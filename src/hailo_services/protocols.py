import asyncio
import json
import logging
import re
import ssl
from contextlib import suppress

import aiomqtt
import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event, async_write_event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info

from .config import STT_MODEL
from .media import audio_file, decode_base64, normalize_audio
from .schemas import ChatRequest, TranscribeRequest

_LOG = logging.getLogger(__name__)
LANGUAGES = ["de", "en", "fr", "es", "it", "nl", "pt", "pl", "ru", "uk", "tr", "zh", "ja", "ko"]


def wyoming_info():
    attribution = Attribution(name="Hailo / OpenAI", url="https://github.com/hailo-ai/hailo-apps")
    return Info(
        asr=[
            AsrProgram(
                name="hailo-whisper",
                attribution=attribution,
                installed=True,
                description="Resident Whisper Base on Hailo-10H (SHARED)",
                version="0.1.0",
                models=[
                    AsrModel(
                        name=STT_MODEL,
                        attribution=attribution,
                        installed=True,
                        description="Multilingual Whisper Base",
                        version=None,
                        languages=LANGUAGES,
                    )
                ],
            )
        ]
    )


async def read_bounded_event(reader, timeout):
    async def read():
        line = await reader.readline()  # StreamReader limits the JSON header to 64 KiB.
        if not line:
            return None
        header = json.loads(line)
        data_length = header.get("data_length", 0)
        payload_length = header.get("payload_length", 0)
        if not isinstance(data_length, int) or not 0 <= data_length <= 65536:
            raise ValueError("Wyoming data block exceeds limit")
        if not isinstance(payload_length, int) or not 0 <= payload_length <= 1024 * 1024:
            raise ValueError("Wyoming payload exceeds 1 MiB chunk limit")
        data = header.get("data", {})
        if data_length:
            data.update(json.loads(await reader.readexactly(data_length)))
        payload = await reader.readexactly(payload_length) if payload_length else None
        return Event(type=header["type"], data=data, payload=payload)

    return await asyncio.wait_for(read(), timeout)


class WyomingServer:
    def __init__(self, runtime, settings):
        self.runtime, self.settings = runtime, settings
        self.server = None
        self.connections = set()

    async def start(self):
        self.server = await asyncio.start_server(
            self.handle,
            self.settings.wyoming_host,
            self.settings.wyoming_port,
        )

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        tasks = list(self.connections)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.connections.add(task)
        language = self.settings.language
        audio, fmt = bytearray(), None
        try:
            if len(self.connections) > 32:
                raise ValueError("Too many Wyoming connections")
            while True:
                event = await read_bounded_event(reader, self.settings.request_timeout)
                if event is None:
                    break
                if Describe.is_type(event.type):
                    await async_write_event(wyoming_info().event(), writer)
                elif Transcribe.is_type(event.type):
                    request = Transcribe.from_event(event)
                    if request.name not in {None, STT_MODEL, "Whisper-Base"}:
                        raise ValueError("Unknown transcription model")
                    language = request.language or self.settings.language
                    if not re.fullmatch(r"[a-z]{2}", language):
                        raise ValueError("Language must be a two-letter code")
                    audio, fmt = bytearray(), None
                elif AudioStart.is_type(event.type):
                    start = AudioStart.from_event(event)
                    if (
                        start.width != 2
                        or start.channels not in {1, 2}
                        or not 8000 <= start.rate <= 192000
                    ):
                        raise ValueError(
                            "Wyoming requires signed PCM16, mono/stereo, 8000..192000 Hz"
                        )
                    fmt = (start.rate, start.width, start.channels)
                    audio.clear()
                elif AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    if fmt != (chunk.rate, chunk.width, chunk.channels):
                        raise ValueError("Audio format mismatch or missing audio-start")
                    if len(chunk.audio) % (chunk.width * chunk.channels):
                        raise ValueError("Incomplete PCM frame")
                    audio.extend(chunk.audio)
                    limit = min(
                        self.settings.max_body,
                        fmt[0] * fmt[1] * fmt[2] * self.settings.max_audio_seconds,
                    )
                    if len(audio) > limit:
                        raise ValueError("Audio exceeds duration/size limit")
                elif AudioStop.is_type(event.type):
                    if not fmt or not audio:
                        raise ValueError("No audio received")
                    samples = (
                        np.frombuffer(audio, dtype="<i2").astype(np.float32).reshape(-1, fmt[2])
                        / 32768
                    )
                    normalized = normalize_audio(samples, fmt[0], self.settings.max_audio_seconds)
                    text = await self.runtime.transcribe(normalized, language)
                    await async_write_event(
                        Transcript(text=text, language=language).event(), writer
                    )
                    audio, fmt = bytearray(), None
                # Optional ping/select-program events are safely ignored.
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as exc:
            _LOG.warning("Wyoming request failed: %s", exc)
            with suppress(ConnectionError):
                await async_write_event(
                    Error(text=str(exc), code="transcription_failed").event(), writer
                )
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.connections.discard(task)


async def dispatch(runtime, settings, operation, payload):
    if operation == "chat":
        request = ChatRequest.model_validate(payload)
        if request.stream:
            raise ValueError("Use HTTP SSE for token streaming")
        return {"text": await runtime.chat(request)}
    if operation == "transcribe":
        request = TranscribeRequest.model_validate(payload)
        data = decode_base64(request.audio_base64, settings.max_body)
        audio = audio_file(data, settings.max_audio_seconds)
        return {"text": await runtime.transcribe(audio, request.language)}
    if operation == "health":
        return runtime.status()
    raise ValueError("Unknown operation; use chat/transcribe/health")


class MQTTBridge:
    def __init__(self, runtime, settings):
        self.runtime, self.settings = runtime, settings
        self.connected = False

    async def run(self):
        prefix = self.settings.mqtt_prefix.rstrip("/")
        while True:
            try:
                async with aiomqtt.Client(
                    hostname=self.settings.mqtt_host,
                    port=self.settings.mqtt_port,
                    username=self.settings.mqtt_username or None,
                    password=self.settings.mqtt_password or None,
                    identifier="hailo-10h-services",
                    tls_context=ssl.create_default_context() if self.settings.mqtt_tls else None,
                    will=aiomqtt.Will(f"{prefix}/status", payload="offline", qos=1, retain=True),
                    max_queued_incoming_messages=self.settings.queue_size,
                ) as client:
                    self.connected = True
                    await client.subscribe(f"{prefix}/request/+", qos=0)
                    await client.publish(f"{prefix}/status", "online", retain=True, qos=1)
                    # QoS 0 avoids replaying old inference requests on reconnect.
                    async for message in client.messages:
                        if message.retain:
                            continue
                        if len(message.payload) > self.settings.max_body:
                            _LOG.warning("Oversized MQTT request discarded")
                            continue
                        identifier = None
                        try:
                            envelope = json.loads(message.payload)
                            identifier = envelope.get("id")
                            if not isinstance(identifier, str) or not re.fullmatch(
                                r"[A-Za-z0-9_-]{1,64}", identifier
                            ):
                                raise ValueError(
                                    "id must be 1..64 alphanumeric, underscore or hyphen characters"
                                )
                            operation = str(message.topic).rsplit("/", 1)[-1]
                            result = await dispatch(
                                self.runtime, self.settings, operation, envelope["payload"]
                            )
                            response = {"id": identifier, "ok": True, "result": result}
                        except Exception as exc:
                            response = {"id": identifier, "ok": False, "error": str(exc)}
                        if isinstance(identifier, str) and re.fullmatch(
                            r"[A-Za-z0-9_-]{1,64}", identifier
                        ):
                            await client.publish(
                                f"{prefix}/response/{identifier}",
                                json.dumps(response),
                                qos=0,
                                retain=False,
                            )
            except aiomqtt.MqttError as exc:
                _LOG.warning("MQTT disconnected: %s; retrying in 5s", exc)
                await asyncio.sleep(5)
            finally:
                self.connected = False
