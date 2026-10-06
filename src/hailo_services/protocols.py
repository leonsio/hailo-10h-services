"""Wyoming speech and MQTT/WebSocket operation dispatch."""

import asyncio
import io
import json
import logging
import re
import ssl
import time
import uuid
import wave
from contextlib import suppress
from pathlib import Path

import aiomqtt
import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event, async_write_event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.tts import Synthesize

from .config import Settings
from .media import audio_file, audio_metadata, decode_base64, normalize_audio
from .schemas import ChatRequest, SpeechRequest, TranscribeRequest
from .speech_common import language_matches
from .speech_piper import installed_voices

_LOG = logging.getLogger(__name__)
LANGUAGES = ["de", "en", "fr", "es", "it", "nl", "pt", "pl", "ru", "uk", "tr", "zh", "ja", "ko"]


def _debug(settings, message, *args):
    """Emit a formatted diagnostic when debug logging is enabled.

    Args:
        settings (Settings): Validated service settings controlling enabled models and limits.
        message (str | dict[str, Any]): Formatted diagnostic text or ASGI message.
        *args (Any): Positional arguments forwarded to the native operation.

    Returns:
        None: Writes a debug record when enabled.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if settings.debug_log:
        _LOG.debug(message, *args)


def _tts_voices(settings, speech):
    """Describe installed voices only when CPU speech synthesis is ready.

    Args:
        settings (Settings): Configured default voice and provisioned voice directory.
        speech (SpeechRuntime | None): Independent CPU TTS runtime.

    Returns:
        list[dict[str, str]]: Default-first voice IDs and declared languages.
    """
    if not settings.piper_enabled or speech is None or not speech.ready:
        return []
    default = {
        "id": Path(settings.piper_voice).stem,
        "language": getattr(speech.backend, "default_language", None) or settings.piper_language,
    }
    return [default] + [
        voice for voice in installed_voices(settings) if voice["id"] != default["id"]
    ]


def wyoming_info(settings=None, speech=None):
    """Build Wyoming discovery metadata for Whisper and ready Piper voices.

    Args:
        settings (Settings): Validated service settings controlling enabled models and limits.
        speech (SpeechRuntime | None): CPU TTS readiness and voice metadata.

    Returns:
        Info: ASR and TTS programs with installed models/voices and languages.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    settings = settings or Settings()
    attribution = Attribution(
        name="Hailo / OpenAI", url="https://github.com/hailo-ai/hailo_model_zoo_genai"
    )
    info = Info(
        asr=[
            AsrProgram(
                name="hailo-whisper",
                attribution=attribution,
                installed=True,
                description=f"Resident {settings.whisper_hef} on Hailo-10H (SHARED)",
                version="0.1.0",
                models=[
                    AsrModel(
                        name=settings.stt_model,
                        attribution=attribution,
                        installed=True,
                        description=f"Multilingual {settings.whisper_hef}",
                        version=None,
                        languages=LANGUAGES,
                    )
                ],
            )
        ]
        if settings.whisper_enabled
        else []
    )

    voices = _tts_voices(settings, speech)
    if voices:
        piper_attribution = Attribution(name="Piper", url="https://github.com/OHF-Voice/piper1-gpl")
        info.tts = [
            TtsProgram(
                name="hailo-piper",
                attribution=piper_attribution,
                installed=True,
                description="Piper text-to-speech on CPU",
                version="0.1.0",
                supports_synthesize_streaming=False,
                voices=[
                    TtsVoice(
                        name=voice["id"],
                        attribution=piper_attribution,
                        installed=True,
                        description=voice["id"],
                        version=None,
                        languages=[voice["language"]],
                    )
                    for voice in voices
                ],
            )
        ]
    return info


async def read_bounded_event(reader, timeout):
    """Read a Wyoming event with bounded block lengths and a deadline.

    Args:
        reader (asyncio.StreamReader): Connected stream providing protocol input.
        timeout (float): Operation deadline or proxy timeout in seconds.

    Returns:
        Event | None: Decoded event, or None at end of stream.

    Raises:
        ValueError: The JSON header or declared block sizes are invalid.
        asyncio.IncompleteReadError: The peer closes before a declared block is complete.
        asyncio.TimeoutError: The event is not received before the deadline.
    """

    async def read():
        """Read and validate one Wyoming header, JSON block and payload.

        Returns:
            Event | None: Decoded event, or None at end of stream.

        Raises:
            ValueError: Wyoming data block exceeds limit.
        """
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
    """Serve Wyoming discovery, Whisper transcription and Piper speech output."""

    def __init__(self, runtime, settings, speech=None):
        """Initialize WyomingServer configuration and owned dependencies.

        Args:
            runtime (Runtime): Owner-thread scheduler serving chat and speech requests.
            settings (Settings): Validated service settings controlling enabled models and limits.
            speech (SpeechRuntime | None): Shared CPU TTS runtime used by HTTP and Wyoming.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.runtime, self.settings = runtime, settings
        self.speech = speech
        self.server = None
        self.connections = set()

    async def start(self):
        """Initialize resident resources or start the configured transport listener.

        Returns:
            None: Marks the service ready after successful initialization.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.server = await asyncio.start_server(
            self.handle,
            self.settings.wyoming_host,
            self.settings.wyoming_port,
        )

    async def close(self):
        """Release resources owned by this service or native context.

        Returns:
            None: Closes native resources, connections or owner executors.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        tasks = list(self.connections)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def synthesize(self, event, writer):
        """Translate a Wyoming synthesis request into bounded CPU audio output.

        Args:
            event (Event): Text and optional voice/language selection.
            writer (asyncio.StreamWriter): Connection receiving PCM audio events.

        Returns:
            None: Sends audio-start, bounded audio-chunk events and audio-stop.

        Raises:
            ValueError: Voice, language, speaker or text format is unsupported.
            BusyError: CPU speech synthesis is not available or its queue is full.
        """
        if self.speech is None:
            raise ValueError("Piper TTS is not configured")
        request = Synthesize.from_event(event)
        if request.text_format not in {None, "text"}:
            raise ValueError("Piper supports plain text only, not SSML")
        voices = _tts_voices(self.settings, self.speech)
        selected = None
        if request.voice is not None:
            if request.voice.speaker is not None:
                raise ValueError("Piper speaker selection is not supported")
            if request.voice.name:
                selected = next(
                    (voice for voice in voices if voice["id"] == request.voice.name), None
                )
                if selected is None:
                    raise ValueError("Unknown Piper voice")
            elif request.voice.language:
                selected = next(
                    (
                        voice
                        for voice in voices
                        if language_matches(request.voice.language, voice["language"])
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError("No installed Piper voice matches the requested language")
        voice_id = selected["id"] if selected else None
        # The configured default may be an absolute ONNX path outside voice_dir.
        if voice_id == Path(self.settings.piper_voice).stem:
            voice_id = None
        body = SpeechRequest(input=request.text, voice=voice_id, response_format="wav")
        started = time.perf_counter()
        _debug(
            self.settings,
            "protocol=wyoming event=synthesize_start voice=%s text_chars=%d",
            selected["id"] if selected else Path(self.settings.piper_voice).stem,
            len(body.input),
        )
        data, _, _ = await self.speech.synthesize(body)
        with wave.open(io.BytesIO(data), "rb") as audio:
            rate, width, channels = audio.getframerate(), audio.getsampwidth(), audio.getnchannels()
            await asyncio.wait_for(
                async_write_event(
                    AudioStart(rate=rate, width=width, channels=channels).event(), writer
                ),
                self.settings.request_timeout,
            )
            while chunk := audio.readframes(4096):
                await asyncio.wait_for(
                    async_write_event(
                        AudioChunk(rate=rate, width=width, channels=channels, audio=chunk).event(),
                        writer,
                    ),
                    self.settings.request_timeout,
                )
            await asyncio.wait_for(
                async_write_event(AudioStop().event(), writer), self.settings.request_timeout
            )
        _debug(
            self.settings,
            "protocol=wyoming event=synthesize_complete processing_ms=%.1f sample_rate_hz=%d bytes=%d",
            (time.perf_counter() - started) * 1000,
            rate,
            len(data),
        )

    async def handle(self, reader, writer):
        """Process discovery, speech output and completed PCM recordings on one connection.

        Args:
            reader (asyncio.StreamReader): Connected stream providing protocol input.
            writer (asyncio.StreamWriter): Connected stream used for responses.

        Returns:
            None: Sends discovery, transcripts or errors and closes the connection.

        Raises:
            ValueError: No audio received.
        """
        task = asyncio.current_task()
        self.connections.add(task)
        peer = writer.get_extra_info("peername") or ("unknown", 0)
        request_id = uuid.uuid4().hex[:12]
        language = self.settings.language
        audio, fmt = bytearray(), None
        operation = "transcription"
        try:
            if len(self.connections) > 32:
                raise ValueError("Too many Wyoming connections")
            while True:
                event = await read_bounded_event(reader, self.settings.request_timeout)
                if event is None:
                    break
                if Describe.is_type(event.type):
                    _debug(
                        self.settings,
                        "protocol=wyoming event=describe request_id=%s peer=%s",
                        request_id,
                        peer[0],
                    )
                    await async_write_event(
                        wyoming_info(self.settings, self.speech).event(), writer
                    )
                elif Synthesize.is_type(event.type):
                    operation = "synthesis"
                    await self.synthesize(event, writer)
                    audio, fmt = bytearray(), None
                elif event.type.startswith("synthesize-"):
                    operation = "synthesis"
                    raise ValueError("Streaming text synthesis is not supported; use synthesize")
                elif Transcribe.is_type(event.type):
                    operation = "transcription"
                    request = Transcribe.from_event(event)
                    if request.name not in {
                        None,
                        self.settings.stt_model,
                        self.settings.whisper_hef,
                    }:
                        raise ValueError("Unknown transcription model")
                    language = request.language or self.settings.language
                    if not re.fullmatch(r"[a-z]{2}", language):
                        raise ValueError("Language must be a two-letter code")
                    audio, fmt = bytearray(), None
                    request_id = uuid.uuid4().hex[:12]
                    _debug(
                        self.settings,
                        "protocol=wyoming event=transcribe_start request_id=%s peer=%s model=%s language=%s",
                        request_id,
                        peer[0],
                        request.name or self.settings.stt_model,
                        language,
                    )
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
                    _debug(
                        self.settings,
                        "protocol=wyoming event=audio_start request_id=%s encoding=pcm_s16le sample_rate_hz=%d channels=%d",
                        request_id,
                        start.rate,
                        start.channels,
                    )
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
                    started = time.perf_counter()
                    _debug(
                        self.settings,
                        "protocol=wyoming event=transcribe_inference request_id=%s input_sample_rate_hz=%d channels=%d duration_seconds=%.3f bytes=%d language=%s",
                        request_id,
                        fmt[0],
                        fmt[2],
                        len(audio) / (fmt[0] * fmt[1] * fmt[2]),
                        len(audio),
                        language,
                    )
                    text = await self.runtime.transcribe(normalized, language)
                    _debug(
                        self.settings,
                        "protocol=wyoming event=transcribe_complete request_id=%s inference_ms=%.1f transcript_chars=%d",
                        request_id,
                        (time.perf_counter() - started) * 1000,
                        len(text),
                    )
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
                    Error(text=str(exc), code=f"{operation}_failed").event(), writer
                )
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.connections.discard(task)


async def dispatch(runtime, settings, operation, payload):
    """Validate and execute a chat, transcription or health operation.

    Args:
        runtime (Runtime): Owner-thread scheduler serving chat and speech requests.
        settings (Settings): Validated service settings controlling enabled models and limits.
        operation (str): Requested operation: chat, transcribe or health.
        payload (Any): Decoded protocol data or structured diagnostic payload.

    Returns:
        dict[str, Any]: Operation result suitable for WebSocket or MQTT transport.

    Raises:
        ValueError: The operation, payload or media is invalid.
        BusyError: The selected inference backend is unavailable.
        asyncio.TimeoutError: The request exceeds the inference deadline.
    """
    if operation == "chat":
        request = ChatRequest.model_validate(payload)
        if request.stream:
            raise ValueError("Use HTTP SSE for token streaming")
        return {"text": await runtime.chat(request)}
    if operation == "transcribe":
        request = TranscribeRequest.model_validate(payload)
        data = decode_base64(request.audio_base64, settings.max_body)
        _debug(
            settings,
            "operation=whisper_transcribe model=%s language=%s audio=%s bytes=%d",
            settings.stt_model,
            request.language or settings.language,
            audio_metadata(data),
            len(data),
        )
        audio = audio_file(data, settings.max_audio_seconds)
        return {"text": await runtime.transcribe(audio, request.language)}
    if operation == "health":
        return runtime.status()
    raise ValueError("Unknown operation; use chat/transcribe/health")


class MQTTBridge:
    """Bridge bounded JSON requests to resident services with broker reconnection."""

    def __init__(self, runtime, settings):
        """Initialize MQTTBridge configuration and owned dependencies.

        Args:
            runtime (Runtime): Owner-thread scheduler serving chat and speech requests.
            settings (Settings): Validated service settings controlling enabled models and limits.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.runtime, self.settings = runtime, settings
        self.connected = False

    async def run(self):
        """Run the MQTT request bridge with reconnect and bounded payload handling.

        Returns:
            None: Publishes responses while connected; retries broker disconnects.

        Raises:
            ValueError: id must be 1..64 alphanumeric, underscore or hyphen characters.
        """
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
                    _debug(
                        self.settings,
                        "protocol=mqtt event=connected broker=%s port=%d topic_prefix=%s",
                        self.settings.mqtt_host,
                        self.settings.mqtt_port,
                        prefix,
                    )
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
                            started = time.perf_counter()
                            _debug(
                                self.settings,
                                "protocol=mqtt event=request_start request_id=%s operation=%s topic=%s",
                                identifier,
                                operation,
                                message.topic,
                            )
                            result = await dispatch(
                                self.runtime, self.settings, operation, envelope["payload"]
                            )
                            response = {"id": identifier, "ok": True, "result": result}
                            _debug(
                                self.settings,
                                "protocol=mqtt event=request_end request_id=%s operation=%s duration_ms=%.1f ok=true",
                                identifier,
                                operation,
                                (time.perf_counter() - started) * 1000,
                            )
                        except Exception as exc:
                            _debug(
                                self.settings,
                                "protocol=mqtt event=request_error request_id=%s operation=%s error_type=%s",
                                identifier,
                                locals().get("operation", "unknown"),
                                type(exc).__name__,
                            )
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
                if self.connected:
                    _debug(
                        self.settings,
                        "protocol=mqtt event=disconnected broker=%s",
                        self.settings.mqtt_host,
                    )
                self.connected = False
