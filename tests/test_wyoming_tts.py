import asyncio
import io
import json
import wave

import pytest
from test_services import FakeBackend
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from hailo_services.api.protocols import WyomingServer
from hailo_services.config import Settings
from hailo_services.runtime.runtime import Runtime
from hailo_services.speech.speech_runtime import SpeechRuntime


class VoiceBackend:
    language = "de_DE"

    def __init__(self):
        self.calls = []

    def start(self):
        pass

    def synthesize(self, request):
        self.calls.append(request)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(22050)
            audio.writeframes(b"\x01\x00" * 9000)
        return buffer.getvalue(), "audio/wav", 22050

    def close(self):
        pass


async def connect(settings, backend):
    runtime = Runtime(settings, FakeBackend())
    speech = SpeechRuntime(settings, backend)
    await runtime.start()
    await speech.start()
    server = WyomingServer(runtime, settings, speech)
    await server.start()
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", server.server.sockets[0].getsockname()[1]
    )
    return runtime, speech, server, reader, writer


async def cleanup(runtime, speech, server, writer):
    writer.close()
    await writer.wait_closed()
    await server.close()
    await speech.close()
    await runtime.close()


async def read(reader):
    return await asyncio.wait_for(async_read_event(reader), 2)


def test_wyoming_discovery_tts_audio_and_stt_on_same_socket(tmp_path):
    (tmp_path / "en_US-lessac-medium.onnx").touch()
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text(
        json.dumps({"language": {"code": "en_US"}})
    )

    async def run():
        settings = Settings(
            piper_enabled=True,
            piper_voice_dir=str(tmp_path),
            wyoming_host="127.0.0.1",
            wyoming_port=0,
        )
        backend = VoiceBackend()
        runtime, speech, server, reader, writer = await connect(settings, backend)
        try:
            await async_write_event(Describe().event(), writer)
            info = Info.from_event(await read(reader))
            assert info.asr[0].models[0].name == "whisper-base"
            assert info.tts[0].name == "hailo-piper"
            assert info.tts[0].installed
            assert not info.tts[0].supports_synthesize_streaming
            assert [(voice.name, voice.languages) for voice in info.tts[0].voices] == [
                ("de_DE-thorsten-medium", ["de_DE"]),
                ("en_US-lessac-medium", ["en_US"]),
            ]
            for selection, expected_voice in [
                (None, None),
                (SynthesizeVoice(name="de_DE-thorsten-medium"), None),
                (SynthesizeVoice(language="en"), "en_US-lessac-medium"),
            ]:
                await async_write_event(
                    Synthesize(text="Hallo Welt", voice=selection).event(), writer
                )
                start = AudioStart.from_event(await read(reader))
                assert (start.rate, start.width, start.channels) == (22050, 2, 1)
                chunks = []
                while True:
                    event = await read(reader)
                    if AudioStop.is_type(event.type):
                        break
                    chunk = AudioChunk.from_event(event)
                    assert (chunk.rate, chunk.width, chunk.channels) == (22050, 2, 1)
                    assert len(chunk.audio) <= 8192
                    chunks.append(chunk.audio)
                assert b"".join(chunks) == b"\x01\x00" * 9000
                assert len(chunks) == 3
                assert backend.calls[-1].voice == expected_voice
            await async_write_event(Transcribe(language="de").event(), writer)
            await async_write_event(AudioStart(rate=16000, width=2, channels=1).event(), writer)
            await async_write_event(
                AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0" * 1600).event(), writer
            )
            await async_write_event(AudioStop().event(), writer)
            assert Transcript.from_event(await read(reader)).text == "Hallo Welt"
        finally:
            await cleanup(runtime, speech, server, writer)

    asyncio.run(run())


@pytest.mark.parametrize(
    "synthesis_request",
    [
        Synthesize(text="Hallo", voice=SynthesizeVoice(name="missing")),
        Synthesize(text="Hallo", voice=SynthesizeVoice(language="ru")),
        Synthesize(text="Hallo", voice=SynthesizeVoice(name="de_DE-thorsten-medium", speaker="1")),
        Synthesize(text="<speak>Hallo</speak>", text_format="ssml"),
        Synthesize(text="x" * 101),
    ],
)
def test_wyoming_tts_invalid_requests_return_protocol_error(synthesis_request):
    async def run():
        settings = Settings(
            piper_enabled=True, piper_max_input_chars=100, wyoming_host="127.0.0.1", wyoming_port=0
        )
        backend = VoiceBackend()
        runtime, speech, server, reader, writer = await connect(settings, backend)
        try:
            await async_write_event(synthesis_request.event(), writer)
            error = Error.from_event(await read(reader))
            assert error.code == "synthesis_failed"
            assert error.text
            assert backend.calls == []
        finally:
            await cleanup(runtime, speech, server, writer)

    asyncio.run(run())


@pytest.mark.parametrize("enabled,failed", [(False, False), (True, True)])
def test_wyoming_does_not_advertise_unavailable_tts(enabled, failed):
    class FailedBackend(VoiceBackend):
        def start(self):
            raise RuntimeError("No voice installed")

    async def run():
        settings = Settings(piper_enabled=enabled, wyoming_host="127.0.0.1", wyoming_port=0)
        runtime, speech, server, reader, writer = await connect(
            settings, FailedBackend() if failed else VoiceBackend()
        )
        try:
            await async_write_event(Describe().event(), writer)
            info = Info.from_event(await read(reader))
            assert info.tts == []
            assert info.asr
            await async_write_event(Synthesize(text="Hallo").event(), writer)
            assert Error.from_event(await read(reader)).code == "synthesis_failed"
        finally:
            await cleanup(runtime, speech, server, writer)

    asyncio.run(run())
