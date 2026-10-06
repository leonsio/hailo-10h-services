import asyncio
import io
import json
import threading
import wave
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_services import FakeBackend

from hailo_services.app import create_app
from hailo_services.config import Settings
from hailo_services.errors import BusyError
from hailo_services.schemas import SpeechRequest
from hailo_services.speech_piper import PiperBackend
from hailo_services.speech_runtime import SpeechRuntime


class SpeechBackend:
    def start(self):
        pass

    def synthesize(self, request):
        if request.voice == "missing":
            raise ValueError("Voice is not installed")
        if request.voice == "broken":
            raise RuntimeError("native failure")
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(22050)
            audio.writeframes(b"\0\0" * 100)
        return output.getvalue(), "audio/wav", 22050

    def close(self):
        pass


def test_speech_http_and_model_discovery():
    settings = Settings(
        piper_enabled=True, wyoming_port=0, api_key="secret", piper_max_input_chars=100
    )
    app = create_app(settings, FakeBackend(), speech_backend=SpeechBackend())
    with TestClient(app) as client:
        payload = {"model": "piper", "input": "Guten Abend!"}
        assert client.post("/v1/audio/speech", json=payload).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        response = client.post("/v1/audio/speech", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-audio-sample-rate"] == "22050"
        with wave.open(io.BytesIO(response.content)) as audio:
            assert audio.getnframes() == 100
        assert client.get("/health").json()["piper"]["ready"] is True
        assert "piper" in [
            m["id"] for m in client.get("/v1/models", headers=headers).json()["data"]
        ]
        for change, status in [
            ({"input": " "}, 422),
            ({"input": "a" * 101}, 400),
            ({"voice": "../bad"}, 422),
            ({"voice": "missing"}, 400),
            ({"voice": "broken"}, 502),
            ({"speed": 0}, 422),
            ({"response_format": "mp3"}, 422),
            ({"model": "tts-1"}, 422),
        ]:
            assert (
                client.post("/v1/audio/speech", json=payload | change, headers=headers).status_code
                == status
            )


def test_disabled_and_failed_startup_do_not_break_other_services():
    class FailedBackend(SpeechBackend):
        def start(self):
            raise ImportError("piper missing")

    for enabled in [False, True]:
        app = create_app(
            Settings(piper_enabled=enabled, wyoming_port=0),
            FakeBackend(),
            speech_backend=FailedBackend(),
        )
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/health").json()["piper"]["ready"] is False
            assert client.post("/v1/audio/speech", json={"input": "Hallo"}).status_code == 503


def test_timeout_keeps_capacity_until_native_work_finishes():
    entered, release = threading.Event(), threading.Event()

    class SlowBackend(SpeechBackend):
        def synthesize(self, request):
            entered.set()
            release.wait(2)
            return super().synthesize(request)

    async def run():
        runtime = SpeechRuntime(
            Settings(piper_enabled=True, queue_size=1, request_timeout=0.01), SlowBackend()
        )
        await runtime.start()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await runtime.synthesize(SpeechRequest(input="Hallo"))
            assert entered.is_set() and runtime.pending == 1
            with pytest.raises(BusyError):
                await runtime.synthesize(SpeechRequest(input="Hallo"))
            release.set()
        finally:
            release.set()
            await runtime.close()
        assert runtime.pending == 0

    asyncio.run(run())


def test_piper_voice_language_speed_and_pcm(monkeypatch, tmp_path):
    calls = []
    for name, language in [("de_DE-thorsten-medium", "de_DE"), ("en_US-lessac-medium", "en_US")]:
        (tmp_path / (name + ".onnx")).touch()
        (tmp_path / (name + ".onnx.json")).write_text(json.dumps({"language": {"code": language}}))

    class Voice:
        config = SimpleNamespace(sample_rate=22050)

        @classmethod
        def load(cls, path, use_cuda):
            assert use_cuda is False
            calls.append(path)
            return cls()

        def synthesize(self, text, syn_config):
            assert syn_config.length_scale == 0.5
            yield SimpleNamespace(audio_int16_bytes=b"\0\0" * 2205)

    import sys

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=Voice, SynthesisConfig=lambda **kw: SimpleNamespace(**kw)),
    )
    backend = PiperBackend(Settings(piper_voice_dir=str(tmp_path)))
    backend.start()
    data, mime, rate = backend.synthesize(SpeechRequest(input="Hallo", speed=2))
    assert mime == "audio/wav" and rate == 22050 and data.startswith(b"RIFF")
    data, mime, rate = backend.synthesize(
        SpeechRequest(input="Hallo", speed=2, response_format="pcm", language="de")
    )
    assert mime == "audio/pcm" and rate == 24000 and len(data) == 4800
    assert len(calls) == 1
    with pytest.raises(ValueError, match="not en"):
        backend.synthesize(SpeechRequest(input="Hello", language="en"))
    backend.synthesize(
        SpeechRequest(input="Hello", voice="en_US-lessac-medium", language="en-US", speed=2)
    )
    assert len(calls) == 2
    assert backend.default_language == "de_DE"  # Discovery must survive a resident voice switch.
    with pytest.raises(ValueError):
        backend._path("../escape")
    with pytest.raises(ValueError, match="not installed"):
        backend._path("missing")
    backend.settings = replace(backend.settings, max_audio_seconds=0)
    with pytest.raises(ValueError, match="max_audio_seconds"):
        backend.synthesize(SpeechRequest(input="Hello", speed=2))
    backend.close()
    assert backend.voice is None


def test_piper_settings_yaml_and_env(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "settings:\n  piper_enabled: true\n  piper_language: en_US\n  piper_voice: en_US-lessac-medium\n"
    )
    monkeypatch.setenv("HAILO_CONFIG", str(config))
    monkeypatch.setenv("HAILO_PIPER_MAX_INPUT_CHARS", "100")
    settings = Settings.from_env()
    assert settings.piper_enabled and settings.piper_language == "en_US"
    assert settings.piper_max_input_chars == 100
    with pytest.raises(ValueError):
        Settings(piper_max_input_chars=0)
