import asyncio
import base64
import io
import json
import struct
import sys
import threading
import time
import types
import zlib
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

from hailo_services.app import create_app
from hailo_services.config import VLM_MODEL, Settings
from hailo_services.media import audio_file, image_frame
from hailo_services.protocols import WyomingServer, dispatch, read_bounded_event
from hailo_services.runtime import BusyError, HailoBackend, Runtime
from hailo_services.schemas import ChatRequest


class FakeBackend:
    paths = {"vlm": "/fake/qwen.hef", "whisper": "/fake/whisper.hef"}

    def __init__(self):
        self.started = self.closed = False
        self.threads = []
        self.calls = []

    def start(self):
        self.started = True
        self.threads.append(threading.get_ident())

    def chat(self, request, emit=None, cancelled=None):
        self.calls.append(request)
        self.threads.append(threading.get_ident())
        if emit:
            emit("Hello ")
            emit("world")
        return "Hello world"

    def transcribe(self, audio, language):
        self.calls.append((audio, language))
        self.threads.append(threading.get_ident())
        return "Hallo Welt"

    def close(self):
        self.closed = True
        self.threads.append(threading.get_ident())


def wav(rate=16000, channels=1):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros((rate // 10, channels)), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def snapshot():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 16), "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def settings(**kwargs):
    return replace(Settings(), wyoming_port=0, **kwargs)


def test_http_ws_audio_and_resident_owner():
    backend = FakeBackend()
    with TestClient(create_app(settings(), backend)) as client:
        assert client.get("/health").json()["group_id"] == "SHARED"
        models = client.get("/v1/models").json()["data"]
        assert [entry["id"] for entry in models] == [VLM_MODEL, "whisper-base"]
        payload = {"model": VLM_MODEL, "messages": [{"role": "user", "content": "Hi"}]}
        answer = client.post("/v1/chat/completions", json=payload)
        assert answer.status_code == 200
        assert answer.json()["choices"][0]["message"]["content"] == "Hello world"
        stream = client.post("/v1/chat/completions", json={**payload, "stream": True}).text
        assert '"content": "Hello "' in stream and '"content": "world"' in stream
        assert stream.endswith("data: [DONE]\n\n")
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("voice.wav", wav(48000, 2))},
            data={"language": "de"},
        )
        assert response.json() == {"text": "Hallo Welt"}
        assert backend.calls[-1][0].dtype == np.dtype("<f4")
        assert len(backend.calls[-1][0]) == 1600
        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "id": "a",
                    "op": "transcribe",
                    "payload": {"audio_base64": base64.b64encode(wav()).decode()},
                }
            )
            assert ws.receive_json()["result"]["text"] == "Hallo Welt"
            ws.send_text("not json")
            assert ws.receive_json()["ok"] is False
        assert not backend.closed
    assert backend.closed and len(set(backend.threads)) == 1


def test_auth_limits_and_unsupported_parameters():
    with TestClient(create_app(settings(api_key="secret", max_body=2048), FakeBackend())) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/v1/models", headers=headers).status_code == 200
        assert (
            client.post("/v1/chat/completions", content=b"x" * 2049, headers=headers).status_code
            == 413
        )
        assert (
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "tools": []},
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/chat/completions",
                json={"model": "wrong", "messages": [{"role": "user", "content": "hi"}]},
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/audio/transcriptions", files={"file": ("bad.wav", b"bad")}, headers=headers
            ).status_code
            == 400
        )


def test_playground_public_assets_keep_inference_authenticated():
    backend = FakeBackend()
    with TestClient(create_app(settings(api_key="secret", max_audio_seconds=10), backend)) as client:
        page = client.get("/")
        assert page.status_code == 200 and 'lang="de"' in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert client.head("/").status_code == 200
        for path in ("app.js", "style.css", "recorder-worklet.js"):
            asset = client.get("/ui/" + path)
            assert asset.status_code == 200
            assert asset.headers["x-content-type-options"] == "nosniff"
        config = client.get("/ui/config").json()
        assert config["auth_required"] is True
        assert config["recording_seconds"] == 10
        assert config["whisper_model"] == "whisper-base"
        assert "secret" not in json.dumps(config)
        assert client.post("/ui/config").status_code == 401
        assert client.get("/ui/unknown.js").status_code == 401
        assert client.get("/ui/../config.py").status_code == 401
        payload = {"messages": [{"role": "user", "content": "Hallo"}]}
        assert client.post("/v1/chat/completions", json=payload).status_code == 401
        assert client.post("/v1/audio/transcriptions", files={"file": ("voice.wav", wav())}).status_code == 401
        assert not backend.calls
        headers = {"Authorization": "Bearer secret"}
        assert client.post("/v1/chat/completions", json=payload, headers=headers).status_code == 200
        assert client.post(
            "/v1/audio/transcriptions", files={"file": ("aufnahme.wav", wav(48000))},
            data={"model": "whisper-base", "language": "de"}, headers=headers,
        ).json() == {"text": "Hallo Welt"}


def test_playground_config_without_key():
    with TestClient(create_app(settings(), FakeBackend())) as client:
        config = client.get("/ui/config").json()
        assert config["auth_required"] is False
        assert config["recording_seconds"] == 30


def test_media_validation():
    frame = image_frame(snapshot(), 4096)
    assert frame.shape == (336, 336, 3)
    assert frame.dtype == np.uint8 and frame.flags.c_contiguous and frame.flags.writeable
    frame[0, 0] = [0, 1, 2]  # The native binding may request writable buffer access.
    with pytest.raises(ValueError):
        image_frame("http://localhost/private", 4096)
    with pytest.raises(ValueError):
        audio_file(wav(), 0)


def test_phone_resolution_jpeg_is_downsampled_but_extreme_images_are_rejected(monkeypatch):
    # Lower the production limit in this unit test so the decoder path is
    # exercised without allocating a real 48 MP buffer on the test runner.
    monkeypatch.setattr("hailo_services.media._MAX_IMAGE_PIXELS", 3_000_000)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 3_000_000)
    image = Image.new("RGB", (1800, 1600), "#4a728c")
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=75)
    frame = image_frame(base64.b64encode(encoded.getvalue()).decode(), 1024 * 1024)
    assert frame.shape == (336, 336, 3)
    assert frame.flags.writeable and frame.flags.c_contiguous

    def png_chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    huge_png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">2I5B", 1801, 1800, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(b"\0"))
        + png_chunk(b"IEND", b"")
    )
    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(ValueError, match="Image has too many pixels"):
            image_frame(base64.b64encode(huge_png).decode(), 1024 * 1024)


def test_timeout_retains_queue_slot():
    class Slow(FakeBackend):
        def chat(self, *args):
            time.sleep(0.15)
            return "done"

    async def run():
        runtime = Runtime(settings(queue_size=1, request_timeout=0.01), Slow())
        await runtime.start()
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(asyncio.TimeoutError):
            await runtime.chat(request)
        assert runtime.pending == 1
        with pytest.raises(BusyError):
            await runtime.chat(request)
        await asyncio.sleep(0.2)
        assert runtime.pending == 0
        await runtime.close()

    asyncio.run(run())


def test_cancel_retains_queue_slot():
    class Slow(FakeBackend):
        def chat(self, *args):
            time.sleep(0.1)
            return "done"

    async def run():
        runtime = Runtime(settings(queue_size=1), Slow())
        await runtime.start()
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        task = asyncio.create_task(runtime.chat(request))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.pending == 1
        await runtime.close()

    asyncio.run(run())


def test_real_wyoming_wire_protocol():
    async def run():
        config = settings()
        runtime = Runtime(config, FakeBackend())
        await runtime.start()
        server = WyomingServer(runtime, config)
        await server.start()
        port = server.server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            await async_write_event(Describe().event(), writer)
            info = Info.from_event(await async_read_event(reader))
            assert info.asr[0].models[0].name == "whisper-base"
            await async_write_event(Transcribe(language="de").event(), writer)
            await async_write_event(AudioStart(rate=16000, width=2, channels=1).event(), writer)
            await async_write_event(
                AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0" * 1600).event(), writer
            )
            await async_write_event(AudioStop().event(), writer)
            transcript = Transcript.from_event(await async_read_event(reader))
            assert transcript.text == "Hallo Welt"
            # A second request on the same connection must not inherit samples/language.
            await async_write_event(Transcribe(language="en").event(), writer)
            await async_write_event(AudioStart(rate=48000, width=2, channels=2).event(), writer)
            await async_write_event(
                AudioChunk(rate=48000, width=2, channels=2, audio=b"\0\0" * 9600).event(), writer
            )
            await async_write_event(AudioStop().event(), writer)
            assert Transcript.from_event(await async_read_event(reader)).language == "en"
        finally:
            writer.close()
            await writer.wait_closed()
            await server.close()
            await runtime.close()

    asyncio.run(run())


def test_wyoming_rejects_huge_payload_before_allocation():
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"type":"audio-chunk","payload_length":999999999}\n')
        with pytest.raises(ValueError, match="payload"):
            await read_bounded_event(reader, 1)

    asyncio.run(run())


def test_mcp_protocol_and_tools():
    with TestClient(create_app(settings(mcp_hosts="testserver"), FakeBackend())) as client:
        headers = {"Accept": "application/json, text/event-stream"}
        init = client.post(
            "/mcp/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
        assert init.status_code == 200, init.text
        response = client.post(
            "/mcp/",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        names = [tool["name"] for tool in response.json()["result"]["tools"]]
        assert set(names) == {"analyze_image", "transcribe_audio", "chat_text"}
        response = client.post(
            "/mcp/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "analyze_image",
                    "arguments": {"image_base64": snapshot(), "prompt": "What is visible?"},
                },
            },
        )
        assert response.json()["result"]["content"][0]["text"] == "Hello world"


def test_mqtt_dispatch_uses_same_runtime():
    async def run():
        runtime = Runtime(settings(), FakeBackend())
        await runtime.start()
        result = await dispatch(
            runtime, settings(), "chat", {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert result == {"text": "Hello world"}
        result = await dispatch(
            runtime, settings(), "transcribe", {"audio_base64": base64.b64encode(wav()).decode()}
        )
        assert result == {"text": "Hallo Welt"}
        await runtime.close()

    asyncio.run(run())


def test_native_shared_creation_residency_and_partial_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr("hailo_services.runtime.prepare_model_version", lambda: None)
    events, params_seen = [], []

    class Resource:
        def __init__(self, name):
            self.name = name

        def release(self):
            events.append(self.name)

    class Device(Resource):
        @staticmethod
        def create_params():
            return types.SimpleNamespace(group_id="UNSHARED")

        def __init__(self, params):
            params_seen.append(params.group_id)
            super().__init__("device")

    hef = tmp_path / "model.hef"
    hef.write_bytes(b"compiled test fixture")
    resolved = []

    def resolve(model, **kwargs):
        resolved.append((model, kwargs))
        return hef

    modules = {
        "hailo_platform": types.SimpleNamespace(VDevice=Device),
        "hailo_platform.genai": types.SimpleNamespace(
            VLM=lambda *a: Resource("vlm"), Speech2Text=lambda *a: Resource("whisper")
        ),
        "hailo_apps": types.ModuleType("hailo_apps"),
        "hailo_apps.python": types.ModuleType("hailo_apps.python"),
        "hailo_apps.python.core": types.ModuleType("hailo_apps.python.core"),
        "hailo_apps.python.core.common": types.ModuleType("hailo_apps.python.core.common"),
        "hailo_apps.python.core.common.core": types.SimpleNamespace(resolve_hef_path=resolve),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    backend = HailoBackend(settings())
    backend.start()
    assert params_seen == ["SHARED"]
    assert resolved == [
        (VLM_MODEL, {"app_name": "vlm_chat", "arch": "hailo10h"}),
        ("Whisper-Base", {"app_name": "whisper_chat", "arch": "hailo10h"}),
    ]
    assert not events and backend.vlm and backend.whisper
    backend.close()
    assert events == ["whisper", "vlm", "device"]

    def fail(*args):
        raise RuntimeError("out of memory")

    modules["hailo_platform.genai"].Speech2Text = fail
    events.clear()
    with pytest.raises(RuntimeError, match="out of memory"):
        backend.start()
    assert events == ["vlm", "device"]


def test_model_release_selection_handles_hailort_54():
    from hailo_services.models import select_release

    available = ["v5.1.0", "v5.2.0", "v5.3.0"]
    assert select_release("5.4.0", available) == "v5.3.0"
    assert select_release("5.2.1", available) == "v5.2.0"
    assert select_release("5.4.0", available, "v5.2.0") == "v5.2.0"
    with pytest.raises(ValueError):
        select_release("5.4.0", available, "v5.4.0")
    with pytest.raises(ValueError):
        select_release("5.1.0", available, "v5.3.0")


def test_vlm_preprocessing_context_cleanup_and_native_streaming():
    class Vlm:
        def __init__(self):
            self.clears = 0
            self.request = None

        def clear_context(self):
            self.clears += 1

        def generate(self, **kwargs):
            from contextlib import contextmanager

            self.request = kwargs

            @contextmanager
            def chunks():
                yield iter(["Hallo", " Welt", "<|im_end|>"])

            return chunks()

    backend = HailoBackend(settings())
    backend.vlm = Vlm()
    request = ChatRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": snapshot()}},
                ],
            }
        ]
    )
    streamed = []
    assert backend.chat(request, streamed.append) == "Hallo Welt"
    assert streamed == ["Hallo", " Welt"]
    assert backend.vlm.clears == 2
    frame = backend.vlm.request["frames"][0]
    assert frame.shape == (336, 336, 3) and frame.dtype == np.uint8
    assert not memoryview(frame).readonly and frame.flags.c_contiguous
    assert frame[0, 0].tolist() == [255, 0, 0]
    assert backend.vlm.request["prompt"][0]["content"][1] == {"type": "image"}

    def fail(**kwargs):
        raise RuntimeError("native failure")

    backend.vlm.generate = fail
    with pytest.raises(RuntimeError, match="native failure"):
        backend.chat(request)
    assert backend.vlm.clears == 4


def test_mqtt_envelopes_and_invalid_id_do_not_crash_bridge(monkeypatch):
    from hailo_services.protocols import MQTTBridge

    published = []

    class Messages:
        def __aiter__(self):
            self.items = iter(
                [
                    {"id": 42, "payload": {}},
                    {"id": "good", "payload": {"messages": [{"role": "user", "content": "hi"}]}},
                ]
            )
            return self

        async def __anext__(self):
            try:
                value = next(self.items)
            except StopIteration:
                raise asyncio.CancelledError
            return types.SimpleNamespace(
                payload=json.dumps(value).encode(), retain=False, topic="hailo10h/request/chat"
            )

    class Client:
        def __init__(self, **kwargs):
            self.messages = Messages()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def subscribe(self, *args, **kwargs):
            pass

        async def publish(self, topic, payload, **kwargs):
            published.append((topic, payload, kwargs))

    monkeypatch.setattr("hailo_services.protocols.aiomqtt.Client", Client)

    async def run():
        config = settings(mqtt_host="broker")
        runtime = Runtime(config, FakeBackend())
        await runtime.start()
        bridge = MQTTBridge(runtime, config)
        with pytest.raises(asyncio.CancelledError):
            await bridge.run()
        assert not bridge.connected
        await runtime.close()

    asyncio.run(run())
    assert published[0][0] == "hailo10h/status"
    topic, payload, options = published[1]
    assert topic == "hailo10h/response/good"
    assert json.loads(payload)["result"]["text"] == "Hello world"
    assert options["retain"] is False


def test_version_detection_uses_binding_without_device_probe(monkeypatch):
    from hailo_services.models import prepare_model_version

    modules = {
        "hailo_platform": types.SimpleNamespace(__version__="5.4.0"),
        "hailo_apps": types.ModuleType("hailo_apps"),
        "hailo_apps.python": types.ModuleType("hailo_apps.python"),
        "hailo_apps.python.core": types.ModuleType("hailo_apps.python.core"),
        "hailo_apps.python.core.common": types.ModuleType("hailo_apps.python.core.common"),
        "hailo_apps.python.core.common.defines": types.SimpleNamespace(
            HAILORT_VERSION_KEY="hailort_version",
            MODEL_ZOO_VERSION_KEY="model_zoo_version",
            VALID_H10_MODEL_ZOO_VERSION=["v5.1.0", "v5.2.0", "v5.3.0"],
        ),
        # No CLI/version probe helper exists in this fixture.
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("hailort_version", raising=False)
    monkeypatch.delenv("model_zoo_version", raising=False)
    prepare_model_version()
    import os

    assert os.environ["model_zoo_version"] == "v5.3.0"
