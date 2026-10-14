"""Monitoring counters and private dashboard API behavior."""

import time

import pytest
from fastapi.testclient import TestClient
from test_services import FakeBackend, settings

from hailo_services.api.app import create_app
from hailo_services.config import VLM_MODEL
from hailo_services.diagnostics.monitoring import Monitor, instrument
from hailo_services.schemas import ChatRequest


def test_monitoring_counts_chat_stream_and_whisper_once():
    with TestClient(create_app(settings(), FakeBackend())) as client:
        assert client.get("/monitor").status_code == 200
        assert client.get("/ui/monitor.js").status_code == 200
        payload = {"model": VLM_MODEL, "messages": [{"role": "user", "content": "Hi"}]}
        assert client.post("/v1/chat/completions", json=payload).status_code == 200
        assert client.post("/v1/chat/completions", json={**payload, "stream": True}).status_code == 200
        data = client.get("/v1/monitor").json()
        vlm = data["services"]["VLM"]
        assert vlm["requests_1h"] == 2
        assert vlm["errors_1h"] == 0
        assert vlm["average_ms"] >= 0
        assert vlm["last_request_at"]
        assert data["health"]["ready"]
        assert data["resources"]["process_rss_bytes"] > 0


def test_monitor_api_is_authenticated_when_key_is_configured():
    with TestClient(create_app(settings(api_key="secret"), FakeBackend())) as client:
        assert client.get("/monitor").status_code == 200
        assert client.get("/v1/monitor").status_code == 401
        assert client.get("/v1/monitor", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_hour_window_prunes_old_samples():
    monitor = Monitor(settings())
    try:
        started = monitor.begin("YOLO")
        monitor.end("YOLO", started)
        assert monitor.snapshot()["YOLO"]["requests_1h"] == 1
        with monitor.lock:
            monitor.buckets["YOLO"] = {int(time.time()) - 3601: [1, 0, 100, 100, 0, 0]}
        assert monitor.snapshot()["YOLO"]["requests_1h"] == 0
    finally:
        monitor.close()


def test_runtime_observers_capture_ttft_and_detector_failure():
    class RuntimeStub:
        settings = settings()

        def default_chat_request(self, request):
            return request

        async def chat(self, request):
            request._metrics["ttft_ms"] = 12.5
            return "answer"

        async def stream(self, request):
            yield "answer"

        async def transcribe(self, audio):
            return "transcript"

    class VisionStub:
        async def detect_array(self, frame):
            raise RuntimeError("detector unavailable")

    class SpeechStub:
        async def synthesize(self, request):
            return b"sound"

    async def exercise():
        runtime, vision, speech = RuntimeStub(), VisionStub(), SpeechStub()
        monitor = Monitor(settings())
        try:
            instrument(runtime, vision, speech, monitor)
            request = ChatRequest(model=VLM_MODEL, messages=[{"role": "user", "content": "Hi"}])
            assert await runtime.chat(request) == "answer"
            assert monitor.snapshot()["VLM"]["average_ttft_ms"] == 12.5
            with pytest.raises(RuntimeError):
                await vision.detect_array(None)
            assert monitor.snapshot()["YOLO"]["errors_1h"] == 1
            assert monitor.snapshot()["YOLO"]["average_ms"] >= 0
        finally:
            monitor.close()

    import asyncio

    asyncio.run(exercise())
