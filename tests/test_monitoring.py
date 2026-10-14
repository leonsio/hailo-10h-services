"""Monitoring counters and private dashboard API behavior."""

import logging
import multiprocessing as mp
import os
import queue
import time
from concurrent.futures import Future
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_services import FakeBackend, settings

from hailo_services.api.app import create_app
from hailo_services.config import VLM_MODEL
from hailo_services.diagnostics.monitoring import Monitor, RecentLogs, instrument
from hailo_services.runtime.workers.worker import WorkerSupervisor, _finish, _worker_logging
from hailo_services.schemas import ChatRequest


def test_monitoring_counts_chat_stream_and_whisper_once(caplog):
    with caplog.at_level(logging.INFO, logger="hailo_services.diagnostics.monitoring"), TestClient(create_app(settings(), FakeBackend())) as client:
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
        assert any("event=monitor_request service=VLM" in line for line in data["logs"][0]["lines"])


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


def _emit_worker_log(log_queue):
    _worker_logging(False, log_queue)
    logging.getLogger("hailo_services.worker_test").info("worker inference complete")


def test_worker_logs_reach_monitor_capture():
    context = mp.get_context("spawn")
    log_queue = context.Queue(maxsize=16)
    worker = context.Process(target=_emit_worker_log, args=(log_queue,))
    worker.start()
    try:
        record = log_queue.get(timeout=5)
        logs = RecentLogs()
        logs.handle(record)
        group = logs.snapshot({"hailo": {"pid": worker.pid, "status": "ready"}})[1]
        assert "worker inference complete" in group["lines"][0]
        assert f"[{worker.pid}]" in group["lines"][0]
    finally:
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        log_queue.close()
    assert worker.exitcode == 0


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="Linux /proc required")
def test_resources_list_gateway_and_live_worker_separately(monkeypatch):
    original = Path.read_text
    stat = ["S"] + ["0"] * 49
    stat[11], stat[12], stat[19] = "100", "50", "777"

    def read(path, *args, **kwargs):
        if str(path) == "/proc/424242/stat":
            return "424242 (worker) " + " ".join(stat)
        if str(path) == "/proc/424242/statm":
            return "100 50 0 0 0 0 0"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monitor = Monitor(settings())
    try:
        statuses = {"gateway": {"pid": os.getpid(), "status": "ready"},
                    "hailo": {"pid": 424242, "status": "ready"}}
        first = monitor.resources(statuses)["processes"]
        assert {row["name"] for row in first} == {"gateway", "hailo"}
        assert all(isinstance(row["rss_bytes"], int) and row["rss_bytes"] > 0 for row in first), first
        time.sleep(0.02)
        second = monitor.resources(statuses)["processes"]
        assert all(row["cpu_percent"] is not None for row in second)
    finally:
        monitor.close()


def test_logs_keep_fifty_lines_per_process():
    logs = RecentLogs()
    for number in range(52):
        record = logging.LogRecord("hailo_services.worker", logging.INFO, __file__, 1, f"line {number}", (), None)
        record.process = 424242
        logs.handle(record)
    result = logs.snapshot({"hailo": {"pid": 424242, "status": "ready"}})
    assert result[0]["name"] == "gateway"
    assert result[0]["lines"] == []
    assert len(result[1]["lines"]) == 50
    assert "line 2" in result[1]["lines"][0]


def test_supervisor_forwards_worker_records_into_process_group():
    supervisor = WorkerSupervisor(settings())
    logs = RecentLogs()
    supervisor.set_log_sink(logs)
    try:
        record = logging.LogRecord("hailo_services.worker", logging.INFO, __file__, 1, "queued worker log", (), None)
        record.process = 424242
        supervisor.log_queue.put(record)
        deadline = time.monotonic() + 2
        while not logs.snapshot({"hailo": {"pid": 424242, "status": "ready"}})[1]["lines"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert "queued worker log" in logs.snapshot({"hailo": {"pid": 424242, "status": "ready"}})[1]["lines"][0]
    finally:
        supervisor.set_log_sink(None)


def test_worker_completion_emits_activity_without_debug_logging(caplog):
    future = Future()
    future.set_result("answer")
    responses = queue.Queue()
    with caplog.at_level(logging.INFO, logger="hailo_services.runtime.workers.worker"):
        _finish(responses, "test-id", future, (), "chat", time.perf_counter())
    assert responses.get_nowait()["value"] == "answer"
    assert "event=worker_request operation=chat status=completed" in caplog.text
