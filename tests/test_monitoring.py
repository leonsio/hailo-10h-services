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
from hailo_services.diagnostics.hailo_device import HailoTelemetry, read_device, resident_models
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
        assert "hailo_device" in data
        assert data["hailo_device"]["nnc_percent"] is None
        assert data["system_services"]["wyoming"]["status"] == "disabled"
        assert data["system_services"]["http"]["port"] == 8090
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


def test_logs_keep_hundred_lines_per_process():
    logs = RecentLogs()
    for number in range(102):
        record = logging.LogRecord("hailo_services.worker", logging.INFO, __file__, 1, f"line {number}", (), None)
        record.process = 424242
        logs.handle(record)
    result = logs.snapshot({"hailo": {"pid": 424242, "status": "ready"}})
    assert result[0]["name"] == "gateway"
    assert result[0]["lines"] == []
    assert len(result[1]["lines"]) == 100
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


def test_settings_ui_validates_saves_and_hides_secrets(tmp_path, monkeypatch):
    import yaml
    path = tmp_path / "config.yaml"
    path.write_text('settings:\n  api_key: "secret"\n  mqtt_password: "private"\n  debug_log: false\nmodels:\n  vlm:\n    enabled: true\n', encoding="utf-8")
    monkeypatch.setenv("HAILO_CONFIG", str(path))
    with TestClient(create_app(settings(api_key="secret"), FakeBackend())) as client:
        assert client.get("/settings").status_code == 200
        assert client.get("/v1/settings").status_code == 401
        auth = {"Authorization": "Bearer secret"}
        response = client.get("/v1/settings", headers=auth).json()
        fields = {entry["path"]: entry for entry in response["fields"]}
        assert fields["settings.api_key"]["value"] is None
        assert fields["settings.mqtt_password"]["value"] is None
        assert fields["settings.api_key"]["secret_set"]
        bad = client.put("/v1/settings", headers=auth, json={"version": response["version"], "changes": {"settings.debug_log": "yes"}})
        assert bad.status_code == 400
        assert yaml.safe_load(path.read_text())["settings"]["debug_log"] is False
        saved = client.put("/v1/settings", headers=auth, json={"version": response["version"], "changes": {"models.vlm.enabled": False, "settings.debug_log": True}})
        assert saved.status_code == 200
        result = yaml.safe_load(path.read_text())
        assert result["settings"]["api_key"] == "secret"
        assert result["settings"]["debug_log"] is True
        assert result["models"]["vlm"]["enabled"] is False
        assert client.put("/v1/settings", headers=auth, json={"version": response["version"], "changes": {"settings.debug_log": False}}).status_code == 409
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        assert client.post("/v1/settings/restart", headers=auth).status_code == 409


def test_monitor_polling_is_absent_from_gateway_logs(caplog):
    with caplog.at_level(logging.DEBUG, logger="hailo_services.api.app"), TestClient(create_app(settings(debug_log=True), FakeBackend())) as client:
        for _ in range(2):
            client.get("/v1/monitor")
        assert "path=/v1/monitor" not in caplog.text
        assert not any("path=/v1/monitor" in line for line in client.get("/v1/monitor").json()["logs"][0]["lines"])


def test_settings_write_rejects_invalid_model_combination(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("models:\n  vlm:\n    enabled: true\n  hailo_llm:\n    enabled: false\n", encoding="utf-8")
    monkeypatch.setenv("HAILO_CONFIG", str(path))
    with TestClient(create_app(settings(), FakeBackend())) as client:
        version = client.get("/v1/settings").json()["version"]
        result = client.put("/v1/settings", json={"version": version, "changes": {"models.hailo_llm.enabled": True}})
        assert result.status_code == 400
        assert "enabled: false" in path.read_text()



def test_hailort_control_stats_are_normalized_to_bytes_and_zero_is_preserved():
    from types import SimpleNamespace
    control = SimpleNamespace(
        query_performance_stats=lambda sampling_period_ms: SimpleNamespace(
            cpu_utilization=12, nnc_utilization=37, dsp_utilization=0,
            ram_size_used=5 * 1024 * 1024, ram_size_total=8 * 1024 * 1024),
        get_chip_temperature=lambda: SimpleNamespace(ts0_temperature=51, ts1_temperature=54),
        get_throttling_state=lambda: False,
    )
    result = read_device(control)
    assert result["state"] == "ready"
    assert result["nnc_percent"] == 37
    assert result["cpu_percent"] == 12
    assert "dsp_percent" not in result
    assert result["ram_used_bytes"] == 5 * 1024 ** 3
    assert result["ram_total_bytes"] == 8 * 1024 ** 3
    assert result["temperature_c"] == 54
    assert result["throttling"] is False


def test_unsupported_control_read_does_not_erase_supported_stats_or_retry():
    from types import SimpleNamespace
    calls = []
    def unsupported():
        calls.append(True)
        raise RuntimeError("HAILO_INVALID_OPERATION")
    control = SimpleNamespace(
        query_performance_stats=lambda sampling_period_ms: SimpleNamespace(
            cpu_utilization=-1, nnc_utilization=0, dsp_utilization=float("nan"),
            ram_size_used=-1, ram_size_total=8 * 1024 * 1024),
        get_chip_temperature=unsupported,
        get_throttling_state=unsupported,
    )
    unavailable = {}
    for _ in range(2):
        result = read_device(control, unavailable)
        assert result["state"] == "partial"
        assert result["nnc_percent"] == 0
        assert result["cpu_percent"] is None
        assert "dsp_percent" not in result
        assert result["temperature_c"] is None
        assert result["throttling"] is None
        assert "temperature" in result["errors"]
    assert len(calls) == 2  # Once per unsupported API family, not on every refresh.


def test_enabled_thermal_protection_does_not_mean_actual_throttling():
    from types import SimpleNamespace
    control = SimpleNamespace(get_throttling_state=lambda: True)
    control._get_health_information = lambda: SimpleNamespace(current_temperature_throttling_level=-1, current_overcurrent_zone=0)
    assert read_device(control)["throttling"] is False
    control._get_health_information = lambda: SimpleNamespace(current_temperature_throttling_level=0, current_overcurrent_zone=0)
    assert read_device(control)["throttling"] is True


def test_collector_owns_read_handle_and_returns_cached_snapshot():
    import threading
    from types import SimpleNamespace
    published = threading.Event()
    released = []
    queries = []
    def query(sampling_period_ms):
        queries.append(threading.current_thread().name)
        return SimpleNamespace(nnc_utilization=37)
    control = SimpleNamespace(query_performance_stats=query, get_chip_temperature=lambda: None, get_throttling_state=lambda: False)
    physical = SimpleNamespace(control=control, release=lambda: released.append(True))
    collector = HailoTelemetry(lambda snapshot: published.set(), interval=60)
    collector.start(SimpleNamespace(get_physical_devices=lambda: [physical]))
    try:
        assert published.wait(timeout=2)
        for _ in range(10):
            assert collector.snapshot()["nnc_percent"] == 37
        assert queries == ["hailo-device-telemetry"]
    finally:
        collector.close()
    assert released == [True]


def test_resident_models_depend_on_successful_native_loads():
    from types import SimpleNamespace
    chat = SimpleNamespace(vlm=object(), whisper=object(), llm=None, minilm=None)
    vision = SimpleNamespace(configured_model=object())
    models = resident_models(chat, vision, settings())
    assert {model["role"] for model in models} == {"VLM", "Whisper", "YOLO"}
    assert all(model["state"] == "loaded" for model in models)
    chat.vlm = None
    assert "VLM" not in {model["role"] for model in resident_models(chat, vision, settings())}


def test_disabled_services_are_marked_and_metrics_remain_available():
    monitor = Monitor(settings(vlm_enabled=False, whisper_enabled=False))
    try:
        services = monitor.snapshot()
        assert not services["VLM"]["enabled"]
        assert not services["Whisper"]["enabled"]
        assert not services["LLM"]["enabled"]
        assert not services["Piper"]["enabled"]
        assert services["HA-Assist"]["enabled"]
    finally:
        monitor.close()


def test_worker_listener_receives_telemetry_without_an_inference_request():
    from types import SimpleNamespace

    from hailo_services.runtime.workers.worker import _WorkerClient

    client = _WorkerClient("hailo", settings())
    alive = [False]
    client.process = SimpleNamespace(is_alive=lambda: alive[0])
    client.response_queue = queue.Queue()
    client.metadata = {"resident_models": [{"name": "YOLOv11m", "state": "loaded"}]}
    client.response_queue.put({"kind": "telemetry", "value": {"nnc_percent": 37, "state": "partial"}})
    client._listen()
    alive[0] = True
    assert client.telemetry_status()["nnc_percent"] == 37
    assert client.telemetry_status()["resident_models"][0]["name"] == "YOLOv11m"
    alive[0] = False
    assert client.telemetry_status()["nnc_percent"] is None
    assert client.telemetry_status()["resident_models"] == []


def test_settings_model_choices_match_catalogue_roles_and_validate_writes(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("settings:\n  frigate_assist_text_model: null\nmodels:\n  vlm:\n    model: Qwen2-VL-2B-Instruct\n  whisper:\n    model: Whisper-Base\nvision:\n  model: yolov11m\n", encoding="utf-8")
    monkeypatch.setenv("HAILO_CONFIG", str(path))
    with TestClient(create_app(settings(), FakeBackend())) as client:
        data = client.get("/v1/settings").json()
        fields = {field["path"]: field for field in data["fields"]}
        assert set(fields["models.vlm.model"]["options"]) == {"Qwen2-VL-2B-Instruct", "Qwen3-VL-2B-Instruct"}
        assert set(fields["models.whisper.model"]["options"]) == {"Whisper-Base", "Whisper-Small", "Whisper-Tiny"}
        assert "yolov11m" in fields["vision.model"]["options"]
        assert "gemma-4-E2B-it" in fields["settings.frigate_assist_text_model"]["options"]
        assert fields["settings.frigate_assist_text_model"]["allow_empty"]
        bad = client.put("/v1/settings", json={"version": data["version"], "changes": {"models.vlm.model": "Whisper-Base"}})
        assert bad.status_code == 400
        saved = client.put("/v1/settings", json={"version": data["version"], "changes": {"models.vlm.model": "Qwen3-VL-2B-Instruct", "settings.frigate_assist_text_model": "gemma-4-E2B-it"}})
        assert saved.status_code == 200


def test_settings_choices_use_custom_catalogue_without_reading_model_files(tmp_path, monkeypatch):
    catalog = tmp_path / "models.yaml"
    catalog.write_text("schema_version: 1\nmodels:\n  custom-vlm:\n    kind: vlm\n  custom-whisper:\n    kind: whisper\n  ignored-asset:\n    kind: asset\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(f"settings:\n  model_catalog: {catalog}\nmodels:\n  vlm:\n    model: custom-vlm\n", encoding="utf-8")
    monkeypatch.setenv("HAILO_CONFIG", str(config))
    with TestClient(create_app(settings(), FakeBackend())) as client:
        data = client.get("/v1/settings").json()
        fields = {field["path"]: field for field in data["fields"]}
        assert fields["models.vlm.model"]["options"] == ["custom-vlm"]
        assert data["catalog_error"] is None


def test_monitor_service_caption_lists_only_successfully_loaded_models():
    backend = FakeBackend()
    backend.vlm = object()
    with TestClient(create_app(settings(vlm_hef="Qwen3-VL-2B-Instruct"), backend)) as client:
        data = client.get("/v1/monitor").json()
        assert data["services"]["VLM"]["models"] == ["Qwen3-VL-2B-Instruct"]
        assert data["services"]["Whisper"]["models"] == []
        backend.vlm = None
        assert client.get("/v1/monitor").json()["services"]["VLM"]["models"] == []


def test_gui_available_while_native_models_are_loading():
    import threading
    entered, release = threading.Event(), threading.Event()
    backend = FakeBackend()
    def slow_start():
        entered.set()
        assert release.wait(5)
    backend.start = slow_start
    app = create_app(settings(), backend)
    with TestClient(app) as client:
        try:
            assert entered.wait(2)
            assert client.get('/monitor').status_code == 200
            assert client.get('/health').status_code == 503
            data = client.get('/v1/monitor').json()
            assert data['startup']['state'] == 'loading'
            assert not data['health']['ready']
            payload = {'model': VLM_MODEL, 'messages': [{'role': 'user', 'content': 'Hi'}]}
            assert client.post('/v1/chat/completions', json=payload).status_code == 503
        finally:
            release.set()
            async def wait():
                await app.state.init_task
            client.portal.call(wait)
        assert client.get('/health').status_code == 200


def test_model_startup_error_keeps_monitoring_and_cpu_backend_available():
    from test_services import FakeLiteRTBackend
    backend = FakeBackend()
    def failed_start():
        raise RuntimeError('bad native model')
    backend.start = failed_start
    app = create_app(settings(litert_enabled=True), backend, FakeLiteRTBackend())
    with TestClient(app) as client:
        async def wait():
            await app.state.init_task
        client.portal.call(wait)
        assert client.get('/monitor').status_code == 200
        data = client.get('/v1/monitor').json()
        assert data['startup']['state'] == 'error'
        assert 'bad native model' in data['startup']['errors']['Hailo']
        payload = {'model': 'gemma-4-E2B-it', 'messages': [{'role': 'user', 'content': 'Hi'}]}
        assert client.post('/v1/chat/completions', json=payload).status_code == 200


@pytest.mark.parametrize('assistant', ['HA-Assist', 'Frigate-Assist'])
@pytest.mark.parametrize('direct', [True, False])
def test_assistant_counts_actual_backend_in_addition_to_public_service(monkeypatch, assistant, direct):
    import asyncio

    from hailo_services.runtime.runtime import Runtime
    from hailo_services.speech.speech_runtime import SpeechRuntime
    from hailo_services.vision.vision import VisionRuntime
    cfg = settings(ha_assist_enabled=True, frigate_assist_enabled=True)
    runtime = Runtime(cfg, FakeBackend())
    monitor = Monitor(cfg)
    instrument(runtime, VisionRuntime(cfg), SpeechRuntime(cfg), monitor)
    if assistant == 'HA-Assist':
        monkeypatch.setattr('hailo_services.assistants.ha.ha_assist.target_model', lambda *_: VLM_MODEL)
        monkeypatch.setattr('hailo_services.assistants.ha.ha_assist.prepare', lambda _, request: (request, 'direct' if direct else None))
    else:
        monkeypatch.setattr('hailo_services.assistants.frigate.frigate_assist.prepare', lambda _, request: (request.model_copy(update={'model': VLM_MODEL}), 'direct' if direct else None))
        monkeypatch.setattr('hailo_services.assistants.frigate.frigate_routing.validate_result', lambda result, *_: result)
    async def run():
        await runtime.start_hailo()
        try:
            await runtime.chat(ChatRequest(model=assistant, messages=[{'role': 'user', 'content': 'Hi'}]))
            await asyncio.sleep(0)
            data = monitor.snapshot()
            assert data[assistant]['requests_1h'] == 1
            assert data['VLM']['requests_1h'] == (0 if direct else 1)
        finally:
            await runtime.close()
            monitor.close()
    asyncio.run(run())


def test_gateway_only_settings_are_saved_and_applied_without_restart(tmp_path, monkeypatch):
    import pickle

    from hailo_services.config import Settings
    path = tmp_path / 'config.yaml'
    path.write_text('settings: {request_timeout: 120, piper_max_input_chars: 4000, ha_wait_messages: true}\n')
    monkeypatch.setenv('HAILO_CONFIG', str(path))
    app = create_app(Settings.from_env(), FakeBackend())
    with TestClient(app) as client:
        data = client.get('/v1/settings').json()
        assert next(field for field in data['fields'] if field['path'] == 'settings.request_timeout')['runtime']
        saved = client.put('/v1/settings', json={'version': data['version'], 'changes': {'settings.request_timeout': 45, 'settings.ha_wait_messages': False}})
        assert saved.status_code == 200
        assert not saved.json()['restart_required']
        assert app.state.runtime.settings.request_timeout == 45
        assert not app.state.runtime.settings.ha_wait_messages
        assert Settings.from_env().request_timeout == 45
        assert pickle.loads(pickle.dumps(app.state.runtime.settings)).request_timeout == 45
