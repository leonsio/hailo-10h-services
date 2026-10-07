import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hailo_services.backend_hailo import HailoBackend
from hailo_services.config import Settings
from hailo_services.models import ModelManager


def test_detector_priority_defaults_to_hailo_maximum():
    assert Settings().vision_scheduler_priority == 31
    with pytest.raises(ValueError, match="between 0 and 31"):
        Settings(vision_scheduler_priority=32)


def test_genai_vdevice_uses_shared_round_robin(monkeypatch):
    round_robin = object()
    observed = {}

    class Params:
        group_id = None
        scheduling_algorithm = None

    class VDevice:
        @staticmethod
        def create_params():
            return Params()

        def __init__(self, params):
            observed["group_id"] = params.group_id
            observed["scheduling_algorithm"] = params.scheduling_algorithm

        def release(self):
            pass

    class VLM:
        def __init__(self, device, path):
            observed["vlm_path"] = path

        def release(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "hailo_platform",
        SimpleNamespace(
            HailoSchedulingAlgorithm=SimpleNamespace(ROUND_ROBIN=round_robin),
            VDevice=VDevice,
        ),
    )
    monkeypatch.setitem(sys.modules, "hailo_platform.genai", SimpleNamespace(VLM=VLM))
    monkeypatch.setattr("hailo_services.backend_hailo.prepare_model_version", lambda: "5.4.0")
    monkeypatch.setattr(ModelManager, "resolve", lambda self, model, kind, path=None: "/tmp/vlm.hef")

    settings = replace(
        Settings(),
        whisper_enabled=False,
        minilm_enabled=False,
        hailo_llm_enabled=False,
        vision_enabled=True,
        vision_scheduler_priority=31,
    )
    backend = HailoBackend(settings)
    backend.start()
    try:
        assert observed["group_id"] == "SHARED"
        assert observed["scheduling_algorithm"] is round_robin
        assert observed["vlm_path"] == "/tmp/vlm.hef"
    finally:
        backend.close()
