import pytest

from hailo_services.config import Settings
from hailo_services.models import ModelManager


def test_hailo_54_catalog_excludes_legacy_llama_3b():
    manager = ModelManager(Settings(), "5.4.0")

    with pytest.raises(ValueError, match="Unknown catalogue model"):
        manager.entry("Llama-3_2-3B-Instruct", "llm")

    assert manager.url("Llama3.2-1B-Instruct") == (
        "https://dev-public.hailo.ai/v5.4.0/blob/Llama3.2-1B-Instruct.hef"
    )
