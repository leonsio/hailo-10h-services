"""Verify installer model preparation without installing packages or downloading weights."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hailo_services.config import LLM_MODEL, Settings
from hailo_services.runtime.models import ModelManager


def installer_config():
    script = (Path(__file__).resolve().parents[1] / "scripts/install.sh").read_text()
    return script.split("<<'CONFIGPY'\n", 1)[1].split("\nCONFIGPY", 1)[0]


@pytest.mark.parametrize(
    ("enabled", "configured", "expected"),
    [(True, "", None), (True, "/custom/gemma.litertlm", "/custom/gemma.litertlm"),
     (False, "/legacy/gemma.litertlm", "/legacy/gemma.litertlm"), (False, "", None)],
)
def test_installer_prepares_only_selected_gemma(
    monkeypatch, capsys, enabled, configured, expected
):
    settings = SimpleNamespace(litert_enabled=enabled, litert_model_path=configured)
    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    monkeypatch.setattr("os.path.isfile", lambda path: False)
    monkeypatch.setattr(ModelManager, "__init__", lambda self, settings: None)
    calls = []

    def resolve(self, model, kind, path):
        calls.append((model, kind, path))
        return Path(path or "/store/gemma.litertlm")

    monkeypatch.setattr(ModelManager, "resolve", resolve)
    exec(compile(installer_config(), "installer-config", "exec"), {})
    output = capsys.readouterr().out
    if enabled or configured:
        assert calls == [(LLM_MODEL, "litert", expected)]
        assert output == f"{expected or '/store/gemma.litertlm'}\nenabled\n"
    else:
        assert calls == []
        assert output == ""


def test_installer_propagates_model_download_failure(monkeypatch):
    monkeypatch.setattr(
        Settings, "from_env",
        lambda: SimpleNamespace(litert_enabled=True, litert_model_path=""),
    )
    monkeypatch.setattr("os.path.isfile", lambda path: False)
    monkeypatch.setattr(ModelManager, "__init__", lambda self, settings: None)

    def fail(*args):
        raise OSError("download unavailable")

    monkeypatch.setattr(ModelManager, "resolve", fail)
    with pytest.raises(OSError, match="download unavailable"):
        exec(compile(installer_config(), "installer-config", "exec"), {})
