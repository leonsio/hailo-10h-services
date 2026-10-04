from pathlib import Path

import pytest

from hailo_services.models import ensure_minilm_hef


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.offset = 0
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        chunk = self.payload[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_minilm_hef_downloads_atomically_to_shared_models_path(tmp_path, monkeypatch):
    destination = tmp_path / "hailo10h" / "minilm-l6-ruvector.hef"
    payload = b"h" * (1024 * 1024)
    opened = []

    def fake_urlopen(request, timeout):
        opened.append((request.full_url, timeout))
        return Response(payload)

    monkeypatch.setattr("hailo_services.models.urllib.request.urlopen", fake_urlopen)
    resolved = ensure_minilm_hef(destination, "https://models.example/minilm.hef")
    assert resolved == destination
    assert destination.read_bytes() == payload
    assert opened == [("https://models.example/minilm.hef", 120)]
    assert list(destination.parent.iterdir()) == [destination]
    assert ensure_minilm_hef(destination, "https://models.example/unused.hef") == destination
    assert len(opened) == 1


def test_minilm_hef_rejects_an_incomplete_download_and_cleans_temp_file(tmp_path, monkeypatch):
    destination = tmp_path / "minilm-l6-ruvector.hef"
    monkeypatch.setattr(
        "hailo_services.models.urllib.request.urlopen",
        lambda *args, **kwargs: Response(b"too short"),
    )
    with pytest.raises(RuntimeError, match="Incomplete MiniLM HEF"):
        ensure_minilm_hef(destination, "https://models.example/minilm.hef")
    assert not destination.exists()
    assert list(Path(tmp_path).iterdir()) == []
