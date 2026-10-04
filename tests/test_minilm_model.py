from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hailo_services.minilm import MiniLM
from hailo_services.models import ensure_minilm_hef
from hailo_services.tool_retrieval import retrieve_tools


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


def test_minilm_host_preparation_runs_encoder_and_normalizes_masked_output():
    model = MiniLM.__new__(MiniLM)
    model.word = np.ones((4, 384), dtype=np.float32)
    model.position = np.zeros((128, 384), dtype=np.float32)
    model.segment = np.zeros((2, 384), dtype=np.float32)
    model.gamma = np.ones(384, dtype=np.float32)
    model.beta = np.zeros(384, dtype=np.float32)
    model.tokenizer = SimpleNamespace(encode=lambda _: SimpleNamespace(
        ids=[1, 2] + [0] * 126, attention_mask=[1, 1] + [0] * 126,
        type_ids=[0] * 128,
    ))
    buffers = []

    class Bindings:
        def input(self):
            return SimpleNamespace(set_buffer=lambda frame: buffers.append(frame))

        def output(self, name):
            return SimpleNamespace(get_buffer=lambda: np.tile(
                np.arange(384, dtype=np.float32), (1, 128, 1)
            ))

    binding = Bindings()
    model.infer_model = SimpleNamespace(
        outputs=[SimpleNamespace(name="hidden")],
        output=lambda name: SimpleNamespace(shape=(1, 128, 384)),
    )
    model.configured = SimpleNamespace(
        create_bindings=lambda **kwargs: binding,
        wait_for_async_ready=lambda **kwargs: None,
        run_async=lambda items, callback: (
            callback(completion_info=SimpleNamespace(exception=None))
            or SimpleNamespace(wait=lambda timeout: None)
        ),
    )
    vector = model.embed("hello")
    assert buffers[0].shape == (1, 128, 384)
    assert buffers[0].dtype == np.float32
    assert vector.shape == (384,)
    assert np.linalg.norm(vector) == pytest.approx(1)


def test_semantic_tool_retrieval_limits_tool_schemas():
    class Encoder:
        def embed(self, text):
            return np.array([1., 0.]) if "lamp" in text else np.array([0., 1.])

    tools = [{"type": "function", "function": {
        "name": "lamp" if i == 0 else f"other_{i}", "description": "tool",
        "parameters": {"type": "object"},
    }} for i in range(15)]
    selected, stats = retrieve_tools([{"role": "user", "content": "lamp"}], tools, encoder=Encoder())
    assert selected[0]["function"]["name"] == "lamp"
    # A confident lexical hit is no longer padded with unrelated semantic
    # top-N results just to reach the old fixed 12-tool limit.
    assert stats["tools_after"] == 1
