import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hailo_services.app import create_app
from hailo_services.config import Settings
from hailo_services.models import ModelManager
from hailo_services.protocols import wyoming_info
from hailo_services.runtime import HailoBackend, Runtime


def test_yaml_selects_models_and_env_overrides(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text('''settings:
  port: 8091
  litert_max_input_tokens: 4096
models:
  vlm: {enabled: true, model: Qwen3-VL-2B-Instruct}
  whisper: {enabled: true, model: Whisper-Small}
  gemma: {enabled: false}
  minilm: {enabled: false}
''')
    monkeypatch.setenv("HAILO_CONFIG", str(path))
    monkeypatch.setenv("HAILO_PORT", "8092")
    s = Settings.from_env()
    assert s.port == 8092
    assert s.vlm_model == "Qwen3-VL-2B-Instruct"
    assert s.stt_model == "whisper-small"
    assert not s.litert_enabled and not s.minilm_enabled


@pytest.mark.parametrize("text", [
    "settings: {typo: 1}", "models: {typo: {enabled: true}}",
    "settings: {port: true}", 'settings: {debug_log: "false"}',
    "models: {gemma: {enabled: 'false'}}", "[]",
    "models: {whisper: {model: Whisper-Small, typo: 1}}",
])
def test_yaml_rejects_invalid_config(tmp_path, monkeypatch, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    monkeypatch.setenv("HAILO_CONFIG", str(path))
    with pytest.raises(ValueError):
        Settings.from_env()


def test_all_current_hailo_models_have_documented_54_urls():
    manager = ModelManager(Settings(vlm_release="v5.4.0"), "5.4.0")
    models = ["Qwen2-VL-2B-Instruct", "Qwen3-VL-2B-Instruct",
              "Whisper-Tiny", "Whisper-Base", "Whisper-Small",
              "DeepSeek-R1-Distill-Qwen-1.5B", "Llama3.2-1B-Instruct",
              "Qwen2-1.5B-Instruct", "Qwen2-1.5B-Instruct-Function-Calling-v1",
              "Qwen2.5-1.5B-Instruct", "Qwen2.5-Coder-1.5B-Instruct", "Qwen3-1.7B-Instruct"]
    for model in models:
        assert manager.url(model) == f"https://dev-public.hailo.ai/v5.4.0/blob/{model}.hef"
    with pytest.raises(ValueError, match="No known model release"):
        ModelManager(Settings(), "5.2.0").url("Qwen3-VL-2B-Instruct")
    with pytest.raises(ValueError, match="No known model release"):
        ModelManager(Settings(), "5.5.0").url("Whisper-Small")
    with pytest.raises(ValueError, match="not a whisper"):
        manager.entry("Qwen3-VL-2B-Instruct", "whisper")


def test_cached_named_models_reused_and_download_is_bounded(tmp_path, monkeypatch):
    s = replace(Settings(), model_store=str(tmp_path))
    manager = ModelManager(s, "5.4.0")
    path = tmp_path / "Whisper-Small.hef"
    with path.open('wb') as stream:
        stream.truncate(manager.entry('Whisper-Small')['sizes']['v5.4.0'])
    monkeypatch.setattr("hailo_services.models.urllib.request.urlopen",
                        lambda *a, **k: pytest.fail("cached model caused network call"))
    assert manager.resolve("Whisper-Small", "whisper") == path
    assert manager.resolve(str(path), "whisper") == path
    with pytest.raises(FileNotFoundError):
        manager.resolve(str(tmp_path / "missing.hef"))


def test_disabled_models_never_download_or_allocate(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "hailo_platform", SimpleNamespace(VDevice=SimpleNamespace(
        create_params=lambda: pytest.fail("disabled models allocated accelerator"))))
    monkeypatch.setitem(sys.modules, "hailo_platform.genai", SimpleNamespace(VLM=None, Speech2Text=None))
    monkeypatch.setattr("hailo_services.runtime.prepare_model_version", lambda: "5.4.0")
    monkeypatch.setattr(ModelManager, "resolve", lambda *a, **k: pytest.fail("disabled model downloaded"))
    s = replace(Settings(), vlm_enabled=False, whisper_enabled=False, minilm_enabled=False)
    backend = HailoBackend(s)
    backend.start()
    assert not backend.paths and backend.device is None
    backend.close()
    assert wyoming_info(s).asr == []


def test_selected_models_appear_in_http_and_wyoming():
    class Backend:
        paths = {}
        def start(self):
            pass
        def close(self):
            pass
        def chat(self, request, emit=None, cancelled=None):
            return request.model
    s = replace(Settings(), vlm_hef="Qwen3-VL-2B-Instruct", whisper_hef="Whisper-Small",
                wyoming_port=0)
    with TestClient(create_app(s, backend=Backend())) as client:
        ids = {m["id"] for m in client.get('/v1/models').json()['data']}
        assert ids == {"Qwen3-VL-2B-Instruct", "whisper-small"}
        assert client.get('/ui/config').json()['whisper_model'] == 'whisper-small'
        response = client.post('/v1/chat/completions', json={
            "model": "Qwen3-VL-2B-Instruct", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200
    assert wyoming_info(s).asr[0].models[0].name == 'whisper-small'


def test_gemma_enable_download_setup_and_disable_legacy_path(tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'
    path.write_text('models: {gemma: {enabled: true}}')
    monkeypatch.setenv('HAILO_CONFIG', str(path))
    runtime = Runtime(Settings.from_env())
    assert runtime.litert_backend is not None
    assert runtime.litert_backend.model_manager.url('gemma-4-E2B-it').endswith('gemma-4-E2B-it.litertlm')
    asyncio.run(runtime.close())
    path.write_text('settings: {litert_model_path: /old/gemma.litertlm}\nmodels: {gemma: {enabled: false}}')
    s = Settings.from_env()
    assert s.litert_model_path == ''
    runtime = Runtime(s)
    assert runtime.litert_backend is None
    asyncio.run(runtime.close())


def test_exact_size_rejects_partial_cache_and_failed_replacement_preserves_it(tmp_path, monkeypatch):
    from hailo_services.models import ensure_model_file
    path = tmp_path / 'model.hef'
    path.write_bytes(b'old')
    class Response:
        headers = {'Content-Length': '4'}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size):
            return b''
    monkeypatch.setattr('hailo_services.models.urllib.request.urlopen', lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match='Incomplete'):
        ensure_model_file(path, 'https://example.test/model', 1, 10, expected_size=4)
    assert path.read_bytes() == b'old'
    assert list(tmp_path.iterdir()) == [path]


def test_vlm_uses_selected_model_frame_shape():
    import base64
    import io
    from contextlib import contextmanager

    from PIL import Image

    from hailo_services.schemas import ChatRequest
    class VLM:
        def tokenize(self, text):
            return text.split()
        def input_frame_shape(self):
            return [448, 448, 3]
        def clear_context(self):
            pass
        @contextmanager
        def generate(self, **kwargs):
            assert kwargs['frames'][0].shape == (448, 448, 3)
            yield iter(['image'])
    buffer = io.BytesIO()
    Image.new('RGB', (30, 20)).save(buffer, format='PNG')
    backend = HailoBackend(replace(Settings(), vlm_hef='Qwen3-VL-2B-Instruct'))
    backend.vlm = VLM()
    request = ChatRequest(model='Qwen3-VL-2B-Instruct', messages=[{'role': 'user', 'content': [{
        'type': 'image_url', 'image_url': {'url': base64.b64encode(buffer.getvalue()).decode()}}]}])
    assert backend.chat(request) == 'image'


@pytest.mark.parametrize('vlm_enabled', [True, False])
def test_hailo_hef_llm_is_blocked_even_without_vlm(vlm_enabled):
    with pytest.raises(ValueError, match='HEF LLM support is disabled'):
        Settings(vlm_enabled=vlm_enabled, hailo_llm_enabled=True)


@pytest.mark.parametrize('content', [
    'models: {hailo_llm: {enabled: true}}',
    'settings: {hailo_llm_enabled: true, vlm_enabled: false}',
])
def test_yaml_cannot_enable_hailo_llm(tmp_path, monkeypatch, content):
    path = tmp_path / 'config.yaml'
    path.write_text(content)
    monkeypatch.setenv('HAILO_CONFIG', str(path))
    with pytest.raises(ValueError, match='HEF LLM support is disabled'):
        Settings.from_env()


def test_env_cannot_enable_hailo_llm(tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'
    path.write_text('models: {hailo_llm: {enabled: false}}')
    monkeypatch.setenv('HAILO_CONFIG', str(path))
    monkeypatch.setenv('HAILO_HAILO_LLM_ENABLED', 'true')
    with pytest.raises(ValueError, match='HEF LLM support is disabled'):
        Settings.from_env()


def test_backend_guard_runs_before_vendor_import_or_download(monkeypatch):
    s = Settings()
    # Also protect startup if an embedding application bypasses frozen Settings.
    object.__setattr__(s, 'hailo_llm_enabled', True)
    monkeypatch.setattr(ModelManager, 'resolve', lambda *a, **k: pytest.fail('download started'))
    with pytest.raises(ValueError, match='HEF LLM support is disabled'):
        HailoBackend(s).start()


def test_yaml_example_covers_every_legacy_env_parameter(monkeypatch):
    import re
    from pathlib import Path

    import yaml
    root = Path(__file__).resolve().parents[1]
    path = root / 'deploy/hailo-10h-services.yaml.example'
    document = yaml.safe_load(path.read_text())
    covered = set(document['settings'])
    for role, fields in {
        'vlm': ('vlm_enabled', 'vlm_hef', 'vlm_release', 'vlm_max_input_tokens'),
        'whisper': ('whisper_enabled', 'whisper_hef'),
        'hailo_llm': ('hailo_llm_enabled', 'hailo_llm_model', 'hailo_llm_max_input_tokens'),
        'minilm': ('minilm_enabled', 'minilm_hef_path'),
        'gemma': ('litert_enabled', 'litert_model_path', 'litert_max_input_tokens'),
    }.items():
        assert role in document['models']
        covered.update(fields)
    legacy = {match.lower() for match in re.findall(
        r'^HAILO_(\w+)=', (root / 'deploy/hailo-10h-services.env.example').read_text(), re.M)}
    assert legacy <= covered
    assert set(vars(Settings())) <= covered
    monkeypatch.setenv('HAILO_CONFIG', str(path))
    s = Settings.from_env()
    assert s.mqtt_port == 1883 and s.mcp_hosts.endswith('hailo.local:*')
    # Both Hailo VLM and CPU Gemma are explicitly allowed together.
    s = replace(s, litert_enabled=True)
    runtime = Runtime(s)
    assert runtime.hailo_chat_models == ['Qwen2-VL-2B-Instruct']
    assert runtime.litert_backend is not None
    asyncio.run(runtime.close())
