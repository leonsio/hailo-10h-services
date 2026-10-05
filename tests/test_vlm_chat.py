import asyncio
import base64
import io
import json
from contextlib import contextmanager
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hailo_services.app import create_app
from hailo_services.config import LLM_MODEL, Settings
from hailo_services.input_budget import InputBudgetError
from hailo_services.models import ModelManager
from hailo_services.runtime import HailoBackend, Runtime
from hailo_services.schemas import ChatRequest
from hailo_services.vlm_chat import limit_request, model_prompt, tool_response


class VLM:
    def __init__(self, text='Hallo', tokens=None):
        self.text = text
        self.tokens = tokens
        self.calls = []
        self.clears = 0

    def tokenize(self, text):
        return range(self.tokens) if self.tokens is not None else text.split()

    def max_context_capacity(self):
        return 2048

    def prompt_template(self):
        return "{% for m in messages %}<|im_start|>{{ m.role }}\n{% for p in m.content %}{{ p.text|default('<image>') }}{% endfor %}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"

    def clear_context(self):
        self.clears += 1

    @contextmanager
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield iter([self.text])


def backend(model='Qwen3-VL-2B-Instruct', **kwargs):
    settings = Settings(vlm_hef=model, whisper_enabled=False, minilm_enabled=False, wyoming_port=0)
    value = HailoBackend(settings)
    value.vlm = VLM(**kwargs)
    value.start = lambda: None
    return value


def request(**kwargs):
    return ChatRequest(messages=[{'role': 'user', 'content': 'Hallo'}], **kwargs)


def tool(name='get_time'):
    return {'type': 'function', 'function': {'name': name, 'parameters': {
        'type': 'object', 'properties': {'zone': {'type': 'string'}}, 'required': ['zone'],
        'additionalProperties': False}}}


def image():
    buffer = io.BytesIO()
    Image.new('RGB', (50, 40), 'red').save(buffer, format='PNG')
    return {'type': 'image_url', 'image_url': {'url': base64.b64encode(buffer.getvalue()).decode()}}


@pytest.mark.parametrize('model, shape', [
    ('Qwen2-VL-2B-Instruct', (336, 336, 3)), ('Qwen3-VL-2B-Instruct', (288, 512, 3))])
def test_catalogue_frame_size_without_native_shape(model, shape):
    b = backend(model)
    r = request().model_copy(update={'messages': [{'role': 'user', 'content': [image()]}]})
    assert b.chat(r) == 'Hallo'
    frame = b.vlm.calls[0]['frames'][0]
    assert frame.shape == shape
    assert frame.flags.writeable and frame.flags.c_contiguous


def test_qwen3_rejects_multiple_images_before_generation():
    b = backend()
    r = request().model_copy(update={'messages': [{'role': 'user', 'content': [image(), image()]}]})
    with pytest.raises(ValueError, match='at most 1'):
        b.chat(r)
    assert not b.vlm.calls


@pytest.mark.parametrize('tokens, accepted', [(1663, True), (1664, False)])
def test_native_token_boundary_reserves_output_and_template(tokens, accepted):
    b = backend(tokens=tokens)
    r = request(max_input_tokens=8192, max_tokens=256)
    if accepted:
        assert b.chat(r) == 'Hallo'
        assert b.vlm.calls[0]['frames'] == []
    else:
        with pytest.raises(InputBudgetError) as error:
            b.chat(r)
        assert error.value.limit == 1791
        assert not b.vlm.calls


def test_client_can_only_lower_vlm_input_limit_and_native_capacity_wins():
    model = VLM(tokens=180)
    with pytest.raises(InputBudgetError) as error:
        limit_request(model, request(max_input_tokens=300), 2048, 2048)
    assert error.value.limit == 300
    model.max_context_capacity = lambda: 512
    with pytest.raises(InputBudgetError) as error:
        limit_request(model, request(), 2048, 2048)
    assert error.value.limit == 255


def test_old_turns_trimmed_with_current_tool_round_retained():
    r = request(tools=[tool()]).model_copy(update={'messages': [
        {'role': 'user', 'content': 'OLD ' * 2500}, {'role': 'assistant', 'content': 'Old answer'},
        {'role': 'user', 'content': 'Current request'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'x', 'type': 'function', 'function': {'name': 'get_time', 'arguments': '{"zone":"UTC"}'}}]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': '{"time":"12:00"}'},
    ]})
    trimmed, prompt = limit_request(VLM(), r, 2048, 2048)
    assert trimmed.messages == r.messages[2:]
    assert '12:00' in json.dumps(prompt)
    bad = r.model_copy(update={'messages': r.messages[:-1]})
    with pytest.raises(ValueError, match='Missing results'):
        limit_request(VLM(), bad, 2048, 2048)


def test_native_template_is_counted_including_tools_and_debug_prompt(caplog):
    model = VLM()
    captured = []
    model.tokenize = lambda text: captured.append(text) or text.split()
    r = request(tools=[tool()], tool_choice='required', parallel_tool_calls=False)
    with caplog.at_level('DEBUG'):
        limit_request(model, r, 2048, 2048, debug=True)
    assert 'get_time' in captured[0] and 'at most one' in captured[0]
    assert '<|im_start|>assistant' in captured[0]
    assert 'rendered_prompt' in caplog.text


def test_missing_tokenizer_fails_before_generation():
    b = backend()
    b.vlm.tokenize = None
    with pytest.raises(ValueError, match='VLM.tokenize'):
        b.chat(request())
    assert not b.vlm.calls


def test_json_tool_calls_validated_and_buffered():
    text = json.dumps({'tool_calls': [{'function': {'name': 'get_time', 'arguments': {'zone': 'UTC'}}}]})
    b = backend(text=text)
    r = request(tools=[tool()], tool_choice='required')
    emitted = []
    result = b.chat(r, emitted.append)
    assert result['tool_calls'][0]['function']['name'] == 'get_time'
    assert emitted == [result]
    assert b.vlm.clears == 2
    assert tool_response('```json\n' + text + '\n```', r)['tool_calls']
    b.vlm.text = text.replace('get_time', 'unknown')
    emitted.clear()
    with pytest.raises(ValueError, match='unavailable function'):
        b.chat(r, emitted.append)
    assert emitted == []


@pytest.mark.parametrize('response', [
    '{"tool_calls":[{"function":{"name":"get_time","arguments":{}}}]}',
    'I called it successfully.',
])
def test_invalid_or_missing_required_tool_call(response):
    with pytest.raises(ValueError):
        tool_response(response, request(tools=[tool()], tool_choice='required'))


def test_tool_choice_none_excludes_schema_from_model_prompt():
    prompt = model_prompt(request(tools=[tool()], tool_choice='none'))
    assert 'get_time' not in json.dumps(prompt)


def test_default_text_route_prefers_ready_gemma_then_vlm_but_images_use_vlm():
    runtime = Runtime(Settings(vlm_hef='Qwen3-VL-2B-Instruct'))
    assert runtime.default_chat_request(request()).model == 'Qwen3-VL-2B-Instruct'
    runtime.litert_ready = True
    assert runtime.default_chat_request(request()).model == LLM_MODEL
    r = ChatRequest(messages=[{'role': 'user', 'content': [image()]}])
    assert runtime.default_chat_request(r).model == 'Qwen3-VL-2B-Instruct'
    assert runtime.default_chat_request(request(model=LLM_MODEL)).model == LLM_MODEL
    runtime.litert_ready = False
    asyncio.run(runtime.close())


def test_text_only_http_stream_ui_metadata_and_unavailable_explicit_gemma():
    b = backend()
    with TestClient(create_app(b.settings, b)) as client:
        config = client.get('/ui/config').json()
        assert config['default_text_model'] == 'Qwen3-VL-2B-Instruct'
        assert config['model_limits']['Qwen3-VL-2B-Instruct']['max_input_tokens'] == 2048
        assert config['vlm_max_images'] == 1
        for stream in [False, True]:
            response = client.post('/v1/chat/completions', json={
                'messages': [{'role': 'user', 'content': 'Hallo'}], 'stream': stream,
                'max_input_tokens': 2048})
            assert response.status_code == 200 and 'Hallo' in response.text
        assert all(call['frames'] == [] for call in b.vlm.calls)
        response = client.post('/v1/chat/completions', json={
            'model': LLM_MODEL, 'messages': [{'role': 'user', 'content': 'Hallo'}]})
        assert response.status_code == 503


def test_oversized_tools_are_retrieved_before_vlm_budget():
    b = backend()
    tools = [tool('get_time')] + [tool(f'unrelated_{i}') for i in range(100)]
    for item in tools:
        item['function']['parameters']['properties']['zone']['enum'] = ['UTC'] + [f'Zone_{i}' for i in range(500)]
    body = {'messages': [{'role': 'user', 'content': 'Use get_time for UTC'}],
            'tools': tools, 'tool_choice': {'type': 'function', 'function': {'name': 'get_time'}}}
    assert len(json.dumps(body)) > 500_000
    b.vlm.text = '{"tool_calls":[{"function":{"name":"get_time","arguments":{"zone":"UTC"}}}]}'
    with TestClient(create_app(b.settings, b)) as client:
        response = client.post('/v1/chat/completions', json=body)
        assert response.status_code == 200, response.text
        prompt = json.dumps(b.vlm.calls[0]['prompt'])
        assert 'unrelated_99' not in prompt
        assert response.json()['choices'][0]['finish_reason'] == 'tool_calls'


def test_qwen2_preferred_release_isolated_and_override_does_not_change_whisper(tmp_path, monkeypatch):
    settings = Settings(model_store=str(tmp_path))
    manager = ModelManager(settings, '5.4.0')
    assert '/v5.1.1/' in manager.url('Qwen2-VL-2B-Instruct')
    assert '/v5.4.0/' in manager.url('Whisper-Base')
    assert '/v5.4.0/' in manager.url('Qwen3-VL-2B-Instruct')
    existing = tmp_path / 'Qwen2-VL-2B-Instruct.hef'
    existing.write_bytes(b'user model')
    seen = []
    monkeypatch.setattr('hailo_services.models.ensure_model_file', lambda path, *args: seen.append(path) or path)
    selected = manager.resolve('Qwen2-VL-2B-Instruct', 'vlm')
    assert selected == tmp_path / 'v5.1.1' / existing.name
    assert existing.read_bytes() == b'user model'
    manager = ModelManager(replace(settings, vlm_release='v5.3.0'), '5.4.0')
    assert '/v5.3.0/' in manager.url('Qwen2-VL-2B-Instruct')
    assert '/v5.4.0/' in manager.url('Whisper-Base')


def test_model_adjacent_config_limits_and_env_precedence(tmp_path, monkeypatch):
    path = tmp_path / 'config.yaml'
    path.write_text('''models:
  vlm: {model: Qwen3-VL-2B-Instruct, max_input_tokens: 2048, release: v5.4.0}
  gemma: {enabled: false, max_input_tokens: 4096}
  hailo_llm: {enabled: false, max_input_tokens: 2048}
''')
    monkeypatch.setenv('HAILO_CONFIG', str(path))
    monkeypatch.setenv('HAILO_VLM_MAX_INPUT_TOKENS', '1024')
    settings = Settings.from_env()
    assert settings.vlm_max_input_tokens == 1024 and settings.litert_max_input_tokens == 4096
    assert settings.vlm_release == 'v5.4.0'
    monkeypatch.setenv('HAILO_VLM_MAX_INPUT_TOKENS', '2049')
    with pytest.raises(ValueError, match='compiled HEF limit'):
        Settings.from_env()


def test_catalogue_documents_all_hailo_llm_limits():
    manager = ModelManager(Settings())
    for entry in manager.entries.values():
        if entry['kind'] in {'llm', 'vlm'}:
            assert entry['context_length'] == 2048


def test_vlm_tool_stream_returns_only_validated_action_deltas():
    b = backend(text='{"tool_calls":[{"function":{"name":"get_time","arguments":{"zone":"UTC"}}}]}')
    body = {'messages': [{'role': 'user', 'content': 'Use get_time UTC'}],
            'tools': [tool()], 'tool_choice': 'required', 'stream': True}
    with TestClient(create_app(b.settings, b)) as client:
        response = client.post('/v1/chat/completions', json=body)
        assert '"finish_reason": "tool_calls"' in response.text
        assert '"tool_calls"' in response.text
        assert '"name": "get_time"' in response.text
        b.vlm.text = b.vlm.text.replace('get_time', 'unknown')
        response = client.post('/v1/chat/completions', json=body)
        assert '"error"' in response.text
        assert '"tool_calls"' not in response.text


def test_http_oversized_plain_text_does_not_reach_native_generation():
    b = backend(tokens=2048)
    with TestClient(create_app(b.settings, b)) as client:
        response = client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'content': 'large plain input'}]})
        assert response.status_code == 400
        assert response.json()['error']['code'] == 'input_token_limit_exceeded'
        assert not b.vlm.calls


def test_vlm_http_metrics_survive_prompt_copy_and_are_isolated_per_request():
    import re

    b = backend(text='Hello world')
    with TestClient(create_app(b.settings, b)) as client:
        payload = {'model': b.settings.vlm_model, 'messages': [{'role': 'user', 'content': 'Hallo'}]}
        first = client.post('/v1/chat/completions', json=payload).json()
        metrics = first['metrics']
        assert metrics['input_tokens_source'] == 'tokenizer_text'
        assert metrics['output_tokens'] == 2
        assert metrics['output_tokens_source'] == 'tokenizer'
        assert metrics['input_budget_tokens'] > metrics['input_tokens']
        assert 0 <= metrics['ttft_ms'] <= metrics['inference_ms'] <= metrics['processing_ms']
        assert metrics['ttft_source'] == 'first_text_chunk'
        assert first['usage']['completion_tokens'] == 2
        for key in ('requested_at', 'responded_at'):
            assert re.search(r'\d{2}:\d{2}:\d{2}\.\d{3}\+00:00$', metrics[key])
        b.vlm.text = ''
        second = client.post('/v1/chat/completions', json=payload).json()['metrics']
        assert second['request_id'] != metrics['request_id']
        assert second['output_tokens'] == 0
        assert 'ttft_ms' not in second
        assert metrics['output_tokens'] == 2
