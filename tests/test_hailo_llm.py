import asyncio
import json
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hailo_services.app import create_app
from hailo_services.config import LLM_MODEL, Settings
from hailo_services.hailo_llm_chat import limit_request, model_prompt
from hailo_services.input_budget import InputBudgetError
from hailo_services.models import ModelManager
from hailo_services.runtime import HailoBackend, Runtime
from hailo_services.schemas import ChatRequest
from hailo_services.vlm_chat import render_prompt


class NativeLLM:
    def __init__(self, text='Paris est la capitale.', tokens=None, capacity=2048):
        self.text, self.tokens, self.capacity = text, tokens, capacity
        self.calls, self.rendered, self.threads = [], [], []
        self.clears = 0
        self.closed = False

    def tokenize(self, text):
        self.rendered.append(text)
        return range(self.tokens) if self.tokens is not None else text.split()

    def prompt_template(self):
        return ('{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}'
                '<|im_end|>\n{% endfor %}<|im_start|>assistant\n')

    def max_context_capacity(self):
        return self.capacity

    def clear_context(self):
        self.clears += 1

    @contextmanager
    def generate(self, **kwargs):
        assert 'frames' not in kwargs
        assert isinstance(kwargs['prompt'], str)
        self.calls.append(kwargs)
        self.threads.append(threading.get_ident())
        yield iter([self.text[:5], self.text[5:]])

    def release(self):
        self.closed = True


class NativeLlama(NativeLLM):
    def prompt_template(self):
        # Reduced Llama membership/length check that rejected even plain text
        # after the adapter added tool_calls=[]. Keep the native error verbatim.
        return """
{%- for message in messages %}
{%- if 'tool_calls' in message %}
{%- if message.tool_calls|length != 1 %}
{{- raise_exception('This model only supports single tool-calls at once!') }}
{%- endif %}
{{- message.tool_calls[0].function.name }}
{%- else %}
<|start_header_id|>{{ message.role }}<|end_header_id|>
{{ message.content }}<|eot_id|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>{% endif %}
"""


def backend(**kwargs):
    s = Settings(**{'vlm_enabled': False, 'hailo_llm_enabled': True, 'whisper_enabled': False,
                    'minilm_enabled': False, 'wyoming_port': 0, **kwargs})
    b = HailoBackend(s)
    b.llm = NativeLlama() if s.hailo_llm_model_id == 'Llama3.2-1B-Instruct' else NativeLLM()
    b.start = lambda: None
    return b


def request(model='Qwen2.5-1.5B-Instruct', **kwargs):
    return ChatRequest(model=model, messages=[{'role': 'user', 'content': 'Hauptstadt Frankreich?'}],
                       **kwargs)


def tool():
    return {'type': 'function', 'function': {'name': 'get_time', 'parameters': {
        'type': 'object', 'properties': {'zone': {'type': 'string', 'enum': ['UTC']}},
        'required': ['zone'], 'additionalProperties': False}}}


@pytest.mark.parametrize('model', ['Qwen2.5-1.5B-Instruct', 'Qwen3-1.7B-Instruct',
                                 'Llama3.2-1B-Instruct'])
def test_native_llm_http_ws_stream_metrics_and_model_selection(model):
    b = backend(hailo_llm_model=model)
    native = b.llm
    with TestClient(create_app(b.settings, b)) as client:
        assert [m['id'] for m in client.get('/v1/models').json()['data']] == [model]
        health = client.get('/health').json()
        assert health['models'] == [model]
        assert health['default_text_model'] == model
        assert health['model_limits'][model]['context_length'] == 2048
        config = client.get('/ui/config').json()
        assert config['vision_models'] == []
        assert config['hailo_llm_model'] == model
        assert config['chat_models'] == [model]
        payload = {'messages': [{'role': 'user', 'content': 'Hauptstadt Frankreich?'}]}
        answer = client.post('/v1/chat/completions', json=payload)
        assert answer.status_code == 200
        data = answer.json()
        assert data['model'] == model
        assert data['choices'][0]['message']['content'] == b.llm.text
        assert data['usage']['prompt_tokens'] > 0
        assert data['metrics']['ttft_ms'] >= 0
        assert data['metrics']['input_tokens_source'] == 'tokenizer_text'
        streamed = client.post('/v1/chat/completions', json={**payload, 'model': model,
                                                          'stream': True})
        assert streamed.status_code == 200
        events = [json.loads(line[6:]) for line in streamed.text.splitlines()
                  if line.startswith('data: {')]
        assert ''.join(e['choices'][0]['delta'].get('content', '') for e in events
                       if e.get('choices')) == b.llm.text
        assert streamed.text.endswith('data: [DONE]\n\n')
        with client.websocket_connect('/ws') as ws:
            ws.send_json({'id': 'llm-test', 'op': 'chat', 'payload': {**payload, 'model': model}})
            assert ws.receive_json()['result']['text'] == b.llm.text
    assert len(set(native.threads)) == 1
    assert native.clears == 6  # Three calls, isolated KV contexts.


def test_llm_uses_string_text_parts_and_own_prompt_template():
    b = backend()
    r = request(top_p=0.8, seed=19)
    r.messages[0]['content'] = [{'type': 'text', 'text': 'Hallo'}, {'type': 'text', 'text': 'Welt'}]
    assert b.chat(r) == b.llm.text
    call = b.llm.calls[0]
    assert call['prompt'] == b.llm.rendered[0]
    assert 'Hallo\nWelt' in call['prompt']
    assert call['top_p'] == 0.8 and call['seed'] == 19
    assert b.llm.rendered[0] == '<|im_start|>user\nHallo\nWelt<|im_end|>\n<|im_start|>assistant'


@pytest.mark.parametrize('tokens,accepted', [(1663, True), (1664, False)])
def test_llm_2048_boundary_reserves_output_and_template(tokens, accepted):
    b = backend()
    b.llm.tokens = tokens
    r = request(max_input_tokens=8192, max_tokens=256)
    if accepted:
        assert b.chat(r) == b.llm.text
        assert r._metrics['input_budget_tokens'] == 1791
    else:
        with pytest.raises(InputBudgetError) as exc:
            b.chat(r)
        assert exc.value.limit == 1791
        assert b.llm.calls == []


def test_llm_native_capacity_and_configured_ceiling_are_authoritative():
    r = request(max_input_tokens=2048)
    model = NativeLLM(tokens=500, capacity=800)
    with pytest.raises(InputBudgetError) as exc:
        limit_request(model, r, 2048, 2048)
    assert exc.value.limit == 543
    model.capacity = 2048
    with pytest.raises(InputBudgetError) as exc:
        limit_request(model, r, 600, 2048)
    assert exc.value.limit == 600


def test_llm_trims_old_turns_keeps_system_and_current_turn():
    class LengthLLM(NativeLLM):
        def tokenize(self, text):
            return range(len(text))
    r = request()
    r.messages = [{'role': 'system', 'content': 'Retain system'},
                  {'role': 'user', 'content': 'old ' * 500},
                  {'role': 'assistant', 'content': 'Old answer'},
                  *r.messages]
    trimmed, prompt = limit_request(LengthLLM(), r, 800, 2048)
    assert [m['role'] for m in trimmed.messages] == ['system', 'user']
    assert 'Retain system' in prompt
    assert r._metrics['removed_messages'] == 2
    with pytest.raises(InputBudgetError):
        limit_request(LengthLLM(tokens=2000), request(), 64, 2048)


@pytest.mark.parametrize('stream', [False, True])
def test_llm_rejects_zero_temperature_and_images_before_generation(stream):
    b = backend()
    native = b.llm
    with TestClient(create_app(b.settings, b)) as client:
        payload = request(temperature=0, stream=stream).model_dump()
        response = client.post('/v1/chat/completions', json=payload)
        assert response.status_code == 400
        assert 'Hailo LLM' in response.json()['error']
        payload = {'messages': [{'role': 'user', 'content': [{'type': 'image_url',
                   'image_url': {'url': 'invalid-image'}}]}]}
        response = client.post('/v1/chat/completions', json=payload)
        assert response.status_code == 400
        assert 'enabled VLM' in response.json()['error']
        payload['model'] = b.settings.hailo_llm_model_id
        response = client.post('/v1/chat/completions', json=payload)
        assert response.status_code == 400
        assert 'text-only' in response.json()['error']
    assert native.calls == []


@pytest.mark.parametrize('wrapped', [False, True])
def test_llm_tool_calls_buffer_validate_and_keep_result_history(wrapped):
    b = backend()
    text = json.dumps({'name': 'get_time', 'arguments': {'zone': 'UTC'}})
    b.llm.text = f'<tool_call>{text}</tool_call>' if wrapped else text
    emitted = []
    r = request(tools=[tool()], tool_choice='required')
    result = b.chat(r, emitted.append)
    assert len(emitted) == 1 and emitted[0] == result
    assert result['tool_calls'][0]['function']['name'] == 'get_time'
    followup = r.model_copy(update={'messages': [*r.messages, result, {
        'role': 'tool', 'tool_call_id': result['tool_calls'][0]['id'], 'content': '12:00'}]})
    assert '12:00' in model_prompt(followup)[-1]['content']
    b.llm.text = '{"name":"get_time","arguments":{"zone":"invented"}}'
    with pytest.raises(ValueError):
        b.chat(r, emitted.append)
    assert len(emitted) == 1
    assert b.llm.clears == 4


def test_llm_missing_tokenizer_and_generation_failure_clean_context():
    b = backend()
    b.llm.tokenize = None
    with pytest.raises(ValueError, match='LLM.tokenize'):
        b.chat(request())
    b.llm.tokenize = lambda text: text.split()
    @contextmanager
    def failure(**kwargs):
        raise RuntimeError('native failure')
        yield
    b.llm.generate = failure
    with pytest.raises(RuntimeError, match='native failure'):
        b.chat(request())
    assert b.llm.clears == 2


@pytest.mark.parametrize('vlm_enabled', [False, True])
@pytest.mark.parametrize('fail_load', [False, True])
def test_llm_startup_shared_device_resolution_and_cleanup(monkeypatch, tmp_path, vlm_enabled, fail_load):
    events, resolutions, loaded = [], [], []
    class Resource:
        def __init__(self, name):
            self.name = name
        def release(self):
            events.append('release:' + self.name)
    class Device(Resource):
        @staticmethod
        def create_params():
            return SimpleNamespace(group_id='UNSHARED')
        def __init__(self, params):
            assert params.group_id == 'SHARED'
            events.append('allocate')
            super().__init__('device')
    def llm(device, path):
        loaded.append((device, path))
        if fail_load:
            raise RuntimeError('native load failure')
        return Resource('llm')
    monkeypatch.setitem(sys.modules, 'hailo_platform', SimpleNamespace(VDevice=Device))
    monkeypatch.setitem(sys.modules, 'hailo_platform.genai', SimpleNamespace(
        LLM=llm, VLM=lambda device, path: Resource('vlm')))
    monkeypatch.setattr('hailo_services.runtime.prepare_model_version', lambda: '5.4.0')
    def resolve(self, model, kind):
        assert 'allocate' not in events
        resolutions.append((model, kind))
        return tmp_path / f'{model}.hef'
    monkeypatch.setattr(ModelManager, 'resolve', resolve)
    s = Settings(vlm_enabled=vlm_enabled, hailo_llm_enabled=True, whisper_enabled=False,
                 minilm_enabled=False)
    b = HailoBackend(s)
    if fail_load:
        with pytest.raises(RuntimeError, match='native load failure'):
            b.start()
    else:
        b.start()
        assert loaded == [(b.device, b.paths['llm'])]
        b.close()
        b.close()
    assert resolutions[-1] == ('Qwen2.5-1.5B-Instruct', 'llm')
    expected = ([] if fail_load else ['release:llm']) + (
        ['release:vlm'] if vlm_enabled else []) + ['release:device']
    assert events == ['allocate', *expected]
    assert b.llm is b.vlm is b.device is None


@pytest.mark.parametrize('hailo_llm', [False, True])
def test_gemma_and_hailo_models_have_independent_selection_and_queues(hailo_llm):
    class Backend:
        paths = {}
        def start(self):
            pass
        def close(self):
            pass
        def chat(self, r, emit=None, cancelled=None):
            if emit:
                emit(r.model)
            return r.model
    s = Settings(vlm_enabled=not hailo_llm, hailo_llm_enabled=hailo_llm,
                 whisper_enabled=False, minilm_enabled=False, wyoming_port=0)
    async def check():
        rt = Runtime(s, Backend(), Backend())
        await rt.start()
        try:
            model = s.hailo_llm_model_id if hailo_llm else s.vlm_model
            assert await rt.chat(request(model=model)) == model
            assert await rt.chat(request(model=LLM_MODEL, temperature=0)) == LLM_MODEL
            assert await rt.chat(ChatRequest(messages=request().messages)) == LLM_MODEL
            assert rt.executor is not rt.litert_executor
            rt.pending = s.queue_size
            assert await rt.chat(request(model=LLM_MODEL)) == LLM_MODEL
            assert rt.status()['models'] == [model, LLM_MODEL]
        finally:
            rt.pending = 0
            await rt.close()
    asyncio.run(check())


def test_explicit_model_selects_matching_native_instance_when_both_are_enabled():
    b = backend(vlm_enabled=True)
    class NativeVLM(NativeLLM):
        @contextmanager
        def generate(self, **kwargs):
            assert kwargs['frames'] == []
            assert isinstance(kwargs['prompt'], list)
            self.calls.append(kwargs)
            yield iter(['Vision model text'])
    b.vlm = NativeVLM()
    assert b.chat(request(model=b.settings.vlm_model)) == 'Vision model text'
    assert b.llm.calls == []
    assert b.chat(request()) == b.llm.text
    assert len(b.llm.calls) == len(b.vlm.calls) == 1
    with pytest.raises(ValueError, match='Unknown or disabled'):
        b.chat(request(model='not-loaded'))
    assert len(b.llm.calls) == len(b.vlm.calls) == 1


@pytest.mark.parametrize('value', ['Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct.hef',
                                 '/models/Qwen2.5-1.5B-Instruct.hef'])
def test_llm_model_id_keeps_decimal_parameter_counts(value):
    assert Settings(hailo_llm_model=value).hailo_llm_model_id == 'Qwen2.5-1.5B-Instruct'


def test_llm_sends_non_thinking_template_and_native_empty_special_token_defaults():
    b = backend(hailo_llm_model='Qwen3-1.7B-Instruct')
    b.llm.prompt_template = lambda: (
        '{{ bos_token }}{% for m in messages %}{{ m.content }}{{ eos_token }}{% endfor %}'
        '{% if enable_thinking %}<think>{% else %}<think>\n\n</think>\n{% endif %}')
    b.chat(request(model='Qwen3-1.7B-Instruct'))
    assert b.llm.calls[0]['prompt'] == 'Hauptstadt Frankreich?<think>\n\n</think>\n'


def test_llama_template_omits_empty_calls_but_preserves_real_calls():
    options = ModelManager(Settings()).entry('Llama3.2-1B-Instruct')['prompt_template']
    prompt = [{'role': 'user', 'content': 'Hallo', 'tool_calls': []}]
    # Prove this fixture reproduces the pre-fix production failure.
    with pytest.raises(ValueError, match='single tool-calls'):
        render_prompt(NativeLlama(), prompt, model_kind='LLM')
    assert 'Hallo' in render_prompt(
        NativeLlama(), prompt, model_kind='LLM', template_options=options,
    )
    assert prompt[0]['tool_calls'] == []  # Rendering only normalizes a copy.
    call = {'function': {'name': 'get_time', 'arguments': {'zone': 'UTC'}}}
    prompt = [{'role': 'assistant', 'content': None, 'tool_calls': [call]}]
    assert 'get_time' in render_prompt(
        NativeLlama(), prompt, model_kind='LLM', template_options=options,
    )
    prompt[0]['tool_calls'].append(call)
    with pytest.raises(ValueError, match='single tool-calls'):
        render_prompt(NativeLlama(), prompt, model_kind='LLM', template_options=options)


@pytest.mark.parametrize('history_calls', [0, 1, 2])
def test_llama_tools_and_complete_history_use_single_call_contract(history_calls):
    b = backend(hailo_llm_model='Llama3.2-1B-Instruct')
    b.llm.text = '{"name":"get_time","arguments":{"zone":"UTC"}}'
    messages = [{'role': 'user', 'content': 'Use get_time UTC'}]
    if history_calls:
        calls = [{'id': f'call_{i}', 'type': 'function', 'function': {
            'name': 'get_time', 'arguments': '{"zone":"UTC"}'}}
            for i in range(history_calls)]
        messages.append({'role': 'assistant', 'content': None, 'tool_calls': calls})
        messages.extend({'role': 'tool', 'tool_call_id': call['id'], 'content': f'12:0{i}'}
                        for i, call in enumerate(calls))
    r = ChatRequest(model=b.settings.hailo_llm_model_id, messages=messages,
                    tools=[tool()], tool_choice='required', parallel_tool_calls=True)
    emitted = []
    answer = b.chat(r, emitted.append)
    assert len(answer['tool_calls']) == 1
    assert emitted == [answer]
    prompt = b.llm.calls[0]['prompt']
    assert 'Return at most one function call.' in prompt
    assert prompt == b.llm.rendered[0]  # Budgeting and generation use the same options.
    for i in range(history_calls):
        assert f'call_{i}' in prompt and f'12:0{i}' in prompt
    assert r.parallel_tool_calls is True
    assert r.messages == messages


def test_llama_multiple_generated_calls_are_rejected_before_streaming():
    b = backend(hailo_llm_model='Llama3.2-1B-Instruct')
    call = {'function': {'name': 'get_time', 'arguments': {'zone': 'UTC'}}}
    b.llm.text = json.dumps({'tool_calls': [call, call]})
    emitted = []
    with pytest.raises(ValueError, match='parallel calls when disabled'):
        b.chat(request(model=b.settings.hailo_llm_model_id, tools=[tool()],
                       parallel_tool_calls=True), emitted.append)
    assert emitted == []
    assert b.llm.clears == 2


def test_template_and_single_call_policy_come_from_custom_catalogue(tmp_path):
    import yaml

    entries = ModelManager(Settings()).entries
    # An unrelated ID with the same metadata must get the same behaviour.
    alias = 'Custom-Text-Model'
    entries[alias] = dict(entries['Llama3.2-1B-Instruct'])
    path = tmp_path / 'model_catalog.yaml'
    path.write_text(yaml.safe_dump({'schema_version': 1, 'models': entries}))
    b = backend(hailo_llm_model=alias, model_catalog=str(path))
    b.llm = NativeLlama(text='{"name":"get_time","arguments":{"zone":"UTC"}}')
    assert b.chat(request(model=alias, tools=[tool()]))['tool_calls']
    assert 'Return at most one function call.' in b.llm.calls[0]['prompt']
    # Changing only metadata restores the caller's parallel-call preference.
    entries[alias]['tool_calling'] = {'parallel_calls': True}
    path.write_text(yaml.safe_dump({'schema_version': 1, 'models': entries}))
    call = {'function': {'name': 'get_time', 'arguments': {'zone': 'UTC'}}}
    b.llm.text = json.dumps({'tool_calls': [call, call]})
    assert len(b.chat(request(model=alias, tools=[tool()]))['tool_calls']) == 2
    assert 'Return at most one function call.' not in b.llm.calls[-1]['prompt']
