"""Regression tests for production LiteRT optimizations and timings."""

import json
import sys
from types import SimpleNamespace

from hailo_services.config import LLM_MODEL
from hailo_services.litert_optimizations import instrument_engine, successful_action_followup
from hailo_services.runtime import LiteRTLMBackend
from hailo_services.schemas import ChatRequest


def action_request(*, failed=None, response_type="action_done", speech=None):
    call_id = "call_action"
    payload = {
        "speech": speech or {},
        "response_type": response_type,
        "data": {
            "success": [{"name": "Küche", "type": "area", "id": "kitchen"}],
            "failed": failed or [],
        },
    }
    request = ChatRequest(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": "schalte das Licht in der Küche aus"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "intent__HassTurnOff",
                        "arguments": json.dumps({"area": "Küche", "domain": ["light"]}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(payload),
            },
        ],
    )
    object.__setattr__(request, "_ha_assist", True)
    request._request_id = "fast-123"
    return request


def test_successful_action_followup_skips_second_gemma_inference(caplog):
    backend = LiteRTLMBackend("/does/not/need/to/exist.litertlm", debug_log=True)
    request = action_request()
    with caplog.at_level("DEBUG", logger="hailo_services.litert_optimizations"):
        result = backend.chat(request)
    assert result == "Erledigt."
    assert "tool_followup_fast_path request_id=fast-123" in caplog.text
    assert "skipped_gemma=true" in caplog.text


def test_action_followup_prefers_home_assistant_speech():
    request = action_request(speech={"plain": {"speech": "Alle Lichter sind aus."}})
    fast = successful_action_followup(request)
    assert fast is not None
    assert fast["text"] == "Alle Lichter sind aus."


def test_failed_or_query_tool_result_does_not_use_fast_path():
    failed = action_request(failed=[{"name": "Küche"}])
    assert successful_action_followup(failed) is None
    query = action_request(response_type="query")
    assert successful_action_followup(query) is None


def test_fast_path_requires_all_parallel_tool_results():
    request = action_request()
    request.messages[-2]["tool_calls"].append({
        "id": "call_missing",
        "type": "function",
        "function": {"name": "intent__HassTurnOff", "arguments": "{}"},
    })
    assert successful_action_followup(request) is None


class FakeConversation:
    def __init__(self):
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def render_message_to_string(self, message):
        return f"<|turn>user\n{message['content']}\n<|turn>model\n"

    def send_message(self, prompt, **kwargs):
        self.sent.append(prompt)
        return {"content": "ok"}


class FakeEngine:
    def __init__(self):
        self.conversation = FakeConversation()

    def create_conversation(self, **kwargs):
        return self.conversation

    def tokenize(self, text):
        return text.split()


def test_wall_clock_timing_is_logged_without_native_benchmark(caplog):
    backend = SimpleNamespace(engine=FakeEngine(), debug_log=True)
    instrument_engine(backend)
    with caplog.at_level("DEBUG", logger="hailo_services.litert_optimizations"):
        with backend.engine.create_conversation(messages=[]) as conversation:
            assert conversation.send_message("hello") == {"content": "ok"}
    assert "gemma_timing request_id=-" in caplog.text
    assert "wall_ms=" in caplog.text
    assert "prefill_tokens" not in caplog.text
    assert "decode_tokens" not in caplog.text


def test_exact_rendered_prompt_is_logged_in_debug(caplog):
    backend = SimpleNamespace(engine=FakeEngine(), debug_log=True)
    instrument_engine(backend)
    with caplog.at_level("DEBUG", logger="hailo_services.litert_optimizations"):
        with backend.engine.create_conversation(messages=[]) as conversation:
            rendered = conversation.render_message_to_string(
                {"role": "user", "content": "schalte das Licht aus"}
            )
    assert rendered == "<|turn>user\nschalte das Licht aus\n<|turn>model\n"
    assert "event=gemma_rendered_prompt request_id=-" in caplog.text
    assert '"prompt":"<|turn>user\\nschalte das Licht aus\\n<|turn>model\\n"' in caplog.text
    assert '"raw_tokens":6' in caplog.text


def test_start_does_not_enable_native_benchmark(monkeypatch, tmp_path):
    model = tmp_path / "gemma.litertlm"
    model.write_bytes(b"fake")
    captured = {}

    class EngineContext:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self.engine = FakeEngine()

        def __enter__(self):
            return self.engine

        def __exit__(self, *args):
            return None

    fake_litert = SimpleNamespace(
        Engine=EngineContext,
        Backend=SimpleNamespace(CPU=lambda: "cpu"),
    )
    monkeypatch.setitem(sys.modules, "litert_lm", fake_litert)
    backend = LiteRTLMBackend(model, max_num_tokens=4096, debug_log=True)
    backend.start()
    try:
        assert "enable_benchmark" not in captured["kwargs"]
        assert captured["kwargs"]["max_num_tokens"] == 4096
    finally:
        backend.close()


def test_wall_metrics_are_collected_without_native_counts(monkeypatch):
    from hailo_services.litert_optimizations import _REQUEST, _log_timing

    metrics = {"input_tokens": 20, "input_tokens_source": "tokenizer"}
    monkeypatch.setattr(_REQUEST, "metrics", metrics, raising=False)
    _log_timing(
        SimpleNamespace(debug_log=False),
        wall_ms=50,
        create_call_ms=1,
        enter_ms=2,
        first_chunk_ms=12,
    )
    assert metrics["input_tokens"] == 20
    assert metrics["input_tokens_source"] == "tokenizer"
    assert metrics["inference_ms"] == 50
    assert metrics["ttft_ms"] == 12
    assert metrics["ttft_source"] == "first_text_chunk"
    assert "prefill_tokens_per_second" not in metrics
    assert "decode_tokens_per_second" not in metrics


def test_native_constraints_enabled_only_for_explicit_python_capability():
    class CapableEngine(FakeEngine):
        flags = []
        def create_conversation(self, enable_constrained_decoding=False, **kwargs):
            self.flags.append(enable_constrained_decoding)
            return self.conversation

    class LegacyEngine(FakeEngine):
        def create_conversation(self, **kwargs):
            assert 'enable_constrained_decoding' not in kwargs
            return self.conversation

    for engine, expected in [(CapableEngine(), True), (LegacyEngine(), False)]:
        backend = LiteRTLMBackend('/unused.litertlm')
        backend.engine = engine
        backend.litert_lm = SimpleNamespace(Tool=object, SamplerConfig=lambda **kwargs: kwargs)
        request = ChatRequest(model=LLM_MODEL, messages=[{'role': 'user', 'content': 'Hello'}],
                              tools=[{'type': 'function', 'function': {
                                  'name': 'test', 'parameters': {'type': 'object', 'properties': {}}}}])
        assert backend.chat(request, tools_prepared=True) == 'ok'
        assert request._metrics['constrained_decoding']['enabled'] is expected
        if expected:
            assert all(engine.flags)
