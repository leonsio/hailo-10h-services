"""Regression tests for low-latency LiteRT Home Assistant handling."""

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

    def get_benchmark_info(self):
        return SimpleNamespace(
            init_time_in_second=0.02,
            time_to_first_token_in_second=0.35,
            last_prefill_token_count=900,
            last_prefill_tokens_per_second=300.0,
            last_decode_token_count=30,
            last_decode_tokens_per_second=15.0,
        )


class FakeEngine:
    def __init__(self):
        self.conversation = FakeConversation()

    def create_conversation(self, **kwargs):
        return self.conversation

    def tokenize(self, text):
        return text.split()


def test_native_litert_benchmark_metrics_are_logged(caplog):
    backend = SimpleNamespace(engine=FakeEngine(), debug_log=True)
    instrument_engine(backend)
    with caplog.at_level("DEBUG", logger="hailo_services.litert_optimizations"):
        with backend.engine.create_conversation(messages=[]) as conversation:
            assert conversation.send_message("hello") == {"content": "ok"}
    assert "gemma_timing request_id=-" in caplog.text
    assert "prefill_tokens=900" in caplog.text
    assert "decode_tokens=30" in caplog.text
    assert '"prefill_ms_estimate":3000.0' in caplog.text
    assert '"decode_ms_estimate":2000.0' in caplog.text
    assert '"time_to_first_token_ms":350.0' in caplog.text


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
    assert '"raw_tokens":4' in caplog.text


def test_debug_start_enables_native_litert_benchmark(monkeypatch, tmp_path):
    model = tmp_path / "gemma.litertlm"
    model.write_bytes(b"fake")
    captured = {}

    class EngineContext:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
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
        assert captured["kwargs"]["enable_benchmark"] is True
        assert captured["kwargs"]["max_num_tokens"] == 4096
    finally:
        backend.close()


def test_non_debug_start_disables_native_litert_benchmark(monkeypatch, tmp_path):
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
    backend = LiteRTLMBackend(model, max_num_tokens=4096, debug_log=False)
    backend.start()
    try:
        assert captured["kwargs"]["enable_benchmark"] is False
    finally:
        backend.close()
