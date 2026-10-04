"""End-to-end input budget behavior with a deterministic native-engine stand-in."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_services import FakeBackend, settings

from hailo_services.app import create_app
from hailo_services.config import LLM_MODEL
from hailo_services.input_budget import InputBudgetError, history_candidates
from hailo_services.runtime import LiteRTLMBackend
from hailo_services.schemas import ChatRequest
from hailo_services.tool_retrieval import retrieve_tools

TOOL = {"type": "function", "function": {
    "name": "intent__HassTurnOn", "description": "turn a light on",
    "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}}},
}}


class BudgetBackend(LiteRTLMBackend):
    def __init__(self, context=16384):
        super().__init__("/fake/gemma.litertlm", context)
        self.created = []
        self.closed = 0
        self.engine = SimpleNamespace(tokenize=lambda text: text.split())
        self.litert_lm = SimpleNamespace(
            Tool=type("Tool", (), {}), SamplerConfig=lambda **kwargs: kwargs,
        )

    def start(self):
        backend = self

        class Conversation:
            def __init__(self, options):
                self.options = options
                self.response = {"content": "ok"}

            def __enter__(self):
                backend.created.append(self.options)
                return self

            def __exit__(self, *args):
                backend.closed += 1

            def render_message_to_string(self, prompt):
                return json.dumps({
                    "messages": self.options["messages"], "tools": [
                        t.get_tool_description() for t in self.options.get("tools", [])
                    ], "prompt": prompt,
                }, ensure_ascii=False)

            def send_message(self, prompt, **kwargs):
                self.prompt = prompt
                return self.response

        self.engine.create_conversation = lambda **opts: Conversation(opts)


def req(messages, **kwargs):
    return ChatRequest(model=LLM_MODEL, messages=messages, max_tokens=64, **kwargs)


def test_api_trims_old_complete_rounds_and_keeps_tools_and_active_tool_turn():
    backend = BudgetBackend()
    messages = [
        {"role": "system", "content": "Keep this system prompt."},
        {"role": "user", "content": "old " * 5000},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "active request"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": TOOL["function"]["name"], "arguments": "{}",
            },
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    payload = {
        "model": LLM_MODEL, "messages": messages, "tools": [TOOL],
        "max_input_tokens": 4096, "max_tokens": 64,
    }
    with TestClient(create_app(settings(), FakeBackend(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    inference_options = backend.created[-1]
    assert [m["role"] for m in inference_options["messages"]] == ["system", "user", "assistant"]
    assert inference_options["messages"][-1]["tool_calls"][0]["id"] == "call_1"
    assert inference_options["tools"][0].get_tool_description() == TOOL
    assert backend.closed == len(backend.created)
    assert len(backend.created) >= 2


def test_required_prompt_and_tool_schema_are_not_truncated_when_budget_is_impossible():
    backend = BudgetBackend()
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": "mandatory " * 4500},
                     {"role": "user", "content": "turn"}],
        "tools": [TOOL], "max_input_tokens": 4096, "max_tokens": 64,
    }
    with TestClient(create_app(settings(), FakeBackend(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "input_token_limit_exceeded"
    assert response.json()["error"]["input_tokens"] > 4096
    assert response.json()["error"]["input_limit"] == 4096
    assert backend.closed == len(backend.created)
    # No inference call was made after all safe trimming candidates failed.
    assert backend.created[-1]["messages"] == payload["messages"][:-1]


def test_context_reserves_output_space_when_cap_exceeds_available_input():
    backend = BudgetBackend(context=4096)
    backend.start()
    request = req([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "earlier " * 4200},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "current"},
    ], max_input_tokens=4096)
    # 4096 context - 64 requested output - 1 start token is the actual input ceiling.
    trimmed = backend._limit_input(request, {})
    assert trimmed.messages == [request.messages[0], *request.messages[-1:]]


@pytest.mark.parametrize(("prompt_tokens", "accepted"), [(3839, True), (3840, False)])
def test_exact_budget_boundary_and_requested_value_are_applied(prompt_tokens, accepted):
    backend = BudgetBackend()
    backend.start()
    backend.engine.tokenize = lambda text: range(prompt_tokens)
    request = req([{"role": "user", "content": "hello"}], max_input_tokens=4096)
    if accepted:
        trimmed = backend._limit_input(request, {})
        assert trimmed.messages == request.messages
    else:
        with pytest.raises(InputBudgetError) as error:
            backend._limit_input(request, {})
        assert error.value.tokens == 4096
    probe = backend.created[-1]
    assert probe["max_output_tokens"] == request.max_tokens


def test_malformed_old_tool_history_is_rejected_before_it_can_be_trimmed():
    backend = BudgetBackend()
    request = req([
        {"role": "user", "content": "old " * 5000},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "pending", "type": "function", "function": {
                "name": TOOL["function"]["name"], "arguments": "{}",
            },
        }]},
        {"role": "user", "content": "current"},
    ], tools=[TOOL], max_input_tokens=4096)
    with pytest.raises(ValueError, match="Missing results"):
        backend._limit_input(request, {"tools": []})
    assert not backend.created


def test_candidates_keep_system_and_every_message_from_the_active_turn():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "system", "content": "also system"},
        {"role": "user", "content": "active"},
        {"role": "assistant", "tool_calls": []},
        {"role": "tool", "tool_call_id": "id", "content": "result"},
    ]
    candidates = list(history_candidates(messages))
    assert len(candidates) == 2
    assert candidates[-1] == [messages[0], messages[3], *messages[4:]]


@pytest.mark.parametrize("value", [0, -1, 131073, True, 4096.0, "4096.0"])
def test_invalid_max_input_tokens_values(value):
    with pytest.raises(ValidationError):
        req([{"role": "user", "content": "hello"}], max_input_tokens=value)


def test_max_input_tokens_is_optional_for_existing_clients():
    assert req([{"role": "user", "content": "hello"}]).max_input_tokens is None
    assert req([{"role": "user", "content": "hello"}], max_input_tokens="4096").max_input_tokens == 4096


def test_engine_tokenizer_and_renderer_are_required_only_when_budget_is_requested():
    backend = BudgetBackend()
    backend.start()
    backend.engine.tokenize = None
    request = req([{"role": "user", "content": "hello"}], max_input_tokens=4096)
    with pytest.raises(ValueError, match="Engine.tokenize"):
        backend._limit_input(request, {})
    assert not backend.created
    assert backend.chat(req([{"role": "user", "content": "hello"}])) == "ok"


def test_german_user_request_reduces_irrelevant_ha_tools_and_entity_enums():
    tools = [
        {"type": "function", "function": {
            "name": "intent__HassTurnOn", "description": "Turn on a living room light",
            "parameters": {"type": "object", "properties": {"entity_id": {
                "type": "string", "enum": [
                    "light.wohnzimmer_stehlampe", "light.wohnzimmer_decke",
                    "light.schlafzimmer_stehlampe", "light.kueche_decke",
                ]
            }}}
        }},
        {"type": "function", "function": {
            "name": "intent__HassClimateSetTemperature", "description": "Set climate temperature",
            "parameters": {"type": "object", "properties": {"temperature": {"type": "number"}}}
        }},
        {"type": "function", "function": {
            "name": "intent__HassCoverClose", "description": "Close a window cover",
            "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}}}
        }},
    ]
    selected, stats = retrieve_tools(
        [{"role": "user", "content": "Mach bitte die Stehlampe im Wohnzimmer an."}], tools
    )
    assert [item["function"]["name"] for item in selected] == ["intent__HassTurnOn"]
    assert selected[0]["function"]["parameters"]["properties"]["entity_id"]["enum"] == [
        "light.wohnzimmer_stehlampe",
    ]
    assert stats == {"tools_before": 3, "tools_after": 1, "enum_values_removed": 3}
    # Input schemas are copied; requests can be safely retried with the original.
    assert len(tools[0]["function"]["parameters"]["properties"]["entity_id"]["enum"]) == 4


def test_tool_retrieval_preserves_tools_without_a_confident_match():
    tools = [TOOL, {"type": "function", "function": {"name": "other", "parameters": {}}}]
    selected, stats = retrieve_tools([{"role": "user", "content": "Guten Morgen"}], tools)
    assert selected == tools
    assert stats["tools_after"] == 2


def test_default_gemma_input_budget_is_4096_even_if_client_omits_custom_field():
    backend = BudgetBackend()
    backend.start()
    request = req([{"role": "user", "content": "active request"}])
    backend.engine.tokenize = lambda text: range(4096)
    assert request.max_input_tokens is None
    with pytest.raises(InputBudgetError) as error:
        backend.chat(request)
    assert error.value.limit == 4096
    assert error.value.tokens == 4352
    assert backend.closed == len(backend.created)
    assert backend.max_input_tokens == 4096
