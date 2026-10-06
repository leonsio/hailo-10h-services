import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_services import FakeBackend, settings

from hailo_services.app import create_app
from hailo_services.config import LLM_MODEL, VLM_MODEL
from hailo_services.runtime import LiteRTLMBackend
from hailo_services.schemas import ChatRequest
from hailo_services.tool_calling import native_messages, response_message

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "intent__HassTurnOn",
            "description": "Turns on a device",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    }
]


def payload(**kwargs):
    return {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": "Schalte die Lampe ein"}],
        "tools": TOOLS,
        "user": "ha-user",
        **kwargs,
    }


def model_response(name="intent__HassTurnOn", arguments=None):
    return {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": {"name": "Lampe"} if arguments is None else arguments,
                },
            }
        ]
    }


class NativeBackend(LiteRTLMBackend):
    """Exercise the real adapter with a stand-in for the resident LiteRT engine."""

    def __init__(self, response=None):
        super().__init__("/fake/gemma.litertlm")
        self.response = model_response() if response is None else response
        self.calls = []

    def start(self):
        backend = self

        class Tool:
            pass

        class Conversation:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def render_message_to_string(self, message):
                return json.dumps(message)

            def send_message(self, prompt, **kwargs):
                backend.prompt = prompt
                return backend.response

        class Engine:
            def tokenize(self, text):
                return text.split()

            def create_conversation(self, **kwargs):
                backend.calls.append(kwargs)
                return Conversation()

        self.engine = Engine()
        self.litert_lm = SimpleNamespace(Tool=Tool, SamplerConfig=lambda **kwargs: kwargs)

    def close(self):
        self.engine = None


@pytest.mark.parametrize("stream", [False, True])
def test_home_assistant_tool_roundtrip(stream):
    backend = NativeBackend()
    with TestClient(create_app(settings(), FakeBackend(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload(stream=stream))
        assert response.status_code == 200
        if stream:
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            choice = events[-1]["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            call = events[-2]["choices"][0]["delta"]["tool_calls"][0]
            assert call.pop("index") == 0
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            message = choice["message"]
            call = message["tool_calls"][0]
        assert call["id"].startswith("call_")
        assert json.loads(call["function"]["arguments"]) == {"name": "Lampe"}
        options = backend.calls[-1]
        assert options["automatic_tool_calling"] is False
        assert options["tools"][0].get_tool_description() == TOOLS[0]
        with pytest.raises(RuntimeError, match="executed by Home Assistant"):
            options["tools"][0].execute({"name": "Lampe"})

        backend.response = {"content": [{"type": "text", "text": "Die Lampe ist eingeschaltet."}]}
        followup = payload(
            messages=[
                {"role": "user", "content": "Schalte die Lampe ein"},
                message,
                {"role": "tool", "tool_call_id": call["id"], "content": '{"success": true}'},
            ]
        )
        result = client.post("/v1/chat/completions", json=followup)
        assert result.status_code == 200
        assert result.json()["choices"][0]["finish_reason"] == "stop"
        assert result.json()["choices"][0]["message"]["content"] == "Die Lampe ist eingeschaltet."
        assert backend.prompt == {
            "role": "tool",
            "content": [
                {
                    "type": "tool_response",
                    "name": "intent__HassTurnOn",
                    "response": '{"success": true}',
                }
            ],
        }
        history_call = backend.calls[-1]["messages"][-1]["tool_calls"][0]
        assert history_call["function"]["arguments"] == {"name": "Lampe"}


@pytest.mark.parametrize(
    "name,args",
    [
        ("unknown", {"name": "Lampe"}),
        ("intent__HassTurnOn", {}),
        ("intent__HassTurnOn", {"name": 42}),
        ("intent__HassTurnOn", {"name": "Lampe", "other": 1}),
        ("intent__HassTurnOn", "broken JSON"),
        ("intent__HassTurnOn", "[]"),
    ],
)
def test_invalid_generated_actions_are_rejected(name, args):
    backend = NativeBackend(model_response(name, args))
    with TestClient(create_app(settings(), FakeBackend(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload())
        assert response.status_code == 400
        assert "tool_calls" not in response.json()


def test_multi_name_model_argument_expands_to_parallel_calls():
    request = ChatRequest(**payload())
    object.__setattr__(request, "_ha_assist", True)
    result = response_message(
        model_response(arguments={"name": ["Backofen Licht", "Oberlicht"]}),
        request,
        "",
    )
    calls = result["tool_calls"]
    assert len(calls) == 2
    assert len({call["id"] for call in calls}) == 2
    assert [json.loads(call["function"]["arguments"]) for call in calls] == [
        {"name": "Backofen Licht"},
        {"name": "Oberlicht"},
    ]


def test_single_name_list_is_normalized_even_without_parallel_calls():
    request = ChatRequest(**payload(parallel_tool_calls=False))
    object.__setattr__(request, "_ha_assist", True)
    result = response_message(
        model_response(arguments={"name": ["Oberlicht"]}),
        request,
        "",
    )
    assert len(result["tool_calls"]) == 1
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"name": "Oberlicht"}


def test_multi_name_model_argument_requires_parallel_calls():
    request = ChatRequest(**payload(parallel_tool_calls=False))
    object.__setattr__(request, "_ha_assist", True)
    with pytest.raises(ValueError, match="multiple device names"):
        response_message(
            model_response(arguments={"name": ["Backofen Licht", "Oberlicht"]}),
            request,
            "",
        )


AREA_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "intent__HassTurnOff",
            "description": "Turns off a device",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "area": {"type": "string"},
                    "domain": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
    }
]


def area_request(user_text="schalte das Licht in der Küche aus", entities=None):
    if entities is None:
        entities = [
            ("Licht - Rechts", "light", "Küche"),
            ("Licht - Links", "light", "Küche"),
            ("Oberlicht", "light", "Küche"),
        ]
    static = "\n".join(
        f"- names: {name}\n  domain: {domain}\n  areas: {area}" for name, domain, area in entities
    )
    return ChatRequest(
        **{
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Static Context: Relevant entities for the current user request:\n"
                    + static,
                },
                {"role": "user", "content": user_text},
            ],
            "tools": AREA_TOOL,
            "parallel_tool_calls": True,
        }
    )


def test_generic_area_command_rewrites_single_name_to_area_domain():
    request = area_request()
    object.__setattr__(request, "_ha_assist", True)
    response = model_response(
        name="intent__HassTurnOff",
        arguments={"name": "Licht - Rechts", "domain": ["light"]},
    )
    result = response_message(response, request, "")
    assert len(result["tool_calls"]) == 1
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {
        "area": "Küche",
        "domain": ["light"],
    }


def test_explicit_device_name_is_not_rewritten_to_area():
    request = area_request("schalte Licht - Rechts in der Küche aus")
    response = model_response(
        name="intent__HassTurnOff",
        arguments={"name": "Licht - Rechts", "domain": ["light"]},
    )
    result = response_message(response, request, "")
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Licht - Rechts",
        "domain": ["light"],
    }


def test_single_relevant_entity_is_not_rewritten_to_area():
    request = area_request(entities=[("Oberlicht", "light", "Küche")])
    response = model_response(
        name="intent__HassTurnOff",
        arguments={"name": "Oberlicht", "domain": ["light"]},
    )
    result = response_message(response, request, "")
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Oberlicht",
        "domain": ["light"],
    }


def test_none_and_required_tool_choices():
    request = ChatRequest(**payload(tool_choice="none"))
    with pytest.raises(ValueError, match="unavailable"):
        response_message(model_response(), request, "")
    assert response_message({"content": "Hi"}, request, "Hi") == "Hi"
    request = ChatRequest(**payload(tool_choice="required"))
    with pytest.raises(ValueError, match="required tool call"):
        response_message({}, request, "Hi")
    backend = NativeBackend({"content": "Hallo"})
    with TestClient(create_app(settings(), FakeBackend(), backend)) as client:
        assert (
            client.post("/v1/chat/completions", json=payload(tool_choice="none")).status_code == 200
        )
        assert backend.calls[-1]["tools"] == []


def test_parallel_calls_and_message_object():
    request = ChatRequest(**payload(parallel_tool_calls=False))
    response = model_response()
    response["tool_calls"] *= 2
    with pytest.raises(ValueError, match="parallel"):
        response_message(response, request, "")
    request = ChatRequest(**payload())
    wrapped = SimpleNamespace(to_json=lambda: response)
    result = response_message(wrapped, request, "")
    assert len({call["id"] for call in result["tool_calls"]}) == 2
    messages = [request.messages[0], result] + [
        {"role": "tool", "tool_call_id": call["id"], "content": "OK"}
        for call in result["tool_calls"]
    ]
    native = native_messages(messages)
    assert len(native[-1]["content"]) == 2
    with pytest.raises(ValueError, match="Missing results"):
        native_messages(messages[:-1])
    messages[-1]["tool_call_id"] = "unknown"
    with pytest.raises(ValueError, match="pending"):
        native_messages(messages)


def test_named_choice_is_restricted():
    backend = NativeBackend()
    choice = {"type": "function", "function": {"name": "intent__HassTurnOn"}}
    with TestClient(create_app(settings(), FakeBackend(), backend)) as client:
        assert (
            client.post("/v1/chat/completions", json=payload(tool_choice=choice)).status_code == 200
        )
        choice["function"]["name"] = "unknown"
        assert (
            client.post("/v1/chat/completions", json=payload(tool_choice=choice)).status_code == 400
        )


@pytest.mark.parametrize(
    "change",
    [
        {"tools": [{"type": "wrong"}]},
        {"tools": TOOLS * 2},
        {"tool_choice": "invalid"},
        {"messages": [{"role": "assistant", "content": None}]},
        {"messages": [{"role": "tool", "content": "OK"}]},
        {"messages": [{"role": "assistant", "tool_calls": [1]}]},
    ],
)
def test_invalid_request_schema(change):
    with pytest.raises(ValidationError):
        ChatRequest(**payload(**change))


def test_remote_schema_refs_are_rejected():
    tool = {
        "type": "function",
        "function": {
            "name": "external",
            "parameters": {"$ref": "https://example.invalid/schema.json"},
        },
    }
    with pytest.raises(ValidationError, match="local JSON schema"):
        ChatRequest(**payload(tools=[tool]))


def test_vlm_routes_tools_and_accepts_user_field():
    backend = FakeBackend()
    with TestClient(create_app(settings(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload(model=VLM_MODEL))
        assert response.status_code == 200
        assert backend.calls
        response = client.post("/v1/chat/completions", json=payload(model=VLM_MODEL, tools=None))
        assert response.status_code == 200


def test_invalid_tool_stream_has_error_and_no_action_delta():
    backend = NativeBackend(model_response("unknown"))
    with TestClient(create_app(settings(), FakeBackend(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload(stream=True))
        assert '"error"' in response.text
        assert '"tool_calls"' not in response.text


def test_old_litert_without_tools_reports_upgrade():
    backend = NativeBackend()
    backend.start()
    del backend.litert_lm.Tool
    with pytest.raises(ValueError, match="upgrade"):
        backend.chat(ChatRequest(**payload()))


def test_native_failure_returns_json_error_and_correct_debug_status(caplog):
    backend = NativeBackend()
    original_start = backend.start

    def start():
        original_start()

        class Conversation:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def render_message_to_string(self, message):
                return json.dumps(message)

            def send_message(self, *args, **kwargs):
                raise RuntimeError("litert_lm_conversation_send_message failed")

        backend.engine.create_conversation = lambda **kwargs: Conversation()

    backend.start = start
    import logging

    with caplog.at_level(logging.DEBUG, logger="hailo_services.app"):
        with TestClient(create_app(settings(debug_log=True), FakeBackend(), backend)) as client:
            health = client.get("/health").json()
            assert health["litert_lm"]["max_num_tokens"] == 16384
            response = client.post("/v1/chat/completions", json=payload())
            assert response.status_code == 502
            assert response.json()["error"]["type"] == "inference_error"
            assert "16384" in response.json()["error"]["message"]
            assert "HAILO_LITERT_MAX_NUM_TOKENS" in response.json()["error"]["message"]
    assert "status=502" in caplog.text


def test_context_configuration(monkeypatch):
    from hailo_services.config import Settings
    from hailo_services.runtime import Runtime

    monkeypatch.setenv("HAILO_LITERT_MAX_NUM_TOKENS", "32768")
    config = Settings.from_env()
    assert config.litert_max_num_tokens == 32768
    runtime = Runtime(
        settings(litert_model_path="/fake/model", litert_max_num_tokens=32768), FakeBackend()
    )
    assert runtime.litert_backend.max_num_tokens == 32768
    runtime.executor.shutdown()
    runtime.litert_executor.shutdown()
    monkeypatch.setenv("HAILO_LITERT_MAX_NUM_TOKENS", "0")
    with pytest.raises(ValueError, match="context"):
        Settings.from_env()


@pytest.mark.parametrize("stream", [False, True])
def test_llama_cpp_top_p_reaches_native_sampler(stream):
    backend = NativeBackend()
    with TestClient(create_app(settings(), FakeBackend(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload(top_p=1.0, stream=stream))
        assert response.status_code == 200
        assert backend.calls[-1]["sampler_config"]["top_p"] == 1.0
        if stream:
            assert '"finish_reason": "tool_calls"' in response.text
        else:
            assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan")])
def test_invalid_top_p_is_rejected(value):
    with pytest.raises(ValidationError):
        ChatRequest(**payload(top_p=value))


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, None])
def test_valid_top_p(value):
    assert ChatRequest(**payload(top_p=value)).top_p == value
