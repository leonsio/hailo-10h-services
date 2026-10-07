"""Frigate virtual-model routing, dependency, isolation and streaming regressions."""

import copy
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hailo_services.app import create_app
from hailo_services.config import FRIGATE_ASSIST_MODEL, LLM_MODEL, VLM_MODEL, Settings
from hailo_services.frigate_assist import prepare
from hailo_services.frigate_prompt import compact_tool
from hailo_services.runtime import HailoBackend
from hailo_services.schemas import ChatRequest

SYSTEM = (
    """You are a helpful assistant for Frigate, a security camera NVR system.
Current server local date and time: 2026-10-07 at 09:02:19 PM
Available cameras:
  - Front Door (ID: front_door, zones: Entry (ID: entry))
  - Garden (ID: garden, zones: Lawn (ID: lawn))
"""
    + "Generic instructions not relevant to this question.\n" * 70
)


def tool(name, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Long explanation of all possible situations. " * 20,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool("get_profile_status"),
    tool(
        "get_recap",
        {"after": {"type": "string"}, "before": {"type": "string"}, "cameras": {"type": "string"}},
        ["after", "before"],
    ),
    tool("get_live_context", {"camera": {"type": "string"}}, ["camera"]),
    tool(
        "search_objects",
        {
            "camera": {"type": "string"},
            "label": {"type": "string"},
            "semantic_query": {"type": "string", "description": "Appearance in English"},
            "sub_label": {"type": "string"},
            "after": {"type": "string"},
        },
    ),
    tool("find_similar_objects", {"event_id": {"type": "string"}}, ["event_id"]),
    tool(
        "set_camera_state",
        {
            "camera": {"type": "string"},
            "feature": {"type": "string", "enum": ["detect", "record"]},
            "value": {"type": "string", "enum": ["ON", "OFF"]},
        },
        ["camera", "feature", "value"],
    ),
    tool(
        "start_camera_watch",
        {"camera": {"type": "string"}, "condition": {"type": "string"}},
        ["camera", "condition"],
    ),
    tool("stop_camera_watch"),
]


def payload(question="Was ist passiert, während ich weg war?", **extra):
    return {
        "model": FRIGATE_ASSIST_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        "tools": copy.deepcopy(TOOLS),
        "tool_choice": "auto",
        **extra,
    }


def followup(body, call, data):
    updated = copy.deepcopy(body)
    updated["messages"] += [
        call,
        {"role": "tool", "tool_call_id": call["tool_calls"][0]["id"], "content": json.dumps(data)},
    ]
    return updated


def name_and_args(message):
    function = message["tool_calls"][0]["function"]
    return function["name"], json.loads(function["arguments"])


class Backend(HailoBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self.calls = []

    def start(self):
        pass

    def close(self):
        pass

    def chat(self, request, emit=None, cancelled=None):
        self.calls.append(request)
        return "visible objects only"


class LLM:
    def __init__(self):
        self.calls = []
        self.result = "summary from actual events"

    def start(self):
        pass

    def close(self):
        pass

    def chat(self, request, emit=None, cancelled=None):
        self.calls.append(request)
        request._metrics.update(input_tokens=123, output_tokens=7)
        return self.result


@pytest.fixture
def service():
    settings = Settings(
        litert_enabled=True, wyoming_port=0, minilm_enabled=False, whisper_enabled=False
    )
    backend, llm = Backend(settings), LLM()
    with TestClient(create_app(settings, backend, llm)) as client:
        yield client, backend, llm


def message(response):
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]


def test_catalogue_health_and_ui(service):
    client, _, _ = service
    assert FRIGATE_ASSIST_MODEL in [m["id"] for m in client.get("/v1/models").json()["data"]]
    health = client.get("/health").json()
    assert health["frigate_assist"]["experimental"]
    assert health["frigate_assist"]["text_ready"]
    assert FRIGATE_ASSIST_MODEL in health["models"]
    assert FRIGATE_ASSIST_MODEL in client.get("/ui/config").json()["vision_models"]


def test_absence_three_rounds_without_invented_times(service):
    client, backend, llm = service
    body = payload()
    first_response = client.post("/v1/chat/completions", json=body)
    first = message(first_response)
    assert name_and_args(first) == ("get_profile_status", {})
    assert first_response.json()["metrics"]["frigate_route"]["inference_calls"] == 0
    assert not backend.calls and not llm.calls
    profile = {
        "active_profile": "Home",
        "profiles": ["Home", "Away"],
        "last_activated": {"Away": "2026-10-07 05:00:00 PM", "Home": "2026-10-07 08:30:00 PM"},
    }
    body = followup(body, first, profile)
    second = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(second) == (
        "get_recap",
        {"after": "2026-10-07T17:00:00", "before": "2026-10-07T20:30:00"},
    )
    body = followup(
        body,
        second,
        {
            "events": [
                {
                    "camera": "Front Door",
                    "start_time_local": "2026-10-07 06:10:00 PM",
                    "description": "Person at the gate",
                    "severity": "detection",
                }
            ]
        },
    )
    final = message(client.post("/v1/chat/completions", json=body))
    assert final["content"] == llm.result
    assert len(llm.calls) == 1 and not backend.calls
    compiled = llm.calls[0]
    assert compiled.model == LLM_MODEL
    assert not compiled.tools
    assert "2026-10-07 06:10:00 PM" in json.dumps(compiled.messages)
    assert "Generic instructions" not in json.dumps(compiled.messages)
    assert not getattr(compiled, "_ha_assist", False)


@pytest.mark.parametrize(
    "profile",
    [
        {"active_profile": "Home", "last_activated": {"Holiday": "2026-10-07 05:00:00 PM"}},
        {"active_profile": "Home", "last_activated": {"Away": "invalid"}},
        {"error": "profile manager unavailable"},
        {"active_profile": "Home", "last_activated": {"Away": "2026-10-08 05:00:00 PM"}},
    ],
)
def test_absence_ambiguous_requires_clarification(service, profile):
    client, backend, llm = service
    body = payload()
    first = message(client.post("/v1/chat/completions", json=body))
    result = message(client.post("/v1/chat/completions", json=followup(body, first, profile)))
    assert "Von wann bis wann" in result["content"]
    assert not result.get("tool_calls") and not backend.calls and not llm.calls


def test_active_away_uses_supplied_server_clock():
    body = ChatRequest(**payload())
    first = prepare(Settings(), body)[1]
    body = ChatRequest(
        **followup(
            payload(),
            first,
            {"active_profile": "Away", "last_activated": {"Away": "2026-10-07 05:00:00 PM"}},
        )
    )
    assert name_and_args(prepare(Settings(), body)[1])[1]["before"] == "2026-10-07T21:02:19"


def test_custom_away_profile_is_explicit_configuration():
    settings = Settings(frigate_assist_away_profiles="Urlaub")
    first = prepare(settings, ChatRequest(**payload()))[1]
    body = ChatRequest(
        **followup(
            payload(),
            first,
            {"active_profile": "Urlaub", "last_activated": {"Urlaub": "2026-10-07 05:00:00 PM"}},
        )
    )
    assert name_and_args(prepare(settings, body)[1])[0] == "get_recap"


def test_live_tool_then_image_routes_to_vlm(service):
    client, backend, llm = service
    body = payload("Was ist gerade an Kamera Front Door zu sehen?")
    first = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(first) == ("get_live_context", {"camera": "front_door"})
    body = followup(body, first, {"camera": "front_door", "detections": []})
    body["messages"].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the current live image from camera 'front_door'.",
                },
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
            ],
        }
    )
    message(client.post("/v1/chat/completions", json=body))
    assert not llm.calls and len(backend.calls) == 1
    compiled = backend.calls[0]
    assert compiled.model == VLM_MODEL and not compiled.tools and compiled.tool_choice == "none"
    assert len(compiled.messages) == 2
    assert "Was ist gerade" in compiled.messages[1]["content"][0]["text"]
    assert "Never invent" in compiled.messages[0]["content"]
    assert "Generic instructions" not in str(compiled.messages)


def test_image_description_without_tools_and_zero_temperature(service):
    client, backend, llm = service
    body = payload("")
    body.pop("tools")
    body["temperature"] = 0
    body["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the car, one sentence."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
            ],
        }
    ]
    result = client.post("/v1/chat/completions", json=body)
    assert message(result)["content"] == "visible objects only"
    assert result.json()["model"] == FRIGATE_ASSIST_MODEL
    assert backend.calls[0].temperature == 0.01
    assert not llm.calls


def test_native_chat_is_not_compiled(service):
    client, backend, llm = service
    body = payload("General question", model=VLM_MODEL)
    message(client.post("/v1/chat/completions", json=body))
    assert backend.calls[0].messages == body["messages"]
    assert backend.calls[0].tools == body["tools"]
    assert "frigate_route" not in backend.calls[0]._metrics
    body["model"] = LLM_MODEL
    message(client.post("/v1/chat/completions", json=body))
    assert llm.calls[0].messages == body["messages"]


def test_text_routing_and_tool_selection(service):
    client, backend, llm = service
    message(client.post("/v1/chat/completions", json=payload("Zeige alle Autos heute")))
    assert not backend.calls and len(llm.calls) == 1
    compiled = llm.calls[0]
    assert [t["function"]["name"] for t in compiled.tools] == ["search_objects"]
    assert (
        len(json.dumps(compiled.model_dump()))
        < len(json.dumps(payload("Zeige alle Autos heute"))) / 2
    )
    assert "semantic_query in English" in compiled.tools[0]["function"]["description"]


def test_unknown_camera_generated_call_is_rejected(service):
    client, _, llm = service
    llm.result = {
        "tool_calls": [
            {"function": {"name": "search_objects", "arguments": {"camera": "invented"}}}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload("Zeige alle Autos heute"))
    assert response.status_code == 400
    assert "unknown" in response.text


def test_schema_violation_generated_call_is_rejected(service):
    client, _, llm = service
    llm.result = {
        "tool_calls": [{"function": {"name": "search_objects", "arguments": {"invented": True}}}]
    }
    assert (
        client.post("/v1/chat/completions", json=payload("Zeige alle Autos heute")).status_code
        == 400
    )


def test_technical_status_is_a_clarification(service):
    client, backend, llm = service
    result = message(
        client.post(
            "/v1/chat/completions", json=payload("Wie ist der aktuelle Status meiner Kameras?")
        )
    )
    assert "technischen" in result["content"]
    assert not llm.calls and not backend.calls


def test_deterministic_setting_and_unsupported_watch(service):
    client, backend, llm = service
    result = message(
        client.post(
            "/v1/chat/completions", json=payload("Schalte die Erkennung für Kamera Front Door aus")
        )
    )
    assert name_and_args(result) == (
        "set_camera_state",
        {"camera": "front_door", "feature": "detect", "value": "OFF"},
    )
    result = message(
        client.post("/v1/chat/completions", json=payload("Benachrichtige mich wenn Gäste kommen"))
    )
    assert "noch nicht sicher unterstützt" in result["content"]
    assert not backend.calls and not llm.calls


def test_truncated_events_are_explicit_and_errors_remain(service):
    client, _, llm = service
    body = payload()
    call = prepare(Settings(), ChatRequest(**body))[1]
    data = {
        "events": [
            {"start_time_local": "unchanged", "description": str(i), "box": [0, 0, 1, 1]}
            for i in range(30)
        ],
        "error": "partial database failure",
    }
    # Summary follows a completed recap (a valid client-supplied round).
    call["tool_calls"][0]["function"] = {
        "name": "get_recap",
        "arguments": '{"after":"2026-10-07T17:00:00","before":"2026-10-07T20:00:00"}',
    }
    message(client.post("/v1/chat/completions", json=followup(body, call, data)))
    compiled = llm.calls[0]
    result = json.loads(compiled.messages[-1]["content"])
    assert result["events"][-1] == {"_omitted_records": 18, "_total_records": 30}
    assert result["error"] == "partial database failure"
    assert "box" not in result["events"][0]


def test_stream_tool_calls_usage_and_virtual_name(service):
    client, backend, llm = service
    response = client.post(
        "/v1/chat/completions", json=payload(stream=True, stream_options={"include_usage": True})
    )
    chunks = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and "[DONE]" not in line
    ]
    assert all(chunk["model"] == FRIGATE_ASSIST_MODEL for chunk in chunks)
    assert any(c["choices"] and c["choices"][0]["finish_reason"] == "tool_calls" for c in chunks)
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 0
    assert chunks[-1]["metrics"]["frigate_route"]["route"] == "direct"
    assert response.text.endswith("data: [DONE]\n\n")
    assert not llm.calls and not backend.calls


def test_stream_text_usage(service):
    client, _, _ = service
    response = client.post(
        "/v1/chat/completions",
        json=payload("Zeige alle Autos heute", stream=True, stream_options={"include_usage": True}),
    )
    chunks = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and "[DONE]" not in line
    ]
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 123,
        "completion_tokens": 7,
        "total_tokens": 130,
    }


def test_stream_options_only_for_proxy():
    with pytest.raises(ValidationError, match="only by Frigate-Assist"):
        ChatRequest(**payload(model=VLM_MODEL, stream_options={"include_usage": True}))
    with pytest.raises(ValidationError):
        ChatRequest(**payload(stream_options={"unknown": True}))


def test_schema_compaction_retains_constraints():
    original = tool(
        "test",
        {"label": {"type": "string", "enum": ["person", "car"], "description": "Verbose"}},
        ["label"],
    )
    compact = compact_tool(original)
    assert compact["function"]["parameters"]["properties"]["label"] == {
        "type": "string",
        "enum": ["person", "car"],
    }
    assert compact["function"]["parameters"]["required"] == ["label"]
    assert "description" in original["function"]["parameters"]["properties"]["label"]


def test_schema_compaction_preserves_annotation_named_parameters_and_enum_objects():
    original = tool(
        "test",
        {
            "description": {"type": "string", "description": "prose"},
            "title": {"const": {"description": "required value"}},
        },
        ["description"],
    )
    schema = compact_tool(original)["function"]["parameters"]
    assert schema["properties"]["description"] == {"type": "string"}
    assert schema["properties"]["title"]["const"] == {"description": "required value"}


def test_single_camera_does_not_replace_an_unknown_requested_camera():
    body = payload("What is visible now on camera Nonexistent?")
    body["messages"][0]["content"] = SYSTEM.split("  - Garden")[0]
    _, direct = prepare(Settings(), ChatRequest(**body))
    assert isinstance(direct, str) and "Which camera" in direct


def test_missing_user_question_is_a_client_error():
    with pytest.raises(ValueError, match="requires a user"):
        prepare(
            Settings(),
            ChatRequest(
                model=FRIGATE_ASSIST_MODEL, messages=[{"role": "system", "content": "system only"}]
            ),
        )


def test_historical_image_is_labelled_and_earlier_frames_are_marked_omitted():
    image = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
        ],
    }
    body = payload()
    body["messages"] = [
        copy.deepcopy(image),
        copy.deepcopy(image),
        {"role": "user", "content": "What changed?"},
    ]
    prepared, _ = prepare(Settings(), ChatRequest(**body))
    task = prepared.messages[1]["content"][0]["text"]
    assert "earlier conversation" in task and "1 earlier images omitted" in task
    assert len(prepared.messages[1]["content"]) == 2


def test_invalid_tool_dependency_is_not_hidden():
    body = payload()
    body["messages"].append({"role": "tool", "tool_call_id": "missing", "content": "{}"})
    with pytest.raises(ValueError, match="pending"):
        prepare(Settings(), ChatRequest(**body))


def test_oversized_vision_constraints_fail_clearly():
    body = payload()
    body["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Important output contract " * 100},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
            ],
        }
    ]
    with pytest.raises(ValueError, match="vision task exceeds"):
        prepare(Settings(), ChatRequest(**body))


def test_no_text_backend_fallback_and_disabled_proxy():
    with pytest.raises(ValueError, match="require Gemma"):
        prepare(Settings(), ChatRequest(**payload("Explain a camera concept")))
    with pytest.raises(ValueError, match="disabled"):
        prepare(Settings(frigate_assist_enabled=False), ChatRequest(**payload()))


def test_no_invented_instruction_from_an_old_image_question():
    body = payload("Old question about a person")
    body["messages"].append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this new car."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
            ],
        }
    )
    prepared, direct = prepare(Settings(), ChatRequest(**body))
    assert direct is None
    assert "Old question" not in str(prepared.messages)
