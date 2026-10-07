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
            "before": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
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
        litert_enabled=True,
        frigate_assist_text_model=LLM_MODEL,
        wyoming_port=0,
        minilm_enabled=False,
        whisper_enabled=False,
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
    assert compiled.model == VLM_MODEL and not compiled.tools and compiled.tool_choice is None
    assert len(compiled.messages) == 2
    assert "Was ist gerade" in compiled.messages[1]["content"][0]["text"]
    assert "Keine erfundenen" in compiled.messages[0]["content"]
    assert "Antworte auf Deutsch" in compiled.messages[0]["content"]
    assert "Here is the current live image" not in compiled.messages[1]["content"][0]["text"]
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


def test_live_image_short_reply_single_and_multiple_cameras(service):
    client, backend, llm = service
    body = payload("Wie ist der aktuelle Status meiner Kameras?")
    first = message(client.post("/v1/chat/completions", json=body))
    body["messages"] += [first, {"role": "user", "content": "das aktuelle Kamerabild"}]
    result = message(client.post("/v1/chat/completions", json=body))
    assert "Welche Kamera" in result["content"]
    body["messages"][0]["content"] = SYSTEM.split("  - Garden")[0]
    result = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(result) == ("get_live_context", {"camera": "front_door"})
    assert not llm.calls and not backend.calls


@pytest.mark.parametrize(
    ("language", "instruction"),
    [("de", "Antworte auf Deutsch"), ("en", "Answer in English"), ("ru", "Answer in Russian")],
)
def test_vision_explicit_language_overrides_caption(service, language, instruction):
    client, backend, _ = service
    body = payload("Describe the live image", language=language)
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
    assert instruction in backend.calls[0].messages[0]["content"]


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


@pytest.mark.parametrize(
    ("question", "after", "before"),
    [
        (
            "Zeige mir die Ereignisse der letzten Stunde",
            "2026-10-07T20:02:19",
            "2026-10-07T21:02:19",
        ),
        ("Show me all events in the last hour", "2026-10-07T20:02:19", "2026-10-07T21:02:19"),
        (
            "Zeige alle Ereignisse der letzten 30 Minuten",
            "2026-10-07T20:32:19",
            "2026-10-07T21:02:19",
        ),
        ("Zeige die Ereignisse heute", "2026-10-07T00:00:00", "2026-10-07T21:02:19"),
        ("Show events yesterday", "2026-10-06T00:00:00", "2026-10-07T00:00:00"),
        (
            "Zeige die Ereignisse ab heute 06:00 Uhr bis jetzt",
            "2026-10-07T06:00:00",
            "2026-10-07T21:02:19",
        ),
    ],
)
def test_event_intervals_without_inference(service, question, after, before):
    client, backend, llm = service
    result = message(client.post("/v1/chat/completions", json=payload(question)))
    assert name_and_args(result) == ("get_recap", {"after": after, "before": before})
    assert not backend.calls and not llm.calls
    result = message(
        client.post(
            "/v1/chat/completions", json=followup(payload(question), result, {"events": []})
        )
    )
    assert "keine Aktivitäten" in result["content"] or "No activity" in result["content"]
    assert not llm.calls


@pytest.mark.parametrize(
    "data",
    [
        {"events": [], "error": "database failed"},
        {"events": [], "partial": True},
        {"events": [], "message": "Partial query; remaining cameras unavailable"},
        {"events": [{"description": "Person detected"}]},
    ],
)
def test_empty_recap_does_not_hide_errors_or_partial_results(service, data):
    client, _, llm = service
    body = payload("Zeige mir die Ereignisse der letzten Stunde")
    call = message(client.post("/v1/chat/completions", json=body))
    result = message(client.post("/v1/chat/completions", json=followup(body, call, data)))
    assert result["content"] == "summary from actual events" and len(llm.calls) == 1


def test_empty_recap_stream_completion_is_logged(service, caplog):
    client, _, llm = service
    caplog.set_level("INFO", logger="hailo_services.app")
    body = payload("Zeige mir die Ereignisse der letzten Stunde")
    call = message(client.post("/v1/chat/completions", json=body))
    body = followup(
        body, call, {"events": [], "message": "No activity was found during this time period."}
    )
    body.update(stream=True, stream_options={"include_usage": True})
    response = client.post("/v1/chat/completions", json=body)
    assert response.text.endswith("data: [DONE]\n\n")
    assert "status=completed finish_reason=stop done_emitted=true" in caplog.text
    assert not llm.calls


def test_watch_request_does_not_promise_unsupported_monitoring(service):
    client, backend, llm = service
    result = message(
        client.post(
            "/v1/chat/completions",
            json=payload("Pass auf die Haustür auf und sag mir Bescheid, wenn jemand kommt"),
        )
    )
    assert "noch nicht sicher unterstützt" in result["content"]
    assert not backend.calls and not llm.calls


def test_event_interval_midnight_and_search_fallback(service):
    client, _, llm = service
    body = payload("Zeige mir die Ereignisse der letzten Stunde")
    body["messages"][0]["content"] = SYSTEM.replace("09:02:19 PM", "12:20:00 AM")
    body["tools"] = [t for t in body["tools"] if t["function"]["name"] != "get_recap"]
    result = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(result) == (
        "search_objects",
        {"after": "2026-10-06T23:20:00", "before": "2026-10-07T00:20:00"},
    )
    assert not llm.calls


def test_time_followups_do_not_copy_bad_assistant_dates(service):
    client, _, llm = service
    body = payload("Zeige mir die Ereignisse der letzten Stunde")
    body["messages"] += [
        {"role": "assistant", "content": "Bitte gib einen Zeitraum an: 10:02 PM bis 11:02 PM."},
        {"role": "user", "content": "von heute morgen bis jetzt"},
    ]
    result = message(client.post("/v1/chat/completions", json=body))
    assert "Ab welcher Uhrzeit" in result["content"]
    body["messages"] += [result, {"role": "user", "content": "ab 06:00 Uhr bis jetzt"}]
    result = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(result) == (
        "get_recap",
        {"after": "2026-10-07T06:00:00", "before": "2026-10-07T21:02:19"},
    )
    assert not llm.calls


def test_event_interval_needs_clock_and_honors_tool_none(service):
    client, _, llm = service
    body = payload("Zeige mir die Ereignisse der letzten Stunde")
    body["messages"][0]["content"] = "No clock supplied"
    result = message(client.post("/v1/chat/completions", json=body))
    assert "Serverzeit" in result["content"] and not llm.calls
    body["tool_choice"] = "none"
    result = message(client.post("/v1/chat/completions", json=body))
    assert "tool_calls" not in result and len(llm.calls) == 1


def test_filtered_or_unrelated_questions_do_not_inherit_event_interval(service):
    client, _, llm = service
    body = payload("Zeige mir die Ereignisse der letzten Stunde von Kamera Garden")
    message(client.post("/v1/chat/completions", json=body))
    assert len(llm.calls) == 1
    body = payload("Zeige mir die Ereignisse der letzten Stunde")
    body["messages"] += [
        {"role": "assistant", "content": "Ab wann?"},
        {"role": "user", "content": "Eine andere Frage"},
        {"role": "assistant", "content": "Antwort"},
        {"role": "user", "content": "ab 06:00 Uhr bis jetzt"},
    ]
    message(client.post("/v1/chat/completions", json=body))
    assert len(llm.calls) == 2


def test_relative_event_stream_finishes_and_next_request_is_accepted(service):
    client, _, llm = service
    response = client.post(
        "/v1/chat/completions",
        json=payload(
            "Zeige mir die Ereignisse der letzten Stunde",
            stream=True,
            stream_options={"include_usage": True},
        ),
    )
    chunks = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and "[DONE]" not in line
    ]
    call = next(
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
    )
    assert name_and_args({"tool_calls": [call]}) == (
        "get_recap",
        {"after": "2026-10-07T20:02:19", "before": "2026-10-07T21:02:19"},
    )
    assert chunks[-1]["usage"]["total_tokens"] == 0
    assert response.text.endswith("data: [DONE]\n\n")
    assert message(client.post("/v1/chat/completions", json=payload("Weitere Frage")))["content"]
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"after": "2026-10-07 10:02:08 PM", "before": "2026-10-07 11:02:08 PM"},
        {"after": "2026-10-07T20:00:00", "before": "2026-10-07T23:00:00"},
        {"after": "2026-10-07T20:00:00", "before": "2026-10-07T19:00:00"},
        {"after": "2026-10-07T20:00:00Z"},
        {"after": "2026-02-30T00:00:00"},
    ],
)
def test_generated_historical_dates_rejected_before_streaming(service, arguments):
    client, _, llm = service
    llm.result = {"tool_calls": [{"function": {"name": "search_objects", "arguments": arguments}}]}
    response = client.post(
        "/v1/chat/completions", json=payload("Zeige alle Autos heute", stream=True)
    )
    assert '"error"' in response.text
    assert '"tool_calls"' not in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    # A failed round releases the service for the next request.
    llm.result = "summary from actual events"
    assert message(client.post("/v1/chat/completions", json=payload("Eine weitere Frage")))[
        "content"
    ]


@pytest.mark.parametrize(
    "question,name",
    [
        ("wann wurde Leo zuletzt gesehen?", "Leo"),
        ('Ich meine die Person "Leo" wann wurde es zuletzt erkannt?', "Leo"),
        ('Wann wurde die Person "Alex Müller" zuletzt erkannt?', "Alex Müller"),
        ("When was the person Alex last seen?", "Alex"),
        ("Wann wurde DHL zuletzt gesehen?", "DHL"),
    ],
)
def test_named_last_seen_search_is_deterministic(service, question, name):
    client, backend, llm = service
    result = message(client.post("/v1/chat/completions", json=payload(question)))
    assert name_and_args(result) == ("search_objects", {"sub_label": name, "limit": 1})
    assert not llm.calls and not backend.calls


@pytest.mark.parametrize(
    "question",
    [
        "Wann wurde ein Auto zuletzt gesehen?",
        "Wann wurde die Person mit roter Jacke zuletzt gesehen?",
        "Wann wurde Leo zuletzt gesehen an Kamera Unknown?",
    ],
)
def test_last_seen_recognizer_does_not_drop_extra_filters(service, question):
    client, _, llm = service
    message(client.post("/v1/chat/completions", json=payload(question)))
    assert len(llm.calls) == 1


def test_generated_camera_friendly_name_is_normalized_without_mutating_model_output(service):
    client, _, llm = service
    llm.result = {
        "tool_calls": [
            {
                "function": {
                    "name": "search_objects",
                    "arguments": {"camera": "Front Door", "label": "car"},
                }
            }
        ]
    }
    original = copy.deepcopy(llm.result)
    result = message(client.post("/v1/chat/completions", json=payload("Zeige alle Autos heute")))
    assert name_and_args(result) == ("search_objects", {"camera": "front_door", "label": "car"})
    assert llm.result == original


def test_ambiguous_friendly_camera_name_remains_rejected(service):
    client, _, llm = service
    body = payload("Zeige alle Autos heute")
    body["messages"][0]["content"] = SYSTEM.replace("Garden (ID: garden", "Front Door (ID: garden")
    llm.result = {
        "tool_calls": [
            {"function": {"name": "search_objects", "arguments": {"camera": "Front Door"}}}
        ]
    }
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400 and "ambiguous" in response.text


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
    with pytest.raises(ValueError, match="not an enabled LLM"):
        prepare(
            Settings(frigate_assist_text_model=LLM_MODEL),
            ChatRequest(**payload("Explain a camera concept")),
        )
    with pytest.raises(ValueError, match="disabled"):
        prepare(Settings(frigate_assist_enabled=False), ChatRequest(**payload()))


def test_native_hailo_text_target_routes_through_hailo_and_reports_health():
    settings = Settings(
        vlm_enabled=False,
        hailo_llm_enabled=True,
        hailo_llm_model="Qwen3-1.7B-Instruct",
        frigate_assist_text_model="Qwen3-1.7B-Instruct",
        litert_enabled=True,
        minilm_enabled=False,
        whisper_enabled=False,
        wyoming_port=0,
    )
    backend, gemma = Backend(settings), LLM()
    with TestClient(create_app(settings, backend, gemma)) as client:
        health = client.get("/health").json()["frigate_assist"]
        assert health["text_model"] == "Qwen3-1.7B-Instruct" and health["text_ready"]
        assert not health["vision_ready"]
        response = client.post("/v1/chat/completions", json=payload("Zeige alle Autos heute"))
        message(response)
        assert len(backend.calls) == 1 and not gemma.calls
        assert backend.calls[0].model == "Qwen3-1.7B-Instruct"
        assert response.json()["metrics"]["frigate_route"]["backend_model"] == "Qwen3-1.7B-Instruct"
        assert not getattr(backend.calls[0], "_ha_assist", False)


def test_native_hailo_target_readiness_does_not_follow_unrelated_gemma():
    settings = Settings(
        frigate_assist_text_model="Qwen3-1.7B-Instruct",
        litert_enabled=True,
        minilm_enabled=False,
        whisper_enabled=False,
        wyoming_port=0,
    )
    backend, gemma = Backend(settings), LLM()
    with TestClient(create_app(settings, backend, gemma)) as client:
        assert not client.get("/health").json()["frigate_assist"]["text_ready"]
        response = client.post("/v1/chat/completions", json=payload("Explain a camera concept"))
        assert response.status_code == 400 and "not an enabled LLM" in response.text
        assert not backend.calls and not gemma.calls


def test_text_and_vision_targets_are_selected_independently_before_native_startup():
    # Planning for a future compatible runtime; this does not bypass HailoRT's
    # actual startup compatibility guard or claim current simultaneous support.
    settings = Settings(
        hailo_llm_enabled=True,
        hailo_llm_model="Qwen3-1.7B-Instruct",
        frigate_assist_text_model="Qwen3-1.7B-Instruct",
    )
    text, direct = prepare(settings, ChatRequest(**payload("Explain a camera concept")))
    assert direct is None and text.model == "Qwen3-1.7B-Instruct"
    image = payload()
    image["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this car"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
            ],
        }
    ]
    vision, direct = prepare(settings, ChatRequest(**image))
    assert direct is None and vision.model == VLM_MODEL


@pytest.mark.parametrize("target", ["unknown", "Qwen3-VL-2B-Instruct"])
def test_invalid_or_wrong_role_text_target_is_rejected(target):
    with pytest.raises(ValueError, match="not an enabled LLM"):
        prepare(
            Settings(litert_enabled=True, frigate_assist_text_model=target),
            ChatRequest(**payload("Explain a camera concept")),
        )


@pytest.mark.parametrize(
    "key,target",
    [
        ("frigate_assist_text_model", FRIGATE_ASSIST_MODEL),
        ("frigate_assist_text_model", "HA-Assist"),
        ("frigate_assist_vision_model", FRIGATE_ASSIST_MODEL),
    ],
)
def test_invalid_proxy_target_configuration(key, target):
    with pytest.raises(ValueError):
        Settings(**{key: target})


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


@pytest.mark.parametrize("model", ["Qwen2-VL-2B-Instruct", "Qwen3-VL-2B-Instruct"])
@pytest.mark.parametrize("selection", ["", "  ", None, "explicit"])
def test_vlm_only_defaults_and_explicit_text_selection(model, selection):
    selected = model if selection == "explicit" else selection
    settings = Settings(
        vlm_hef=model,
        frigate_assist_text_model=selected,
        frigate_assist_vision_model=selected,
        wyoming_port=0,
    )
    body = payload("Explain a camera concept")
    prepared, direct = prepare(settings, ChatRequest(**body))
    assert direct is None and prepared.model == model
    assert settings.frigate_text_model == settings.frigate_vision_model == model
    assert prepared.tools and "return ONLY JSON" in prepared.messages[0]["content"]
    backend = Backend(settings)
    with TestClient(create_app(settings, backend)) as client:
        status = client.get("/health").json()["frigate_assist"]
        assert status["text_model"] == status["vision_model"] == model
        assert status["text_ready"] and status["vision_ready"]
        assert FRIGATE_ASSIST_MODEL in client.get("/ui/config").json()["vision_models"]
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        assert len(backend.calls) == 1 and backend.calls[0].model == model


def test_enabled_gemma_does_not_override_an_omitted_frigate_target():
    settings = Settings(litert_enabled=True, vlm_hef="Qwen3-VL-2B-Instruct")
    prepared, direct = prepare(settings, ChatRequest(**payload("Explain a camera concept")))
    assert direct is None and prepared.model == settings.vlm_model


def test_disabled_default_vlm_and_explicit_unavailable_targets_do_not_fall_back():
    with pytest.raises(ValueError, match="not an enabled LLM/VLM"):
        prepare(
            Settings(vlm_enabled=False, litert_enabled=True),
            ChatRequest(**payload("Explain a camera concept")),
        )
    with pytest.raises(ValueError, match="not an enabled LLM/VLM"):
        prepare(
            Settings(frigate_assist_text_model=LLM_MODEL),
            ChatRequest(**payload("Explain a camera concept")),
        )


def test_yaml_empty_targets_resolve_to_loaded_vlm(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        "settings:\n  frigate_assist_text_model:\n  frigate_assist_vision_model: ''\nmodels:\n  vlm:\n    enabled: true\n    model: Qwen3-VL-2B-Instruct\n"
    )
    monkeypatch.setenv("HAILO_CONFIG", str(path))
    settings = Settings.from_env()
    assert settings.frigate_text_model == settings.frigate_vision_model == "Qwen3-VL-2B-Instruct"


@pytest.mark.parametrize(
    "loaded,target",
    [
        ("Qwen3-VL-2B-Instruct", "Qwen2-VL-2B-Instruct"),
        ("Qwen2-VL-2B-Instruct", "Qwen3-VL-2B-Instruct"),
    ],
)
def test_conflicting_frigate_image_target_is_rejected_at_configuration_load(loaded, target):
    with pytest.raises(ValueError, match="does not match the enabled VLM"):
        Settings(vlm_hef=loaded, frigate_assist_vision_model=target)


def test_yaml_conflicting_vision_target_rejected_at_startup(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        "settings:\n  frigate_assist_vision_model: Qwen2-VL-2B-Instruct\nmodels:\n  vlm:\n    enabled: true\n    model: Qwen3-VL-2B-Instruct\n"
    )
    monkeypatch.setenv("HAILO_CONFIG", str(path))
    with pytest.raises(ValueError, match="does not match the enabled VLM"):
        Settings.from_env()


@pytest.mark.parametrize(
    "question",
    [
        "Wann hast du zuletzt Morgan gesehen?",
        "Wann ist Morgan zuletzt gesehen worden?",
        "When did you last see Morgan?",
        "Wann wurde Morgan zuletzt gesehen? Gib mir das letzte Bild dazu",
    ],
)
def test_general_last_sighting_forms_and_image_suffix(service, question):
    client, backend, llm = service
    call = message(client.post("/v1/chat/completions", json=payload(question)))
    assert name_and_args(call) == ("search_objects", {"sub_label": "Morgan", "limit": 1})
    assert not backend.calls and not llm.calls


def test_last_sighting_exact_camera_filter_and_followup_preserved(service):
    client, _, llm = service
    call = message(
        client.post(
            "/v1/chat/completions", json=payload("When was Morgan last seen at camera Garden?")
        )
    )
    assert name_and_args(call)[1] == {"sub_label": "Morgan", "camera": "garden", "limit": 1}
    body = payload("Wann wurde Morgan zuletzt gesehen? Gib mir das letzte Bild dazu")
    body["messages"] += [
        {"role": "assistant", "content": "Which camera?"},
        {"role": "user", "content": "Garden"},
    ]
    call = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(call)[1] == {"sub_label": "Morgan", "camera": "garden", "limit": 1}
    assert not llm.calls


@pytest.mark.parametrize(
    "question",
    [
        "Wann wurde zuletzt eine Person vor dem Garden erkannt?",
        "When was a person last detected at Garden?",
    ],
)
def test_class_last_sighting_requests_one_result(service, question):
    client, _, llm = service
    body = payload(question)
    call = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(call) == (
        "search_objects",
        {"label": "person", "camera": "garden", "limit": 1},
    )
    data = [
        {
            "label": "person",
            "camera": "garden",
            "start_time_local": "2026-10-07 09:13:52 PM",
            "end_time_local": "2026-10-07 09:14:14 PM",
        }
    ]
    answer = message(client.post("/v1/chat/completions", json=followup(body, call, data)))
    assert "09:13:52 PM" in answer["content"] and "09:14:14 PM" in answer["content"]
    assert not llm.calls


@pytest.mark.parametrize(
    "data",
    [
        [],
        [
            {
                "sub_label": "Morgan",
                "label": "person",
                "camera": "garden",
                "start_time_local": "2026-10-07 12:42:48 PM",
            }
        ],
    ],
)
def test_named_last_sighting_summarized_without_model(service, data):
    client, _, llm = service
    body = payload("When was Morgan last seen?")
    call = message(client.post("/v1/chat/completions", json=body))
    answer = message(client.post("/v1/chat/completions", json=followup(body, call, data)))
    assert answer["content"] and not llm.calls


@pytest.mark.parametrize(
    "data",
    [
        {"error": "database failure"},
        [
            {
                "sub_label": "SomeoneElse",
                "camera": "garden",
                "start_time_local": "2026-10-07 12:42:48 PM",
            }
        ],
        [{"sub_label": "Morgan", "camera": "garden", "start_time_local": "invalid"}],
    ],
)
def test_last_sighting_errors_and_mismatches_are_not_presented_as_proof(service, data):
    client, _, llm = service
    body = payload("When was Morgan last seen?")
    call = message(client.post("/v1/chat/completions", json=body))
    message(client.post("/v1/chat/completions", json=followup(body, call, data)))
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "question", ["Zeige mir das aktuelle Live-Bild an", "Show me the current live image"]
)
def test_single_camera_live_request_and_multi_camera_reply_are_direct(service, question):
    client, _, llm = service
    body = payload(question)
    body["messages"][0]["content"] = SYSTEM.split("  - Garden")[0]
    call = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(call) == ("get_live_context", {"camera": "front_door"})
    body = payload(question)
    body["messages"] += [
        {"role": "assistant", "content": "Which camera?"},
        {"role": "user", "content": "Garden"},
    ]
    call = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(call) == ("get_live_context", {"camera": "garden"})
    assert not llm.calls


def test_attached_event_clothing_never_uses_similarity_or_an_unrelated_old_image(service):
    client, backend, llm = service
    body = payload()
    body["messages"] += [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Old live frame"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
            ],
        },
        {"role": "assistant", "content": "Old description"},
        {
            "role": "user",
            "content": "[attached_event:event-123] Welche Kleidung hatte die Person an?",
        },
    ]
    answer = message(client.post("/v1/chat/completions", json=body))
    assert "kein Werkzeug" in answer["content"]
    assert not backend.calls and not llm.calls
    body["tools"].append(tool("get_event_image", {"event_id": {"type": "string"}}, ["event_id"]))
    answer = message(client.post("/v1/chat/completions", json=body))
    assert name_and_args(answer) == ("get_event_image", {"event_id": "event-123"})


@pytest.mark.parametrize(
    "question",
    [
        "Hast du es gerade erfunden?",
        "Warum?",
        "When was Morgan last seen?",
        "Explain camera settings",
    ],
)
def test_historical_image_does_not_route_text_followups_to_vlm(service, question):
    client, backend, llm = service
    body = payload("Original image question")
    body["messages"] += [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the current live image from camera 'garden'."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
            ],
        },
        {"role": "assistant", "content": "An unverified visual claim."},
        {"role": "user", "content": question},
    ]
    message(client.post("/v1/chat/completions", json=body))
    assert not backend.calls
    if llm.calls:
        assert not any(isinstance(m.get("content"), list) for m in llm.calls[0].messages)
        assert "unverified model claims" in llm.calls[0].messages[0]["content"]


def test_last_image_request_requires_image_tool_and_retains_actual_event_id(service):
    client, _, llm = service
    body = payload("When was Morgan last seen? Show me the last image")
    body["tools"].append(tool("get_event_image", {"event_id": {"type": "string"}}, ["event_id"]))
    call = message(client.post("/v1/chat/completions", json=body))
    data = [
        {
            "id": "event-123",
            "sub_label": "Morgan",
            "camera": "garden",
            "start_time_local": "2026-10-07 12:42:48 PM",
        }
    ]
    next_call = message(client.post("/v1/chat/completions", json=followup(body, call, data)))
    assert name_and_args(next_call) == ("get_event_image", {"event_id": "event-123"})
    assert not llm.calls


def test_vision_chat_output_cap_preserves_description_contracts():
    body = payload()
    body["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="}},
            ],
        }
    ]
    compiled, _ = prepare(Settings(frigate_assist_vision_max_tokens=64), ChatRequest(**body))
    assert compiled.max_tokens == 64
    body.pop("tools")
    body["messages"].insert(
        0, {"role": "system", "content": "Return JSON with a description field."}
    )
    compiled, _ = prepare(Settings(frigate_assist_vision_max_tokens=64), ChatRequest(**body))
    assert compiled.max_tokens == 256


@pytest.mark.parametrize(
    "question",
    [
        "Welche Farbe hatte die Kleidung von Morgan als er zuletzt gesehen wurde?",
        "What was Morgan wearing when they were last seen?",
    ],
)
def test_historical_clothing_searches_the_event_before_requesting_an_image(service, question):
    client, _, llm = service
    call = message(client.post("/v1/chat/completions", json=payload(question)))
    assert name_and_args(call) == ("search_objects", {"sub_label": "Morgan", "limit": 1})
    assert not llm.calls
