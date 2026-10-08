"""Regression tests for multilingual deterministic Frigate/NVR routing."""

import json

import pytest

from hailo_services.config import FRIGATE_ASSIST_MODEL, LLM_MODEL, Settings
from hailo_services.frigate_assist import prepare
from hailo_services.schemas import ChatRequest

SYSTEM = """You are a helpful assistant for Frigate, a security camera NVR system.
Current server local date and time: 2026-10-08 at 01:14:20 PM
Available cameras:
  - Front Door (ID: front_door, zones: Entry (ID: entry))
  - Garden (ID: garden, zones: Lawn (ID: lawn))
"""


def tool(name, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Frigate client tool",
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
        {
            "after": {"type": "string"},
            "before": {"type": "string"},
            "cameras": {"type": "string"},
        },
        ["after", "before"],
    ),
    tool("get_live_context", {"camera": {"type": "string"}}, ["camera"]),
    tool(
        "search_objects",
        {
            "camera": {"type": "string"},
            "label": {"type": "string"},
            "sub_label": {"type": "string"},
            "after": {"type": "string"},
            "before": {"type": "string"},
            "semantic_query": {"type": "string"},
            "limit": {"type": "integer"},
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

SETTINGS = Settings(litert_enabled=True, frigate_assist_text_model=LLM_MODEL)


def request(question, messages=None, tools=None):
    return ChatRequest(
        model=FRIGATE_ASSIST_MODEL,
        messages=messages
        or [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        tools=tools or TOOLS,
        tool_choice="auto",
    )


def call(message):
    function = message["tool_calls"][0]["function"]
    arguments = function["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return function["name"], arguments


@pytest.mark.parametrize(
    "question",
    [
        "was ist in den letzten 3 Stunden passiert",
        "what happened in the last 3 hours",
        "que s'est-il passé pendant les 3 dernières heures",
        "qué pasó en las últimas 3 horas",
        "cosa è successo nelle ultime 3 ore",
        "wat is er de afgelopen 3 uur gebeurd",
        "o que aconteceu nas últimas 3 horas",
        "что произошло за последние 3 часа",
    ],
)
def test_last_three_hours_is_direct_in_multiple_languages(question):
    prepared, direct = prepare(SETTINGS, request(question))
    assert call(direct) == (
        "get_recap",
        {"after": "2026-10-08T10:14:20", "before": "2026-10-08T13:14:20"},
    )
    assert prepared._metrics["frigate_route"]["inference_calls"] == 0


@pytest.mark.parametrize(
    "question",
    [
        "from 11:00 to 13:00",
        "von 11:00 bis 13:00",
        "de 11:00 à 13:00",
        "de 11:00 a 13:00",
        "dalle 11:00 alle 13:00",
        "van 11:00 tot 13:00",
        "das 11:00 às 13:00",
        "с 11:00 до 13:00",
    ],
)
def test_time_only_followup_inherits_historical_recap(question):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "what happened in the last 3 hours"},
        {"role": "assistant", "content": "Please specify a time window."},
        {"role": "user", "content": question},
    ]
    _, direct = prepare(SETTINGS, request(question, messages=messages))
    assert call(direct) == (
        "get_recap",
        {"after": "2026-10-08T11:00:00", "before": "2026-10-08T13:00:00"},
    )


def test_historical_object_query_exposes_one_required_tool_with_fixed_times():
    prepared, direct = prepare(SETTINGS, request("show cars in the last 3 hours"))
    assert direct is None
    assert prepared.tool_choice == "required"
    assert [item["function"]["name"] for item in prepared.tools] == ["search_objects"]
    properties = prepared.tools[0]["function"]["parameters"]["properties"]
    assert properties["after"]["enum"] == ["2026-10-08T10:14:20"]
    assert properties["before"]["enum"] == ["2026-10-08T13:14:20"]


def test_compound_historical_and_future_intents_are_not_discarded():
    prepared, direct = prepare(
        SETTINGS,
        request("Show cars yesterday and notify me when a person arrives"),
    )
    assert direct is None
    names = {item["function"]["name"] for item in prepared.tools}
    assert {"search_objects", "start_camera_watch"} <= names
    assert prepared.tool_choice == "auto"


def test_live_context_bypasses_model_for_exact_camera():
    _, direct = prepare(SETTINGS, request("what is visible live on Front Door camera"))
    assert call(direct) == ("get_live_context", {"camera": "front_door"})


def test_explicit_camera_toggle_bypasses_model():
    _, direct = prepare(SETTINGS, request("turn detection off for Front Door"))
    assert call(direct) == (
        "set_camera_state",
        {"camera": "front_door", "feature": "detect", "value": "OFF"},
    )


def test_watch_intent_limits_model_to_required_watch_tool():
    prepared, direct = prepare(
        SETTINGS,
        request("notify me when a person arrives at Front Door"),
    )
    assert direct is None
    assert prepared.tool_choice == "required"
    assert [item["function"]["name"] for item in prepared.tools] == ["start_camera_watch"]


def test_absence_recap_starts_with_profile_lookup_without_model():
    _, direct = prepare(SETTINGS, request("what happened while I was away"))
    assert call(direct) == ("get_profile_status", {})
