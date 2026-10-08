"""Shared HassIL recognition and Frigate deterministic routing regressions."""

import json

import pytest

from hailo_services.config import FRIGATE_ASSIST_MODEL, Settings
from hailo_services.frigate_intents import hassil_plan, recognize_request
from hailo_services.schemas import ChatRequest

SYSTEM = """You are a helpful assistant for Frigate, a security camera NVR system.
Current server local date and time: 2026-10-08 at 08:30:00 PM
Available cameras:
  - Front Door (ID: front_door, zones: Entry (ID: entry))
  - Garden (ID: garden, zones: Lawn (ID: lawn))
"""


def tool(name, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool(
        "get_recap",
        {"after": {"type": "string"}, "before": {"type": "string"}},
        ["after", "before"],
    ),
    tool("get_live_context", {"camera": {"type": "string"}}, ["camera"]),
    tool(
        "search_objects",
        {
            "camera": {"type": "string"},
            "label": {"type": "string"},
            "after": {"type": "string"},
            "before": {"type": "string"},
            "limit": {"type": "integer"},
        },
    ),
    tool("get_profile_status"),
    tool(
        "set_camera_state",
        {
            "camera": {"type": "string"},
            "feature": {"type": "string", "enum": ["detect", "record"]},
            "value": {"type": "string", "enum": ["ON", "OFF"]},
        },
        ["camera", "feature", "value"],
    ),
    tool("stop_camera_watch"),
]


def request(text, *, language=None):
    data = {
        "model": FRIGATE_ASSIST_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    if language:
        data["language"] = language
    return ChatRequest(**data)


def call(message):
    function = message["tool_calls"][0]["function"]
    return function["name"], json.loads(function["arguments"])


def test_frigate_hassil_uses_request_scoped_camera_and_canonical_slots():
    match, trace = recognize_request(request("Zeige alle Autos heute", language="de"))
    assert trace["matched"]
    assert match == {
        "intent": "FrigateSearchObjects",
        "slots": {"label": "car", "period": "today"},
        "language": "de",
    }

    match, _ = recognize_request(request("Zeige mir das Bild von Front Door", language="de"))
    assert match["intent"] == "FrigateLiveContext"
    assert match["slots"]["camera"] == "front_door"


def test_compound_request_falls_through_instead_of_dropping_capabilities():
    match, trace = recognize_request(
        request("Show cars yesterday and notify me when a person arrives", language="en")
    )
    assert match is None
    assert trace["reason"] in {"no_match", "ambiguous"}


def test_hassil_object_search_constrains_the_model_without_losing_filters():
    tools, direct, reason = hassil_plan(
        request("Zeige alle Autos heute", language="de"), Settings(), False
    )
    assert reason == "hassil_object_search_tool_selection"
    assert direct is None
    assert [item["function"]["name"] for item in tools] == ["search_objects"]
    properties = tools[0]["function"]["parameters"]["properties"]
    assert properties["label"]["enum"] == ["car"]
    assert properties["after"]["enum"] == ["2026-10-08T00:00:00"]
    assert properties["before"]["enum"] == ["2026-10-08T20:30:00"]


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("de", "Zeige die Ereignisse gestern"),
        ("en", "Show events yesterday"),
        ("fr", "Montre les événements hier"),
        ("es", "Muestra los eventos ayer"),
        ("it", "Mostra gli eventi ieri"),
        ("nl", "Toon de gebeurtenissen gisteren"),
        ("pt", "Mostra os eventos ontem"),
        ("ru", "Покажи события вчера"),
    ],
)
def test_recap_grammar_covers_existing_multilingual_router_languages(language, text):
    match, trace = recognize_request(request(text, language=language))
    assert trace["matched"], trace
    assert match["intent"] == "FrigateRecap"
    assert match["slots"]["period"] == "yesterday"


def test_tool_none_keeps_hassil_advisory_only():
    body = request("Zeige die Ereignisse gestern", language="de")
    body = body.model_copy(update={"tool_choice": "none"})
    match, trace = recognize_request(body)
    assert match is None and trace["reason"] == "ineligible"
    assert hassil_plan(body, Settings(), False) is None
