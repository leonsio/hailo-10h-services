"""Regression tests for deterministic Frigate recap and named-home shortcuts."""

import copy
import json

import pytest

from hailo_services.assistants.frigate.frigate_assist import prepare
from hailo_services.assistants.frigate.frigate_shortcuts import shortcut_plan
from hailo_services.config import FRIGATE_ASSIST_MODEL, Settings
from hailo_services.schemas import ChatRequest

SYSTEM = """You are a helpful assistant for Frigate, a security camera NVR system.
Current server local date and time: 2026-10-08 at 10:14:29 PM
Available cameras:
  - Eingang (ID: eingang, zones: Alles (ID: alles))
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
        {
            "after": {"type": "string"},
            "before": {"type": "string"},
            "cameras": {"type": "string"},
        },
        ["after", "before"],
    ),
    tool(
        "search_objects",
        {
            "camera": {"type": "string"},
            "label": {"type": "string"},
            "sub_label": {"type": "string"},
            "after": {"type": "string"},
            "before": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    ),
]


def payload(question):
    return {
        "model": FRIGATE_ASSIST_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        "tools": copy.deepcopy(TOOLS),
        "tool_choice": "auto",
    }


def followup(body, call, data):
    updated = copy.deepcopy(body)
    updated["messages"] += [
        call,
        {
            "role": "tool",
            "tool_call_id": call["tool_calls"][0]["id"],
            "content": json.dumps(data),
        },
    ]
    return updated


def name_and_args(message):
    function = message["tool_calls"][0]["function"]
    arguments = function["arguments"]
    return function["name"], json.loads(arguments) if isinstance(arguments, str) else arguments


def recap_events(count=14):
    return [
        {
            "camera": "eingang",
            "severity": "alert" if index % 2 == 0 else "detection",
            "objects": ["person"] if index < 7 else ["bicycle"],
            "zones": ["alles"],
            "time": f"{5 + index // 4:02d}:{59 - index:02d} PM",
            "duration_seconds": index,
        }
        for index in range(count)
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Zeige mir die Ereignisse der letzten 5 Stunden",
        "Zeige mir die Ereignisse der letzten 5 Stunden an",
    ],
)
def test_nonempty_recap_listing_is_direct_and_keeps_every_event(question):
    body = payload(question)
    request = ChatRequest(**body)
    _prepared, call = prepare(Settings(), request)
    assert name_and_args(call) == (
        "get_recap",
        {"after": "2026-10-08T17:14:29", "before": "2026-10-08T22:14:29"},
    )

    events = recap_events()
    result_request = ChatRequest(**followup(body, call, {"events": events}))
    _prepared, direct = prepare(Settings(), result_request)

    assert isinstance(direct, str)
    assert direct.count("\n") == 13
    assert "Eingang" in direct and "@ Alles" in direct
    assert events[0]["time"] in direct and events[-1]["time"] in direct
    assert result_request._metrics["frigate_route"]["reason"] == "deterministic_recap_list"
    assert result_request._metrics["frigate_route"]["inference_calls"] == 0


def test_recap_time_followup_inherits_previous_listing_and_stays_direct():
    first_body = payload("Zeige mir die Ereignisse der letzten Stunde")
    first_request = ChatRequest(**first_body)
    _prepared, first_call = prepare(Settings(), first_request)
    assert name_and_args(first_call) == (
        "get_recap",
        {"after": "2026-10-08T21:14:29", "before": "2026-10-08T22:14:29"},
    )

    empty_request = ChatRequest(
        **followup(
            first_body,
            first_call,
            {"events": [], "message": "No activity was found during this time period."},
        )
    )
    _prepared, empty_answer = prepare(Settings(), empty_request)
    assert isinstance(empty_answer, str)

    second_body = followup(
        first_body,
        first_call,
        {"events": [], "message": "No activity was found during this time period."},
    )
    second_body["messages"] += [
        {"role": "assistant", "content": empty_answer},
        {"role": "user", "content": "und die letzten 5 Stunden?"},
    ]
    second_request = ChatRequest(**second_body)
    _prepared, second_call = prepare(Settings(), second_request)
    assert name_and_args(second_call) == (
        "get_recap",
        {"after": "2026-10-08T17:14:29", "before": "2026-10-08T22:14:29"},
    )

    result_request = ChatRequest(**followup(second_body, second_call, {"events": recap_events()}))
    _prepared, direct = prepare(Settings(), result_request)
    assert isinstance(direct, str)
    assert direct.count("\n") == 13
    assert result_request._metrics["frigate_route"]["reason"] == "deterministic_recap_list"
    assert result_request._metrics["frigate_route"]["inference_calls"] == 0


def test_recap_summary_wording_is_not_forced_into_direct_listing():
    body = payload("Fasse die Ereignisse der letzten 5 Stunden zusammen")
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "recap1",
                "type": "function",
                "function": {
                    "name": "get_recap",
                    "arguments": json.dumps(
                        {"after": "2026-10-08T17:14:29", "before": "2026-10-08T22:14:29"}
                    ),
                },
            }
        ],
    }
    request = ChatRequest(
        **followup(
            body,
            call,
            {"events": [{"camera": "eingang", "time": "06:15 PM", "objects": ["person"]}]},
        )
    )
    assert shortcut_plan(request, Settings(), False) is None


def test_semantic_time_followup_is_not_misclassified_as_plain_recap_listing():
    body = payload("Zeige mir die Ereignisse der letzten Stunde")
    body["messages"] += [
        {"role": "assistant", "content": "Keine Aktivitäten."},
        {"role": "user", "content": "und was war auffällig in den letzten 5 Stunden?"},
    ]
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "recap-semantic",
                "type": "function",
                "function": {
                    "name": "get_recap",
                    "arguments": json.dumps(
                        {"after": "2026-10-08T17:14:29", "before": "2026-10-08T22:14:29"}
                    ),
                },
            }
        ],
    }
    request = ChatRequest(**followup(body, call, {"events": recap_events(2)}))
    assert shortcut_plan(request, Settings(), False) is None


@pytest.mark.parametrize(
    "question",
    [
        "Wann ist Leo nach Hause gekommen?",
        "Wann ist die Person Leo nach hause gekommen",
        "When did Leo get home?",
    ],
)
def test_named_home_question_uses_latest_named_sighting_without_llm(question):
    body = payload(question)
    request = ChatRequest(**body)
    _prepared, call = prepare(Settings(), request)
    assert name_and_args(call) == ("search_objects", {"sub_label": "Leo", "limit": 1})
    assert request._metrics["frigate_route"]["reason"] == "home_named_sighting"
    assert request._metrics["frigate_route"]["inference_calls"] == 0

    event = {
        "start_time_local": "2026-10-08 06:15:02 PM",
        "end_time_local": "2026-10-08 06:15:37 PM",
        "sub_label": "Leo",
        "label": "person",
        "zones": ["alles"],
        "camera": "eingang",
        "id": "event-leo",
    }
    result_request = ChatRequest(**followup(body, call, [event]))
    _prepared, direct = prepare(Settings(), result_request)

    assert isinstance(direct, str)
    assert "Leo" in direct and "Eingang" in direct
    assert "2026-10-08 06:15:02 PM" in direct
    assert result_request._metrics["frigate_route"]["reason"] == "home_named_sighting_summary"
    assert result_request._metrics["frigate_route"]["inference_calls"] == 0
