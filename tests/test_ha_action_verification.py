import json

import hailo_services.ha_action_verification as action_verification
from hailo_services.config import LLM_MODEL
from hailo_services.schemas import ChatRequest


LIVE_TOOL = {
    "type": "function",
    "function": {
        "name": "homeassistant__GetLiveContext",
        "description": "Get current state",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "area": {"type": "string"},
                "domain": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
            },
            "additionalProperties": False,
        },
    },
}
TURN_OFF_TOOL = {
    "type": "function",
    "function": {
        "name": "intent__HassTurnOff",
        "description": "Turn off",
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


def call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def action_done(call_id):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps({
            "speech": {},
            "response_type": "action_done",
            "data": {"success": [{"name": "Küche", "type": "area", "id": "kuche"}], "failed": []},
        }),
    }


def live_result(call_id, states):
    lines = ["Live Context: An overview of the areas and the devices in this smart home:"]
    for name, state in states:
        lines.extend([
            f"- names: {name}",
            "  domain: light",
            f"  state: '{state}'",
            "  areas: Küche",
        ])
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps({"success": True, "result": "\n".join(lines) + "\n"}),
    }


def request(messages):
    return ChatRequest(
        model=LLM_MODEL,
        messages=messages,
        tools=[TURN_OFF_TOOL, LIVE_TOOL],
        parallel_tool_calls=True,
    )


def base_action_messages():
    action = call(
        "call_action",
        "intent__HassTurnOff",
        {"area": "Küche", "domain": ["light"]},
    )
    return [
        {"role": "user", "content": "Schalte das Licht in der Küche aus"},
        {"role": "assistant", "content": None, "tool_calls": [action]},
        action_done("call_action"),
    ]


def test_successful_action_requests_live_verification_before_acknowledgement():
    decision = action_verification.action_verification_response(request(base_action_messages()))
    assert decision is not None
    assert decision["kind"] == "verify"
    verify = decision["response"]["tool_calls"][0]
    assert verify["function"]["name"] == "homeassistant__GetLiveContext"
    assert json.loads(verify["function"]["arguments"]) == {
        "area": "Küche",
        "domain": ["light"],
    }


def test_verified_action_finishes_without_gemma():
    messages = base_action_messages()
    verify = call(
        "call_verify",
        "homeassistant__GetLiveContext",
        {"area": "Küche", "domain": ["light"]},
    )
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [verify]},
        live_result("call_verify", [
            ("Backofen Licht", "off"),
            ("Licht - Links", "off"),
            ("Licht - Rechts", "off"),
            ("Oberlicht", "off"),
        ]),
    ])
    decision = action_verification.action_verification_response(request(messages))
    assert decision["kind"] == "verified"
    assert decision["response"] == "Erledigt."


def test_mismatching_light_is_retried_once_with_area_disambiguation():
    messages = base_action_messages()
    verify = call(
        "call_verify",
        "homeassistant__GetLiveContext",
        {"area": "Küche", "domain": ["light"]},
    )
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [verify]},
        live_result("call_verify", [
            ("Backofen Licht", "off"),
            ("Licht - Links", "off"),
            ("Licht - Rechts", "on"),
            ("Oberlicht", "off"),
        ]),
    ])
    decision = action_verification.action_verification_response(request(messages))
    assert decision["kind"] == "retry"
    calls = decision["response"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "intent__HassTurnOff"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "name": "Licht - Rechts",
        "domain": ["light"],
        "area": "Küche",
    }


def test_second_failed_verification_stops_retry_loop():
    messages = base_action_messages()
    verify1 = call(
        "call_verify1",
        "homeassistant__GetLiveContext",
        {"area": "Küche", "domain": ["light"]},
    )
    retry = call(
        "call_retry",
        "intent__HassTurnOff",
        {"name": "Licht - Rechts", "area": "Küche", "domain": ["light"]},
    )
    verify2 = call(
        "call_verify2",
        "homeassistant__GetLiveContext",
        {"area": "Küche", "domain": ["light"]},
    )
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [verify1]},
        live_result("call_verify1", [("Licht - Rechts", "on")]),
        {"role": "assistant", "content": None, "tool_calls": [retry]},
        action_done("call_retry"),
        {"role": "assistant", "content": None, "tool_calls": [verify2]},
        live_result("call_verify2", [("Licht - Rechts", "on")]),
    ])
    decision = action_verification.action_verification_response(request(messages))
    assert decision["kind"] == "failed_verification"
    assert "Licht - Rechts" in decision["response"]
