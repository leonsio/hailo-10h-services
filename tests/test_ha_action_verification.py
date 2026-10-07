import json
from types import SimpleNamespace

from hailo_services.config import LLM_MODEL, Settings
from hailo_services.ha_action_verification import action_verification_response
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
                "domain": {
                    "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]
                },
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


def action_done(call_id, *, failed=None):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(
            {
                "speech": {},
                "response_type": "action_done",
                "data": {
                    "success": [{"name": "Küche", "type": "area", "id": "kuche"}],
                    "failed": failed or [],
                },
            }
        ),
    }


def live_result(call_id, states):
    lines = ["Live Context: An overview of the areas and the devices in this smart home:"]
    for name, state in states:
        lines.extend(
            [
                f"- names: {name}",
                "  domain: light",
                f"  state: '{state}'",
                "  areas: Küche",
            ]
        )
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


def settings(attempts=2):
    return SimpleNamespace(ha_assist_verify_attempts=attempts)


def base_action_messages(*, failed=None):
    action = call(
        "call_action",
        "intent__HassTurnOff",
        {"area": "Küche", "domain": ["light"]},
    )
    return [
        {"role": "user", "content": "Schalte das Licht in der Küche aus"},
        {"role": "assistant", "content": None, "tool_calls": [action]},
        action_done("call_action", failed=failed),
    ]


def test_successful_action_requests_live_verification_before_acknowledgement():
    decision = action_verification_response(request(base_action_messages()), settings())
    assert decision is not None
    assert decision["kind"] == "verify"
    assert decision["verify_attempt"] == 1
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
    messages.extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [verify]},
            live_result(
                "call_verify",
                [
                    ("Backofen Licht", "off"),
                    ("Licht - Links", "off"),
                    ("Licht - Rechts", "off"),
                    ("Oberlicht", "off"),
                ],
            ),
        ]
    )
    decision = action_verification_response(request(messages), settings())
    assert decision["kind"] == "verified"
    assert decision["response"] == "Erledigt."


def test_mismatching_light_retries_only_live_verification():
    messages = base_action_messages()
    verify = call(
        "call_verify",
        "homeassistant__GetLiveContext",
        {"area": "Küche", "domain": ["light"]},
    )
    messages.extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [verify]},
            live_result(
                "call_verify",
                [
                    ("Backofen Licht", "off"),
                    ("Licht - Links", "off"),
                    ("Licht - Rechts", "on"),
                    ("Oberlicht", "off"),
                ],
            ),
        ]
    )
    decision = action_verification_response(request(messages), settings())
    assert decision["kind"] == "verify_retry"
    assert decision["verify_attempt"] == 2
    calls = decision["response"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "homeassistant__GetLiveContext"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "area": "Küche",
        "domain": ["light"],
    }


def test_second_failed_verification_reports_unconfirmed_state_without_reissuing_action():
    messages = base_action_messages()
    verify1 = call(
        "call_verify1",
        "homeassistant__GetLiveContext",
        {"area": "Küche", "domain": ["light"]},
    )
    verify2 = call(
        "call_verify2",
        "homeassistant__GetLiveContext",
        {"area": "Küche", "domain": ["light"]},
    )
    messages.extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [verify1]},
            live_result("call_verify1", [("Licht - Rechts", "on")]),
            {"role": "assistant", "content": None, "tool_calls": [verify2]},
            live_result("call_verify2", [("Licht - Rechts", "on")]),
        ]
    )
    decision = action_verification_response(request(messages), settings())
    assert decision["kind"] == "state_unconfirmed"
    assert decision["verify_attempt"] == 2
    assert "Licht - Rechts" in decision["response"]
    assert (
        len(
            [
                call
                for message in messages
                if message.get("role") == "assistant"
                for call in message.get("tool_calls") or []
                if call["function"]["name"] == "intent__HassTurnOff"
            ]
        )
        == 1
    )


def test_configured_third_verify_is_used_before_giving_up():
    messages = base_action_messages()
    for index in (1, 2):
        verify = call(
            f"call_verify{index}",
            "homeassistant__GetLiveContext",
            {"area": "Küche", "domain": ["light"]},
        )
        messages.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": [verify]},
                live_result(f"call_verify{index}", [("Licht - Rechts", "on")]),
            ]
        )
    decision = action_verification_response(request(messages), settings(3))
    assert decision["kind"] == "verify_retry"
    assert decision["verify_attempt"] == 3
    assert (
        decision["response"]["tool_calls"][0]["function"]["name"] == "homeassistant__GetLiveContext"
    )


def test_verify_can_be_disabled():
    decision = action_verification_response(request(base_action_messages()), settings(0))
    assert decision is None


def test_failed_target_is_reported_as_action_failure_without_verification():
    messages = base_action_messages(
        failed=[
            {
                "name": "Licht - Rechts",
                "type": "entity",
                "id": "light.licht_rechts",
            }
        ]
    )
    decision = action_verification_response(request(messages), settings())
    assert decision["kind"] == "action_failed"
    assert (
        "Home Assistant konnte die angeforderte Aktion nicht erfolgreich ausführen"
        in decision["response"]
    )
    assert "Licht - Rechts" in decision["response"]
    assert (
        "tool_calls" not in decision["response"] if isinstance(decision["response"], dict) else True
    )


def test_verify_settings_have_safe_defaults_and_bounds():
    configured = Settings()
    assert configured.ha_assist_verify_attempts == 2
    assert configured.ha_assist_verify_delay == 0.5

    try:
        Settings(ha_assist_verify_attempts=11)
    except ValueError as error:
        assert "verify attempts" in str(error)
    else:
        raise AssertionError("Expected invalid verify attempt count to fail")

    try:
        Settings(ha_assist_verify_delay=31.0)
    except ValueError as error:
        assert "verify delay" in str(error)
    else:
        raise AssertionError("Expected invalid verify delay to fail")
