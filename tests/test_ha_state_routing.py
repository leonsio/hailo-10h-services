import json

from hailo_services.assistants.ha.ha_state_routing import (
    compact_live_followup_request,
    deterministic_live_response,
    direct_live_context_response,
)
from hailo_services.config import LLM_MODEL
from hailo_services.schemas import ChatRequest

SYSTEM = """Du bist Sprach Assistent für Home Assistant.

Static Context: An overview of the areas and the devices in this smart home:
- names: Licht Dimmer
  domain: light
  areas: Wohnzimmer
- names: Licht Tisch
  domain: light
  areas: Wohnzimmer
- names: Fenster - Twinkly
  domain: light
  areas: Wohnzimmer
- names: Temperatur
  domain: climate
  areas: Wohnzimmer
- names: Oberlicht
  domain: light
  areas: Küche

When controlling Home Assistant always call the intent tools.
"""

LIVE = {
    "type": "function",
    "function": {
        "name": "homeassistant__GetLiveContext",
        "description": "Provides real-time information about current state.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "domain": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "area": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
}
TURN_ON = {
    "type": "function",
    "function": {
        "name": "intent__HassTurnOn",
        "description": "Turns on a device.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def request(text, messages=None):
    return ChatRequest(
        model=LLM_MODEL,
        messages=messages
        or [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        tools=[LIVE, TURN_ON],
    )


def live_followup(text, arguments, payload):
    call_id = "call_live"
    return request(
        text,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "homeassistant__GetLiveContext",
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
    )


def test_living_room_light_state_uses_live_context_area_directly():
    result = direct_live_context_response(request("Ist das Licht im Wohnzimmer an?"))
    assert result is not None
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "homeassistant__GetLiveContext"
    assert json.loads(call["function"]["arguments"]) == {
        "area": "Wohnzimmer",
        "domain": ["light"],
    }


def test_state_question_is_not_a_turn_on_action():
    result = direct_live_context_response(request("Ist das Licht im Wohnzimmer an?"))
    assert result["tool_calls"][0]["function"]["name"] != "intent__HassTurnOn"


def test_explicit_light_name_uses_name_and_domain():
    result = direct_live_context_response(request("Ist Licht Tisch an?"))
    assert result is not None
    arguments = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"name": "Licht Tisch", "domain": ["light"]}


def test_control_command_is_not_treated_as_state_question():
    assert direct_live_context_response(request("Schalte das Licht im Wohnzimmer an")) is None


def test_whole_home_aggregate_light_state_uses_domain_only():
    result = direct_live_context_response(request("Sind alle Lichter aus?"))
    assert result is not None
    arguments = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"domain": ["light"]}


def test_temperature_query_selects_unique_measurement_entity():
    result = direct_live_context_response(request("Wie warm ist es im Wohnzimmer?"))
    assert result is not None
    arguments = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"name": "Temperatur", "domain": ["climate"]}


def test_mixed_area_state_is_answered_without_llm():
    routed = live_followup(
        "Ist das Licht im Wohnzimmer an?",
        {"area": "Wohnzimmer", "domain": ["light"]},
        {"Licht Tisch": "on", "Licht Dimmer": "off"},
    )
    response = deterministic_live_response(routed)
    assert response is not None
    assert "Teilweise" in response
    assert "1 von 2" in response
    assert "Licht Tisch" in response
    assert "Licht Dimmer" in response


def test_explicit_device_boolean_state_is_answered_without_llm():
    routed = live_followup(
        "Ist Licht Tisch aus?",
        {"name": "Licht Tisch", "domain": ["light"]},
        {"Licht Tisch": "off"},
    )
    assert deterministic_live_response(routed) == "Ja, Licht Tisch ist aus."


def test_list_active_lights_is_answered_without_llm():
    routed = live_followup(
        "Welche Lichter im Wohnzimmer sind an?",
        {"area": "Wohnzimmer", "domain": ["light"]},
        {"Licht Tisch": "on", "Licht Dimmer": "off", "Fenster - Twinkly": "on"},
    )
    response = deterministic_live_response(routed)
    assert response == "An: Licht Tisch, Fenster - Twinkly."


def test_count_active_lights_is_answered_without_llm():
    routed = live_followup(
        "Wie viele Lichter im Wohnzimmer sind an?",
        {"area": "Wohnzimmer", "domain": ["light"]},
        {"Licht Tisch": "on", "Licht Dimmer": "off", "Fenster - Twinkly": "on"},
    )
    assert deterministic_live_response(routed) == "2 von 3 Lichter sind an."


def test_generic_device_status_is_answered_without_llm():
    routed = live_followup(
        "Wie ist der Status von Licht Tisch?",
        {"name": "Licht Tisch", "domain": ["light"]},
        {"Licht Tisch": "on"},
    )
    assert deterministic_live_response(routed) == "Licht Tisch ist an."


def test_temperature_value_is_answered_without_llm():
    live_text = """Live Context: An overview of the areas and devices:
- names: Temperatur
  domain: climate
  state: 'heat'
  areas: Wohnzimmer
  current_temperature: 21.5
  unit_of_measurement: °C
"""
    routed = live_followup(
        "Wie warm ist es im Wohnzimmer?",
        {"name": "Temperatur", "domain": ["climate"]},
        {"success": True, "result": live_text},
    )
    assert deterministic_live_response(routed) == "Die Temperatur beträgt 21,5 °C."


def test_live_followup_is_reduced_to_tiny_text_only_prompt_as_fallback():
    call_id = "call_live"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Ist das Licht im Wohnzimmer an?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "homeassistant__GetLiveContext",
                        "arguments": json.dumps({"area": "Wohnzimmer", "domain": ["light"]}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": '{"Licht Tisch":"on","Licht Dimmer":"off"}',
        },
    ]
    routed = compact_live_followup_request(request("unused", messages=messages))
    assert routed is not None
    assert routed.tools is None
    assert routed.tool_choice is None
    assert len(routed.messages) == 2
    assert "Static Context" not in routed.messages[0]["content"]
    assert "Ist das Licht im Wohnzimmer an?" in routed.messages[1]["content"]
    assert '"Licht Tisch":"on"' in routed.messages[1]["content"]
