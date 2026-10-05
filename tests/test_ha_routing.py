import json

from hailo_services.config import LLM_MODEL
from hailo_services.ha_routing import (
    assess_ha_relevance,
    direct_action_response,
    general_passthrough_request,
)
from hailo_services.schemas import ChatRequest

SYSTEM = """Du bist Sprach Assistent für Home Assistant.
Antworte kurz.

Static Context: An overview of the areas and the devices in this smart home:
- names: Backofen Licht
  domain: light
  areas: Küche
- names: Licht - Rechts
  domain: light
  areas: Küche
- names: Licht - Links
  domain: light
  areas: Küche
- names: Oberlicht
  domain: light
  areas: Küche
- names: Kaffeemaschine
  domain: switch
  areas: Küche
- names: Oberlicht
  domain: light
  areas: Ankleide

When controlling Home Assistant always call the intent tools.
"""

TURN_ON = {
    "type": "function",
    "function": {
        "name": "intent__HassTurnOn",
        "description": "Turns on/opens a device or entity.",
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

TURN_OFF = {
    "type": "function",
    "function": {
        "name": "intent__HassTurnOff",
        "description": "Turns off/closes a device or entity.",
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


def request(text, *, tools=None, system=SYSTEM):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    return ChatRequest(model=LLM_MODEL, messages=messages, tools=tools)


def test_general_knowledge_has_no_ha_lexical_signal():
    req = request("was ist die Hauptstadt von Frankreich", tools=[TURN_OFF])
    relevance = assess_ha_relevance(req.messages, req.tools, encoder=None)
    assert relevance["relevant"] is False
    assert relevance["entity_lexical_max"] == 0
    assert relevance["tool_lexical_max"] == 0


def test_general_passthrough_drops_generated_ha_prompt_and_tools():
    req = request("was ist die Hauptstadt von Frankreich", tools=[TURN_OFF])
    routed = general_passthrough_request(req)
    assert routed.messages == [{"role": "user", "content": "was ist die Hauptstadt von Frankreich"}]
    assert routed.tools is None
    assert routed.tool_choice is None


def test_kitchen_light_command_is_ha_relevant():
    req = request("Schalte das Licht in der Küche aus", tools=[TURN_OFF])
    relevance = assess_ha_relevance(req.messages, req.tools, encoder=None)
    assert relevance["relevant"] is True
    assert relevance["entity_lexical_max"] > 0
    assert relevance["tool_lexical_max"] > 0


def test_direct_area_turn_off_selects_light_from_mixed_domain_area():
    req = request("Schalte das Licht in der Küche aus", tools=[TURN_OFF])
    result = direct_action_response(req)
    assert result is not None
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "intent__HassTurnOff"
    assert json.loads(call["function"]["arguments"]) == {
        "area": "Küche",
        "domain": ["light"],
    }


def test_direct_area_turn_on_matches_reported_home_assistant_request():
    req = request("schalte das Licht in der Küche an", tools=[TURN_ON])
    result = direct_action_response(req)
    assert result is not None
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "intent__HassTurnOn"
    assert json.loads(call["function"]["arguments"]) == {
        "area": "Küche",
        "domain": ["light"],
    }


def test_direct_area_switch_command_selects_switch_not_light():
    req = request("Schalte den Schalter in der Küche aus", tools=[TURN_OFF])
    result = direct_action_response(req)
    assert result is not None
    call = result["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {
        "area": "Küche",
        "domain": ["switch"],
    }


def test_direct_area_resolution_can_use_full_source_catalogue():
    source = request("Schalte das Licht in der Küche aus", tools=[TURN_OFF])
    reduced_system = """Du bist Sprach Assistent für Home Assistant.

Static Context: Relevant entities for the current user request:
- names: Kaffeemaschine
  domain: switch
  areas: Küche

When controlling Home Assistant always call the intent tools.
"""
    prepared = request(
        "Schalte das Licht in der Küche aus",
        tools=[TURN_OFF],
        system=reduced_system,
    )
    result = direct_action_response(prepared, source_messages=source.messages)
    assert result is not None
    call = result["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {
        "area": "Küche",
        "domain": ["light"],
    }


def test_mixed_domain_area_without_explicit_domain_does_not_bypass_llm():
    req = request("Schalte alle Geräte in der Küche aus", tools=[TURN_OFF])
    trace = {}
    assert direct_action_response(req, trace=trace) is None
    assert trace["direct_action_reason"] == "area_domain_ambiguous"
    assert trace["candidate_domains"] == ["light", "switch"]


def test_direct_explicit_device_uses_name_and_domain():
    req = request("Schalte Backofen Licht aus", tools=[TURN_OFF])
    result = direct_action_response(req)
    assert result is not None
    call = result["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {
        "name": "Backofen Licht",
        "domain": ["light"],
    }


def test_ambiguous_duplicate_name_does_not_bypass_llm():
    req = request("Schalte Oberlicht aus", tools=[TURN_OFF])
    assert direct_action_response(req) is None


def test_state_question_is_not_mistaken_for_direct_action():
    req = request("Ist das Licht in der Küche aus?", tools=[TURN_OFF])
    assert direct_action_response(req) is None
