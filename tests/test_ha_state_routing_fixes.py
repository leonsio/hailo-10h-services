import json

from hailo_services.assistants.ha.ha_state_routing import (
    deterministic_live_response,
    direct_live_context_response,
)
from hailo_services.config import LLM_MODEL
from hailo_services.schemas import ChatRequest

SYSTEM = """Du bist Sprach Assistent für Home Assistant.

Static Context: An overview of the areas and the devices in this smart home:
- names: Licht
  domain: light
  areas: Bad Obergeschoss
- names: Licht
  domain: light
  areas: Flur Obergeschoss
- names: Licht
  domain: light
  areas: Flur Erdgeschoss
- names: Licht
  domain: light
  areas: Maxim
- names: Licht
  domain: light
  areas: Elisa
- names: Fenster - Twinkly
  domain: light
  areas: Wohnzimmer
- names: Licht Dimmer
  domain: light
  areas: Wohnzimmer
- names: Licht Tisch
  domain: light
  areas: Wohnzimmer
- names: Backofen Licht
  domain: light
  areas: Küche
- names: Licht - Links
  domain: light
  areas: Küche
- names: Licht - Rechts
  domain: light
  areas: Küche
- names: Oberlicht
  domain: light
  areas: Küche
- names: Dimmer Oberlicht
  domain: light
  areas: Dachgeschoss
- names: Nachtlicht Julia
  domain: light
  areas: Dachgeschoss
- names: Nachtlicht Leon
  domain: light
  areas: Dachgeschoss
- names: DG Schlafzimmer - Fenster links Fensterzustand
  domain: binary_sensor
  areas: Dachgeschoss
- names: DG Schlafzimmer - Fenster rechts Fensterzustand
  domain: binary_sensor
  areas: Dachgeschoss
- names: KG Gästezimmer Licht KG Gästezimmer - Licht
  domain: light
  areas: Gästezimmer

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


def request(text, messages=None):
    return ChatRequest(
        model=LLM_MODEL,
        messages=messages
        or [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        tools=[LIVE],
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


def arguments_for(text):
    result = direct_live_context_response(request(text))
    assert result is not None
    call = result["tool_calls"][0]
    assert call["function"]["name"] == "homeassistant__GetLiveContext"
    return json.loads(call["function"]["arguments"])


def test_living_room_area_beats_generic_light_names_elsewhere():
    assert arguments_for("Ist das Licht im Wohnzimmer an?") == {
        "area": "Wohnzimmer",
        "domain": ["light"],
    }


def test_kitchen_area_beats_generic_light_names_elsewhere():
    assert arguments_for("Ist das Licht in der Küche an?") == {
        "area": "Küche",
        "domain": ["light"],
    }


def test_attic_area_beats_generic_light_names_elsewhere():
    assert arguments_for("Ist das Licht im Dachgeschoss an?") == {
        "area": "Dachgeschoss",
        "domain": ["light"],
    }


def test_specific_device_inside_area_stays_specific():
    assert arguments_for("Ist Licht Tisch im Wohnzimmer an?") == {
        "name": "Licht Tisch",
        "domain": ["light"],
    }


def test_unknown_room_is_queried_directly_instead_of_gemma_guessing():
    assert arguments_for("Ist das Licht im Schlafzimmer an?") == {
        "area": "Schlafzimmer",
        "domain": ["light"],
    }


def test_whole_house_boolean_query_uses_domain_only():
    assert arguments_for("Ist das Licht im Haus an?") == {
        "domain": ["light"],
    }


def test_no_exposed_entity_error_is_not_treated_as_entity_state():
    routed = live_followup(
        "Ist das Licht im Schlafzimmer an?",
        {"area": "Schlafzimmer", "domain": ["light"]},
        {
            "success": False,
            "error": "no exposed entities matched area 'Schlafzimmer'",
        },
    )
    assert deterministic_live_response(routed) == (
        "Ich konnte im Bereich „Schlafzimmer“ keine passenden Lichter finden."
    )


def test_error_key_is_not_rendered_as_entity_named_error():
    routed = live_followup(
        "Ist das Licht im Schlafzimmer an?",
        {"area": "Schlafzimmer", "domain": ["light"]},
        {
            "error": "no exposed entities matched name 'KG Gästezimmer Licht'",
        },
    )
    response = deterministic_live_response(routed)
    assert response == "Ich konnte im Bereich „Schlafzimmer“ keine passenden Lichter finden."
    assert "error ist" not in response
