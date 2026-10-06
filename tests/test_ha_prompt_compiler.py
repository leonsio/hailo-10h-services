import json

import numpy as np

from hailo_services.config import LLM_MODEL
from hailo_services.ha_prompt_compiler import compile_ha_prompt
from hailo_services.schemas import ChatRequest

SYSTEM = """Du bist Sprach Assistent für Home Assistant.
Antworte einfach, kurz und wahrheitsgetreu.
You ARE equipped to answer questions about the current state of the home.
If the user asks about the CURRENT state, value, or mode, call GetLiveContext.
For general knowledge questions not about the home: answer truthfully.

Static Context: An overview of the areas and the devices in this smart home:
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
- names: Backofen Temperatur
  domain: sensor
  areas: Küche
- names: Wohnzimmer Thermostat
  domain: climate
  areas: Wohnzimmer
- names: Wohnzimmer Temperatur
  domain: sensor
  areas: Wohnzimmer
- names: Rollladen
  domain: cover
  areas: Wohnzimmer

When controlling Home Assistant always call the intent tools.
Use intent__HassTurnOn to lock and intent__HassTurnOff to unlock a lock.
When controlling a device, prefer passing just name and domain.
When controlling an area, prefer passing just area name and domain.
When a user asks to turn on all devices of a specific type, ask for an area.
This device is not able to start timers.
"""

LIGHT = {
    "type": "function",
    "function": {
        "name": "light__HassLightSet",
        "description": "Sets the brightness percentage or color of a light",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name"},
                "area": {"type": "string", "description": "Area name"},
                "floor": {"type": "string", "description": "Floor"},
                "domain": {"type": "array", "items": {"type": "string", "enum": ["light"]}},
                "color": {"type": "string"},
                "temperature": {"type": "integer", "minimum": 0},
                "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "additionalProperties": False,
        },
    },
}

CLIMATE = {
    "type": "function",
    "function": {
        "name": "climate__HassClimateSetTemperature",
        "description": "Sets the target temperature of a climate device or entity",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "area": {"type": "string"},
                "name": {"type": "string"},
                "floor": {"type": "string"},
            },
            "required": ["temperature"],
            "additionalProperties": False,
        },
    },
}

LIVE = {
    "type": "function",
    "function": {
        "name": "homeassistant__GetLiveContext",
        "description": "Provides real-time information about current state and values.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "domain": {"type": "array", "items": {"type": "string"}},
                "area": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
}


def request(text, tools, messages=None):
    return ChatRequest(
        model=LLM_MODEL,
        messages=messages
        or [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        tools=tools,
    )


def system_text(compiled):
    return next(message["content"] for message in compiled.messages if message["role"] == "system")


def tool_properties(compiled):
    return compiled.tools[0]["function"]["parameters"]["properties"]


def test_brightness_compiles_to_light_only_minimal_prompt_and_schema():
    source = request("Stelle das Licht in der Küche auf 40 Prozent", [LIGHT, CLIMATE, LIVE])
    prepared = request("Stelle das Licht in der Küche auf 40 Prozent", [LIGHT, CLIMATE])

    compiled, plan = compile_ha_prompt(source, prepared)

    assert plan is not None
    assert plan["capability"] == "light.brightness"
    assert plan["domain"] == "light"
    assert plan["area"] == "Küche"
    assert [tool["function"]["name"] for tool in compiled.tools] == ["light__HassLightSet"]
    assert set(tool_properties(compiled)) == {"name", "area", "domain", "brightness"}
    assert tool_properties(compiled)["domain"]["items"]["enum"] == ["light"]
    assert tool_properties(compiled)["area"]["enum"] == ["Küche"]
    assert set(tool_properties(compiled)["name"]["enum"]) == {
        "Backofen Licht",
        "Licht - Links",
        "Licht - Rechts",
        "Oberlicht",
    }
    prompt = system_text(compiled)
    assert "Backofen Temperatur" not in prompt
    assert "Wohnzimmer Thermostat" not in prompt
    assert "Rollladen" not in prompt
    assert "Backofen Licht" in prompt
    assert len(prompt) < 500
    assert plan["system_characters_after"] < plan["system_characters_before"]
    assert plan["tool_schema_after"]["properties"] < plan["tool_schema_before"]["properties"]


def test_climate_query_removes_light_entities_and_irrelevant_properties():
    source = request("Stelle den Thermostat im Wohnzimmer auf 22 Grad", [LIGHT, CLIMATE])
    prepared = request("Stelle den Thermostat im Wohnzimmer auf 22 Grad", [CLIMATE, LIGHT])

    compiled, plan = compile_ha_prompt(source, prepared)

    assert plan["capability"] == "climate.temperature"
    assert plan["domain"] == "climate"
    assert plan["area"] == "Wohnzimmer"
    assert [tool["function"]["name"] for tool in compiled.tools] == [
        "climate__HassClimateSetTemperature"
    ]
    assert set(tool_properties(compiled)) == {"temperature", "area", "name"}
    prompt = system_text(compiled)
    assert "Wohnzimmer Thermostat" in prompt
    assert "Backofen Licht" not in prompt
    assert "Wohnzimmer Temperatur" not in prompt


def test_selected_single_tool_can_define_capability_without_semantic_model():
    source = request("Mach die Beleuchtung in der Küche gemütlicher", [LIGHT, CLIMATE])
    prepared = request("Mach die Beleuchtung in der Küche gemütlicher", [LIGHT])

    compiled, plan = compile_ha_prompt(source, prepared)

    assert plan["capability"] == "light.adjust"
    assert plan["capability_source"] == "selected_tool"
    assert plan["domain"] == "light"
    assert plan["area"] == "Küche"
    assert "Wohnzimmer Thermostat" not in system_text(compiled)


class FakeEncoder:
    def embed(self, text):
        text = str(text).casefold()
        if "gemüt" in text:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "warm white" in text or "cool white" in text:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "brightness" in text:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def test_minilm_capability_fallback_is_used_when_selected_tools_are_ambiguous():
    source = request("Mach es in der Küche gemütlicher", [LIGHT, CLIMATE])
    prepared = request("Mach es in der Küche gemütlicher", [LIGHT, CLIMATE])

    compiled, plan = compile_ha_prompt(
        source,
        prepared,
        encoder=FakeEncoder(),
        embedding_cache={},
    )

    assert plan["capability"] == "light.temperature"
    assert plan["capability_source"] == "minilm"
    assert [tool["function"]["name"] for tool in compiled.tools] == ["light__HassLightSet"]
    assert set(tool_properties(compiled)) == {"name", "area", "domain", "temperature"}


def test_direct_response_requests_are_not_recompiled():
    source = request("Ist das Licht in der Küche an?", [LIVE])
    prepared = request("Ist das Licht in der Küche an?", [LIVE])
    object.__setattr__(prepared, "_direct_ha_response", {"role": "assistant"})

    compiled, plan = compile_ha_prompt(source, prepared)

    assert compiled is prepared
    assert plan is None


def test_active_tool_history_is_compacted_with_dependencies():
    call_id = "call_live"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Ist das Licht in der Küche an?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "homeassistant__GetLiveContext",
                        "arguments": json.dumps({"area": "Küche", "domain": ["light"]}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": '{"success":true}'},
    ]
    source = request("unused", [LIVE], messages=messages)
    prepared = request("unused", [LIVE], messages=messages)

    compiled, plan = compile_ha_prompt(source, prepared)

    assert compiled.messages[1:] == prepared.messages[1:]
    assert plan is not None
    assert len(system_text(compiled)) < len(SYSTEM)
    assert compiled.tools[0]["function"]["name"] == "homeassistant__GetLiveContext"
