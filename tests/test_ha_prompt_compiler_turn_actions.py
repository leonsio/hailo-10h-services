from hailo_services.assistants.ha.ha_prompt_compiler import compile_ha_prompt
from hailo_services.config import LLM_MODEL
from hailo_services.schemas import ChatRequest

SYSTEM = """Du bist Sprach Assistent für Home Assistant.

Static Context: An overview of the areas and the devices in this smart home:
- names: Deckenlicht
  domain: light
  areas: Küche
- names: Kaffeemaschine
  domain: switch
  areas: Küche
- names: Saugroboter
  domain: vacuum
  areas: Küche

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


def request(text):
    return ChatRequest(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        tools=[TURN_ON],
    )


class MustNotBeUsedEncoder:
    def embed(self, text):
        raise AssertionError("MiniLM capability inference must not run for a selected turn-on tool")


def test_turn_on_tool_defines_capability_before_semantic_inference():
    source = request("Schalte das Licht in der Küche an")
    prepared = request("Schalte das Licht in der Küche an")

    compiled, plan = compile_ha_prompt(
        source,
        prepared,
        encoder=MustNotBeUsedEncoder(),
        embedding_cache={},
    )

    assert plan["capability"] == "device.turn_on"
    assert plan["capability_source"] == "selected_tool"
    assert plan["semantic"] is None
    assert plan["domain"] == "light"
    assert plan["area"] == "Küche"
    assert plan["selected_tool_names"] == ["intent__HassTurnOn"]
    assert "vacuum.control" not in plan["system_prompt"]
    assert compiled.tools[0]["function"]["name"] == "intent__HassTurnOn"


def test_turn_on_tool_is_not_reclassified_as_vacuum_control():
    source = request("Schalte den Staubsauger in der Küche an")
    prepared = request("Schalte den Staubsauger in der Küche an")

    _, plan = compile_ha_prompt(
        source,
        prepared,
        encoder=MustNotBeUsedEncoder(),
        embedding_cache={},
    )

    assert plan["capability"] == "device.turn_on"
    assert plan["capability_source"] == "selected_tool"
    assert plan["domain"] == "vacuum"
    assert plan["area"] == "Küche"
