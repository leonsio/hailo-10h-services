import numpy as np
import pytest

from hailo_services import ha_prompt_compiler as compiler
from hailo_services import ha_routing as routing
from hailo_services.config import LLM_MODEL
from hailo_services.schemas import ChatRequest

SYSTEM = """Du bist Sprach Assistent für Home Assistant.

Static Context: An overview of the areas and the devices in this smart home:
- names: Licht Tisch
  domain: light
  areas: Wohnzimmer

When controlling Home Assistant always call the intent tools.
"""

LIVE = {
    "type": "function",
    "function": {
        "name": "homeassistant__GetLiveContext",
        "description": "Read current Home Assistant state.",
        "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
    },
}
LIGHT = {
    "type": "function",
    "function": {
        "name": "light__HassLightSet",
        "description": "Set light properties.",
        "parameters": {
            "type": "object",
            "properties": {
                "area": {"type": "string"},
                "domain": {"type": "array", "items": {"type": "string", "enum": ["light"]}},
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
        "description": "Set target temperature.",
        "parameters": {
            "type": "object",
            "properties": {"temperature": {"type": "number"}, "area": {"type": "string"}},
            "required": ["temperature"],
            "additionalProperties": False,
        },
    },
}


class ToolScoreEncoder:
    def __init__(self, query, scores):
        self.query = query
        self.scores = scores

    def embed(self, text):
        if text == self.query:
            return np.array([1.0], dtype=np.float32)
        for marker, score in self.scores.items():
            if marker in text:
                return np.array([score], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)


def request(text, tools):
    return ChatRequest(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        tools=tools,
    )


def test_semantic_nearest_tool_without_clear_margin_does_not_route_general_question_to_ha():
    query = "was ist die Hauptstadt von Bolivien"
    encoder = ToolScoreEncoder(query, {"GetLiveContext": 0.561, "HassLightSet": 0.522})
    req = request(query, [LIVE, LIGHT])
    result = routing.assess_ha_relevance(req.messages, req.tools, encoder=encoder, embedding_cache={})
    assert result["relevant"] is False
    assert result["reason"] == "semantic_ambiguous"
    assert result["semantic_tool_name"] == "homeassistant__GetLiveContext"
    assert result["semantic_tool_margin"] < result["semantic_margin_threshold"]


def test_distinct_semantic_tool_match_is_still_allowed():
    query = "prüfe bitte was dort gerade los ist"
    encoder = ToolScoreEncoder(query, {"GetLiveContext": 0.71, "HassLightSet": 0.49})
    req = request(query, [LIVE, LIGHT])
    result = routing.assess_ha_relevance(req.messages, req.tools, encoder=encoder, embedding_cache={})
    assert result["relevant"] is True
    assert result["reason"] == "semantic_tool"
    assert result["semantic_tool_margin"] >= result["semantic_margin_threshold"]


def test_semantic_capability_is_limited_to_capabilities_of_selected_tools():
    query = "mehrdeutiger test"
    scores = {
        compiler._CAPABILITIES["vacuum.control"]: 0.95,
        compiler._CAPABILITIES["light.adjust"]: 0.72,
        compiler._CAPABILITIES["climate.temperature"]: 0.60,
    }
    encoder = ToolScoreEncoder(query, scores)
    capability, source, semantic = compiler._capability(query, [LIGHT, CLIMATE], encoder, {})
    assert capability == "light.adjust"
    assert source == "minilm"
    assert semantic["best"] == pytest.approx(0.72, abs=1e-5)


def test_compiler_never_rewrites_tool_domain_to_incompatible_enum():
    compact = compiler._compact_tool(LIGHT, "vacuum.control", "vacuum", None, [])
    enum = compact["function"]["parameters"]["properties"]["domain"]["items"]["enum"]
    assert enum == ["light"]
