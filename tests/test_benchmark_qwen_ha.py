import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark-qwen-ha.py"
spec = importlib.util.spec_from_file_location("benchmark_qwen_ha", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_qwen_payload_cannot_trigger_production_ha_envelope_shortcut():
    scenario = next(item for item in benchmark.SCENARIOS if item.id == "light_brightness")
    payload = benchmark.payload_for(scenario, 1200, benchmark.DEFAULT_MODEL, 128, 0.1, 42)
    assert payload["model"] == benchmark.DEFAULT_MODEL
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert "Static Context:" not in payload["messages"][0]["content"]
    assert {item["function"]["name"] for item in payload["tools"]} == {
        "light__HassLightSet", "homeassistant__GetLiveContext", "intent__HassTurnOn"
    }


def test_prompt_load_grows_monotonically_with_targets():
    scenario = next(item for item in benchmark.SCENARIOS if item.id == "light_brightness")
    sizes = []
    estimates = []
    for target in (850, 1100, 1400, 1700):
        payload = benchmark.payload_for(scenario, target, benchmark.DEFAULT_MODEL, 128, 0.1, 42)
        system = payload["messages"][0]["content"]
        sizes.append(len(system))
        estimates.append(benchmark.estimate_tokens(system, scenario.prompt, payload["tools"]))
    assert sizes == sorted(sizes)
    assert estimates == sorted(estimates)
    assert estimates[-1] > estimates[0]


def test_valid_expected_tool_call_passes_and_unknown_area_is_hallucination():
    scenario = next(item for item in benchmark.SCENARIOS if item.id == "light_brightness")
    good = {"choices": [{"message": {"tool_calls": [{"type": "function", "function": {
        "name": "light__HassLightSet",
        "arguments": '{"area":"Wohnzimmer","brightness":70,"domain":["light"]}'
    }}]}}]}
    assert benchmark.validate(scenario, good)["correct"] is True

    bad = {"choices": [{"message": {"tool_calls": [{"type": "function", "function": {
        "name": "light__HassLightSet",
        "arguments": '{"area":"Wiesentor","brightness":70,"domain":["light"]}'
    }}]}}]}
    result = benchmark.validate(scenario, bad)
    assert result["correct"] is False
    assert result["reason"] == "hallucinated area/name"


def test_all_tools_mode_expands_selection_without_changing_requested_model():
    scenario = next(item for item in benchmark.SCENARIOS if item.id == "vacuum_start")
    focused = benchmark.payload_for(scenario, 1200, benchmark.DEFAULT_MODEL, 128, 0.1, 42)
    expanded = benchmark.payload_for(scenario, 1200, benchmark.DEFAULT_MODEL, 128, 0.1, 42, True)
    assert len(focused["tools"]) == 3
    assert len(expanded["tools"]) == len(benchmark.TOOLS)
    assert expanded["model"] == benchmark.DEFAULT_MODEL
