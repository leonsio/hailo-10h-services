import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark-qwen-ha.py"
spec = importlib.util.spec_from_file_location("benchmark_qwen_ha", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def scenario(name="light_brightness"):
    return next(item for item in benchmark.SCENARIOS if item.id == name)


def test_qwen_payload_cannot_trigger_production_ha_envelope_shortcut():
    payload = benchmark.payload_for(
        scenario(), 700, benchmark.DEFAULT_MODEL, 128, 0.1, 42
    )
    assert payload["model"] == benchmark.DEFAULT_MODEL
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert "Static Context:" not in payload["messages"][0]["content"]
    assert [item["function"]["name"] for item in payload["tools"]] == [
        "light__HassLightSet"
    ]


def test_focused_baseline_contains_only_relevant_entities():
    payload = benchmark.payload_for(
        scenario(), 100, benchmark.DEFAULT_MODEL, 128, 0.1, 42
    )
    system = payload["messages"][0]["content"]
    assert "Licht Tisch" in system
    assert "Fenster - Twinkly" in system
    assert "Rollladen" not in system
    assert "Deebot mini" not in system


def test_prompt_load_grows_monotonically_with_targets():
    sizes = []
    estimates = []
    for target in (400, 700, 1000, 1300):
        payload = benchmark.payload_for(
            scenario(), target, benchmark.DEFAULT_MODEL, 128, 0.1, 42
        )
        system = payload["messages"][0]["content"]
        sizes.append(len(system))
        estimates.append(
            benchmark.estimate_tokens(system, scenario().prompt, payload["tools"])
        )
    assert sizes == sorted(sizes)
    assert estimates == sorted(estimates)
    assert estimates[-1] > estimates[0]


def test_valid_expected_tool_call_passes_and_unknown_area_is_hallucination():
    good = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "light__HassLightSet",
                                "arguments": (
                                    '{"area":"Wohnzimmer","brightness":70,'
                                    '"domain":["light"]}'
                                ),
                            },
                        }
                    ]
                }
            }
        ]
    }
    assert benchmark.validate(scenario(), good)["correct"] is True

    bad = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "light__HassLightSet",
                                "arguments": (
                                    '{"area":"Wiesentor","brightness":70,'
                                    '"domain":["light"]}'
                                ),
                            },
                        }
                    ]
                }
            }
        ]
    }
    result = benchmark.validate(scenario(), bad)
    assert result["correct"] is False
    assert result["reason"] == "hallucinated area/name"
    assert result["arguments"]["area"] == "Wiesentor"


def test_tool_modes_separate_context_limit_from_tool_selection_complexity():
    focused = benchmark.payload_for(
        scenario("vacuum_start"), 700, benchmark.DEFAULT_MODEL, 128, 0.1, 42
    )
    distractors = benchmark.payload_for(
        scenario("vacuum_start"),
        700,
        benchmark.DEFAULT_MODEL,
        128,
        0.1,
        42,
        tool_mode="distractors",
    )
    expanded = benchmark.payload_for(
        scenario("vacuum_start"), 700, benchmark.DEFAULT_MODEL, 128, 0.1, 42, True
    )
    assert len(focused["tools"]) == 1
    assert len(distractors["tools"]) == 3
    assert len(expanded["tools"]) == len(benchmark.TOOLS)
    assert expanded["model"] == benchmark.DEFAULT_MODEL


def test_model_output_http_errors_are_classified_separately_from_token_limit():
    missing = benchmark.HttpFailure(
        400, '{"error":"Model did not return the required tool call"}'
    )
    invalid = benchmark.HttpFailure(
        400,
        '{"error":"Model returned invalid arguments for light__HassLightSet: x"}',
    )
    limit = benchmark.HttpFailure(
        400,
        '{"error":{"message":"Input token limit exceeded","code":"input_token_limit_exceeded"}}',
    )
    assert benchmark.classify_http_failure(missing) == (
        "model output: required tool call missing"
    )
    assert benchmark.classify_http_failure(invalid) == (
        "model output: invalid tool arguments"
    )
    assert benchmark.classify_http_failure(limit) == "input token limit"
