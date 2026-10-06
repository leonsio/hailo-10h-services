import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark-qwen-text-intent.py"
spec = importlib.util.spec_from_file_location("benchmark_qwen_text_intent", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def scenario(name):
    return next(item for item in benchmark.SCENARIOS if item.id == name)


def response(content, model="Qwen3-VL-2B-Instruct"):
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


def test_payload_is_a_true_model_only_request_without_tools_or_ha_envelope():
    item = scenario("intent_light_brightness")
    payload = benchmark.payload_for(item, "Qwen3-VL-2B-Instruct", 32, 0.1, 42)
    assert payload["model"] == "Qwen3-VL-2B-Instruct"
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "Static Context:" not in payload["messages"][0]["content"]
    assert "INTENT|TARGET_TYPE|TARGET|VALUE" in payload["messages"][0]["content"]


def test_general_text_validation_is_short_and_deterministic():
    item = scenario("text_capital_france")
    assert benchmark.validate(item, response("Paris"))["correct"] is True
    assert benchmark.validate(item, response("paris."))["correct"] is True
    result = benchmark.validate(item, response("Die Antwort ist Paris."))
    assert result["correct"] is False
    assert result["reason"] == "wrong answer"


def test_intent_validation_accepts_safe_case_only_canonicalization():
    item = scenario("intent_light_brightness")
    good = benchmark.validate(
        item,
        response("LIGHT_BRIGHTNESS|area|wohnzimmer|70"),
    )
    assert good["correct"] is True

    bad = benchmark.validate(
        item,
        response("LIGHT_BRIGHTNESS|area|Wohn Zimmer|70"),
    )
    assert bad["correct"] is False
    assert bad["reason"] == "wrong target"


def test_intent_validation_rejects_extra_text_and_wrong_slots():
    item = scenario("intent_climate_temperature")
    exact = benchmark.validate(
        item,
        response("CLIMATE_TEMPERATURE|area|Schlafzimmer|19.5"),
    )
    assert exact["correct"] is True

    extra = benchmark.validate(
        item,
        response("Hier ist das Ergebnis:\nCLIMATE_TEMPERATURE|area|Schlafzimmer|19,5"),
    )
    assert extra["correct"] is False
    assert extra["reason"] == "expected exactly one intent line"

    wrong = benchmark.validate(
        item,
        response("CLIMATE_TEMPERATURE|area|Schlafzimmer|21"),
    )
    assert wrong["correct"] is False
    assert wrong["reason"] == "wrong value"


def test_single_code_fence_is_tolerated_but_multiple_answers_are_not():
    item = scenario("intent_vacuum_start")
    fenced = benchmark.validate(
        item,
        response("```text\nVACUUM_START|name|Deebot mini|-\n```"),
    )
    assert fenced["correct"] is True

    multiple = benchmark.validate(
        item,
        response("VACUUM_START|name|Deebot mini|-\nVACUUM_START|name|Deebot mini|-"),
    )
    assert multiple["correct"] is False
    assert multiple["reason"] == "expected exactly one intent line"


def test_summary_separates_text_and_intent_accuracy_and_latency():
    rows = [
        {"group": "text", "ok": True, "client_total_ms": 100, "inference_ms": 80, "ttft_ms": 20, "input_tokens": 10, "input_budget_tokens": 138, "output_tokens": 1, "validation": {"reason": "ok"}},
        {"group": "text", "ok": False, "client_total_ms": 200, "inference_ms": 180, "ttft_ms": 30, "input_tokens": 10, "input_budget_tokens": 138, "output_tokens": 2, "validation": {"reason": "wrong answer"}},
        {"group": "intent", "ok": True, "client_total_ms": 300, "inference_ms": 250, "ttft_ms": 40, "input_tokens": 20, "input_budget_tokens": 148, "output_tokens": 8, "validation": {"reason": "ok"}},
    ]
    summary = benchmark.summarize(rows)
    assert summary["text"]["accuracy"] == 0.5
    assert summary["intent"]["accuracy"] == 1.0
    assert summary["all"]["accuracy"] == 2 / 3
    assert summary["text"]["avg_client_total_ms"] == 150
    assert summary["all"]["failure_reasons"] == {"wrong answer": 1}
