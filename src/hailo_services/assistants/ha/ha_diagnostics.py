"""Explain HA preparation without generative inference or tool execution."""

import copy
import time

from hailo_services.config import HA_ASSIST_MODEL


def diagnose(backend, request):
    """Prepare a copied request and expose proposals, candidates and prompt evidence.

    Args:
        backend: Hailo owner-thread backend; optional MiniLM retrieval is permitted.
        request: HA-Assist request to inspect, independent of production chat history.

    Returns:
        dict: Proposed route, actions and prepared prompt before native token budgeting.

    Raises:
        ValueError: The request targets a physical model, streaming, or disabled HA-Assist.
    """
    from hailo_services.assistants.ha.ha_assist import prepare, target_model

    if request.model != HA_ASSIST_MODEL or request.stream:
        raise ValueError("Diagnosis requires model HA-Assist and stream=false")
    started = time.perf_counter()
    copied = request.model_copy(deep=True)
    copied._metrics = {}
    object.__setattr__(copied, "_ha_diagnostic", True)
    copied = copied.model_copy(update={"model": target_model(backend.settings, copied)})
    prepared, direct = prepare(backend, copied)
    metrics = copy.deepcopy(copied._metrics)
    route = metrics.get("ha_route", {})
    route["would_inference_calls"] = route.get("inference_calls", 0)
    route["inference_calls"] = 0
    return {
        "object": "ha_assist.diagnosis",
        "model": HA_ASSIST_MODEL,
        "generative_calls": 0,
        "tools_executed": 0,
        "duration_ms": (time.perf_counter() - started) * 1000,
        "proposed_response": direct,
        "metrics": metrics,
        "prompt_stage": "ha_prepared_before_native_template_and_budget",
        "prepared_request": {
            "model": prepared.model,
            "messages": prepared.messages,
            "tools": prepared.tools,
            "tool_choice": prepared.tool_choice,
            "max_tokens": prepared.max_tokens,
        },
    }
