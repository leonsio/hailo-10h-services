"""Experimental virtual Frigate model with deterministic planning and isolated prompts."""

import json
import logging
import time

from .config import FRIGATE_ASSIST_MODEL, LLM_MODEL
from .frigate_prompt import compile_request, has_images
from .frigate_routing import plan

_LOG = logging.getLogger(__name__)


def prepare(settings, request):
    """Prepare a deterministic call or exactly one native text/vision request.

    Args:
        settings: Validated service settings.
        request: Incoming virtual-model request.

    Returns:
        tuple: Prepared native request and optional deterministic response.

    Raises:
        ValueError: Proxy is disabled or the necessary model is not enabled.
    """
    if not settings.frigate_assist_enabled:
        raise ValueError("Frigate-Assist is disabled")
    if not any(message.get("role") == "user" for message in request.messages):
        raise ValueError("Frigate-Assist requires a user question or image")
    started = time.perf_counter()
    images = has_images(request)
    tools, direct, reason = plan(request, settings, images)
    target = settings.frigate_assist_vision_model if images else LLM_MODEL
    # Compilation validates the entire active round even for deterministic calls.
    prepared = compile_request(request, settings, tools, images)
    if direct is None:
        if images and (not settings.vlm_enabled or target != settings.vlm_model):
            raise ValueError("Frigate-Assist image requests require the configured enabled VLM")
        if not images and not (settings.litert_enabled or settings.litert_model_path):
            raise ValueError(
                "Frigate-Assist text requests require Gemma/LiteRT-LM; no VLM fallback"
            )
    prepared = prepared.model_copy(
        update={
            "model": target,
            "temperature": min(0.1, max(0.01, request.temperature))
            if images
            else request.temperature,
        }
    )
    trace = {
        "requested_model": FRIGATE_ASSIST_MODEL,
        "backend_model": target if direct is None else None,
        "route": "direct" if direct is not None else "vlm" if images else "llm",
        "reason": reason,
        "inference_calls": 0 if direct is not None else 1,
        "prepare_ms": (time.perf_counter() - started) * 1000,
    }
    request._metrics["frigate_route"] = trace
    _LOG.info(
        "frigate_assist request_id=%s route=%s target=%s reason=%s tools=%d->%d",
        request._request_id,
        trace["route"],
        trace["backend_model"],
        reason,
        len(request.tools or []),
        len(prepared.tools or []),
    )
    if settings.debug_log:
        _LOG.debug(
            "event=frigate_assist request_id=%s json=%s",
            request._request_id,
            json.dumps({**trace, **request._metrics["frigate_prompt"]}, ensure_ascii=False),
        )
    return prepared, direct
