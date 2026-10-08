"""Experimental virtual Frigate model with deterministic planning and isolated prompts."""

import json
import logging
import time

from .config import FRIGATE_ASSIST_MODEL, LLM_MODEL
from .frigate_deterministic import deterministic_plan, resolved_facts
from .frigate_prompt import compile_request, text_content, uses_images
from .frigate_routing import plan, resolved_context

_LOG = logging.getLogger(__name__)


_MODEL_TOOL_ROUTES = {
    "deterministic_historical_search_tool_selection",
    "deterministic_watch_tool_selection",
}


def _tool_names(tools):
    """Return selected tool names for conservative planner comparisons.

    Args:
        tools: OpenAI function tool declarations.

    Returns:
        set[str]: Function names contained in the declarations.
    """
    return {tool["function"]["name"] for tool in tools or []}


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
    images = uses_images(request)
    deterministic = deterministic_plan(request, settings, images)
    if deterministic is None:
        tools, direct, reason = plan(request, settings, images)
    else:
        tools, direct, reason = deterministic
        # The deterministic layer may narrow a single obvious read tool before
        # semantic inference. If the conservative legacy planner sees additional
        # tools, retain them: that indicates a compound intent or an unknown/new
        # Frigate capability which this layer must not silently discard.
        if direct is None and reason == "deterministic_historical_search_tool_selection":
            legacy_tools, legacy_direct, legacy_reason = plan(request, settings, images)
            if legacy_direct is None and _tool_names(legacy_tools) - _tool_names(tools):
                tools, direct, reason = legacy_tools, legacy_direct, legacy_reason
    target = settings.frigate_vision_model if images else settings.frigate_text_model
    # Compilation validates the entire active round even for deterministic calls.
    context = resolved_context(request)
    context.update(resolved_facts(request))
    prepared = compile_request(request, settings, tools, images, context=context)
    if direct is None and prepared.tools and reason in _MODEL_TOOL_ROUTES:
        # The deterministic planner has already chosen the only safe capability.
        # Requiring the call prevents a small model from answering from memory or
        # asking for information that is already encoded in the constrained schema.
        prepared = prepared.model_copy(update={"tool_choice": "required"})
    if direct is None:
        if images and (not settings.vlm_enabled or target != settings.vlm_model):
            raise ValueError(
                f"Frigate-Assist image target {target!r} must match the enabled VLM "
                f"{settings.vlm_model!r}; configure frigate_assist_vision_model or leave it empty"
            )
        if not images:
            allowed = {settings.vlm_model} if settings.vlm_enabled else set()
            if settings.litert_enabled or settings.litert_model_path:
                allowed.add(LLM_MODEL)
            if settings.hailo_llm_enabled:
                allowed.add(settings.hailo_llm_model_id)
            if target not in allowed:
                raise ValueError(
                    f"Frigate-Assist text target {target!r} is not an enabled LLM/VLM; "
                    "configure frigate_assist_text_model and enable its backend; explicit unavailable targets do not fall back"
                )
    image_chat = images and (
        bool(request.tools)
        or any(
            m.get("role") == "system" and "helpful assistant for Frigate" in text_content(m)
            for m in request.messages
        )
    )
    prepared = prepared.model_copy(
        update={
            "model": target,
            "max_tokens": min(request.max_tokens, settings.frigate_assist_vision_max_tokens)
            if image_chat
            else request.max_tokens,
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
