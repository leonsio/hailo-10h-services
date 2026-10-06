"""Virtual-model boundary: zero deterministic or exactly one generative backend call."""

import json
import logging
import time

from .config import HA_ASSIST_MODEL, LLM_MODEL
from .ha_action_verification import (
    _VERIFY_SETTLE_SECONDS,
    action_verification_response,
    successful_action_followup,
)
from .ha_pipeline import _DIRECT_ATTRIBUTES
from .i18n import detect_language, using_language
from .tool_calling import has_tool_context, native_messages
from .tool_retrieval import latest_user_text

_LOG = logging.getLogger(__name__)


def has_images(request):
    """Check whether any message contains an image URL content part.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        bool: Whether the request needs a vision-capable backend.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return any(
        part.get("type") == "image_url"
        for message in request.messages
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
    )


def target_model(settings, request):
    """Resolve and validate the HA-Assist backend for text or image input.

    Args:
        settings (Settings): Validated service settings controlling enabled models and limits.
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        str: Enabled configured backend model identifier.

    Raises:
        ValueError: HA-Assist is disabled.
    """
    if not settings.ha_assist_enabled:
        raise ValueError("HA-Assist is disabled")
    images = has_images(request)
    target = settings.ha_assist_vision_model if images else settings.ha_assist_text_model
    # Validate role, not just presence in /v1/models: text never falls back to a VLM.
    allowed = {settings.vlm_model} if settings.vlm_enabled and images else set()
    if not images:
        if settings.hailo_llm_enabled:
            allowed.add(settings.hailo_llm_model_id)
        if settings.litert_enabled or settings.litert_model_path:
            allowed.add(LLM_MODEL)
    if target not in allowed:
        raise ValueError(
            f"HA-Assist {'vision' if images else 'text'} target {target!r} is not an enabled "
            f"{'VLM' if images else 'LLM'}; configure the target and enable that backend"
        )
    return target


def prepare(backend, request):
    """Executed on the Hailo owner thread, so optional MiniLM is serialized.

    Args:
        backend (ChatBackend): Resident backend used for generation or context preparation.
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        tuple[ChatRequest, ChatResult | None]: Prepared backend request and optional deterministic answer.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    from .runtime import HailoBackend

    started = time.perf_counter()
    images = has_images(request)
    if not images and has_tool_context(request):
        native_messages(request.messages)  # Fast paths must preserve call/result dependencies too.
    language = request.language or detect_language(
        latest_user_text(request.messages), backend.settings.service_language
    )
    object.__setattr__(request, "_ha_assist", True)
    object.__setattr__(request, "_ha_request", True)
    object.__setattr__(request, "_response_language", language)
    with using_language(language):
        if images:
            # Never let a text-only shortcut answer a question about the image.
            prepared = (
                backend.retrieve_context(request) if isinstance(backend, HailoBackend) else request
            )
            direct = None
        else:
            prepared = (
                backend.select_tools(request) if isinstance(backend, HailoBackend) else request
            )
            decision = (
                action_verification_response(prepared) if request.tool_choice != "none" else None
            )
            if decision and decision["kind"] == "verify":
                time.sleep(_VERIFY_SETTLE_SECONDS)
                request._metrics["ha_verify_settle_ms"] = _VERIFY_SETTLE_SECONDS * 1000
            fast = (
                successful_action_followup(prepared)
                if not decision and request.tool_choice != "none"
                else None
            )
            direct = (
                decision["response"]
                if decision
                else fast["text"]
                if fast
                else next(
                    (
                        getattr(prepared, name)
                        for name in _DIRECT_ATTRIBUTES
                        if getattr(prepared, name, None) is not None
                    ),
                    None,
                )
            )
    trace = {
        "requested_model": HA_ASSIST_MODEL,
        "backend_model": request.model,
        "route": "direct" if direct is not None else "vlm" if images else "llm",
        "reason": "deterministic"
        if direct is not None
        else "image_present"
        if images
        else "text_only",
        "inference_calls": 0 if direct is not None else 1,
        "prepare_ms": (time.perf_counter() - started) * 1000,
        "language": language,
        "tools_after": len(prepared.tools or []),
    }
    request._metrics["ha_route"] = trace
    _LOG.info(
        "ha_assist request_id=%s route=%s target=%s inference_calls=%d",
        request._request_id,
        trace["route"],
        request.model,
        trace["inference_calls"],
    )
    if backend.settings.debug_log:
        _LOG.debug(
            "event=ha_assist request_id=%s json=%s",
            request._request_id,
            json.dumps(trace, ensure_ascii=False),
        )
    return prepared, direct
