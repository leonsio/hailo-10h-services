"""Structured debug logging without backend-specific imports."""

import json
import logging
from typing import Any


def debug_json(
    logger: logging.Logger, enabled: bool, event: str, payload: Any, *, request_id: str = "-"
) -> None:
    """Log a JSON event only when request diagnostics are enabled.

    Args:
        logger: Logger owned by the calling module.
        enabled: Whether verbose diagnostics are enabled.
        event: Stable event name for log filtering.
        payload: JSON data; unsupported objects are converted to strings.
        request_id: Correlation identifier for the current request.

    Returns:
        None. Emits a debug record when enabled.
    """
    if enabled:
        logger.debug(
            "event=%s request_id=%s json=%s",
            event,
            request_id,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
        )


def cache_status(hailo_backend, litert_backend):
    """Report observed context APIs without saving, loading or sharing user state.

    Args:
        hailo_backend: Resident Hailo backend, possibly with LLM/VLM disabled.
        litert_backend: Optional LiteRT backend with a per-request conversation engine.

    Returns:
        dict: Bounded preprocessing cache counts and native context capabilities/policies.
    """
    from hailo_services.assistants.ha.ha_recognition import _templates
    from hailo_services.shared.tool_retrieval import _tool_index

    native = {}
    for kind in ("llm", "vlm"):
        model = getattr(hailo_backend, kind, None)
        native[kind] = {
            "context_snapshot_api": bool(
                model
                and callable(getattr(model, "save_context", None))
                and callable(getattr(model, "load_context", None))
            ),
            "service_policy": "clear_per_request",
            "service_snapshot_reuse": False,
        }
    return {
        "static_tool_index": _tool_index.cache_info()._asdict(),
        "sentence_templates": _templates.cache_info()._asdict(),
        "minilm_embeddings": {
            "entries": len(getattr(hailo_backend, "_retrieval_embedding_cache", {})),
            "limit": 512,
        },
        "hailo": native,
        "litert": {
            "conversation_api": callable(
                getattr(getattr(litert_backend, "engine", None), "create_conversation", None)
            ),
            "service_policy": "new_conversation_per_request",
            "native_prefix_reuse": "runtime_dependent_not_measured",
        },
        "live_state_or_action_response_cache": False,
    }
