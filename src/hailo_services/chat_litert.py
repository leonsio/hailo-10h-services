"""LiteRT text-chat preparation, action acknowledgements and timing correlation."""

import json
import logging

from .diagnostics_litert import request_diagnostics
from .ha_action_verification import successful_action_followup

_LOG = logging.getLogger(__name__)


def run_chat(backend, request, generate, emit=None, cancelled=None, tools_prepared=False):
    """Apply successful-action follow-up and scope LiteRT timing correlation.

    Args:
        backend (ChatBackend): Resident backend used for generation or context preparation.
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        generate (Callable[..., ChatResult]): Native generation function invoked after fast paths.
        emit (ChatEmitter | None): Optional callback receiving text chunks or validated assistant messages.
        cancelled (threading.Event | None): Optional cancellation event checked between generated chunks.
        tools_prepared (bool): Whether context/tool preparation has already run for this request.

    Returns:
        ChatResult: Deterministic acknowledgement or generated response.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    request_id = getattr(request, "_request_id", "-")
    fast = successful_action_followup(request)
    if fast is not None:
        _LOG.info(
            "tool_followup_fast_path request_id=%s tools=%s successes=%d skipped_gemma=true",
            request_id,
            ",".join(fast["tool_names"]),
            len(fast["successes"]),
        )
        if getattr(backend, "debug_log", False):
            _LOG.debug(
                "event=tool_followup_fast_path request_id=%s json=%s",
                request_id,
                json.dumps(fast, ensure_ascii=False, separators=(",", ":"), default=str),
            )
        if emit:
            emit(fast["text"])
        return fast["text"]

    with request_diagnostics(request):
        return generate(request, emit, cancelled, tools_prepared)
