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
