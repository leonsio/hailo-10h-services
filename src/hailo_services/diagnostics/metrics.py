"""Request-local diagnostics; missing native metrics remain unavailable."""

import math
from datetime import datetime, timezone


def timestamp():
    """Return the current UTC timestamp with millisecond precision.

    Returns:
        str: ISO 8601 UTC timestamp.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def record(metrics, **values):
    """Store valid non-negative finite measurements without overwriting unavailable values.

    Args:
        metrics (dict[str, Any]): Mutable request-local measurements.
        **values (Any): Formatting substitutions, metric updates or catalogue values, depending on the helper.

    Returns:
        None: Updates the supplied metrics mapping in place.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or value < 0:
                continue
        if value is not None:
            metrics[key] = value


def count_output(metrics, tokenize, text):
    """Retokenized output is identified separately from native decode counts.

    Args:
        metrics (dict[str, Any]): Mutable request-local measurements.
        tokenize (Callable[[str], Any] | None): Optional native tokenizer used for diagnostic token counts.
        text (str): Text to parse, normalize, match or render.

    Returns:
        None: Records retokenized output when available; ignores diagnostic failures.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if "output_tokens" in metrics or not callable(tokenize):
        return
    try:
        record(metrics, output_tokens=len(tokenize(text)), output_tokens_source="tokenizer")
    except Exception:
        pass  # Diagnostics must never make successful inference fail.


def response_metrics(metrics, state):
    """Add correlation, response timestamps and total request duration.

    Args:
        metrics (dict[str, Any]): Mutable request-local measurements.
        state (dict[str, Any]): Request start/correlation state.

    Returns:
        dict[str, Any]: Copy of metrics enriched with final request timing.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    import time

    result = dict(metrics)
    record(
        result,
        request_id=state.get("request_id", "-"),
        requested_at=state.get("requested_at"),
        responded_at=timestamp(),
        processing_ms=(time.perf_counter() - state["request_started"]) * 1000,
    )
    return result
