"""Request-local diagnostics; missing native metrics remain unavailable."""

import math
from datetime import datetime, timezone


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def record(metrics, **values):
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or value < 0:
                continue
        if value is not None:
            metrics[key] = value


def count_output(metrics, tokenize, text):
    """Retokenized output is identified separately from native decode counts."""
    if "output_tokens" in metrics or not callable(tokenize):
        return
    try:
        record(metrics, output_tokens=len(tokenize(text)), output_tokens_source="tokenizer")
    except Exception:
        pass  # Diagnostics must never make successful inference fail.


def response_metrics(metrics, state):
    import time

    result = dict(metrics)
    record(result, request_id=state.get("request_id", "-"),
           requested_at=state.get("requested_at"), responded_at=timestamp(),
           processing_ms=(time.perf_counter() - state["request_started"]) * 1000)
    return result
