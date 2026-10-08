"""Non-throwing parsing for JSON objects in tool results and arguments."""

import json
from typing import Any


def json_object(value: object) -> dict[str, Any] | None:
    """Parse an object or JSON string; return None for other JSON values.

    Args:
        value: A decoded object or JSON-encoded string.

    Returns:
        The original dictionary or decoded dictionary; None for invalid input.

    Raises:
        No exceptions for invalid JSON or unsupported input types.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
