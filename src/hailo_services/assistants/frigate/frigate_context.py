"""Shared Frigate timestamp, catalogue and active tool-context helpers."""

import json
import re
from datetime import datetime

from hailo_services.shared.tool_calling import response_message


def local_datetime(value):
    """Parse a supplied Frigate local timestamp without timezone assumptions.

    Args:
        value: Timestamp string from the Frigate prompt or tool result.

    Returns:
        datetime | None: Parsed naive local datetime, or None when unsupported.
    """
    if not isinstance(value, str):
        return None
    for fmt in (
        "%Y-%m-%d at %I:%M:%S %p",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def tool_results(request):
    """Decode tool results that belong to the latest user turn.

    Args:
        request: Incoming chat request containing assistant calls and tool responses.

    Returns:
        dict: Tool names mapped to decoded JSON results for the active turn.
    """
    users = [i for i, m in enumerate(request.messages) if m.get("role") == "user"]
    if not users:
        return {}
    calls, results = {}, {}
    for message in request.messages[users[-1] :]:
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            if isinstance(call.get("id"), str) and isinstance(function.get("name"), str):
                calls[call["id"]] = function["name"]
        if message.get("role") == "tool" and message.get("tool_call_id") in calls:
            try:
                results[calls[message["tool_call_id"]]] = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                results[calls[message["tool_call_id"]]] = {"error": "Tool returned non-JSON data"}
    return results


def matched_cameras(text, catalogue):
    """Match exact camera IDs or friendly names without fuzzy guessing.

    Args:
        text: Latest user text.
        catalogue: Mapping of camera IDs to friendly names.

    Returns:
        list[str]: Exact camera IDs mentioned in the text.
    """
    return [
        identifier
        for identifier, friendly in catalogue.items()
        if any(
            re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I)
            for value in {identifier, friendly}
        )
    ]


def tool_call(request, name, arguments):
    """Build a schema-validated assistant tool call.

    Args:
        request: Request whose selected tool schema validates the call.
        name: Tool name.
        arguments: Exact deterministic arguments.

    Returns:
        dict: OpenAI-compatible assistant tool-call message.
    """
    return response_message(
        {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}, request, ""
    )


def select_tool(tools, name):
    """Select declarations for one exact tool name.

    Args:
        tools: Available OpenAI function tool declarations.
        name: Function name to retain.

    Returns:
        list[dict]: Matching tool declarations.
    """
    return [tool for tool in tools if tool["function"]["name"] == name]
