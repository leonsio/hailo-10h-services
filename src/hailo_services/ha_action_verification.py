"""Deterministic verification and retry for Home Assistant on/off actions.

Home Assistant intent results report whether a service target was accepted, not
whether the entity actually reached the requested state.  For simple light and
switch TurnOn/TurnOff actions this module therefore performs a GetLiveContext
round after ``action_done``.  If an entity did not change state, it is retried
once with name + area + domain and verified again.  No Gemma inference is used
for this control loop.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from .i18n import t
from .json_utils import json_object as _json_object

_LOG = logging.getLogger(__name__)
_LIVE_TOOL = "homeassistant__GetLiveContext"
_ACTION_TO_STATE = {
    "intent__HassTurnOn": "on",
    "intent__HassTurnOff": "off",
}
_VERIFY_DOMAINS = {"light", "switch"}
# HA service calls can return action_done before the entity-state event has
# propagated through the state machine/integration. A short, generic settling
# window avoids immediately verifying stale state while keeping deterministic
# actions far faster than an LLM round-trip.
_VERIFY_SETTLE_SECONDS = 0.4


def _arguments(call):
    """Decode a historical action call argument object.

    Args:
        call (dict[str, Any]): Historical function call and its JSON arguments.

    Returns:
        dict[str, Any]: Decoded arguments, or an empty mapping for invalid input.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    function = call.get("function", {}) if isinstance(call, dict) else {}
    return _json_object(function.get("arguments")) or {}


def _tool_name(call):
    """Read the function name from a historical tool call.

    Args:
        call (dict[str, Any]): Historical function call and its JSON arguments.

    Returns:
        str | None: Function name if the call has a valid function mapping.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return call.get("function", {}).get("name") if isinstance(call, dict) else None


def _latest_user_index(messages):
    """Find the most recent user boundary in conversation history.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        int: Latest user index, or zero when no user message exists.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return 0


def _completed_round(messages):
    """Return the final assistant tool-call round and its tool result messages.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        tuple[list[dict], list[dict]] | None: Last complete call round and corresponding result messages.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if not messages or messages[-1].get("role") != "tool":
        return None
    first_tool = len(messages) - 1
    while first_tool > 0 and messages[first_tool - 1].get("role") == "tool":
        first_tool -= 1
    assistant_index = first_tool - 1
    if assistant_index < 0 or messages[assistant_index].get("role") != "assistant":
        return None
    calls = messages[assistant_index].get("tool_calls") or []
    if not isinstance(calls, list) or not calls:
        return None
    results = {message.get("tool_call_id"): message for message in messages[first_tool:]}
    if any(call.get("id") not in results for call in calls):
        return None
    return calls, [results[call.get("id")] for call in calls]


def _successful_action_results(results):
    """Check that every HA action result confirms successful execution.

    Args:
        results (list[Any]): Client tool results aligned to the active call round.

    Returns:
        bool: Whether all results contain success without failures.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    for message in results:
        payload = _json_object(message.get("content"))
        if payload is None or payload.get("response_type") != "action_done":
            return False
        data = payload.get("data")
        if not isinstance(data, dict) or data.get("failed"):
            return False
        success = data.get("success")
        if not isinstance(success, list) or not success:
            return False
    return True


def _live_result(results):
    """Extract live-context text from one HA tool result.

    Args:
        results (list[Any]): Client tool results aligned to the active call round.

    Returns:
        str | None: Live state text, or None for another result shape.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if len(results) != 1:
        return None
    payload = _json_object(results[0].get("content"))
    if payload is None or payload.get("success") is not True:
        return None
    result = payload.get("result")
    return result if isinstance(result, str) else None


def _live_entities(text):
    """Parse live-state text for deterministic on/off verification.

    Args:
        text (str): Text to parse, normalize, match or render.

    Returns:
        list[dict[str, str]]: Names, domains, areas and states for verification.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    entities = []
    blocks = re.split(r"(?m)(?=^- names:\s*)", text or "")
    for block in blocks:
        if not block.startswith("- names:"):
            continue
        name = re.search(r"(?m)^- names:\s*(.+)$", block)
        domain = re.search(r"(?m)^\s+domain:\s*(.+)$", block)
        state = re.search(r"(?m)^\s+state:\s*['\"]?([^'\"\n]+)['\"]?\s*$", block)
        area = re.search(r"(?m)^\s+areas:\s*(.+)$", block)
        if name and domain and state:
            entities.append(
                {
                    "name": name.group(1).strip(),
                    "domain": domain.group(1).strip(),
                    "state": state.group(1).strip().casefold(),
                    "area": area.group(1).strip() if area else "",
                }
            )
    return entities


def _action_calls_since_user(messages):
    """Collect action calls in the active user turn.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        list[dict[str, Any]]: Historical on/off action calls since the latest user request.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    start = _latest_user_index(messages)
    calls = []
    for message in messages[start:]:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if _tool_name(call) in _ACTION_TO_STATE:
                calls.append(call)
    return calls


def _initial_action(messages):
    """Resolve the first verifiable on/off action in the active turn.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        dict[str, Any] | None: Action metadata including expected state, or None.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    calls = _action_calls_since_user(messages)
    if not calls:
        return None
    first = calls[0]
    name = _tool_name(first)
    args = _arguments(first)
    domains = args.get("domain")
    if isinstance(domains, str):
        domains = [domains]
    if (
        not isinstance(domains, list)
        or not domains
        or any(not isinstance(domain, str) for domain in domains)
    ):
        return None
    if not set(domains) <= _VERIFY_DOMAINS:
        return None
    return {
        "tool": name,
        "expected_state": _ACTION_TO_STATE[name],
        "arguments": args,
        "attempts": len(calls),
    }


def _tool_call(name, arguments):
    """Build a function call using a new request-local call identifier.

    Args:
        name (str): Function, attribute, device or model identifier.
        arguments (dict[str, Any]): Function argument values to validate or encode.

    Returns:
        dict[str, Any]: OpenAI function call with JSON-encoded arguments.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return {
        "id": "call_" + uuid.uuid4().hex,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def action_verification_response(request):
    """Return the next deterministic verification/retry response, if applicable.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        dict[str, Any] | None: Verification, retry or final-response decision, or None when inapplicable.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    initial = _initial_action(request.messages)
    completed = _completed_round(request.messages)
    if initial is None or completed is None:
        return None
    calls, results = completed
    names = {_tool_name(call) for call in calls}

    if names <= set(_ACTION_TO_STATE):
        if not _successful_action_results(results):
            return None
        live_available = any(
            tool.get("function", {}).get("name") == _LIVE_TOOL for tool in (request.tools or [])
        )
        if not live_available:
            return None
        verify_args = {
            key: value
            for key, value in initial["arguments"].items()
            if key in {"name", "area", "domain"}
        }
        return {
            "kind": "verify",
            "response": {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call(_LIVE_TOOL, verify_args)],
            },
            "expected_state": initial["expected_state"],
            "arguments": verify_args,
            "attempts": initial["attempts"],
        }

    if names != {_LIVE_TOOL}:
        return None
    live = _live_result(results)
    if live is None:
        return None
    entities = _live_entities(live)
    if not entities:
        return None
    expected = initial["expected_state"]
    mismatches = [entity for entity in entities if entity["state"] != expected]
    if not mismatches:
        return {
            "kind": "verified",
            "response": t("ha_action_verification.212"),
            "expected_state": expected,
            "entities": entities,
            "attempts": initial["attempts"],
        }

    if initial["attempts"] < 2:
        retry_calls = []
        for entity in mismatches:
            arguments = {"name": entity["name"], "domain": [entity["domain"]]}
            if entity["area"]:
                arguments["area"] = entity["area"]
            retry_calls.append(_tool_call(initial["tool"], arguments))
        return {
            "kind": "retry",
            "response": {"role": "assistant", "content": None, "tool_calls": retry_calls},
            "expected_state": expected,
            "mismatches": mismatches,
            "attempts": initial["attempts"],
        }

    names_text = ", ".join(entity["name"] for entity in mismatches)
    return {
        "kind": "failed_verification",
        "response": t("ha_action_verification.236", names_text=names_text),
        "expected_state": expected,
        "mismatches": mismatches,
        "attempts": initial["attempts"],
    }


def _needs_live_tool(messages):
    """Check whether an action turn needs a live-state verification tool.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        bool: Whether GetLiveContext must be retained for verification.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    completed = _completed_round(messages)
    initial = _initial_action(messages)
    if initial is None or completed is None:
        return False
    calls, results = completed
    return {_tool_name(call) for call in calls} <= set(
        _ACTION_TO_STATE
    ) and _successful_action_results(results)


def prepare_request(backend, request, next_stage):
    """Apply this HA preparation stage and invoke its fallback when needed.

    Args:
        backend (ChatBackend): Resident backend used for generation or context preparation.
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        next_stage (Callable[[ChatRequest], ChatRequest]): Fallback preparation stage when this stage does not finish routing.

    Returns:
        ChatRequest: Prepared or unchanged request, possibly carrying a direct response.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    live_tool = next(
        (
            tool
            for tool in (request.tools or [])
            if tool.get("function", {}).get("name") == _LIVE_TOOL
        ),
        None,
    )
    prepared = next_stage(request)
    if live_tool is not None and _needs_live_tool(prepared.messages):
        tools = list(prepared.tools or [])
        if not any(tool.get("function", {}).get("name") == _LIVE_TOOL for tool in tools):
            tools.append(live_tool)
            prepared = prepared.model_copy(update={"tools": tools})
            prepared._request_id = getattr(request, "_request_id", "-")
    return prepared


def _speech(result):
    """Read non-empty plain speech from a Home Assistant action result.

    Args:
        result (Any): Native response or parsed HA action/intent result.

    Returns:
        str | None: Client-provided speech acknowledgement, or None.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    speech = result.get("speech")
    if not isinstance(speech, dict):
        return None
    plain = speech.get("plain")
    if isinstance(plain, dict):
        value = plain.get("speech")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def successful_action_followup(request):
    """Return a deterministic acknowledgement for a completed HA action turn.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        dict[str, Any] | None: Acknowledgement text and result evidence, or None when not safely complete.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    from .ha_pipeline import is_home_assistant_request

    if not is_home_assistant_request(request):
        return None
    messages = request.messages
    if not messages or messages[-1].get("role") != "tool":
        return None

    first_tool = len(messages) - 1
    while first_tool > 0 and messages[first_tool - 1].get("role") == "tool":
        first_tool -= 1
    assistant_index = first_tool - 1
    if assistant_index < 0:
        return None
    assistant = messages[assistant_index]
    calls = assistant.get("tool_calls") if assistant.get("role") == "assistant" else None
    if not isinstance(calls, list) or not calls:
        return None

    pending = {}
    for call in calls:
        if not isinstance(call, dict):
            return None
        identifier = call.get("id")
        function = call.get("function")
        if not isinstance(identifier, str) or not isinstance(function, dict):
            return None
        name = function.get("name")
        if not isinstance(name, str):
            return None
        pending[identifier] = name

    results = {}
    speeches = []
    successes = []
    for message in messages[first_tool:]:
        identifier = message.get("tool_call_id")
        if not isinstance(identifier, str) or identifier not in pending or identifier in results:
            return None
        payload = _json_object(message.get("content"))
        if payload is None or payload.get("response_type") != "action_done":
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        if data.get("failed"):
            return None
        success = data.get("success")
        spoken = _speech(payload)
        if not (isinstance(success, list) and success) and not spoken:
            return None
        results[identifier] = payload
        if spoken:
            speeches.append(spoken)
        if isinstance(success, list):
            successes.extend(item for item in success if isinstance(item, dict))

    if set(results) != set(pending):
        return None

    text = " ".join(dict.fromkeys(speeches)) if speeches else t("litert_optimizations.122")
    return {
        "text": text,
        "tool_names": [pending[call["id"]] for call in calls],
        "successes": successes,
        "tool_results": [results[call["id"]] for call in calls],
    }
