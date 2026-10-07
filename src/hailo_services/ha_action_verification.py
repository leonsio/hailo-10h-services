"""Deterministic verification for Home Assistant on/off actions.

Home Assistant intent results report whether a service target was accepted, not
whether the entity already exposes the requested state. For simple light and
switch TurnOn/TurnOff actions this module can therefore perform one or more
GetLiveContext verification rounds after ``action_done``. Verification retries
only read state; a successfully accepted action is never re-issued merely
because state propagation is delayed. No generative inference is used for this
control loop.
"""

from __future__ import annotations

import json
import re
import uuid

from .i18n import t
from .json_utils import json_object as _json_object

_LIVE_TOOL = "homeassistant__GetLiveContext"
_ACTION_TO_STATE = {
    "intent__HassTurnOn": "on",
    "intent__HassTurnOff": "off",
}
_VERIFY_DOMAINS = {"light", "switch"}
_DEFAULT_VERIFY_ATTEMPTS = 2


def _arguments(call):
    """Decode a historical action call argument object.

    Args:
        call (dict[str, Any]): Historical function call.

    Returns:
        dict[str, Any]: Decoded arguments, or an empty mapping for invalid input.
    """
    function = call.get("function", {}) if isinstance(call, dict) else {}
    return _json_object(function.get("arguments")) or {}


def _tool_name(call):
    """Read the function name from a historical tool call.

    Args:
        call (dict[str, Any]): Historical function call.

    Returns:
        str | None: Function name when available.
    """
    return call.get("function", {}).get("name") if isinstance(call, dict) else None


def _latest_user_index(messages):
    """Find the most recent user boundary in conversation history.

    Args:
        messages (list[dict[str, Any]]): Ordered conversation messages.

    Returns:
        int: Index of the most recent user message, or zero.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return 0


def _completed_round(messages):
    """Return the final assistant tool-call round and its tool result messages.

    Args:
        messages (list[dict[str, Any]]): Ordered conversation messages.

    Returns:
        tuple[list[dict], list[dict]] | None: Completed calls/results, or None.
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


def _action_result_summary(results):
    """Parse HA action results into successful and failed targets.

    Args:
        results (list[dict[str, Any]]): Tool-result messages for one action round.

    Returns:
        dict[str, list] | None: Successful and failed targets, or None for an unknown result shape.
    """
    successes = []
    failures = []
    for message in results:
        payload = _json_object(message.get("content"))
        if payload is None or payload.get("response_type") != "action_done":
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        success = data.get("success")
        failed = data.get("failed")
        if not isinstance(success, list) or not isinstance(failed, list):
            return None
        successes.extend(success)
        failures.extend(failed)
    return {"successes": successes, "failures": failures}


def _successful_action_results(results):
    """Check that every HA action result confirms execution without failures.

    Args:
        results (list[dict[str, Any]]): Tool-result messages for one action round.

    Returns:
        bool: Whether action execution succeeded without failed targets.
    """
    summary = _action_result_summary(results)
    return bool(summary and summary["successes"] and not summary["failures"])


def _target_names(items):
    """Return stable unique target names from Home Assistant result entries.

    Args:
        items (list[Any]): Home Assistant target result entries.

    Returns:
        list[str]: Stable unique names or IDs for the supplied targets.
    """
    names = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("name") or item.get("id")
        else:
            value = item
        if isinstance(value, str) and value.strip() and value.strip() not in names:
            names.append(value.strip())
    return names


def _action_failure_text(failures):
    """Render an explicit action failure distinct from a stale verification state.

    Args:
        failures (list[Any]): Explicit failed targets reported by Home Assistant.

    Returns:
        str: Localized text describing explicit failed action targets.
    """
    names = _target_names(failures)
    if not names:
        return t("ha_plan.tool_failed")
    return f"{t('ha_plan.tool_failed')} {t('device.plural')}: {', '.join(names)}."


def _live_result(results):
    """Extract live-context text from one HA tool result.

    Args:
        results (list[dict[str, Any]]): Tool-result messages for a live-state call.

    Returns:
        str | None: Live-context text when the result is valid.
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
        text (str): Home Assistant live-context text.

    Returns:
        list[dict[str, str]]: Parsed entity names, domains, areas and states.
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
    """Collect on/off action calls in the active user turn.

    Args:
        messages (list[dict[str, Any]]): Ordered conversation messages.

    Returns:
        list[dict[str, Any]]: On/off action calls in the active turn.
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


def _live_calls_since_user(messages):
    """Count completed GetLiveContext verification calls in the active turn.

    Args:
        messages (list[dict[str, Any]]): Ordered conversation messages.

    Returns:
        int: Number of live-state verification calls in the active turn.
    """
    start = _latest_user_index(messages)
    count = 0
    for message in messages[start:]:
        if message.get("role") != "assistant":
            continue
        count += sum(
            1 for call in message.get("tool_calls") or [] if _tool_name(call) == _LIVE_TOOL
        )
    return count


def _initial_action(messages):
    """Resolve the first verifiable on/off action in the active turn.

    Args:
        messages (list[dict[str, Any]]): Ordered conversation messages.

    Returns:
        dict[str, Any] | None: First verifiable action metadata, or None.
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
        "action_calls": len(calls),
    }


def _verify_attempt_limit(settings):
    """Return the configured maximum number of live-state reads.

    Args:
        settings (Settings | None): Service settings, or None for the default.

    Returns:
        int: Validated non-negative maximum number of state reads.
    """
    if settings is None:
        return _DEFAULT_VERIFY_ATTEMPTS
    return max(0, int(getattr(settings, "ha_assist_verify_attempts", _DEFAULT_VERIFY_ATTEMPTS)))


def _verify_arguments(initial):
    """Build the narrow live-context filter for an action target.

    Args:
        initial (dict[str, Any]): Initial action metadata.

    Returns:
        dict[str, Any]: Name/area/domain filter used for live-state reads.
    """
    return {
        key: value
        for key, value in initial["arguments"].items()
        if key in {"name", "area", "domain"}
    }


def _tool_call(name, arguments):
    """Build a function call using a new request-local call identifier.

    Args:
        name (str): Function name.
        arguments (dict[str, Any]): Function arguments.

    Returns:
        dict[str, Any]: OpenAI-compatible function call object.
    """
    return {
        "id": "call_" + uuid.uuid4().hex,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _live_tool_available(request):
    """Check whether Home Assistant supplied GetLiveContext for this request.

    Args:
        request (ChatRequest): Active Home Assistant chat request.

    Returns:
        bool: Whether GetLiveContext is available on this request.
    """
    return any(tool.get("function", {}).get("name") == _LIVE_TOOL for tool in (request.tools or []))


def _verify_decision(initial, verify_attempt):
    """Build the next deterministic GetLiveContext call.

    Args:
        initial (dict[str, Any]): Initial action metadata.
        verify_attempt (int): One-based live-state read number.

    Returns:
        dict[str, Any]: Deterministic decision requesting the next state read.
    """
    arguments = _verify_arguments(initial)
    return {
        "kind": "verify" if verify_attempt == 1 else "verify_retry",
        "response": {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call(_LIVE_TOOL, arguments)],
        },
        "expected_state": initial["expected_state"],
        "arguments": arguments,
        "action_calls": initial["action_calls"],
        "verify_attempt": verify_attempt,
    }


def action_verification_response(request, settings=None):
    """Return the next deterministic verification or final response, if applicable.

    Args:
        request (ChatRequest): Active Home Assistant chat request.
        settings (Settings | None): Service settings controlling verification.

    Returns:
        dict[str, Any] | None: Verification/failure/final response decision, or None.
    """
    initial = _initial_action(request.messages)
    completed = _completed_round(request.messages)
    if initial is None or completed is None:
        return None
    calls, results = completed
    names = {_tool_name(call) for call in calls}
    verify_limit = _verify_attempt_limit(settings)

    if names <= set(_ACTION_TO_STATE):
        summary = _action_result_summary(results)
        if summary is None:
            return None
        if summary["failures"]:
            return {
                "kind": "action_failed",
                "response": _action_failure_text(summary["failures"]),
                "failures": summary["failures"],
                "successes": summary["successes"],
                "action_calls": initial["action_calls"],
            }
        if not summary["successes"]:
            return None
        if verify_limit <= 0 or not _live_tool_available(request):
            return None
        return _verify_decision(initial, 1)

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
    verify_attempt = _live_calls_since_user(request.messages)
    if not mismatches:
        return {
            "kind": "verified",
            "response": t("ha_action_verification.212"),
            "expected_state": expected,
            "entities": entities,
            "action_calls": initial["action_calls"],
            "verify_attempt": verify_attempt,
        }

    if verify_attempt < verify_limit and _live_tool_available(request):
        decision = _verify_decision(initial, verify_attempt + 1)
        decision["mismatches"] = mismatches
        return decision

    names_text = ", ".join(entity["name"] for entity in mismatches)
    return {
        "kind": "state_unconfirmed",
        "response": f"{t('ha_action.done')} {t('ha_action_verification.236', names_text=names_text)}",
        "expected_state": expected,
        "mismatches": mismatches,
        "action_calls": initial["action_calls"],
        "verify_attempt": verify_attempt,
    }


def _needs_live_tool(messages, settings=None):
    """Check whether an action turn still needs a live-state verification tool.

    Args:
        messages (list[dict[str, Any]]): Ordered conversation messages.
        settings (Settings | None): Service settings controlling verification.

    Returns:
        bool: Whether GetLiveContext must remain available for another state read.
    """
    verify_limit = _verify_attempt_limit(settings)
    if verify_limit <= 0:
        return False
    completed = _completed_round(messages)
    initial = _initial_action(messages)
    if initial is None or completed is None:
        return False
    calls, results = completed
    names = {_tool_name(call) for call in calls}
    if names <= set(_ACTION_TO_STATE):
        return _successful_action_results(results)
    if names != {_LIVE_TOOL}:
        return False
    live = _live_result(results)
    if live is None:
        return False
    entities = _live_entities(live)
    if not entities:
        return False
    mismatches = [entity for entity in entities if entity["state"] != initial["expected_state"]]
    return bool(mismatches) and _live_calls_since_user(messages) < verify_limit


def prepare_request(backend, request, next_stage):
    """Retain GetLiveContext while deterministic action verification is active.

    Args:
        backend (ChatBackend): Backend providing service settings.
        request (ChatRequest): Active Home Assistant chat request.
        next_stage (Callable): Next request-preparation stage.

    Returns:
        ChatRequest: Prepared request with GetLiveContext retained when needed.
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
    if live_tool is not None and _needs_live_tool(prepared.messages, backend.settings):
        tools = list(prepared.tools or [])
        if not any(tool.get("function", {}).get("name") == _LIVE_TOOL for tool in tools):
            tools.append(live_tool)
            prepared = prepared.model_copy(update={"tools": tools})
            prepared._request_id = getattr(request, "_request_id", "-")
    return prepared


def _speech(result):
    """Read non-empty plain speech from a Home Assistant action result.

    Args:
        result (dict[str, Any]): Home Assistant action result payload.

    Returns:
        str | None: Plain speech acknowledgement when supplied.
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
        request (ChatRequest): Active Home Assistant chat request.

    Returns:
        dict[str, Any] | None: Deterministic acknowledgement data, or None.
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

    text = " ".join(dict.fromkeys(speeches)) if speeches else t("ha_action.done")
    return {
        "text": text,
        "tool_names": [pending[call["id"]] for call in calls],
        "successes": successes,
        "tool_results": [results[call["id"]] for call in calls],
    }
