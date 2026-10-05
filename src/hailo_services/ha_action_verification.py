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
import time
import uuid
from functools import wraps

from .i18n import t

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


def _json_object(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _arguments(call):
    function = call.get("function", {}) if isinstance(call, dict) else {}
    return _json_object(function.get("arguments")) or {}


def _tool_name(call):
    return call.get("function", {}).get("name") if isinstance(call, dict) else None


def _latest_user_index(messages):
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return 0


def _completed_round(messages):
    """Return the final assistant tool-call round and its tool result messages."""
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
    if len(results) != 1:
        return None
    payload = _json_object(results[0].get("content"))
    if payload is None or payload.get("success") is not True:
        return None
    result = payload.get("result")
    return result if isinstance(result, str) else None


def _live_entities(text):
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
            entities.append({
                "name": name.group(1).strip(),
                "domain": domain.group(1).strip(),
                "state": state.group(1).strip().casefold(),
                "area": area.group(1).strip() if area else "",
            })
    return entities


def _action_calls_since_user(messages):
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
    calls = _action_calls_since_user(messages)
    if not calls:
        return None
    first = calls[0]
    name = _tool_name(first)
    args = _arguments(first)
    domains = args.get("domain")
    if isinstance(domains, str):
        domains = [domains]
    if not isinstance(domains, list) or not domains or any(
        not isinstance(domain, str) for domain in domains
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
    return {
        "id": "call_" + uuid.uuid4().hex,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def action_verification_response(request):
    """Return the next deterministic verification/retry response, if applicable."""
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
            key: value for key, value in initial["arguments"].items()
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
            "response": t('ha_action_verification.212'),
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
        "response": t('ha_action_verification.236' , names_text=names_text),
        "expected_state": expected,
        "mismatches": mismatches,
        "attempts": initial["attempts"],
    }


def _needs_live_tool(messages):
    completed = _completed_round(messages)
    initial = _initial_action(messages)
    if initial is None or completed is None:
        return False
    calls, results = completed
    return (
        {_tool_name(call) for call in calls} <= set(_ACTION_TO_STATE)
        and _successful_action_results(results)
    )


def install():
    """Install deterministic on/off verification after the other HA routing hooks."""
    from . import runtime

    backend_cls = runtime.HailoBackend
    litert_cls = runtime.LiteRTLMBackend
    if getattr(backend_cls, "_ha_action_verification_installed", False):
        return

    original_select_tools = backend_cls.select_tools
    original_chat = litert_cls.chat

    @wraps(original_select_tools)
    def select_tools(self, request):
        live_tool = next((
            tool for tool in (request.tools or [])
            if tool.get("function", {}).get("name") == _LIVE_TOOL
        ), None)
        prepared = original_select_tools(self, request)
        if live_tool is not None and _needs_live_tool(prepared.messages):
            tools = list(prepared.tools or [])
            if not any(tool.get("function", {}).get("name") == _LIVE_TOOL for tool in tools):
                tools.append(live_tool)
                prepared = prepared.model_copy(update={"tools": tools})
                prepared._request_id = getattr(request, "_request_id", "-")
        return prepared

    @wraps(original_chat)
    def chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        decision = action_verification_response(request)
        if decision is not None:
            request_id = getattr(request, "_request_id", "-")
            response = decision["response"]
            settle_ms = 0
            if decision["kind"] == "verify":
                time.sleep(_VERIFY_SETTLE_SECONDS)
                settle_ms = round(_VERIFY_SETTLE_SECONDS * 1000)
            _LOG.info(
                "ha_action_verification request_id=%s kind=%s attempts=%d "
                "settle_ms=%d skipped_gemma=true",
                request_id,
                decision["kind"],
                decision.get("attempts", 0),
                settle_ms,
            )
            if getattr(self, "debug_log", False):
                debug_decision = {**decision, "settle_ms": settle_ms}
                _LOG.debug(
                    "event=ha_action_verification request_id=%s json=%s",
                    request_id,
                    json.dumps(
                        debug_decision,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ),
                )
            if isinstance(response, str) and emit:
                emit(response)
            return response
        return original_chat(self, request, emit, cancelled, tools_prepared)

    backend_cls.select_tools = select_tools
    litert_cls.chat = chat
    backend_cls._ha_action_verification_installed = True
