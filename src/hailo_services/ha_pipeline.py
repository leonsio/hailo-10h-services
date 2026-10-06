"""Single boundary for HA-only routing, localization and safe fast actions."""

from __future__ import annotations

import json
import logging
import re
import uuid
from functools import wraps

from jsonschema import ValidationError, validate

from .ha_prompt_compiler import _TOOL_CAPABILITY
from .ha_state_routing import _entries, _query_domain
from .i18n import detect_language, lexicon, normalize_matching, t, using_language
from .tool_retrieval import latest_user_text

_LOG = logging.getLogger(__name__)
_HA_TOOLS = set(_TOOL_CAPABILITY) | {"intent__HassTurnOn", "intent__HassTurnOff"}
_DIRECT_ATTRIBUTES = (
    "_direct_ha_response",
    "_direct_ha_state_response",
    "_direct_weather_response",
)


def is_home_assistant_request(request):
    """Require identifiable HA tools/history, not a device word in user text."""
    if not getattr(request, "_ha_assist", False):
        return False
    names = {tool.get("function", {}).get("name") for tool in request.tools or []}
    for message in request.messages:
        names.update(
            call.get("function", {}).get("name") for call in message.get("tool_calls") or []
        )
    return bool(names & _HA_TOOLS)


def needs_inference(request):
    return not any(getattr(request, name, None) is not None for name in _DIRECT_ATTRIBUTES)


def task_history(request, *, encoder=None, embedding_cache=None):
    """Keep active tool dependencies; retire completed HA turns at the HA boundary."""
    from .ha_routing import assess_ha_relevance

    systems = [m for m in request.messages if m.get("role") == "system"]
    turns = []
    for message in request.messages:
        if message.get("role") == "system":
            continue
        if message.get("role") == "user" or not turns:
            turns.append([])
        turns[-1].append(message)
    if not turns:
        return request

    def relevant(turn, semantic=False):
        return assess_ha_relevance(
            systems + [turn[0]], request.tools,
            encoder=encoder if semantic else None, embedding_cache=embedding_cache,
        )["relevant"]

    active = turns[-1]
    # A result is part of the current user turn, not a new independent request.
    active_tools = any(m.get("role") == "tool" or m.get("tool_calls") for m in active)
    house = active_tools or relevant(active, True)
    retained = []
    if not house:
        for turn in turns[:-1]:
            if not any(m.get("role") == "tool" or m.get("tool_calls") for m in turn) and not relevant(turn):
                retained.extend(turn)
    messages = systems + retained + active
    prepared = request.model_copy(update={"messages": messages})
    request._metrics["ha_history"] = {
        "policy": "current_ha_turn" if house else "general_conversation",
        "messages_before": len(request.messages), "messages_after": len(messages),
    }
    return prepared


def _target(request, domain, query):
    entities = [e for e in _entries(request.messages) if e["domain"] == domain]

    def contains(value):
        return bool(value) and (" " + normalize_matching(value) + " ") in (" " + query + " ")

    areas = {e["area"] for e in entities if e["area"] and contains(e["area"])}
    if len(areas) > 1:
        return None
    if areas:
        area = next(iter(areas))
        members = [e for e in entities if e["area"] == area]
    else:
        area, members = None, entities
    explicit = [
        e
        for e in members
        if contains(e["name"])
        and normalize_matching(e["name"]) not in lexicon("ha_pipeline.words.46.99")
    ]
    unique = {(e["name"], e["area"]) for e in explicit}
    if len(unique) == 1:
        result = {"name": explicit[0]["name"], "domain": [domain]}
        if area:
            result["area"] = area
        return result
    if unique or area is None:
        return None
    return {"area": area, "domain": [domain]}


def direct_numeric_action(request):
    """Absolute brightness/cover position only; uncertain/composite requests defer."""
    if request.tool_choice == "none" or request.messages[-1].get("role") != "user":
        return None
    query = normalize_matching(latest_user_text(request.messages))
    # normalize_matching removes %, so inspect the original value separately.
    original = latest_user_text(request.messages)
    values = re.findall(lexicon("ha_pipeline.match.65.24"), original, re.I)
    if not values:
        # An absolute, terminal value is meaningful only for light/cover targets.
        values = re.findall(r"\b(?:auf|to|на)\s+(\d{1,3})\s*[.!?]?$", original, re.I)
    if len(values) != 1 or len(re.findall(r"\d+(?:[.,]\d+)?", original)) != 1:
        return None
    if not re.search(lexicon("ha_pipeline.pattern.68.21"), query):
        return None
    # No relative changes, conditional instructions or combined actions.
    if re.search(lexicon("ha_pipeline.pattern.71.17"), query):
        return None
    domain = _query_domain(query)
    tool_name, prop = {
        "light": ("light__HassLightSet", "brightness"),
        "cover": ("intent__HassSetPosition", "position"),
    }.get(domain, (None, None))
    if tool_name is None or not 0 <= int(values[0]) <= 100:
        return None
    if (
        isinstance(request.tool_choice, dict)
        and request.tool_choice["function"]["name"] != tool_name
    ):
        return None
    tool = next(
        (t for t in request.tools or [] if t.get("function", {}).get("name") == tool_name), None
    )
    arguments = _target(request, domain, query)
    if tool is None or arguments is None:
        return None
    schema = tool["function"].get("parameters", {})
    properties = schema.get("properties", {})
    arguments = {k: v for k, v in arguments.items() if k in properties}
    # A target must remain after restricting to the actual client's schema.
    if not ({"name", "area"} & arguments.keys()) or prop not in properties:
        return None
    arguments[prop] = int(values[0])
    try:
        validate(arguments, schema)
    except ValidationError:
        return None
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _valid_direct(request, response):
    tools = {tool["function"]["name"]: tool["function"] for tool in request.tools or []}
    query = normalize_matching(latest_user_text(request.messages))
    for call in response.get("tool_calls", []):
        name = call["function"]["name"]
        if name not in tools:
            return False
        if name != "homeassistant__GetLiveContext" and re.search(
            lexicon("ha_pipeline.pattern.71.17"), query
        ):
            return False
        try:
            validate(json.loads(call["function"]["arguments"]), tools[name].get("parameters", {}))
        except (ValidationError, ValueError):
            return False
    return True


def install():
    from . import runtime

    backend = runtime.HailoBackend
    if getattr(backend, "_ha_pipeline_installed", False):
        return
    original_select = backend.select_tools
    original_chat = runtime.LiteRTLMBackend.chat

    @wraps(original_select)
    def select(self, request):
        if not getattr(request, "_ha_assist", False):
            return request
        if request.tool_choice == "none":
            return request
        if not is_home_assistant_request(request):
            return request
        request = task_history(request, encoder=self.minilm,
                               embedding_cache=self._retrieval_embedding_cache)
        if self.settings.debug_log:
            _LOG.debug("event=ha_history request_id=%s json=%s", request._request_id,
                       json.dumps(request._metrics["ha_history"]))
        language = request.language or detect_language(
            latest_user_text(request.messages), self.settings.service_language
        )
        with using_language(language):
            from .ha_intents import deterministic_intent
            from .ha_request_plan import canonical_request, target_clarification, tool_failure

            request = canonical_request(request, self.settings)
            if self.settings.debug_log and getattr(request, "_ha_plan", None):
                _LOG.debug("event=ha_plan request_id=%s json=%s", request._request_id,
                           json.dumps(request._ha_plan, ensure_ascii=False))
            immediate = tool_failure(request) or target_clarification(request)
            if immediate is not None and not isinstance(request.tool_choice, dict):
                prepared = request.model_copy()
                object.__setattr__(prepared, "_direct_ha_response", immediate)
                object.__setattr__(prepared, "_response_language", language)
                return prepared

            unresolved = getattr(request, "_ha_plan", {}).get("target_resolution") == "llm"
            if unresolved:
                direct, intent_trace = None, {"source": "catalogue_scores", "reason": "ambiguous"}
            else:
                direct, intent_trace = deterministic_intent(request, self.settings, language)
            if direct is None and not unresolved:
                direct = direct_numeric_action(request)
                if direct is not None:
                    intent_trace.update(source="direct_numeric", reason="validated",
                                        arguments=json.loads(direct["tool_calls"][0]["function"]["arguments"]))
            request._metrics["ha_intent"] = intent_trace
            if direct is not None and getattr(request, "_ha_plan", None):
                request._ha_plan["action"] = (request._ha_plan.get("action") or
                    direct["tool_calls"][0]["function"]["name"])
            if self.settings.debug_log:
                _LOG.debug("event=ha_intent request_id=%s json=%s", request._request_id,
                           json.dumps(intent_trace, ensure_ascii=False))
            if direct is not None:
                prepared = request.model_copy()
                object.__setattr__(prepared, "_direct_ha_response", direct)
                _LOG.info(
                    "ha_route request_id=%s route=direct_hassil skipped_gemma=true",
                    request._request_id,
                )
            elif unresolved:
                from .ha_prompt_compiler import compile_ha_prompt

                prepared, _ = compile_ha_prompt(request, request)
            elif isinstance(request.tool_choice, dict):
                # A forced tool is authoritative; no competing direct read/action.
                prepared = request.model_copy()
            else:
                prepared = original_select(self, request)
            direct_response = getattr(prepared, "_direct_ha_response", None)
            if direct_response is not None and not _valid_direct(request, direct_response):
                from .ha_prompt_compiler import compile_ha_prompt

                prepared, _ = compile_ha_prompt(request, request)
            if needs_inference(prepared):
                messages = [dict(message) for message in prepared.messages]
                if messages[0].get("role") == "system" and isinstance(
                    messages[0].get("content"), str
                ):
                    messages[0]["content"] += "\n" + t("prompt.reply_language")
                else:
                    messages.insert(0, {"role": "system", "content": t("prompt.reply_language")})
                prepared = prepared.model_copy(update={"messages": messages})
            if getattr(request, "_ha_plan", None):
                object.__setattr__(prepared, "_ha_plan", request._ha_plan)
            object.__setattr__(prepared, "_response_language", language)
            object.__setattr__(prepared, "_ha_request", True)
            return prepared

    @wraps(original_chat)
    def chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        if not getattr(request, "_ha_assist", False):
            return self.plain_chat(request, emit, cancelled, True)
        with using_language(getattr(request, "_response_language", request.language or "de")):
            # The successful-action shortcut also needs HA provenance.
            return original_chat(self, request, emit, cancelled, True)

    backend.select_tools = select
    runtime.LiteRTLMBackend.chat = chat
    backend._ha_pipeline_installed = True
