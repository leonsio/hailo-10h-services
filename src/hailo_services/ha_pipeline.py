"""Single boundary for HA-only routing, localization and safe fast actions."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

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
    """Require identifiable HA tools/history, not a device word in user text.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        Any: Result as described by the operation.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if not getattr(request, "_ha_assist", False):
        return False
    names = {tool.get("function", {}).get("name") for tool in request.tools or []}
    for message in request.messages:
        names.update(
            call.get("function", {}).get("name") for call in message.get("tool_calls") or []
        )
    return bool(names & _HA_TOOLS)


def needs_inference(request):
    """Check whether preparation already produced a deterministic response.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        bool: True when a generative backend is still required.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return not any(getattr(request, name, None) is not None for name in _DIRECT_ATTRIBUTES)


def task_history(request, *, encoder=None, embedding_cache=None):
    """Keep active tool dependencies; retire completed HA turns at the HA boundary.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        encoder (MiniLM | None): Optional resident semantic encoder; None uses lexical matching.
        embedding_cache (dict[str, np.ndarray] | None): Bounded cache for static retrieval embeddings.

    Returns:
        Any: Result as described by the operation.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
        """Check whether a history turn depends on Home Assistant context.

        Args:
            turn (list[dict[str, Any]]): One user turn with its assistant and tool messages.
            semantic (bool): Whether semantic relevance fallback is allowed.

        Returns:
            bool: Whether the turn needs HA entity or tool context.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return assess_ha_relevance(
            systems + [turn[0]],
            request.tools,
            encoder=encoder if semantic else None,
            embedding_cache=embedding_cache,
        )["relevant"]

    active = turns[-1]
    # A result is part of the current user turn, not a new independent request.
    active_tools = any(m.get("role") == "tool" or m.get("tool_calls") for m in active)
    house = active_tools or relevant(active, True)
    retained = []
    if not house:
        for turn in turns[:-1]:
            if not any(
                m.get("role") == "tool" or m.get("tool_calls") for m in turn
            ) and not relevant(turn):
                retained.extend(turn)
    messages = systems + retained + active
    prepared = request.model_copy(update={"messages": messages})
    request._metrics["ha_history"] = {
        "policy": "current_ha_turn" if house else "general_conversation",
        "messages_before": len(request.messages),
        "messages_after": len(messages),
    }
    return prepared


def _target(request, domain, query):
    """Resolve one explicit entity or area target in the requested domain.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        domain (str | None): Resolved Home Assistant domain, or None.
        query (str): User question or normalized utterance to match.

    Returns:
        dict[str, Any] | None: Safe name/area/domain arguments, or None for ambiguity.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    entities = [e for e in _entries(request.messages) if e["domain"] == domain]

    def contains(value):
        """Match a whole normalized catalogue phrase in the current query.

        Args:
            value (Any): Input value inspected or normalized by this helper.

        Returns:
            bool: Whether the phrase occurs with word boundaries.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
    """Absolute brightness/cover position only; uncertain/composite requests defer.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        dict[str, Any] | None: Assistant brightness/position call, or None when ambiguous or unsupported.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
    """Validate direct calls against client schemas and composite-action guards.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        response (Any): Native output or decoded assistant response to normalize.

    Returns:
        bool: Whether every deterministic call is safe for this request.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    from .ha_request_plan import validate_action

    if not isinstance(response, dict):
        return True
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
            args = json.loads(call["function"]["arguments"])
            validate(args, tools[name].get("parameters", {}))
            if not validate_action(request, name, args):
                return False
        except (ValidationError, ValueError):
            return False
    return True


def _prepare_boundary(backend, request, next_stage):
    """Apply HA provenance, history, language and deterministic intent policy.

    Args:
        backend (ChatBackend): Resident backend used for generation or context preparation.
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        next_stage (Callable[[ChatRequest], ChatRequest]): Fallback preparation stage when this stage does not finish routing.

    Returns:
        ChatRequest: Prepared request with language, plan and optional direct result.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if not getattr(request, "_ha_assist", False):
        return request
    if request.tool_choice == "none":
        return request
    if not is_home_assistant_request(request):
        return request
    history_started = time.perf_counter()
    request = task_history(
        request, encoder=backend.minilm, embedding_cache=backend._retrieval_embedding_cache
    )
    request._metrics.setdefault("ha_stages_ms", {})["history"] = (
        time.perf_counter() - history_started
    ) * 1000
    if backend.settings.debug_log:
        _LOG.debug(
            "event=ha_history request_id=%s json=%s",
            request._request_id,
            json.dumps(request._metrics["ha_history"]),
        )
    language = request.language or detect_language(
        latest_user_text(request.messages), backend.settings.service_language
    )
    with using_language(language):
        from .ha_intents import deterministic_intent
        from .ha_request_plan import canonical_request, target_clarification, tool_failure

        phase_started = time.perf_counter()
        request = canonical_request(request, backend.settings)
        request._metrics.setdefault("ha_stages_ms", {})["canonicalize"] = (
            time.perf_counter() - phase_started
        ) * 1000
        phase_started = time.perf_counter()
        if backend.settings.debug_log and getattr(request, "_ha_plan", None):
            _LOG.debug(
                "event=ha_plan request_id=%s json=%s",
                request._request_id,
                json.dumps(request._ha_plan, ensure_ascii=False),
            )
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
            direct, intent_trace = deterministic_intent(request, backend.settings, language)
        if direct is None and not unresolved:
            direct = direct_numeric_action(request)
            if direct is not None:
                intent_trace.update(
                    source="direct_numeric",
                    reason="validated",
                    arguments=json.loads(direct["tool_calls"][0]["function"]["arguments"]),
                )
        request._metrics["ha_stages_ms"]["intent_recognition"] = (
            time.perf_counter() - phase_started
        ) * 1000
        request._metrics["ha_intent"] = intent_trace
        phase_started = time.perf_counter()
        if direct is not None and getattr(request, "_ha_plan", None):
            request._ha_plan["action"] = (
                request._ha_plan.get("action") or direct["tool_calls"][0]["function"]["name"]
            )
        if backend.settings.debug_log:
            _LOG.debug(
                "event=ha_intent request_id=%s json=%s",
                request._request_id,
                json.dumps(intent_trace, ensure_ascii=False),
            )
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
            prepared = next_stage(request)
        direct_response = getattr(prepared, "_direct_ha_response", None)
        if direct_response is not None and not _valid_direct(request, direct_response):
            from .ha_prompt_compiler import compile_ha_prompt

            prepared, _ = compile_ha_prompt(request, request)
        request._metrics["ha_stages_ms"]["context_and_prompt"] = (
            time.perf_counter() - phase_started
        ) * 1000
        if needs_inference(prepared):
            messages = [dict(message) for message in prepared.messages]
            if messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
                messages[0]["content"] += "\n" + t("prompt.reply_language")
            else:
                messages.insert(0, {"role": "system", "content": t("prompt.reply_language")})
            prepared = prepared.model_copy(update={"messages": messages})
        if getattr(request, "_ha_plan", None):
            object.__setattr__(prepared, "_ha_plan", request._ha_plan)
        object.__setattr__(prepared, "_response_language", language)
        object.__setattr__(prepared, "_ha_request", True)
        return prepared


def prepare_request(backend, request):
    """Apply this HA preparation stage and invoke its fallback when needed.

    Args:
        backend (ChatBackend): Resident backend used for generation or context preparation.
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        ChatRequest: Prepared or unchanged request, possibly carrying a direct response.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    from functools import partial

    from . import (
        ha_action_verification,
        ha_prompt_compiler,
        ha_routing,
        ha_state_routing,
        ha_weather_routing,
    )

    stage = backend.retrieve_context
    for prepare in (
        ha_routing.prepare_request,
        ha_state_routing.prepare_request,
        ha_action_verification.prepare_request,
        ha_prompt_compiler.prepare_request,
        ha_weather_routing.prepare_request,
    ):
        stage = partial(prepare, backend, next_stage=stage)
    return _prepare_boundary(backend, request, stage)
