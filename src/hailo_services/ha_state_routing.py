"""Fast deterministic routing for simple Home Assistant state questions.

The Home Assistant client already exposes current state through GetLiveContext.
For unambiguous read-only questions this module therefore avoids Gemma twice:
first for tool selection, then again for formatting the tool result.  Complex or
ambiguous questions still fall back to the compact LLM path.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from functools import lru_cache

from .i18n import (
    catalogue,
    current_language,
    detect_language,
    labels,
    lexicon,
    t,
)
from .i18n import normalize_matching as _normalized
from .json_utils import json_object as _json_object
from .tool_retrieval import _static_context_parts, latest_user_text

_LOG = logging.getLogger(__name__)
_LIVE_TOOL = "homeassistant__GetLiveContext"
_FINAL_SYSTEM_KEY = "ha_state_routing.22"
_DOMAIN_WORDS = lexicon("ha_state_routing._DOMAIN_WORDS")
_STATE_ALIASES = lexicon("ha_state_routing._STATE_ALIASES")


def _contains(text: str, phrase: str) -> bool:
    """Match a non-empty normalized phrase with word boundaries.

    Args:
        text (str): Text to parse, normalize, match or render.
        phrase (str): Normalized phrase to match as complete words.

    Returns:
        bool: Whether the phrase appears as whole words.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return bool(phrase) and f" {phrase} " in f" {text} "


def _has_action_verb(text: str) -> bool:
    """Recognize explicit control verbs in a normalized utterance.

    Args:
        text (str): Text to parse, normalize, match or render.

    Returns:
        bool: Whether the request expresses a device action.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return bool(
        re.search(
            lexicon("ha_state_routing.pattern.37.8"),
            _normalized(text),
        )
    )


def _expected_state(text: str) -> str | None:
    """Read an explicitly requested on/off state from the query.

    Args:
        text (str): Text to parse, normalize, match or render.

    Returns:
        str | None: Canonical state, or None when not specified.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    normalized = _normalized(text)
    for word, state in _STATE_ALIASES.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            return state
    return None


def _query_kind(text: str) -> str | None:
    """Return the deterministic read-only question kind, if recognized.

    Args:
        text (str): Text to parse, normalize, match or render.

    Returns:
        str | None: Recognized status, list, count, all, temperature or humidity category.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    normalized = _normalized(text)
    if not normalized or _has_action_verb(normalized):
        return None
    if re.search(lexicon("ha_state_routing.pattern.57.17"), normalized) and _expected_state(
        normalized
    ):
        return "count"
    if re.search(lexicon("ha_state_routing.pattern.59.17"), normalized) and _expected_state(
        normalized
    ):
        return "list"
    if re.search(lexicon("ha_state_routing.pattern.61.17"), normalized):
        return "temperature"
    if re.search(lexicon("ha_state_routing.pattern.63.17"), normalized):
        return "humidity"
    if re.search(lexicon("ha_state_routing.match.63.17"), normalized):
        return "status"
    if _expected_state(normalized) and (
        re.search(lexicon("ha_state_routing.match.66.18"), normalized) or _query_domain(normalized)
    ):
        return (
            "all" if re.search(lexicon("ha_state_routing.match.68.34"), normalized) else "boolean"
        )
    return None


def _is_state_question(text: str) -> bool:
    """Recognize a read-only state question without action intent.

    Args:
        text (str): Text to parse, normalize, match or render.

    Returns:
        bool: Whether deterministic live-state routing applies.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return _query_kind(text) is not None


def _entries(messages) -> list[dict[str, str]]:
    """Parse static entity entries with a bounded cache for small system contexts.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        list[dict[str, str]]: Entities from the generated HA catalogue.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    systems = tuple(
        m["content"]
        for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    )
    parsed = (
        _cached_entries(systems)
        if sum(map(len, systems)) <= 65536
        else _cached_entries.__wrapped__(systems)
    )
    return [dict(item) for item in parsed]


@lru_cache(maxsize=64)
def _cached_entries(systems):
    """Cache parsed name, domain and area fields from system messages.

    Args:
        systems (tuple[str, ...]): Static system texts used as an immutable cache key.

    Returns:
        list[dict[str, str]]: Parsed static entity entries.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    entities: list[dict[str, str]] = []
    for content in systems:
        message = {"role": "system", "content": content}
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        parts = _static_context_parts(message["content"])
        if parts is None:
            continue
        for entry in parts[1]:
            name = re.search(r"(?m)^- names:\s*(.+)$", entry)
            domain = re.search(r"(?m)^\s+domain:\s*(.+)$", entry)
            area = re.search(r"(?m)^\s+areas:\s*(.+)$", entry)
            if name and domain:
                entities.append(
                    {
                        "name": name.group(1).strip(),
                        "domain": domain.group(1).strip(),
                        "area": area.group(1).strip() if area else "",
                    }
                )
    return entities


def _live_tool(tools):
    """Find the client-provided GetLiveContext function schema.

    Args:
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.

    Returns:
        dict[str, Any] | None: Live-context tool, or None when unavailable.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    for tool in tools or []:
        if tool.get("function", {}).get("name") == _LIVE_TOOL:
            return tool
    return None


def _query_domain(query: str) -> str | None:
    """Infer exactly one domain from localized query vocabulary.

    Args:
        query (str): User question or normalized utterance to match.

    Returns:
        str | None: Domain hint, or None when absent or conflicting.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    words = set(_normalized(query).split())
    matches = [domain for domain, aliases in _DOMAIN_WORDS.items() if words & aliases]
    return matches[0] if len(matches) == 1 else None


def _measurement_members(members, kind: str):
    """Filter static members to plausible temperature or humidity sources.

    Args:
        members (list[dict[str, Any]]): Candidate entities in the selected area or domain.
        kind (str): Model role, measurement category or environment query kind.

    Returns:
        list[dict[str, str]]: Members whose names and domains match the measurement kind.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if kind == "temperature":
        hints = lexicon("ha_state_routing.words.112.16")
        preferred_domains = {"sensor", "climate"}
    elif kind == "humidity":
        hints = lexicon("ha_state_routing.words.115.16")
        preferred_domains = {"sensor", "climate"}
    else:
        return []
    matches = [
        entity
        for entity in members
        if entity["domain"] in preferred_domains
        and set(_normalized(entity["name"]).split()) & hints
    ]
    if matches:
        return matches
    return [entity for entity in members if entity["domain"] in preferred_domains]


def _target_arguments(messages, query: str):
    """Resolve area before global entity names and keep unknown areas direct.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.
        query (str): User question or normalized utterance to match.

    Returns:
        dict[str, Any] | None: Safe live-context filters, or None when the target is ambiguous.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    entities = _entries(messages)
    if not entities:
        return None

    normalized_query = _normalized(query)
    domain_hint = _query_domain(query)
    kind = _query_kind(query)

    # An explicitly mentioned area has precedence over generic entity names
    # elsewhere in the house.  Production installations often contain many
    # entities named simply "Licht", which must not make
    # "Ist das Licht im Wohnzimmer an?" ambiguous.
    area_names = sorted(
        {entity["area"] for entity in entities if entity["area"]},
        key=len,
        reverse=True,
    )
    matching_areas = [area for area in area_names if _contains(normalized_query, _normalized(area))]
    if len(matching_areas) == 1:
        area = matching_areas[0]
        area_entities = [entity for entity in entities if entity["area"] == area]

        explicit = [
            entity
            for entity in area_entities
            if _contains(normalized_query, _normalized(entity["name"]))
            and (domain_hint is None or entity["domain"] == domain_hint)
        ]
        explicit_keys = {(item["name"], item["domain"], item["area"]) for item in explicit}
        if len(explicit_keys) == 1:
            entity = explicit[0]
            return {"name": entity["name"], "domain": [entity["domain"]]}
        if explicit:
            return None

        members = area_entities
        if domain_hint is not None:
            # Querying HA directly remains safe even if the static context has
            # no member of that domain; HA then returns a precise no-match
            # result which we format without involving Gemma.
            return {"area": area, "domain": [domain_hint]}
        if kind in {"temperature", "humidity"}:
            members = _measurement_members(members, kind)
            unique = {(item["name"], item["domain"]) for item in members}
            if len(unique) == 1:
                entity = members[0]
                return {"name": entity["name"], "domain": [entity["domain"]]}
            return None

        domains = {entity["domain"] for entity in members}
        if len(domains) == 1:
            return {"area": area, "domain": [next(iter(domains))]}
        if kind in {"status", "list", "count", "all"}:
            return {"area": area}
        return None
    if matching_areas:
        return None

    # If the sentence explicitly says "im/in der <room>" but that room is not
    # present in the static area list, ask Home Assistant directly instead of
    # letting Gemma guess a similarly named entity (for example Gästezimmer
    # for Schlafzimmer).
    location = _location_phrase(query)
    if location and domain_hint is not None:
        normalized_location = _normalized(location)
        if normalized_location in _HOME_LOCATIONS:
            return {"domain": [domain_hint]}
        return {"area": location, "domain": [domain_hint]}

    # Without an area, retain the original exact-name behaviour.
    explicit = [
        entity
        for entity in entities
        if _contains(normalized_query, _normalized(entity["name"]))
        and (domain_hint is None or entity["domain"] == domain_hint)
    ]
    explicit_keys = {(item["name"], item["domain"], item["area"]) for item in explicit}
    if len(explicit_keys) == 1:
        entity = explicit[0]
        return {"name": entity["name"], "domain": [entity["domain"]]}
    if explicit:
        return None

    if domain_hint is not None and kind in {"list", "count", "all"}:
        return {"domain": [domain_hint]}
    return None


def direct_live_context_response(request):
    """Return a direct GetLiveContext call for an unambiguous state question.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        dict[str, Any] | None: Assistant live-context call, or None when no safe shortcut applies.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    tool = _live_tool(request.tools)
    query = latest_user_text(request.messages).strip()
    if tool is None or not _is_state_question(query):
        return None
    arguments = _target_arguments(request.messages, query)
    if arguments is None:
        return None
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": _LIVE_TOOL,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


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


def _live_followup(messages):
    """Return question, calls and results for the latest pure LiveContext round.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        tuple[str, list[dict], list[Any]] | None: User question, live calls and matching result contents.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    start = _latest_user_index(messages)
    question = latest_user_text(messages).strip()
    if not _is_state_question(question):
        return None
    calls = []
    results = []
    for message in messages[start + 1 :]:
        for call in message.get("tool_calls", []) or []:
            if call.get("function", {}).get("name") != _LIVE_TOOL:
                return None
            calls.append(call)
        if message.get("role") == "tool":
            results.append((message.get("tool_call_id"), message.get("content", "")))
    call_ids = {call.get("id") for call in calls}
    if not calls or {identifier for identifier, _ in results} != call_ids:
        return None
    return question, calls, [content for _, content in results]


def _block_value(block: str, key: str):
    """Read one field from an HA live-context text block.

    Args:
        block (str): One live-context entity block.
        key (str): Resource or field identifier to look up.

    Returns:
        str: Unquoted field value, or an empty string when absent.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    match = re.search(
        rf"(?mi)^\s+{re.escape(key)}:\s*['\"]?([^'\"\n]+)['\"]?\s*$",
        block,
    )
    return match.group(1).strip() if match else None


def _entities_from_live_text(text: str):
    """Parse live-context text and selected measurement attributes.

    Args:
        text (str): Text to parse, normalize, match or render.

    Returns:
        list[dict[str, Any]]: Decoded live entities.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    entities = []
    blocks = re.split(r"(?m)(?=^- names:\s*)", text or "")
    for block in blocks:
        if not block.startswith("- names:"):
            continue
        name = re.search(r"(?m)^- names:\s*(.+)$", block)
        domain = _block_value(block, "domain")
        state = _block_value(block, "state")
        if not name or state is None:
            continue
        attributes = {}
        for key in (
            "current_temperature",
            "temperature",
            "current_humidity",
            "humidity",
            "unit_of_measurement",
            "unit",
            "device_class",
        ):
            value = _block_value(block, key)
            if value is not None:
                attributes[key] = value
        entities.append(
            {
                "name": name.group(1).strip(),
                "domain": domain or "",
                "state": state.casefold(),
                "area": _block_value(block, "areas") or "",
                "attributes": attributes,
            }
        )
    return entities


def _entities_from_mapping(payload: dict, call):
    """Parse structured live state while excluding error metadata.

    Args:
        payload (dict): Decoded protocol data or structured diagnostic payload.
        call (dict[str, Any]): Historical function call and its JSON arguments.

    Returns:
        list[dict[str, Any]]: Entity records extracted from the payload.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    entities = []
    payload = {key: value for key, value in payload.items() if key not in _ERROR_KEYS}
    arguments = _json_object(call.get("function", {}).get("arguments")) or {}
    domains = arguments.get("domain")
    if isinstance(domains, list) and len(domains) == 1:
        domain = domains[0]
    elif isinstance(domains, str):
        domain = domains
    else:
        domain = ""
    for name, value in payload.items():
        if name in {"success", "result", "response_type", "speech", "data"}:
            continue
        if isinstance(value, str):
            entities.append(
                {
                    "name": str(name),
                    "domain": domain,
                    "state": value.casefold(),
                    "area": arguments.get("area", ""),
                    "attributes": {},
                }
            )
        elif isinstance(value, dict) and "state" in value:
            attributes = (
                value.get("attributes") if isinstance(value.get("attributes"), dict) else {}
            )
            entities.append(
                {
                    "name": str(name),
                    "domain": str(value.get("domain") or domain),
                    "state": str(value["state"]).casefold(),
                    "area": str(value.get("area") or arguments.get("area", "")),
                    "attributes": attributes,
                }
            )
    return entities


def _live_entities(results, calls):
    """Decode HA live-state entities from tool result payloads.

    Args:
        results (list[Any]): Client tool results aligned to the active call round.
        calls (list[dict[str, Any]]): Assistant function calls from a tool round.

    Returns:
        list[dict[str, Any]]: Entities with names, domains, states and available attributes.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    entities = []
    call_by_id = {call.get("id"): call for call in calls}
    for index, content in enumerate(results):
        payload = _json_object(content)
        if payload is None:
            continue
        result = payload.get("result")
        if payload.get("success") is True and isinstance(result, str):
            entities.extend(_entities_from_live_text(result))
            continue
        call = calls[index] if index < len(calls) else next(iter(call_by_id.values()), {})
        entities.extend(_entities_from_mapping(payload, call))
    # Stable de-duplication avoids repeated aliases from verbose live context.
    unique = {}
    for entity in entities:
        key = (entity["name"], entity["domain"], entity["area"])
        unique[key] = entity
    return list(unique.values())


def _state_label(state: str) -> str:
    """Translate a canonical entity state for user-facing answers.

    Args:
        state (str): Canonical HA entity state.

    Returns:
        str: Localized state label.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return labels("states").get(str(state).casefold(), str(state))


def _domain_label(entities, count: int) -> str:
    """Choose a localized singular or plural domain label.

    Args:
        entities (list[dict[str, Any]]): Static or live HA entities relevant to the request.
        count (int): Entity count used for localized pluralization.

    Returns:
        str: Device-class label matching the entity count.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    domains = {entity.get("domain", "") for entity in entities if entity.get("domain")}
    if len(domains) == 1:
        singular, plural = labels("domains").get(
            next(iter(domains)), (t("device.singular"), t("device.plural"))
        )
        return singular if count == 1 else plural
    return t("device.singular") if count == 1 else t("device.plural")


def _format_number(value: str) -> str:
    """Format a numeric state compactly for a localized response.

    Args:
        value (str): Input value inspected or normalized by this helper.

    Returns:
        str: Formatted number or the original unparseable value.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    rendered = f"{number:g}"
    return rendered if current_language() == "en" else rendered.replace(".", ",")


def _measurement_value(entity, kind: str):
    """Read temperature or humidity from live attributes or sensor state.

    Args:
        entity (dict[str, Any]): HA entity containing name, domain, area and optional state/attributes.
        kind (str): Model role, measurement category or environment query kind.

    Returns:
        str | None: Formatted measurement with unit, or None when unavailable.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    attrs = entity.get("attributes", {})
    if kind == "temperature":
        value = attrs.get("current_temperature")
        if entity.get("domain") == "climate" and value in {None, ""}:
            return None
        if value is None:
            value = attrs.get("temperature")
    else:
        value = attrs.get("current_humidity")
        if value is None:
            value = attrs.get("humidity")
    if value is None:
        state = entity.get("state", "")
        try:
            float(state)
        except (TypeError, ValueError):
            return None
        value = state
    unit = attrs.get("unit_of_measurement") or attrs.get("unit") or ""
    return f"{_format_number(value)}{(' ' + str(unit)) if unit else ''}"


def deterministic_live_response(request):
    """Answer common LiveContext status questions directly, without Gemma.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        str | None: Localized live-state answer, or None when generative interpretation is needed.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    followup = _live_followup(request.messages)
    if followup is None:
        return None
    question, calls, results = followup
    friendly_error = _friendly_live_error(calls, results)
    if friendly_error is not None:
        return friendly_error
    entities = _live_entities(results, calls)
    if not entities:
        return None
    kind = _query_kind(question)
    expected = _expected_state(question)

    if kind in {"temperature", "humidity"}:
        values = [(entity["name"], _measurement_value(entity, kind)) for entity in entities]
        values = [(name, value) for name, value in values if value is not None]
        if not values:
            return None
        if len(values) == 1:
            noun = t("measurement." + kind)
            return t("measurement.value", noun=noun, value=values[0][1])
        return "; ".join(f"{name}: {value}" for name, value in values) + "."

    if kind == "status" or expected is None:
        if len(entities) == 1:
            entity = entities[0]
            return t("ha_state_routing.449", v0=entity["name"], v1=_state_label(entity["state"]))
        return (
            "; ".join(f"{entity['name']}: {_state_label(entity['state'])}" for entity in entities)
            + "."
        )

    matching = [entity for entity in entities if entity["state"] == expected]
    other = [entity for entity in entities if entity["state"] != expected]
    label = _domain_label(entities, len(entities))
    expected_label = _state_label(expected)

    if kind == "count":
        return t(
            "ha_state_routing.460",
            v0=len(matching),
            v1=len(entities),
            label=label,
            expected_label=expected_label,
        )
    if kind == "list":
        if not matching:
            return t("ha_state_routing.463", label=label, expected_label=expected_label)
        return (
            f"{expected_label.capitalize()}: "
            + ", ".join(entity["name"] for entity in matching)
            + "."
        )

    if len(entities) == 1:
        entity = entities[0]
        if matching:
            return t("ha_state_routing.471", v0=entity["name"], expected_label=expected_label)
        return t("ha_state_routing.472", v0=entity["name"], v1=_state_label(entity["state"]))

    if not other:
        return t(
            "ha_state_routing.475", v0=len(entities), label=label, expected_label=expected_label
        )
    if not matching:
        states = "; ".join(
            f"{entity['name']}: {_state_label(entity['state'])}" for entity in entities
        )
        return t(
            "ha_state_routing.480",
            v0=len(entities),
            label=label,
            expected_label=expected_label,
            states=states,
        )
    matching_names = ", ".join(entity["name"] for entity in matching)
    other_states = "; ".join(
        f"{entity['name']}: {_state_label(entity['state'])}" for entity in other
    )
    return t(
        "ha_state_routing.486",
        v0=len(matching),
        v1=len(entities),
        label=label,
        expected_label=expected_label,
        matching_names=matching_names,
        other_states=other_states,
    )


def compact_live_followup_request(request):
    """Fallback: reduce an unhandled state follow-up before sending it to Gemma.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        ChatRequest | None: Minimal tool-free follow-up, or None when not a live-result turn.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    followup = _live_followup(request.messages)
    if followup is None:
        return None
    question, _, results = followup
    live_text = "\n".join(str(result) for result in results)
    messages = [
        {"role": "system", "content": t(_FINAL_SYSTEM_KEY)},
        {
            "role": "user",
            "content": t("ha_state_routing.502", question=question, live_text=live_text),
        },
    ]
    prepared = request.model_copy(
        update={
            "messages": messages,
            "tools": None,
            "tool_choice": None,
        }
    )
    prepared._request_id = getattr(request, "_request_id", "-")
    return prepared


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
    request_id = getattr(request, "_request_id", "-")

    direct_answer = deterministic_live_response(request)
    if direct_answer is not None:
        prepared = request.model_copy(update={"tools": None, "tool_choice": None})
        prepared._request_id = request_id
        object.__setattr__(prepared, "_direct_ha_state_response", direct_answer)
        _LOG.info(
            "ha_route request_id=%s route=direct_state_response skipped_gemma=true",
            request_id,
        )
        if backend.settings.debug_log:
            _LOG.debug(
                "event=ha_state_route request_id=%s json=%s",
                request_id,
                json.dumps(
                    {
                        "route": "direct_state_response",
                        "response": direct_answer,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
            )
        return prepared

    compact = compact_live_followup_request(request)
    if compact is not None:
        _LOG.info(
            "ha_route request_id=%s route=live_context_followup_minimal tools=0",
            request_id,
        )
        if backend.settings.debug_log:
            _LOG.debug(
                "event=ha_state_route request_id=%s json=%s",
                request_id,
                json.dumps(
                    {
                        "route": "live_context_followup_minimal",
                        "messages": compact.messages,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
            )
        return compact

    query = latest_user_text(request.messages).strip()
    live_tool = _live_tool(request.tools)
    if live_tool is not None and _is_state_question(query):
        direct = direct_live_context_response(request)
        if direct is not None:
            prepared = request.model_copy(
                update={
                    "tools": [live_tool],
                    "tool_choice": None,
                }
            )
            prepared._request_id = request_id
            object.__setattr__(prepared, "_direct_ha_response", direct)
            _LOG.info(
                "ha_route request_id=%s route=direct_live_context skipped_gemma=true",
                request_id,
            )
            if backend.settings.debug_log:
                _LOG.debug(
                    "event=ha_state_route request_id=%s json=%s",
                    request_id,
                    json.dumps(
                        {
                            "route": "direct_live_context",
                            "tool_call": direct["tool_calls"][0],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ),
                )
            return prepared

        # A state question must never expose TurnOn/TurnOff to Gemma even
        # when the target is too ambiguous for a deterministic direct call.
        narrowed = request.model_copy(update={"tools": [live_tool]})
        narrowed._request_id = request_id
        _LOG.info(
            "ha_route request_id=%s route=live_context_only reason=ambiguous_target",
            request_id,
        )
        return next_stage(narrowed)

    return next_stage(request)


def _location_phrase(query: str) -> str | None:
    """Extract a strongly expressed location from a state question.

    Args:
        query (str): User question or normalized utterance to match.

    Returns:
        str | None: Original location text, or None when no explicit location is found.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    language = detect_language(str(query))
    match = re.search(catalogue(language)["patterns"]["state_location"], str(query), re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip(" \t\r\n?!.,:;")
    return value or None


def _friendly_live_error(calls, results) -> str | None:
    """Return a deterministic user-facing message for HA lookup failures.

    Args:
        calls (list[dict[str, Any]]): Assistant function calls from a tool round.
        results (list[Any]): Client tool results aligned to the active call round.

    Returns:
        str | None: Localized lookup error, or None when results report no failure.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    for index, content in enumerate(results):
        payload = _json_object(content)
        if payload is None:
            continue
        error = payload.get("error")
        failed = payload.get("success") is False
        if error is None and not failed:
            continue

        call = calls[index] if index < len(calls) else (calls[0] if calls else {})
        arguments = _json_object(call.get("function", {}).get("arguments")) or {}
        area = arguments.get("area")
        name = arguments.get("name")
        domains = arguments.get("domain")
        if isinstance(domains, str):
            domains = [domains]

        if area and domains == ["light"]:
            return t("ha_state_routing_fixes.159", area=area)
        if area:
            return t("ha_state_routing_fixes.161", area=area)
        if name:
            return t("ha_state_routing_fixes.163", name=name)
        return t("ha_state_routing_fixes.164")
    return None


_HOME_LOCATIONS = lexicon("ha_state_routing_fixes._HOME_LOCATIONS")
_ERROR_KEYS = {"error", "message", "detail", "details"}
