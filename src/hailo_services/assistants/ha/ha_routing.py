"""HA relevance and general-question handling inside the virtual model.

Exact deterministic actions are handled by ha_intents/HassIL at the HA pipeline
boundary. This module retains legacy matching helpers for compatibility and
routes unmatched requests through the existing context retrieval path.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import uuid

from hailo_services.shared.i18n import lexicon
from hailo_services.shared.i18n import normalize_matching as _normalized
from hailo_services.shared.tool_retrieval import (
    _embedding,
    _score,
    _static_context_parts,
    _text,
    _tokens,
    _tool_score,
    latest_user_text,
)

_LOG = logging.getLogger(__name__)
_SEMANTIC_RELEVANCE_THRESHOLD = 0.56
_EXTRA_STOP_WORDS = lexicon("ha_routing._EXTRA_STOP_WORDS")
_DOMAIN_WORDS = lexicon("ha_state_routing._DOMAIN_WORDS")
_DIRECT_TOOLS = {"intent__HassTurnOn", "intent__HassTurnOff"}


def _query_tokens(text: str) -> set[str]:
    """Tokenize the user query while removing routing-specific stop words.

    Args:
        text (str): Text to parse, normalize, match or render.

    Returns:
        set[str]: Normalized significant query terms.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return _tokens(text) - _EXTRA_STOP_WORDS


def _is_ha_system_message(message) -> bool:
    """Identify HA instructions in a generated system message.

    Args:
        message (str | dict[str, Any]): Formatted diagnostic text or ASGI message.

    Returns:
        bool: Whether the message is an HA system envelope.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if message.get("role") != "system" or not isinstance(message.get("content"), str):
        return False
    content = message["content"]
    return "Static Context:" in content and "Home Assistant" in content


def _has_tool_history(messages) -> bool:
    """Check whether messages contain tool calls or results.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        bool: Whether active tool dependencies may constrain preparation.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return any(message.get("role") == "tool" or message.get("tool_calls") for message in messages)


def _entity_entries(messages) -> list[str]:
    """Extract raw static entity blocks from HA system messages.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        list[str]: Text blocks used by lexical and semantic retrieval.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    entries: list[str] = []
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        parts = _static_context_parts(message["content"])
        if parts is not None:
            entries.extend(parts[1])
    return entries


def _tool_description(tool) -> str:
    """Build retrieval text from function name and description.

    Args:
        tool (dict[str, Any]): Client-provided function schema.

    Returns:
        str: Tool description for semantic embedding.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    function = tool.get("function", {}) if isinstance(tool, dict) else {}
    return " ".join(
        (
            function.get("name", ""),
            function.get("description", ""),
            _text(function.get("parameters", {}))[:1400],
        )
    )


def assess_ha_relevance(
    messages,
    tools,
    *,
    encoder=None,
    embedding_cache=None,
    semantic_threshold: float = _SEMANTIC_RELEVANCE_THRESHOLD,
):
    """Classify whether the latest user request needs Home Assistant context.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.
        encoder (MiniLM | None): Optional resident semantic encoder; None uses lexical matching.
        embedding_cache (dict[str, np.ndarray] | None): Bounded cache for static retrieval embeddings.
        semantic_threshold (float): Minimum similarity for semantic HA relevance.

    Returns:
        dict[str, Any]: Relevance decision, lexical/semantic scores and confidence diagnostics.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    query_text = latest_user_text(messages).strip()
    query = _query_tokens(query_text)
    entries = _entity_entries(messages)
    tools = tools or []

    entity_scores = [(_score(query, entry), entry) for entry in entries]
    tool_scores = [(_tool_score(query, tool), tool) for tool in tools]
    entity_lexical = max((score for score, _ in entity_scores), default=0)
    tool_lexical = max((score for score, _ in tool_scores), default=0)
    semantic_tool_score = None
    semantic_tool_name = None

    relevant = bool(entity_lexical > 0 or tool_lexical > 0)
    reason = "lexical" if relevant else "none"

    # Semantic fallback is intentionally thresholded. MiniLM may always find a
    # nearest tool, even for unrelated questions (e.g. geography), so a mere
    # top-1 result must not make the request HA-related.
    if not relevant and encoder is not None and query_text and tools:
        import numpy as np

        query_vector = _embedding(encoder, query_text, embedding_cache)
        semantic = []
        for tool in tools:
            similarity = float(
                np.dot(
                    query_vector,
                    _embedding(encoder, _tool_description(tool), embedding_cache),
                )
            )
            semantic.append((similarity, tool))
        if semantic:
            semantic.sort(key=lambda item: -item[0])
            semantic_tool_score, semantic_tool = semantic[0]
            semantic_tool_name = semantic_tool.get("function", {}).get("name")
            if semantic_tool_score >= semantic_threshold:
                relevant = True
                reason = "semantic_tool"

    result = {
        "relevant": relevant,
        "reason": reason,
        "query_text": query_text,
        "query_tokens": sorted(query),
        "entity_lexical_max": entity_lexical,
        "tool_lexical_max": tool_lexical,
        "semantic_tool_max": semantic_tool_score,
        "semantic_tool_name": semantic_tool_name,
        "semantic_threshold": semantic_threshold,
        "entity_count": len(entries),
        "tool_count": len(tools),
    }

    if reason == "semantic_tool":
        second_score, second_tool = semantic[1] if len(semantic) > 1 else (-1.0, {})
        margin = semantic_tool_score - second_score
        result.update(
            {
                "semantic_tool_second_max": second_score,
                "semantic_tool_second_name": second_tool.get("function", {}).get("name"),
                "semantic_tool_margin": margin,
                "semantic_margin_threshold": _SEMANTIC_RELEVANCE_MARGIN,
                "semantic_strong_threshold": _SEMANTIC_STRONG_RELEVANCE,
            }
        )
        if margin < _SEMANTIC_RELEVANCE_MARGIN and semantic_tool_score < max(
            _SEMANTIC_STRONG_RELEVANCE, semantic_threshold + _SEMANTIC_RELEVANCE_MARGIN
        ):
            result.update(relevant=False, reason="semantic_ambiguous")
    return result


def general_passthrough_request(request):
    """Drop only the generated HA envelope, preserving the actual conversation.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        ChatRequest: Copy preserving conversation content with the generated HA envelope removed.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    messages = [
        copy.deepcopy(message) for message in request.messages if not _is_ha_system_message(message)
    ]
    return request.model_copy(
        update={
            "messages": messages,
            "tools": None,
            "tool_choice": None,
        }
    )


def _parse_entity(entry: str):
    """Parse name, domain and area from an HA static-context entry.

    Args:
        entry (str): Raw static-context entity block.

    Returns:
        dict[str, str] | None: Parsed entity, or None for a malformed entry.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    name_match = re.search(r"(?m)^- names:\s*(.+)$", entry)
    domain_match = re.search(r"(?m)^\s+domain:\s*(.+)$", entry)
    area_match = re.search(r"(?m)^\s+areas:\s*(.+)$", entry)
    if not name_match or not domain_match:
        return None
    return {
        "name": name_match.group(1).strip(),
        "domain": domain_match.group(1).strip(),
        "area": area_match.group(1).strip() if area_match else "",
    }


def _contains_phrase(text: str, phrase: str) -> bool:
    """Check whether a normalized phrase occurs as complete words.

    Args:
        text (str): Text to parse, normalize, match or render.
        phrase (str): Normalized phrase to match as complete words.

    Returns:
        bool: Whether the phrase is present.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return bool(phrase) and f" {phrase} " in f" {text} "


def _explicit_action(query: str, tool_name: str) -> bool:
    """Check that on/off wording agrees with the selected tool.

    Args:
        query (str): User question or normalized utterance to match.
        tool_name (str): Client function identifier.

    Returns:
        bool: Whether the user explicitly requested the selected action.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    query = _normalized(query)
    if tool_name == "intent__HassTurnOff":
        return bool(
            re.search(lexicon("ha_routing.pattern.178.22"), query)
            or re.search(lexicon("ha_routing.pattern.179.25"), query)
            or re.search(lexicon("ha_routing.match.180.25"), query)
        )
    if tool_name == "intent__HassTurnOn":
        return bool(
            re.search(lexicon("ha_routing.pattern.184.22"), query)
            or re.search(lexicon("ha_routing.pattern.185.25"), query)
            or re.search(lexicon("ha_routing.match.186.25"), query)
        )
    return False


def _domain_aliases(domain: str) -> set[str]:
    """Read normalized vocabulary for one Home Assistant domain.

    Args:
        domain (str): Resolved Home Assistant domain, or None.

    Returns:
        set[str]: Accepted aliases for the domain.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    aliases = {_normalized(domain)}
    aliases.update(_normalized(word) for word in _DOMAIN_WORDS.get(domain, set()))
    return {alias for alias in aliases if alias}


def _mentioned_domains(query: str, domains: set[str]) -> list[str]:
    """Return candidate HA domains explicitly named by the user.

    Args:
        query (str): User question or normalized utterance to match.
        domains (set[str]): Candidate domain identifiers from the catalogue.

    Returns:
        list[str]: Domains whose vocabulary appears in the query.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    text = _normalized(query)
    matches: list[str] = []
    for domain in sorted(domains):
        if any(_contains_phrase(text, alias) for alias in _domain_aliases(domain)):
            matches.append(domain)
    return matches


def _generic_domain_entity(entity) -> bool:
    """Treat names such as 'Light'/'Licht' as a class, not a unique device.

    Args:
        entity (dict[str, Any]): HA entity containing name, domain, area and optional state/attributes.

    Returns:
        bool: Whether the name describes a whole class of devices.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return _normalized(entity["name"]) in _domain_aliases(entity["domain"])


def _specific_entity_matches(query: str, entities) -> list[dict]:
    """Find explicitly named devices, preferring the most specific name match.

    Args:
        query (str): User question or normalized utterance to match.
        entities (list[dict[str, Any]]): Static or live HA entities relevant to the request.

    Returns:
        list[dict]: Most-specific explicitly named catalogue entities.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    text = _normalized(query)
    matches = [
        entity
        for entity in entities
        if not _generic_domain_entity(entity)
        and _contains_phrase(text, _normalized(entity["name"]))
    ]
    if not matches:
        return []
    longest = max(len(_normalized(entity["name"])) for entity in matches)
    return [entity for entity in matches if len(_normalized(entity["name"])) == longest]


def direct_action_response(request, *, source_messages=None, trace=None):
    """Build a deterministic tool call for an unambiguous on/off command.

    Resolution order is intentionally location-first: an explicitly mentioned
    area constrains all subsequent entity/domain matching. This prevents generic
    entity names such as "Light" in other rooms from making an otherwise clear
    area command ambiguous.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        source_messages (list[dict[str, Any]] | None): Original context used before retrieval compaction.
        trace (dict | list | None): Optional mutable diagnostic collector.

    Returns:
        dict[str, Any] | None: Validated on/off call message, or None when inference is required.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if trace is not None:
        trace.clear()

    def reject(reason: str, **details):
        """Attach a rejection reason to the deterministic action trace.

        Args:
            reason (str): Diagnostic reason for rejecting a direct action.
            **details (Any): Additional rejection fields appended to the trace.

        Returns:
            None: Records why the shortcut did not apply.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if trace is not None:
            trace.update({"direct_action_reason": reason, **details})
        return None

    tools = request.tools or []
    if len(tools) != 1:
        return reject("tool_count", selected_tool_count=len(tools))
    tool_name = tools[0].get("function", {}).get("name")
    if tool_name not in _DIRECT_TOOLS:
        return reject("unsupported_tool", selected_tool=tool_name)

    query = latest_user_text(request.messages).strip()
    if not _explicit_action(query, tool_name):
        return reject("action_not_explicit", selected_tool=tool_name)

    entity_messages = source_messages if source_messages is not None else request.messages
    entities = [
        parsed
        for parsed in (_parse_entity(entry) for entry in _entity_entries(entity_messages))
        if parsed is not None
    ]
    if not entities:
        return reject("no_entities")

    normalized_query = _normalized(query)
    area_groups: dict[str, list[dict]] = {}
    for entity in entities:
        area = entity["area"]
        if area and _contains_phrase(normalized_query, _normalized(area)):
            area_groups.setdefault(area, []).append(entity)

    arguments = None
    target_kind = None

    if len(area_groups) == 1:
        area, members = next(iter(area_groups.items()))
        domains = {member["domain"] for member in members}
        mentioned_domains = _mentioned_domains(query, domains)
        if len(mentioned_domains) > 1:
            return reject(
                "area_domain_ambiguous",
                matched_area=area,
                candidate_domains=sorted(domains),
                mentioned_domains=mentioned_domains,
            )

        requested_domain = mentioned_domains[0] if mentioned_domains else None
        scoped_members = [
            member
            for member in members
            if requested_domain is None or member["domain"] == requested_domain
        ]
        explicit = _specific_entity_matches(query, scoped_members)
        explicit_keys = {(item["name"], item["domain"], item["area"]) for item in explicit}
        if len(explicit_keys) == 1:
            entity = explicit[0]
            arguments = {"name": entity["name"], "domain": [entity["domain"]]}
            target_kind = "entity"
        elif explicit:
            return reject(
                "entity_name_ambiguous",
                matched_area=area,
                matched_entities=sorted(
                    {f"{item['name']}|{item['domain']}|{item['area']}" for item in explicit}
                ),
            )
        elif requested_domain is not None:
            if not scoped_members:
                return reject(
                    "area_domain_not_found",
                    matched_area=area,
                    requested_domain=requested_domain,
                )
            arguments = {"area": area, "domain": [requested_domain]}
            target_kind = "area"
        elif len(domains) == 1:
            domain = next(iter(domains))
            arguments = {"area": area, "domain": [domain]}
            target_kind = "area"
        else:
            return reject(
                "area_domain_ambiguous",
                matched_area=area,
                candidate_domains=sorted(domains),
                mentioned_domains=[],
            )
    elif len(area_groups) > 1:
        return reject("area_not_unique", matched_areas=sorted(area_groups))
    else:
        explicit = _specific_entity_matches(query, entities)
        explicit_keys = {(item["name"], item["domain"], item["area"]) for item in explicit}
        if len(explicit_keys) == 1:
            entity = explicit[0]
            arguments = {"name": entity["name"], "domain": [entity["domain"]]}
            target_kind = "entity"
        elif explicit:
            return reject(
                "entity_name_ambiguous",
                matched_entities=sorted(
                    {f"{item['name']}|{item['domain']}|{item['area']}" for item in explicit}
                ),
            )
        else:
            return reject("target_not_unique")

    if trace is not None:
        trace.update(
            {
                "direct_action_reason": "unambiguous",
                "direct_target_kind": target_kind,
                "direct_tool": tool_name,
                "direct_arguments": arguments,
            }
        )

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
        "_routing": {
            "kind": "direct_action",
            "target_kind": target_kind,
            "tool": tool_name,
            "arguments": arguments,
        },
    }


def _log_route(debug_log: bool, request_id: str, route: str, payload):
    """Record route decisions and optional structured diagnostics.

    Args:
        debug_log (bool): Whether verbose request diagnostics should be logged.
        request_id (str): Request correlation identifier used in diagnostics.
        route (str): Chosen routing path for diagnostic logs.
        payload (Any): Decoded protocol data or structured diagnostic payload.

    Returns:
        None: Emits route log records.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    _LOG.info("ha_route request_id=%s route=%s", request_id, route)
    if debug_log:
        _LOG.debug(
            "event=ha_route request_id=%s json=%s",
            request_id,
            json.dumps(
                {"route": route, **payload}, ensure_ascii=False, separators=(",", ":"), default=str
            ),
        )


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
    # Active tool rounds must retain their call/result dependencies.
    if _has_tool_history(request.messages) or isinstance(request.tool_choice, dict):
        prepared = next_stage(request)
        _log_route(backend.settings.debug_log, request_id, "ha_llm", {"reason": "tool_history"})
        return prepared

    is_ha_envelope = bool(request.tools) and any(
        _is_ha_system_message(message) for message in request.messages
    )
    if not is_ha_envelope:
        return next_stage(request)

    relevance = assess_ha_relevance(
        request.messages,
        request.tools,
        encoder=backend.minilm,
        embedding_cache=backend._retrieval_embedding_cache,
    )
    if not relevance["relevant"]:
        prepared = general_passthrough_request(request)
        _log_route(backend.settings.debug_log, request_id, "general_passthrough", relevance)
        return prepared

    prepared = next_stage(request)
    _log_route(backend.settings.debug_log, request_id, "ha_llm", relevance)
    return prepared


_SEMANTIC_RELEVANCE_MARGIN = 0.05
_SEMANTIC_STRONG_RELEVANCE = 0.66
