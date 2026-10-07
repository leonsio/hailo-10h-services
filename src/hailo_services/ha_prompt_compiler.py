"""Compile Home Assistant requests into a minimal Gemma prompt.

Home Assistant clients send a large generic system prompt, a complete entity
catalogue and verbose JSON schemas.  Retrieval narrows those objects, but Gemma
still pays the prefill cost for rules and schema properties that are irrelevant
to the current request.  This module runs after deterministic HA routing and the
existing MiniLM retrieval layer and rebuilds the remaining LLM request from the
smallest useful context.
"""

from __future__ import annotations

import copy
import json
import logging
import re

from .i18n import lexicon, t
from .i18n import normalize_matching as _normalized
from .tool_retrieval import _embedding, _static_context_parts, latest_user_text

_LOG = logging.getLogger(__name__)

_CAPABILITIES = lexicon("ha_prompt_compiler._CAPABILITIES")

_TOOL_CAPABILITY = {
    "homeassistant__GetLiveContext": "state.query",
    "intent__HassTurnOn": "device.turn_on",
    "intent__HassTurnOff": "device.turn_off",
    "light__HassLightSet": "light.adjust",
    "climate__HassClimateSetTemperature": "climate.temperature",
    "intent__HassSetPosition": "cover.position",
    "intent__HassStopMoving": "cover.stop",
    "vacuum__HassVacuumCleanArea": "vacuum.control",
    "vacuum__HassVacuumReturnToBase": "vacuum.control",
    "vacuum__HassVacuumStart": "vacuum.control",
    "todo__get_items": "todo.read",
    "todo__HassListAddItem": "todo.modify",
    "todo__HassListCompleteItem": "todo.modify",
    "todo__HassListRemoveItem": "todo.modify",
    "assist_satellite__HassBroadcast": "broadcast",
    "llm__GetDateTime": "datetime",
}

_CAPABILITY_TOOLS = {
    "state.query": {"homeassistant__GetLiveContext"},
    "device.turn_on": {"intent__HassTurnOn"},
    "device.turn_off": {"intent__HassTurnOff"},
    "light.brightness": {"light__HassLightSet"},
    "light.color": {"light__HassLightSet"},
    "light.temperature": {"light__HassLightSet"},
    "light.adjust": {"light__HassLightSet"},
    "climate.temperature": {"climate__HassClimateSetTemperature"},
    "cover.position": {"intent__HassSetPosition"},
    "cover.stop": {"intent__HassStopMoving"},
    "vacuum.control": {
        "vacuum__HassVacuumCleanArea",
        "vacuum__HassVacuumReturnToBase",
        "vacuum__HassVacuumStart",
    },
    "todo.read": {"todo__get_items"},
    "todo.modify": {
        "todo__HassListAddItem",
        "todo__HassListCompleteItem",
        "todo__HassListRemoveItem",
    },
    "broadcast": {"assist_satellite__HassBroadcast"},
    "datetime": {"llm__GetDateTime"},
}

_TOOL_PROPERTIES = {
    "homeassistant__GetLiveContext": {"name", "domain", "area"},
    "intent__HassTurnOn": {"name", "area", "domain"},
    "intent__HassTurnOff": {"name", "area", "domain"},
    "light__HassLightSet": {"name", "area", "domain", "brightness", "color", "temperature"},
    "climate__HassClimateSetTemperature": {"temperature", "area", "name"},
    "intent__HassSetPosition": {"name", "area", "domain", "position"},
    "intent__HassStopMoving": {"name", "area", "domain"},
    "vacuum__HassVacuumCleanArea": {"area", "name"},
    "vacuum__HassVacuumReturnToBase": {"name", "area", "domain"},
    "vacuum__HassVacuumStart": {"name", "area", "domain"},
    "todo__get_items": {"todo_list", "status"},
    "todo__HassListAddItem": {"item", "name"},
    "todo__HassListCompleteItem": {"item", "name"},
    "todo__HassListRemoveItem": {"item", "name"},
    "assist_satellite__HassBroadcast": {"message"},
    "llm__GetDateTime": set(),
}

_SHORT_TOOL_DESCRIPTIONS = lexicon("ha_prompt_compiler._SHORT_TOOL_DESCRIPTIONS")

_DOMAIN_ALIASES = lexicon("ha_prompt_compiler.domain_aliases")


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


def _is_ha_system(message) -> bool:
    """Identify an HA-generated system envelope containing tool or entity context.

    Args:
        message (str | dict[str, Any]): Formatted diagnostic text or ASGI message.

    Returns:
        bool: Whether the system message belongs to the HA envelope.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return (
        message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and "Static Context:" in message["content"]
        and "Home Assistant" in message["content"]
    )


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


def _parse_entity(entry: str):
    """Parse name, domain and area from an HA static-context entry.

    Args:
        entry (str): Raw static-context entity block.

    Returns:
        dict[str, str] | None: Parsed entity, or None for a malformed entry.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    name = re.search(r"(?m)^- names:\s*(.+)$", entry)
    domain = re.search(r"(?m)^\s+domain:\s*(.+)$", entry)
    area = re.search(r"(?m)^\s+areas:\s*(.+)$", entry)
    if not name or not domain:
        return None
    return {
        "name": name.group(1).strip(),
        "domain": domain.group(1).strip(),
        "area": area.group(1).strip() if area else "",
    }


def _entities(messages):
    """Parse the HA static catalogue from system messages.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        list[dict[str, str]]: Entities with name, domain and area.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    result = []
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        parts = _static_context_parts(message["content"])
        if parts is None:
            continue
        result.extend(
            parsed for parsed in (_parse_entity(entry) for entry in parts[1]) if parsed is not None
        )
    return result


def _tool_names(tools):
    """Read function identifiers from selected OpenAI tools.

    Args:
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.

    Returns:
        list[str]: Selected tool names in their input order.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return [tool.get("function", {}).get("name", "") for tool in tools or []]


def _deterministic_capability(query: str) -> str | None:
    """Recognize an explicit HA capability from normalized action phrases.

    Args:
        query (str): User question or normalized utterance to match.

    Returns:
        str | None: Capability identifier, or None when no precise action is recognized.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    text = _normalized(query)
    light_target = bool(re.search(lexicon("ha_prompt_compiler.pattern.152.34"), text))
    if light_target and (
        re.search(r"\d+\s*%", query) or re.search(lexicon("ha_prompt_compiler.pattern.154.8"), text)
    ):
        return "light.brightness"
    if re.search(lexicon("ha_prompt_compiler.pattern.157.17"), text):
        return "light.temperature"
    if re.search(lexicon("ha_prompt_compiler.match.159.17"), text) and light_target:
        return "light.color"
    if re.search(lexicon("ha_prompt_compiler.pattern.161.17"), text) and re.search(
        lexicon("ha_prompt_compiler.pattern.162.8"), text
    ):
        return "climate.temperature"
    if re.search(lexicon("ha_prompt_compiler.match.165.17"), text) and re.search(
        lexicon("ha_prompt_compiler.match.166.8"), text
    ):
        return "cover.position"
    if re.search(lexicon("ha_prompt_compiler.match.169.17"), text) and re.search(
        lexicon("ha_prompt_compiler.match.170.8"), text
    ):
        return "cover.stop"
    if re.search(lexicon("ha_prompt_compiler.match.173.17"), text):
        return "vacuum.control"
    if re.search(lexicon("ha_prompt_compiler.match.175.17"), text):
        if re.search(lexicon("ha_prompt_compiler.pattern.176.21"), text):
            return "todo.read"
        return "todo.modify"
    if re.search(lexicon("ha_prompt_compiler.pattern.179.17"), text):
        return "datetime"
    if re.search(lexicon("ha_prompt_compiler.match.181.17"), text):
        return "broadcast"
    return None


def _capability(query: str, tools, encoder, embedding_cache):
    """Infer semantics only inside capability families of selected tools.

    Args:
        query (str): User question or normalized utterance to match.
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.
        encoder (MiniLM | None): Optional resident semantic encoder; None uses lexical matching.
        embedding_cache (dict[str, np.ndarray] | None): Bounded cache for static retrieval embeddings.

    Returns:
        tuple[str | None, str, dict | None]: Capability, selection source and optional semantic scores.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    tool_names = _tool_names(tools)
    tool_caps = {_TOOL_CAPABILITY.get(name) for name in tool_names} - {None}
    action_caps = tool_caps & {"device.turn_on", "device.turn_off"}
    if len(action_caps) == 1:
        return next(iter(action_caps)), "selected_tool", None

    allowed = _allowed_capabilities(tool_names)
    deterministic = _deterministic_capability(query)
    if deterministic is not None and deterministic in allowed:
        return deterministic, "deterministic", None

    # Preserve the established fast path when retrieval left exactly one tool
    # family. For example a lone light tool intentionally compiles as
    # light.adjust unless the query was deterministically more specific above.
    if len(tool_caps) == 1:
        return next(iter(tool_caps)), "selected_tool", None

    if encoder is None or not query or not allowed:
        return None, "selected_tools", {"best": None, "second": None}

    import numpy as np

    query_vector = _embedding(encoder, query, embedding_cache)
    ranked = []
    for name in allowed:
        description = _CAPABILITIES.get(name)
        if not description:
            continue
        similarity = float(
            np.dot(
                query_vector,
                _embedding(encoder, description, embedding_cache),
            )
        )
        ranked.append((similarity, name))
    ranked.sort(reverse=True)
    if not ranked:
        return None, "selected_tools", {"best": None, "second": None}

    best_score, best_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -1.0
    semantic = {"best": best_score, "second": second_score}
    if best_score >= 0.50 and best_score - second_score >= 0.025:
        return best_name, "minilm", semantic
    return None, "selected_tools", semantic


def _domain(query: str, capability: str | None):
    """Infer the HA domain from query words and selected capability.

    Args:
        query (str): User question or normalized utterance to match.
        capability (str | None): Canonical action capability selected for this request.

    Returns:
        str | None: Unambiguous domain hint, or None.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    words = set(_normalized(query).split())
    matches = [name for name, aliases in _DOMAIN_ALIASES.items() if words & aliases]
    if len(matches) == 1:
        return matches[0]
    if capability and capability.startswith("light."):
        return "light"
    if capability and capability.startswith("climate."):
        return "climate"
    if capability and capability.startswith("cover."):
        return "cover"
    if capability == "vacuum.control":
        return "vacuum"
    return None


def _area(query: str, entities):
    """Find a uniquely mentioned area in the static entity catalogue.

    Args:
        query (str): User question or normalized utterance to match.
        entities (list[dict[str, Any]]): Static or live HA entities relevant to the request.

    Returns:
        str | None: Area name, or None when absent or ambiguous.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    text = _normalized(query)
    areas = sorted({item["area"] for item in entities if item["area"]}, key=len, reverse=True)
    matches = [area for area in areas if _contains(text, _normalized(area))]
    return matches[0] if len(matches) == 1 else None


def _generic_entity_name(name: str, domain: str | None):
    """Check whether an entity name only names a device class.

    Args:
        name (str): Function, attribute, device or model identifier.
        domain (str | None): Resolved Home Assistant domain, or None.

    Returns:
        bool: Whether the name is generic for the requested domain.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    normalized = _normalized(name)
    if domain is None:
        return False
    return normalized in {_normalized(alias) for alias in _DOMAIN_ALIASES.get(domain, set())}


def _relevant_entities(
    source_messages, prepared_messages, query: str, domain: str | None, area: str | None
):
    """Select catalogue entities matching the requested domain and target.

    Args:
        source_messages (list[dict[str, Any]] | None): Original context used before retrieval compaction.
        prepared_messages (list[dict[str, Any]]): Messages after context retrieval.
        query (str): User question or normalized utterance to match.
        domain (str | None): Resolved Home Assistant domain, or None.
        area (str | None): Resolved catalogue area, or None when no area is selected.

    Returns:
        list[dict[str, str]]: Relevant parsed entities for validation or compact prompting.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    source = _entities(source_messages)
    text = _normalized(query)

    if area is not None:
        matches = [item for item in source if item["area"] == area]
        if domain is not None:
            matches = [item for item in matches if item["domain"] == domain]
        explicit = [
            item
            for item in matches
            if not _generic_entity_name(item["name"], domain)
            and _contains(text, _normalized(item["name"]))
        ]
        if explicit:
            return explicit[:8]
        return matches[:12]

    explicit = [
        item
        for item in source
        if (domain is None or item["domain"] == domain)
        and not _generic_entity_name(item["name"], domain)
        and _contains(text, _normalized(item["name"]))
    ]
    keys = {(item["name"], item["domain"], item["area"]) for item in explicit}
    if len(keys) == 1:
        return explicit[:1]

    # Fall back to the already MiniLM-ranked catalogue rather than re-ranking
    # every entity a second time.
    prepared = _entities(prepared_messages)
    if domain is not None:
        prepared = [item for item in prepared if item["domain"] == domain]
    return prepared[:8]


def _filter_tools(tools, capability: str | None):
    """Prefer tools belonging to the selected capability.

    Args:
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.
        capability (str | None): Canonical action capability selected for this request.

    Returns:
        list[dict[str, Any]]: Matching tools, falling back to the original selection.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    tools = list(tools or [])
    preferred = _CAPABILITY_TOOLS.get(capability)
    if not preferred:
        return tools
    selected = [tool for tool in tools if tool.get("function", {}).get("name") in preferred]
    return selected or tools


def _capability_properties(tool_name: str, capability: str | None):
    """Select schema properties needed for one capability and tool.

    Args:
        tool_name (str): Client function identifier.
        capability (str | None): Canonical action capability selected for this request.

    Returns:
        set[str]: Allowed parameter names before preserving required fields.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    allowed = set(_TOOL_PROPERTIES.get(tool_name, ()))
    if tool_name == "light__HassLightSet":
        if capability == "light.brightness":
            allowed -= {"color", "temperature"}
        elif capability == "light.color":
            allowed -= {"brightness", "temperature"}
        elif capability == "light.temperature":
            allowed -= {"brightness", "color"}
    return allowed


def _compact_tool(tool, capability, domain, area, entities):
    """Restrict a client schema to capability properties and valid target enums.

    Args:
        tool (dict[str, Any]): Client-provided function schema.
        capability (str | None): Canonical action capability selected for this request.
        domain (str | None): Resolved Home Assistant domain, or None.
        area (str | None): Resolved catalogue area, or None when no area is selected.
        entities (list[dict[str, Any]]): Static or live HA entities relevant to the request.

    Returns:
        dict[str, Any]: Copied schema preserving required fields and compatible domains.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    compact = copy.deepcopy(tool)
    function = compact.get("function", {})
    name = function.get("name", "")
    parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
    properties = (
        parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
    )
    required = set(parameters.get("required") or [])
    allowed = _capability_properties(name, capability)
    # Never remove a required field from an externally supplied schema.
    allowed |= required

    new_properties = {}
    for prop_name, schema in properties.items():
        if prop_name not in allowed:
            continue
        value = copy.deepcopy(schema)
        if isinstance(value, dict):
            value.pop("description", None)
        if (
            prop_name == "domain"
            and domain is not None
            and (not _source_domain_enum(tool) or domain in _source_domain_enum(tool))
        ):
            value = {"type": "array", "items": {"type": "string", "enum": [domain]}}
        elif prop_name == "area" and area is not None:
            value = {"type": "string", "enum": [area]}
        elif prop_name == "name" and entities:
            names = list(dict.fromkeys(item["name"] for item in entities))
            if len(names) <= 8:
                value = {"type": "string", "enum": names}
        new_properties[prop_name] = value

    function["description"] = _SHORT_TOOL_DESCRIPTIONS.get(name, function.get("description", ""))
    function["parameters"] = {
        "type": "object",
        "properties": new_properties,
        **(
            {
                "required": [
                    item for item in parameters.get("required", []) if item in new_properties
                ]
            }
            if parameters.get("required")
            else {}
        ),
        "additionalProperties": False,
    }
    return compact


def _minimal_system(capability, domain, area, entities, tools):
    """Compose minimal HA instructions and resolved target context.

    Args:
        capability (str | None): Canonical action capability selected for this request.
        domain (str | None): Resolved Home Assistant domain, or None.
        area (str | None): Resolved catalogue area, or None when no area is selected.
        entities (list[dict[str, Any]]): Static or live HA entities relevant to the request.
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.

    Returns:
        str: Request-specific system text with entities and capability.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    lines = [
        t("ha_prompt_compiler.384"),
        t("ha_prompt_compiler.385"),
    ]
    tool_names = set(_tool_names(tools))
    if "homeassistant__GetLiveContext" in tool_names:
        lines.append(t("ha_prompt_compiler.389"))
    elif tools:
        lines.append(t("ha_prompt_compiler.391"))
    if capability:
        lines.append(t("ha_prompt_compiler.393", capability=capability))
    context = []
    if domain:
        context.append(f"domain={domain}")
    if area:
        context.append(f"area={area}")
    if context:
        lines.append(t("ha_prompt_compiler.400") + "; ".join(context))
    if entities:
        lines.append(
            t("ha_prompt_compiler.402")
            + " | ".join(
                item["name"] + (f" [{item['area']}]" if item.get("area") and not area else "")
                for item in entities
            )
        )
    if (
        domain
        and not area
        and len(entities) > 1
        and any(name in tool_names for name in {"intent__HassTurnOn", "intent__HassTurnOff"})
    ):
        lines.append(t("ha_prompt_compiler.406"))
    return "\n".join(lines)


def _replace_ha_system(messages, system_text):
    """Replace the generated HA envelope while preserving unrelated messages.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.
        system_text (str): Compiled request-specific HA instructions.

    Returns:
        list[dict[str, Any]]: Messages with a compact HA system prompt.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    result = []
    replaced = False
    for message in messages:
        if _is_ha_system(message):
            if not replaced:
                result.append({"role": "system", "content": system_text})
                replaced = True
            continue
        result.append(copy.deepcopy(message))
    if not replaced:
        result.insert(0, {"role": "system", "content": system_text})
    return result


def _schema_stats(tools):
    """Count tool schemas, properties and enums for diagnostics.

    Args:
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.

    Returns:
        dict[str, int]: Schema size counters.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    properties = 0
    for tool in tools or []:
        params = tool.get("function", {}).get("parameters", {})
        if isinstance(params, dict) and isinstance(params.get("properties"), dict):
            properties += len(params["properties"])
    rendered = json.dumps(tools or [], ensure_ascii=False, separators=(",", ":"), default=str)
    return {"tools": len(tools or []), "properties": properties, "characters": len(rendered)}


def compile_ha_prompt(source_request, prepared_request, *, encoder=None, embedding_cache=None):
    """Return a request rebuilt from request-specific HA context and schemas.

    Args:
        source_request (ChatRequest): Original request before retrieval/compilation.
        prepared_request (ChatRequest): Request after retrieval and deterministic preparation.
        encoder (MiniLM | None): Optional resident semantic encoder; None uses lexical matching.
        embedding_cache (dict[str, np.ndarray] | None): Bounded cache for static retrieval embeddings.

    Returns:
        tuple[ChatRequest, dict | None]: Compiled request and diagnostic plan, or unchanged request and None.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    active_tools = _has_tool_history(source_request.messages)
    if not any(_is_ha_system(message) for message in source_request.messages):
        return prepared_request, None
    if not prepared_request.tools:
        return prepared_request, None
    if any(
        getattr(prepared_request, attribute, None) is not None
        for attribute in ("_direct_ha_response", "_direct_ha_state_response")
    ):
        return prepared_request, None

    query = latest_user_text(source_request.messages).strip()
    capability, capability_source, semantic = _capability(
        query,
        prepared_request.tools,
        encoder,
        embedding_cache,
    )
    intent_plan = getattr(source_request, "_ha_plan", {})
    if intent_plan.get("action") in {"light.brightness", "cover.position"}:
        capability, capability_source = intent_plan["action"], "canonical_slots"
    domain = _domain(query, capability)
    source_entities = _entities(source_request.messages)
    area = _area(query, source_entities)
    relevant = _relevant_entities(
        source_request.messages,
        prepared_request.messages,
        query,
        domain,
        area,
    )
    candidates = intent_plan.get("target_candidates", [])
    if candidates:
        area = None
        allowed_areas = {c["area"] for c in candidates}
        relevant = [
            e
            for e in source_entities
            if any(
                (c["kind"] == "area" and e["area"] == c["area"])
                or (c["kind"] == "name" and e["name"] == c["value"] and e["area"] == c["area"])
                for c in candidates
            )
            and (not domain or e["domain"] == domain)
        ]
    if intent_plan.get("name"):
        relevant = [e for e in relevant if e["name"] == intent_plan["name"]]
    tools = _filter_tools(prepared_request.tools, capability)
    compact_tools = [_compact_tool(tool, capability, domain, area, relevant) for tool in tools]
    if active_tools:
        # Retain declarations required by the current call/result chain.
        called = {
            c["function"]["name"]
            for m in source_request.messages
            for c in m.get("tool_calls") or []
        }
        compact_tools.extend(
            copy.deepcopy(tool)
            for tool in prepared_request.tools
            if tool["function"]["name"] in called
            and tool["function"]["name"] not in _tool_names(compact_tools)
        )
    system_text = _minimal_system(capability, domain, area, relevant, compact_tools)
    if candidates:
        choices = list(
            {(c["value"], c["kind"], c["area"], c["score"]): c for c in candidates}.values()
        )
        system_text += (
            "\nAmbiguous target candidates (similarity scores, not probabilities): "
            + json.dumps(
                [
                    {
                        "target": c["value"],
                        "kind": c["kind"],
                        "area": c["area"],
                        "score": round(c["score"], 2),
                    }
                    for c in choices
                ],
                ensure_ascii=False,
            )
        )
        system_text += "\nSelect the intended candidate using the user's request. If context cannot distinguish them, ask for clarification."
        for tool in compact_tools:
            properties = tool["function"]["parameters"]["properties"]
            if "area" in properties:
                properties["area"] = {"type": "string", "enum": sorted(allowed_areas)}
    messages = _replace_ha_system(prepared_request.messages, system_text)
    compiled = prepared_request.model_copy(update={"messages": messages, "tools": compact_tools})
    compiled._request_id = getattr(prepared_request, "_request_id", "-")

    before = _schema_stats(prepared_request.tools)
    after = _schema_stats(compact_tools)
    original_system_chars = sum(
        len(message.get("content", ""))
        for message in source_request.messages
        if _is_ha_system(message)
    )
    plan = {
        "capability": capability,
        "capability_source": capability_source,
        "semantic": semantic,
        "domain": domain,
        "area": area,
        "entities": relevant,
        "selected_tool_names": _tool_names(compact_tools),
        "system_characters_before": original_system_chars,
        "system_characters_after": len(system_text),
        "tool_schema_before": before,
        "tool_schema_after": after,
        "system_prompt": system_text,
        "tools": compact_tools,
    }
    object.__setattr__(compiled, "_ha_prompt_plan", plan)
    return compiled, plan


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
    source_request = request
    prepared = next_stage(request)
    compiled, plan = compile_ha_prompt(
        source_request,
        prepared,
        encoder=backend.minilm,
        embedding_cache=backend._retrieval_embedding_cache,
    )
    if plan is None:
        return prepared
    request_id = getattr(prepared, "_request_id", "-")
    _LOG.info(
        "HA prompt compiler request_id=%s capability=%s domain=%s area=%s "
        "entities=%d tools=%d->%d properties=%d->%d system_chars=%d->%d",
        request_id,
        plan["capability"] or "unknown",
        plan["domain"] or "-",
        plan["area"] or "-",
        len(plan["entities"]),
        plan["tool_schema_before"]["tools"],
        plan["tool_schema_after"]["tools"],
        plan["tool_schema_before"]["properties"],
        plan["tool_schema_after"]["properties"],
        plan["system_characters_before"],
        plan["system_characters_after"],
    )
    if backend.settings.debug_log:
        _LOG.debug(
            "event=ha_prompt_plan request_id=%s json=%s",
            request_id,
            json.dumps(plan, ensure_ascii=False, separators=(",", ":"), default=str),
        )
    return compiled


def _allowed_capabilities(tool_names):
    """Find capability families reachable through selected tool names.

    Args:
        tool_names (Iterable[str]): Selected client function identifiers.

    Returns:
        set[str]: Capabilities represented by at least one selected tool.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    selected = set(tool_names)
    return {
        capability
        for capability, capability_tools in _CAPABILITY_TOOLS.items()
        if selected & capability_tools
    }


def _source_domain_enum(tool):
    """Read the original domain enum from a client tool schema.

    Args:
        tool (dict[str, Any]): Client-provided function schema.

    Returns:
        list[str] | None: Allowed domain values, or None when unrestricted.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    parameters = tool.get("function", {}).get("parameters", {}) if isinstance(tool, dict) else {}
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    domain = properties.get("domain", {}) if isinstance(properties, dict) else {}
    items = domain.get("items", {}) if isinstance(domain, dict) else {}
    enum = items.get("enum") if isinstance(items, dict) else None
    return list(enum) if isinstance(enum, list) and enum else None
