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
from functools import wraps

from .tool_retrieval import _embedding, _static_context_parts, latest_user_text

_LOG = logging.getLogger(__name__)

_CAPABILITIES = {
    "state.query": "read current Home Assistant device state or value",
    "light.brightness": "set lamp or light brightness percentage, brighter or darker",
    "light.color": "set lamp or light color",
    "light.temperature": "set warm white or cool white light color temperature",
    "light.adjust": "adjust a light brightness color or color temperature",
    "climate.temperature": "set thermostat heating cooling or room temperature",
    "cover.position": "set blind shutter curtain awning or cover position percentage",
    "cover.stop": "stop a moving blind shutter curtain awning or cover",
    "vacuum.control": "start robot vacuum clean an area or return vacuum to base",
    "todo.read": "read query or list todo shopping list items",
    "todo.modify": "add remove or complete todo shopping list items",
    "broadcast": "broadcast or announce a message in the home",
    "datetime": "get current date or time",
}

_TOOL_CAPABILITY = {
    "homeassistant__GetLiveContext": "state.query",
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

_SHORT_TOOL_DESCRIPTIONS = {
    "homeassistant__GetLiveContext": "Read current Home Assistant state.",
    "intent__HassTurnOn": "Turn on the target.",
    "intent__HassTurnOff": "Turn off the target.",
    "light__HassLightSet": "Set light properties.",
    "climate__HassClimateSetTemperature": "Set target temperature.",
    "intent__HassSetPosition": "Set cover position.",
    "intent__HassStopMoving": "Stop cover movement.",
    "vacuum__HassVacuumCleanArea": "Clean an area.",
    "vacuum__HassVacuumReturnToBase": "Return vacuum to base.",
    "vacuum__HassVacuumStart": "Start vacuum.",
    "todo__get_items": "Read todo items.",
    "todo__HassListAddItem": "Add todo item.",
    "todo__HassListCompleteItem": "Complete todo item.",
    "todo__HassListRemoveItem": "Remove todo item.",
    "assist_satellite__HassBroadcast": "Broadcast a message.",
    "llm__GetDateTime": "Get current date and time.",
}

_DOMAIN_ALIASES = {
    "light": {"licht", "lichter", "lampe", "lampen", "led"},
    "climate": {"thermostat", "heizung", "klima", "klimaanlage"},
    "cover": {"rollladen", "rolllaeden", "jalousie", "jalousien", "markise", "vorhang"},
    "vacuum": {"staubsauger", "saugroboter", "vacuum"},
    "lock": {"schloss", "tuerschloss", "turschloss", "lock"},
    "switch": {"schalter", "switch"},
}


def _normalized(value: str) -> str:
    text = str(value).casefold().replace("ß", "ss")
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _contains(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _is_ha_system(message) -> bool:
    return (
        message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and "Static Context:" in message["content"]
        and "Home Assistant" in message["content"]
    )


def _has_tool_history(messages) -> bool:
    return any(message.get("role") == "tool" or message.get("tool_calls") for message in messages)


def _parse_entity(entry: str):
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
    return [tool.get("function", {}).get("name", "") for tool in tools or []]


def _deterministic_capability(query: str) -> str | None:
    text = _normalized(query)
    if re.search(r"\b(wie hell|heller|dunkler|helligkeit|helligkeit|brightness)\b", text):
        return "light.brightness"
    if re.search(r"\b(warmweiss|warm weiss|kaltweiss|kalt weiss|farbtemperatur)\b", text):
        return "light.temperature"
    if re.search(r"\b(farbe|rot|gruen|blau|gelb|orange|violett|lila|pink|weiss)\b", text) and re.search(
        r"\b(licht|lichter|lampe|lampen|led)\b", text
    ):
        return "light.color"
    if re.search(r"\b(temperatur|grad|waermer|kaelter)\b", text) and re.search(
        r"\b(heizung|thermostat|klima|klimaanlage|temperatur)\b", text
    ):
        return "climate.temperature"
    if re.search(r"\b(prozent|position)\b", text) and re.search(
        r"\b(rollladen|rolllaeden|jalousie|jalousien|markise|vorhang)\b", text
    ):
        return "cover.position"
    if re.search(r"\b(stop|stopp|stoppe|anhalten)\b", text) and re.search(
        r"\b(rollladen|rolllaeden|jalousie|jalousien|markise|vorhang)\b", text
    ):
        return "cover.stop"
    if re.search(r"\b(staubsauger|saugroboter|vacuum)\b", text):
        return "vacuum.control"
    if re.search(r"\b(einkaufsliste|todo|aufgabenliste|liste)\b", text):
        if re.search(r"\b(was|welche|zeige|lies|vorlesen|steht)\b", text):
            return "todo.read"
        return "todo.modify"
    if re.search(r"\b(uhrzeit|datum|welcher tag|wie spaet)\b", text):
        return "datetime"
    if re.search(r"\b(durchsage|broadcast|sage .* bescheid|verkuende)\b", text):
        return "broadcast"
    return None


def _semantic_capability(query: str, encoder, embedding_cache):
    if encoder is None or not query:
        return None, None, None
    import numpy as np

    query_vector = _embedding(encoder, query, embedding_cache)
    ranked = []
    for name, description in _CAPABILITIES.items():
        similarity = float(np.dot(
            query_vector,
            _embedding(encoder, description, embedding_cache),
        ))
        ranked.append((similarity, name))
    ranked.sort(reverse=True)
    best_score, best_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -1.0
    # A semantic route must be both reasonably strong and distinct from the
    # runner-up.  Otherwise the already selected HA tools remain authoritative.
    if best_score >= 0.50 and best_score - second_score >= 0.025:
        return best_name, best_score, second_score
    return None, best_score, second_score


def _capability(query: str, tools, encoder, embedding_cache):
    deterministic = _deterministic_capability(query)
    if deterministic is not None:
        return deterministic, "deterministic", None

    tool_caps = {_TOOL_CAPABILITY.get(name) for name in _tool_names(tools)} - {None}
    if len(tool_caps) == 1:
        return next(iter(tool_caps)), "selected_tool", None

    semantic, best, second = _semantic_capability(query, encoder, embedding_cache)
    if semantic is not None:
        return semantic, "minilm", {"best": best, "second": second}
    return None, "selected_tools", {"best": best, "second": second}


def _domain(query: str, capability: str | None):
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
    text = _normalized(query)
    areas = sorted({item["area"] for item in entities if item["area"]}, key=len, reverse=True)
    matches = [area for area in areas if _contains(text, _normalized(area))]
    return matches[0] if len(matches) == 1 else None


def _generic_entity_name(name: str, domain: str | None):
    normalized = _normalized(name)
    if domain is None:
        return False
    return normalized in {_normalized(alias) for alias in _DOMAIN_ALIASES.get(domain, set())}


def _relevant_entities(source_messages, prepared_messages, query: str, domain: str | None, area: str | None):
    source = _entities(source_messages)
    text = _normalized(query)

    if area is not None:
        matches = [item for item in source if item["area"] == area]
        if domain is not None:
            matches = [item for item in matches if item["domain"] == domain]
        explicit = [
            item for item in matches
            if not _generic_entity_name(item["name"], domain)
            and _contains(text, _normalized(item["name"]))
        ]
        if explicit:
            return explicit[:8]
        return matches[:12]

    explicit = [
        item for item in source
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
    tools = list(tools or [])
    preferred = _CAPABILITY_TOOLS.get(capability)
    if not preferred:
        return tools
    selected = [tool for tool in tools if tool.get("function", {}).get("name") in preferred]
    return selected or tools


def _capability_properties(tool_name: str, capability: str | None):
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
    compact = copy.deepcopy(tool)
    function = compact.get("function", {})
    name = function.get("name", "")
    parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
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
        if prop_name == "domain" and domain is not None:
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
        **({"required": [item for item in parameters.get("required", []) if item in new_properties]}
           if parameters.get("required") else {}),
        "additionalProperties": False,
    }
    return compact


def _minimal_system(capability, domain, area, entities, tools):
    lines = [
        "Home Assistant. Nutze nur bereitgestellte Tools. Erfinde keine Geräte oder Werte.",
        "Bei Bereichszielen nutze area; bei einem bestimmten Gerät name. Antworte kurz.",
    ]
    tool_names = set(_tool_names(tools))
    if "homeassistant__GetLiveContext" in tool_names:
        lines.append("Für aktuelle Zustände oder Werte GetLiveContext aufrufen.")
    elif tools:
        lines.append("Für die angeforderte Steuerung ein bereitgestelltes Tool aufrufen.")
    if capability:
        lines.append(f"Aufgabe: {capability}")
    context = []
    if domain:
        context.append(f"domain={domain}")
    if area:
        context.append(f"area={area}")
    if context:
        lines.append("Kontext: " + "; ".join(context))
    if entities:
        lines.append("Geräte: " + " | ".join(item["name"] for item in entities))
    if domain and not area and len(entities) > 1 and any(
        name in tool_names for name in {"intent__HassTurnOn", "intent__HassTurnOff"}
    ):
        lines.append("Keine Sammelaktion ohne eindeutigen Bereich ausführen.")
    return "\n".join(lines)


def _replace_ha_system(messages, system_text):
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
    properties = 0
    for tool in tools or []:
        params = tool.get("function", {}).get("parameters", {})
        if isinstance(params, dict) and isinstance(params.get("properties"), dict):
            properties += len(params["properties"])
    rendered = json.dumps(tools or [], ensure_ascii=False, separators=(",", ":"), default=str)
    return {"tools": len(tools or []), "properties": properties, "characters": len(rendered)}


def compile_ha_prompt(source_request, prepared_request, *, encoder=None, embedding_cache=None):
    """Return a request rebuilt from request-specific HA context and schemas."""
    if _has_tool_history(source_request.messages):
        return prepared_request, None
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
    tools = _filter_tools(prepared_request.tools, capability)
    compact_tools = [
        _compact_tool(tool, capability, domain, area, relevant)
        for tool in tools
    ]
    system_text = _minimal_system(capability, domain, area, relevant, compact_tools)
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


def install():
    """Install the compiler after deterministic HA routing/verification hooks."""
    from . import runtime

    backend_cls = runtime.HailoBackend
    if getattr(backend_cls, "_ha_prompt_compiler_installed", False):
        return
    original_select_tools = backend_cls.select_tools

    @wraps(original_select_tools)
    def select_tools(self, request):
        source_request = request
        prepared = original_select_tools(self, request)
        compiled, plan = compile_ha_prompt(
            source_request,
            prepared,
            encoder=self.minilm,
            embedding_cache=self._retrieval_embedding_cache,
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
        if self.settings.debug_log:
            _LOG.debug(
                "event=ha_prompt_plan request_id=%s json=%s",
                request_id,
                json.dumps(plan, ensure_ascii=False, separators=(",", ":"), default=str),
            )
        return compiled

    backend_cls.select_tools = select_tools
    backend_cls._ha_prompt_compiler_installed = True
