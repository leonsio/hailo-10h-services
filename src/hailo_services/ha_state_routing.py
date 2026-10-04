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
from functools import wraps

from .tool_retrieval import _static_context_parts, latest_user_text

_LOG = logging.getLogger(__name__)
_LIVE_TOOL = "homeassistant__GetLiveContext"
_FINAL_SYSTEM = (
    "Beantworte die Nutzerfrage kurz und ausschließlich anhand der folgenden "
    "Home-Assistant-Live-Daten. Erfinde keine Zustände oder Werte."
)
_DOMAIN_WORDS = {
    "light": {"licht", "lichter", "lampe", "lampen", "led"},
    "switch": {"schalter", "switch"},
    "cover": {"rollladen", "rolllaeden", "jalousie", "jalousien", "markise"},
    "lock": {"schloss", "schloesser", "tuerschloss", "turschloss", "lock"},
    "climate": {"thermostat", "heizung", "klima", "klimaanlage"},
    "vacuum": {"staubsauger", "saugroboter", "vacuum"},
}
_STATE_ALIASES = {
    "an": "on",
    "on": "on",
    "ein": "on",
    "aus": "off",
    "off": "off",
    "offen": "open",
    "geoeffnet": "open",
    "geschlossen": "closed",
    "zu": "closed",
    "gesperrt": "locked",
    "verriegelt": "locked",
    "locked": "locked",
    "entsperrt": "unlocked",
    "entriegelt": "unlocked",
    "unlocked": "unlocked",
}
_STATE_LABELS = {
    "on": "an",
    "off": "aus",
    "open": "offen",
    "closed": "geschlossen",
    "locked": "gesperrt",
    "unlocked": "entsperrt",
    "home": "zu Hause",
    "not_home": "nicht zu Hause",
    "idle": "inaktiv",
    "cleaning": "reinigt",
    "docked": "an der Ladestation",
    "unavailable": "nicht verfügbar",
    "unknown": "unbekannt",
}
_DOMAIN_LABELS = {
    "light": ("Licht", "Lichter"),
    "switch": ("Schalter", "Schalter"),
    "cover": ("Rollladen", "Rollläden"),
    "lock": ("Schloss", "Schlösser"),
    "vacuum": ("Staubsauger", "Staubsauger"),
    "climate": ("Thermostat", "Thermostate"),
    "sensor": ("Sensor", "Sensoren"),
    "binary_sensor": ("Sensor", "Sensoren"),
}


def _normalized(value: str) -> str:
    text = str(value).casefold().replace("ß", "ss")
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _contains(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _has_action_verb(text: str) -> bool:
    return bool(re.search(
        r"\b(schalt\w*|mach\w*|einschalten|ausschalten|anschalten|anmachen|"
        r"ausmachen|aktiviere\w*|deaktiviere\w*|stell\w*|setze\w*|oeffne\w*|"
        r"schliesse\w*|verriegel\w*|entriegel\w*)\b",
        _normalized(text),
    ))


def _expected_state(text: str) -> str | None:
    normalized = _normalized(text)
    for word, state in _STATE_ALIASES.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            return state
    return None


def _query_kind(text: str) -> str | None:
    """Return the deterministic read-only question kind, if recognized."""
    normalized = _normalized(text)
    if not normalized or _has_action_verb(normalized):
        return None
    if re.search(r"\b(wie viele|wieviele)\b", normalized) and _expected_state(normalized):
        return "count"
    if re.search(r"\b(welche|welcher|welches)\b", normalized) and _expected_state(normalized):
        return "list"
    if re.search(r"\b(temperatur|wie warm|wie kalt)\b", normalized):
        return "temperature"
    if re.search(r"\b(luftfeuchtigkeit|feuchtigkeit)\b", normalized):
        return "humidity"
    if re.search(r"\b(status|zustand|modus)\b", normalized):
        return "status"
    if _expected_state(normalized) and re.search(r"\b(ist|sind|steht|stehen)\b", normalized):
        return "all" if re.search(r"\balle\b", normalized) else "boolean"
    return None


def _is_state_question(text: str) -> bool:
    return _query_kind(text) is not None


def _entries(messages) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    for message in messages:
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
                entities.append({
                    "name": name.group(1).strip(),
                    "domain": domain.group(1).strip(),
                    "area": area.group(1).strip() if area else "",
                })
    return entities


def _live_tool(tools):
    for tool in tools or []:
        if tool.get("function", {}).get("name") == _LIVE_TOOL:
            return tool
    return None


def _query_domain(query: str) -> str | None:
    words = set(_normalized(query).split())
    matches = [domain for domain, aliases in _DOMAIN_WORDS.items() if words & aliases]
    return matches[0] if len(matches) == 1 else None


def _measurement_members(members, kind: str):
    if kind == "temperature":
        hints = {"temperatur", "temperature", "thermostat"}
        preferred_domains = {"sensor", "climate"}
    elif kind == "humidity":
        hints = {"luftfeuchtigkeit", "feuchtigkeit", "humidity"}
        preferred_domains = {"sensor", "climate"}
    else:
        return []
    matches = [
        entity for entity in members
        if entity["domain"] in preferred_domains
        and set(_normalized(entity["name"]).split()) & hints
    ]
    if matches:
        return matches
    return [entity for entity in members if entity["domain"] in preferred_domains]


def _target_arguments(messages, query: str):
    entities = _entries(messages)
    if not entities:
        return None
    normalized_query = _normalized(query)
    domain_hint = _query_domain(query)
    kind = _query_kind(query)

    explicit = [
        entity for entity in entities
        if _contains(normalized_query, _normalized(entity["name"]))
        and (domain_hint is None or entity["domain"] == domain_hint)
    ]
    explicit_keys = {(item["name"], item["domain"], item["area"]) for item in explicit}
    if len(explicit_keys) == 1:
        entity = explicit[0]
        return {"name": entity["name"], "domain": [entity["domain"]]}
    if explicit:
        return None

    area_names = sorted({entity["area"] for entity in entities if entity["area"]}, key=len, reverse=True)
    matching_areas = [area for area in area_names if _contains(normalized_query, _normalized(area))]
    if len(matching_areas) == 1:
        area = matching_areas[0]
        members = [entity for entity in entities if entity["area"] == area]
        if domain_hint is not None:
            members = [entity for entity in members if entity["domain"] == domain_hint]
        elif kind in {"temperature", "humidity"}:
            members = _measurement_members(members, kind)
            unique = {(item["name"], item["domain"]) for item in members}
            if len(unique) == 1:
                entity = members[0]
                return {"name": entity["name"], "domain": [entity["domain"]]}
            return None
        domains = {entity["domain"] for entity in members}
        if len(domains) == 1:
            return {"area": area, "domain": [next(iter(domains))]}
        if kind in {"status", "list", "count", "all"} and domain_hint is None:
            return {"area": area}
        return None
    if matching_areas:
        return None

    # Safe whole-home aggregate queries can use a domain-only target.  We do
    # not do this for singular yes/no questions such as "Ist das Licht an?".
    if domain_hint is not None and kind in {"list", "count", "all"}:
        return {"domain": [domain_hint]}
    return None


def direct_live_context_response(request):
    """Return a direct GetLiveContext call for an unambiguous state question."""
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
        "tool_calls": [{
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": _LIVE_TOOL,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
    }


def _latest_user_index(messages):
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return 0


def _live_followup(messages):
    """Return question, calls and results for the latest pure LiveContext round."""
    start = _latest_user_index(messages)
    question = latest_user_text(messages).strip()
    if not _is_state_question(question):
        return None
    calls = []
    results = []
    for message in messages[start + 1:]:
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


def _block_value(block: str, key: str):
    match = re.search(
        rf"(?mi)^\s+{re.escape(key)}:\s*['\"]?([^'\"\n]+)['\"]?\s*$",
        block,
    )
    return match.group(1).strip() if match else None


def _entities_from_live_text(text: str):
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
            "current_temperature", "temperature", "current_humidity", "humidity",
            "unit_of_measurement", "unit", "device_class",
        ):
            value = _block_value(block, key)
            if value is not None:
                attributes[key] = value
        entities.append({
            "name": name.group(1).strip(),
            "domain": domain or "",
            "state": state.casefold(),
            "area": _block_value(block, "areas") or "",
            "attributes": attributes,
        })
    return entities


def _entities_from_mapping(payload: dict, call):
    entities = []
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
            entities.append({
                "name": str(name), "domain": domain, "state": value.casefold(),
                "area": arguments.get("area", ""), "attributes": {},
            })
        elif isinstance(value, dict) and "state" in value:
            attributes = value.get("attributes") if isinstance(value.get("attributes"), dict) else {}
            entities.append({
                "name": str(name),
                "domain": str(value.get("domain") or domain),
                "state": str(value["state"]).casefold(),
                "area": str(value.get("area") or arguments.get("area", "")),
                "attributes": attributes,
            })
    return entities


def _live_entities(results, calls):
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
    return _STATE_LABELS.get(str(state).casefold(), str(state))


def _domain_label(entities, count: int) -> str:
    domains = {entity.get("domain", "") for entity in entities if entity.get("domain")}
    if len(domains) == 1:
        singular, plural = _DOMAIN_LABELS.get(next(iter(domains)), ("Gerät", "Geräte"))
        return singular if count == 1 else plural
    return "Gerät" if count == 1 else "Geräte"


def _format_number(value: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    rendered = f"{number:g}"
    return rendered.replace(".", ",")


def _measurement_value(entity, kind: str):
    attrs = entity.get("attributes", {})
    if kind == "temperature":
        value = attrs.get("current_temperature") or attrs.get("temperature")
    else:
        value = attrs.get("current_humidity") or attrs.get("humidity")
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
    """Answer common LiveContext status questions directly, without Gemma."""
    followup = _live_followup(request.messages)
    if followup is None:
        return None
    question, calls, results = followup
    entities = _live_entities(results, calls)
    if not entities:
        return None
    kind = _query_kind(question)
    expected = _expected_state(question)

    if kind in {"temperature", "humidity"}:
        values = [
            (entity["name"], _measurement_value(entity, kind))
            for entity in entities
        ]
        values = [(name, value) for name, value in values if value is not None]
        if not values:
            return None
        if len(values) == 1:
            noun = "Temperatur" if kind == "temperature" else "Luftfeuchtigkeit"
            return f"Die {noun} beträgt {values[0][1]}."
        return "; ".join(f"{name}: {value}" for name, value in values) + "."

    if kind == "status" or expected is None:
        if len(entities) == 1:
            entity = entities[0]
            return f"{entity['name']} ist {_state_label(entity['state'])}."
        return "; ".join(
            f"{entity['name']}: {_state_label(entity['state'])}" for entity in entities
        ) + "."

    matching = [entity for entity in entities if entity["state"] == expected]
    other = [entity for entity in entities if entity["state"] != expected]
    label = _domain_label(entities, len(entities))
    expected_label = _state_label(expected)

    if kind == "count":
        return f"{len(matching)} von {len(entities)} {label} sind {expected_label}."
    if kind == "list":
        if not matching:
            return f"Keine {label} sind {expected_label}."
        return f"{expected_label.capitalize()}: " + ", ".join(
            entity["name"] for entity in matching
        ) + "."

    if len(entities) == 1:
        entity = entities[0]
        if matching:
            return f"Ja, {entity['name']} ist {expected_label}."
        return f"Nein, {entity['name']} ist {_state_label(entity['state'])}."

    if not other:
        return f"Ja, alle {len(entities)} {label} sind {expected_label}."
    if not matching:
        states = "; ".join(
            f"{entity['name']}: {_state_label(entity['state'])}" for entity in entities
        )
        return f"Nein, keines der {len(entities)} {label} ist {expected_label}. {states}."
    matching_names = ", ".join(entity["name"] for entity in matching)
    other_states = "; ".join(
        f"{entity['name']}: {_state_label(entity['state'])}" for entity in other
    )
    return (
        f"Teilweise: {len(matching)} von {len(entities)} {label} sind {expected_label}: "
        f"{matching_names}. Abweichend: {other_states}."
    )


def compact_live_followup_request(request):
    """Fallback: reduce an unhandled state follow-up before sending it to Gemma."""
    followup = _live_followup(request.messages)
    if followup is None:
        return None
    question, _, results = followup
    live_text = "\n".join(str(result) for result in results)
    messages = [
        {"role": "system", "content": _FINAL_SYSTEM},
        {
            "role": "user",
            "content": f"Frage: {question}\nHome-Assistant-Live-Daten:\n{live_text}",
        },
    ]
    prepared = request.model_copy(update={
        "messages": messages,
        "tools": None,
        "tool_choice": None,
    })
    prepared._request_id = getattr(request, "_request_id", "-")
    return prepared


def install():
    """Install deterministic state-query routing after the general HA hooks."""
    from . import runtime

    backend_cls = runtime.HailoBackend
    litert_cls = runtime.LiteRTLMBackend
    if getattr(backend_cls, "_ha_state_routing_installed", False):
        return
    original_select_tools = backend_cls.select_tools
    original_litert_chat = litert_cls.chat

    @wraps(original_select_tools)
    def select_tools(self, request):
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
            if self.settings.debug_log:
                _LOG.debug(
                    "event=ha_state_route request_id=%s json=%s",
                    request_id,
                    json.dumps({
                        "route": "direct_state_response",
                        "response": direct_answer,
                    }, ensure_ascii=False, separators=(",", ":"), default=str),
                )
            return prepared

        compact = compact_live_followup_request(request)
        if compact is not None:
            _LOG.info(
                "ha_route request_id=%s route=live_context_followup_minimal tools=0",
                request_id,
            )
            if self.settings.debug_log:
                _LOG.debug(
                    "event=ha_state_route request_id=%s json=%s",
                    request_id,
                    json.dumps({
                        "route": "live_context_followup_minimal",
                        "messages": compact.messages,
                    }, ensure_ascii=False, separators=(",", ":"), default=str),
                )
            return compact

        query = latest_user_text(request.messages).strip()
        live_tool = _live_tool(request.tools)
        if live_tool is not None and _is_state_question(query):
            direct = direct_live_context_response(request)
            if direct is not None:
                prepared = request.model_copy(update={
                    "tools": [live_tool],
                    "tool_choice": None,
                })
                prepared._request_id = request_id
                object.__setattr__(prepared, "_direct_ha_response", direct)
                _LOG.info(
                    "ha_route request_id=%s route=direct_live_context skipped_gemma=true",
                    request_id,
                )
                if self.settings.debug_log:
                    _LOG.debug(
                        "event=ha_state_route request_id=%s json=%s",
                        request_id,
                        json.dumps({
                            "route": "direct_live_context",
                            "tool_call": direct["tool_calls"][0],
                        }, ensure_ascii=False, separators=(",", ":"), default=str),
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
            return original_select_tools(self, narrowed)

        return original_select_tools(self, request)

    @wraps(original_litert_chat)
    def litert_chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        direct_answer = getattr(request, "_direct_ha_state_response", None)
        if direct_answer is not None:
            request_id = getattr(request, "_request_id", "-")
            _LOG.info("direct_ha_state_response request_id=%s skipped_gemma=true", request_id)
            if emit:
                emit(direct_answer)
            return direct_answer
        return original_litert_chat(self, request, emit, cancelled, tools_prepared)

    backend_cls.select_tools = select_tools
    litert_cls.chat = litert_chat
    backend_cls._ha_state_routing_installed = True
