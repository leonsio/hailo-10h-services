"""Deterministic Home Assistant routing for weather and ambient measurements.

Current weather and ambient temperature questions should not be interpreted as
climate-control commands.  This layer prefers dedicated HA weather/ambient
sensors, rejects clearly stale/off climate devices for generic measurements,
and only falls back to a tiny Gemma prompt when several plausible live sources
remain ambiguous.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from functools import wraps

from .ha_state_routing import (
    _entries,
    _has_action_verb,
    _live_entities,
    _live_tool,
    _measurement_value,
    _normalized,
)
from .tool_retrieval import latest_user_text

_LOG = logging.getLogger(__name__)
_LIVE_TOOL = "homeassistant__GetLiveContext"
_INVALID_STATES = {"", "unknown", "unavailable", "none", "null"}
_WEATHER_WORDS = {
    "wetter",
    "draussen",
    "aussen",
    "aussentemperatur",
    "wetterstation",
}
_EQUIPMENT_WORDS = {
    "grill",
    "probe",
    "pit",
    "deye",
    "solarflow",
    "zelle",
    "zelltemperatur",
    "geraetetemperatur",
    "kessel",
    "vorlauf",
    "warmwasser",
    "speicher",
    "backofen",
    "food",
    "akku",
    "batterie",
    "wechselrichter",
}


def _contains_word(text: str, words: set[str]) -> bool:
    tokens = set(_normalized(text).split())
    return bool(tokens & words)


def _weather_query(text: str) -> bool:
    normalized = _normalized(text)
    if _has_action_verb(normalized):
        return False
    return bool(re.search(r"\b(wetter|wetterlage|draussen|aussen)\b", normalized)) and not bool(
        re.search(r"\b(licht|lampe|rollladen|schalter)\b", normalized)
    )


def _ambient_temperature_query(text: str) -> bool:
    normalized = _normalized(text)
    if _has_action_verb(normalized):
        return False
    if re.search(r"\b(thermostat|heizung|klima|klimaanlage)\b", normalized):
        return False
    return bool(
        re.search(r"\b(wie warm|wie kalt|welche temperatur|was fuer eine temperatur|temperatur ist)\b", normalized)
    )


def _explicit_area(query: str, entities: list[dict[str, str]]) -> str | None:
    normalized = _normalized(query)
    areas = sorted({item["area"] for item in entities if item.get("area")}, key=len, reverse=True)
    matches = [area for area in areas if f" {_normalized(area)} " in f" {normalized} "]
    return matches[0] if len(matches) == 1 else None


def _is_weather_sensor(entity: dict[str, str]) -> bool:
    name = _normalized(entity.get("name", ""))
    domain = entity.get("domain", "")
    if domain == "weather":
        return True
    if domain != "sensor":
        return False
    return (
        "wetterstation" in name
        or "aussentemperatur" in name
        or "aussen temperatur" in name
        or "outdoor temperature" in name
    )


def _weather_sources(entities: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    area = _explicit_area(query, entities)
    sources = [item for item in entities if _is_weather_sensor(item)]
    if area is not None:
        in_area = [item for item in sources if item.get("area") == area]
        if in_area:
            sources = in_area
    weather_entities = [item for item in sources if item.get("domain") == "weather"]
    if weather_entities:
        return weather_entities[:2]
    station = [item for item in sources if "wetterstation" in _normalized(item.get("name", ""))]
    # Prefer temperature/humidity/dew/frost-point values from one weather station.
    station.sort(key=lambda item: _weather_sensor_order(item.get("name", "")))
    return station[:4]


def _weather_sensor_order(name: str) -> int:
    normalized = _normalized(name)
    if "temperatur" in normalized and "frost" not in normalized:
        return 0
    if "luftfeuchte" in normalized or "humidity" in normalized:
        return 1
    if "frostpunkt" in normalized or "taupunkt" in normalized or "dew" in normalized:
        return 2
    return 3


def _temperature_score(entity: dict[str, str], area: str | None) -> int:
    name = _normalized(entity.get("name", ""))
    domain = entity.get("domain", "")
    if area is not None and entity.get("area") != area:
        return -1000
    if domain not in {"sensor", "climate"}:
        return -1000
    if not re.search(r"\b(temperatur|temperature|thermostat)\b", name):
        return -1000

    score = 80 if domain == "sensor" else 15
    if "wetterstation" in name:
        score += 80
    if "aussentemperatur" in name or "aussen temperatur" in name:
        score += 70
    if "wandthermostat temperatur" in name or "raumtemperatur" in name:
        score += 45
    if name == "temperatur" or name.endswith(" temperatur"):
        score += 20
    if "stellantrieb" in name:
        score -= 25
    if set(name.split()) & _EQUIPMENT_WORDS:
        score -= 100
    return score


def _temperature_sources(entities: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    area = _explicit_area(query, entities)
    ranked = sorted(
        ((_temperature_score(item, area), index, item) for index, item in enumerate(entities)),
        key=lambda row: (-row[0], row[1]),
    )
    viable = [item for score, _, item in ranked if score >= 50]
    if viable:
        best_score = _temperature_score(viable[0], area)
        close = [item for item in viable if _temperature_score(item, area) >= best_score - 15]
        return close[:4]

    # If no proper ambient sensor exists, query one climate device only as a
    # last resort.  Its live state is checked later; state=off is not accepted
    # as a current ambient reading for a generic temperature question.
    climate = [
        item for score, _, item in ranked
        if item.get("domain") == "climate" and score > -1000
    ]
    return climate[:1]


def _tool_call(source: dict[str, str]) -> dict:
    arguments: dict[str, object] = {
        "name": source["name"],
        "domain": [source["domain"]],
    }
    if source.get("area"):
        arguments["area"] = source["area"]
    return {
        "id": "call_" + uuid.uuid4().hex,
        "type": "function",
        "function": {
            "name": _LIVE_TOOL,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _direct_calls(sources: list[dict[str, str]]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call(source) for source in sources],
    }


def _latest_round(messages):
    user_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            user_index = index
            break
    if user_index < 0:
        return None
    query = latest_user_text(messages).strip()
    calls = []
    result_by_id = {}
    for message in messages[user_index + 1 :]:
        for call in message.get("tool_calls", []) or []:
            if call.get("function", {}).get("name") != _LIVE_TOOL:
                return None
            calls.append(call)
        if message.get("role") == "tool":
            result_by_id[message.get("tool_call_id")] = message.get("content", "")
    if not calls:
        return None
    call_ids = {call.get("id") for call in calls}
    if set(result_by_id) != call_ids:
        return None
    return query, calls, [result_by_id[call.get("id")] for call in calls]


def _live_value(entity: dict, kind: str, *, generic_temperature: bool = False) -> str | None:
    state = str(entity.get("state", "")).casefold()
    if state in _INVALID_STATES:
        return None
    if generic_temperature and entity.get("domain") == "climate" and state == "off":
        return None
    return _measurement_value(entity, kind)


def _weather_response(entities: list[dict]) -> str:
    usable = [
        entity for entity in entities
        if entity.get("domain") == "weather" or "wetterstation" in _normalized(entity.get("name", ""))
    ]
    if not usable:
        return "Aktuell habe ich keine aktuellen Wetterdaten aus Home Assistant."

    weather = next((item for item in usable if item.get("domain") == "weather"), None)
    temperature = None
    humidity = None
    frostpoint = None
    condition = None
    if weather is not None:
        state = str(weather.get("state", "")).casefold()
        if state not in _INVALID_STATES:
            condition = _weather_condition(state)
            temperature = _live_value(weather, "temperature")
            humidity = _live_value(weather, "humidity")

    for entity in usable:
        name = _normalized(entity.get("name", ""))
        if temperature is None and "temperatur" in name and "frost" not in name:
            temperature = _live_value(entity, "temperature", generic_temperature=True)
        if humidity is None and ("luftfeuchte" in name or "humidity" in name):
            humidity = _live_value(entity, "humidity") or _numeric_state(entity)
        if frostpoint is None and ("frostpunkt" in name or "taupunkt" in name or "dew" in name):
            frostpoint = _live_value(entity, "temperature", generic_temperature=True)

    if not any((condition, temperature, humidity, frostpoint)):
        return "Aktuell sind die Wetterdaten in Home Assistant nicht verfügbar."
    parts = []
    if condition:
        parts.append(f"Draußen ist es {condition}")
    if temperature:
        parts.append(("die Temperatur beträgt " if parts else "Draußen sind es ") + temperature)
    if humidity:
        parts.append(f"die Luftfeuchtigkeit beträgt {humidity}")
    sentence = ", ".join(parts) + "." if parts else ""
    if frostpoint:
        sentence += (" " if sentence else "") + f"Der Frostpunkt liegt bei {frostpoint}."
    return sentence


def _numeric_state(entity: dict) -> str | None:
    state = str(entity.get("state", ""))
    if state.casefold() in _INVALID_STATES:
        return None
    try:
        value = float(state)
    except (TypeError, ValueError):
        return None
    rendered = f"{value:g}".replace(".", ",")
    unit = entity.get("attributes", {}).get("unit_of_measurement") or entity.get("attributes", {}).get("unit")
    return rendered + (f" {unit}" if unit else "")


def _weather_condition(state: str) -> str | None:
    return {
        "sunny": "sonnig",
        "clear-night": "klar",
        "cloudy": "bewölkt",
        "partlycloudy": "teilweise bewölkt",
        "rainy": "regnerisch",
        "pouring": "stark regnerisch",
        "snowy": "verschneit",
        "fog": "neblig",
        "windy": "windig",
    }.get(state)


def _temperature_response(entities: list[dict], query: str) -> str | None:
    values = []
    for entity in entities:
        value = _live_value(entity, "temperature", generic_temperature=True)
        if value is not None:
            values.append((entity, value))
    if not values:
        area = _explicit_area(query, _entries_from_live_entities(entities))
        suffix = f" für {area}" if area else ""
        return f"Aktuell habe ich keine aktuellen Temperaturdaten{suffix}."
    if len(values) == 1:
        return f"Die aktuelle Temperatur beträgt {values[0][1]}."
    return None


def _entries_from_live_entities(entities: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "name": str(item.get("name", "")),
            "domain": str(item.get("domain", "")),
            "area": str(item.get("area", "")),
        }
        for item in entities
    ]


def _compact_temperature_decision(request, entities: list[dict], query: str):
    lines = []
    for entity in entities:
        value = _live_value(entity, "temperature", generic_temperature=True)
        lines.append(
            f"- {entity.get('name')}; domain={entity.get('domain')}; "
            f"state={entity.get('state')}; area={entity.get('area')}; value={value or 'nicht aktuell/verfügbar'}"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "Beantworte eine Frage nach der aktuellen Umgebungstemperatur anhand der wenigen "
                "Home-Assistant-Live-Kandidaten. Bevorzuge echte Raum-/Außentemperatursensoren. "
                "Geräte-, Grill-, Akku-, Wechselrichter- oder interne Temperaturen sind keine "
                "Umgebungstemperatur. Ein Climate-Gerät mit state=off/unavailable/unknown ist keine "
                "verlässliche aktuelle Quelle. Wenn keine geeignete aktuelle Quelle existiert, sage "
                "klar, dass aktuell keine aktuellen Temperaturdaten verfügbar sind. Erfinde nichts."
            ),
        },
        {
            "role": "user",
            "content": f"Frage: {query}\nLive-Kandidaten:\n" + "\n".join(lines),
        },
    ]
    prepared = request.model_copy(update={"messages": messages, "tools": None, "tool_choice": None})
    prepared._request_id = getattr(request, "_request_id", "-")
    return prepared


def install():
    from . import runtime

    backend_cls = runtime.HailoBackend
    litert_cls = runtime.LiteRTLMBackend
    if getattr(backend_cls, "_ha_weather_routing_installed", False):
        return
    original_select_tools = backend_cls.select_tools
    original_litert_chat = litert_cls.chat

    @wraps(original_select_tools)
    def select_tools(self, request):
        request_id = getattr(request, "_request_id", "-")
        query = latest_user_text(request.messages).strip()
        live_tool = _live_tool(request.tools)

        round_data = _latest_round(request.messages)
        if round_data is not None and (_weather_query(query) or _ambient_temperature_query(query)):
            _, calls, results = round_data
            entities = _live_entities(results, calls)
            if _weather_query(query):
                answer = _weather_response(entities)
                prepared = request.model_copy(update={"tools": None, "tool_choice": None})
                prepared._request_id = request_id
                object.__setattr__(prepared, "_direct_weather_response", answer)
                _LOG.info(
                    "ha_weather_route request_id=%s route=direct_weather_response entities=%d skipped_gemma=true",
                    request_id,
                    len(entities),
                )
                return prepared
            answer = _temperature_response(entities, query)
            if answer is not None:
                prepared = request.model_copy(update={"tools": None, "tool_choice": None})
                prepared._request_id = request_id
                object.__setattr__(prepared, "_direct_weather_response", answer)
                _LOG.info(
                    "ha_weather_route request_id=%s route=direct_temperature_response entities=%d skipped_gemma=true",
                    request_id,
                    len(entities),
                )
                return prepared
            compact = _compact_temperature_decision(request, entities, query)
            _LOG.info(
                "ha_weather_route request_id=%s route=temperature_decision_minimal entities=%d tools=0",
                request_id,
                len(entities),
            )
            return compact

        if live_tool is None or any(message.get("role") == "tool" for message in request.messages):
            return original_select_tools(self, request)

        static_entities = _entries(request.messages)
        if _weather_query(query):
            sources = _weather_sources(static_entities, query)
            if not sources:
                answer = "Ich habe keine passenden aktuellen Wetterdaten in Home Assistant gefunden."
                prepared = request.model_copy(update={"tools": None, "tool_choice": None})
                prepared._request_id = request_id
                object.__setattr__(prepared, "_direct_weather_response", answer)
                _LOG.info(
                    "ha_weather_route request_id=%s route=no_weather_source skipped_gemma=true",
                    request_id,
                )
                return prepared
            prepared = request.model_copy(update={"tools": [live_tool], "tool_choice": None})
            prepared._request_id = request_id
            object.__setattr__(prepared, "_direct_ha_response", _direct_calls(sources))
            _LOG.info(
                "ha_weather_route request_id=%s route=direct_weather_live sources=%d skipped_gemma=true",
                request_id,
                len(sources),
            )
            return prepared

        if _ambient_temperature_query(query):
            sources = _temperature_sources(static_entities, query)
            if not sources:
                area = _explicit_area(query, static_entities)
                suffix = f" für {area}" if area else ""
                answer = f"Ich habe keinen geeigneten aktuellen Temperatursensor{suffix} gefunden."
                prepared = request.model_copy(update={"tools": None, "tool_choice": None})
                prepared._request_id = request_id
                object.__setattr__(prepared, "_direct_weather_response", answer)
                _LOG.info(
                    "ha_weather_route request_id=%s route=no_temperature_source skipped_gemma=true",
                    request_id,
                )
                return prepared
            prepared = request.model_copy(update={"tools": [live_tool], "tool_choice": None})
            prepared._request_id = request_id
            object.__setattr__(prepared, "_direct_ha_response", _direct_calls(sources))
            _LOG.info(
                "ha_weather_route request_id=%s route=direct_temperature_live sources=%d skipped_gemma=true",
                request_id,
                len(sources),
            )
            return prepared

        return original_select_tools(self, request)

    @wraps(original_litert_chat)
    def litert_chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        answer = getattr(request, "_direct_weather_response", None)
        if answer is not None:
            request_id = getattr(request, "_request_id", "-")
            _LOG.info("direct_ha_weather_response request_id=%s skipped_gemma=true", request_id)
            if emit:
                emit(answer)
            return answer
        return original_litert_chat(self, request, emit, cancelled, tools_prepared)

    backend_cls.select_tools = select_tools
    litert_cls.chat = litert_chat
    backend_cls._ha_weather_routing_installed = True
