"""Generic Home Assistant routing for weather and ambient measurements.

The routing intentionally avoids installation-specific device names and brands.
It relies on Home Assistant domains, areas, device classes, units, and generic
environmental semantics. Ambiguous live data is reduced before Gemma sees it.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from functools import wraps

from . import ha_state_routing as _state
from .i18n import labels, lexicon, t
from .tool_retrieval import latest_user_text

_LOG = logging.getLogger(__name__)
_LIVE_TOOL = "homeassistant__GetLiveContext"
_INVALID_STATES = {"", "unknown", "unavailable", "none", "null"}

_TEMPERATURE_DEVICE_CLASSES = {"temperature"}
_HUMIDITY_DEVICE_CLASSES = {"humidity"}
_DEW_POINT_DEVICE_CLASSES = {"dew_point"}
_WEATHER_DEVICE_CLASSES = {
    "temperature",
    "humidity",
    "dew_point",
    "atmospheric_pressure",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "precipitation_intensity",
}

_TEMPERATURE_UNITS = {"°c", "c", "°f", "f", "k", "kelvin"}
_HUMIDITY_UNITS = {"%"}

_AMBIENT_WORDS = lexicon('ha_weather_routing._AMBIENT_WORDS')
_OUTDOOR_WORDS = lexicon('ha_weather_routing._OUTDOOR_WORDS')
_PROCESS_WORDS = lexicon('ha_weather_routing._PROCESS_WORDS')


def _normalized(value: object) -> str:
    return _state._normalized(str(value))


def _tokens(value: object) -> set[str]:
    return set(_normalized(value).split())


def _has_any_word(value: object, words: set[str]) -> bool:
    return bool(_tokens(value) & words)


def _ambient_temperature_query(text: str) -> bool:
    normalized = _normalized(text)
    if _state._has_action_verb(normalized):
        return False
    if re.search(lexicon('ha_weather_routing.pattern.63.17'), normalized):
        return False
    return bool(
        re.search(
            lexicon('ha_weather_routing.pattern.67.12'),
            normalized,
        )
    )


def _weather_query(text: str) -> bool:
    normalized = _normalized(text)
    if _state._has_action_verb(normalized) or _ambient_temperature_query(text):
        return False
    return bool(
        re.search(lexicon('ha_weather_routing.pattern.79.18'), normalized)
        or re.search(lexicon('ha_weather_routing.pattern.80.21'), normalized)
    )


def _explicit_area(query: str, entities: list[dict[str, str]]) -> str | None:
    normalized = _normalized(query)
    areas = sorted({item["area"] for item in entities if item.get("area")}, key=len, reverse=True)
    matches = [area for area in areas if f" {_normalized(area)} " in f" {normalized} "]
    return matches[0] if len(matches) == 1 else None


def _outdoor_area(query: str, entities: list[dict[str, str]]) -> str | None:
    explicit = _explicit_area(query, entities)
    if explicit:
        return explicit
    if not _has_any_word(query, _OUTDOOR_WORDS):
        return None
    areas = sorted({item["area"] for item in entities if item.get("area")}, key=len, reverse=True)
    matches = [area for area in areas if _has_any_word(area, _OUTDOOR_WORDS)]
    return matches[0] if len(matches) == 1 else None


def _environment_name_score(name: str) -> int:
    words = _tokens(name)
    score = 0
    if words & _AMBIENT_WORDS:
        score += 30
    if words & _OUTDOOR_WORDS:
        score += 20
    if words & _PROCESS_WORDS:
        score -= 45
    return score


def _temperature_score(entity: dict[str, str], area: str | None) -> int:
    if area is not None and entity.get("area") != area:
        return -1000
    domain = entity.get("domain", "")
    if domain not in {"sensor", "climate", "weather"}:
        return -1000
    name = _normalized(entity.get("name", ""))
    if domain == "weather":
        return 120 + _environment_name_score(name)
    if domain == "climate":
        return 45 + _environment_name_score(name)
    if re.search(lexicon('ha_weather_routing.pattern.125.17'), name):
        return 75 + _environment_name_score(name)
    return 20 + _environment_name_score(name)


def _temperature_sources(entities: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    area = _outdoor_area(query, entities) or _explicit_area(query, entities)
    ranked = sorted(
        ((_temperature_score(item, area), index, item) for index, item in enumerate(entities)),
        key=lambda row: (-row[0], row[1]),
    )
    viable = [item for score, _, item in ranked if score >= 40]
    if not viable:
        return []
    best = _temperature_score(viable[0], area)
    return [item for item in viable if _temperature_score(item, area) >= best - 15][:4]


def _is_weather_sensor(entity: dict[str, str]) -> bool:
    if entity.get("domain") == "weather":
        return True
    if entity.get("domain") != "sensor":
        return False
    words = _tokens(entity.get("name", ""))
    return bool(
        words
        & (
            _AMBIENT_WORDS
            | {"humidity", "luftfeuchte", "dew", "taupunkt", "frostpunkt", "pressure", "wind"}
        )
    )


def _weather_sensor_order(name: str) -> int:
    normalized = _normalized(name)
    if re.search(lexicon('ha_weather_routing.pattern.160.17'), normalized) and not re.search(
        lexicon('ha_weather_routing.match.160.8'), normalized
    ):
        return 0
    if re.search(lexicon('ha_weather_routing.pattern.164.17'), normalized):
        return 1
    if re.search(lexicon('ha_weather_routing.match.165.17'), normalized):
        return 2
    return 3


def _weather_sources(entities: list[dict[str, str]], query: str) -> list[dict[str, str]]:
    area = _outdoor_area(query, entities) or _explicit_area(query, entities)
    sources = [item for item in entities if _is_weather_sensor(item)]
    if area is not None:
        scoped = [item for item in sources if item.get("area") == area]
        if scoped:
            sources = scoped
    weather_entities = [item for item in sources if item.get("domain") == "weather"]
    if weather_entities:
        return weather_entities[:2]
    sources.sort(key=lambda item: (_weather_sensor_order(item.get("name", "")), item.get("name", "")))
    return sources[:4]


def _tool_call(arguments: dict[str, object]) -> dict:
    return {
        "id": "call_" + uuid.uuid4().hex,
        "type": "function",
        "function": {
            "name": _LIVE_TOOL,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _direct_call(arguments: dict[str, object]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call(arguments)],
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


def _device_class(entity: dict) -> str:
    return str(entity.get("attributes", {}).get("device_class", "")).casefold().strip()


def _unit(entity: dict) -> str:
    attrs = entity.get("attributes", {})
    return str(attrs.get("unit_of_measurement") or attrs.get("unit") or "").strip()


def _numeric_state(entity: dict) -> str | None:
    state = str(entity.get("state", ""))
    if state.casefold() in _INVALID_STATES:
        return None
    try:
        value = float(state)
    except (TypeError, ValueError):
        return None
    rendered = _state._format_number(value)
    unit = _unit(entity)
    return rendered + (f" {unit}" if unit else "")


def _temperature_value(entity: dict) -> str | None:
    state = str(entity.get("state", "")).casefold()
    if state in _INVALID_STATES:
        return None
    attrs = entity.get("attributes", {})
    domain = entity.get("domain")
    if domain == "climate":
        value = attrs.get("current_temperature")
        if value in {None, ""}:
            return None
        unit = _unit(entity)
        return f"{_state._format_number(value)}{(' ' + unit) if unit else ''}"
    if domain == "weather":
        value = attrs.get("temperature")
        if value in {None, ""}:
            return None
        unit = (
            attrs.get("temperature_unit")
            or attrs.get("unit_of_measurement")
            or attrs.get("unit")
            or ""
        )
        return f"{_state._format_number(value)}{(' ' + str(unit)) if unit else ''}"
    device_class = _device_class(entity)
    unit = _normalized(_unit(entity))
    if device_class in _TEMPERATURE_DEVICE_CLASSES or unit in _TEMPERATURE_UNITS:
        return _numeric_state(entity)
    name = _normalized(entity.get("name", ""))
    if re.search(lexicon('ha_weather_routing.pattern.280.17'), name):
        return _numeric_state(entity)
    return None


def _humidity_value(entity: dict) -> str | None:
    state = str(entity.get("state", "")).casefold()
    if state in _INVALID_STATES:
        return None
    attrs = entity.get("attributes", {})
    if entity.get("domain") == "weather":
        value = attrs.get("humidity")
        if value in {None, ""}:
            return None
        return f"{_state._format_number(value)} %"
    device_class = _device_class(entity)
    normalized_unit = _normalized(_unit(entity))
    name = _normalized(entity.get("name", ""))
    if (
        device_class in _HUMIDITY_DEVICE_CLASSES
        or normalized_unit in _HUMIDITY_UNITS
        or re.search(lexicon('ha_weather_routing.pattern.301.21'), name)
    ):
        return _numeric_state(entity)
    return None


def _dew_point_value(entity: dict) -> str | None:
    if _device_class(entity) in _DEW_POINT_DEVICE_CLASSES:
        return _numeric_state(entity)
    name = _normalized(entity.get("name", ""))
    if re.search(lexicon('ha_weather_routing.match.310.17'), name):
        return _numeric_state(entity)
    return None


def _live_temperature_score(entity: dict, query: str) -> int:
    value = _temperature_value(entity)
    if value is None:
        return -1000
    domain = entity.get("domain", "")
    device_class = _device_class(entity)
    score = 0
    if domain == "weather":
        score += 120
    elif domain == "sensor":
        score += 75
        if device_class in _TEMPERATURE_DEVICE_CLASSES:
            score += 20
    elif domain == "climate":
        score += 45
    score += _environment_name_score(str(entity.get("name", "")))
    if _has_any_word(query, _OUTDOOR_WORDS):
        if _has_any_word(entity.get("area", ""), _OUTDOOR_WORDS):
            score += 30
        if _has_any_word(entity.get("name", ""), _OUTDOOR_WORDS):
            score += 20
    return score


def _strong_ambient_evidence(entity: dict, query: str) -> bool:
    if entity.get("domain") == "weather":
        return True
    name = _normalized(entity.get("name", ""))
    if _has_any_word(name, _AMBIENT_WORDS | _OUTDOOR_WORDS):
        return True
    if _has_any_word(query, _OUTDOOR_WORDS) and _has_any_word(
        entity.get("area", ""), _OUTDOOR_WORDS
    ):
        return True
    return name in lexicon('ha_weather_routing.words.349.19')


def _temperature_response(entities: list[dict], query: str) -> str | None:
    ranked = sorted(
        (
            (_live_temperature_score(entity, query), index, entity, _temperature_value(entity))
            for index, entity in enumerate(entities)
        ),
        key=lambda row: (-row[0], row[1]),
    )
    ranked = [row for row in ranked if row[0] > -1000 and row[3] is not None]
    if not ranked:
        area = _explicit_area(query, _entries_from_live_entities(entities))
        suffix = t('ha_weather_routing.420' , area=area) if area else ""
        return t('ha_weather_routing.421' , suffix=suffix)
    best_score, _, best_entity, best_value = ranked[0]
    if len(ranked) == 1:
        if _strong_ambient_evidence(best_entity, query):
            return t('ha_weather_routing.425' , best_value=best_value)
        return None
    second_score = ranked[1][0]
    if best_score - second_score >= 15 and _strong_ambient_evidence(best_entity, query):
        return t('ha_weather_routing.429' , best_value=best_value)
    return None


def _environmental_sensor(entity: dict) -> bool:
    if entity.get("domain") == "weather":
        return True
    if entity.get("domain") != "sensor":
        return False
    device_class = _device_class(entity)
    if device_class in _WEATHER_DEVICE_CLASSES:
        return True
    name = _normalized(entity.get("name", ""))
    return bool(
        re.search(
            lexicon('ha_weather_routing.pattern.388.12'),
            name,
        )
    )


def _weather_condition(state: str) -> str | None:
    return labels("weather").get(state)


def _pick_weather_group(entities: list[dict], query: str) -> list[dict] | None:
    relevant = [entity for entity in entities if _environmental_sensor(entity)]
    if not relevant:
        return []
    weather_entities = [
        entity for entity in relevant
        if entity.get("domain") == "weather"
        and str(entity.get("state", "")).casefold() not in _INVALID_STATES
    ]
    if len(weather_entities) > 1:
        # Neither catalogue order nor provider order determines the user's source.
        return None
    if weather_entities:
        weather = weather_entities[0]
        area = weather.get("area")
        companions = [
            entity for entity in relevant
            if area and entity.get("area") == area and entity.get("domain") == "sensor"
        ]
        return [weather, *companions]
    relevant = [entity for entity in relevant if entity.get("domain") != "weather"]
    groups: dict[str, list[dict]] = {}
    for entity in relevant:
        area = str(entity.get("area") or "")
        groups.setdefault(area, []).append(entity)
    scored = []
    for area, members in groups.items():
        kinds = set()
        for entity in members:
            if _temperature_value(entity):
                kinds.add("temperature")
            if _humidity_value(entity):
                kinds.add("humidity")
            if _dew_point_value(entity):
                kinds.add("dew_point")
            device_class = _device_class(entity)
            if device_class in {
                "atmospheric_pressure",
                "pressure",
                "wind_speed",
                "wind_direction",
                "precipitation",
                "precipitation_intensity",
            }:
                kinds.add(device_class)
        score = len(kinds) * 20
        if _has_any_word(area, _OUTDOOR_WORDS):
            score += 30
        if any(
            _has_any_word(item.get("name", ""), lexicon('ha_weather_routing.words.432.48'))
            for item in members
        ):
            score += 20
        if _has_any_word(query, _OUTDOOR_WORDS) and _has_any_word(area, _OUTDOOR_WORDS):
            score += 20
        scored.append((score, area, members))
    scored.sort(key=lambda row: (-row[0], row[1]))
    if not scored or scored[0][0] <= 0:
        return []
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][2]


def _weather_response(entities: list[dict], query: str = "") -> str | None:
    group = _pick_weather_group(entities, query)
    if group is None:
        return None
    if not group:
        return t('ha_weather_routing.520')
    weather = next((item for item in group if item.get("domain") == "weather"), None)
    condition = None
    temperature = None
    humidity = None
    dew_point = None
    if weather is not None:
        state = str(weather.get("state", "")).casefold()
        if state not in _INVALID_STATES:
            condition = _weather_condition(state)
        temperature = _temperature_value(weather)
        humidity = _humidity_value(weather)
    temp_candidates = sorted(
        (
            (_live_temperature_score(entity, query), index, entity)
            for index, entity in enumerate(group)
            if _temperature_value(entity) is not None
        ),
        key=lambda row: (-row[0], row[1]),
    )
    if temperature is None and temp_candidates:
        temperature = _temperature_value(temp_candidates[0][2])
    for entity in group:
        if humidity is None:
            humidity = _humidity_value(entity)
        if dew_point is None:
            dew_point = _dew_point_value(entity)
    if not any((condition, temperature, humidity, dew_point)):
        return t('ha_weather_routing.548')
    parts = []
    if condition:
        parts.append(t('ha_weather_routing.551' , condition=condition))
    if temperature:
        parts.append((t('ha_weather_routing.553') if parts else t('ha_weather_routing.553')) + temperature)
    if humidity:
        parts.append(t('ha_weather_routing.555' , humidity=humidity))
    sentence = ", ".join(parts) + "." if parts else ""
    if dew_point:
        sentence += (" " if sentence else "") + t('ha_weather_routing.558' , dew_point=dew_point)
    return sentence


def _entries_from_live_entities(entities: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "name": str(item.get("name", "")),
            "domain": str(item.get("domain", "")),
            "area": str(item.get("area", "")),
        }
        for item in entities
    ]


def _compact_environment_decision(request, entities: list[dict], query: str, kind: str):
    relevant = [
        entity
        for entity in entities
        if _environmental_sensor(entity) or entity.get("domain") == "climate"
    ][:16]
    lines = []
    for entity in relevant:
        attrs = entity.get("attributes", {})
        lines.append(
            f"- {entity.get('name')}; domain={entity.get('domain')}; "
            f"area={entity.get('area')}; state={entity.get('state')}; "
            f"device_class={attrs.get('device_class', '')}; unit={_unit(entity)}; "
            f"current_temperature={attrs.get('current_temperature', '')}"
        )
    task = (
        t('ha_weather_routing.589')
        if kind == "temperature"
        else t('ha_weather_routing.591')
    )
    messages = [
        {
            "role": "system",
            "content": (
                t('ha_weather_routing.597' , task=task)
            ),
        },
        {
            "role": "user",
            "content": t('ha_weather_routing.608' , query=query) + "\n".join(lines),
        },
    ]
    prepared = request.model_copy(update={"messages": messages, "tools": None, "tool_choice": None})
    prepared._request_id = getattr(request, "_request_id", "-")
    return prepared


def _initial_live_arguments(
    static_entities: list[dict[str, str]],
    query: str,
    *,
    kind: str,
) -> dict[str, object]:
    area = _outdoor_area(query, static_entities) or _explicit_area(query, static_entities)
    if kind == "temperature":
        arguments: dict[str, object] = {"domain": ["sensor", "climate", "weather"]}
    else:
        arguments = {"domain": ["weather", "sensor"]}
    if area:
        arguments["area"] = area
    elif kind == "weather" and any(item.get("domain") == "weather" for item in static_entities):
        arguments = {"domain": ["weather"]}
    return arguments


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
        live_tool = _state._live_tool(request.tools)

        round_data = _latest_round(request.messages)
        if round_data is not None and (_weather_query(query) or _ambient_temperature_query(query)):
            _, calls, results = round_data
            entities = _state._live_entities(results, calls)
            if _weather_query(query):
                answer = _weather_response(entities, query)
                if answer is None:
                    compact = _compact_environment_decision(request, entities, query, "weather")
                    _LOG.info(
                        "ha_weather_route request_id=%s route=weather_decision_minimal entities=%d tools=0",
                        request_id,
                        len(entities),
                    )
                    return compact
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

            compact = _compact_environment_decision(request, entities, query, "temperature")
            _LOG.info(
                "ha_weather_route request_id=%s route=temperature_decision_minimal entities=%d tools=0",
                request_id,
                len(entities),
            )
            return compact

        if live_tool is None or any(message.get("role") == "tool" for message in request.messages):
            return original_select_tools(self, request)

        static_entities = _state._entries(request.messages)
        if _weather_query(query):
            arguments = _initial_live_arguments(static_entities, query, kind="weather")
            prepared = request.model_copy(update={"tools": [live_tool], "tool_choice": None})
            prepared._request_id = request_id
            object.__setattr__(prepared, "_direct_ha_response", _direct_call(arguments))
            _LOG.info(
                "ha_weather_route request_id=%s route=generic_weather_live filters=%s skipped_gemma=true",
                request_id,
                json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            )
            return prepared

        if _ambient_temperature_query(query):
            arguments = _initial_live_arguments(static_entities, query, kind="temperature")
            prepared = request.model_copy(update={"tools": [live_tool], "tool_choice": None})
            prepared._request_id = request_id
            object.__setattr__(prepared, "_direct_ha_response", _direct_call(arguments))
            _LOG.info(
                "ha_weather_route request_id=%s route=generic_temperature_live filters=%s skipped_gemma=true",
                request_id,
                json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
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
