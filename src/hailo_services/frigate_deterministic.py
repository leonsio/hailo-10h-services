"""Language-neutral deterministic shortcuts for common Frigate/NVR requests."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from .frigate_prompt import camera_catalogue, server_time, text_content
from .tool_calling import response_message, selected_tools


# Routing uses shared intent categories. Language-specific surface forms live only in
# these data tables so adding another language never changes the routing algorithm.
_LANGUAGE_HINTS = {
    "de": {"was", "zeige", "letzten", "heute", "gestern", "passiert", "kamera", "ereignisse"},
    "en": {"what", "show", "last", "today", "yesterday", "happened", "camera", "events"},
    "fr": {"quoi", "montre", "derniere", "dernières", "aujourd'hui", "hier", "passe", "passé", "camera", "caméra", "evenements", "événements"},
    "es": {"que", "qué", "muestra", "ultimas", "últimas", "hoy", "ayer", "paso", "pasó", "camara", "cámara", "eventos"},
    "it": {"cosa", "mostra", "ultime", "oggi", "ieri", "successo", "telecamera", "eventi"},
    "nl": {"wat", "toon", "afgelopen", "vandaag", "gisteren", "gebeurd", "camera", "gebeurtenissen"},
    "pt": {"que", "mostra", "ultimas", "últimas", "hoje", "ontem", "aconteceu", "camera", "câmera", "eventos"},
    "ru": {"что", "покажи", "последние", "сегодня", "вчера", "произошло", "камера", "события"},
}

_RECAP_WORDS = {
    "event", "events", "activity", "activities", "history", "happened", "incident", "incidents", "recap",
    "ereignis", "ereignisse", "aktivitaet", "aktivitaeten", "aktivität", "aktivitäten", "historie", "passiert", "vorfaelle", "vorfälle",
    "evenement", "evenements", "événement", "événements", "activite", "activites", "activité", "activités", "historique", "passe", "passé",
    "evento", "eventos", "actividad", "actividades", "historial", "paso", "pasó", "ocurrio", "ocurrió",
    "eventi", "attivita", "attività", "cronologia", "successo",
    "gebeurtenis", "gebeurtenissen", "activiteit", "activiteiten", "geschiedenis", "gebeurd",
    "atividade", "atividades", "historico", "histórico", "aconteceu",
    "событие", "события", "активность", "история", "произошло", "случилось",
}

_LIVE_WORDS = {
    "now", "live", "current", "currently", "visible",
    "jetzt", "gerade", "aktuell", "livebild", "sichtbar",
    "maintenant", "actuel", "actuelle", "direct", "visible",
    "ahora", "actual", "directo", "visible",
    "adesso", "ora", "attuale", "diretta", "visibile",
    "nu", "live", "actueel", "zichtbaar",
    "agora", "atual", "direto", "visivel", "visível",
    "сейчас", "прямо", "текущий", "видно",
}

_PRESENCE_WORDS = {
    "there", "see", "show", "visible", "present", "happening",
    "da", "sehen", "zeige", "sichtbar", "passiert",
    "voir", "montre", "visible", "présent", "present", "passe", "passé",
    "hay", "ver", "muestra", "visible", "pasa", "pasando",
    "c'e", "c’è", "vedi", "mostra", "visibile", "succede",
    "er", "zie", "toon", "zichtbaar", "gebeurt",
    "ha", "há", "ver", "mostra", "visivel", "visível", "acontecendo",
    "есть", "видно", "покажи", "вижу", "происходит",
}

_SIMILAR_WORDS = {
    "similar", "same", "like",
    "aehnlich", "ähnlich", "gleich", "derselbe", "dieselbe",
    "similaire", "semblable", "meme", "même",
    "parecido", "mismo",
    "simile", "stesso",
    "vergelijkbaar", "zelfde",
    "semelhante", "mesmo",
    "похож", "похожие", "такой", "же",
}

_LAST_MARKERS = {
    "last", "past", "letzte", "letzten", "letzter", "letztes",
    "dernier", "derniere", "dernière", "derniers", "dernieres", "dernières",
    "ultimo", "ultima", "último", "última", "ultimos", "ultimas", "últimos", "últimas",
    "ultime", "ultimi", "afgelopen", "laatste",
    "последний", "последняя", "последние", "последних",
}

_UNIT_MINUTES = {
    "minute": 1, "minutes": 1, "min": 1, "mins": 1, "minuten": 1,
    "minuto": 1, "minutos": 1, "minuti": 1, "minuut": 1,
    "минута": 1, "минуты": 1, "минут": 1,
    "hour": 60, "hours": 60, "hr": 60, "hrs": 60, "stunde": 60, "stunden": 60,
    "heure": 60, "heures": 60, "hora": 60, "horas": 60, "ora": 60, "ore": 60,
    "uur": 60, "uren": 60, "час": 60, "часа": 60, "часов": 60,
    "day": 1440, "days": 1440, "tag": 1440, "tage": 1440, "tagen": 1440,
    "jour": 1440, "jours": 1440, "dia": 1440, "dias": 1440, "giorno": 1440,
    "giorni": 1440, "dag": 1440, "dagen": 1440, "день": 1440, "дня": 1440,
    "дней": 1440,
    "week": 10080, "weeks": 10080, "woche": 10080, "wochen": 10080,
    "semaine": 10080, "semaines": 10080, "semana": 10080, "semanas": 10080,
    "settimana": 10080, "settimane": 10080, "неделя": 10080, "недели": 10080,
    "недель": 10080,
}

_ONE_WORDS = {
    "one", "a", "an", "ein", "eine", "einer", "einen", "einem",
    "un", "une", "uno", "una", "een", "um", "uma", "один", "одна", "одну",
}

_TODAY = {"today", "heute", "aujourd'hui", "aujourdhui", "hoy", "oggi", "vandaag", "hoje", "сегодня"}
_YESTERDAY = {"yesterday", "gestern", "hier", "ayer", "ieri", "gisteren", "ontem", "вчера"}
_RANGE_FROM = {"from", "von", "de", "desde", "da", "van", "a partir de", "с"}
_RANGE_TO = {"to", "until", "bis", "a", "à", "hasta", "alle", "tot", "ate", "até", "до"}
_SINCE = {"since", "seit", "depuis", "desde", "da", "vanaf", "с"}

_STOP_WORDS = {
    "stop", "cancel", "stoppe", "beende", "annule", "arrete", "arrête", "deten", "detén",
    "ferma", "pare", "останови", "отмени",
}
_WATCH_WORDS = {
    "watch", "monitor", "überwachung", "ueberwachung", "überwache", "ueberwache", "surveille",
    "vigila", "monitora", "bewaak", "monitoriza", "наблюдение", "следи",
}

_FEATURE_ALIASES = {
    "detect": {"detect", "detection", "erkennung", "détection", "deteccion", "detección", "rilevamento", "detectie", "deteccao", "detecção", "обнаружение"},
    "record": {"record", "recording", "aufnahme", "enregistrement", "grabacion", "grabación", "registrazione", "opname", "gravacao", "gravação", "запись"},
    "snapshots": {"snapshot", "snapshots", "schnappschuss", "schnappschuesse", "schnappschüsse", "instantane", "instantané", "captura", "istantanea", "momentopname", "снимок", "снимки"},
    "audio": {"audio", "ton", "son", "sonido", "suono", "geluid", "som", "звук"},
    "motion": {"motion", "bewegung", "mouvement", "movimiento", "movimento", "beweging", "движение"},
    "notifications": {"notification", "notifications", "benachrichtigung", "benachrichtigungen", "notificacion", "notificación", "notifiche", "melding", "meldingen", "notificacao", "notificação", "уведомление", "уведомления"},
}
_ON_WORDS = {
    "on", "enable", "enabled", "ein", "an", "active", "actif", "activo", "attivo", "aan",
    "ligado", "ativado", "включи", "включено",
}
_OFF_WORDS = {
    "off", "disable", "disabled", "aus", "inaktiv", "desactive", "désactive", "desactivado",
    "disattiva", "uit", "desligado", "desativado", "выключи", "выключено",
}
_ACTION_WORDS = {
    "turn", "set", "enable", "disable", "schalte", "aktiviere", "deaktiviere", "active",
    "désactive", "activa", "desactiva", "attiva", "disattiva", "zet", "ative", "desative",
    "включи", "выключи",
}

_MESSAGES = {
    "empty_recap": {
        "de": "Für den abgefragten Zeitraum wurden keine Aktivitäten zurückgegeben.",
        "en": "No activity was returned for the requested time period.",
        "fr": "Aucune activité n’a été renvoyée pour la période demandée.",
        "es": "No se devolvió ninguna actividad para el período solicitado.",
        "it": "Non è stata restituita alcuna attività per il periodo richiesto.",
        "nl": "Er is geen activiteit teruggegeven voor de gevraagde periode.",
        "pt": "Nenhuma atividade foi retornada para o período solicitado.",
        "ru": "За запрошенный период активность не найдена.",
    },
    "future_range": {
        "de": "Der angegebene historische Zeitraum liegt teilweise in der Zukunft. Bitte nenne einen Zeitraum bis zur aktuellen Serverzeit.",
        "en": "The requested historical interval is partly in the future. Please use a time range ending no later than the current server time.",
        "fr": "La période historique demandée se situe en partie dans le futur. Indiquez une période se terminant au plus tard à l’heure actuelle du serveur.",
        "es": "El intervalo histórico solicitado está parcialmente en el futuro. Indica un intervalo que termine como máximo en la hora actual del servidor.",
        "it": "L’intervallo storico richiesto è in parte nel futuro. Indica un intervallo che termini non oltre l’ora corrente del server.",
        "nl": "Het gevraagde historische tijdvak ligt gedeeltelijk in de toekomst. Gebruik een tijdvak dat uiterlijk op de huidige servertijd eindigt.",
        "pt": "O intervalo histórico solicitado está parcialmente no futuro. Use um intervalo que termine, no máximo, na hora atual do servidor.",
        "ru": "Запрошенный исторический интервал частично находится в будущем. Укажите интервал, заканчивающийся не позже текущего времени сервера.",
    },
}


def _normalize(text: str) -> str:
    text = text.casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ÖØ-öø-ÿА-Яа-яЁё'’]+", _normalize(text), flags=re.UNICODE))


def _language(request, text: str) -> str:
    explicit = getattr(request, "language", None)
    if isinstance(explicit, str) and explicit:
        code = explicit.casefold().split("-")[0].split("_")[0]
        if code in _LANGUAGE_HINTS:
            return code
    tokens = _tokens(text)
    scores = {code: len(tokens & words) for code, words in _LANGUAGE_HINTS.items()}
    if re.search(r"[А-Яа-яЁё]", text):
        scores["ru"] += 2
    best = max(scores, key=scores.get)
    return best if scores[best] else "en"


def _message(request, text: str, key: str) -> str:
    lang = _language(request, text)
    return _MESSAGES[key].get(lang, _MESSAGES[key]["en"])


def _local_datetime(value):
    if not isinstance(value, str):
        return None
    for fmt in (
        "%Y-%m-%d at %I:%M:%S %p",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _latest_user_text(request) -> str:
    return next(
        (text_content(m).strip() for m in reversed(request.messages) if m.get("role") == "user"),
        "",
    )


def _user_texts(request) -> list[str]:
    return [text_content(m).strip() for m in request.messages if m.get("role") == "user"]


def _active_tool_results(request) -> dict[str, object]:
    user_indexes = [i for i, m in enumerate(request.messages) if m.get("role") == "user"]
    if not user_indexes:
        return {}
    start = user_indexes[-1]
    calls: dict[str, str] = {}
    results: dict[str, object] = {}
    for message in request.messages[start:]:
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            if isinstance(call.get("id"), str) and isinstance(function.get("name"), str):
                calls[call["id"]] = function["name"]
        if message.get("role") == "tool":
            name = calls.get(message.get("tool_call_id"))
            if not name:
                continue
            try:
                results[name] = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                results[name] = {"error": "Tool returned non-JSON data"}
    return results


def _matched_cameras(text: str, cameras: dict[str, str]) -> list[str]:
    matches = []
    for identifier, friendly in cameras.items():
        if any(
            re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I)
            for value in {identifier, friendly}
        ):
            matches.append(identifier)
    return matches


def _has_any(text: str, words: set[str]) -> bool:
    return bool(_tokens(text) & words)


def _relative_interval(text: str, now: datetime):
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿА-Яа-яЁё]+", _normalize(text), flags=re.UNICODE)
    for i, token in enumerate(tokens):
        if token not in _LAST_MARKERS:
            continue
        for j in range(i + 1, min(len(tokens), i + 4)):
            raw = tokens[j]
            if raw.isdigit():
                amount = int(raw)
            elif raw in _ONE_WORDS:
                amount = 1
            else:
                continue
            if j + 1 >= len(tokens):
                continue
            unit = tokens[j + 1]
            if unit not in _UNIT_MINUTES or amount <= 0:
                continue
            try:
                return now - timedelta(minutes=amount * _UNIT_MINUTES[unit]), now
            except OverflowError:
                return None
    token_set = set(tokens)
    if token_set & _YESTERDAY:
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return today - timedelta(days=1), today
    if token_set & _TODAY:
        return now.replace(hour=0, minute=0, second=0, microsecond=0), now
    return None


def _time_parts(value: str):
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", value)
    if not match:
        return None
    hour, minute = int(match[1]), int(match[2] or 0)
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _explicit_time_range(text: str, now: datetime):
    normalized = _normalize(text)
    time = r"(\d{1,2}(?::\d{2})?)"
    patterns = [rf"(?<!\d){time}\s*-\s*{time}(?!\d)"]
    from_words = "|".join(sorted((re.escape(v) for v in _RANGE_FROM), key=len, reverse=True))
    to_words = "|".join(sorted((re.escape(v) for v in _RANGE_TO), key=len, reverse=True))
    patterns.append(rf"(?:{from_words})\s+{time}\s+(?:{to_words})\s+{time}")
    patterns.append(rf"(?<!\d){time}\s+(?:{to_words})\s+{time}(?!\d)")
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.I)
        if not match:
            continue
        first, second = _time_parts(match[1]), _time_parts(match[2])
        if not first or not second:
            continue
        after = now.replace(hour=first[0], minute=first[1], second=0, microsecond=0)
        before = now.replace(hour=second[0], minute=second[1], second=0, microsecond=0)
        if after < before:
            return after, before
    return None


def _since_interval(text: str, now: datetime):
    normalized = _normalize(text)
    since_words = "|".join(sorted((re.escape(v) for v in _SINCE), key=len, reverse=True))
    match = re.search(rf"(?:{since_words})\s+(\d{{1,2}}(?::\d{{2}})?)", normalized, flags=re.I)
    if not match:
        return None
    value = _time_parts(match[1])
    if not value:
        return None
    after = now.replace(hour=value[0], minute=value[1], second=0, microsecond=0)
    return (after, now) if after < now else None


def _time_interval(text: str, now: datetime):
    return _relative_interval(text, now) or _explicit_time_range(text, now) or _since_interval(text, now)


def _is_recap_intent(text: str) -> bool:
    return _has_any(text, _RECAP_WORDS)


def _previous_recap_intent(request) -> bool:
    users = _user_texts(request)
    return any(_is_recap_intent(value) for value in users[:-1])


def _call(request, name: str, arguments: dict):
    return response_message(
        {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
        request,
        "",
    )


def _tool_by_name(tools, name):
    return [tool for tool in tools if tool["function"]["name"] == name]


def _feature_toggle_plan(request, text: str, tools, cameras):
    """Build one explicit camera feature toggle without semantic inference."""
    tool = next((t for t in tools if t["function"]["name"] == "set_camera_state"), None)
    if not tool or not _has_any(text, _ACTION_WORDS):
        return None
    tokens = _tokens(text)
    on_hit, off_hit = bool(tokens & _ON_WORDS), bool(tokens & _OFF_WORDS)
    if on_hit == off_hit:
        return None
    feature = None
    for candidate, aliases in _FEATURE_ALIASES.items():
        if tokens & aliases:
            if feature is not None:
                return None
            feature = candidate
    if feature is None:
        return None
    allowed = (
        tool["function"].get("parameters", {}).get("properties", {}).get("feature", {}).get("enum")
    )
    if isinstance(allowed, list) and feature not in allowed:
        return None
    matches = _matched_cameras(text, cameras)
    if len(matches) != 1:
        return None
    chosen = _tool_by_name(tools, "set_camera_state")
    prepared = request.model_copy(update={"tools": chosen})
    direct = _call(
        prepared,
        "set_camera_state",
        {"camera": matches[0], "feature": feature, "value": "ON" if on_hit else "OFF"},
    )
    return chosen, direct, "deterministic_camera_state"


def resolved_facts(request) -> dict:
    """Return multilingual, conservative deterministic facts for prompt compression."""
    text = _latest_user_text(request)
    now = _local_datetime(server_time(request.messages))
    facts = {}
    if now:
        interval = _time_interval(text, now)
        if interval:
            facts["local_time_window"] = {
                "after": interval[0].isoformat(timespec="seconds"),
                "before": interval[1].isoformat(timespec="seconds"),
                "meaning": "unambiguous interval resolved from the latest user request",
            }
    cameras = camera_catalogue(request.messages)
    matches = _matched_cameras(text, cameras)
    if len(matches) == 1:
        facts["camera_id"] = matches[0]
    return facts


def deterministic_plan(request, settings, images):
    """Return an override for high-confidence NVR requests, otherwise None.

    The algorithm is language-neutral. Surface forms are data, and uncertain
    semantics deliberately fall back to the existing Frigate planner/model.
    """
    if images or request.tool_choice == "required" or isinstance(request.tool_choice, dict):
        return None
    tools = selected_tools(request)
    if not tools:
        return None
    names = {tool["function"]["name"] for tool in tools}
    text = _latest_user_text(request)
    normalized = _normalize(text)
    cameras = camera_catalogue(request.messages)
    results = _active_tool_results(request)

    recap = results.get("get_recap")
    if isinstance(recap, dict) and set(results).issubset({"get_recap", "get_profile_status"}):
        if (
            set(recap).issubset({"events", "message"})
            and recap.get("events") == []
            and recap.get("message") in (None, "No activity was found during this time period.")
        ):
            return [], _message(request, text, "empty_recap"), "deterministic_empty_recap"
        if not recap.get("error"):
            return [], None, "deterministic_recap_summary"

    anchor = re.match(r"\[attached_event:([^]\s]+)\]", text, re.I)
    if anchor and "find_similar_objects" in names and _has_any(text, _SIMILAR_WORDS) and not results:
        arguments = {"event_id": anchor[1]}
        now = _local_datetime(server_time(request.messages))
        if now:
            interval = _time_interval(text, now)
            if interval:
                arguments.update(
                    after=interval[0].isoformat(timespec="seconds"),
                    before=interval[1].isoformat(timespec="seconds"),
                )
        matches = _matched_cameras(text, cameras)
        if len(matches) == 1:
            arguments["cameras"] = [matches[0]]
        chosen = _tool_by_name(tools, "find_similar_objects")
        prepared = request.model_copy(update={"tools": chosen})
        return (
            chosen,
            _call(prepared, "find_similar_objects", arguments),
            "deterministic_similarity",
        )

    now = _local_datetime(server_time(request.messages))
    if now and "get_recap" in names and not results:
        interval = _time_interval(text, now)
        recap_intent = _is_recap_intent(text)
        followup = interval is not None and _previous_recap_intent(request)
        if interval and (recap_intent or followup):
            after, before = interval
            if before > now:
                return [], _message(request, text, "future_range"), "deterministic_future_range"
            arguments = {
                "after": after.isoformat(timespec="seconds"),
                "before": before.isoformat(timespec="seconds"),
            }
            matches = _matched_cameras(text, cameras)
            if len(matches) == 1:
                arguments["cameras"] = matches[0]
            chosen = _tool_by_name(tools, "get_recap")
            prepared = request.model_copy(update={"tools": chosen})
            return (
                chosen,
                _call(prepared, "get_recap", arguments),
                "deterministic_recap_interval",
            )

    if "get_live_context" in names and not results and _has_any(text, _LIVE_WORDS):
        if _has_any(text, _PRESENCE_WORDS) or "camera" in normalized or "kamera" in normalized:
            matches = _matched_cameras(text, cameras)
            if len(matches) == 1 or (not matches and len(cameras) == 1):
                camera = matches[0] if matches else next(iter(cameras))
                chosen = _tool_by_name(tools, "get_live_context")
                prepared = request.model_copy(update={"tools": chosen})
                return (
                    chosen,
                    _call(prepared, "get_live_context", {"camera": camera}),
                    "deterministic_live_context",
                )

    toggle = _feature_toggle_plan(request, text, tools, cameras)
    if toggle is not None and not results:
        return toggle

    if "stop_camera_watch" in names and not results:
        if _has_any(text, _STOP_WORDS) and _has_any(text, _WATCH_WORDS):
            chosen = _tool_by_name(tools, "stop_camera_watch")
            prepared = request.model_copy(update={"tools": chosen})
            return (
                chosen,
                _call(prepared, "stop_camera_watch", {}),
                "deterministic_stop_watch",
            )

    if "start_camera_watch" in names and not results and _has_any(text, _WATCH_WORDS):
        chosen = _tool_by_name(tools, "start_camera_watch")
        return chosen, None, "deterministic_watch_tool_selection"

    return None
