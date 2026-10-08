"""Multilingual deterministic shortcuts for common Frigate/NVR requests."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timedelta

from .frigate_prompt import camera_catalogue, server_time, text_content
from .frigate_routing import recap_interval
from .tool_calling import response_message, selected_tools

_LANG_HINTS = {
    "de": "was zeige letzten heute gestern passiert kamera ereignisse",
    "en": "what show last today yesterday happened camera events",
    "fr": "quoi montre dernières aujourd'hui hier passé caméra événements",
    "es": "qué muestra últimas hoy ayer pasó cámara eventos",
    "it": "cosa mostra ultime oggi ieri successo telecamera eventi",
    "nl": "wat toon afgelopen vandaag gisteren gebeurd gebeurde camera gebeurtenissen",
    "pt": "que mostra últimas hoje ontem aconteceu câmera eventos",
    "ru": "что покажи последние сегодня вчера произошло камера события",
}
_LANG_HINTS = {key: set(value.split()) for key, value in _LANG_HINTS.items()}

_RECAP = set(
    "event events activity activities history happened incident incidents recap "
    "ereignis ereignisse aktivitaet aktivitaeten aktivität aktivitäten historie passiert vorfaelle vorfälle "
    "evenement evenements événement événements activite activites activité activités historique passe passé "
    "evento eventos actividad actividades historial paso pasó ocurrio ocurrió "
    "eventi attivita attività cronologia successo "
    "gebeurtenis gebeurtenissen activiteit activiteiten geschiedenis gebeurd gebeurde "
    "atividade atividades historico histórico aconteceu "
    "событие события активность история произошло случилось".split()
)
_LIVE = set(
    "now live current currently visible jetzt gerade aktuell livebild sichtbar maintenant actuel actuelle direct "
    "ahora actual directo adesso ora attuale diretta visibile nu actueel zichtbaar agora atual direto visivel visível "
    "сейчас прямо текущий видно".split()
)
_PRESENCE = set(
    "there see show visible present happening da sehen zeige sichtbar passiert voir montre présent passe passé "
    "hay ver muestra pasa pasando c'e c’è vedi mostra visibile succede er zie toon zichtbaar gebeurt "
    "ha há mostra visivel visível acontecendo есть видно покажи вижу происходит".split()
)
_SIMILAR = set(
    "similar same like aehnlich ähnlich gleich derselbe dieselbe similaire semblable meme même parecido mismo "
    "simile stesso vergelijkbaar zelfde semelhante mesmo похож похожие такой же".split()
)
_SEARCH = set(
    "show find search list zeige finde suche liste montre cherche muestra busca lista mostra cerca toon zoek lijst "
    "procura покажи найди список".split()
)
_AWAY = set(
    "away absence absent abwesenheit weg ausencia ausente assenza afwezigheid afwezig ausência "
    "отсутствие отсутствовал отсутствовала".split()
)
_LAST = set(
    "last past letzte letzten letzter letztes dernier derniere dernière derniers dernieres dernières ultimo ultima "
    "último última ultimos ultimas últimos últimas ultime ultimi afgelopen laatste последний последняя последние последних".split()
)
_ONE = set("one a an ein eine einer einen einem un une uno una een um uma один одна одну".split())
_TODAY = {
    "today",
    "heute",
    "aujourd'hui",
    "aujourdhui",
    "hoy",
    "oggi",
    "vandaag",
    "hoje",
    "сегодня",
}
_YESTERDAY = {"yesterday", "gestern", "hier", "ayer", "ieri", "gisteren", "ontem", "вчера"}
_UNITS = {
    **{
        word: 1
        for word in "minute minutes min mins minuten minuto minutos minuti minuut минута минуты минут".split()
    },
    **{
        word: 60
        for word in "hour hours hr hrs stunde stunden heure heures hora horas ora ore uur uren час часа часов".split()
    },
    **{
        word: 1440
        for word in "day days tag tage tagen jour jours dia dias giorno giorni dag dagen день дня дней".split()
    },
    **{
        word: 10080
        for word in "week weeks woche wochen semaine semaines semana semanas settimana settimane неделя недели недель".split()
    },
}
_FROM = {"from", "von", "de", "desde", "da", "dal", "dalle", "van", "das", "с"}
_TO = {"to", "until", "bis", "a", "à", "hasta", "alle", "tot", "as", "às", "ate", "até", "до"}
_BETWEEN = {"between", "zwischen", "entre", "tra", "tussen", "между"}
_AND = {"and", "und", "et", "y", "e", "en", "и"}
_SINCE = {"since", "seit", "depuis", "desde", "da", "vanaf", "с"}
_WATCH = set(
    "watch monitor notify notification überwachung ueberwachung überwache ueberwache benachrichtige surveille avertis "
    "notifie vigila notifica avvisa monitora bewaak meld monitoriza avise notifique наблюдение следи уведомляй".split()
)
_STOP = set(
    "stop cancel stoppe beende annule arrete arrête deten detén ferma pare останови отмени".split()
)
_ACTION = set(
    "turn set enable disable schalte aktiviere deaktiviere active désactive activa desactiva attiva disattiva zet "
    "ative desative включи выключи".split()
)
_ON = set(
    "on enable enabled ein an active actif activo attivo aan ligado ativado включи включено".split()
)
_OFF = set(
    "off disable disabled aus inaktiv desactive désactive desactivado disattiva uit desligado desativado выключи выключено".split()
)
_FEATURES = {
    "detect": set(
        "detect detection erkennung détection deteccion detección rilevamento detectie deteccao detecção обнаружение".split()
    ),
    "record": set(
        "record recording aufnahme enregistrement grabacion grabación registrazione opname gravacao gravação запись".split()
    ),
    "snapshots": set(
        "snapshot snapshots schnappschuss schnappschuesse schnappschüsse instantane instantané captura istantanea momentopname снимок снимки".split()
    ),
    "audio": set("audio ton son sonido suono geluid som звук".split()),
    "motion": set("motion bewegung mouvement movimiento movimento beweging движение".split()),
    "notifications": set(
        "notification notifications benachrichtigung benachrichtigungen notificacion notificación notifiche melding meldingen notificacao notificação уведомление уведомления".split()
    ),
}
_MESSAGES = {
    "empty": {
        "de": "Für den abgefragten Zeitraum wurden keine Aktivitäten zurückgegeben.",
        "en": "No activity was returned for the requested time period.",
        "fr": "Aucune activité n’a été renvoyée pour la période demandée.",
        "es": "No se devolvió ninguna actividad para el período solicitado.",
        "it": "Non è stata restituita alcuna attività per il periodo richiesto.",
        "nl": "Er is geen activiteit teruggegeven voor de gevraagde periode.",
        "pt": "Nenhuma atividade foi retornada para o período solicitado.",
        "ru": "За запрошенный период активность не найдена.",
    },
    "future": {
        "de": "Der angegebene historische Zeitraum liegt teilweise in der Zukunft.",
        "en": "The requested historical interval is partly in the future.",
        "fr": "La période historique demandée se situe en partie dans le futur.",
        "es": "El intervalo histórico solicitado está parcialmente en el futuro.",
        "it": "L’intervallo storico richiesto è in parte nel futuro.",
        "nl": "Het gevraagde historische tijdvak ligt gedeeltelijk in de toekomst.",
        "pt": "O intervalo histórico solicitado está parcialmente no futuro.",
        "ru": "Запрошенный исторический интервал частично находится в будущем.",
    },
    "absence": {
        "de": "Für die Abwesenheit lässt sich kein eindeutiger Zeitraum aus den Profilen bestimmen. Von wann bis wann soll ich nachsehen?",
        "en": "The profiles do not establish an unambiguous absence interval. What start and end time should I use?",
        "fr": "Les profils ne permettent pas de déterminer une période d’absence sans ambiguïté. Quelles heures de début et de fin dois-je utiliser ?",
        "es": "Los perfiles no permiten determinar un intervalo de ausencia inequívoco. ¿Qué horas de inicio y fin debo usar?",
        "it": "I profili non definiscono un intervallo di assenza univoco. Quali orari di inizio e fine devo usare?",
        "nl": "De profielen bepalen geen eenduidige afwezigheidsperiode. Welke begin- en eindtijd moet ik gebruiken?",
        "pt": "Os perfis não definem um intervalo de ausência inequívoco. Quais horários de início e fim devo usar?",
        "ru": "Профили не задают однозначный интервал отсутствия. Какое время начала и окончания использовать?",
    },
}


def _normalize(text):
    return re.sub(r"\s+", " ", text.casefold().replace("–", "-").replace("—", "-")).strip()


def _tokens(text):
    return set(re.findall(r"[\wÀ-ÖØ-öø-ÿА-Яа-яЁё'’]+", _normalize(text), re.UNICODE))


def _language(request, text):
    explicit = getattr(request, "language", None)
    if isinstance(explicit, str) and explicit:
        code = explicit.casefold().split("-")[0].split("_")[0]
        if code in _LANG_HINTS:
            return code
    tokens = _tokens(text)
    scores = {code: len(tokens & words) for code, words in _LANG_HINTS.items()}
    if re.search(r"[А-Яа-яЁё]", text):
        scores["ru"] += 2
    code = max(scores, key=scores.get)
    return code if scores[code] else "en"


def _message(request, text, key):
    language = _language(request, text)
    return _MESSAGES[key].get(language, _MESSAGES[key]["en"])


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
            pass
    return None


def _latest_text(request):
    return next(
        (text_content(m).strip() for m in reversed(request.messages) if m.get("role") == "user"), ""
    )


def _results(request):
    users = [i for i, m in enumerate(request.messages) if m.get("role") == "user"]
    if not users:
        return {}
    calls, results = {}, {}
    for message in request.messages[users[-1] :]:
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            if isinstance(call.get("id"), str) and isinstance(function.get("name"), str):
                calls[call["id"]] = function["name"]
        if message.get("role") == "tool" and message.get("tool_call_id") in calls:
            try:
                results[calls[message["tool_call_id"]]] = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                results[calls[message["tool_call_id"]]] = {"error": "Tool returned non-JSON data"}
    return results


def _cameras(text, catalogue):
    return [
        identifier
        for identifier, friendly in catalogue.items()
        if any(
            re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I)
            for value in {identifier, friendly}
        )
    ]


def _relative(text, now):
    words = re.findall(r"[\wÀ-ÖØ-öø-ÿА-Яа-яЁё]+", _normalize(text), re.UNICODE)
    for i, word in enumerate(words):
        if word not in _LAST:
            continue
        pairs = [
            (words[j], words[j + 1] if j + 1 < len(words) else None)
            for j in range(i + 1, min(len(words), i + 4))
        ]
        if i > 0 and i + 1 < len(words):
            pairs.append((words[i - 1], words[i + 1]))
        for raw, unit in pairs:
            amount = int(raw) if raw.isdigit() else 1 if raw in _ONE else None
            if amount and unit in _UNITS:
                try:
                    return now - timedelta(minutes=amount * _UNITS[unit]), now
                except OverflowError:
                    return None
    tokens = set(words)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if tokens & _YESTERDAY:
        return midnight - timedelta(days=1), midnight
    if tokens & _TODAY:
        return midnight, now
    return None


def _time(value):
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", value)
    if not match:
        return None
    result = int(match[1]), int(match[2] or 0)
    return result if result[0] < 24 and result[1] < 60 else None


def _alternatives(values):
    return "|".join(sorted((re.escape(value) for value in values), key=len, reverse=True))


def _explicit_range(text, now):
    normalized = _normalize(text)
    stamp = r"(\d{1,2}(?::\d{2})?)"
    patterns = [
        rf"(?<!\d){stamp}\s*-\s*{stamp}(?!\d)",
        rf"(?:{_alternatives(_FROM)})\s+{stamp}\s+(?:{_alternatives(_TO)})\s+{stamp}",
        rf"(?<!\d){stamp}\s+(?:{_alternatives(_TO)})\s+{stamp}(?!\d)",
        rf"(?:{_alternatives(_BETWEEN)})\s+{stamp}\s+(?:{_alternatives(_AND)})\s+{stamp}",
    ]
    base = now - timedelta(days=1) if _tokens(text) & _YESTERDAY else now
    for pattern in patterns:
        match = re.search(pattern, normalized, re.I)
        if not match:
            continue
        first, second = _time(match[1]), _time(match[2])
        if first and second:
            after = base.replace(hour=first[0], minute=first[1], second=0, microsecond=0)
            before = base.replace(hour=second[0], minute=second[1], second=0, microsecond=0)
            if after < before:
                return after, before
    return None


def _since(text, now):
    match = re.search(
        rf"(?:{_alternatives(_SINCE)})\s+(\d{{1,2}}(?::\d{{2}})?)", _normalize(text), re.I
    )
    value = _time(match[1]) if match else None
    if not value:
        return None
    after = now.replace(hour=value[0], minute=value[1], second=0, microsecond=0)
    return (after, now) if after < now else None


def _interval(text, now):
    return _explicit_range(text, now) or _since(text, now) or _relative(text, now)


def _call(request, name, arguments):
    return response_message(
        {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}, request, ""
    )


def _tools(tools, name):
    return [tool for tool in tools if tool["function"]["name"] == name]


def _fixed_tool(tools, name, fixed):
    chosen = copy.deepcopy(_tools(tools, name))
    if chosen:
        properties = chosen[0]["function"].get("parameters", {}).get("properties", {})
        for key, value in fixed.items():
            if isinstance(properties.get(key), dict):
                properties[key]["enum"] = [value]
    return chosen


def _toggle(request, text, tools, catalogue):
    tool = next((tool for tool in tools if tool["function"]["name"] == "set_camera_state"), None)
    tokens = _tokens(text)
    if not tool or not tokens & _ACTION:
        return None
    on, off = bool(tokens & _ON), bool(tokens & _OFF)
    if on == off:
        return None
    feature = None
    for candidate, aliases in _FEATURES.items():
        if tokens & aliases:
            if feature:
                return None
            feature = candidate
    allowed = (
        tool["function"].get("parameters", {}).get("properties", {}).get("feature", {}).get("enum")
    )
    cameras = _cameras(text, catalogue)
    if not feature or len(cameras) != 1 or isinstance(allowed, list) and feature not in allowed:
        return None
    chosen = _tools(tools, "set_camera_state")
    prepared = request.model_copy(update={"tools": chosen})
    return (
        chosen,
        _call(
            prepared,
            "set_camera_state",
            {"camera": cameras[0], "feature": feature, "value": "ON" if on else "OFF"},
        ),
        "deterministic_camera_state",
    )


def resolved_facts(request):
    """Resolve multilingual time and camera facts without model inference.

    Args:
        request: Incoming Frigate request.

    Returns:
        dict: Unambiguous facts safe to inject into the compact prompt.
    """
    text = _latest_text(request)
    now = _local_datetime(server_time(request.messages))
    facts = {}
    interval = _interval(text, now) if now else None
    if interval:
        facts["local_time_window"] = {
            "after": interval[0].isoformat(timespec="seconds"),
            "before": interval[1].isoformat(timespec="seconds"),
            "meaning": "unambiguous interval resolved from the latest user request",
        }
    matches = _cameras(text, camera_catalogue(request.messages))
    if len(matches) == 1:
        facts["camera_id"] = matches[0]
    return facts


def deterministic_plan(request, settings, images):
    """Plan high-confidence NVR requests without unnecessary model inference.

    Args:
        request: Incoming Frigate request.
        settings: Service settings used for absence-profile resolution.
        images: Whether the current turn requires image observation.

    Returns:
        tuple | None: Selected tools, optional direct response and route reason, or None.
    """
    if images or request.tool_choice == "required" or isinstance(request.tool_choice, dict):
        return None
    tools = selected_tools(request)
    if not tools:
        return None
    names = {tool["function"]["name"] for tool in tools}
    text, tokens = _latest_text(request), _tokens(_latest_text(request))
    catalogue, results = camera_catalogue(request.messages), _results(request)
    recap = results.get("get_recap")
    if isinstance(recap, dict) and set(results).issubset({"get_recap", "get_profile_status"}):
        if (
            set(recap).issubset({"events", "message"})
            and recap.get("events") == []
            and recap.get("message") in (None, "No activity was found during this time period.")
        ):
            return [], _message(request, text, "empty"), "deterministic_empty_recap"
        if not recap.get("error"):
            return [], None, "deterministic_recap_summary"
    if any(
        name in results for name in {"set_camera_state", "start_camera_watch", "stop_camera_watch"}
    ):
        return [], None, "deterministic_action_summary"
    similar = bool(tokens & _SIMILAR)
    if "search_objects" in results and not similar:
        return [], None, "deterministic_search_summary"
    absence = bool(tokens & _RECAP) and bool(tokens & _AWAY)
    if absence:
        if "get_profile_status" in results and "get_recap" not in results:
            interval = recap_interval(results["get_profile_status"], request, settings)
            if interval and "get_recap" in names:
                chosen = _tools(tools, "get_recap")
                prepared = request.model_copy(update={"tools": chosen})
                return (
                    chosen,
                    _call(prepared, "get_recap", interval),
                    "deterministic_absence_interval",
                )
            return [], _message(request, text, "absence"), "deterministic_absence_ambiguous"
        if not results and "get_profile_status" in names:
            chosen = _tools(tools, "get_profile_status")
            prepared = request.model_copy(update={"tools": chosen})
            return (
                chosen,
                _call(prepared, "get_profile_status", {}),
                "deterministic_absence_profile",
            )
    anchor = re.match(r"\[attached_event:([^]\s]+)\]", text, re.I)
    if anchor and similar and "find_similar_objects" in names and not results:
        arguments = {"event_id": anchor[1]}
        now = _local_datetime(server_time(request.messages))
        interval = _interval(text, now) if now else None
        if interval:
            arguments.update(
                after=interval[0].isoformat(timespec="seconds"),
                before=interval[1].isoformat(timespec="seconds"),
            )
        matches = _cameras(text, catalogue)
        if len(matches) == 1:
            arguments["cameras"] = [matches[0]]
        chosen = _tools(tools, "find_similar_objects")
        prepared = request.model_copy(update={"tools": chosen})
        return (
            chosen,
            _call(prepared, "find_similar_objects", arguments),
            "deterministic_similarity",
        )
    now = _local_datetime(server_time(request.messages))
    interval = _interval(text, now) if now else None
    previous_recap = any(
        _tokens(text_content(message)) & _RECAP
        for message in request.messages[:-1]
        if message.get("role") == "user"
    )
    if interval and "get_recap" in names and not results and (tokens & _RECAP or previous_recap):
        after, before = interval
        if before > now:
            return [], _message(request, text, "future"), "deterministic_future_range"
        arguments = {
            "after": after.isoformat(timespec="seconds"),
            "before": before.isoformat(timespec="seconds"),
        }
        matches = _cameras(text, catalogue)
        if len(matches) == 1:
            arguments["cameras"] = matches[0]
        chosen = _tools(tools, "get_recap")
        prepared = request.model_copy(update={"tools": chosen})
        return chosen, _call(prepared, "get_recap", arguments), "deterministic_recap_interval"
    if (
        interval
        and "search_objects" in names
        and not results
        and tokens & _SEARCH
        and not tokens & _RECAP
    ):
        after, before = interval
        fixed = {
            "after": after.isoformat(timespec="seconds"),
            "before": before.isoformat(timespec="seconds"),
        }
        matches = _cameras(text, catalogue)
        if len(matches) == 1:
            fixed["camera"] = matches[0]
        return (
            _fixed_tool(tools, "search_objects", fixed),
            None,
            "deterministic_historical_search_tool_selection",
        )
    if "get_live_context" in names and not results and tokens & _LIVE:
        matches = _cameras(text, catalogue)
        if tokens & _PRESENCE or matches:
            if len(matches) == 1 or not matches and len(catalogue) == 1:
                camera = matches[0] if matches else next(iter(catalogue))
                chosen = _tools(tools, "get_live_context")
                prepared = request.model_copy(update={"tools": chosen})
                return (
                    chosen,
                    _call(prepared, "get_live_context", {"camera": camera}),
                    "deterministic_live_context",
                )
    toggle = _toggle(request, text, tools, catalogue)
    if toggle and not results:
        return toggle
    if "stop_camera_watch" in names and not results and tokens & _STOP and tokens & _WATCH:
        chosen = _tools(tools, "stop_camera_watch")
        prepared = request.model_copy(update={"tools": chosen})
        return chosen, _call(prepared, "stop_camera_watch", {}), "deterministic_stop_watch"
    if "start_camera_watch" in names and not results and tokens & _WATCH:
        return _tools(tools, "start_camera_watch"), None, "deterministic_watch_tool_selection"
    return None
