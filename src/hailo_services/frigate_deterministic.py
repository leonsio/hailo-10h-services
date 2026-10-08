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
    "de": "was zeige letzten heute gestern passiert kamera ereignisse erkennung überwache schalte wann gesehen",
    "en": "what show last today yesterday happened camera events detection watch turn when seen",
    "fr": "quoi montre dernières aujourd'hui hier passé caméra événements détection surveille quand vu",
    "es": "qué muestra últimas hoy ayer pasó cámara eventos detección vigila cuándo visto",
    "it": "cosa mostra ultime oggi ieri successo telecamera eventi rilevamento avvisa quando visto",
    "nl": "wat toon afgelopen vandaag gisteren gebeurd camera gebeurtenissen detectie bewaak wanneer gezien",
    "pt": "que mostra últimas hoje ontem aconteceu câmera eventos detecção monitora quando visto",
    "ru": "что покажи последние сегодня вчера произошло камера события обнаружение следи когда видел",
}
_LANG_HINTS = {key: set(value.split()) for key, value in _LANG_HINTS.items()}

_RECAP = set(
    "event events activity activities history happened incident incidents recap detection detections alert alerts "
    "ereignis ereignisse aktivitaet aktivitaeten aktivität aktivitäten historie passiert vorfaelle vorfälle erkennung erkennungen alarm alarme "
    "evenement evenements événement événements activite activites activité activités historique passe passé détection détections alerte alertes "
    "evento eventos actividad actividades historial paso pasó ocurrio ocurrió deteccion detección detecciones alerta alertas "
    "eventi attivita attività cronologia successo rilevamento rilevamenti avviso avvisi "
    "gebeurtenis gebeurtenissen activiteit activiteiten geschiedenis gebeurd gebeurde detectie detecties melding meldingen "
    "atividade atividades historico histórico aconteceu deteccao detecção detecções alerta alertas "
    "событие события активность история произошло случилось обнаружение обнаружения тревога тревоги".split()
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
    "last past letzte letzten letzter letztes zuletzt dernier derniere dernière derniers dernieres dernières ultimo ultima "
    "último última ultimos ultimas últimos últimas ultime ultimi scorso scorsa afgelopen laatste laatst "
    "последний последняя последние последних последний раз".split()
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
        for word in "minute minutes min mins minuten minuto minutos minuti minuut minuten minuto minutos минута минуты минут".split()
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
_SINCE = {"since", "seit", "ab", "depuis", "desde", "da", "vanaf", "с"}
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
    "on enable enabled ein an active actif activa activo attiva attivo aan ligado ative ativado включи включено".split()
)
_OFF = set(
    "off disable disabled aus inaktiv desactive désactive desactiva desactivado disattiva uit desligado desative desativado выключи выключено".split()
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
_WHEN = set("when wann quand cuando cuándo quando wanneer когда".split())
_SEEN = set(
    "seen detected spotted gesehen erkannt vue vu détecté detecte detectado vista visto rilevato gezien gedetecteerd "
    "visto detectado увиден замечен обнаружен обнаружена обнаружено".split()
)
_LABEL_ALIASES = {
    "person": set(
        "person people mensch menschen personne personnes persona personas persona persone persoon personen pessoa pessoas человек люди".split()
    ),
    "car": set(
        "car cars auto autos voiture voitures coche coches auto automobili wagen wagens carro carros машина машины автомобиль автомобили".split()
    ),
    "dog": set(
        "dog dogs hund hunde chien chiens perro perros cane cani hond honden cao cão cães собака собаки".split()
    ),
    "cat": set(
        "cat cats katze katzen chat chats gato gatos gatto gatti kat katten кошка кошки кот коты".split()
    ),
    "package": set(
        "package packages paket pakete colis paquete paquetes pacco pacchi pakket pakketten pacote pacotes посылка посылки".split()
    ),
    "bicycle": set(
        "bicycle bicycles bike bikes fahrrad fahrräder velo vélo vélos bicicleta bicicletas bicicletta biciclette fiets fietsen bicicleta bicicletas велосипед велосипеды".split()
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
    """Normalize user text for multilingual lexical matching.

    Args:
        text: User-supplied text.

    Returns:
        str: Case-folded text with normalized whitespace and dash characters.
    """
    return re.sub(r"\s+", " ", text.casefold().replace("–", "-").replace("—", "-")).strip()


def _tokens(text):
    """Tokenize supported scripts without translating the request.

    Args:
        text: User-supplied text.

    Returns:
        set[str]: Normalized lexical tokens found in the text.
    """
    normalized = _normalize(text)
    tokens = set(re.findall(r"[\wÀ-ÖØ-öø-ÿА-Яа-яЁё'’]+", normalized, re.UNICODE))
    split_apostrophes = normalized.replace("'", " ").replace("’", " ")
    tokens.update(re.findall(r"[\wÀ-ÖØ-öø-ÿА-Яа-яЁё]+", split_apostrophes, re.UNICODE))
    return tokens


def _language(request, text):
    """Resolve a response language from an explicit hint or lexical evidence.

    Args:
        request: Incoming chat request which may contain an explicit language.
        text: Latest user text used only when no explicit language is supplied.

    Returns:
        str: Supported two-letter language code, defaulting to English.
    """
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
    """Return a deterministic response in the resolved request language.

    Args:
        request: Incoming chat request.
        text: Latest user text used for language detection.
        key: Message catalogue key.

    Returns:
        str: Localized deterministic response text.
    """
    language = _language(request, text)
    return _MESSAGES[key].get(language, _MESSAGES[key]["en"])


def _local_datetime(value):
    """Parse a supplied Frigate local timestamp without timezone assumptions.

    Args:
        value: Timestamp string from the Frigate prompt or tool result.

    Returns:
        datetime | None: Parsed naive local datetime, or None when unsupported.
    """
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
    """Read the latest user text without copying assistant suggestions.

    Args:
        request: Incoming chat request.

    Returns:
        str: Latest user message text, or an empty string when absent.
    """
    return next(
        (text_content(m).strip() for m in reversed(request.messages) if m.get("role") == "user"), ""
    )


def _results(request):
    """Decode tool results that belong to the latest user turn.

    Args:
        request: Incoming chat request containing assistant calls and tool responses.

    Returns:
        dict: Tool names mapped to decoded JSON results for the active turn.
    """
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
    """Match exact camera IDs or friendly names without fuzzy guessing.

    Args:
        text: Latest user text.
        catalogue: Mapping of camera IDs to friendly names.

    Returns:
        list[str]: Exact camera IDs mentioned in the text.
    """
    return [
        identifier
        for identifier, friendly in catalogue.items()
        if any(
            re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I)
            for value in {identifier, friendly}
        )
    ]


def _relative(text, now):
    """Resolve multilingual elapsed, today, and yesterday expressions.

    Args:
        text: Latest user text.
        now: Authoritative Frigate server-local time.

    Returns:
        tuple[datetime, datetime] | None: Resolved local interval, or None when ambiguous.
    """
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
    """Parse an unambiguous 24-hour clock fragment.

    Args:
        value: Hour or hour-and-minute text.

    Returns:
        tuple[int, int] | None: Hour and minute, or None for an invalid clock value.
    """
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", value)
    if not match:
        return None
    result = int(match[1]), int(match[2] or 0)
    return result if result[0] < 24 and result[1] < 60 else None


def _alternatives(values):
    """Build a longest-first escaped regex alternation for a lexical set.

    Args:
        values: Literal lexical values.

    Returns:
        str: Regex-safe alternation fragment.
    """
    return "|".join(sorted((re.escape(value) for value in values), key=len, reverse=True))


def _explicit_range(text, now):
    """Resolve explicit two-clock ranges without inventing a date.

    Args:
        text: Latest user text.
        now: Authoritative Frigate server-local time.

    Returns:
        tuple[datetime, datetime] | None: Resolved range on today or yesterday.
    """
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
    """Resolve a multilingual start-clock expression ending at supplied now.

    Args:
        text: Latest user text.
        now: Authoritative Frigate server-local time.

    Returns:
        tuple[datetime, datetime] | None: Start-to-now interval, or None when invalid.
    """
    match = re.search(
        rf"(?:{_alternatives(_SINCE)})\s+(?:(?:{_alternatives(_TODAY)})\s+)?(\d{{1,2}}(?::\d{{2}})?)",
        _normalize(text),
        re.I,
    )
    value = _time(match[1]) if match else None
    if not value:
        return None
    after = now.replace(hour=value[0], minute=value[1], second=0, microsecond=0)
    return (after, now) if after < now else None


def _interval(text, now):
    """Resolve the supported deterministic time expressions by specificity.

    Args:
        text: Latest user text.
        now: Authoritative Frigate server-local time.

    Returns:
        tuple[datetime, datetime] | None: Resolved interval, or None when ambiguous.
    """
    return _explicit_range(text, now) or _since(text, now) or _relative(text, now)


def _call(request, name, arguments):
    """Build a schema-validated assistant tool call.

    Args:
        request: Request whose selected tool schema validates the call.
        name: Tool name.
        arguments: Exact deterministic arguments.

    Returns:
        dict: OpenAI-compatible assistant tool-call message.
    """
    return response_message(
        {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}, request, ""
    )


def _tools(tools, name):
    """Select declarations for one exact tool name.

    Args:
        tools: Available OpenAI function tool declarations.
        name: Function name to retain.

    Returns:
        list[dict]: Matching tool declarations.
    """
    return [tool for tool in tools if tool["function"]["name"] == name]


def _fixed_tool(tools, name, fixed):
    """Constrain known tool properties to deterministic enum values.

    Args:
        tools: Available OpenAI function tool declarations.
        name: Function name to retain.
        fixed: Property values already resolved without inference.

    Returns:
        list[dict]: Deep-copied declaration with resolved properties constrained.
    """
    chosen = copy.deepcopy(_tools(tools, name))
    if chosen:
        properties = chosen[0]["function"].get("parameters", {}).get("properties", {})
        for key, value in fixed.items():
            if isinstance(properties.get(key), dict):
                properties[key]["enum"] = [value]
    return chosen


def _toggle(request, text, tools, catalogue):
    """Resolve an exact camera feature on/off action across supported languages.

    Args:
        request: Incoming chat request.
        text: Latest user text.
        tools: Available tool declarations.
        catalogue: Mapping of camera IDs to friendly names.

    Returns:
        tuple | None: Deterministic tool selection/call/reason, or None when ambiguous.
    """
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


def _last_sighting(text, tools, catalogue):
    """Resolve a class-based latest-sighting query without translating free text.

    The routine intentionally handles only canonical object classes. Named people,
    appearance descriptions, and compound conditions remain model/legacy-planner work.

    Args:
        text: Latest user text.
        tools: Available tool declarations.
        catalogue: Mapping of camera IDs to friendly names.

    Returns:
        tuple[list[dict], dict] | None: Selected tool and exact arguments, or None.
    """
    tokens = _tokens(text)
    if not tokens & _WHEN or not tokens & _LAST or not tokens & _SEEN:
        return None
    labels = [canonical for canonical, aliases in _LABEL_ALIASES.items() if tokens & aliases]
    if len(labels) != 1:
        return None
    selected = _tools(tools, "search_objects")
    if not selected:
        return None
    properties = selected[0]["function"].get("parameters", {}).get("properties", {})
    if "label" not in properties or "limit" not in properties or "camera" not in properties:
        return None
    matches = _cameras(text, catalogue)
    if len(matches) != 1:
        return None
    arguments = {"label": labels[0], "camera": matches[0], "limit": 1}
    return selected, arguments


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
    text = _latest_text(request)
    tokens = _tokens(text)
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
    if not results and "search_objects" in names:
        sighting = _last_sighting(text, tools, catalogue)
        if sighting:
            chosen, arguments = sighting
            prepared = request.model_copy(update={"tools": chosen})
            return (
                chosen,
                _call(prepared, "search_objects", arguments),
                "deterministic_class_last_sighting",
            )
    now = _local_datetime(server_time(request.messages))
    interval = _interval(text, now) if now else None
    user_texts = [
        text_content(message)
        for message in request.messages
        if message.get("role") == "user"
    ]
    previous_recap = len(user_texts) >= 2 and bool(_tokens(user_texts[-2]) & _RECAP)
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
        matches = _cameras(text, catalogue)
        chosen = (
            _fixed_tool(tools, "start_camera_watch", {"camera": matches[0]})
            if len(matches) == 1
            else _tools(tools, "start_camera_watch")
        )
        return chosen, None, "deterministic_watch_tool_selection"
    return None
