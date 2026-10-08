"""HassIL recognition for high-confidence Frigate intents.

The grammar is intentionally narrower than Frigate's complete chat surface. A miss is
not an error: the established deterministic planner and then the configured LLM/VLM
remain available. This layer only bypasses inference when both intent and arguments
are explicit and schema-validatable.
"""

from __future__ import annotations

import copy
import time
from functools import lru_cache

from .frigate_deterministic import (
    _FEATURES,
    _LABEL_ALIASES,
    _OFF,
    _ON,
    _TODAY,
    _YESTERDAY,
    _interval,
    _language,
    _local_datetime,
)
from .frigate_prompt import camera_catalogue, server_time, text_content
from .intent_engine import mapped_text_slot, restrict_grammar, result_slots, unique_recognition
from .tool_calling import response_message, selected_tools

# These sentence templates cover only requests whose full meaning can be preserved
# deterministically. Free-form descriptions, compound requests and semantic searches
# intentionally fall through to the existing planners/models.
_SENTENCES = {
    "de": {
        "FrigateLiveContext": [
            "(zeige|zeig|beschreibe) [mir] [das] (livebild|kamerabild|bild) (von|an) [kamera] {camera}",
            "was ist (jetzt|gerade|aktuell) (an|auf) [kamera] {camera} [zu sehen]",
            "was siehst du (jetzt|gerade|aktuell) (an|auf) [kamera] {camera}",
        ],
        "FrigateAbsenceRecap": [
            "was ist passiert (während|waehrend|als) ich weg war",
            "was ist passiert (während|waehrend) meiner abwesenheit",
        ],
        "FrigateRecap": [
            "(zeige|zeig) [mir] [die|alle] (ereignisse|aktivitäten|aktivitaeten|erkennungen|historie) [von] {period}",
            "was ist {period} passiert",
        ],
        "FrigateSearchObjects": [
            "(zeige|zeig|finde|suche) [mir] [alle|die] {label} {period}",
        ],
        "FrigateSetCameraState": [
            "(schalte|setze) [die] {feature} (für|fuer) [kamera] {camera} {state}",
        ],
        "FrigateStopWatch": [
            "(stoppe|beende) [die] (überwachung|ueberwachung)",
            "(stoppe|beende) [die] kamera (überwachung|ueberwachung)",
        ],
        "FrigateLastSighting": [
            "wann wurde [zuletzt] (ein|eine|die|der|das) {label} (an|bei|auf|vor) [kamera] {camera} zuletzt (gesehen|erkannt)",
            "wann wurde zuletzt (ein|eine|die|der|das) {label} (an|bei|auf|vor) [kamera] {camera} (gesehen|erkannt)",
        ],
    },
    "en": {
        "FrigateLiveContext": [
            "(show|describe) [me] [the] (live image|camera image|image|picture) (from|of|at) [camera] {camera}",
            "what (is|do you see) (now|currently) (at|on) [camera] {camera} [visible]",
            "what is visible (now|currently) (at|on) [camera] {camera}",
        ],
        "FrigateAbsenceRecap": [
            "what happened while i was away",
            "what happened during my absence",
        ],
        "FrigateRecap": [
            "show [me] [all|the] (events|activity|detections|history) [from] {period}",
            "what happened {period}",
        ],
        "FrigateSearchObjects": [
            "(show|find|search for) [me] [all|the] {label} {period}",
        ],
        "FrigateSetCameraState": [
            "(turn|set) [the] {feature} (for|on) [camera] {camera} {state}",
        ],
        "FrigateStopWatch": [
            "(stop|cancel) [the] [camera] (watch|monitoring)",
            "stop watching",
        ],
        "FrigateLastSighting": [
            "when was (a|the) {label} last (seen|detected) (at|on|in front of) [camera] {camera}",
        ],
    },
    "fr": {
        "FrigateLiveContext": [
            "(montre|décris|decris) [moi] [l'] (image en direct|image de caméra|image) (de|sur) [la caméra] {camera}",
            "qu'est ce qui est visible maintenant (sur|à) [la caméra] {camera}",
        ],
        "FrigateAbsenceRecap": ["qu'est ce qui s'est passé pendant mon absence"],
        "FrigateRecap": [
            "montre [moi] [les] (événements|evenements|activités|activites|détections|detections) {period}",
        ],
        "FrigateSearchObjects": ["(montre|cherche) [moi] [les] {label} {period}"],
        "FrigateSetCameraState": [
            "(active|désactive|desactive) [la] {feature} (sur|pour) [la caméra] {camera} {state}"
        ],
        "FrigateStopWatch": ["(arrête|arrete|annule) [la] surveillance"],
        "FrigateLastSighting": [
            "quand (un|une|le|la) {label} a été vu pour la dernière fois (sur|à) [la caméra] {camera}"
        ],
    },
    "es": {
        "FrigateLiveContext": [
            "(muestra|describe) [me] [la] (imagen en directo|imagen de cámara|imagen) (de|en) [la cámara] {camera}",
            "qué se ve ahora (en|por) [la cámara] {camera}",
        ],
        "FrigateAbsenceRecap": ["qué pasó durante mi ausencia", "qué pasó mientras estaba fuera"],
        "FrigateRecap": ["muestra [me] [los] (eventos|actividad|detecciones|historial) {period}"],
        "FrigateSearchObjects": ["(muestra|busca) [me] [los] {label} {period}"],
        "FrigateSetCameraState": ["(activa|desactiva) [la] {feature} (en|para) [la cámara] {camera} {state}"],
        "FrigateStopWatch": ["(detén|deten|cancela) [la] vigilancia"],
        "FrigateLastSighting": [
            "cuándo fue visto por última vez (un|una|el|la) {label} (en|por) [la cámara] {camera}"
        ],
    },
    "it": {
        "FrigateLiveContext": [
            "(mostra|descrivi) [mi] [la] (immagine live|immagine della telecamera|immagine) (di|dalla) [telecamera] {camera}",
            "cosa si vede adesso (su|alla) [telecamera] {camera}",
        ],
        "FrigateAbsenceRecap": ["cosa è successo durante la mia assenza"],
        "FrigateRecap": ["mostra [mi] [gli] (eventi|attività|attivita|rilevamenti) {period}"],
        "FrigateSearchObjects": ["(mostra|cerca) [mi] [tutti] {label} {period}"],
        "FrigateSetCameraState": ["(attiva|disattiva) [il] {feature} (su|per) [telecamera] {camera} {state}"],
        "FrigateStopWatch": ["(ferma|annulla) [il] monitoraggio"],
        "FrigateLastSighting": [
            "quando è stato visto l'ultima volta (un|una|il|la) {label} (su|alla) [telecamera] {camera}"
        ],
    },
    "nl": {
        "FrigateLiveContext": [
            "(toon|beschrijf) [mij] [het] (livebeeld|camerabeeld|beeld) (van|op) [camera] {camera}",
            "wat is er nu zichtbaar (op|bij) [camera] {camera}",
        ],
        "FrigateAbsenceRecap": ["wat is er gebeurd tijdens mijn afwezigheid"],
        "FrigateRecap": ["toon [mij] [de] (gebeurtenissen|activiteit|detecties|geschiedenis) {period}"],
        "FrigateSearchObjects": ["(toon|zoek) [mij] [alle] {label} {period}"],
        "FrigateSetCameraState": ["zet [de] {feature} (op|voor) [camera] {camera} {state}"],
        "FrigateStopWatch": ["(stop|annuleer) [de] bewaking"],
        "FrigateLastSighting": [
            "wanneer is (een|de) {label} voor het laatst gezien (op|bij) [camera] {camera}"
        ],
    },
    "pt": {
        "FrigateLiveContext": [
            "(mostra|descreve) [me] [a] (imagem ao vivo|imagem da câmera|imagem) (da|na) [câmera] {camera}",
            "o que está visível agora (na|pela) [câmera] {camera}",
        ],
        "FrigateAbsenceRecap": ["o que aconteceu durante minha ausência"],
        "FrigateRecap": ["mostra [me] [os] (eventos|atividade|detecções|historico|histórico) {period}"],
        "FrigateSearchObjects": ["(mostra|procura) [me] [todos] {label} {period}"],
        "FrigateSetCameraState": ["(ative|desative) [a] {feature} (na|para) [câmera] {camera} {state}"],
        "FrigateStopWatch": ["(pare|cancele) [o] monitoramento"],
        "FrigateLastSighting": [
            "quando (um|uma|o|a) {label} foi visto pela última vez (na|pela) [câmera] {camera}"
        ],
    },
    "ru": {
        "FrigateLiveContext": [
            "(покажи|опиши) [мне] (текущее изображение|живое изображение|изображение) [с] [камеры] {camera}",
            "что сейчас видно [на] [камере] {camera}",
        ],
        "FrigateAbsenceRecap": ["что произошло во время моего отсутствия"],
        "FrigateRecap": ["покажи [мне] (события|активность|обнаружения|историю) {period}"],
        "FrigateSearchObjects": ["(покажи|найди) [мне] [все] {label} {period}"],
        "FrigateSetCameraState": ["(включи|выключи) {feature} [на] [камере] {camera} {state}"],
        "FrigateStopWatch": ["(останови|отмени) наблюдение"],
        "FrigateLastSighting": [
            "когда {label} последний раз был замечен [на] [камере] {camera}"
        ],
    },
}


@lru_cache(maxsize=8)
def _grammar(language):
    """Compile the Frigate-specific HassIL grammar for one supported language.

    Args:
        language: Supported two-letter language code.

    Returns:
        Intents | None: Compiled grammar, or None for an unsupported language.
    """
    sentences = _SENTENCES.get(language)
    if not sentences:
        return None
    return restrict_grammar(
        {
            "language": language,
            "intents": {
                intent: {"data": [{"sentences": forms}]} for intent, forms in sentences.items()
            },
        }
    )


def _slot_lists(language, cameras):
    """Build request-scoped camera and canonical Frigate vocabulary slots.

    Args:
        language: HassIL grammar language.
        cameras: Camera IDs mapped to Frigate friendly names.

    Returns:
        dict: Slot names mapped to request-scoped HassIL text lists.
    """
    camera_values = []
    for identifier, friendly in cameras.items():
        camera_values.extend([(identifier, identifier), (friendly, identifier)])
    label_values = [
        (alias, canonical) for canonical, aliases in _LABEL_ALIASES.items() for alias in aliases
    ]
    feature_values = [
        (alias, canonical) for canonical, aliases in _FEATURES.items() for alias in aliases
    ]
    state_values = [(alias, "ON") for alias in _ON] + [(alias, "OFF") for alias in _OFF]
    period_values = [(value, "today") for value in _TODAY] + [
        (value, "yesterday") for value in _YESTERDAY
    ]
    return {
        "camera": mapped_text_slot(language, "camera", camera_values),
        "label": mapped_text_slot(language, "label", label_values),
        "feature": mapped_text_slot(language, "feature", feature_values),
        "state": mapped_text_slot(language, "state", state_values),
        "period": mapped_text_slot(language, "period", period_values),
    }


def recognize_request(request):
    """Recognize one exact Frigate intent without interpreting unmatched free text.

    Args:
        request: Incoming Frigate OpenAI-compatible request.

    Returns:
        tuple: ``(match, trace)`` where match is a plain intent/slot dictionary or None.
    """
    started = time.perf_counter()
    text = next(
        (text_content(message).strip() for message in reversed(request.messages) if message.get("role") == "user"),
        "",
    )
    trace = {"stage": "hassil", "matched": False, "reason": "no_match"}
    if not text or len(text) > 512 or request.tool_choice == "none":
        trace["reason"] = "ineligible"
        return None, trace
    language = _language(request, text)
    grammar = _grammar(language)
    cameras = camera_catalogue(request.messages)
    if grammar is None:
        trace["reason"] = "unsupported_language"
        return None, trace
    result, reason = unique_recognition(
        text.rstrip("?.!"),
        grammar,
        slot_lists=_slot_lists(language, cameras),
        language=language,
    )
    trace.update(reason=reason, language=language, duration_ms=(time.perf_counter() - started) * 1000)
    if result is None:
        return None, trace
    slots = result_slots(result)
    match = {"intent": result.intent.name, "slots": slots, "language": language}
    trace.update(matched=True, intent=result.intent.name, slots=slots)
    return match, trace


def _tool(tools, name):
    """Select declarations for one exact Frigate tool name.

    Args:
        tools: Available OpenAI function declarations.
        name: Function name to retain.

    Returns:
        list[dict]: Matching tool declarations.
    """
    return [item for item in tools if item["function"]["name"] == name]


def _supports(tool, arguments):
    """Check deterministic arguments against the exposed schema shape before calling.

    Args:
        tool: Selected OpenAI tool declaration.
        arguments: Deterministically resolved arguments.

    Returns:
        bool: Whether the exposed schema can safely accept all arguments.
    """
    if not tool:
        return False
    schema = tool[0]["function"].get("parameters", {})
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not set(arguments).issubset(properties) or not required.issubset(arguments):
        return False
    for key, value in arguments.items():
        rule = properties.get(key, {})
        allowed = rule.get("enum")
        if isinstance(allowed, list) and value not in allowed:
            return False
    return True


def _call(request, tools, name, arguments):
    """Build one schema-checked deterministic Frigate call.

    Args:
        request: Incoming Frigate request.
        tools: Available OpenAI tool declarations.
        name: Function name to call.
        arguments: Exact deterministic arguments.

    Returns:
        tuple | None: Selected tools and validated call, or None when unsupported.
    """
    chosen = _tool(tools, name)
    if not _supports(chosen, arguments):
        return None
    prepared = request.model_copy(update={"tools": chosen})
    direct = response_message(
        {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}, prepared, ""
    )
    return chosen, direct


def hassil_plan(request, settings, images):
    """Turn a unique Frigate HassIL match into a safe deterministic plan.

    Args:
        request: Incoming Frigate request.
        settings: Service settings (kept for planner interface symmetry).
        images: Whether this turn requires image observation.

    Returns:
        tuple | None: Selected tools, direct response/call and route reason, or None.
    """
    del settings
    if images or request.tool_choice == "required" or isinstance(request.tool_choice, dict):
        return None
    match, trace = recognize_request(request)
    request._metrics["frigate_hassil"] = trace
    if match is None:
        return None
    tools = selected_tools(request)
    slots = match["slots"]
    intent = match["intent"]
    if intent == "FrigateLiveContext":
        result = _call(request, tools, "get_live_context", {"camera": slots["camera"]})
        return (*result, "hassil_live_context") if result else None
    if intent == "FrigateAbsenceRecap":
        result = _call(request, tools, "get_profile_status", {})
        return (*result, "hassil_absence_profile") if result else None
    if intent == "FrigateStopWatch":
        result = _call(request, tools, "stop_camera_watch", {})
        return (*result, "hassil_stop_watch") if result else None
    if intent == "FrigateSetCameraState":
        arguments = {
            "camera": slots["camera"],
            "feature": slots["feature"],
            "value": slots["state"],
        }
        result = _call(request, tools, "set_camera_state", arguments)
        return (*result, "hassil_camera_state") if result else None
    if intent == "FrigateLastSighting":
        arguments = {"label": slots["label"], "camera": slots["camera"], "limit": 1}
        result = _call(request, tools, "search_objects", arguments)
        return (*result, "hassil_last_sighting") if result else None
    if intent in {"FrigateRecap", "FrigateSearchObjects"}:
        now = _local_datetime(server_time(request.messages))
        text = next(
            (text_content(message) for message in reversed(request.messages) if message.get("role") == "user"),
            "",
        )
        interval = _interval(text, now) if now else None
        if interval is None:
            return None
        after, before = (value.isoformat(timespec="seconds") for value in interval)
        if intent == "FrigateRecap":
            arguments = {"after": after, "before": before}
            result = _call(request, tools, "get_recap", arguments)
            if result:
                return *result, "hassil_recap_interval"
            result = _call(request, tools, "search_objects", arguments)
            return (*result, "hassil_recap_search_fallback") if result else None
        fixed = {
            "label": slots["label"],
            "after": after,
            "before": before,
        }
        chosen = _tool(tools, "search_objects")
        if not chosen:
            return None
        # Preserve the established contract for filtered historical searches: the
        # model may still need to carry additional explicit filters, but it only
        # sees the one capability and cannot change already resolved arguments.
        chosen = copy.deepcopy(chosen)
        properties = chosen[0]["function"].get("parameters", {}).get("properties", {})
        for key, value in fixed.items():
            if isinstance(properties.get(key), dict):
                properties[key]["enum"] = [value]
        return chosen, None, "hassil_object_search_tool_selection"
    return None
