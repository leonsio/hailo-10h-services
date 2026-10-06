"""Official HassIL grammars -> validated OpenAI calls using request-scoped slots."""

import copy
import json
import time
import uuid
from functools import lru_cache
from itertools import islice

from hassil import Intents, TextSlotList, recognize_all
from hassil.errors import HassilError
from home_assistant_intents import get_intents
from jsonschema import ValidationError, validate

from .ha_fuzzy import slot_repairs
from .ha_state_routing import _entries
from .tool_retrieval import latest_user_text

# Only tool names supplied by this client are reachable. No HA connection here.
_DOMAINS = {
    "HassLightSet": "light",
    "HassSetPosition": "cover",
    "HassClimateSetTemperature": "climate",
    "HassClimateGetTemperature": "climate",
    "HassMediaPause": "media_player",
    "HassMediaUnpause": "media_player",
    "HassMediaNext": "media_player",
    "HassMediaPrevious": "media_player",
    "HassSetVolume": "media_player",
    "HassVacuumStart": "vacuum",
    "HassVacuumReturnToBase": "vacuum",
}


@lru_cache(maxsize=32)
def _grammar(language, supported):
    """Load and cache official HassIL grammars for the available client tools.

    Args:
        language (str | None): Language code; None uses the configured or detected language.
        supported (tuple[str, ...]): Intent names exposed by the requesting client.

    Returns:
        Intents | None: Restricted intent grammar, or None when language is unsupported.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    document = get_intents(language)
    if document is None:
        return None
    document = copy.deepcopy(document)
    document["intents"] = {
        name: data for name, data in document["intents"].items() if name in supported
    }
    # Small declarative supplements for existing service phrases absent from the
    # upstream grammar. They use the same HassIL slots and validation path.
    brightness = {
        "de": [
            "(stelle|setze) [das] Licht (im|in [dem]|in der) {area} auf {brightness} (Prozent|%)"
        ],
        "en": ["set [the] (light|lights) in [the] {area} to {brightness} (percent|%)"],
        "ru": ["(установи|поставь) свет в {area} на {brightness} (процентов|%)"],
    }
    if "HassLightSet" in document["intents"] and language in brightness:
        document["intents"]["HassLightSet"]["data"].append(
            {"sentences": brightness[language], "slots": {"domain": "light"}}
        )
    return Intents.from_dict(document)


def _arguments(result, tool, entities):
    """Resolve arguments from a historical call or recognized intent.

    Args:
        result (Any): Native response or parsed HA action/intent result.
        tool (dict[str, Any]): Client-provided function schema.
        entities (list[dict[str, Any]]): Static or live HA entities relevant to the request.

    Returns:
        dict[str, Any] | None: Decoded arguments, or None when intent slots are unsafe.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    slots = {key: value.value for key, value in result.entities.items()}
    intent = result.intent.name
    schema = tool["function"].get("parameters", {})
    properties = schema.get("properties", {})
    # Silently dropping a qualifier (device_class, color, etc.) could broaden an action.
    if set(slots) - set(properties):
        return None
    arguments = dict(slots)
    domain = slots.get("domain") or result.context.get("domain") or _DOMAINS.get(intent)
    if domain and "domain" in properties and "domain" not in arguments:
        arguments["domain"] = domain
    if properties.get("domain", {}).get("type") == "array" and isinstance(
        arguments.get("domain"), str
    ):
        arguments["domain"] = [arguments["domain"]]
    members = entities
    if "area" in arguments:
        members = [e for e in members if e["area"] == arguments["area"]]
    if domain:
        members = [e for e in members if e["domain"] == domain]
    if "name" in arguments:
        members = [e for e in members if e["name"] == arguments["name"]]
        keys = {(e["name"], e["area"], e["domain"]) for e in members}
        if len(keys) != 1:
            return None
        # Preserve area qualification for duplicate names across rooms.
        if "area" not in arguments and "area" in properties and members[0]["area"]:
            arguments["area"] = members[0]["area"]
    if intent.startswith(
        ("HassTurn", "HassLight", "HassSetPosition", "HassClimate", "HassLock", "HassUnlock")
    ):
        if not ({"area", "name"} & arguments.keys()) or not members:
            return None
    try:
        validate(arguments, schema)
    except ValidationError:
        return None
    return arguments


def deterministic_intent(request, settings, language):
    """Exact first; conservative fuzzy slots on miss. Ambiguous calls never execute.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        settings (Settings): Validated service settings controlling enabled models and limits.
        language (str | None): Language code; None uses the configured or detected language.

    Returns:
        tuple[dict | None, dict]: Unique validated call message or None, plus the recognition trace.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    started = time.perf_counter()
    trace = {"stage": "hassil", "exact": "miss", "fuzzy": [], "candidates": []}
    if request.tool_choice == "none" or request.messages[-1].get("role") != "user":
        trace["reason"] = "tool_choice_or_followup"
        return None, trace
    query = latest_user_text(request.messages).strip()
    if len(query) > 512:
        trace["reason"] = "query_too_long"
        return None, trace
    tools = {}
    for tool in request.tools or []:
        name = tool["function"]["name"]
        intent = name.rsplit("__", 1)[-1]
        if intent.startswith("Hass"):
            tools.setdefault(intent, []).append(tool)
    if isinstance(request.tool_choice, dict):
        forced = request.tool_choice["function"]["name"]
        tools = {
            intent: [t for t in group if t["function"]["name"] == forced]
            for intent, group in tools.items()
        }
        tools = {intent: group for intent, group in tools.items() if group}
    grammar = _grammar(language.split("-")[0], tuple(sorted(tools))) if tools else None
    if grammar is None or not grammar.intents:
        trace["reason"] = "no_supported_intents"
        return None, trace
    entities = _entries(request.messages)
    # Name slots include actual domain context for grammar requires_context rules.
    names = [
        {"in": e["name"], "out": e["name"], "context": {"domain": e["domain"]}} for e in entities
    ]
    lists = {
        "area": TextSlotList.from_strings(sorted({e["area"] for e in entities if e["area"]})),
        "floor": TextSlotList.from_strings([]),
    }
    lists["name"] = Intents.from_dict(
        {"language": language, "intents": {}, "lists": {"name": {"values": names}}}
    ).slot_lists["name"]

    def match(text):
        """Recognize and validate distinct candidate calls for one utterance.

        Args:
            text (str): Text to parse, normalize, match or render.

        Returns:
            dict[tuple, tuple]: Distinct validated calls indexed by tool name and arguments.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        calls = {}
        try:
            results = list(
                islice(recognize_all(text, grammar, slot_lists=lists, language=language), 65)
            )
        except HassilError as exc:
            # A grammar may require a context/slot absent from this OpenAI client.
            trace["reason"] = "grammar_context_unavailable"
            trace["error"] = str(exc)
            return {}
        if len(results) > 64:
            trace["reason"] = "too_many_matches"
            return {}
        for result in results:
            for tool in tools[result.intent.name]:
                args = _arguments(result, tool, entities)
                trace["candidates"].append(
                    {
                        "intent": result.intent.name,
                        "slots": {k: v.value for k, v in result.entities.items()},
                        "validated": args is not None,
                    }
                )
                if args is not None:
                    key = (tool["function"]["name"], json.dumps(args, sort_keys=True))
                    calls[key] = (tool, args, result.intent.name)
        return calls

    calls = match(query)
    source = "hassil_exact"
    if calls:
        trace["exact"] = "hit"
    elif settings.ha_assist_fuzzy_enabled:
        values = [e[key] for e in entities for key in ("name", "area")]
        trace["fuzzy"] = slot_repairs(
            query,
            values,
            threshold=settings.ha_assist_fuzzy_threshold,
            margin=settings.ha_assist_fuzzy_margin,
        )
        if trace["fuzzy"]:
            calls = match(trace["fuzzy"][0]["text"])
            source = "hassil_fuzzy"
    trace["duration_ms"] = (time.perf_counter() - started) * 1000
    if len(calls) != 1:
        trace.setdefault("reason", "ambiguous" if calls else "no_validated_match")
        return None, trace
    tool, args, intent = next(iter(calls.values()))
    trace.update(source=source, intent=intent, arguments=args, reason="validated")
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        ],
    }, trace
