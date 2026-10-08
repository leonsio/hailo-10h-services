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

from hailo_services.assistants.frigate.frigate_context import (
    select_tool as _tool,
)
from hailo_services.assistants.frigate.frigate_deterministic import (
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
from hailo_services.assistants.frigate.frigate_prompt import (
    camera_catalogue,
    server_time,
    text_content,
)
from hailo_services.shared.i18n import SUPPORTED_LANGUAGES, catalogue
from hailo_services.shared.intent_engine import (
    mapped_text_slot,
    restrict_grammar,
    result_slots,
    unique_recognition,
)
from hailo_services.shared.tool_calling import response_message, selected_tools

# These sentence templates cover only requests whose full meaning can be preserved
# deterministically. Free-form descriptions, compound requests and semantic searches
# intentionally fall through to the existing planners/models.
_SENTENCES = {
    language: catalogue(language)["frigate"]["sentences"] for language in SUPPORTED_LANGUAGES
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
        (
            text_content(message).strip()
            for message in reversed(request.messages)
            if message.get("role") == "user"
        ),
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
    trace.update(
        reason=reason, language=language, duration_ms=(time.perf_counter() - started) * 1000
    )
    if result is None:
        return None, trace
    slots = result_slots(result)
    match = {"intent": result.intent.name, "slots": slots, "language": language}
    trace.update(matched=True, intent=result.intent.name, slots=slots)
    return match, trace


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
            (
                text_content(message)
                for message in reversed(request.messages)
                if message.get("role") == "user"
            ),
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
