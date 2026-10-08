"""High-confidence Frigate shortcuts that finish simple tool rounds without inference."""

from __future__ import annotations

import re
from numbers import Real

from hailo_services.assistants.frigate.frigate_context import (
    matched_cameras,
    select_tool,
    tool_call,
    tool_results,
)
from hailo_services.assistants.frigate.frigate_deterministic import (
    _LABEL_ALIASES,
    _WHEN,
    _language,
)
from hailo_services.assistants.frigate.frigate_prompt import camera_catalogue, text_content
from hailo_services.assistants.frigate.frigate_routing import (
    event_interval,
    last_sighting_result,
)
from hailo_services.shared.i18n import catalogue
from hailo_services.shared.tool_calling import arguments_object, selected_tools


def _latest_text(request):
    """Return the latest user text without copying assistant text."""
    return next(
        (text_content(message).strip() for message in reversed(request.messages) if message.get("role") == "user"),
        "",
    )


def _active_call_arguments(request, name):
    """Return arguments for the latest active-round call of ``name``."""
    last_user = max(
        (index for index, message in enumerate(request.messages) if message.get("role") == "user"),
        default=-1,
    )
    for message in reversed(request.messages[last_user + 1 :]):
        for call in reversed(message.get("tool_calls", [])):
            function = call.get("function", {})
            if function.get("name") == name:
                try:
                    return arguments_object(function.get("arguments", {}))
                except ValueError:
                    return None
    return None


def _zone_catalogue(messages):
    """Extract zone IDs and friendly names from Frigate's supplied camera catalogue."""
    zones = {}
    for message in messages:
        if message.get("role") != "system":
            continue
        for line in text_content(message).splitlines():
            if "zones:" not in line:
                continue
            tail = line.split("zones:", 1)[1]
            for match in re.finditer(r"([^,()]+?)\s*\(ID:\s*([^,()]+)\)", tail):
                zones[match[2].strip()] = match[1].strip()
    return zones


def _string_list(value):
    """Validate a compact list of strings returned by Frigate."""
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return [item.strip() for item in value if item.strip()]


def _render_recap(request, events):
    """Render every canonical recap event without truncation or model inference."""
    cameras = camera_catalogue(request.messages)
    zones = _zone_catalogue(request.messages)
    lines = []
    for event in events:
        if not isinstance(event, dict) or any(
            key in event for key in ("error", "partial", "_omitted", "_omitted_records")
        ):
            return None
        stamp = event.get("time") or event.get("start_time_local")
        camera = event.get("camera")
        if not isinstance(stamp, str) or not stamp.strip() or not isinstance(camera, str):
            return None
        objects = _string_list(event.get("objects"))
        event_zones = _string_list(event.get("zones"))
        if objects is None or event_zones is None:
            return None
        severity = event.get("severity")
        description = event.get("description")
        duration = event.get("duration_seconds")
        if severity is not None and not isinstance(severity, str):
            return None
        if description is not None and not isinstance(description, str):
            return None
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, Real)):
            return None

        parts = [stamp.strip(), cameras.get(camera, camera)]
        if severity:
            parts.append(severity.strip())
        if objects:
            parts.append(", ".join(objects))
        if event_zones:
            parts.append("@ " + ", ".join(zones.get(zone, zone) for zone in event_zones))
        if duration is not None:
            parts.append(f"{duration:g}s")
        if description and description.strip():
            parts.append(re.sub(r"\s+", " ", description).strip())
        lines.append("- " + " · ".join(parts))
    return "\n".join(lines) if lines else None


def _simple_recap_result(request, results):
    """Finish an explicit event-list request when get_recap returned canonical data."""
    if set(results) != {"get_recap"}:
        return None
    recap = results.get("get_recap")
    if not isinstance(recap, dict) or not set(recap).issubset({"events", "message"}):
        return None
    events = recap.get("events")
    if not isinstance(events, list) or not events:
        return None
    recognized, interval, clarification = event_interval(request)
    if not recognized or interval is None or clarification is not None:
        return None
    actual = _active_call_arguments(request, "get_recap")
    if actual != interval:
        return None
    return _render_recap(request, events)


def _home_span(request, text):
    """Locate a localized home/house reference using the existing locale aliases."""
    language = _language(request, text)
    aliases = set(catalogue(language).get("aliases", {}).get("haus", []))
    aliases.add("haus")
    matches = []
    for alias in aliases:
        alias = alias.strip()
        if not alias:
            continue
        match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I)
        if match:
            matches.append(match.span())
    # The German canonical stem also covers grammatical/compound forms such as
    # "Hause" and "zuhause" without storing those phrases in Python code.
    for match in re.finditer(r"(?<!\w)[\wÀ-ÖØ-öø-ÿА-Яа-яЁё'-]*haus[\wÀ-ÖØ-öø-ÿА-Яа-яЁё'-]*(?!\w)", text, re.I):
        matches.append(match.span())
    return min(matches, key=lambda span: span[0]) if matches else None


def _named_home_query(request, text):
    """Resolve a named-person question about being/coming home to a last sighting query.

    The shortcut intentionally returns evidence (the latest matching sighting), not
    a claim that the sighting proves an arrival. Ambiguous/lower-case free-form names
    fall through to the normal model path.
    """
    words = re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", text, re.UNICODE)
    if not any(word.casefold() in _WHEN for word in words):
        return None
    home = _home_span(request, text)
    if home is None:
        return None
    prefix = text[: home[0]]
    cameras = camera_catalogue(request.messages)
    camera_matches = matched_cameras(text, cameras)
    if len(camera_matches) > 1:
        return None

    quoted = re.findall(r"[\"“„]([^\"”]+)[\"”]", prefix)
    if len(quoted) == 1:
        candidate = quoted[0].strip()
        if re.fullmatch(r"[^\W\d_][\wÀ-ÖØ-öø-ÿА-Яа-яЁё .'-]{0,79}", candidate, re.UNICODE):
            result = {"sub_label": candidate}
            if camera_matches:
                result["camera"] = camera_matches[0]
            return result

    language = _language(request, text)
    data = catalogue(language)
    blocked = {word.casefold() for word in _WHEN}
    blocked.update(word.casefold() for word in _LABEL_ALIASES.get("person", set()))
    blocked.update(word.casefold() for word in data.get("frigate", {}).get("hints", []))
    for canonical in ("haus", "ist", "im", "auf", "bitte"):
        blocked.update(word.casefold() for word in data.get("aliases", {}).get(canonical, []))
    for identifier, friendly in cameras.items():
        blocked.update(word.casefold() for word in re.findall(r"[^\W\d_]+", identifier, re.UNICODE))
        blocked.update(word.casefold() for word in re.findall(r"[^\W\d_]+", friendly, re.UNICODE))

    names = []
    for word in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", prefix, re.UNICODE):
        plain = word.casefold()
        if not word[:1].isupper() or plain in blocked:
            continue
        if "haus" in plain:
            continue
        names.append(word)
    if not 1 <= len(names) <= 3:
        return None
    candidate = " ".join(names)
    result = {"sub_label": candidate}
    if camera_matches:
        result["camera"] = camera_matches[0]
    return result


def _named_home_plan(request, tools, results):
    """Call/search and then deterministically summarize a named home sighting."""
    text = _latest_text(request)
    query = _named_home_query(request, text)
    if query is None:
        return None
    cameras = camera_catalogue(request.messages)
    if "search_objects" in results:
        direct = last_sighting_result(request, (query, False), results, tools, cameras)
        if direct:
            chosen, response, _reason = direct
            return chosen, response, "home_named_sighting_summary"
        return None
    if results:
        return None
    chosen = select_tool(tools, "search_objects")
    if not chosen:
        return None
    parameters = chosen[0]["function"].get("parameters", {})
    properties = parameters.get("properties", {})
    arguments = {**query, "limit": 1}
    if not set(arguments).issubset(properties) or not set(parameters.get("required", [])).issubset(arguments):
        return None
    prepared = request.model_copy(update={"tools": chosen})
    return chosen, tool_call(prepared, "search_objects", arguments), "home_named_sighting"


def shortcut_plan(request, settings, images):
    """Return a safe deterministic shortcut or ``None`` for the established planners."""
    del settings
    if images:
        return None
    tools = selected_tools(request)
    results = tool_results(request)
    recap = _simple_recap_result(request, results)
    if recap is not None:
        return [], recap, "deterministic_recap_list"
    return _named_home_plan(request, tools, results)
