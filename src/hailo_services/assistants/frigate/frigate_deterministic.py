"""Multilingual deterministic shortcuts for common Frigate/NVR requests."""

from __future__ import annotations

import copy
import re
from datetime import timedelta

from hailo_services.assistants.frigate.frigate_context import (
    local_datetime as _local_datetime,
)
from hailo_services.assistants.frigate.frigate_context import (
    matched_cameras as _cameras,
)
from hailo_services.assistants.frigate.frigate_context import (
    select_tool as _tools,
)
from hailo_services.assistants.frigate.frigate_context import (
    tool_call as _call,
)
from hailo_services.assistants.frigate.frigate_context import (
    tool_results as _results,
)
from hailo_services.assistants.frigate.frigate_prompt import (
    camera_catalogue,
    server_time,
    text_content,
)
from hailo_services.assistants.frigate.frigate_routing import recap_interval
from hailo_services.shared.i18n import detect_language, language_code, translate, vocabulary
from hailo_services.shared.tool_calling import selected_tools

_RECAP = vocabulary("_RECAP")
_LIVE = vocabulary("_LIVE")
_PRESENCE = vocabulary("_PRESENCE")
_SIMILAR = vocabulary("_SIMILAR")
_SEARCH = vocabulary("_SEARCH")
_AWAY = vocabulary("_AWAY")
_LAST = vocabulary("_LAST")
_ONE = vocabulary("_ONE")
_TODAY = vocabulary("_TODAY")
_YESTERDAY = vocabulary("_YESTERDAY")
_UNITS = vocabulary("_UNITS")
_FROM = vocabulary("_FROM")
_TO = vocabulary("_TO")
_BETWEEN = vocabulary("_BETWEEN")
_AND = vocabulary("_AND")
_SINCE = vocabulary("_SINCE")
_WATCH = vocabulary("_WATCH")
_STOP = vocabulary("_STOP")
_ACTION = vocabulary("_ACTION")
_ON = vocabulary("_ON")
_OFF = vocabulary("_OFF")
_FEATURES = vocabulary("_FEATURES")
_WHEN = vocabulary("_WHEN")
_SEEN = vocabulary("_SEEN")
_LABEL_ALIASES = vocabulary("_LABEL_ALIASES")


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
    return language_code(explicit, fallback=detect_language(text, "en"))


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
    return translate("frigate." + key, language=language)


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
        text_content(message) for message in request.messages if message.get("role") == "user"
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
