"""Targeted fixes for deterministic Home Assistant state-query routing.

These overrides keep common area-scoped state questions on the fast path even
when the global entity list contains generic names such as ``Licht``. They also
turn unknown room names into a direct GetLiveContext lookup so Home Assistant,
not Gemma, decides whether the area exists. Tool errors are rendered as concise
user-facing messages instead of being mistaken for entity states.
"""

from __future__ import annotations

import re
from functools import wraps

from . import ha_state_routing as _state

_HOME_LOCATIONS = {
    "haus",
    "ganzen haus",
    "gesamten haus",
    "zuhause",
    "zu hause",
    "home",
}
_LOCATION_RE = re.compile(
    r"\b(?:im|in\s+der|in\s+dem|in\s+den|in)\s+(.+?)"
    r"(?:\s+(?:ist|sind|steht|stehen))?\s+"
    r"(?:an|aus|offen|geschlossen|gesperrt|verriegelt|entsperrt|entriegelt|"
    r"on|off|locked|unlocked)\b",
    re.IGNORECASE,
)
_ERROR_KEYS = {"error", "message", "detail", "details"}


def _location_phrase(query: str) -> str | None:
    """Extract a strongly expressed location from a state question."""
    match = _LOCATION_RE.search(str(query))
    if not match:
        return None
    value = match.group(1).strip(" \t\r\n?!.,:;")
    return value or None


def _target_arguments(messages, query: str):
    """Resolve area before global entity names and keep unknown areas direct."""
    entities = _state._entries(messages)
    if not entities:
        return None

    normalized_query = _state._normalized(query)
    domain_hint = _state._query_domain(query)
    kind = _state._query_kind(query)

    # An explicitly mentioned area has precedence over generic entity names
    # elsewhere in the house.  Production installations often contain many
    # entities named simply "Licht", which must not make
    # "Ist das Licht im Wohnzimmer an?" ambiguous.
    area_names = sorted(
        {entity["area"] for entity in entities if entity["area"]},
        key=len,
        reverse=True,
    )
    matching_areas = [
        area for area in area_names
        if _state._contains(normalized_query, _state._normalized(area))
    ]
    if len(matching_areas) == 1:
        area = matching_areas[0]
        area_entities = [entity for entity in entities if entity["area"] == area]

        explicit = [
            entity for entity in area_entities
            if _state._contains(normalized_query, _state._normalized(entity["name"]))
            and (domain_hint is None or entity["domain"] == domain_hint)
        ]
        explicit_keys = {
            (item["name"], item["domain"], item["area"]) for item in explicit
        }
        if len(explicit_keys) == 1:
            entity = explicit[0]
            return {"name": entity["name"], "domain": [entity["domain"]]}
        if explicit:
            return None

        members = area_entities
        if domain_hint is not None:
            # Querying HA directly remains safe even if the static context has
            # no member of that domain; HA then returns a precise no-match
            # result which we format without involving Gemma.
            return {"area": area, "domain": [domain_hint]}
        if kind in {"temperature", "humidity"}:
            members = _state._measurement_members(members, kind)
            unique = {(item["name"], item["domain"]) for item in members}
            if len(unique) == 1:
                entity = members[0]
                return {"name": entity["name"], "domain": [entity["domain"]]}
            return None

        domains = {entity["domain"] for entity in members}
        if len(domains) == 1:
            return {"area": area, "domain": [next(iter(domains))]}
        if kind in {"status", "list", "count", "all"}:
            return {"area": area}
        return None
    if matching_areas:
        return None

    # If the sentence explicitly says "im/in der <room>" but that room is not
    # present in the static area list, ask Home Assistant directly instead of
    # letting Gemma guess a similarly named entity (for example Gästezimmer
    # for Schlafzimmer).
    location = _location_phrase(query)
    if location and domain_hint is not None:
        normalized_location = _state._normalized(location)
        if normalized_location in _HOME_LOCATIONS:
            return {"domain": [domain_hint]}
        return {"area": location, "domain": [domain_hint]}

    # Without an area, retain the original exact-name behaviour.
    explicit = [
        entity for entity in entities
        if _state._contains(normalized_query, _state._normalized(entity["name"]))
        and (domain_hint is None or entity["domain"] == domain_hint)
    ]
    explicit_keys = {
        (item["name"], item["domain"], item["area"]) for item in explicit
    }
    if len(explicit_keys) == 1:
        entity = explicit[0]
        return {"name": entity["name"], "domain": [entity["domain"]]}
    if explicit:
        return None

    if domain_hint is not None and kind in {"list", "count", "all"}:
        return {"domain": [domain_hint]}
    return None


def _friendly_live_error(calls, results) -> str | None:
    """Return a deterministic user-facing message for HA lookup failures."""
    for index, content in enumerate(results):
        payload = _state._json_object(content)
        if payload is None:
            continue
        error = payload.get("error")
        failed = payload.get("success") is False
        if error is None and not failed:
            continue

        call = calls[index] if index < len(calls) else (calls[0] if calls else {})
        arguments = _state._json_object(call.get("function", {}).get("arguments")) or {}
        area = arguments.get("area")
        name = arguments.get("name")
        domains = arguments.get("domain")
        if isinstance(domains, str):
            domains = [domains]

        if area and domains == ["light"]:
            return f"Ich konnte im Bereich „{area}“ keine passenden Lichter finden."
        if area:
            return f"Ich konnte im Bereich „{area}“ keine passenden Geräte finden."
        if name:
            return f"Ich konnte „{name}“ in Home Assistant nicht finden."
        return "Home Assistant konnte keine passenden freigegebenen Geräte finden."
    return None


def install():
    """Install the routing overrides once."""
    if getattr(_state, "_area_state_fixes_installed", False):
        return

    original_deterministic_live_response = _state.deterministic_live_response
    original_entities_from_mapping = _state._entities_from_mapping

    @wraps(original_deterministic_live_response)
    def deterministic_live_response(request):
        followup = _state._live_followup(request.messages)
        if followup is not None:
            _, calls, results = followup
            friendly_error = _friendly_live_error(calls, results)
            if friendly_error is not None:
                return friendly_error
        return original_deterministic_live_response(request)

    @wraps(original_entities_from_mapping)
    def entities_from_mapping(payload, call):
        clean_payload = {
            key: value for key, value in payload.items() if key not in _ERROR_KEYS
        }
        return original_entities_from_mapping(clean_payload, call)

    _state._target_arguments = _target_arguments
    _state.deterministic_live_response = deterministic_live_response
    _state._entities_from_mapping = entities_from_mapping
    _state._area_state_fixes_installed = True
