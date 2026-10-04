"""Fast and safe routing for simple Home Assistant state questions.

State questions such as ``Ist das Licht im Wohnzimmer an?`` must never be
interpreted as actions merely because they contain words such as ``an`` or
``aus``.  For unambiguous targets this module emits GetLiveContext directly,
without running Gemma for tool selection.  The follow-up containing live data
is then reduced to a tiny text-only prompt before Gemma formats the answer.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from functools import wraps

from .tool_retrieval import _static_context_parts, latest_user_text

_LOG = logging.getLogger(__name__)
_LIVE_TOOL = "homeassistant__GetLiveContext"
_FINAL_SYSTEM = (
    "Beantworte die Nutzerfrage kurz und ausschließlich anhand der folgenden "
    "Home-Assistant-Live-Daten. Erfinde keine Zustände oder Werte."
)
_DOMAIN_WORDS = {
    "light": {"licht", "lichter", "lampe", "lampen", "led"},
    "switch": {"schalter", "switch"},
    "cover": {"rollladen", "rolllaeden", "jalousie", "jalousien", "markise"},
    "climate": {"thermostat", "heizung", "klima"},
    "vacuum": {"staubsauger", "saugroboter", "vacuum"},
}


def _normalized(value: str) -> str:
    text = str(value).casefold().replace("ß", "ss")
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _contains(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _is_state_question(text: str) -> bool:
    """Recognize explicit state/value questions, not control commands."""
    text = _normalized(text)
    if not text:
        return False
    # Explicit action verbs always win over a state-looking suffix.
    if re.search(
        r"\b(schalt\w*|mach\w*|einschalten|ausschalten|anschalten|anmachen|"
        r"ausmachen|aktiviere\w*|deaktiviere\w*|stell\w*|setze\w*)\b",
        text,
    ):
        return False
    if re.search(r"\b(ist|sind|steht|stehen)\b.*\b(an|aus|on|off|offen|geschlossen)\b", text):
        return True
    if re.search(r"\b(status|zustand|modus)\b", text):
        return True
    if re.search(r"\b(wie warm|wie kalt|temperatur|luftfeuchtigkeit)\b", text):
        return True
    return False


def _entries(messages) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        parts = _static_context_parts(message["content"])
        if parts is None:
            continue
        for entry in parts[1]:
            name = re.search(r"(?m)^- names:\s*(.+)$", entry)
            domain = re.search(r"(?m)^\s+domain:\s*(.+)$", entry)
            area = re.search(r"(?m)^\s+areas:\s*(.+)$", entry)
            if name and domain:
                entities.append({
                    "name": name.group(1).strip(),
                    "domain": domain.group(1).strip(),
                    "area": area.group(1).strip() if area else "",
                })
    return entities


def _live_tool(tools):
    for tool in tools or []:
        if tool.get("function", {}).get("name") == _LIVE_TOOL:
            return tool
    return None


def _query_domain(query: str) -> str | None:
    words = set(_normalized(query).split())
    matches = [domain for domain, aliases in _DOMAIN_WORDS.items() if words & aliases]
    return matches[0] if len(matches) == 1 else None


def _target_arguments(messages, query: str):
    entities = _entries(messages)
    if not entities:
        return None
    normalized_query = _normalized(query)
    domain_hint = _query_domain(query)

    explicit = [
        entity for entity in entities
        if _contains(normalized_query, _normalized(entity["name"]))
        and (domain_hint is None or entity["domain"] == domain_hint)
    ]
    explicit_keys = {(item["name"], item["domain"], item["area"]) for item in explicit}
    if len(explicit_keys) == 1:
        entity = explicit[0]
        return {"name": entity["name"], "domain": [entity["domain"]]}
    if explicit:
        return None

    area_names = sorted({entity["area"] for entity in entities if entity["area"]}, key=len, reverse=True)
    matching_areas = [area for area in area_names if _contains(normalized_query, _normalized(area))]
    if len(matching_areas) != 1:
        return None
    area = matching_areas[0]
    members = [entity for entity in entities if entity["area"] == area]
    if domain_hint is not None:
        members = [entity for entity in members if entity["domain"] == domain_hint]
    domains = {entity["domain"] for entity in members}
    if len(domains) != 1:
        return None
    return {"area": area, "domain": [next(iter(domains))]}


def direct_live_context_response(request):
    """Return a direct GetLiveContext call for an unambiguous state question."""
    tool = _live_tool(request.tools)
    query = latest_user_text(request.messages).strip()
    if tool is None or not _is_state_question(query):
        return None
    arguments = _target_arguments(request.messages, query)
    if arguments is None:
        return None
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": _LIVE_TOOL,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
    }


def _live_followup(messages):
    """Return (question, results) for a pure GetLiveContext follow-up."""
    question = latest_user_text(messages).strip()
    if not _is_state_question(question):
        return None
    calls = []
    results = []
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            if call.get("function", {}).get("name") != _LIVE_TOOL:
                return None
            calls.append(call.get("id"))
        if message.get("role") == "tool":
            results.append((message.get("tool_call_id"), message.get("content", "")))
    if not calls or {identifier for identifier, _ in results} != set(calls):
        return None
    return question, [content for _, content in results]


def compact_live_followup_request(request):
    """Reduce a simple live-state follow-up to system + user/tool-result text."""
    followup = _live_followup(request.messages)
    if followup is None:
        return None
    question, results = followup
    live_text = "\n".join(str(result) for result in results)
    messages = [
        {"role": "system", "content": _FINAL_SYSTEM},
        {
            "role": "user",
            "content": f"Frage: {question}\nHome-Assistant-Live-Daten:\n{live_text}",
        },
    ]
    prepared = request.model_copy(update={
        "messages": messages,
        "tools": None,
        "tool_choice": None,
    })
    prepared._request_id = getattr(request, "_request_id", "-")
    return prepared


def install():
    """Install state-query routing after the general HA routing hooks."""
    from . import runtime

    cls = runtime.HailoBackend
    if getattr(cls, "_ha_state_routing_installed", False):
        return
    original_select_tools = cls.select_tools

    @wraps(original_select_tools)
    def select_tools(self, request):
        request_id = getattr(request, "_request_id", "-")

        compact = compact_live_followup_request(request)
        if compact is not None:
            _LOG.info(
                "ha_route request_id=%s route=live_context_followup_minimal tools=0",
                request_id,
            )
            if self.settings.debug_log:
                _LOG.debug(
                    "event=ha_state_route request_id=%s json=%s",
                    request_id,
                    json.dumps({
                        "route": "live_context_followup_minimal",
                        "messages": compact.messages,
                    }, ensure_ascii=False, separators=(",", ":"), default=str),
                )
            return compact

        query = latest_user_text(request.messages).strip()
        live_tool = _live_tool(request.tools)
        if live_tool is not None and _is_state_question(query):
            direct = direct_live_context_response(request)
            if direct is not None:
                prepared = request.model_copy(update={
                    "tools": [live_tool],
                    "tool_choice": None,
                })
                prepared._request_id = request_id
                object.__setattr__(prepared, "_direct_ha_response", direct)
                _LOG.info(
                    "ha_route request_id=%s route=direct_live_context skipped_gemma=true",
                    request_id,
                )
                if self.settings.debug_log:
                    _LOG.debug(
                        "event=ha_state_route request_id=%s json=%s",
                        request_id,
                        json.dumps({
                            "route": "direct_live_context",
                            "tool_call": direct["tool_calls"][0],
                        }, ensure_ascii=False, separators=(",", ":"), default=str),
                    )
                return prepared

            # Even if the target is too ambiguous for a deterministic direct
            # call, a state question must never expose TurnOn/TurnOff to Gemma.
            narrowed = request.model_copy(update={"tools": [live_tool]})
            narrowed._request_id = request_id
            _LOG.info(
                "ha_route request_id=%s route=live_context_only reason=ambiguous_target",
                request_id,
            )
            return original_select_tools(self, narrowed)

        return original_select_tools(self, request)

    cls.select_tools = select_tools
    cls._ha_state_routing_installed = True
