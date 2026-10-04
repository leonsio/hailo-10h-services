"""Low-latency routing for Home Assistant text requests.

The Home Assistant client sends a large static entity catalogue and many tool
schemas with every request. MiniLM is used as a cheap relevance gate before
Gemma:

* unrelated/general questions drop the HA system prompt and tools and are sent
  to Gemma like a plain web chat;
* unambiguous turn-on/turn-off commands are converted directly to an OpenAI
  tool call without running Gemma;
* everything else keeps the existing entity/tool retrieval + Gemma path.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from functools import wraps

from .tool_retrieval import (
    _embedding,
    _score,
    _static_context_parts,
    _text,
    _tool_score,
    _tokens,
    latest_user_text,
)

_LOG = logging.getLogger(__name__)
_SEMANTIC_RELEVANCE_THRESHOLD = 0.56
_EXTRA_STOP_WORDS = {
    "von", "vom", "zur", "zum", "zu", "auf", "für", "fuer", "mit", "ohne",
    "was", "wer", "wie", "wo", "wann", "welche", "welcher", "welches",
}
_DIRECT_TOOLS = {"intent__HassTurnOn", "intent__HassTurnOff"}


def _query_tokens(text: str) -> set[str]:
    return _tokens(text) - _EXTRA_STOP_WORDS


def _is_ha_system_message(message) -> bool:
    if message.get("role") != "system" or not isinstance(message.get("content"), str):
        return False
    content = message["content"]
    return "Static Context:" in content and "Home Assistant" in content


def _has_tool_history(messages) -> bool:
    return any(
        message.get("role") == "tool" or message.get("tool_calls")
        for message in messages
    )


def _entity_entries(messages) -> list[str]:
    entries: list[str] = []
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        parts = _static_context_parts(message["content"])
        if parts is not None:
            entries.extend(parts[1])
    return entries


def _tool_description(tool) -> str:
    function = tool.get("function", {}) if isinstance(tool, dict) else {}
    return " ".join((
        function.get("name", ""),
        function.get("description", ""),
        _text(function.get("parameters", {}))[:1400],
    ))


def assess_ha_relevance(
    messages,
    tools,
    *,
    encoder=None,
    embedding_cache=None,
    semantic_threshold: float = _SEMANTIC_RELEVANCE_THRESHOLD,
):
    """Classify whether the latest user request needs Home Assistant context."""
    query_text = latest_user_text(messages).strip()
    query = _query_tokens(query_text)
    entries = _entity_entries(messages)
    tools = tools or []

    entity_scores = [(_score(query, entry), entry) for entry in entries]
    tool_scores = [(_tool_score(query, tool), tool) for tool in tools]
    entity_lexical = max((score for score, _ in entity_scores), default=0)
    tool_lexical = max((score for score, _ in tool_scores), default=0)
    semantic_tool_score = None
    semantic_tool_name = None

    relevant = bool(entity_lexical > 0 or tool_lexical > 0)
    reason = "lexical" if relevant else "none"

    # Semantic fallback is intentionally thresholded. MiniLM may always find a
    # nearest tool, even for unrelated questions (e.g. geography), so a mere
    # top-1 result must not make the request HA-related.
    if not relevant and encoder is not None and query_text and tools:
        import numpy as np

        query_vector = _embedding(encoder, query_text, embedding_cache)
        semantic = []
        for tool in tools:
            similarity = float(np.dot(
                query_vector,
                _embedding(encoder, _tool_description(tool), embedding_cache),
            ))
            semantic.append((similarity, tool))
        if semantic:
            semantic.sort(key=lambda item: -item[0])
            semantic_tool_score, semantic_tool = semantic[0]
            semantic_tool_name = semantic_tool.get("function", {}).get("name")
            if semantic_tool_score >= semantic_threshold:
                relevant = True
                reason = "semantic_tool"

    return {
        "relevant": relevant,
        "reason": reason,
        "query_text": query_text,
        "query_tokens": sorted(query),
        "entity_lexical_max": entity_lexical,
        "tool_lexical_max": tool_lexical,
        "semantic_tool_max": semantic_tool_score,
        "semantic_tool_name": semantic_tool_name,
        "semantic_threshold": semantic_threshold,
        "entity_count": len(entries),
        "tool_count": len(tools),
    }


def general_passthrough_request(request):
    """Drop only the generated HA envelope, preserving the actual conversation."""
    messages = [
        copy.deepcopy(message)
        for message in request.messages
        if not _is_ha_system_message(message)
    ]
    return request.model_copy(update={
        "messages": messages,
        "tools": None,
        "tool_choice": None,
    })


def _normalized(value: str) -> str:
    value = str(value).casefold().replace("ß", "ss")
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _parse_entity(entry: str):
    name_match = re.search(r"(?m)^- names:\s*(.+)$", entry)
    domain_match = re.search(r"(?m)^\s+domain:\s*(.+)$", entry)
    area_match = re.search(r"(?m)^\s+areas:\s*(.+)$", entry)
    if not name_match or not domain_match:
        return None
    return {
        "name": name_match.group(1).strip(),
        "domain": domain_match.group(1).strip(),
        "area": area_match.group(1).strip() if area_match else "",
    }


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _explicit_action(query: str, tool_name: str) -> bool:
    query = _normalized(query)
    if tool_name == "intent__HassTurnOff":
        return bool(
            re.search(r"\b(ausschalten|ausmachen|deaktivieren|deaktiviere)\b", query)
            or re.search(r"\b(schalt\w*|mach\w*)\b.*\b(aus)\b", query)
            or re.search(r"\b(turn|switch)\b.*\boff\b", query)
        )
    if tool_name == "intent__HassTurnOn":
        return bool(
            re.search(r"\b(einschalten|anschalten|anmachen|aktivieren|aktiviere)\b", query)
            or re.search(r"\b(schalt\w*|mach\w*)\b.*\b(an|ein)\b", query)
            or re.search(r"\b(turn|switch)\b.*\bon\b", query)
        )
    return False


def direct_action_response(request):
    """Build a deterministic tool call for an unambiguous on/off command."""
    tools = request.tools or []
    if len(tools) != 1:
        return None
    tool_name = tools[0].get("function", {}).get("name")
    if tool_name not in _DIRECT_TOOLS:
        return None

    query = latest_user_text(request.messages).strip()
    if not _explicit_action(query, tool_name):
        return None

    entities = [
        parsed for parsed in (_parse_entity(entry) for entry in _entity_entries(request.messages))
        if parsed is not None
    ]
    if not entities:
        return None

    normalized_query = _normalized(query)
    explicit = [
        entity for entity in entities
        if _contains_phrase(normalized_query, _normalized(entity["name"]))
    ]
    # Same spoken/name target in multiple areas is ambiguous; let Gemma/HA deal
    # with it rather than issuing a potentially wrong action.
    explicit_keys = {(item["name"], item["domain"], item["area"]) for item in explicit}
    if len(explicit_keys) == 1:
        entity = explicit[0]
        arguments = {"name": entity["name"], "domain": [entity["domain"]]}
        target_kind = "entity"
    elif explicit:
        return None
    else:
        area_groups = {}
        for entity in entities:
            area = entity["area"]
            if area and _contains_phrase(normalized_query, _normalized(area)):
                area_groups.setdefault(area, []).append(entity)
        if len(area_groups) != 1:
            return None
        area, members = next(iter(area_groups.items()))
        domains = {member["domain"] for member in members}
        if len(domains) != 1:
            return None
        domain = next(iter(domains))
        arguments = {"area": area, "domain": [domain]}
        target_kind = "area"

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
        "_routing": {
            "kind": "direct_action",
            "target_kind": target_kind,
            "tool": tool_name,
            "arguments": arguments,
        },
    }


def _log_route(debug_log: bool, request_id: str, route: str, payload):
    _LOG.info("ha_route request_id=%s route=%s", request_id, route)
    if debug_log:
        _LOG.debug(
            "event=ha_route request_id=%s json=%s",
            request_id,
            json.dumps({"route": route, **payload}, ensure_ascii=False, separators=(",", ":"), default=str),
        )


def install():
    """Install routing hooks after the existing LiteRT optimizations."""
    from . import runtime

    backend_cls = runtime.HailoBackend
    litert_cls = runtime.LiteRTLMBackend
    if getattr(backend_cls, "_ha_routing_installed", False):
        return

    original_select_tools = backend_cls.select_tools
    original_litert_chat = litert_cls.chat

    @wraps(original_select_tools)
    def select_tools(self, request):
        request_id = getattr(request, "_request_id", "-")
        # Active tool rounds must retain their call/result dependencies.
        if _has_tool_history(request.messages) or isinstance(request.tool_choice, dict):
            prepared = original_select_tools(self, request)
            _log_route(self.settings.debug_log, request_id, "ha_llm", {"reason": "tool_history"})
            return prepared

        is_ha_envelope = bool(request.tools) and any(
            _is_ha_system_message(message) for message in request.messages
        )
        if not is_ha_envelope:
            return original_select_tools(self, request)

        relevance = assess_ha_relevance(
            request.messages,
            request.tools,
            encoder=self.minilm,
            embedding_cache=self._retrieval_embedding_cache,
        )
        if not relevance["relevant"]:
            prepared = general_passthrough_request(request)
            _log_route(self.settings.debug_log, request_id, "general_passthrough", relevance)
            return prepared

        prepared = original_select_tools(self, request)
        direct = direct_action_response(prepared)
        if direct is not None:
            routing = direct.pop("_routing")
            object.__setattr__(prepared, "_direct_ha_response", direct)
            _log_route(self.settings.debug_log, request_id, "direct_action", {
                **relevance,
                **routing,
            })
        else:
            _log_route(self.settings.debug_log, request_id, "ha_llm", relevance)
        return prepared

    @wraps(original_litert_chat)
    def litert_chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        direct = getattr(request, "_direct_ha_response", None)
        if direct is not None:
            request_id = getattr(request, "_request_id", "-")
            _LOG.info("direct_ha_action request_id=%s skipped_gemma=true", request_id)
            if getattr(self, "debug_log", False):
                _LOG.debug(
                    "event=direct_ha_action request_id=%s json=%s",
                    request_id,
                    json.dumps(direct, ensure_ascii=False, separators=(",", ":"), default=str),
                )
            return direct
        return original_litert_chat(self, request, emit, cancelled, tools_prepared)

    backend_cls.select_tools = select_tools
    litert_cls.chat = litert_chat
    backend_cls._ha_routing_installed = True
