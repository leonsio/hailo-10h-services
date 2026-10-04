"""Retrieve relevant Home Assistant tool schemas before passing them to Gemma.

The incoming HA request may contain a large number of nearly identical tools or
large enums of entity IDs. Keep only tools and enum values that match the latest
user turn. The optional encoder runs MiniLM on the Hailo accelerator.
"""

from __future__ import annotations

import copy
import re
import unicodedata

_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)
_STOP_WORDS = {
    "bitte", "mal", "doch", "ein", "eine", "einen", "einem", "einer", "der", "die", "das",
    "den", "dem", "des", "im", "in", "am", "und", "oder", "mir", "mich",
    "jetzt", "es", "sie", "er", "ist", "sind", "wird", "werden", "mach", "mache",
    "macht", "schalte", "schalt", "stell", "stelle", "setze", "fahr", "fahre",
}
_SYNONYMS = {
    "an": {"turn_on", "on"},
    "aus": {"turn_off", "off"},
    "einschalten": {"turn_on", "on", "light", "switch"},
    "anschalten": {"turn_on", "on", "light", "switch"},
    "anmachen": {"turn_on", "on", "light", "switch"},
    "ausschalten": {"turn_off", "off", "light", "switch"},
    "ausmachen": {"turn_off", "off", "light", "switch"},
    "öffnen": {"open", "cover", "garage"},
    "oeffnen": {"open", "cover", "garage"},
    "hoch": {"open", "cover", "up"},
    "hochfahren": {"open", "cover", "up"},
    "runter": {"close", "cover", "down", "shutter"},
    "herunter": {"close", "cover", "down", "shutter"},
    "schließen": {"close", "cover", "shutter"},
    "schliessen": {"close", "cover", "shutter"},
    "runterfahren": {"close", "cover", "down"},
    "herunterfahren": {"close", "cover", "down"},
    "temperatur": {"climate", "temperature"},
    "wärmer": {"heat", "temperature", "climate"},
    "kaelter": {"cool", "temperature", "climate"},
    "kälter": {"cool", "temperature", "climate"},
    "heller": {"brightness", "light"},
    "dunkler": {"brightness", "light"},
}


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    return ""


def _tokens(value) -> set[str]:
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", _text(value)).casefold().replace("ß", "ss")
    raw = raw.replace("_", " ").replace(".", " ").replace("-", " ")
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    result: set[str] = set()
    for word in _WORD_RE.findall(normalized):
        word = word.strip("._-")
        if not word:
            continue
        if word not in _STOP_WORDS and len(word) > 1:
            result.add(word)
        for ending in ("ern", "en", "er", "es", "e", "n", "s"):
            if len(word) - len(ending) >= 4 and word.endswith(ending):
                result.add(word[:-len(ending)])
                break
        result.update(_SYNONYMS.get(word, ()))
    return result


def latest_user_text(messages) -> str:
    """Read only text from the latest user message (ignore image payloads)."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""
    return ""


def _score(query: set[str], candidate) -> int:
    if not query:
        return 0
    words = _tokens(candidate)
    exact = len(query & words)
    # Entity IDs often contain concatenated area/name fragments.
    joined = " ".join(words)
    substring = sum(1 for word in query if len(word) >= 4 and word in joined)
    return exact * 3 + substring


def _prune_enums(node, query: set[str], stats: dict, limit: int = 12):
    if isinstance(node, dict):
        enum = node.get("enum")
        if isinstance(enum, list) and len(enum) > 1:
            ranked = [(_score(query, value), index, value) for index, value in enumerate(enum)]
            relevant = [item for item in ranked if item[0] > 0]
            if relevant:
                relevant.sort(key=lambda item: (-item[0], item[1]))
                # Leave a little breadth for similar devices/areas and stable ties.
                best = relevant[0][0]
                selected = [item for item in relevant if item[0] >= max(1, best - 2)][:limit]
                node["enum"] = [item[2] for item in selected]
                stats["enum_values_removed"] += len(enum) - len(node["enum"])
        for value in list(node.values()):
            _prune_enums(value, query, stats, limit)
    elif isinstance(node, list):
        for value in node:
            _prune_enums(value, query, stats, limit)


def retrieve_tools(messages, tools, *, max_tools: int = 12, enum_limit: int = 12, encoder=None):
    """Return a copy of tools with irrelevant definitions and enum values removed.

    If no reliable lexical hit exists, the full tool list is preserved. A forced
    tool selection or existing tool round is handled by the caller and must not
    be changed by this best-effort first pass.
    """
    if not tools:
        return tools, {"tools_before": 0, "tools_after": 0, "enum_values_removed": 0}
    query = _tokens(latest_user_text(messages))
    if encoder is not None and latest_user_text(messages).strip():
        import numpy as np

        vector = encoder.embed(latest_user_text(messages))
        ranked = []
        for index, tool in enumerate(tools):
            fn = tool.get("function", {})
            description = " ".join((fn.get("name", ""), fn.get("description", ""),
                                    _text(fn.get("parameters", {}))[:1400]))
            similarity = float(np.dot(vector, encoder.embed(description)))
            # Exact matches remain useful for entity IDs and proper names.
            ranked.append((similarity + min(_score(query, tool), 12) * 0.025, index, tool))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        chosen = [item[2] for item in ranked[:max_tools]]
        stats = {"tools_before": len(tools), "tools_after": len(chosen), "enum_values_removed": 0}
        compact = copy.deepcopy(chosen)
        for tool in compact:
            _prune_enums(tool.get("function", {}).get("parameters", {}), query, stats, enum_limit)
        return compact, stats

    ranked = [(_score(query, tool), index, tool) for index, tool in enumerate(tools)]
    hits = [item for item in ranked if item[0] > 0]
    if not hits:
        chosen = list(tools)
    else:
        hits.sort(key=lambda item: (-item[0], item[1]))
        best = hits[0][0]
        # Keep exact/high-confidence hits plus near-ties, capped to bound schemas.
        selected = [item for item in hits if item[0] >= max(1, best - 2)][:max_tools]
        chosen = [item[2] for item in selected]

    stats = {
        "tools_before": len(tools), "tools_after": len(chosen), "enum_values_removed": 0,
    }
    compact = copy.deepcopy(chosen)
    for tool in compact:
        function = tool.get("function", {})
        _prune_enums(function.get("parameters", {}), query, stats, enum_limit)
    return compact, stats
