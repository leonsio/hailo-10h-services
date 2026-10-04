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


def _embedding(encoder, text: str, cache=None):
    if cache is not None and text in cache:
        return cache[text]
    vector = encoder.embed(text)
    if cache is not None:
        cache[text] = vector
    return vector


def _static_context_parts(content: str):
    """Split Home Assistant's generated Static Context into prefix, entries and suffix."""
    marker = "Static Context:"
    start = content.find(marker)
    if start < 0:
        return None
    entries_start = content.find("- names:", start)
    if entries_start < 0:
        return None
    # hass_local_openai_llm appends control rules after a blank line. Keep those
    # rules verbatim; only the generated entity catalogue is replaceable.
    suffix_start = content.find("\n\nWhen controlling Home Assistant", entries_start)
    if suffix_start < 0:
        suffix_start = len(content)
    catalogue = content[entries_start:suffix_start].strip()
    entries = [
        block.strip()
        for block in re.split(r"(?m)(?=^- names:\s*)", catalogue)
        if block.strip().startswith("- names:")
    ]
    if not entries:
        return None
    return content[:start], entries, content[suffix_start:]


def compact_static_context(
    messages,
    *,
    encoder=None,
    max_entities: int = 8,
    semantic_candidates: int = 24,
    embedding_cache=None,
):
    """Replace HA's full static entity catalogue with request-relevant entries.

    The full inbound request remains available to MiniLM. Only the copy that is
    forwarded to Gemma is compacted.
    """
    query_text = latest_user_text(messages).strip()
    query = _tokens(query_text)
    updated = copy.deepcopy(messages)
    totals = {
        "entities_before": 0,
        "entities_after": 0,
        "characters_removed": 0,
        "system_prompts_compacted": 0,
    }
    if not query_text:
        return updated, totals

    for message in updated:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        parts = _static_context_parts(message["content"])
        if parts is None:
            continue
        prefix, entries, suffix = parts
        totals["entities_before"] += len(entries)

        lexical = [(_score(query, entry), index, entry) for index, entry in enumerate(entries)]
        lexical_hits = [item for item in lexical if item[0] > 0]

        if lexical_hits:
            lexical_hits.sort(key=lambda item: (-item[0], item[1]))
            best = lexical_hits[0][0]
            candidates = [
                item for item in lexical_hits if item[0] >= max(1, best - 2)
            ][:semantic_candidates]
        else:
            candidates = lexical

        if encoder is not None and candidates:
            import numpy as np

            query_vector = _embedding(encoder, query_text, embedding_cache)
            ranked = []
            for lexical_score, index, entry in candidates:
                similarity = float(np.dot(
                    query_vector, _embedding(encoder, entry, embedding_cache)
                ))
                ranked.append((
                    similarity + min(lexical_score, 12) * 0.03,
                    lexical_score,
                    index,
                    entry,
                ))
            ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
            selected = [item[3] for item in ranked[:max_entities]]
        else:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = [item[2] for item in candidates[:max_entities]]

        # If there was no lexical signal, semantic retrieval still returns a
        # small candidate set instead of forwarding the entire house inventory.
        if not selected:
            selected = entries[:max_entities]

        compact_catalogue = "\n".join(selected)
        replacement = (
            "Static Context: Relevant entities for the current user request:\n"
            + compact_catalogue
        )
        old_content = message["content"]
        message["content"] = prefix + replacement + suffix
        totals["entities_after"] += len(selected)
        totals["characters_removed"] += max(0, len(old_content) - len(message["content"]))
        totals["system_prompts_compacted"] += 1

    return updated, totals


def retrieve_tools(
    messages,
    tools,
    *,
    max_tools: int = 4,
    enum_limit: int = 8,
    encoder=None,
    embedding_cache=None,
):
    """Return a compact copy of tools relevant to the latest user turn.

    Lexical matches are used as a high-confidence prefilter. MiniLM ranks only
    those candidates when possible; when lexical matching finds nothing, MiniLM
    provides a small semantic fallback instead of forwarding the whole tool set.
    """
    if not tools:
        return tools, {"tools_before": 0, "tools_after": 0, "enum_values_removed": 0}

    query_text = latest_user_text(messages).strip()
    query = _tokens(query_text)
    ranked = [(_score(query, tool), index, tool) for index, tool in enumerate(tools)]
    hits = [item for item in ranked if item[0] > 0]

    if hits:
        hits.sort(key=lambda item: (-item[0], item[1]))
        best = hits[0][0]
        candidates = [item for item in hits if item[0] >= max(1, best - 2)]
    else:
        candidates = ranked

    if encoder is not None and query_text and candidates:
        import numpy as np

        vector = _embedding(encoder, query_text, embedding_cache)
        semantic = []
        for lexical_score, index, tool in candidates:
            fn = tool.get("function", {})
            description = " ".join((
                fn.get("name", ""),
                fn.get("description", ""),
                _text(fn.get("parameters", {}))[:1400],
            ))
            similarity = float(np.dot(
                vector, _embedding(encoder, description, embedding_cache)
            ))
            semantic.append((
                similarity + min(lexical_score, 12) * 0.03,
                lexical_score,
                index,
                tool,
            ))
        semantic.sort(key=lambda item: (-item[0], -item[1], item[2]))
        chosen = [item[3] for item in semantic[:max_tools]]
    elif hits:
        chosen = [item[2] for item in candidates[:max_tools]]
    else:
        # Without an encoder there is no safe semantic basis for discarding
        # unmatched tools; preserve legacy behavior in that fallback path.
        chosen = list(tools)

    stats = {
        "tools_before": len(tools),
        "tools_after": len(chosen),
        "enum_values_removed": 0,
    }
    compact = copy.deepcopy(chosen)
    for tool in compact:
        function = tool.get("function", {})
        _prune_enums(function.get("parameters", {}), query, stats, enum_limit)
    return compact, stats
