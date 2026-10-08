"""Retrieve relevant Home Assistant tool schemas before passing them to Gemma.

The incoming HA request may contain a large number of nearly identical tools or
large enums of entity IDs. Keep only tools and enum values that match the latest
user turn. The optional encoder runs MiniLM on the Hailo accelerator.
"""

from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from hailo_services.shared.i18n import lexicon, normalize_matching

_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)
_STOP_WORDS = lexicon("tool_retrieval._STOP_WORDS")
_SYNONYMS = lexicon("tool_retrieval._SYNONYMS")


def _text(value) -> str:
    """Flatten schema and catalogue values into retrieval text.

    Args:
        value (Any): Input value inspected or normalized by this helper.

    Returns:
        str: Text representation of supported nested values.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
    """Normalize and split text into retrieval or weather-matching terms.

    Args:
        value (Any): Input value inspected or normalized by this helper.

    Returns:
        set[str]: Significant normalized terms.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", _text(value)).casefold().replace("ß", "ss")
    raw = normalize_matching(raw.replace("_", " ").replace(".", " ").replace("-", " "))
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
                result.add(word[: -len(ending)])
                break
        result.update(_SYNONYMS.get(word, ()))
    return result


def latest_user_text(messages) -> str:
    """Read only text from the latest user message (ignore image payloads).

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        str: Result as described by the operation.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""
    return ""


def _score(query: set[str], candidate) -> int:
    """Score normalized token overlap and useful catalogue substrings.

    Args:
        query (set[str]): Normalized significant user terms.
        candidate (Any): Catalogue candidate whose lexical relevance is scored.

    Returns:
        int: Non-negative lexical relevance score.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if not query:
        return 0
    words = _tokens(candidate)
    exact = len(query & words)
    # Entity IDs often contain concatenated area/name fragments.
    joined = " ".join(words)
    substring = sum(1 for word in query if len(word) >= 4 and word in joined)
    return exact * 3 + substring


def _tool_score(query: set[str], tool) -> int:
    """Prefer explicit action/function-name matches over incidental prose matches.

    Home Assistant tool descriptions can be verbose and mention unrelated
    actions as examples. Weighting the function name keeps e.g. HassTurnOff
    ahead of GetLiveContext for an explicit "schalte ... aus" request.

    Args:
        query (set[str]): Normalized significant user terms.
        tool (dict[str, Any]): Client-provided function schema.

    Returns:
        int: Weighted lexical score emphasizing explicit function/action matches.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    function = tool.get("function", {}) if isinstance(tool, dict) else {}
    name_score = _score(query, function.get("name", ""))
    description_score = _score(query, function.get("description", ""))
    parameter_score = _score(query, function.get("parameters", {}))
    return name_score * 4 + description_score + min(parameter_score, 4)


# Router hints are independent of descriptions rendered for the language model.
_ROUTING_KEYWORDS = {
    "HassTurnOn": "turn on switch on einschalten anschalten anmachen включи",
    "HassTurnOff": "turn off switch off ausschalten abschalten ausmachen выключи",
    "HassLightSet": "brightness dim color helligkeit dimmen farbe яркость цвет",
    "HassSetPosition": "position blinds cover rollladen jalousie жалюзи",
    "HassClimateSetTemperature": "set temperature heizen temperatur einstellen температура",
    "GetLiveContext": "state status zustand humidity feuchtigkeit состояние влажность",
}


@lru_cache(maxsize=64)
def _tool_index(language, serialized):
    """Cache immutable lexical schema features, not live state or selected actions.

    Args:
        language: Locale partition; identical schemas in different languages stay separate.
        serialized: Complete canonical schema JSON; any tool/enum change invalidates the key.

    Returns:
        tuple: Immutable tool-field tokens and corpus frequencies; bounded by entry/key limits.
    """
    tools = json.loads(serialized)
    fields = []
    for tool in tools:
        fn = tool.get("function", {})
        fields.append(
            (
                _tokens(fn.get("name", "")),
                _tokens(fn.get("description", "")),
                _tokens(_ROUTING_KEYWORDS.get(fn.get("name", "").rsplit("__", 1)[-1], "")),
                _tokens(fn.get("parameters", {})),
            )
        )
    frequency = Counter(word for parts in fields for word in set().union(*parts))

    return tuple(tuple(frozenset(words) for words in parts) for parts in fields), tuple(
        frequency.items()
    )


def _corpus_scores(query, tools):
    """Score tools with field weights and inverse corpus frequency.

    Args:
        query (set[str]): Normalized significant user terms.
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.

    Returns:
        list[float]: Relevance scores in the original tool order.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    from hailo_services.shared.i18n import current_language

    serialized = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    build = _tool_index if len(serialized) <= 65536 else _tool_index.__wrapped__
    fields, frequency_items = build(current_language(), serialized)
    frequency = dict(frequency_items)

    def score(parts):
        """Compute weighted rare-term overlap for one tool.

        Args:
            parts (Any): Schema fields or normalized text parts to score.

        Returns:
            float: Corpus-weighted relevance score.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        result = 0.0
        for words, weight in zip(parts, (12, 4, 8, 1)):
            for word in query & words:
                # Shared boilerplate carries little weight; rare exact terms dominate.
                idf = math.log((len(tools) + 1) / (frequency[word] + 1)) + 0.25
                result += weight * idf + (6 * idf if weight >= 8 else 0)
        return result

    return [score(parts) for parts in fields]


def _prune_enums(
    node,
    query: set[str],
    stats: dict,
    limit: int = 12,
    *,
    trace=None,
    path: str = "$",
):
    """Restrict matching schema enums and update removal diagnostics in place.

    Args:
        node (Any): Nested schema node to inspect or modify.
        query (set[str]): Normalized significant user terms.
        stats (dict): Mutable retrieval/removal counters.
        limit (int): Upper bound for bytes, input tokens or retained enum values.
        trace (dict | list | None): Optional mutable diagnostic collector.
        path (str): Filesystem destination or diagnostic schema path.

    Returns:
        None: Mutates schema enum values, counters and optional trace.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
                if trace is not None:
                    trace.append(
                        {
                            "path": path,
                            "before": enum,
                            "after": node["enum"],
                            "scores": [
                                {"value": item[2], "lexical_score": item[0]}
                                for item in sorted(ranked, key=lambda item: (-item[0], item[1]))
                            ],
                        }
                    )
        for key, value in list(node.items()):
            _prune_enums(value, query, stats, limit, trace=trace, path=f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _prune_enums(value, query, stats, limit, trace=trace, path=f"{path}[{index}]")


def _embedding(encoder, text: str, cache=None):
    """Reuse bounded cached embeddings for short static retrieval text.

    Args:
        encoder (MiniLM | None): Optional resident semantic encoder; None uses lexical matching.
        text (str): Text to parse, normalize, match or render.
        cache (dict[str, np.ndarray] | None): Optional bounded embedding cache.

    Returns:
        np.ndarray: Encoder vector, cached when eligible.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if len(text) > 8192:
        return encoder.embed(text)
    if cache is not None and text in cache:
        return cache[text]
    vector = encoder.embed(text)
    if cache is not None:
        # Bound resident memory; never cache live HA state or tool results.
        if len(cache) >= 512 and text not in cache:
            cache.pop(next(iter(cache)))
        cache[text] = vector
    return vector


def _static_context_parts(content: str):
    """Split Home Assistant's generated Static Context into prefix, entries and suffix.

    Args:
        content (str): File contents or generated static-context text.

    Returns:
        tuple[str, list[str], str] | None: System prefix, entity blocks and control-rule suffix, or None.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
    trace=None,
):
    """Replace HA's full static entity catalogue with request-relevant entries.

    The full inbound request remains available to MiniLM. Only the copy that is
    forwarded to Gemma is compacted.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.
        encoder (MiniLM | None): Optional resident semantic encoder; None uses lexical matching.
        max_entities (int): Maximum static entities retained in the compact context.
        semantic_candidates (int): Maximum candidates retained for semantic retrieval.
        embedding_cache (dict[str, np.ndarray] | None): Bounded cache for static retrieval embeddings.
        trace (dict | list | None): Optional mutable diagnostic collector.

    Returns:
        tuple[list[dict], dict]: Compact messages and entity/character removal statistics.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    query_text = latest_user_text(messages).strip()
    query = _tokens(query_text)
    updated = copy.deepcopy(messages)
    if trace is not None:
        trace.clear()
        trace.update(
            {
                "stage": "entity_retrieval",
                "query_text": query_text,
                "query_tokens": sorted(query),
                "max_entities": max_entities,
                "semantic_candidates": semantic_candidates,
                "system_prompts": [],
            }
        )
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
        prompt_trace = None
        if trace is not None:
            prompt_trace = {
                "entities_before": len(entries),
                "all_entities": [
                    {
                        "index": index,
                        "lexical_score": score,
                        "entity": entry,
                    }
                    for score, index, entry in lexical
                ],
                "lexical_hits": [],
                "semantic_ranking": [],
                "selected_entities": [],
            }
            trace["system_prompts"].append(prompt_trace)

        if lexical_hits:
            lexical_hits.sort(key=lambda item: (-item[0], item[1]))
            best = lexical_hits[0][0]
            candidates = [item for item in lexical_hits if item[0] >= max(1, best - 4)][
                :semantic_candidates
            ]
            if prompt_trace is not None:
                prompt_trace["lexical_hits"] = [
                    {
                        "index": index,
                        "lexical_score": score,
                        "entity": entry,
                    }
                    for score, index, entry in lexical_hits
                ]
        else:
            candidates = lexical

        if encoder is not None and candidates:
            import numpy as np

            query_vector = _embedding(encoder, query_text, embedding_cache)
            ranked = []
            for lexical_score, index, entry in candidates:
                similarity = float(
                    np.dot(query_vector, _embedding(encoder, entry, embedding_cache))
                )
                ranked.append(
                    (
                        similarity + min(lexical_score, 12) * 0.03,
                        lexical_score,
                        index,
                        entry,
                    )
                )
            ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
            selected = [item[3] for item in ranked[:max_entities]]
            if prompt_trace is not None:
                prompt_trace["semantic_ranking"] = [
                    {
                        "combined_score": score,
                        "lexical_score": lexical_score,
                        "index": index,
                        "entity": entry,
                    }
                    for score, lexical_score, index, entry in ranked
                ]
        else:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = [item[2] for item in candidates[:max_entities]]

        # If there was no lexical signal, semantic retrieval still returns a
        # small candidate set instead of forwarding the entire house inventory.
        if not selected:
            selected = entries[:max_entities]

        if prompt_trace is not None:
            prompt_trace["selected_entities"] = list(selected)
        compact_catalogue = "\n".join(selected)
        replacement = (
            "Static Context: Relevant entities for the current user request:\n" + compact_catalogue
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
    trace=None,
    semantic_candidates: int = 12,
):
    """Return a compact copy of tools relevant to the latest user turn.

    Lexical matches are used as a high-confidence prefilter. MiniLM ranks only
    those candidates when possible; when lexical matching finds nothing, MiniLM
    provides a small semantic fallback instead of forwarding the whole tool set.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.
        tools (list[dict[str, Any]] | None): Client-provided OpenAI function schemas.
        max_tools (int): Maximum retrieved tool schemas.
        enum_limit (int): Maximum retained values for a matching schema enum.
        encoder (MiniLM | None): Optional resident semantic encoder; None uses lexical matching.
        embedding_cache (dict[str, np.ndarray] | None): Bounded cache for static retrieval embeddings.
        trace (dict | list | None): Optional mutable diagnostic collector.
        semantic_candidates (int): Maximum candidates retained for semantic retrieval.

    Returns:
        tuple[list[dict], dict]: Selected copied schemas and tool/enum removal statistics.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if not tools:
        return tools, {"tools_before": 0, "tools_after": 0, "enum_values_removed": 0}

    query_text = latest_user_text(messages).strip()
    query = _tokens(query_text)
    if trace is not None:
        trace.clear()
        trace.update(
            {
                "stage": "tool_retrieval",
                "query_text": query_text,
                "query_tokens": sorted(query),
                "max_tools": max_tools,
                "enum_limit": enum_limit,
                "all_tools": [],
                "candidates": [],
                "semantic_ranking": [],
                "selected_tool_names": [],
                "selected_tools": [],
                "enum_pruning": [],
            }
        )
    before_index = _tool_index.cache_info()
    scores = _corpus_scores(query, tools)
    index_hit = _tool_index.cache_info().hits > before_index.hits
    ranked = [(scores[index], tool["function"]["name"], tool) for index, tool in enumerate(tools)]
    kept_names = {
        call.get("function", {}).get("name")
        for message in messages
        for call in message.get("tool_calls") or []
    }
    if trace is not None:
        trace["all_tools"] = [
            {
                "index": index,
                "name": tool.get("function", {}).get("name"),
                "lexical_score": score,
            }
            for score, index, tool in ranked
        ]
    hits = [item for item in ranked if item[0] > 0]

    if hits:
        hits.sort(key=lambda item: (-item[0], item[1]))
        best = hits[0][0]
        candidates = [item for item in hits if item[0] >= max(1, best * 0.5)][:semantic_candidates]
    else:
        candidates = sorted(ranked, key=lambda item: item[1])[:semantic_candidates]

    if trace is not None:
        trace["candidates"] = [
            {
                "index": index,
                "name": tool.get("function", {}).get("name"),
                "lexical_score": score,
            }
            for score, index, tool in candidates
        ]

    if encoder is not None and query_text and candidates:
        import numpy as np

        vector = _embedding(encoder, query_text, embedding_cache)
        semantic = []
        for lexical_score, index, tool in candidates:
            fn = tool.get("function", {})
            description = " ".join(
                (
                    fn.get("name", ""),
                    fn.get("description", ""),
                    _text(fn.get("parameters", {}))[:1400],
                )
            )
            similarity = float(np.dot(vector, _embedding(encoder, description, embedding_cache)))
            semantic.append(
                (
                    similarity + min(lexical_score, 12) * 0.03,
                    lexical_score,
                    index,
                    tool,
                )
            )
        semantic.sort(key=lambda item: (-item[0], -item[1], item[2]))
        chosen = [item[3] for item in semantic[:max_tools]]
        if trace is not None:
            trace["semantic_ranking"] = [
                {
                    "combined_score": score,
                    "lexical_score": lexical_score,
                    "index": index,
                    "name": tool.get("function", {}).get("name"),
                }
                for score, lexical_score, index, tool in semantic
            ]
    elif hits:
        chosen = [item[2] for item in candidates[:max_tools]]
    else:
        # Without an encoder there is no safe semantic basis for discarding
        # unmatched tools; preserve legacy behavior in that fallback path.
        chosen = list(tools)

    for tool in sorted(tools, key=lambda item: item["function"]["name"]):
        if tool["function"]["name"] in kept_names and tool not in chosen:
            chosen.append(tool)

    stats = {
        "tool_index_cache_hit": index_hit,
        "tools_before": len(tools),
        "tools_after": len(chosen),
        "enum_values_removed": 0,
    }
    compact = copy.deepcopy(chosen)
    enum_trace = trace["enum_pruning"] if trace is not None else None
    for tool in compact:
        function = tool.get("function", {})
        if function.get("name") in kept_names:
            continue
        _prune_enums(
            function.get("parameters", {}),
            query,
            stats,
            enum_limit,
            trace=enum_trace,
            path=f"$.tools.{function.get('name', '<unnamed>')}.parameters",
        )
    if trace is not None:
        trace["selected_tool_names"] = [tool.get("function", {}).get("name") for tool in compact]
        trace["selected_tools"] = copy.deepcopy(compact)
        trace["stats"] = dict(stats)
    return compact, stats
