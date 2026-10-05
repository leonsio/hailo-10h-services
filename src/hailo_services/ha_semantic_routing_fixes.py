"""Regression guards for semantic Home Assistant routing and prompt compilation.

MiniLM is intentionally used as a fallback when lexical/entity matching cannot
classify a request. A nearest-neighbour model always has a top result, though,
so unrelated general questions can sit just above a fixed similarity threshold.
This module adds a confidence-margin guard and keeps semantic prompt capabilities
constrained to tools that retrieval actually selected.
"""

from __future__ import annotations

import copy
from functools import wraps

from . import ha_prompt_compiler as _compiler
from . import ha_routing as _routing
from .tool_retrieval import _embedding

_SEMANTIC_RELEVANCE_MARGIN = 0.05
_SEMANTIC_STRONG_RELEVANCE = 0.66


def _semantic_ranking(query_text, tools, encoder, embedding_cache):
    if encoder is None or not query_text or not tools:
        return []
    import numpy as np

    query_vector = _embedding(encoder, query_text, embedding_cache)
    ranked = []
    for tool in tools:
        similarity = float(np.dot(
            query_vector,
            _embedding(encoder, _routing._tool_description(tool), embedding_cache),
        ))
        ranked.append((similarity, tool.get("function", {}).get("name")))
    ranked.sort(key=lambda item: -item[0])
    return ranked


def _capability(query: str, tools, encoder, embedding_cache):
    """Infer semantics only among capabilities represented by selected tools."""
    tool_caps = {
        _compiler._TOOL_CAPABILITY.get(name)
        for name in _compiler._tool_names(tools)
    } - {None}
    action_caps = tool_caps & {"device.turn_on", "device.turn_off"}
    if len(action_caps) == 1:
        return next(iter(action_caps)), "selected_tool", None

    deterministic = _compiler._deterministic_capability(query)
    if deterministic is not None:
        return deterministic, "deterministic", None

    if len(tool_caps) == 1:
        return next(iter(tool_caps)), "selected_tool", None

    if encoder is None or not query or not tool_caps:
        return None, "selected_tools", {"best": None, "second": None}

    import numpy as np

    query_vector = _embedding(encoder, query, embedding_cache)
    ranked = []
    for name in tool_caps:
        description = _compiler._CAPABILITIES.get(name)
        if not description:
            continue
        similarity = float(np.dot(
            query_vector,
            _embedding(encoder, description, embedding_cache),
        ))
        ranked.append((similarity, name))
    ranked.sort(reverse=True)
    if not ranked:
        return None, "selected_tools", {"best": None, "second": None}

    best_score, best_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -1.0
    semantic = {"best": best_score, "second": second_score}
    if best_score >= 0.50 and best_score - second_score >= 0.025:
        return best_name, "minilm", semantic
    return None, "selected_tools", semantic


def _source_domain_enum(tool):
    parameters = tool.get("function", {}).get("parameters", {}) if isinstance(tool, dict) else {}
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    domain = properties.get("domain", {}) if isinstance(properties, dict) else {}
    items = domain.get("items", {}) if isinstance(domain, dict) else {}
    enum = items.get("enum") if isinstance(items, dict) else None
    return list(enum) if isinstance(enum, list) and enum else None


def install():
    """Install semantic-routing regression guards once."""
    if getattr(_routing, "_semantic_routing_fixes_installed", False):
        return

    original_assess = _routing.assess_ha_relevance
    original_compact_tool = _compiler._compact_tool

    @wraps(original_assess)
    def assess_ha_relevance(messages, tools, *, encoder=None, embedding_cache=None,
                            semantic_threshold=_routing._SEMANTIC_RELEVANCE_THRESHOLD):
        result = original_assess(
            messages,
            tools,
            encoder=encoder,
            embedding_cache=embedding_cache,
            semantic_threshold=semantic_threshold,
        )
        if result.get("reason") != "semantic_tool":
            return result

        ranked = _semantic_ranking(result.get("query_text", ""), tools or [], encoder, embedding_cache)
        if not ranked:
            return result
        best_score, best_name = ranked[0]
        second_score, second_name = ranked[1] if len(ranked) > 1 else (-1.0, None)
        margin = best_score - second_score
        result.update({
            "semantic_tool_max": best_score,
            "semantic_tool_name": best_name,
            "semantic_tool_second_max": second_score,
            "semantic_tool_second_name": second_name,
            "semantic_tool_margin": margin,
            "semantic_margin_threshold": _SEMANTIC_RELEVANCE_MARGIN,
            "semantic_strong_threshold": _SEMANTIC_STRONG_RELEVANCE,
        })
        if margin < _SEMANTIC_RELEVANCE_MARGIN and best_score < max(
            _SEMANTIC_STRONG_RELEVANCE, semantic_threshold + _SEMANTIC_RELEVANCE_MARGIN
        ):
            result["relevant"] = False
            result["reason"] = "semantic_ambiguous"
        return result

    @wraps(original_compact_tool)
    def compact_tool(tool, capability, domain, area, entities):
        compact = original_compact_tool(tool, capability, domain, area, entities)
        allowed_domains = _source_domain_enum(tool)
        if allowed_domains and domain is not None and domain not in allowed_domains:
            source = copy.deepcopy(
                tool.get("function", {}).get("parameters", {}).get("properties", {}).get("domain", {})
            )
            if isinstance(source, dict):
                source.pop("description", None)
            properties = compact.get("function", {}).get("parameters", {}).get("properties", {})
            if isinstance(properties, dict) and "domain" in properties:
                properties["domain"] = source
        return compact

    _routing.assess_ha_relevance = assess_ha_relevance
    _compiler._capability = _capability
    _compiler._compact_tool = compact_tool
    _routing._semantic_routing_fixes_installed = True
