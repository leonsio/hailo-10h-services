"""Optional exposed HA metadata; canonical aliases stay request-scoped."""

import copy
import re

from hailo_services.shared.i18n import normalize_matching
from hailo_services.shared.tool_retrieval import _static_context_parts


def prepare_catalogue(request):
    """Materialize a client's exposed structured catalogue for existing HA stages.

    Args:
        request: HA-boundary request; physical models never call this function.

    Returns:
        ChatRequest: Copied messages and rich private catalogue, or unchanged legacy request.
    """
    context = request.ha_context
    if context is None:
        return request
    entities = [entity.model_dump() for entity in context.entities]
    entries = [
        f"- names: {e['name']}\n  domain: {e['domain']}\n  areas: {e['area']}\n" for e in entities
    ]
    messages = copy.deepcopy(request.messages)
    inserted = False
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        parts = _static_context_parts(message["content"])
        if parts is not None:
            message["content"] = parts[0] + ("".join(entries) if not inserted else "") + parts[2]
            inserted = True
    if not inserted:
        messages.insert(
            0, {"role": "system", "content": "Home Assistant\nStatic Context:\n" + "".join(entries)}
        )
    prepared = request.model_copy(update={"messages": messages})
    object.__setattr__(prepared, "_ha_catalogue", entities)
    prepared._metrics["ha_catalogue"] = {
        "source": "structured",
        "version": context.version,
        "entities": len(entities),
    }
    return prepared


def resolve_aliases(text, entities, domain=None):
    """Resolve only configured aliases with unique target evidence.

    Args:
        text: Current user utterance.
        entities: Exposed catalogue with optional name/area aliases.
        domain: Explicit domain constraint; semantic similarity cannot override it.

    Returns:
        tuple: Canonical text, alias repairs and tied target candidates.
    """
    mappings = {}
    normalized = normalize_matching(text)
    exact_areas = {
        e["area"]
        for e in entities
        if e["area"] and f" {normalize_matching(e['area'])} " in f" {normalized} "
    }
    for entity in entities:
        if domain and entity["domain"] != domain:
            continue
        for kind, key in (("name", "aliases"), ("area", "area_aliases")):
            if kind == "name" and exact_areas and entity["area"] not in exact_areas:
                continue
            for alias in entity.get(key, []):
                item = {
                    "value": entity[kind],
                    "kind": kind,
                    "area": entity["area"],
                    "domain": entity["domain"],
                    "score": 100.0,
                    "input": alias,
                }
                if item not in mappings.setdefault(alias, []):
                    mappings[alias].append(item)
    repairs, candidates = [], []
    canonical_names = {e["name"].casefold() for e in entities} | {
        e["area"].casefold() for e in entities
    }
    for alias in sorted(mappings, key=lambda a: (-len(a), a)):
        if alias.casefold() in canonical_names:
            continue
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        if not re.search(pattern, text, re.I):
            continue
        items = mappings[alias]
        targets = {(c["kind"], c["value"], c["area"] if c["kind"] == "name" else "") for c in items}
        if len(targets) == 1:
            text = re.sub(pattern, items[0]["value"], text, flags=re.I)
            repairs.append({"input": alias, "resolved": items[0]["value"], "source": "ha_alias"})
        else:
            candidates.extend(items)
    return text, repairs, candidates
