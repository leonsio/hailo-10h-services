"""Request-scoped canonical HA slots, target checks and deterministic failures."""

import copy
import json
import re
import unicodedata
from functools import lru_cache

from .ha_fuzzy import slot_repairs
from .i18n import normalize_matching, t
from .tool_retrieval import latest_user_text


def _fold(text):
    text = text.casefold().replace("ae", "a").replace("oe", "o").replace("ue", "u")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


@lru_cache(maxsize=64)
def _catalogue(systems):
    from .ha_state_routing import _entries

    return tuple(
        (e["name"], e["domain"], e["area"])
        for e in _entries([{"role": "system", "content": s} for s in systems])
    )


def catalogue(messages):
    systems = tuple(
        m["content"]
        for m in messages
        if m.get("role") == "system"
        and isinstance(m.get("content"), str)
        and "Static Context:" in m["content"]
    )
    parsed = (
        _catalogue(systems) if sum(map(len, systems)) <= 65536 else _catalogue.__wrapped__(systems)
    )
    return [dict(zip(("name", "domain", "area"), e)) for e in parsed]


def canonical_request(request, settings):
    """Resolve catalogue spellings once; never correct verbs or negations."""
    user_indices = [i for i, m in enumerate(request.messages) if m.get("role") == "user"]
    if not user_indices:
        return request
    user_index = user_indices[-1]
    text = latest_user_text(request.messages)
    if len(text) > 512:
        return request
    before = _catalogue.cache_info()
    entities = catalogue(request.messages)
    after = _catalogue.cache_info()
    values = sorted({e[k] for e in entities for k in ("name", "area") if e[k]})
    original = text
    repairs = []
    if settings.ha_assist_fuzzy_enabled:
        # Orthographic aliases are accepted only when they map uniquely.
        aliases = {}
        for value in values:
            aliases.setdefault(_fold(value), set()).add(value)
        for count in sorted({len(v.split()) for v in values}, reverse=True):
            words = list(re.finditer(r"\w+", text))
            for index in range(len(words) - count + 1):
                start, end = words[index].start(), words[index + count - 1].end()
                fragment = text[start:end]
                matches = aliases.get(_fold(fragment), set())
                if len(matches) == 1:
                    value = next(iter(matches))
                    if fragment.casefold() != value.casefold():
                        text = text[:start] + value + text[end:]
                        repairs.append(
                            {"input": fragment, "resolved": value, "source": "orthography"}
                        )
                        break
        fuzzy = slot_repairs(
            text,
            values,
            threshold=settings.ha_assist_fuzzy_threshold,
            margin=settings.ha_assist_fuzzy_margin,
        )
        if fuzzy:
            text = fuzzy[0]["text"]
            repairs.extend(fuzzy)
        # Generic domain words are separate from catalogue slots. Short action
        # words (on/off/not) are deliberately excluded from spelling repair.
        from rapidfuzz.distance import Levenshtein

        generic = {"licht", "lampe", "beleuchtung", "light", "lights", "свет"}
        for word in re.findall(r"\w+", text):
            candidates = [
                v
                for v in generic
                if len(v) >= 5 and len(word) >= 5 and Levenshtein.distance(word.casefold(), v) == 1
            ]
            if word.casefold() not in generic and len(candidates) == 1:
                text = re.sub(r"\b" + re.escape(word) + r"\b", candidates[0], text)
                repairs.append({"input": word, "resolved": candidates[0], "source": "domain_word"})
    normalized = normalize_matching(text)
    from .ha_state_routing import _query_domain

    domain = _query_domain(normalized)
    areas = sorted(
        {
            e["area"]
            for e in entities
            if e["area"] and (" " + normalize_matching(e["area"]) + " ") in (" " + normalized + " ")
        }
    )
    numeric = re.findall(r"(?<![\w.,-])(\d{1,3})\s*(?:%|prozent|percent|процент\w*)", text, re.I)
    from .ha_state_routing import _has_action_verb

    if not numeric and domain in ("light", "cover") and _has_action_verb(normalized):
        numeric = re.findall(r"\b(?:auf|to|на)\s+(\d{1,3})\s*[.!?]?$", text, re.I)
    plan = {
        "original": original,
        "canonical": text,
        "repairs": repairs,
        "domain": domain,
        "area": areas[0] if len(areas) == 1 else None,
        "value": int(numeric[0]) if len(numeric) == 1 else None,
        "unit": "percent" if len(numeric) == 1 else None,
        "target_ambiguous": len(areas) > 1,
        "catalogue_cache_hit": after.hits > before.hits,
    }
    from .ha_prompt_compiler import _deterministic_capability

    plan["action"] = (
        "light.brightness"
        if numeric and domain == "light"
        else "cover.position"
        if numeric and domain == "cover"
        else _deterministic_capability(text)
    )
    if text != original:
        messages = copy.deepcopy(request.messages)
        content = messages[user_index]["content"]
        if isinstance(content, str):
            messages[user_index]["content"] = text
        else:
            # Only plain text requests enter this path; preserve all other parts.
            text_parts = [p for p in content if p.get("type") == "text"]
            if len(text_parts) == 1:
                text_parts[0]["text"] = text
            else:
                return request
        request = request.model_copy(update={"messages": messages})
    object.__setattr__(request, "_ha_plan", plan)
    object.__setattr__(request, "_ha_catalogue", entities)
    request._metrics["ha_plan"] = plan
    return request


def target_clarification(request):
    plan = getattr(request, "_ha_plan", {})
    text = normalize_matching(latest_user_text(request.messages))
    # An unresolved area must not become a guessed individual-device action.
    from .ha_state_routing import _has_action_verb

    if request.messages[-1].get("role") == "user" and _has_action_verb(text):
        if plan.get("target_ambiguous") or (
            not plan.get("area") and re.search(r"\b(im|in der|in dem|in the|in|в)\s+\w+", text)
        ):
            return t("ha_plan.clarify_target")
    return None


def tool_failure(request):
    if request.messages[-1].get("role") != "tool":
        return None
    results = []
    for message in reversed(request.messages):
        if message.get("role") != "tool":
            break
        try:
            result = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            return None
        results.append(result)
    if not results or not all(isinstance(r, dict) for r in results):
        return None
    if any(
        r.get("error")
        or r.get("success") is False
        or (isinstance(r.get("data"), dict) and r["data"].get("failed"))
        for r in results
    ):
        return t("ha_plan.tool_failed")
    return None


def validate_action(request, name, args):
    """Semantic checks supplement JSON Schema; never guess replacement targets."""
    if not getattr(request, "_ha_assist", False) or name == "homeassistant__GetLiveContext":
        return True
    entities = getattr(request, "_ha_catalogue", None) or catalogue(request.messages)
    if not entities:
        return True  # Clients without a catalogue are still checked against their schema.
    plan = getattr(request, "_ha_plan", {})
    area = plan.get("area")
    domain = plan.get("domain")
    given_domains = args.get("domain")
    if domain and given_domains is not None:
        if (given_domains if isinstance(given_domains, list) else [given_domains]) != [domain]:
            return False
    if area and args.get("area") not in (None, area):
        return False
    members = [
        e
        for e in entities
        if (not area or e["area"] == area) and (not domain or e["domain"] == domain)
    ]
    if args.get("name") and not any(e["name"] == args["name"] for e in members):
        return False
    if area and not args.get("area") and not args.get("name"):
        return False
    if plan.get("unit") == "percent" and plan.get("domain") == "light":
        if (
            name != "light__HassLightSet"
            or args.get("brightness") != plan["value"]
            or "color" in args
        ):
            return False
    if plan.get("unit") == "percent" and domain == "cover":
        if name != "intent__HassSetPosition" or args.get("position") != plan["value"]:
            return False
    return True
