"""Request-scoped canonical HA slots, target checks and deterministic failures."""

import copy
import json
import re
import unicodedata
from functools import lru_cache

from .ha_fuzzy import slot_rankings
from .i18n import normalize_matching, t
from .tool_retrieval import latest_user_text


def _fold(text):
    """Fold orthographic variants for catalogue-only slot matching.

    Args:
        text (str): Text to parse, normalize, match or render.

    Returns:
        str: Case-folded text without accents and transliteration variants.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    text = text.casefold().replace("ae", "a").replace("oe", "o").replace("ue", "u")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


@lru_cache(maxsize=64)
def _catalogue(systems):
    """Cache name, domain and area tuples parsed from system context.

    Args:
        systems (tuple[str, ...]): Static system texts used as an immutable cache key.

    Returns:
        tuple[tuple[str, str, str], ...]: Immutable catalogue entries suitable for bounded caching.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    from .ha_state_routing import _entries

    return tuple(
        (e["name"], e["domain"], e["area"])
        for e in _entries([{"role": "system", "content": s} for s in systems])
    )


def catalogue(messages):
    """Parse the request catalogue with a bounded cache for small contexts.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Returns:
        list[dict[str, str]]: Static catalogue entities with name, domain and area.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
    """Resolve catalogue spellings once; never correct verbs or negations.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        settings (Settings): Validated service settings controlling enabled models and limits.

    Returns:
        ChatRequest: Request carrying canonical slots, catalogue and a target-resolution plan.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
    target_candidates = []
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
                if len(matches) > 1 and fragment.casefold() not in {v.casefold() for v in matches}:
                    for value in sorted(matches):
                        for entity in entities:
                            for kind in ("area", "name"):
                                if entity[kind] == value:
                                    item = dict(
                                        value=value,
                                        score=100.0,
                                        input=fragment,
                                        kind=kind,
                                        area=entity["area"],
                                        domain=entity["domain"],
                                    )
                                    if item not in target_candidates:
                                        target_candidates.append(item)
                if len(matches) == 1:
                    value = next(iter(matches))
                    if fragment.casefold() != value.casefold():
                        text = text[:start] + value + text[end:]
                        repairs.append(
                            {"input": fragment, "resolved": value, "source": "orthography"}
                        )
                        break
        fuzzy = slot_rankings(
            text,
            values,
            threshold=settings.ha_assist_fuzzy_threshold,
            margin=settings.ha_assist_fuzzy_margin,
        )
        if len(fuzzy) == 1 and not fuzzy[0]["ambiguous"]:
            text = fuzzy[0]["text"]
            repairs.extend(fuzzy)
        else:
            for match in fuzzy:
                for candidate in match["candidates"]:
                    for entity in entities:
                        for kind in ("area", "name"):
                            if entity[kind] == candidate["value"]:
                                item = dict(
                                    candidate,
                                    input=match["input"],
                                    kind=kind,
                                    area=entity["area"],
                                    domain=entity["domain"],
                                )
                                if item not in target_candidates:
                                    target_candidates.append(item)
        # Generic domain words are separate from catalogue slots. Short action
        # words (on/off/not) are deliberately excluded from spelling repair.
        from rapidfuzz.distance import Levenshtein

        from .ha_state_routing import _DOMAIN_WORDS, _has_action_verb

        generic = set().union(*_DOMAIN_WORDS.values())
        for word in re.findall(r"\w+", text):
            if _has_action_verb(normalize_matching(word)):
                continue
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
    target_candidates = [c for c in target_candidates if not domain or c["domain"] == domain]
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
        "target_ambiguous": len(areas) > 1 or bool(target_candidates),
        "target_candidates": target_candidates,
        "target_resolution": "llm" if target_candidates else "catalogue",
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
    """Request clarification for an unsafe or unknown explicit area target.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        str | None: Localized clarification, or None when routing can continue.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    plan = getattr(request, "_ha_plan", {})
    if plan.get("target_resolution") == "llm":
        return None
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
    """Translate unsuccessful HA tool results into a deterministic response.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

    Returns:
        str | None: Localized failure response, or None for successful/unsupported results.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
    """Semantic checks supplement JSON Schema; never guess replacement targets.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        name (str): Function, attribute, device or model identifier.
        args (dict[str, Any]): Function arguments for semantic target/value validation.

    Returns:
        bool: Whether generated targets and values agree with the request plan.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
    candidates = plan.get("target_candidates", [])
    if candidates:
        allowed_areas = {c["area"] for c in candidates}
        allowed_names = {
            e["name"]
            for e in entities
            if any(
                (c["kind"] == "area" and e["area"] == c["area"])
                or (c["kind"] == "name" and e["name"] == c["value"] and e["area"] == c["area"])
                for c in candidates
            )
            and (not domain or e["domain"] == domain)
        }
        if args.get("area") not in allowed_areas and args.get("name") not in allowed_names:
            return False
        if args.get("area") and args["area"] not in allowed_areas:
            return False
        if args.get("name") and args["name"] not in allowed_names:
            return False
        if all(c["kind"] == "name" for c in candidates) and not args.get("name"):
            return False
        if (
            args.get("area")
            and args.get("name")
            and not any(e["area"] == args["area"] and e["name"] == args["name"] for e in entities)
        ):
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
