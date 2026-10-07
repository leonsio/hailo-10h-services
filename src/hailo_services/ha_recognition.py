"""Bounded official-template recovery with protected actions, numbers and qualifiers."""

import re
from functools import lru_cache
from itertools import islice

from hassil.errors import HassilError
from hassil.sample import sample_sentence
from rapidfuzz.distance import DamerauLevenshtein
from rapidfuzz.fuzz import ratio

from .i18n import lexicon, normalize_matching


def similarity(text, candidate):
    """Score character edits and adjacent transpositions separately.

    Args:
        text: Incoming catalogue span or sentence.
        candidate: Known spelling or official sentence.

    Returns:
        tuple: Combined score and independent character scores, all from 0 to 100.
    """
    left, right = text.casefold(), candidate.casefold()
    character = ratio(left, right)
    transposition = 100 * DamerauLevenshtein.normalized_similarity(left, right)
    return max(character, transposition), {"characters": character, "transpositions": transposition}


def action_signature(text):
    """Require explicit action evidence without fuzzy rewriting on/off words.

    Args:
        text: User command or reconstructed official sentence.

    Returns:
        str | None: Explicit capability, or None for absent/conflicting polarity.
    """
    from .ha_prompt_compiler import _deterministic_capability
    from .ha_state_routing import _STATE_ALIASES, _has_action_verb

    capability = _deterministic_capability(text)
    if capability:
        return capability
    normalized = normalize_matching(text)
    states = {
        state
        for word, state in _STATE_ALIASES.items()
        if f" {normalize_matching(word)} " in f" {normalized} "
    }
    if _has_action_verb(normalized) and len(states) == 1:
        state = next(iter(states))
        if state in {"on", "off"}:
            return "device.turn_" + state
    return None


def recovery_guard(text):
    """Reject complex or uncertain commands before fuzzy sentence reconstruction.

    Args:
        text: Original user text; action and number evidence must remain explicit.

    Returns:
        str | None: Rejection reason, or None for a simple request.
    """
    from .ha_state_routing import _has_action_verb

    normalized = normalize_matching(text)
    if len(text) > 512:
        return "query_too_long"
    if re.search(lexicon("ha_pipeline.pattern.71.17"), normalized):
        return "protected_qualifier"
    if not _has_action_verb(normalized):
        return "no_explicit_action"
    if len(re.findall(r"\d+(?:[.,]\d+)?", text)) > 1:
        return "multiple_values"
    return None


@lru_cache(maxsize=32)
def _templates(language, supported):
    """Build a bounded index of grammar skeletons without expanding catalogue slots.

    Args:
        language: Base HA language code.
        supported: Tool-backed intent names allowed by this request.

    Returns:
        tuple: Official intent names and sentence skeletons; no device permutations.
    """
    from .ha_intents import _grammar

    grammar = _grammar(language, supported)
    if grammar is None:
        return ()
    templates = []
    quota = min(512, max(1, 4096 // max(1, len(grammar.intents))))
    for intent in grammar.intents.values():
        sentence_count = sum(len(data.sentences) for data in intent.data)
        per_sentence = max(1, min(16, quota // max(1, sentence_count)))
        intent_templates = []
        for data in intent.data:
            rules = {**grammar.expansion_rules, **data.expansion_rules}
            lists = {**grammar.slot_lists, **data.slot_lists}
            for sentence in data.sentences:
                try:
                    samples = islice(
                        sample_sentence(
                            sentence,
                            slot_lists=lists,
                            expansion_rules=rules,
                            language=language,
                            expand_lists=False,
                            expand_ranges=False,
                        ),
                        per_sentence,
                    )
                    intent_templates.extend((intent.name, s) for s in samples)
                except (HassilError, ValueError):
                    continue
        templates.extend(dict.fromkeys(intent_templates[:quota]))
    return tuple(templates)


def grammatical_words(grammar):
    """Read short article forms from official rules without maintaining spelling lists.

    Args:
        grammar: Restricted official language grammar.

    Returns:
        set: Exact grammatical article forms eligible for structure-only scoring.
    """
    words = set()
    for name, sentence in grammar.expansion_rules.items():
        if "artikel" not in name and "article" not in name:
            continue
        try:
            forms = [
                normalize_matching(s)
                for s in islice(
                    sample_sentence(sentence, expand_lists=False, expand_ranges=False), 16
                )
            ]
        except (HassilError, ValueError):
            continue
        if forms and all(len(s.split()) == 1 and 1 <= len(s) <= 3 for s in forms):
            words.update(forms)
    return words


def template_candidates(text, grammar, language, entities, *, threshold=94.0):
    """Rank sentence skeletons while preserving explicit targets, values and polarity.

    Args:
        text: Canonical request with known target spellings.
        grammar: Restricted HassIL grammar used for the mandatory reparse.
        language: Base HA language code.
        entities: Current request catalogue, never a global device list.
        threshold: Minimum sentence similarity; final acceptance also needs a margin.

    Returns:
        tuple: Candidate canonical sentences and diagnostic rejection reason.
    """
    reason = recovery_guard(text)
    if reason:
        return [], reason
    query = normalize_matching(text)
    action = action_signature(text)
    if not action:
        return [], "action_unresolved"
    areas = {
        e["area"]
        for e in entities
        if e["area"] and f" {normalize_matching(e['area'])} " in f" {query} "
    }
    names = {
        e["name"]
        for e in entities
        if e["name"] and f" {normalize_matching(e['name'])} " in f" {query} "
    }
    if len(areas) > 1 or len(names) > 1 or not (areas or names):
        return [], "target_unresolved"
    numbers = re.findall(r"\d+(?:[.,]\d+)?", text)
    bindings = {"area": next(iter(areas), ""), "name": next(iter(names), "")}
    if len(numbers) == 1:
        for key in ("brightness", "position", "temperature", "volume"):
            bindings[key] = numbers[0]
    articles = grammatical_words(grammar)
    candidates = []
    for intent, skeleton in _templates(language, tuple(sorted(grammar.intents))):
        slots = re.findall(r"\{([^}]+)\}", skeleton)
        if (names and "name" not in slots) or (areas and "area" not in slots):
            continue
        if not slots or any(not bindings.get(slot) for slot in slots):
            continue
        canonical = re.sub(r"\{([^}]+)\}", lambda m: bindings[m[1]], skeleton).strip()
        # Polarity is an explicit constraint, never inferred from a fuzzy score.
        if action_signature(canonical) != action:
            continue
        if re.findall(r"\d+(?:[.,]\d+)?", canonical) != numbers:
            continue
        normalized_candidate = normalize_matching(canonical)
        score, signals = similarity(query, normalized_candidate)
        query_body, candidate_body = query, normalized_candidate
        for target in sorted(areas | names, key=len, reverse=True):
            pattern = r"(?<!\w)" + re.escape(normalize_matching(target)) + r"(?!\w)"
            query_body = re.sub(pattern, " ", query_body)
            candidate_body = re.sub(pattern, " ", candidate_body)
        if articles:
            query_body = " ".join(w for w in query_body.split() if w not in articles)
            candidate_body = " ".join(w for w in candidate_body.split() if w not in articles)
        body_score, _ = similarity(
            normalize_matching(query_body), normalize_matching(candidate_body)
        )
        signals["sentence_structure"] = body_score
        # Long catalogue names must not drown out a short negation or qualifier.
        score = min(score, body_score)
        if score >= threshold:
            candidates.append(
                {
                    "intent": intent,
                    "text": canonical,
                    "score": score,
                    "signals": signals,
                    "action": action,
                }
            )
    candidates.sort(key=lambda c: (-c["score"], c["intent"], c["text"]))
    if len(candidates) > 32:
        return [], "candidate_limit"
    return candidates, "ranked" if candidates else "low_sentence_score"
