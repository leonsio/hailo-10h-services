"""Shared HassIL helpers for deterministic application intent recognition."""

from __future__ import annotations

import copy
from itertools import islice
from typing import Any, Iterable, Mapping

from hassil import Intents, TextSlotList, recognize_all
from hassil.errors import HassilError


def base_language(language: str | None, *, default: str = "en") -> str:
    """Return a normalized base language code.

    Args:
        language: Optional BCP-47/underscore language identifier.
        default: Language returned when no identifier is available.

    Returns:
        str: Lower-case base language code.
    """
    if not isinstance(language, str) or not language.strip():
        return default
    return language.strip().casefold().split("-", 1)[0].split("_", 1)[0]


def restrict_grammar(document: Mapping[str, Any], supported: Iterable[str] | None = None) -> Intents:
    """Compile a HassIL document, optionally restricting it to named intents.

    Args:
        document: HassIL-compatible dictionary.
        supported: Intent names that remain reachable. None keeps all intents.

    Returns:
        Intents: Compiled HassIL grammar.
    """
    prepared = copy.deepcopy(dict(document))
    if supported is not None:
        allowed = set(supported)
        prepared["intents"] = {
            name: data for name, data in prepared.get("intents", {}).items() if name in allowed
        }
    return Intents.from_dict(prepared)


def text_slot(values: Iterable[str], *, name: str | None = None) -> TextSlotList:
    """Build a request-scoped HassIL text slot list.

    Args:
        values: Literal values accepted by the slot.
        name: Optional HassIL list name.

    Returns:
        TextSlotList: HassIL slot list.
    """
    # External slot-list mappings are keyed by name at the call site. Omitting
    # the optional constructor name keeps compatibility with the HassIL 3.x API
    # already used by this project.
    del name
    return TextSlotList.from_strings(tuple(values))


def mapped_text_slot(
    language: str,
    name: str,
    values: Iterable[tuple[str, Any] | dict[str, Any]],
) -> TextSlotList:
    """Build a slot list whose spoken forms map to canonical application values.

    Args:
        language: HassIL grammar language.
        name: Slot/list name.
        values: ``(input, output)`` pairs or HassIL list-value dictionaries.

    Returns:
        TextSlotList: Parsed request-scoped mapping.
    """
    serialised = []
    for value in values:
        if isinstance(value, dict):
            serialised.append(dict(value))
        else:
            serialised.append({"in": value[0], "out": value[1]})
    grammar = Intents.from_dict(
        {
            "language": language,
            "intents": {},
            "lists": {name: {"values": serialised}},
        }
    )
    return grammar.slot_lists[name]


def result_slots(result: Any) -> dict[str, Any]:
    """Convert HassIL entities from a recognition result to plain slot values.

    Args:
        result: HassIL recognition result.

    Returns:
        dict[str, Any]: Slot names mapped to canonical values.
    """
    return {key: value.value for key, value in result.entities.items()}


def recognize_bounded(
    text: str,
    grammar: Intents,
    *,
    slot_lists: Mapping[str, Any] | None = None,
    language: str | None = None,
    max_results: int = 64,
) -> tuple[list[Any], str | None]:
    """Recognize with a hard ambiguity bound and normalized HassIL failures.

    Args:
        text: User utterance.
        grammar: Compiled HassIL grammar.
        slot_lists: Request-scoped slot lists overriding/augmenting the grammar.
        language: Language passed to HassIL recognition.
        max_results: Maximum accepted result count.

    Returns:
        tuple: Results and an optional stable failure reason.
    """
    try:
        results = list(
            islice(
                recognize_all(
                    text,
                    grammar,
                    slot_lists=dict(slot_lists or {}),
                    language=language or grammar.language,
                ),
                max_results + 1,
            )
        )
    except HassilError:
        return [], "grammar_context_unavailable"
    if len(results) > max_results:
        return [], "too_many_matches"
    return results, None


def unique_recognition(
    text: str,
    grammar: Intents,
    *,
    slot_lists: Mapping[str, Any] | None = None,
    language: str | None = None,
    max_results: int = 64,
) -> tuple[Any | None, str]:
    """Return one semantic HassIL match, collapsing exact duplicate parses.

    Args:
        text: User utterance.
        grammar: Compiled HassIL grammar.
        slot_lists: Request-scoped slot lists.
        language: Language passed to HassIL.
        max_results: Maximum accepted parse count.

    Returns:
        tuple: Unique result (or None) and a stable recognition reason.
    """
    results, error = recognize_bounded(
        text,
        grammar,
        slot_lists=slot_lists,
        language=language,
        max_results=max_results,
    )
    if error:
        return None, error
    distinct: dict[tuple[str, tuple[tuple[str, str], ...]], Any] = {}
    for result in results:
        slots = result_slots(result)
        key = (
            result.intent.name,
            tuple(sorted((key, repr(value)) for key, value in slots.items())),
        )
        distinct.setdefault(key, result)
    if len(distinct) == 1:
        return next(iter(distinct.values())), "matched"
    return None, "ambiguous" if distinct else "no_match"
