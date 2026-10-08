"""External language resources and request-scoped localization.

Matching canonicalizes vocabulary only. Entity names and tool arguments always
come from HA, and the original user text is preserved for model inference.
"""

from __future__ import annotations

import json
import random
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from hailo_services.shared.intent_engine import base_language

SUPPORTED_LANGUAGES = tuple(
    sorted(path.stem for path in (Path(__file__).parents[1] / "locales").glob("*.json"))
)
_LANGUAGE = ContextVar("hailo_language", default="de")


@lru_cache(maxsize=None)
def catalogue(language):
    """Load and cache one supported language resource file.

    Args:
        language (str | None): Language code; None uses the configured or detected language.

    Returns:
        dict[str, Any]: Language text, aliases, labels, UI strings and routing resources.

    Raises:
        OSError: The language resource file cannot be read.
        ValueError: The resource contains invalid JSON.
    """
    with (Path(__file__).parents[1] / "locales" / f"{language}.json").open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def language_code(value, fallback="de"):
    """Normalize a supported language code, retaining a configured fallback.

    Args:
        value (Any): Input value inspected or normalized by this helper.
        fallback (str): Language used when detection or normalization cannot choose one.

    Returns:
        str: Bundled language code, or the supplied fallback.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    code = base_language(value, default=fallback)
    return code if code in SUPPORTED_LANGUAGES else fallback


@contextmanager
def using_language(language):
    """Set request-local language and restore the previous context on exit.

    Args:
        language (str | None): Language code; None uses the configured or detected language.

    Yields:
        None: Runs the context body with the selected language.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    token = _LANGUAGE.set(language_code(language))
    try:
        yield
    finally:
        _LANGUAGE.reset(token)


def current_language():
    """Read the current request-local language.

    Returns:
        str: Active service language code.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return _LANGUAGE.get()


def t(key, **values):
    """Format localized text with German fallback for missing translations.

    Args:
        key (str): Resource or field identifier to look up.
        **values (Any): Formatting substitutions, metric updates or catalogue values, depending on the helper.

    Returns:
        str: Translated text with substituted placeholders.

    Raises:
        KeyError: The key is absent from both catalogues or a format value is missing.
    """
    return translate(key, **values)


def translate(key, *, language=None, **values):
    """Format a resource in an explicit or request-local language.

    Args:
        key: Stable text resource key.
        language: Optional explicit language; does not change the current context.
        **values: Template substitutions.

    Returns:
        str: Formatted localized text.
    """
    code = language_code(language, current_language())
    template = catalogue(code)["text"].get(key, catalogue("de")["text"].get(key))
    if template is None:
        raise KeyError(key)
    return template.format(**values)


def localized(request, key, **values):
    """Render a response using the client's language or latest human question.

    Args:
        request: Incoming chat request, independent of HA or Frigate state.
        key: Stable response resource key.
        **values: Template substitutions.

    Returns:
        str: Localized response without mutating any language context.
    """
    parts = []
    for message in reversed(request.messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(part.get("text", "") for part in content if part.get("type") == "text")
        if parts:
            break
    text = " ".join(parts)
    language = language_code(
        getattr(request, "_response_language", None) or request.language,
        detect_language(text, "en"),
    )
    return translate(key, language=language, **values)


@lru_cache(maxsize=None)
def vocabulary(key):
    """Combine per-language Frigate vocabulary, retaining canonical mappings.

    Args:
        key: Canonical vocabulary identifier.

    Returns:
        set | dict: Multilingual vocabulary for deterministic recognition.
    """
    result = None
    for language in SUPPORTED_LANGUAGES:
        value = catalogue(language)["frigate"]["vocabulary"][key]
        if isinstance(value, list):
            if result is None:
                result = set()
            result.update(value)
        else:
            if result is None:
                result = {}
            for canonical, aliases in value.items():
                if isinstance(aliases, list):
                    result.setdefault(canonical, set()).update(aliases)
                elif canonical in result and result[canonical] != aliases:
                    raise ValueError(f"Conflicting vocabulary mapping: {key}.{canonical}")
                else:
                    result[canonical] = aliases
    return result


@lru_cache(maxsize=None)
def lexicon(key):
    """Load routing vocabulary and decode tagged sets and tuples.

    Args:
        key (str): Resource or field identifier to look up.

    Returns:
        Any: Cached routing vocabulary for the supplied key.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """

    def unpack(value):
        """Decode tagged sets and tuples in declarative routing vocabulary.

        Args:
            value (Any): Input value inspected or normalized by this helper.

        Returns:
            Any: Recursively decoded vocabulary value.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if isinstance(value, dict):
            if "$set" in value:
                return set(value["$set"])
            if "$tuple" in value:
                return tuple(value["$tuple"])
            return {k: unpack(v) for k, v in value.items()}
        return value

    return unpack(catalogue("de")["routing"][key])


def labels(section):
    """Read localized labels for a domain or measurement section.

    Args:
        section (str): Language catalogue label section.

    Returns:
        dict[str, Any]: Requested localized label mapping.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return catalogue(current_language())[section]


_LAST_WAIT = {}


def wait_sentence(language):
    """Choose a localized wait message without repeating the last one.

    Args:
        language (str | None): Language code; None uses the configured or detected language.

    Returns:
        str: Selected wait notification.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    language = language_code(language)
    choices = [text for text in catalogue(language)["wait"] if text != _LAST_WAIT.get(language)]
    sentence = random.choice(choices)
    _LAST_WAIT[language] = sentence
    return sentence


def _plain(value):
    """Case-fold text and normalize punctuation and German umlauts.

    Args:
        value (Any): Input value inspected or normalized by this helper.

    Returns:
        str: Plain text for language and alias matching.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    text = str(value).casefold().replace("ß", "ss")
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text)).strip()


@lru_cache(maxsize=1)
def _aliases():
    """Build a cached multilingual alias-to-canonical-word mapping.

    Returns:
        dict[str, str]: Normalized alias mapping.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    result = {}
    for language in SUPPORTED_LANGUAGES:
        for canonical, aliases in catalogue(language)["aliases"].items():
            for alias in aliases:
                result[_plain(alias)] = canonical
    return result


@lru_cache(maxsize=1)
def _matching_pattern():
    """Compile a cached whole-word pattern for all routing aliases.

    Returns:
        re.Pattern[str]: Regex matching the longest aliases first.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    aliases = _aliases()
    pattern = (
        r"(?<!\w)(?:"
        + "|".join(re.escape(x) for x in sorted(aliases, key=len, reverse=True))
        + r")(?!\w)"
    )
    return re.compile(pattern)


def normalize_matching(value):
    """Canonicalize localized vocabulary without modifying catalogue targets.

    Args:
        value (Any): Input value inspected or normalized by this helper.

    Returns:
        str: Normalized language-independent matching text.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    return _matching_pattern().sub(lambda match: _aliases()[match[0]], _plain(value))


def detect_language(text, fallback="de"):
    """Infer a bundled language from script and known vocabulary.

    Args:
        text (str): Text to parse, normalize, match or render.
        fallback (str): Language used when detection or normalization cannot choose one.

    Returns:
        str: Detected language, or normalized fallback.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if re.search("[а-яА-ЯёЁ]", text):
        return "ru"
    plain = " " + _plain(text) + " "
    scores = {}
    for language in SUPPORTED_LANGUAGES:
        data = catalogue(language)
        candidates = {alias for aliases in data["aliases"].values() for alias in aliases}
        candidates.update(data.get("frigate", {}).get("hints", []))
        scores[language] = sum(
            len(alias.split()) for alias in candidates if " " + _plain(alias) + " " in plain
        )
    best = max(scores.values(), default=0)
    winners = [language for language, score in scores.items() if score == best and best > 0]
    fallback = language_code(fallback)
    if len(winners) == 1:
        return winners[0]
    return fallback if fallback in winners or not winners else winners[0]
