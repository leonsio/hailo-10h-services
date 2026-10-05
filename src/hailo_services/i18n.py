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

SUPPORTED_LANGUAGES = ("de", "en", "ru")
_LANGUAGE = ContextVar("hailo_language", default="de")


@lru_cache(maxsize=3)
def catalogue(language):
    with (Path(__file__).parent / "locales" / f"{language}.json").open(encoding="utf-8") as file:
        return json.load(file)


def language_code(value, fallback="de"):
    code = str(value or "").lower().split("-")[0].split("_")[0]
    return code if code in SUPPORTED_LANGUAGES else fallback


@contextmanager
def using_language(language):
    token = _LANGUAGE.set(language_code(language))
    try:
        yield
    finally:
        _LANGUAGE.reset(token)


def current_language():
    return _LANGUAGE.get()


def t(key, **values):
    template = catalogue(current_language())["text"].get(key)
    if template is None:
        template = catalogue("de")["text"][key]
    return template.format(**values)


@lru_cache(maxsize=None)
def lexicon(key):
    def unpack(value):
        if isinstance(value, dict):
            if "$set" in value:
                return set(value["$set"])
            if "$tuple" in value:
                return tuple(value["$tuple"])
            return {k: unpack(v) for k, v in value.items()}
        return value

    return unpack(catalogue("de")["routing"][key])


def labels(section):
    return catalogue(current_language())[section]


_LAST_WAIT = {}


def wait_sentence(language):
    language = language_code(language)
    choices = [text for text in catalogue(language)["wait"] if text != _LAST_WAIT.get(language)]
    sentence = random.choice(choices)
    _LAST_WAIT[language] = sentence
    return sentence


def _plain(value):
    text = str(value).casefold().replace("ß", "ss")
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text)).strip()


@lru_cache(maxsize=1)
def _aliases():
    result = {}
    for language in SUPPORTED_LANGUAGES:
        for canonical, aliases in catalogue(language)["aliases"].items():
            for alias in aliases:
                result[_plain(alias)] = canonical
    return result


@lru_cache(maxsize=1)
def _matching_pattern():
    aliases = _aliases()
    pattern = (
        r"(?<!\w)(?:"
        + "|".join(re.escape(x) for x in sorted(aliases, key=len, reverse=True))
        + r")(?!\w)"
    )
    return re.compile(pattern)


def normalize_matching(value):
    return _matching_pattern().sub(lambda match: _aliases()[match[0]], _plain(value))


def detect_language(text, fallback="de"):
    if re.search("[а-яА-ЯёЁ]", text):
        return "ru"
    plain = " " + _plain(text) + " "
    scores = {}
    for language in ("de", "en"):
        scores[language] = sum(
            len(alias.split())
            for aliases in catalogue(language)["aliases"].values()
            for alias in aliases
            if " " + _plain(alias) + " " in plain
        )
    if scores["en"] > scores["de"]:
        return "en"
    if scores["de"] > scores["en"]:
        return "de"
    return language_code(fallback)
