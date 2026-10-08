"""Cross-domain and catalogue regressions for modular assistant composition."""

import asyncio
import copy
import json
import pickle
import string
from concurrent.futures import ThreadPoolExecutor

import pytest

from hailo_services.api import app as app_module
from hailo_services.api.process_app import create_process_app
from hailo_services.assistants.frigate.frigate_assist import prepare
from hailo_services.assistants.frigate.frigate_intents import recognize_request
from hailo_services.assistants.ha.ha_intents import deterministic_intent
from hailo_services.chat.backend_hailo import HailoBackend
from hailo_services.config import FRIGATE_ASSIST_MODEL, Settings
from hailo_services.schemas import ChatRequest
from hailo_services.shared.i18n import (
    SUPPORTED_LANGUAGES,
    catalogue,
    current_language,
    localized,
    t,
    using_language,
)
from hailo_services.shared.tool_calling import response_message


def tool(name, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def frigate_request(language="en", camera="shared"):
    return ChatRequest(
        model=FRIGATE_ASSIST_MODEL,
        language=language,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for Frigate.\n"
                f"Available cameras:\n  - Office (ID: {camera}, zones: none)\n",
            },
            {"role": "user", "content": "Show me the live image from camera Office"},
        ],
        tools=[tool("get_live_context", {"camera": {"type": "string"}}, ("camera",))],
    )


def test_process_composition_does_not_replace_global_runtime_classes(monkeypatch):
    originals = (app_module.Runtime, app_module.SpeechRuntime, app_module.VisionRuntime)
    captured = {}

    def factory(settings, **kwargs):
        captured.update(kwargs)
        return settings

    monkeypatch.setattr(app_module, "create_app", factory)
    settings = Settings()
    assert create_process_app(settings) is settings
    assert (app_module.Runtime, app_module.SpeechRuntime, app_module.VisionRuntime) == originals
    assert set(captured) == {"runtime_factory", "speech_factory", "vision_factory"}
    assert captured["runtime_factory"] is not originals[0]


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_every_catalogue_has_complete_reply_ui_and_grammar_resources(language):
    reference = catalogue("en")
    translated = catalogue(language)
    for section in ("text", "ui", "states", "domains", "weather", "aliases", "language_names"):
        assert translated[section].keys() == reference[section].keys()
    for section in ("text", "ui"):
        for key, template in reference[section].items():

            def fields(value):
                return {
                    field for _, field, _, _ in string.Formatter().parse(value) if field is not None
                }

            assert fields(template) == fields(translated[section][key]), (language, key)
    assert translated["frigate"]["sentences"].keys() == reference["frigate"]["sentences"].keys()
    assert translated["frigate"]["vocabulary"].keys() == reference["frigate"]["vocabulary"].keys()
    assert translated["ha_intents"]["brightness"]
    assert len(translated["wait"]) >= 3


def test_interleaved_frigate_catalogues_never_reuse_a_previous_camera_slot():
    with ThreadPoolExecutor(max_workers=4) as executor:
        matches = list(
            executor.map(
                lambda index: recognize_request(frigate_request(camera=f"camera_{index}"))[0],
                range(24),
            )
        )
    assert [match["slots"]["camera"] for match in matches] == [
        f"camera_{index}" for index in range(24)
    ]


def test_frigate_bypasses_ha_preparation_and_target_normalization():
    request = frigate_request()
    original = copy.deepcopy(request.model_dump())
    backend = HailoBackend(Settings())
    assert backend.select_tools(request) is request
    assert request.model_dump() == original
    prepared, direct = prepare(Settings(), request)
    assert not getattr(prepared, "_ha_assist", False)
    assert json.loads(direct["tool_calls"][0]["function"]["arguments"]) == {"camera": "shared"}
    validated = response_message(direct, prepared, "")
    assert validated["tool_calls"][0]["function"]["name"] == "get_live_context"
    assert "ha_validation" not in prepared._metrics


@pytest.mark.parametrize(
    "language,question",
    [
        ("de", "Stelle das Licht im Büro auf 40 Prozent"),
        ("en", "Set the light in Büro to 40 percent"),
        ("ru", "Установи свет в Büro на 40 процентов"),
        ("fr", "Règle la lumière dans Büro à 40 pour cent"),
        ("es", "Ajusta la luz en Büro al 40 por ciento"),
        ("it", "Imposta la luce in Büro al 40 percento"),
        ("nl", "Zet het licht in Büro op 40 procent"),
        ("pt", "Ajuste a luz em Büro em 40 por cento"),
    ],
)
def test_ha_supplements_in_all_languages_preserve_catalogue_targets(language, question):
    request = ChatRequest(
        language=language,
        messages=[
            {
                "role": "system",
                "content": "Home Assistant\nStatic Context:\n"
                "- names: Lamp\n  domain: light\n  areas: Büro\n",
            },
            {"role": "user", "content": question},
        ],
        tools=[
            tool(
                "intent__HassLightSet",
                {
                    "area": {"type": "string"},
                    "domain": {"type": "array", "items": {"type": "string"}},
                    "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                },
            )
        ],
    )
    result, trace = deterministic_intent(request, Settings(), language)
    assert result is not None, trace
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {
        "area": "Büro",
        "domain": ["light"],
        "brightness": 40,
    }


def test_parallel_localization_and_ipc_roundtrip_do_not_leak_request_language():
    async def answer(language):
        with using_language(language):
            await asyncio.sleep(0)
            request = pickle.loads(pickle.dumps(frigate_request(language)))
            assert current_language() == language
            assert (
                localized(request, "frigate.camera_missing")
                == catalogue(language)["text"]["frigate.camera_missing"]
            )
            return t("ha_action.done")

    async def run():
        return await asyncio.gather(*(answer(language) for language in SUPPORTED_LANGUAGES))

    before = current_language()
    assert asyncio.run(run()) == [
        catalogue(language)["text"]["ha_action.done"] for language in SUPPORTED_LANGUAGES
    ]
    assert current_language() == before
