import asyncio
import json
import threading
from pathlib import Path

import pytest

from hailo_services.app import create_app
from hailo_services.config import LLM_MODEL, Settings
from hailo_services.ha_pipeline import direct_numeric_action, is_home_assistant_request
from hailo_services.ha_state_routing import _query_kind, deterministic_live_response
from hailo_services.i18n import catalogue, detect_language, normalize_matching, using_language
from hailo_services.protocols import wyoming_info
from hailo_services.runtime import HailoBackend
from hailo_services.schemas import ChatRequest

SYSTEM = """Home Assistant
Static Context:
- names: Pendant
  domain: light
  areas: Living Room
- names: Desk Lamp
  domain: light
  areas: Living Room
- names: Blind
  domain: cover
  areas: Living Room
"""


def tool(name, properties=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties
                or {
                    "area": {"type": "string"},
                    "name": {"type": "string"},
                    "domain": {"type": "array", "items": {"type": "string"}},
                    "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                    "position": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "additionalProperties": False,
            },
        },
    }


def request(text, **kwargs):
    return ChatRequest(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        **kwargs,
    )


@pytest.mark.parametrize(
    "text,language",
    [
        ("Stelle das Licht im Wohnzimmer auf 40 Prozent", "de"),
        ("Set lights in the living room to 40 percent", "en"),
        ("Установи свет в гостиной на 40 процентов", "ru"),
    ],
)
def test_brightness_in_three_languages_uses_original_area_name(text, language):
    req = request(text, tools=[tool("light__HassLightSet")])
    result = direct_numeric_action(req)
    assert result is not None
    arguments = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"area": "Living Room", "domain": ["light"], "brightness": 40}
    assert detect_language(text) == language
    assert req.messages[-1]["content"] == text


@pytest.mark.parametrize(
    "text",
    [
        "Stelle den Rollladen im Wohnzimmer auf 60 Prozent",
        "Set blinds in the living room to 60 percent",
        "Установи жалюзи в гостиной на 60 процентов",
    ],
)
def test_cover_position(text):
    result = direct_numeric_action(request(text, tools=[tool("intent__HassSetPosition")]))
    assert json.loads(result["tool_calls"][0]["function"]["arguments"])["position"] == 60


@pytest.mark.parametrize(
    "text",
    [
        "Set lights in living room to 140 percent",
        "Set lights in living room to -40 percent",
        "Set lights in living room to 40.5 percent",
        "Set lights in living room to 40 percent and blue",
        "Set lights to 40 percent",
        "Do not set lights in living room to 40 percent",
        "Установи свет в гостиной на 40 процентов если темно",
        "Mach das Licht im Wohnzimmer um 40 Prozent heller",
    ],
)
def test_unsafe_or_ambiguous_commands_defer(text):
    assert direct_numeric_action(request(text, tools=[tool("light__HassLightSet")])) is None


def test_tool_choice_and_client_schema_are_authoritative():
    req = request(
        "Set lights in living room to 40 percent",
        tools=[tool("light__HassLightSet")],
        tool_choice="none",
    )
    assert direct_numeric_action(req) is None
    req = req.model_copy(
        update={
            "tool_choice": {
                "type": "function",
                "function": {"name": "homeassistant__GetLiveContext"},
            }
        }
    )
    assert direct_numeric_action(req) is None
    req = req.model_copy(
        update={
            "tool_choice": None,
            "tools": [
                tool(
                    "light__HassLightSet",
                    {"name": {"type": "string"}, "brightness": {"type": "integer"}},
                )
            ],
        }
    )
    assert direct_numeric_action(req) is None


class RoutingBackend(HailoBackend):
    paths = {}

    def start(self):
        pass

    def close(self):
        pass


@pytest.mark.parametrize(
    "text", ["Explain lights in physics", "Was ist Temperatur?", "Что такое свет?"]
)
def test_non_ha_request_is_returned_identically_even_with_generic_tools(text):
    req = request(text, tools=[tool("search")])
    backend = RoutingBackend(Settings())
    assert not is_home_assistant_request(req)
    assert backend.select_tools(req) is req


@pytest.mark.parametrize(
    "text,kind",
    [
        ("How many lights are on?", "count"),
        ("Which lights are off?", "list"),
        ("Сколько ламп включены?", "count"),
        ("Какие лампы выключены?", "list"),
        ("Какая температура в саду?", "temperature"),
        ("What is the humidity?", "humidity"),
    ],
)
def test_multilingual_read_classification(text, kind):
    assert _query_kind(text) == kind


@pytest.mark.parametrize(
    "language,question,expected",
    [
        ("de", "Ist die Lampe an?", "Ja, Pendant ist an."),
        ("en", "Is the lamp on?", "Yes, Pendant is on."),
        ("ru", "Свет включен?", "Да, Pendant: включено."),
    ],
)
def test_localized_live_answer(language, question, expected):
    req = request(question, tools=[tool("homeassistant__GetLiveContext")])
    req.messages.extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {
                            "name": "homeassistant__GetLiveContext",
                            "arguments": '{"domain":["light"]}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": '{"Pendant":"on"}'},
        ]
    )
    with using_language(language):
        assert deterministic_live_response(req) == expected


def test_locale_keys_placeholders_and_assets_are_consistent():
    import string

    base = catalogue("de")
    for language in ("en", "ru"):
        data = catalogue(language)
        for section in ("text", "ui", "states", "domains", "weather"):
            assert data[section].keys() == base[section].keys()
        for section in ("text", "ui"):
            for key, value in base[section].items():

                def fields(text):
                    return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}

                assert fields(value) == fields(data[section][key]), key
        assert len(data["wait"]) >= 3
    assert normalize_matching("Light") == normalize_matching("Licht") == normalize_matching("свет")
    assert Path(__file__).parent.parent.joinpath("src/hailo_services/locales/ru.json").is_file()


def test_wyoming_advertises_all_three_languages_independently_of_default():
    languages = wyoming_info().asr[0].models[0].languages
    assert {"de", "en", "ru"} <= set(languages)


def test_sse_wait_precedes_inference_completion_and_tools_remain_buffered():
    """Release the blocking model only after ASGI has sent the wait sentence."""

    async def run():
        release = threading.Event()

        class LLM:
            debug_log = False

            def chat(self, req, emit=None, cancelled=None, tools_prepared=False):
                req._on_inference(req._response_language)
                assert release.wait(2), "SSE status was buffered until after inference"
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {
                                "name": "homeassistant__GetLiveContext",
                                "arguments": "{}",
                            },
                        }
                    ],
                }

        settings = Settings(wyoming_port=0)
        app = create_app(settings, RoutingBackend(settings), LLM())
        runtime = app.state.runtime
        runtime.ready = runtime.litert_ready = True
        req = request(
            "Please analyze all my devices carefully",
            tools=[tool("homeassistant__GetLiveContext")],
            language="en",
            stream=True,
        )
        req.tool_choice = {
            "type": "function",
            "function": {"name": "homeassistant__GetLiveContext"},
        }
        body = req.model_dump_json().encode()
        sent = []
        consumed = False

        async def receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Future()

        async def send(message):
            sent.append(message)
            chunk = message.get("body", b"").decode()
            if any(sentence in chunk for sentence in catalogue("en")["wait"]):
                assert "tool_calls" not in chunk
                release.set()

        try:
            await asyncio.wait_for(
                app(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0", "spec_version": "2.3"},
                        "http_version": "1.1",
                        "method": "POST",
                        "scheme": "http",
                        "path": "/v1/chat/completions",
                        "raw_path": b"/v1/chat/completions",
                        "query_string": b"",
                        "headers": [(b"content-type", b"application/json")],
                        "client": ("127.0.0.1", 123),
                        "server": ("localhost", 8090),
                        "root_path": "",
                    },
                    receive,
                    send,
                ),
                4,
            )
            output = b"".join(m.get("body", b"") for m in sent).decode()
            assert release.is_set()
            sentence = next(text for text in catalogue("en")["wait"] if text in output)
            assert output.index(sentence) < output.index("tool_calls")
            assert "[DONE]" in output
        finally:
            release.set()
            runtime.executor.shutdown(wait=True)
            runtime.litert_executor.shutdown(wait=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "text,area",
    [
        ("Is the light in Unknown Room on?", "Unknown Room"),
        ("Свет в неизвестной комнате включен?", "неизвестной комнате"),
    ],
)
def test_unknown_area_preserves_original_name(text, area):
    from hailo_services.ha_state_routing_fixes import _location_phrase

    assert _location_phrase(text) == area


def test_zero_current_temperature_and_humidity_are_not_lost():
    from hailo_services.ha_state_routing import _measurement_value

    entity = {
        "domain": "climate",
        "state": "heat",
        "attributes": {
            "current_temperature": 0,
            "temperature": 24,
            "current_humidity": 0,
            "humidity": 40,
        },
    }
    assert _measurement_value(entity, "temperature") == "0"
    assert _measurement_value(entity, "humidity") == "0"


def test_wait_sentences_do_not_repeat_consecutively():
    from hailo_services.i18n import wait_sentence

    for language in ("de", "en", "ru"):
        sentences = [wait_sentence(language) for _ in range(10)]
        assert all(left != right for left, right in zip(sentences, sentences[1:]))


def test_public_locale_routes_work_with_api_key_and_reject_unknown_language():
    from fastapi.testclient import TestClient

    app = create_app(Settings(api_key="secret", wyoming_port=0), RoutingBackend(Settings()))
    with TestClient(app) as client:
        assert client.get("/ui/locales/en.json").json()["ui"] == catalogue("en")["ui"]
        assert client.get("/ui/locales/ru.json").status_code == 200
        assert client.get("/ui/config").json()["stt_languages"]
        assert client.get("/ui/locales/../../config.py").status_code != 200
        assert client.get("/v1/models").status_code == 401


def test_negated_onoff_command_never_gets_a_direct_action():
    req = request("Schalte das Licht im Wohnzimmer nicht aus", tools=[tool("intent__HassTurnOff")])
    prepared = RoutingBackend(Settings()).select_tools(req)
    assert getattr(prepared, "_direct_ha_response", None) is None
