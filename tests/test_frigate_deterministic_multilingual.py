"""Regression coverage for language-neutral deterministic Frigate/NVR routing."""

import json

import pytest

from hailo_services.assistants.frigate.frigate_deterministic import deterministic_plan
from hailo_services.config import FRIGATE_ASSIST_MODEL, Settings
from hailo_services.schemas import ChatRequest

SYSTEM = """You are a helpful assistant for Frigate, a security camera NVR system.
Current server local date and time: 2026-10-07 at 09:02:19 PM
Available cameras:
  - Front Door (ID: front_door, zones: Entry (ID: entry))
  - Garden (ID: garden, zones: Lawn (ID: lawn))
"""


def tool(name, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Frigate capability {name}",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool(
        "get_recap",
        {
            "after": {"type": "string"},
            "before": {"type": "string"},
            "cameras": {"type": "string"},
        },
        ["after", "before"],
    ),
    tool("get_live_context", {"camera": {"type": "string"}}, ["camera"]),
    tool(
        "search_objects",
        {
            "camera": {"type": "string"},
            "label": {"type": "string"},
            "semantic_query": {"type": "string"},
            "after": {"type": "string"},
            "before": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    ),
    tool(
        "set_camera_state",
        {
            "camera": {"type": "string"},
            "feature": {"type": "string", "enum": ["detect", "record"]},
            "value": {"type": "string", "enum": ["ON", "OFF"]},
        },
        ["camera", "feature", "value"],
    ),
    tool(
        "start_camera_watch",
        {"camera": {"type": "string"}, "condition": {"type": "string"}},
        ["camera", "condition"],
    ),
    tool("stop_camera_watch"),
]


def request(question, tools=None, language=None):
    body = {
        "model": FRIGATE_ASSIST_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        "tools": tools or TOOLS,
        "tool_choice": "auto",
    }
    if language:
        body["language"] = language
    return ChatRequest(**body)


def name_and_args(message):
    function = message["tool_calls"][0]["function"]
    return function["name"], json.loads(function["arguments"])


@pytest.mark.parametrize(
    "question",
    [
        "Zeige die Ereignisse der letzten 3 Stunden",
        "Show the events from the last 3 hours",
        "Montre les événements des dernières 3 heures",
        "Muestra los eventos de las últimas 3 horas",
        "Mostra gli eventi delle ultime 3 ore",
        "Toon gebeurtenissen van de afgelopen 3 uren",
        "Mostra os eventos das últimas 3 horas",
        "Покажи события за последние 3 часа",
    ],
)
def test_relative_recap_interval_is_multilingual(question):
    selected, direct, reason = deterministic_plan(request(question), Settings(), False)
    assert reason == "deterministic_recap_interval"
    assert [item["function"]["name"] for item in selected] == ["get_recap"]
    assert name_and_args(direct) == (
        "get_recap",
        {"after": "2026-10-07T18:02:19", "before": "2026-10-07T21:02:19"},
    )


@pytest.mark.parametrize(
    "question",
    [
        "Zeige die Ereignisse ab heute 06:00 Uhr bis jetzt",
        "Show events since today 06:00 until now",
        "Montre les événements depuis aujourd'hui 06:00",
        "Muestra los eventos desde hoy 06:00",
        "Mostra gli eventi da oggi 06:00",
        "Toon gebeurtenissen vanaf vandaag 06:00",
        "Mostra os eventos desde hoje 06:00",
    ],
)
def test_since_today_interval_is_multilingual(question):
    _, direct, reason = deterministic_plan(request(question), Settings(), False)
    assert reason == "deterministic_recap_interval"
    assert name_and_args(direct) == (
        "get_recap",
        {"after": "2026-10-07T06:00:00", "before": "2026-10-07T21:02:19"},
    )


@pytest.mark.parametrize(
    ("question", "feature", "value"),
    [
        ("Schalte die Erkennung für Front Door aus", "detect", "OFF"),
        ("Turn detection off for Front Door", "detect", "OFF"),
        ("Désactive la détection sur Front Door", "detect", "OFF"),
        ("Desactiva la detección en Front Door", "detect", "OFF"),
        ("Disattiva il rilevamento su Front Door", "detect", "OFF"),
        ("Zet detectie uit op Front Door", "detect", "OFF"),
        ("Desative a detecção em Front Door", "detect", "OFF"),
        ("Выключи обнаружение Front Door", "detect", "OFF"),
    ],
)
def test_camera_feature_toggle_is_multilingual(question, feature, value):
    selected, direct, reason = deterministic_plan(request(question), Settings(), False)
    assert reason == "deterministic_camera_state"
    assert [item["function"]["name"] for item in selected] == ["set_camera_state"]
    assert name_and_args(direct) == (
        "set_camera_state",
        {"camera": "front_door", "feature": feature, "value": value},
    )


@pytest.mark.parametrize(
    "question",
    [
        "Überwache Garden und benachrichtige mich wenn jemand kommt",
        "Watch Garden and notify me when someone arrives",
        "Surveille Garden et avertis-moi si quelqu'un arrive",
        "Vigila Garden y notifica si llega alguien",
        "Monitora Garden e avvisa se arriva qualcuno",
        "Bewaak Garden en meld als iemand aankomt",
        "Monitora Garden e avise se alguém chegar",
        "Следи за Garden и уведомляй если кто-то придет",
    ],
)
def test_watch_route_constrains_an_exact_camera_in_all_languages(question):
    selected, direct, reason = deterministic_plan(request(question), Settings(), False)
    assert direct is None
    assert reason == "deterministic_watch_tool_selection"
    assert [item["function"]["name"] for item in selected] == ["start_camera_watch"]
    camera = selected[0]["function"]["parameters"]["properties"]["camera"]
    assert camera["enum"] == ["garden"]


@pytest.mark.parametrize(
    ("question", "label"),
    [
        ("Wann wurde zuletzt eine Person bei Garden erkannt?", "person"),
        ("When was a person last detected at Garden?", "person"),
        ("Quand une personne a été vue pour la dernière fois à Garden?", "person"),
        ("Cuándo fue detectado por última vez un coche en Garden?", "car"),
        ("Quando è stata vista l'ultima volta una persona a Garden?", "person"),
        ("Wanneer is een persoon voor het laatst gezien bij Garden?", "person"),
        ("Quando uma pessoa foi vista pela última vez em Garden?", "person"),
        ("Когда автомобиль был обнаружен последний раз в Garden?", "car"),
    ],
)
def test_class_last_sighting_is_multilingual(question, label):
    selected, direct, reason = deterministic_plan(request(question), Settings(), False)
    assert reason == "deterministic_class_last_sighting"
    assert [item["function"]["name"] for item in selected] == ["search_objects"]
    assert name_and_args(direct) == (
        "search_objects",
        {"label": label, "limit": 1, "camera": "garden"},
    )


@pytest.mark.parametrize(
    "question",
    [
        "Zeige was gerade auf Garden sichtbar ist",
        "Show what is visible now on Garden",
        "Montre ce qui est visible maintenant sur Garden",
        "Muestra qué es visible ahora en Garden",
        "Mostra cosa è visibile adesso su Garden",
        "Toon wat nu zichtbaar is op Garden",
        "Mostra o que está visível agora em Garden",
        "Покажи что сейчас видно на Garden",
    ],
)
def test_live_context_is_multilingual(question):
    selected, direct, reason = deterministic_plan(request(question), Settings(), False)
    assert reason == "deterministic_live_context"
    assert [item["function"]["name"] for item in selected] == ["get_live_context"]
    assert name_and_args(direct) == ("get_live_context", {"camera": "garden"})
