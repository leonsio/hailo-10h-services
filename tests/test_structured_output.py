"""HTTP coverage for Home Assistant structured data generation."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_services import FakeBackend, settings

from hailo_services.api.app import create_app
from hailo_services.config import VLM_MODEL
from hailo_services.schemas import ChatRequest

FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "RoomData",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"room": {"type": "string"}},
            "required": ["room"],
            "additionalProperties": False,
        },
    },
}


class JSONBackend(FakeBackend):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def chat(self, request, emit=None, cancelled=None):
        self.calls.append(request)
        request._metrics.update(input_tokens=10, output_tokens=5)
        return self.output


def payload(**kwargs):
    return {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": "Extract the room: kitchen"}],
        "response_format": FORMAT,
        **kwargs,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_structured_generation_and_ha_usage(stream):
    backend = JSONBackend('{"room":"kitchen"}')
    with TestClient(create_app(settings(), backend)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=payload(stream=stream, stream_options={"include_usage": True}),
        )
    assert response.status_code == 200
    if stream:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        content = "".join(
            e["choices"][0]["delta"].get("content", "") for e in events if e["choices"]
        )
        assert events[-1]["usage"]["total_tokens"] == 15
    else:
        content = response.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"room": "kitchen"}
    assert "JSON Schema" in backend.calls[0].messages[0]["content"]


@pytest.mark.parametrize("output", ['{"room":42}', "```json\n{}\n```", "{}", "NaN"])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_output_is_never_published(output, stream):
    with TestClient(create_app(settings(), JSONBackend(output))) as client:
        response = client.post("/v1/chat/completions", json=payload(stream=stream))
    if stream:
        assert '"error"' in response.text
        assert '"finish_reason": "stop"' not in response.text
        assert output not in response.text
    else:
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "structured_output_invalid"


@pytest.mark.parametrize(
    "fmt",
    [
        {"type": "json_schema"},
        {"type": "json_object", "json_schema": FORMAT["json_schema"]},
        {"type": "json_schema", "json_schema": {"name": "bad", "schema": {"type": "wrong"}}},
        {
            "type": "json_schema",
            "json_schema": {"name": "bad", "schema": {"$ref": "https://example.com/schema"}},
        },
    ],
)
def test_invalid_formats_rejected(fmt):
    with pytest.raises(ValidationError):
        ChatRequest(**payload(response_format=fmt))


@pytest.mark.parametrize(
    "fmt,output,status",
    [
        ({"type": "json_object"}, "{}", 200),
        ({"type": "json_object"}, "[]", 502),
        ({"type": "text"}, "ordinary text", 200),
    ],
)
def test_object_and_text_formats(fmt, output, status):
    with TestClient(create_app(settings(), JSONBackend(output))) as client:
        response = client.post("/v1/chat/completions", json=payload(response_format=fmt))
    assert response.status_code == status


def test_intermediate_tools_preserved():
    output = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ],
    }
    with TestClient(create_app(settings(), JSONBackend(output))) as client:
        response = client.post("/v1/chat/completions", json=payload())
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_gemma_direct_model():
    from test_services import FakeLiteRTBackend

    from hailo_services.config import LLM_MODEL

    class GemmaJSONBackend(FakeLiteRTBackend):
        def chat(self, request, emit=None, cancelled=None):
            self.calls.append(request)
            return '{"room":"kitchen"}'

    backend = GemmaJSONBackend()
    with TestClient(create_app(settings(litert_enabled=True), FakeBackend(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload(model=LLM_MODEL))
    assert response.status_code == 200
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {"room": "kitchen"}
    assert "JSON Schema" in str(backend.calls[0].messages)


@pytest.mark.parametrize("model", ["HA-Assist", "Frigate-Assist"])
def test_virtual_models_reject_structured_generation(model):
    backend = JSONBackend("{}")
    with TestClient(create_app(settings(), backend)) as client:
        response = client.post("/v1/chat/completions", json=payload(model=model))
    assert response.status_code == 400
    assert not backend.calls


def test_local_schema_references():
    fmt = {
        "type": "json_schema",
        "json_schema": {
            "name": "LocalReference",
            "schema": {
                "$defs": {"room": {"type": "string"}},
                "type": "object",
                "properties": {"room": {"$ref": "#/$defs/room"}},
            },
        },
    }
    with TestClient(create_app(settings(), JSONBackend('{"room":"kitchen"}'))) as client:
        response = client.post("/v1/chat/completions", json=payload(response_format=fmt))
    assert response.status_code == 200
