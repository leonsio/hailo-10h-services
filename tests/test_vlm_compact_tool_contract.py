import json

import pytest

from hailo_services.chat_hailo_vlm import model_prompt, tool_response
from hailo_services.schemas import ChatRequest


def tool(name="get_time"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read the time.",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
                "additionalProperties": False,
            },
        },
    }


def request(**kwargs):
    return ChatRequest(
        messages=[{"role": "user", "content": "Wie spät ist es in UTC?"}],
        tools=[tool()],
        tool_choice="required",
        parallel_tool_calls=False,
        **kwargs,
    )


def test_vlm_tool_prompt_uses_compact_contract_not_full_json_schema():
    prompt = model_prompt(request())
    text = prompt[0]["content"][0]["text"]
    assert "get_time(zone!:string)" in text
    assert '{"name":"get_time","arguments":{...}}' in text
    assert "additionalProperties" not in text
    assert '"tool_calls"' not in text


@pytest.mark.parametrize(
    "raw",
    [
        '{"name":"get_time","arguments":{"zone":"UTC"}}',
        '{"function":{"name":"get_time","arguments":{"zone":"UTC"}}}',
        '{"zone":"UTC"}',
        '{"tool_calls":[{"function":{"name":"get_time","arguments":{"zone":"UTC"}}}]}',
    ],
)
def test_vlm_tool_response_accepts_compact_safe_shapes(raw):
    result = tool_response(raw, request())
    call = result["tool_calls"][0]["function"]
    assert call["name"] == "get_time"
    assert json.loads(call["arguments"]) == {"zone": "UTC"}


def test_vlm_compact_tool_response_still_rejects_unknown_tools_and_invalid_arguments():
    with pytest.raises(ValueError, match="unavailable function"):
        tool_response('{"name":"not_available","arguments":{"zone":"UTC"}}', request())
    with pytest.raises(ValueError, match="invalid arguments"):
        tool_response('{"name":"get_time","arguments":{}}', request())


def test_vlm_invalid_json_reports_and_logs_raw_model_output(caplog):
    with caplog.at_level("DEBUG"):
        with pytest.raises(ValueError, match=r"valid JSON; raw_output=I would call get_time"):
            tool_response("I would call get_time", request())
    assert "event=vlm_raw_output" in caplog.text
    assert "I would call get_time" in caplog.text


def test_vlm_schema_error_contains_raw_model_output_for_diagnostics():
    raw = '{"name":"get_time","arguments":{}}'
    with pytest.raises(ValueError, match=r"raw_output=.*get_time"):
        tool_response(raw, request())
