"""HTTP and native-boundary regressions for virtual routing and plain isolation."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from hailo_services.app import create_app
from hailo_services.config import HA_ASSIST_MODEL, LLM_MODEL, Settings
from hailo_services.ha_fuzzy import slot_repairs
from hailo_services.ha_intents import deterministic_intent
from hailo_services.runtime import HailoBackend, LiteRTLMBackend
from hailo_services.schemas import ChatRequest
from hailo_services.tool_calling import response_message
from hailo_services.tool_retrieval import retrieve_tools
from hailo_services.vlm_chat import model_prompt

SYSTEM = """Home Assistant
Static Context:
- names: Deckenlampe
  domain: light
  areas: Wohnzimmer
- names: Stehlampe
  domain: light
  areas: Wohnzimmer
- names: Heizung
  domain: climate
  areas: Wohnzimmer
- names: Rollladen
  domain: cover
  areas: Wohnzimmer
"""


def tool(name, **properties):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {
            "area": {"type": "string"}, "name": {"type": "string"},
            "domain": {"type": "array", "items": {"type": "string"}}, **properties},
        "additionalProperties": False}}}


TOOLS = [tool("intent__HassTurnOn"), tool("intent__HassTurnOff"),
         tool("light__HassLightSet", brightness={"type": "integer", "minimum": 0, "maximum": 100}),
         tool("intent__HassSetPosition", position={"type": "integer", "minimum": 0, "maximum": 100}),
         tool("climate__HassClimateSetTemperature", temperature={"type": "number"})]


def payload(text="Mach das Licht im Wohnzimmer aus", model=HA_ASSIST_MODEL, **extra):
    return {"model": model, "messages": [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": text}],
            "tools": copy.deepcopy(TOOLS), **extra}


class Backend(HailoBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self.calls = []
        self.selections = []

    def start(self):
        pass

    def close(self):
        pass

    def select_tools(self, request):
        self.selections.append(request)
        return super().select_tools(request)

    def chat(self, request, emit=None, cancelled=None):
        self.calls.append(request)
        if emit:
            emit("vision answer")
        return "vision answer"


class LLM:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def start(self):
        pass

    def close(self):
        pass

    def chat(self, request, emit=None, cancelled=None):
        self.calls.append(request)
        if self.fail:
            raise ValueError("selected target failed")
        if emit:
            emit("text answer")
        return "text answer"


@pytest.fixture
def service():
    settings = Settings(litert_enabled=True, wyoming_port=0, minilm_enabled=False,
                        whisper_enabled=False)
    backend, llm = Backend(settings), LLM()
    with TestClient(create_app(settings, backend, llm)) as client:
        yield client, backend, llm


def test_catalogue_and_status(service):
    client, backend, _ = service
    assert HA_ASSIST_MODEL in [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert client.get("/health").json()["ha_assist"]["text_model"] == LLM_MODEL
    assert HA_ASSIST_MODEL in client.get("/ui/config").json()["vision_models"]


@pytest.mark.parametrize("text,tool_name,key,value", [
    ("Mach das Licht im Wohnzimmer aus", "intent__HassTurnOff", "area", "Wohnzimmer"),
    ("Mach das Licht im Wohnzimer aus", "intent__HassTurnOff", "area", "Wohnzimmer"),
    ("Stelle das Licht im Wohnzimmer auf 40 Prozent", "light__HassLightSet", "brightness", 40),
    ("Stelle den Rollladen im Wohnzimmer auf 60 Prozent", "intent__HassSetPosition", "position", 60),
    ("Stelle die Temperatur im Wohnzimmer auf 22 Grad", "climate__HassClimateSetTemperature", "temperature", 22),
])
def test_hassil_direct_calls_without_inference(service, text, tool_name, key, value):
    client, backend, llm = service
    result = client.post("/v1/chat/completions", json=payload(text)).json()
    assert result["model"] == HA_ASSIST_MODEL
    call = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == tool_name
    assert json.loads(call["arguments"])[key] == value
    assert not backend.calls and not llm.calls
    assert result["metrics"]["ha_route"]["inference_calls"] == 0
    assert result["usage"]["total_tokens"] == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", [LLM_MODEL, Settings().vlm_model])
def test_plain_requests_preserve_messages_and_tools_without_selection(service, model, stream):
    client, backend, llm = service
    body = payload(model=model, stream=stream)
    result = client.post("/v1/chat/completions", json=body)
    assert result.status_code == 200
    calls = llm.calls if model == LLM_MODEL else backend.calls
    assert len(calls) == 1 and not backend.selections
    assert calls[0].messages == body["messages"]
    assert calls[0].tools == body["tools"]
    assert not getattr(calls[0], "_ha_assist", False)
    assert not calls[0]._metrics.get("ha_route")


@pytest.mark.parametrize("stream", [False, True])
def test_general_text_uses_one_llm_and_retains_virtual_response_name(service, stream):
    client, backend, llm = service
    result = client.post("/v1/chat/completions", json=payload("Was ist die Hauptstadt von Frankreich?", stream=stream))
    assert result.status_code == 200
    assert len(llm.calls) == 1 and not backend.calls
    assert llm.calls[0].model == LLM_MODEL
    assert llm.calls[0].tools is None
    assert HA_ASSIST_MODEL in result.text
    assert llm.calls[0]._metrics["ha_route"]["route"] == "llm"


def test_image_bypasses_text_shortcuts_and_uses_one_vlm(service):
    client, backend, llm = service
    body = payload()
    body["messages"][-1]["content"] = [{"type": "text", "text": "Mach das Licht im Wohnzimmer aus"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    result = client.post("/v1/chat/completions", json=body).json()
    assert result["metrics"]["ha_route"]["route"] == "vlm"
    assert len(backend.calls) == 1 and not llm.calls and not backend.selections
    assert backend.calls[0].messages[-1] == body["messages"][-1]
    prompt = model_prompt(backend.calls[0])
    assert prompt[-1]["content"][-1] == {"type": "image"}


def test_text_target_can_be_native_hailo_llm():
    settings = Settings(hailo_llm_enabled=True, ha_assist_text_model="Qwen2.5-1.5B-Instruct",
                        wyoming_port=0, minilm_enabled=False, whisper_enabled=False)
    backend = Backend(settings)
    with TestClient(create_app(settings, backend)) as client:
        response = client.post("/v1/chat/completions", json=payload("Erkläre Relativität"))
        assert response.status_code == 200
        assert len(backend.calls) == 1
        assert backend.calls[0].model == settings.hailo_llm_model_id


def test_failed_llm_never_falls_back_to_vlm():
    settings = Settings(litert_enabled=True, wyoming_port=0, minilm_enabled=False)
    backend, llm = Backend(settings), LLM(fail=True)
    with TestClient(create_app(settings, backend, llm)) as client:
        result = client.post("/v1/chat/completions", json=payload("Erkläre Quantenphysik"))
        assert result.status_code == 400
        assert len(llm.calls) == 1 and not backend.calls


@pytest.mark.parametrize("settings", [Settings(ha_assist_enabled=False),
                                       Settings(ha_assist_text_model=Settings().vlm_model),
                                       Settings(ha_assist_text_model="unknown")])
def test_disabled_or_wrong_role_rejects_without_model_calls(settings):
    backend = Backend(settings)
    with TestClient(create_app(settings, backend)) as client:
        assert client.post("/v1/chat/completions", json=payload()).status_code == 400
        assert not backend.calls


@pytest.mark.parametrize("text", ["Schalte das Licht im Wohnzimmer nicht aus",
                                   "Mach das Licht im Wohnzimmer aus und die Heizung an",
                                   "Wenn es dunkel ist mach das Licht im Wohnzimmer an",
                                   "Schalte die Deckenlampe und Stehlampe aus",
                                   "Stelle das Licht im Wohnzimmer auf 140 Prozent"])
def test_negation_conditional_composite_and_out_of_range_do_not_match(text):
    response, trace = deterministic_intent(ChatRequest(**payload(text)), Settings(), "de")
    assert response is None, trace


def test_fuzzy_ambiguity_and_disabled_repair():
    assert not slot_repairs("schalte licht in Wohnzimer an", ["Wohnzimmer", "Wohnzimmr"])
    request = ChatRequest(**payload("Mach das Licht im Wohnzimer aus"))
    assert deterministic_intent(request, Settings(ha_assist_fuzzy_enabled=False), "de")[0] is None


def test_tool_choice_and_schema_constraints(service):
    client, backend, llm = service
    body = payload(tool_choice="none")
    response = client.post("/v1/chat/completions", json=body).json()
    assert not response["choices"][0]["message"].get("tool_calls")
    assert len(llm.calls) == 1 and not backend.calls
    body = payload()
    body["tools"] = [tool("intent__HassTurnOff", area={"type": "string", "enum": ["Küche"]})]
    assert deterministic_intent(ChatRequest(**body), Settings(), "de")[0] is None


def test_plain_output_is_not_repaired_as_ha_area_target():
    request = ChatRequest(**payload(model=LLM_MODEL))
    response = {"tool_calls": [{"function": {"name": "intent__HassTurnOff",
                                             "arguments": {"name": "Deckenlampe", "domain": ["light"]}}}]}
    call = response_message(response, request, "")["tool_calls"][0]
    assert json.loads(call["function"]["arguments"])["name"] == "Deckenlampe"
    object.__setattr__(request, "_ha_assist", True)
    call = response_message(response, request, "")["tool_calls"][0]
    assert json.loads(call["function"]["arguments"])["area"] == "Wohnzimmer"


def test_plain_litert_tool_followup_reaches_engine_instead_of_acknowledgement(monkeypatch):
    body = payload(model=LLM_MODEL)
    body["messages"].extend([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function",
         "function": {"name": "intent__HassTurnOn", "arguments": '{"name":"Deckenlampe"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": json.dumps({"response_type": "action_done",
         "data": {"success": [{"name": "Deckenlampe"}], "failed": []}})}])
    calls = []
    monkeypatch.setattr(LiteRTLMBackend, "_chat", lambda self, req, *args: calls.append(req) or "native")
    assert LiteRTLMBackend("/unused").chat(ChatRequest(**body)) == "native"
    assert len(calls) == 1


def test_tool_retrieval_stable_ties_and_preserves_history_schema():
    tools = [tool("zeta"), tool("alpha")]
    messages = [{"role": "user", "content": "area"}]
    selected, _ = retrieve_tools(messages, tools, max_tools=1)
    reversed_selected, _ = retrieve_tools(messages, list(reversed(tools)), max_tools=1)
    assert selected == reversed_selected
    messages += [{"role": "assistant", "tool_calls": [{"function": {"name": "zeta"}}]}]
    selected, _ = retrieve_tools(messages, tools, max_tools=1)
    assert any(t == tools[0] for t in selected)


def live_history(question, content):
    body = payload(question)
    body["tools"].append(tool("homeassistant__GetLiveContext"))
    body["messages"].extend([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "read", "type": "function",
         "function": {"name": "homeassistant__GetLiveContext", "arguments": '{"domain":["light"]}'}}]},
        {"role": "tool", "tool_call_id": "read", "content": json.dumps(content)}])
    return body


def test_live_state_followup_preserves_existing_deterministic_handling(service):
    client, backend, llm = service
    body = live_history("Ist die Deckenlampe an?", {"Deckenlampe": "on"})
    result = client.post("/v1/chat/completions", json=body).json()
    assert "Deckenlampe" in result["choices"][0]["message"]["content"]
    assert result["metrics"]["ha_route"]["inference_calls"] == 0
    assert not backend.calls and not llm.calls


def test_action_done_still_verifies_state_without_a_model(service):
    client, backend, llm = service
    body = payload()
    body["tools"].append(tool("homeassistant__GetLiveContext"))
    body["messages"].extend([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "action", "type": "function",
         "function": {"name": "intent__HassTurnOff", "arguments": '{"name":"Deckenlampe","domain":["light"]}'}}]},
        {"role": "tool", "tool_call_id": "action", "content": json.dumps({"response_type": "action_done",
         "data": {"success": [{"name": "Deckenlampe", "type": "entity", "id": "light.deckenlampe"}], "failed": []}})}])
    result = client.post("/v1/chat/completions", json=body).json()
    function = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert function["name"] == "homeassistant__GetLiveContext"
    assert not backend.calls and not llm.calls
    assert result["metrics"]["ha_verify_settle_ms"] == 400


def test_unmatched_tool_history_cannot_take_a_direct_action(service):
    client, backend, llm = service
    body = payload()
    body["messages"].insert(1, {"role": "assistant", "content": None, "tool_calls": [
        {"id": "pending", "type": "function", "function": {
            "name": "intent__HassTurnOn", "arguments": "{}"}}]})
    assert client.post("/v1/chat/completions", json=body).status_code == 400
    assert not backend.calls and not llm.calls


@pytest.mark.parametrize("stream", [False, True])
def test_image_in_history_keeps_same_vision_target(service, stream):
    client, backend, llm = service
    body = payload("Und was siehst du rechts?", stream=stream)
    body["messages"].insert(1, {"role": "user", "content": [
        {"type": "text", "text": "Betrachte dieses Bild"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]})
    body["messages"].insert(2, {"role": "assistant", "content": "Ein Raum"})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert len(backend.calls) == 1 and not llm.calls
    assert backend.calls[0].model == backend.settings.vlm_model
    assert "HA-Assist" in response.text


@pytest.mark.parametrize("language,area,query", [
    ("en", "Living Room", "turn off the lights in the living room"),
    ("ru", "гостиной", "выключи свет в гостиной"),
])
def test_official_grammar_uses_request_language_and_dynamic_areas(language, area, query):
    body = payload(query)
    body["messages"][0]["content"] = SYSTEM.replace("Wohnzimmer", area)
    result, trace = deterministic_intent(ChatRequest(**body), Settings(), language)
    assert result is not None, trace
    args = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert args["area"] == area
    assert trace["source"] == "hassil_exact"


def test_configuration_env_overrides_yaml_targets(tmp_path, monkeypatch):
    path = tmp_path / "service.yaml"
    path.write_text("settings:\n  ha_assist_text_model: gemma-4-E2B-it\n  ha_assist_fuzzy_enabled: true\n")
    monkeypatch.setenv("HAILO_CONFIG", str(path))
    monkeypatch.setenv("HAILO_HA_ASSIST_TEXT_MODEL", "Qwen3-1.7B-Instruct")
    monkeypatch.setenv("HAILO_HA_ASSIST_FUZZY_ENABLED", "false")
    settings = Settings.from_env()
    assert settings.ha_assist_text_model == "Qwen3-1.7B-Instruct"
    assert not settings.ha_assist_fuzzy_enabled
    with pytest.raises(ValueError, match="itself"):
        Settings(ha_assist_text_model=HA_ASSIST_MODEL)


def test_optional_image_tools_can_return_a_validated_description(service):
    from hailo_services.vlm_chat import tool_response

    client, backend, _ = service
    body = payload()
    body["messages"][-1]["content"] = [{"type": "text", "text": "Was siehst du?"},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    client.post("/v1/chat/completions", json=body)
    req = backend.calls[0]
    prompt = model_prompt(req)
    assert '{"content":"your answer"}' in prompt[0]["content"][0]["text"]
    assert tool_response('{"content":"Ein heller Raum."}', req) == "Ein heller Raum."
    required = req.model_copy(update={"tool_choice": "required"})
    with pytest.raises(ValueError, match="required tool call"):
        tool_response('{"content":"Ein heller Raum."}', required)
