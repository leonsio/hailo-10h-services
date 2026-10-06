"""Cross-catalogue recovery, diagnosis isolation and static-cache regressions."""

import copy
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_ha_assist import SYSTEM, TOOLS, payload
from test_ha_assist import service as service

from hailo_services.config import HA_ASSIST_MODEL, LLM_MODEL, Settings
from hailo_services.ha_fuzzy import slot_repairs
from hailo_services.ha_intents import deterministic_intent
from hailo_services.ha_recognition import _templates
from hailo_services.ha_request_plan import validate_action
from hailo_services.i18n import using_language
from hailo_services.schemas import ChatRequest
from hailo_services.tool_retrieval import _tool_index, retrieve_tools


@pytest.mark.parametrize(
    "text", ["Schalte das Licht in den Wohnzimmer aus", "Schalte das Licht imn Wohnzimmer aus"]
)
def test_recovery_uses_official_templates_and_preserves_action(service, text):
    client, backend, llm = service
    result = client.post("/v1/chat/completions", json=payload(text)).json()
    call = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "intent__HassTurnOff"
    assert json.loads(call["arguments"]) == {"area": "Wohnzimmer", "domain": ["light"]}
    assert result["metrics"]["ha_intent"]["source"] == "hassil_sentence_recovery"
    assert not backend.calls and not llm.calls


@pytest.mark.parametrize(
    "text",
    [
        "Schalte das Licht imn Wohnzimmer nicht aus",
        "Schalte das Licht imn Wohnzimmer aus und an",
        "Wenn es dunkel ist schalte das Licht imn Wohnzimmer aus",
        "Schalte das Licht imn Wohnzimmer auf 120 Prozent",
        "Schalte das Licht imn Wohnzimmer um 20 Prozent heller",
        "Schalte das Licht imn Wohnzimmer auz",
        "Erkläre warum das Licht imn Wohnzimmer aus ist",
    ],
)
def test_recovery_does_not_guess_protected_action_evidence(text):
    response, trace = deterministic_intent(ChatRequest(**payload(text)), Settings(), "de")
    assert response is None
    assert trace.get("source") != "hassil_sentence_recovery"


def test_sentence_recovery_can_be_disabled():
    response, trace = deterministic_intent(
        ChatRequest(**payload("Schalte das Licht imn Wohnzimmer aus")),
        Settings(ha_assist_sentence_fuzzy_enabled=False),
        "de",
    )
    assert response is None and "sentence_recovery" not in trace


@settings(max_examples=50, deadline=None)
@given(st.text(alphabet="abcdefghjkmnpqrstuvwxyz", min_size=9, max_size=20))
def test_unseen_catalogues_resolve_deletion_without_variant_lists(area):
    assert slot_repairs(f"Licht im {area[:-1]} aus", [area])[0]["resolved"] == area


def test_space_variation_and_transposition_are_catalogue_driven():
    assert slot_repairs("Licht im Wohn Zimmer aus", ["Wohnzimmer"])[0]["resolved"] == "Wohnzimmer"
    assert slot_repairs("Licht im Bibliotehk aus", ["Bibliothek"])[0]["resolved"] == "Bibliothek"
    assert not slot_repairs("Licht im RaumC aus", ["RaumA", "RaumB"])


def test_diagnosis_proposes_call_without_generation_or_execution(service):
    client, backend, llm = service
    result = client.post(
        "/v1/ha-assist/diagnose", json=payload("Schalte das Licht imn Wohnzimmer aus")
    ).json()
    assert result["object"] == "ha_assist.diagnosis"
    assert result["generative_calls"] == result["tools_executed"] == 0
    assert result["proposed_response"]["tool_calls"]
    assert result["metrics"]["ha_route"]["would_inference_calls"] == 0
    assert result["metrics"]["ha_stages_ms"]["intent_recognition"] >= 0
    assert not backend.calls and not llm.calls


def test_ambiguous_diagnosis_reports_compact_prompt_without_llm(service):
    client, backend, llm = service
    body = payload("Schalte das Licht im ArbeitsraumC auf 70%")
    body["messages"][0]["content"] = (
        SYSTEM.replace("Wohnzimmer", "ArbeitsraumA")
        + """
- names: Leuchte B
  domain: light
  areas: ArbeitsraumB
- names: Fremde Lampe
  domain: light
  areas: Garten
"""
    )
    result = client.post("/v1/ha-assist/diagnose", json=body).json()
    assert result["generative_calls"] == 0
    assert result["metrics"]["ha_route"]["would_inference_calls"] == 1
    assert len(result["metrics"]["ha_plan"]["target_candidates"]) == 2
    assert result["prompt_stage"] == "ha_prepared_before_native_template_and_budget"
    assert "Fremde Lampe" not in json.dumps(result["prepared_request"])
    assert not backend.calls and not llm.calls


@pytest.mark.parametrize("model,stream", [(LLM_MODEL, False), (HA_ASSIST_MODEL, True)])
def test_diagnosis_rejects_physical_model_or_streaming(service, model, stream):
    client, _, llm = service
    assert (
        client.post("/v1/ha-assist/diagnose", json=payload(model=model, stream=stream)).status_code
        == 400
    )
    assert not llm.calls


def metadata():
    return {
        "version": "v1",
        "entities": [
            {
                "entity_id": "light.reading",
                "name": "Leselampe",
                "domain": "light",
                "area": "Bibliothek",
                "aliases": ["Sofalicht"],
                "area_aliases": ["Leseraum"],
                "floor": "Obergeschoss",
                "capabilities": ["brightness"],
            }
        ],
    }


@pytest.mark.parametrize(
    "text,key,value",
    [
        ("Schalte Sofalicht aus", "name", "Leselampe"),
        ("Schalte das Licht im Leseraum aus", "area", "Bibliothek"),
    ],
)
def test_structured_exposed_aliases_use_canonical_tool_targets(service, text, key, value):
    client, backend, llm = service
    result = client.post("/v1/chat/completions", json=payload(text, ha_context=metadata())).json()
    function = result["choices"][0]["message"]["tool_calls"][0]["function"]
    assert json.loads(function["arguments"])[key] == value
    assert result["metrics"]["ha_catalogue"]["version"] == "v1"
    assert not llm.calls and not backend.calls


def test_catalogue_content_changes_invalidate_even_with_same_version(service):
    client, _, llm = service
    context = metadata()
    for area in ("Bibliothek", "Werkstatt"):
        context["entities"][0]["area"] = area
        body = payload(f"Schalte das Licht im {area} aus", ha_context=context)
        result = client.post("/v1/chat/completions", json=body).json()
        assert (
            json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])[
                "area"
            ]
            == area
        )
    assert not llm.calls


def test_direct_and_generated_actions_preserve_named_target_and_capabilities(service):
    from hailo_services.ha_catalogue import prepare_catalogue
    from hailo_services.ha_request_plan import canonical_request

    request = ChatRequest(
        **payload("Schalte die Leselampe im Bibliothek auf 70%", ha_context=metadata())
    )
    object.__setattr__(request, "_ha_assist", True)
    request = canonical_request(prepare_catalogue(request), Settings())
    assert validate_action(request, "light__HassLightSet", {"name": "Leselampe", "brightness": 70})
    assert not validate_action(
        request, "light__HassLightSet", {"area": "Bibliothek", "brightness": 70}
    )
    request._ha_catalogue[0]["capabilities"] = []
    assert not validate_action(
        request, "light__HassLightSet", {"name": "Leselampe", "brightness": 70}
    )


def test_physical_model_ignores_ha_metadata(service):
    client, backend, llm = service
    body = payload("Schalte Sofalicht aus", model=LLM_MODEL, ha_context=metadata())
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert len(llm.calls) == 1
    assert llm.calls[0].messages == body["messages"]
    assert not getattr(llm.calls[0], "_ha_plan", None) and not backend.selections


def test_tool_index_cache_is_bounded_localized_and_schema_sensitive():
    _tool_index.cache_clear()
    tools = copy.deepcopy(TOOLS)
    with using_language("de"):
        first, _ = retrieve_tools(payload()["messages"], tools)
        second, stats = retrieve_tools(payload()["messages"], tools)
        assert stats["tool_index_cache_hit"] and first == second
        tools[0]["function"]["parameters"]["properties"]["area"]["enum"] = ["Werkstatt"]
        retrieve_tools(payload()["messages"], tools)
    assert _tool_index.cache_info().misses == 2
    with using_language("en"):
        retrieve_tools(payload()["messages"], tools)
    assert _tool_index.cache_info().misses == 3
    for index in range(70):
        tools[0]["function"]["description"] = f"changed {index}"
        retrieve_tools(payload()["messages"], tools)
    assert _tool_index.cache_info().currsize <= 64
    assert _templates.cache_info().maxsize == 32


@pytest.mark.parametrize("qualifier", ["nihct", "not", "wenn", "und", "um 20 Prozent heller"])
def test_long_target_cannot_hide_qualifiers_from_sentence_score(qualifier):
    from hailo_services.ha_intents import _grammar
    from hailo_services.ha_recognition import template_candidates

    area = "ArbeitszimmerMitEinemExtremLangenIndividuellKonfiguriertenNamen"
    text = f"Schalte das Licht imn {area} {qualifier} aus"
    candidates, _ = template_candidates(
        text,
        _grammar("de", ("HassTurnOff", "HassTurnOn")),
        "de",
        [{"name": "Lampe", "area": area, "domain": "light"}],
    )
    assert not candidates


def test_known_unsupported_brightness_does_not_reach_llm(service):
    client, backend, llm = service
    context = metadata()
    context["entities"][0]["capabilities"] = []
    result = client.post(
        "/v1/chat/completions",
        json=payload("Schalte das Licht im Bibliothek auf 70%", ha_context=context),
    ).json()
    assert "tool_calls" not in result["choices"][0]["message"]
    assert result["metrics"]["ha_plan"]["unsupported_property"] == "brightness"
    assert not llm.calls and not backend.calls


def test_exact_catalogue_span_is_never_replaced_with_neighbor_action_words():
    area = "a" * 20
    assert not slot_repairs(f"Licht im {area} aus", [area])


def test_cache_status_reports_apis_without_using_them():
    from types import SimpleNamespace

    from hailo_services.diagnostics import cache_status

    def forbidden():
        raise AssertionError("native context must not be probed or changed")

    backend = SimpleNamespace(
        llm=SimpleNamespace(save_context=forbidden, load_context=forbidden), vlm=None
    )
    status = cache_status(backend, None)
    assert status["hailo"]["llm"]["context_snapshot_api"]
    assert not status["hailo"]["llm"]["service_snapshot_reuse"]
    assert not status["live_state_or_action_response_cache"]


def test_ambiguous_area_capability_check_uses_the_chosen_candidate(service):
    client, _, llm = service
    context = metadata()
    other = copy.deepcopy(context["entities"][0])
    other.update(entity_id="light.other", name="Andere Lampe", area="Werkstatt", capabilities=[])
    context["entities"].append(other)
    result = client.post(
        "/v1/ha-assist/diagnose",
        json=payload("Schalte das Licht im Leseraum auf 70%", ha_context=context),
    ).json()
    assert result["metrics"]["ha_plan"]["target_resolution"] == "llm"
    assert not llm.calls
    from hailo_services.ha_catalogue import prepare_catalogue
    from hailo_services.ha_request_plan import canonical_request

    request = ChatRequest(**payload("Schalte das Licht im Leseraum auf 70%", ha_context=context))
    object.__setattr__(request, "_ha_assist", True)
    request = canonical_request(prepare_catalogue(request), Settings())
    assert validate_action(request, "light__HassLightSet", {"area": "Bibliothek", "brightness": 70})
    assert not validate_action(
        request, "light__HassLightSet", {"area": "Werkstatt", "brightness": 70}
    )
    assert not validate_action(
        request, "light__HassLightSet", {"area": "Erfunden", "brightness": 70}
    )


def test_explicit_empty_exposed_catalogue_rejects_actions():
    from hailo_services.ha_catalogue import prepare_catalogue
    from hailo_services.ha_request_plan import canonical_request

    request = ChatRequest(**payload(ha_context={"entities": []}))
    object.__setattr__(request, "_ha_assist", True)
    request = canonical_request(prepare_catalogue(request), Settings())
    assert not validate_action(request, "intent__HassTurnOff", {"area": "Wohnzimmer"})
