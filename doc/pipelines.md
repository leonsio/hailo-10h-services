# Request pipelines

For the complete rationale and worked examples, see
[How HA-Assist works](ha-assist.md).

## Boundary

`ha_assist.py` is the virtual-model entry point. Only `model="HA-Assist"`
activates HA processing. The resolved physical target carries an internal marker,
which scopes the existing `ha_pipeline.py` and output normalization hooks.
Known HA declarations/history select HA handlers within that boundary; HA-like
words or HA tools alone never activate it for a physical model request.
`tool_choice="none"` bypasses HA tools; a forced tool remains authoritative.

Plain OpenAI text/image requests retain their system messages, user text,
history, tools and schemas. They do not run HA entity/tool retrieval, the HA
prompt compiler, direct HA actions, or HA wait messages. Standard native
function validation, token budgets and model selection still apply.

## HailoRT 5.4.0 backend constraint

The HA-Assist routing layer has separate text and vision targets, but HailoRT/GenAI
5.4.0 does **not** support a native Hailo LLM and Hailo VLM resident on the same
device at the same time. Configure either `models.vlm` or `models.hailo_llm`.

For simultaneous HA text and image handling today, use a Hailo VLM plus Gemma on
CPU, for example:

```yaml
settings:
  ha_assist_text_model: gemma-4-E2B-it
  ha_assist_vision_model: Qwen2-VL-2B-Instruct
models:
  vlm: {enabled: true, model: Qwen2-VL-2B-Instruct}
  hailo_llm: {enabled: false}
  gemma: {enabled: true, max_input_tokens: 4096}
```

A native Hailo LLM may instead be the text target with the VLM disabled. The
software keeps both adapters and routing paths so the two Hailo backends can run
in parallel when a future Hailo runtime supports that combination. Models are not
unloaded/reloaded per request.

## HA sequence

1. Resolve the configured target: images → configured VLM, text → configured text
   backend. There is no complexity-based model switching.
2. Choose request-local language; validate tool call/result dependencies.
3. For text, try official HassIL exact intent/slots using request entity/area lists.
4. On miss, try one conservative fuzzy slot correction and reparse with HassIL.
5. Validate unique direct calls against the client's original schema/choice.
6. Retain weather/measurement, live state and action-result verification paths.
7. Use lexical name/keyword scoring and corpus-frequency weighting, then MiniLM
   ranking to reduce relevant HA context/tools; preserve schemas needed by history.
8. Compile a minimal prompt when inference remains necessary, then apply the
   selected backend's existing native token budget (Gemma remains capped at 4096
   input tokens by default).
9. Invoke exactly one generative target. Image requests bypass all text shortcuts.
10. Validate model tool output and return `model="HA-Assist"`. HA executes tools.

A deterministic answer makes zero generative calls. Failed, unavailable or
oversized requests never switch to another model. The target settings reference
enabled local backends, not remote endpoints.

| Request | Deterministic path |
|---|---|
| Switch a named device or an unambiguous area on/off | Validated HA action tool |
| Set area lights to an absolute percentage | `HassLightSet`, brightness 0–100 |
| Set area covers to an absolute percentage | `HassSetPosition`, position 0–100 |
| Ask state, count/list devices in a state | `GetLiveContext`, then direct formatting |
| Ask temperature/humidity | Live measurement lookup and interpretation |
| Ask current outdoor weather | HA weather/environment metadata, direct if clear |
| Successful action result | HA speech or localized acknowledgement |
| Ambiguous, relative or composite task | Compact request to configured text backend |

A numeric fast path requires exactly one whole percentage, an explicit action,
a known target, and a compatible supplied schema. Relative changes, negation,
conditional/composite instructions, missing targets and invalid values defer to
the configured text target. Direct calls are validated against the original client
schema. No direct path invents an area or rewrites an entity name in outgoing
arguments.

Ambient measurements rank HA domains, areas, device classes, units and generic
environment/process vocabulary. There are no vendor/model-specific selectors.
`climate.temperature` is a setpoint, never an ambient measurement;
`current_temperature` is the measurement. An unavailable/unknown thermostat
cannot supply it. A controller in `off` may still measure temperature, so `off`
alone is not an availability or freshness signal. Missing `current_temperature`
never falls back to the setpoint. Ambiguous data remains a generative-backend
decision; unavailable data produces a localized explanation. Freshness depends on
the availability/measurement metadata supplied by HA; a numeric value without
freshness information cannot prove when a sensor last ran.

Weather lookups include both weather providers and environmental sensors, so an
unavailable provider does not hide a working sensor. Multiple available providers
or equally plausible measurements defer to the configured text target with their
measurement attributes; catalogue order never chooses between them.

## Streaming and diagnostics

`HAILO_HA_WAIT_MESSAGES=true` enables varying wait sentences. The callback is
invoked at native inference, after queueing/preparation/input-budget validation;
fast paths never trigger it. It is one text delta in the existing SSE response,
not another HTTP response. HA may include it in the same assistant turn, and
clients that buffer speech will only speak after completion. Disable it for
clients where this is undesirable. Native tool calls remain fully buffered.

`ha_assist`, `ha_intent`, `ha_route`, `ha_weather_route`, `ha_prompt_plan`,
`gemma_rendered_prompt` and `gemma_timing` explain routing, context removal, actual
prompt tokens and Gemma time. Enable `HAILO_DEBUG_LOG` for full detail; debug
prompts can contain private conversation/device data. `max_input_tokens=4096`
remains independent of the incoming HA catalogue's size.

The non-streaming response includes `metrics.ha_route` (selected physical backend,
route, zero/one inference calls, preparation duration and language) and
`metrics.ha_intent` (exact/fuzzy stage, candidate intents/slots and elapsed time).
Fuzzy repair never changes action vocabulary and requires a unique candidate
above the configured threshold/margin. Original outgoing names remain unchanged.
The official grammar package is pinned for reproducible matching; local HassIL
supplements retain a few existing service phrases for absolute brightness.
