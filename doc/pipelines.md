# Request pipelines

## Boundary

`ha_pipeline.py` is the outer request boundary. A known HA function declaration
or HA function-call history identifies an HA request. User words such as “light”
or “temperature” alone do not identify the client. `tool_choice="none"` bypasses
HA routing; an explicitly forced tool remains authoritative. Unrelated custom
tools do not activate the HA pipeline.

Plain OpenAI text/image requests retain their system messages, user text,
history, tools and schemas. They do not run HA entity/tool retrieval, the HA
prompt compiler, direct HA actions, or HA wait messages. Standard native
function validation, token budgets and model selection still apply.

## HA sequence

1. Choose request-local language; canonical vocabulary is used only for matching.
2. Try an absolute numeric action with an unambiguous exposed entity/area.
3. Apply weather/measurement, state lookup and action-result handling.
4. Use lexical analysis and MiniLM to reduce relevant HA context and tools.
5. Compile a minimal task prompt/schema when inference remains necessary.
6. At actual Gemma inference start, optionally send a localized SSE wait delta.
7. Validate the completed model response before exposing a tool call to HA.

| Request | Deterministic path |
|---|---|
| Switch a named device or an unambiguous area on/off | Validated HA action tool |
| Set area lights to an absolute percentage | `HassLightSet`, brightness 0–100 |
| Set area covers to an absolute percentage | `HassSetPosition`, position 0–100 |
| Ask state, count/list devices in a state | `GetLiveContext`, then direct formatting |
| Ask temperature/humidity | Live measurement lookup and interpretation |
| Ask current outdoor weather | HA weather/environment metadata, direct if clear |
| Successful action result | HA speech or localized acknowledgement |
| Ambiguous, relative or composite task | Compact Gemma request |

A numeric fast path requires exactly one whole percentage, an explicit action,
a known target, and a compatible supplied schema. Relative changes, negation,
conditional/composite instructions, missing targets and invalid values defer to
Gemma. Direct calls are validated against the original client schema. No direct
path invents an area or rewrites an entity name in outgoing arguments.

Ambient measurements rank HA domains, areas, device classes, units and generic
environment/process vocabulary. There are no vendor/model-specific selectors.
`climate.temperature` is a setpoint, never an ambient measurement;
`current_temperature` is the measurement. An unavailable/unknown thermostat
cannot supply it. A controller in `off` may still measure temperature, so `off`
alone is not an availability or freshness signal. Missing `current_temperature`
never falls back to the setpoint. Ambiguous data
remains an LLM decision; unavailable data produces a localized explanation.
Freshness depends on the availability/measurement metadata supplied by HA; a
numeric value without freshness information cannot prove when a sensor last ran.

Weather lookups include both weather providers and environmental sensors, so an
unavailable provider does not hide a working sensor. Multiple available providers
or equally plausible measurements defer to Gemma with their measurement attributes;
catalogue order never chooses between them.

## Streaming and diagnostics

`HAILO_HA_WAIT_MESSAGES=true` enables varying wait sentences. The callback is
invoked at native inference, after queueing/preparation/input-budget validation;
fast paths never trigger it. It is one text delta in the existing SSE response,
not another HTTP response. HA may include it in the same assistant turn, and
clients that buffer speech will only speak after completion. Disable it for
clients where this is undesirable. Native tool calls remain fully buffered.

`ha_route`, `ha_weather_route`, `ha_prompt_plan`, `gemma_rendered_prompt` and
`gemma_timing` explain routing, context removal, actual prompt tokens and Gemma
time. Enable `HAILO_DEBUG_LOG` for full detail; debug prompts can contain private
conversation/device data. `max_input_tokens=4096` remains independent of the
incoming HA catalogue's size.
