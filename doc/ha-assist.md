# How HA-Assist works

`HA-Assist` is a **virtual OpenAI-compatible model** implemented by
Hailo-10H-Services. It has no model weights of its own. Its purpose is to make Home
Assistant requests smaller, faster, safer and more deterministic before a generative
model is considered.

The central rule is:

> **A request is handled deterministically when the answer or action can be derived
> reliably from Home Assistant context. Otherwise HA-Assist performs exactly one
> generative call to the configured text or vision backend.**

This is deliberately different from forwarding the complete Home Assistant request,
all entities and every tool schema directly to an LLM.

## Why not send everything directly to an LLM?

Home Assistant conversation requests can be very large. They may contain:

- a long system prompt;
- hundreds of exposed entities and areas;
- current states and attributes;
- many function/tool schemas;
- previous assistant/tool-call/tool-result history;
- the current user request.

That creates several problems for small local models.

### 1. Context limits

Current Hailo GenAI LLM/VLM HEFs in the model catalog have a **2048-token compiled
context**. Gemma is configured with a larger LiteRT context, but the service still uses
a conservative default `max_input_tokens` of 4096 to protect RAM and latency.

A raw Home Assistant request can exceed these limits before the actual user question is
even considered. Simply truncating the end or removing arbitrary messages can break
function-call dependencies or discard the entity that the user actually meant.

HA-Assist therefore reduces context **before** the final model prompt is rendered.

### 2. Latency

A request such as:

```text
Turn off the living-room light.
```

should not require several seconds of LLM inference when Home Assistant has already
supplied the exact area/entity catalogue and a compatible `HassTurnOff` tool.

For deterministic requests, HA-Assist returns the validated action directly and makes
**zero generative calls**.

### 3. Reliability and hallucinations

An LLM can choose the wrong entity, slightly alter a device name, invent an area or
return arguments that do not match the tool schema. Those errors are particularly
undesirable for home-control actions.

HA-Assist prefers exact catalogue data and validates direct and generated actions
against the tools actually supplied by Home Assistant.

### 4. Most HA requests are structured, not open-ended

Many common Home Assistant requests are better treated as structured data operations:

- turn a known device or area on/off;
- set an absolute brightness or cover position;
- ask whether a device is on;
- count or list devices in a state;
- read temperature/humidity or current weather;
- acknowledge the result of an action.

These tasks do not need open-ended reasoning when the required data is already present
and unambiguous.

### 5. A smaller prompt improves the requests that really do need an LLM

When a request is ambiguous, relative, conditional, composite or simply a general
knowledge question, HA-Assist still uses a generative backend. The difference is that
it forwards a compact task-specific prompt and only the relevant tools/context instead
of the entire HA catalogue.

## Processing overview

```text
Home Assistant
      │
      │ OpenAI /v1/chat/completions
      │ model = HA-Assist
      ▼
┌───────────────────────────────┐
│ 1. Virtual-model boundary     │
│    + request validation       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 2. Detect request type        │
│    text / image / tool result │
└───────────────┬───────────────┘
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
   text request      image request
       │                 │
       ▼                 │
┌─────────────────┐      │
│ 3. Deterministic│      │
│    HA processing│      │
│ HassIL / states │      │
│ measurements    │      │
│ action results  │      │
└────────┬────────┘      │
         │               │
   solved?               │
    │    │               │
  yes    no              │
    │    │               │
    │    ▼               │
    │ ┌──────────────────────────┐
    │ │ 4. Context/tool retrieval│
    │ │ lexical + MiniLM ranking │
    │ │ history dependency guard │
    │ └────────────┬─────────────┘
    │              │
    │              ▼
    │ ┌──────────────────────────┐
    │ │ 5. Minimal prompt compile│
    │ │ + native token budgeting │
    │ └────────────┬─────────────┘
    │              │
    │       ┌──────┴─────────┐
    │       │                │
    │       ▼                ▼
    │   text backend     vision backend
    │   Gemma / Hailo    configured VLM
    │   native LLM            │
    │       │                 │
    │       └────────┬────────┘
    │                ▼
    │ ┌──────────────────────────┐
    │ │ 6. Validate model output │
    │ │ schema / target / value  │
    │ └────────────┬─────────────┘
    │              │
    └──────────────┴──────────────► Home Assistant
                                    executes tools
```

## Step 1: the HA-Assist boundary

HA-specific processing is activated **only** when the request explicitly uses:

```json
{"model": "HA-Assist"}
```

A request sent directly to `gemma-4-E2B-it`, `Qwen2-VL-2B-Instruct` or a native Hailo
LLM is treated as an ordinary physical-model request. HA-like words or tool names do
not silently activate the virtual model.

This boundary makes behavior predictable and lets the same physical models also serve
normal non-Home-Assistant requests.

## Step 2: select text or vision routing

If the request contains an image, HA-Assist selects
`settings.ha_assist_vision_model`. A text-only shortcut is never allowed to answer a
question about an image.

Text-only requests select `settings.ha_assist_text_model` when generation is necessary.
Before that happens, deterministic HA paths are attempted.

### HailoRT 5.4.0 model limitation

With HailoRT/GenAI **5.4.0**, a native Hailo LLM and a Hailo VLM cannot be resident at
the same time. Hailo's current GenAI documentation states that multiple LLM or VLM
models on one device are not supported.

The recommended HA-Assist text+image profile is therefore:

```yaml
settings:
  ha_assist_text_model: gemma-4-E2B-it
  ha_assist_vision_model: Qwen2-VL-2B-Instruct

models:
  vlm:
    enabled: true
    model: Qwen2-VL-2B-Instruct
  hailo_llm:
    enabled: false
  gemma:
    enabled: true
    max_input_tokens: 4096
```

Gemma runs on the CPU, so the Hailo VLM can remain resident for image requests.

A native Hailo LLM can instead be selected as the text backend:

```yaml
settings:
  ha_assist_text_model: Qwen3-1.7B-Instruct

models:
  vlm:
    enabled: false
  hailo_llm:
    enabled: true
    model: Qwen3-1.7B-Instruct
```

That profile is text-only from HA-Assist's perspective because no VLM is resident.

The service is already architecturally prepared for separate text and vision targets.
When Hailo supports simultaneous native VLM+LLM operation, both adapters can be used
without redesigning the HA-Assist routing layer. The current service does **not** swap
HEFs between requests to work around the 5.4.0 limitation.

## Step 3: deterministic Home Assistant processing

For text requests, HA-Assist first checks whether the operation can be answered or
expressed as a validated tool call without a generative model.

### Exact HassIL intent matching

The service uses the packaged Home Assistant intent grammar for supported languages.
Known entity and area names supplied in the request are used as slots.

Examples that can be deterministic when the target is unambiguous:

```text
Turn off the living-room light.
Set the living-room lights to 70 percent.
Set the bedroom blind to 40 percent.
```

The generated call must still match the original function schema supplied by Home
Assistant. HA-Assist does not invent a new tool.

### Conservative fuzzy repair

If exact matching fails because one known entity/area slot is slightly misspelled,
HA-Assist may repair **one** catalogue slot when there is a unique candidate above the
configured threshold and margin. It then reparses the sentence with HassIL.

The fuzzy stage does not rewrite action vocabulary and is intentionally conservative.
Ambiguous matches fall through to normal inference rather than guessing.

### State and measurement queries

Live state, device-count/list, temperature/humidity and weather paths can be answered
deterministically when the supplied Home Assistant data identifies one clear result.

Important safeguards include:

- a thermostat setpoint is not treated as ambient temperature;
- unavailable/unknown devices are not presented as measurements;
- multiple equally plausible measurements are not resolved by catalogue order;
- missing freshness metadata is not invented.

If the data is ambiguous, the request can proceed to the configured text backend with
the relevant context.

### Action-result verification

When Home Assistant sends back a tool result, HA-Assist can verify/format known results
and acknowledgements without a second generative call. Active tool-call/result
relationships are preserved while older unrelated history may be removed.

## Step 4: select only relevant tools and context

If no deterministic answer is available, HA-Assist reduces the model input.

The selection pipeline uses several signals rather than sending the full catalogue:

1. exact entity/area/domain names;
2. lexical keyword relevance;
3. domain/attribute and corpus-frequency weighting;
4. MiniLM semantic ranking when available;
5. mandatory preservation of schemas referenced by active tool history.

MiniLM is a **retrieval model**, not an answer-generating model. Its job is to rank
which Home Assistant entities/tools are relevant to the current turn.

This is important because an incoming HA request can be much larger than the final LLM
context while still containing only a few entities relevant to the user's question.

## Step 5: compile a minimal model prompt

After retrieval, HA-Assist builds a task-specific prompt instead of forwarding large
generic Home Assistant examples.

The final prompt keeps the information required for the active task and preserves:

- system requirements that must survive compaction;
- the current user turn;
- active assistant tool calls;
- matching tool results;
- selected relevant schemas/context.

Older completed house-control turns can be removed from the backend prompt while the
client can continue displaying the complete conversation.

The selected backend then applies its own native tokenizer and token budget. If the
required prompt still does not fit, the request fails with an input-limit error; it is
not silently damaged by arbitrary truncation.

## Step 6: at most one generative backend call

After deterministic processing and context reduction, HA-Assist makes either:

- **zero** generative calls when the result is deterministic; or
- **one** call to the configured text or vision backend.

There is no complexity-based chain such as:

```text
small model → failed → larger model → VLM → retry
```

and there is no hidden fallback from one physical model to another after a generation
failure. This keeps latency and behavior observable.

### Requests that usually need generation

Examples include:

```text
Make the living room more comfortable.
If it gets too warm later, do something sensible.
Turn off everything downstairs except what we still need.
What is the capital of France?
Explain why my office is warmer than the bedroom.
```

The first three are ambiguous/conditional/composite Home Assistant tasks. The fourth is
a normal general-knowledge question. The fifth may require reasoning over selected live
state/measurement context.

## Step 7: validate generated actions

A generative model is not trusted to execute an action directly.

Generated function calls are checked against:

- the tool names actually offered by Home Assistant;
- the corresponding JSON Schema;
- `tool_choice` requirements;
- parallel-call policy;
- resolved HA target/value expectations when a deterministic plan exists.

Invalid or incomplete calls are rejected rather than executed.

The service returns validated `tool_calls` to Home Assistant. **Home Assistant executes
the action using its own permissions.** Hailo-10H-Services does not need an HA access
token and does not call HA services directly.

## Example request paths

| User request | HA-Assist path | Generative calls |
|---|---|---:|
| `Turn off the living-room light` | HassIL → schema-validated tool call | 0 |
| `Set the living-room lights to 70%` | absolute-value fast path → validated tool | 0 |
| `Is the kitchen light on?` | live-state selection → direct answer | 0 when unambiguous |
| `What is the living-room temperature?` | measurement selection → direct answer | 0 when unambiguous |
| successful prior action result | deterministic result verification/acknowledgement | 0 when recognized |
| `Make the living room more comfortable` | retrieve context/tools → minimal prompt → text backend | 1 |
| `What is the capital of France?` | minimal general text request → text backend | 1 |
| image + `What can you see?` | image path → configured VLM | 1 |

## History handling

HA-Assist distinguishes between useful conversational history and completed house-action
history.

A current tool round must remain structurally valid:

```text
assistant tool_call
        ↓
tool result
```

Those dependencies are preserved together. Completed older house-control turns can be
removed from the backend prompt when they are no longer needed. General conversational
history can remain when it is relevant to a follow-up question.

The visible client conversation and the final backend prompt are therefore not
necessarily identical.

## Diagnostics and metrics

Non-streaming HA-Assist responses expose diagnostics in `metrics` when applicable:

| Metric | Purpose |
|---|---|
| `ha_route` | direct/LLM/VLM route, physical backend, preparation time, inference-call count |
| `ha_history` | history filtering policy and message counts |
| `ha_plan` | canonicalized request, catalogue corrections, resolved area/domain/value |
| `ha_intent` | HassIL exact/fuzzy candidates, slots and timing |
| `ha_validation` | generated action validation results |

A deterministic route reports zero generative inference work rather than inventing LLM
token usage.

With debug logging enabled, additional routing, prompt-planning and rendered-prompt
information is logged. Debug output can contain private device names, states and
conversation text and should be handled accordingly.

## Design goals

HA-Assist is designed around the following priorities:

1. **Determinism first** for operations that are already structurally known.
2. **Do not guess targets** when Home Assistant data is ambiguous.
3. **Reduce context before generation**, rather than truncating arbitrary data later.
4. **Use the LLM for language/reasoning**, not for work that can be done reliably with
   structured state and schemas.
5. **Validate generated actions** before exposing them to Home Assistant.
6. **At most one generative backend call** per HA-Assist request.
7. **Keep physical model selection explicit and observable**.
8. **Do not unload/reload Hailo VLM/LLM models per request**.

This architecture is especially useful with small local models because it spends their
limited context and inference time only on the part of the request that actually needs
generative reasoning.

## Related documentation

- [README / feature overview](../README.md#home-assistant-virtual-model)
- [API and Home Assistant configuration](api.md#home-assistant-virtual-model)
- [Request pipeline implementation notes](pipelines.md)
- [Model evaluation](model-benchmark-evaluation.md)
- [Troubleshooting](troubleshooting.md)

Hailo's current GenAI concurrency limitation is documented in the upstream
[Hailo Model Zoo GenAI usage guide](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/USAGE.rst).
