# Frigate-Assist: experimental chat and description proxy

`Frigate-Assist` is a virtual OpenAI-compatible model at `/v1/chat/completions`.
It has no model weights of its own. It prepares Frigate requests before routing
text to the configured **LLM or VLM** (default: the selected resident VLM), or images to
the configured resident **Hailo VLM**.
It never swaps resident models and does not change the preparation of requests
addressed directly to Gemma, Qwen, or `HA-Assist`.

**Chat is experimental and only conditionally usable.** Frigate chat is optional.
Small models can misunderstand language, choose poor filters, or hallucinate image
details. The proxy reduces unnecessary context and rejects invalid calls; it cannot
guarantee factual model output. Use Frigate's regular views/settings when correctness
matters, and inspect actual images and tool results when evaluating an answer.

Frigate versions supporting named GenAI providers can assign `descriptions` and
`chat` to different providers/models. This implementation attempts to combine both
roles behind `Frigate-Assist`. It is **not an embeddings model**. The examples below
target the named-provider/roles configuration; check your installed Frigate version's
schema if it still uses an older configuration layout.

## Configuration

Enable the local CPU Gemma backend and the resident Hailo VLM:

```yaml
settings:
  frigate_assist_enabled: true
  frigate_assist_text_model: gemma-4-E2B-it
  frigate_assist_vision_model: Qwen2-VL-2B-Instruct
  frigate_assist_max_events: 12
  frigate_assist_text_chars: 10000
  frigate_assist_vision_chars: 1200
  frigate_assist_vision_max_tokens: 128
  frigate_assist_away_profiles: "away,abwesend"

models:
  gemma:
    enabled: true
    path: /var/lib/hailo-10h-services/models/gemma-4-E2B-it.litertlm
    max_input_tokens: 4096
  vlm:
    enabled: true
    model: Qwen2-VL-2B-Instruct
    max_input_tokens: 2048
  hailo_llm:
    enabled: false
```

Use the path of your existing Gemma artifact. Restart the service after changing
settings. The equivalent environment names are `HAILO_FRIGATE_ASSIST_ENABLED`,
`HAILO_FRIGATE_ASSIST_TEXT_MODEL`,
`HAILO_FRIGATE_ASSIST_VISION_MODEL`, `HAILO_FRIGATE_ASSIST_MAX_EVENTS`,
`HAILO_FRIGATE_ASSIST_TEXT_CHARS`, `HAILO_FRIGATE_ASSIST_VISION_CHARS` and
`HAILO_FRIGATE_ASSIST_AWAY_PROFILES`.

Frigate combined-role example:

```yaml
genai:
  hailo_assist:
    provider: openai
    base_url: http://HAILO_HOST:8090/v1
    api_key: "{FRIGATE_HAILO_API_KEY}"
    model: Frigate-Assist
    roles:
      - descriptions
      - chat
```

The base URL includes `/v1`; do not append `/chat/completions`. Keep an existing
embeddings provider or Frigate's local semantic-search model. No `context_size`
override is required for this proxy example; see the distinction below. The service
applies the real selected backend's token limit after compilation. Enable
object/review description generation separately in Frigate;
assigning a provider role does not enable every description feature automatically.

To try chat separately while keeping a working description provider:

```yaml
genai:
  existing_descriptions:
    provider: openai
    base_url: http://HAILO_HOST:8090/v1
    api_key: "{FRIGATE_HAILO_API_KEY}"
    model: Qwen2-VL-2B-Instruct
    roles:
      - descriptions
  experimental_chat:
    provider: openai
    base_url: http://HAILO_HOST:8090/v1
    api_key: "{FRIGATE_HAILO_API_KEY}"
    model: Frigate-Assist
    roles:
      - chat
```

Each role must belong to only one provider. Chat may be omitted altogether.
Direct Qwen requests retain their existing API behavior; the Frigate-specific
`stream_options` extension described below is accepted only by `Frigate-Assist`.

### Choosing the text and vision backends

Both `frigate_assist_text_model` and `frigate_assist_vision_model` default to
empty selections. If omitted, empty, whitespace-only or YAML null, each resolves
to the native VLM selected in `models.vlm.model`. This applies even when Gemma is
enabled: choosing Gemma requires `frigate_assist_text_model: gemma-4-E2B-it` explicitly.
HA-Assist's defaults are unchanged.

The text target may be the enabled CPU Gemma, an enabled native Hailo LLM, or the
exact ID of the enabled VLM. With only a VLM enabled, both text and image inference
use that resident model; deterministic requests still avoid inference. For example:

```yaml
settings:
  frigate_assist_text_model: ""  # Omit, leave empty, or explicitly select Qwen3-VL-2B-Instruct
  frigate_assist_vision_model: ""  # Automatically follows models.vlm.model
models:
  vlm:
    enabled: true
    model: Qwen3-VL-2B-Instruct
    max_input_tokens: 2048
```

An explicit image target must match the enabled native VLM. A conflicting ID is
rejected while loading configuration, before startup. An explicit text target is
never silently replaced if its backend is disabled or unavailable. Virtual-model
targets are rejected. When the VLM is disabled and no text target is selected,
generative text requests fail rather than choosing an enabled Gemma implicitly.
Deterministic calls do not require either generative target to be loaded.

Text routed to a VLM keeps text-specific prompt compilation, history and tool
selection; it does not receive the image-only observation prompt. The native VLM
adapter enforces its actual tokenizer/template budget, `models.vlm.max_input_tokens`,
context capacity and reserved output tokens, including tools and history. A larger
request `max_input_tokens` or `frigate_assist_text_chars` cannot raise these limits.
Required context exceeding the budget is rejected. Tool-aware summaries use the
native JSON response contract. VLM text reasoning and tool selection remain
experimental and can be less reliable than the explicitly selected Gemma backend.

For a current text-only Hailo profile, for example:

```yaml
settings:
  frigate_assist_text_model: Qwen3-1.7B-Instruct
  frigate_assist_vision_model: Qwen2-VL-2B-Instruct
models:
  hailo_llm:
    enabled: true
    model: Qwen3-1.7B-Instruct
  vlm:
    enabled: false
```

This example supports text only: its configured vision target is not loaded.
HailoRT/GenAI 5.4.0 currently prevents loading native Hailo LLM and VLM together;
the existing startup checks and resident-model lifecycle still enforce that limit.
Gemma/CPU plus VLM/Hailo remains the current text-and-image profile. The proxy no
longer hardcodes Gemma, so a future runtime supporting resident Hailo LLM+VLM can
use both target settings without rewriting Frigate-Assist routing. Availability
and runtime compatibility must still be verified for that future deployment.

### What Frigate's `context_size` means

`genai.<provider>.provider_options.context_size` is an **optional Frigate-side
planning hint**. The OpenAI provider consumes it locally and excludes it from API
requests. It does not set `max_input_tokens`, change a HEF's capacity, or increase
Gemma's memory/context allowance.

There are two different sizes with a proxy:

| Size | Where it is handled |
|---|---|
| Incoming Frigate envelope: general instructions, catalogue, tools, results and images | Proxy compilation, bounded HTTP body and preparation limits |
| Actual prepared prompt plus tool schemas and image reserves | Selected native LLM/VLM's token budget, after compilation |

The incoming envelope can therefore be larger than 2048/4096 model tokens, as with
HA-Assist. Setting Frigate's hint to the VLM's 2048 tokens unnecessarily conflates
these layers; it is **not required** for `Frigate-Assist`. The examples leave the
override out. The inspected Frigate OpenAI provider falls back to 8192 for an
unknown non-GPT model without reported `max_model_len`; this is Frigate's estimate,
not an advertised 8192-token native context for the proxy.

The hint still changes Frigate behavior. Review descriptions use it to plan the
number of input frames after reserving prompt/response space. In the inspected dev
version, the UI also requires at least 32000 to enable manual review-description
regeneration. Omitting the override does not remove those Frigate-side gates.
Do not claim an arbitrarily large context merely to enable such a feature: more
frames and complex review JSON may exceed this experimental proxy's capabilities.
Content Frigate omits before sending cannot be recovered by the proxy.

The actual safety limits remain `models.gemma.max_input_tokens` (4096 by default),
`models.hailo_llm.max_input_tokens`, and `models.vlm.max_input_tokens` (2048 in the
example). Requests may lower the selected backend's token ceiling but not raise it.

## Processing a request

| Incoming request | Processing | Generative backend calls |
|---|---|---|
| Recognized absence question, no results yet | Return `get_profile_status` | 0 |
| Same question with an unambiguous profile interval | Return `get_recap` with supplied local times | 0 |
| Missing/ambiguous absence interval | Ask for start/end time | 0 |
| Recognized event query with last N minutes/hours, today/yesterday or a start clock time | Calculate the local interval and return `get_recap` (otherwise `search_objects`) | 0 |
| Known live camera query | Return `get_live_context` for the exact camera ID | 0 |
| Exact supported setting/cancellation instruction | Return the schema-validated action | 0 |
| Text reasoning or event summary | Compile text/history and selected schemas | 1 configured LLM/VLM call |
| Successful canonical empty `get_recap` result | Report that no activity was returned for the requested period | 0 |
| Recognized named last-sighting query | Search `sub_label` directly, with `limit: 1` when supported | 0 |
| Exact named/class last-sighting result with supplied local times | Return the verified single result directly | 0 |
| Attached event with a clothing/image question but no frame | Request the event image through a supplied image tool, otherwise explain the missing image | 0 |
| Pure image observation or explicit visual follow-up | Focus on observable image contents; no tools | 1 VLM call |
| Image search/action or forced tool choice | Keep the relevant supplied schemas within native model limits | 1 VLM call |
| Text follow-up about an earlier model answer | Strip historical frames and use the configured text target | 1 configured LLM/VLM call |

The service **returns** tools; **Frigate executes** them and submits the results in
the next request. The proxy does not call Frigate's API, read its database, or cache
camera states. Each generative request uses one backend with no fallback. An explicitly selected LLM never
falls back to a VLM if unavailable. `/health.frigate_assist` reports independent
text and vision readiness; the virtual model's appearance in `/v1/models` does not
mean that both targets are ready.

### Text preparation

The compiler replaces Frigate's long chat system prompt with short task rules.
It retains the supplied server-local clock, camera/friendly-name and zone mappings,
the current user question, and complete tool call/result dependencies for the
active turn. The preceding complete exchange, including compact tool results, is
retained when it fits the history allowance (up to 4000 characters, bounded by the
configured text limit). Larger preceding exchanges are explicitly marked as omitted;
the model is instructed to ask for relevant details when needed. Older rounds are removed. Missing or unmatched tool results are rejected,
not repaired by guessing.

Tool descriptions are shortened and schema annotations removed. Required fields,
types, enums, ranges, and other validation constraints remain intact. Tools are
selected by question category, for example historical search, live context, or
similarity. Compound questions retain tools for each recognized intent. Unknown categories and
new tools remain available rather than being silently removed. Tool results do not
end the workflow: the model can return another supplied tool call when needed. Native LLM token counting remains the final budget check.

Result lists retain at most `frigate_assist_max_events` entries per list. Omitted
record counts are explicit. Long strings/nested data receive omission markers;
image embeddings and geometric/noisy fields are removed. Errors, supplied local
timestamps and descriptions remain in the retained records. Large remaining
context produces an explicit error requesting a narrower question/time range.
Character ceilings are preparation limits, **not token counts**.

Generated tools are validated against the selected schemas and supplied camera IDs.
An exact, unique camera friendly name is normalized to its supplied ID before
proxy schema/camera validation. Unknown or ambiguous names remain errors; no
fuzzy matching, camera fallback or invented camera filter is added.
Historical calls also require valid local ISO timestamps with seconds, an increasing
interval and no timestamps later than the supplied Frigate server clock. Invalid
calls are rejected before being emitted in a stream.
Exact action forms bypass generation, for example:

- `Turn detection for camera <friendly name or ID> off`
- `Turn recording for camera <friendly name or ID> on`
- `Stop camera watch`

Other setting changes, watches and exports use the configured model with the
supplied tool schemas; they are not rejected merely because there is no exact
recognizer. The model is instructed to perform actions only when requested and
never claim success before a result or repeat a completed action. Schema and
camera validation still apply. `set_camera_state` accepts Frigate's `*` camera
wildcard. Frigate retains execution, permissions and approval handling.

### Example: “What happened while I was away?”

1. The proxy returns `get_profile_status` without running a model.
2. Frigate returns `active_profile`, `profiles`, and `last_activated` timestamps.
3. The proxy recognizes absence profiles only by the configured exact names
   (`away,abwesend` by default). Add custom names such as `vacation` yourself.
4. A completed absence runs from the latest known absence activation to activation
   of the current non-away profile, only if that end follows departure and is not
   later than the supplied server clock. An active away profile ends at that clock.
   Unknown profiles, invalid clocks or inconsistent timestamps lead to clarification.
5. `get_recap` receives local ISO strings such as `2026-10-07T17:00:00`, without an
   invented `Z` suffix or timezone conversion.
6. The configured text model summarizes the actual returned activity, including any partial-result markers.

The proxy cannot recover a full profile history that Frigate did not send. Multiple
absences or changing profiles within an absence may require an explicit time range.
The deterministic recognizer covers a small set of German/English question forms;
other wording uses the compact LLM path and remains experimental.

### Named last-sighting queries

`When was Alex last seen?` searches the literal name as `sub_label`, without an
invented camera, generic `label: person`, or semantic search for the name.
A clarification such as `I mean the person "Alex": when were they last detected?`
can provide context for the experimental reasoning path. Names are not hardcoded
to a user's setup. Frigate's historical event search defaults to newest-first;
the proxy requests `limit: 1` if the supplied tool schema supports it. Returned
data is summarized by the selected LLM, preserving supplied local time strings.

This searches Frigate's recorded recognition labels; it does not independently
verify an identity or prove that a person was never present when no match is
returned. Generic object classes, appearance descriptions and extra camera/time
filters stay on the experimental reasoning path rather than silently losing
those constraints.

### Relative event time windows and follow-ups

`Show me the events from the last hour` is handled directly: with a supplied
Frigate clock of `2026-10-07 at 10:02:08 PM`, the tool receives
`after: 2026-10-07T21:02:08` and `before: 2026-10-07T22:02:08`. No model calculates
dates, no camera filter is invented, and no semantic search for the word "event"
is added. The preferred tool is `get_recap` for review activity; if Frigate only
supplies `search_objects`, the returned data instead covers tracked detections.
Returned results are then summarized by the selected LLM.

Recognized narrow German/English event forms cover the last N minutes/hours
(positive durations within the supported calendar), today, yesterday, and a start time such as
`Show me the events from 06:00 until now`. Midnight crossings are
calculated directly. A time-only follow-up such as `from 06:00 until now`
inherits the preceding event question through consecutive time clarifications,
without taking dates from an assistant's suggestions. An unrelated question ends
this inheritance. Filtered or more complex questions still use the experimental
LLM path; this recognizer does not silently discard extra requested filters. The model also
receives resolved camera IDs and time windows for unambiguous references in filtered
questions, preserving the original question. `Show me the history of yesterday`
is handled directly: 00:00 of the previous day to 00:00 of the next day, covering
all of the final minute. All calculations use the supplied Frigate local clock,
not the proxy host's clock. Ambiguous morning/start times still require clarification.

`from this morning until now` has no defined start hour. The proxy asks specifically
which hour the user means rather than silently assuming midnight or 06:00. Missing
or invalid server clocks also require clarification. The proxy uses the clock in
Frigate's request, including if Frigate reuses an older conversation clock; it does
not replace it with the proxy host's wall clock or assume the host's timezone.

### Image preparation and descriptions

Only the current turn's need for visual observation selects vision. Questions such as
`Why did you say that?` use the text target even when earlier messages contain images.
Previous assistant descriptions are unverified claims; neither model can verify them
without the relevant frame. An explicit visual follow-up can reuse an earlier frame,
but a last-sighting query does not reuse an unrelated live image.

Named searches also recognize `When did you last see Morgan?` and
`When was Morgan last seen? Show me the last image too.` without a configured person
name. Exact camera replies continue the preceding live or last-sighting question.
Recognized class queries such as `When was a person last seen at Entrance?` use
the supplied camera catalogue and `limit: 1`. Valid single results use the exact
supplied local timestamps; errors and unknown shapes keep the normal model path.
An event ID is metadata, not an image. Clothing questions require the actual event
frame; the proxy never substitutes a similarity search or today's live frame.
It uses `get_event_image` only if Frigate supplied that tool with an `event_id`
parameter. Otherwise it asks the user to open or attach the event image.

Image chat output is capped at `frigate_assist_vision_max_tokens` (default 128;
environment variable `HAILO_FRIGATE_ASSIST_VISION_MAX_TOKENS`), or the lower requested
limit. This reduces generation time but does not increase the native context budget.
Structured description requests keep their requested output limit and format contract.

Vision gets a brief observation instruction and the relevant question/frame caption,
not Frigate's generic search instructions. Pure observation uses no tool schemas;
explicit image-search/action tasks retain the relevant schemas, and an explicit
required/forced tool choice is respected. These mixed requests still need to fit
the native VLM context and remain experimental. The prompt asks for visible
objects/actions, uncertainty where needed, and no invented identity, intention,
off-screen event, timestamp or technical camera status. The virtual model retains
explicit short description/format instructions. Overlong vision tasks fail with
`frigate_assist_vision_chars` guidance instead of silently removing a JSON contract.

The latest image-bearing message is retained, including its frame sequence. Earlier
image messages are omitted with an explicit warning in the task. Historical images
are labelled as historical when the latest question has no fresh frame. Comparisons
across separate image messages are therefore limited. Native per-model frame and
token limits still apply; Qwen2's multi-frame overflow sampling is unchanged.

Frigate's live tool adds the frame as a separate user message after its tool result.
The proxy retains the original question and this caption for that follow-up. For
plain descriptions without tools, the concise description request goes directly
through the same virtual model to the VLM. No second Gemma call rewrites the image answer.

Complex review-description JSON contracts may exceed the small VLM context or be
poorly followed by Qwen2. Use a separate stronger description provider where needed.
Prompt instructions reduce hallucination pressure but cannot eliminate hallucinations.
“Camera status” is clarified: the supplied live-context tool provides images and
detections, not a complete technical camera-health report.
The short reply `the current camera image` selects
`get_live_context` directly when exactly one camera is supplied. With multiple
cameras the proxy asks for a camera name. A named unknown camera is never replaced
with the sole available camera.

The vision prompt specifies the answer language explicitly (the optional request
`language`, otherwise the human question and service-language fallback). Frigate's
synthetic English live-frame caption does not choose the language; for German
requests it is rewritten into a short German caption. Observations are limited to
two short sentences unless an explicit format requires more. The prompt asks the
model to report people only when a human body is clearly visible, rather than
inferring people from objects or shadows. This is a generation instruction, not
a verified person detector; night/infrared scenes can still be misinterpreted.
The proxy rejects clearly damaged vision output: an empty answer or a Unicode
replacement character (`U+FFFD`) is not presented as a usable image description.
Chat receives a localized message asking the user to inspect the image directly;
description requests fail explicitly rather than receiving prose that violates
their output contract. `event=frigate_vision_quality` and
`metrics.frigate_vision_quality` expose the reason/action. There is no repair by
guessing missing text and no second model call. This guard detects damaged text,
not plausible-sounding hallucinations, and leaves native model requests unchanged.
`get_live_context` requests Frigate's current processed camera frame rather than
searching for the latest historical detection. That frame/context can lag behind a
later live-view screenshot. Compare the tool's context timestamp and the actual
attached frame before treating differing observations as hallucinations. Empty
tracked detections alone do not prove that no person is visible in an image.

## Streaming and diagnostics

Frigate's `stream: true` and `stream_options: {include_usage: true}` are supported
for `model: Frigate-Assist`. Tools and final text are buffered until validation
finishes. The public completion model name remains `Frigate-Assist` on every round.
The final usage chunk has empty `choices`; usage is reported when measured, otherwise
null. Deterministic calls report zero model tokens. No HA voice wait message is emitted.

Responses expose `metrics.frigate_route` (route, reason, target, inference count,
preparation time) and `metrics.frigate_prompt` (message/tool counts, prepared size,
record cap). The normal native model metrics report actual token counts when available.
The usage stream's final chunk also includes these metrics.

With debug logging enabled, `event=frigate_assist` reports preparation/routing counts.
`event=frigate_stream_end` reports completion/error/cancellation, the emitted
finish reason, and whether the `[DONE]` marker was emitted by the server. This
confirms server-side stream progress, not that Frigate/browser consumed the marker.
The usual native input-budget events show whether the compiled prompt fits. A failure
never triggers a silent switch to a less suitable model. Validate real answer quality
and latency on the Raspberry Pi/Hailo device; automated tests use backend doubles.

Native chat also reports `metrics.native_phases` and `event=native_chat_phases`:
input preparation, media preparation, context clearing, entry into generation, and
generation streaming. These measure Python/API boundaries, not separate hardware
vision-encoder or prefill kernels.

YOLO startup reports its scheduler priority (default 31), batch size and SHARED
device group. With debug logging, `event=vision_execution` separates owner-thread
queue waiting from backend execution; execution exceeding 500 ms is logged at warning
level. `event=vision_native_phases` splits preparation, async-ready waiting,
submission and job-completion waiting. ZMQ request IDs correlate these events.
Completion waiting includes native scheduling and execution, so it is not pure
compute time. Priority determines selection of eligible work; it does not promise
immediate interruption of an ongoing VLM operation. Concurrent YOLO/VLM latency
still needs measurement on real hardware. No scheduler or model-unloading workaround
is applied by these diagnostics.

Frigate's detector CPU/RAM display measures its local detector process, including
the ZMQ client, rather than the remote Hailo service. Detector latency can include
transport and remote waiting; a low local CPU percentage does not prove that Hailo
is idle. Check remote service metrics and correlated logs as well.

If chat remains busy and no new service requests appear, check the **Frigate** log
and its outstanding `/api/chat/completion` browser request. A completed service
response can be followed by Frigate tool execution before another model call; a
slow semantic search or another tool can therefore leave this service's log quiet.
Cancel the running chat or reload/start a new conversation to recover the UI.
The proxy cannot cancel or reset Frigate's internal tool execution. A service-side
HTTP 200 alone does not demonstrate that the entire Frigate chat turn completed.
Successful empty recap results are answered without a model; error, partial or
unknown result shapes keep the normal summary path. A watch request uses the configured model and supplied watch schema; an actual
watch requires a tool call executed by Frigate, not a text promise from the model.

go2rtc `producer.go ... error=EOF` entries concern the upstream camera stream.
They are not chat API errors and do not by themselves establish why a chat stalls.
Inspect the Frigate application log (container: `/dev/shm/logs/frigate/current`)
and browser console/network separately from `/dev/shm/logs/go2rtc/current`.

## References

- [Frigate named GenAI providers and roles](https://docs.frigate.video/configuration/genai/genai_config/)
- [Frigate OpenAI provider source](https://github.com/blakeblackshear/frigate/blob/dev/frigate/genai/plugins/openai.py)
- [Frigate chat tool/result format](https://github.com/blakeblackshear/frigate/blob/dev/frigate/api/chat.py)
- [Frigate event search ordering](https://github.com/blakeblackshear/frigate/blob/dev/frigate/api/event.py)
- [Frigate review frame budgeting](https://github.com/blakeblackshear/frigate/blob/dev/frigate/data_processing/post/review_descriptions.py)
- [Frigate review regeneration UI gate](https://github.com/blakeblackshear/frigate/blob/dev/web/src/hooks/use-review-descriptions.ts)
- [Service APIs](api.md)
- [HA-Assist](ha-assist.md)
