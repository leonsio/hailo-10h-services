# Frigate-Assist: experimental chat and description proxy

`Frigate-Assist` is a virtual OpenAI-compatible model at `/v1/chat/completions`.
It has no model weights of its own. It prepares Frigate requests before routing
text to **Gemma / LiteRT-LM on CPU**, or images to the configured resident **Hailo VLM**.
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
  frigate_assist_vision_model: Qwen2-VL-2B-Instruct
  frigate_assist_max_events: 12
  frigate_assist_text_chars: 10000
  frigate_assist_vision_chars: 1200
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
    provider_options:
      context_size: 2048
```

The base URL includes `/v1`; do not append `/chat/completions`. Keep an existing
embeddings provider or Frigate's local semantic-search model. `context_size: 2048`
is a conservative value for the vision backend in this example, not a promise that
every chat fits. The service applies the real selected backend's token limit after
compilation. Enable object/review description generation separately in Frigate;
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
    provider_options:
      context_size: 2048
```

Each role must belong to only one provider. Chat may be omitted altogether.
Direct Qwen requests retain their existing API behavior; the Frigate-specific
`stream_options` extension described below is accepted only by `Frigate-Assist`.

## Processing a request

| Incoming request | Processing | Generative backend calls |
|---|---|---|
| Recognized absence question, no results yet | Return `get_profile_status` | 0 |
| Same question with an unambiguous profile interval | Return `get_recap` with supplied local times | 0 |
| Missing/ambiguous absence interval | Ask for start/end time | 0 |
| Known live camera query | Return `get_live_context` for the exact camera ID | 0 |
| Exact supported setting/cancellation instruction | Return the schema-validated action | 0 |
| Text reasoning or event summary | Compile text/history and selected schemas | 1 Gemma call |
| Request containing image parts | Focus on observable image contents; no tools | 1 VLM call |

The service **returns** tools; **Frigate executes** them and submits the results in
the next request. The proxy does not call Frigate's API, read its database, or cache
camera states. Each generative request uses one backend with no fallback. Text never
falls back to Qwen if Gemma is unavailable. `/health.frigate_assist` reports independent
text and vision readiness; the virtual model's appearance in `/v1/models` does not
mean that both targets are ready.

### Text preparation

The compiler replaces Frigate's long chat system prompt with short task rules.
It retains the supplied server-local clock, camera/friendly-name and zone mappings,
the current user question, and complete tool call/result dependencies for the
active turn. One short preceding plain question/answer can be retained for references.
Older completed tool rounds are removed. Missing or unmatched tool results are rejected,
not repaired by guessing.

Tool descriptions are shortened and schema annotations removed. Required fields,
types, enums, ranges, and other validation constraints remain intact. Read tools are
selected by question category, for example historical search, live context, or
similarity. Unknown categories keep the available read tools rather than guessing
one tool. Native Gemma token counting remains the final budget check.

Result lists retain at most `frigate_assist_max_events` entries per list. Omitted
record counts are explicit. Long strings/nested data receive omission markers;
image embeddings and geometric/noisy fields are removed. Errors, supplied local
timestamps and descriptions remain in the retained records. Large remaining
context produces an explicit error requesting a narrower question/time range.
Character ceilings are preparation limits, **not token counts**.

Generated tools are validated against the selected schemas and supplied camera IDs.
The proxy does not expose writable tools to generative inference. This first version
supports conservative exact forms such as:

- `Schalte die Erkennung für Kamera <friendly name or ID> aus`
- `Schalte die Aufnahme für Kamera <friendly name or ID> ein`
- `Turn detection for camera <friendly name or ID> off`
- `Stop camera watch` / `Stoppe die Überwachung`

More complex setting changes, creating watches/exports, wildcard actions, or unclear
targets remain unsupported. The camera must come from the request's catalogue.
Frigate still enforces permissions and user approval for actions. This proxy does
not bypass those checks. Subsequent action results are summarized by Gemma rather
than interpreted as proof of success before Frigate returns them.

### Example: “Was ist passiert, während ich weg war?”

1. The proxy returns `get_profile_status` without running a model.
2. Frigate returns `active_profile`, `profiles`, and `last_activated` timestamps.
3. The proxy recognizes absence profiles only by the configured exact names
   (`away,abwesend` by default). Add custom names such as `Urlaub` yourself.
4. A completed absence runs from the latest known absence activation to activation
   of the current non-away profile, only if that end follows departure and is not
   later than the supplied server clock. An active away profile ends at that clock.
   Unknown profiles, invalid clocks or inconsistent timestamps lead to clarification.
5. `get_recap` receives local ISO strings such as `2026-10-07T17:00:00`, without an
   invented `Z` suffix or timezone conversion.
6. Gemma summarizes the actual returned activity, including any partial-result markers.

The proxy cannot recover a full profile history that Frigate did not send. Multiple
absences or changing profiles within an absence may require an explicit time range.
The deterministic recognizer covers a small set of German/English question forms;
other wording uses the compact Gemma path and remains experimental.

### Image preparation and descriptions

Vision gets a brief observation instruction and the relevant question/frame caption,
not Frigate's search instructions or eight tool schemas. The prompt asks for visible
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
The usual native input-budget events show whether the compiled prompt fits. A failure
never triggers a silent switch to a less suitable model. Validate real answer quality
and latency on the Raspberry Pi/Hailo device; automated tests use backend doubles.

## References

- [Frigate named GenAI providers and roles](https://docs.frigate.video/configuration/genai/genai_config/)
- [Frigate OpenAI provider source](https://github.com/blakeblackshear/frigate/blob/dev/frigate/genai/plugins/openai.py)
- [Frigate chat tool/result format](https://github.com/blakeblackshear/frigate/blob/dev/frigate/api/chat.py)
- [Service APIs](api.md)
- [HA-Assist](ha-assist.md)
