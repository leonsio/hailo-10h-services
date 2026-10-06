# API and integrations

## Endpoints

| Protocol | Endpoint / port | VLM | LLM | Whisper |
|---|---|---|---|---|
| OpenAI-style HTTP | `:8090/v1/chat/completions` | Text, images, SSE, function tools | Text, SSE, function tools | — |
| OpenAI-style HTTP | `:8090/v1/audio/transcriptions` | — | — | File upload |
| Models / readiness | `/v1/models`, `/health` | Model status | Model status | Model status |
| WebSocket | `ws://HOST:8090/ws` | `chat` | `chat` | `transcribe` |
| MCP Streamable HTTP | `http://HOST:8090/mcp/` | `analyze_image`, `chat_text` | `chat_text` | `transcribe_audio` |
| MQTT (optional) | `hailo10h/request/chat`, `…/transcribe` | JSON requests | JSON requests | Base64 audio |
| Wyoming TCP | `HOST:10300` | — | — | Home Assistant Assist STT |

The **LLM** column covers both native Hailo HEF LLMs and optional Gemma 4 E2B
through LiteRT-LM on CPU. Select an enabled model by its exact `/v1/models` ID:
`Qwen2.5-1.5B-Instruct` (or another configured HEF LLM), `gemma-4-E2B-it`, or the
configured VLM. Gemma is independent of the enabled Hailo chat model. The service
does not enforce VLM/HEF-LLM mutual exclusion; configure the enabled hardware
models yourself. Disabled/unknown IDs never silently select another backend.

HTTP and WebSocket require `Authorization: Bearer API_KEY` when configured.
MCP skips API-key authentication for loopback and private LAN peers by default
(`HAILO_MCP_NO_AUTH_NETWORKS`). Other peers still require the API key.
Set this variable to an empty value to require the key for all MCP peers.
Wyoming has no token authentication: keep port 10300 on a trusted network and
limit access to HA. MQTT uses broker credentials, optional TLS and broker topic
ACLs; HTTP keys do not apply to it. HTTP defaults to LAN binding; use a reverse
proxy for TLS. When launching manually, an empty `HAILO_API_KEY` disables auth.

### Home Assistant

**Speech:** Settings → Devices & services → Add integration → **Wyoming Protocol**.
Enter the machine's IP and port **10300**. Then select `hailo-whisper` / Whisper
Base as speech-to-text in your Assist pipeline, language German (`de`), English (`en`) or Russian (`ru`), subject to the other pipeline providers. See [language diagnostics](languages.md). The service
advertises its ASR capabilities through Wyoming `describe`/`info`. Home Assistant
supplies end-of-speech/VAD; this service buffers audio until `audio-stop`.

**Images:** use a REST call from an automation/custom integration to
`/v1/chat/completions`, supplying a snapshot as a base64 data URL. MCP-capable
clients can call `analyze_image`. Add the actual service IP/name plus `:*` to
`HAILO_MCP_HOSTS` for MCP DNS rebinding protection, then restart the service.

**Local MCP without credentials:** configure the service environment:

```ini
HAILO_MCP_HOSTS=localhost:*,127.0.0.1:*,192.168.1.9:*
HAILO_MCP_NO_AUTH_NETWORKS=127.0.0.0/8,::1/128,192.168.2.4/32
```

Restart `hailo-10h-services`, then add the MCP integration with
`http://192.168.1.9:8090/mcp/`. No API key or OAuth credentials are needed.
The first setting permits the destination Host header; the second permits the
Home Assistant source IP. HTTP/WebSocket authentication and MCP body/Host limits
remain active. The allowlist uses the socket peer; the bundled listener disables proxy-header
trust. Restrict access when using a local proxy, whose peer IP is what the service sees.

This is an inference gateway. For Home Assistant device control, the client or
conversation integration supplies function schemas and executes the resulting
tool calls using its own Home Assistant permissions. Gemma can select and return
those function calls through the OpenAI-compatible chat endpoint, but the gateway
itself does not receive a Home Assistant access token and does not execute device
actions. Qwen remains the VLM path for text/images and does not receive HA entity
access from this service. MCP exposes inference tools to clients; it does not make
Qwen itself a tool-calling agent.

### HTTP examples

```bash
export HAILO_URL=http://HOST:8090
export HAILO_KEY='key from service env file'
curl "$HAILO_URL/v1/models" -H "Authorization: Bearer $HAILO_KEY"
curl "$HAILO_URL/v1/chat/completions" \
  -H "Authorization: Bearer $HAILO_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2-VL-2B-Instruct","messages":[{"role":"user","content":"Antworte auf Deutsch: Was ist Photovoltaik?"}],"max_tokens":128}'
curl "$HAILO_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $HAILO_KEY" \
  -F file=@voice.wav -F model=whisper-base -F language=de
```

An image content list in a `user` message:

```json
[
  {"type":"text","text":"Lies den Zählerstand ab."},
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,BASE64_BYTES"}}
]
```

Images are decoded to writable contiguous RGB UINT8 and resized to the loaded
model's input shape (Qwen2: 336×336, up to four images; Qwen3: 512×288, one image). Only inline base64/data URLs are accepted; snapshot HTTP URLs
are not fetched. `stream:true` enables native-token SSE ending in `[DONE]`.
`max_tokens` is 1..1024; Gemma accepts `temperature` in 0..1, while Hailo LLM/VLM require
`0 < temperature <= 1` (default 0.1). A Hailo request with `temperature=0` returns
HTTP 400 with an explanatory error instead of failing inside HailoRT. The gateway implements a documented
subset of the OpenAI API. Gemma accepts function `tools`/`tool_choice` as described
below; Qwen supports text-only tool requests through the validated JSON adapter. JSON-schema response formats remain unsupported.
Token usage is not fabricated. Text-only Qwen requests pass `frames=[]` and need
target-device validation alongside image requests.

Whisper accepts WAV/FLAC/OGG formats supported by libsndfile, averages channels,
resamples to 16 kHz and submits float32 PCM. `response_format=json|text`.
`whisper-1` is an alias for this local Whisper Base, not an external provider.
Default language `de`; explicit two-letter codes supported. No TTS, wake-word
engine, automatic language detection or audio translation is implemented.

### Request metrics

Successful non-streaming `/v1/chat/completions` JSON responses include a `metrics`
object. `/v1/audio/transcriptions` includes request timing with `response_format=json`;
plain text responses retain their existing format. Streaming/WS/MQTT formats are unchanged.

| Metric | Meaning |
|---|---|
| `request_id` | Identifier also used in the service log |
| `requested_at`, `responded_at` | Server UTC timestamps in ISO 8601 with milliseconds |
| `processing_ms` | Server time from HTTP arrival through upload buffering, parsing, queueing, preparation and inference to JSON response preparation |
| `inference_ms` | Model generation call duration; excludes earlier request preparation and queueing |
| `ttft_ms`, `ttft_source` | Generation start to the first nonempty text chunk when that timing is observable; excludes earlier HTTP/queue time |
| `input_tokens`, `input_tokens_source` | Tokenizer count for the accepted rendered prompt. VLM text counts exclude image tokens |
| `output_tokens`, `output_tokens_source` | Output text retokenized using the loaded model tokenizer where available |
| `input_budget_tokens` | Conservative input budget including safety/image reserves; separate from reported token counts |

Unavailable metrics are omitted, never inferred from character or chunk counts.
The service does not enable LiteRT native benchmark collection. A non-streaming
response includes OpenAI-style `usage` only when both token counts are available;
their provenance is described in `metrics`. Retokenized text counts can differ
from native internal generation counts, and VLM text counts do not represent image tokens.

The playground displays measurements beside each request, including failed ones,
and retains separate transcript/measurement entries for Whisper. Its browser timer
also includes network transfer and JSON decoding. Browser timestamps use the local
timezone with milliseconds; server timestamps are displayed separately to reveal
clock differences. Pending requests show an updating timer. TTFT unavailable from
the model is shown as unavailable; the complete HTTP response time is never used
as a substitute. Switching models or chat modes retains the displayed chat and
text history; Gemma requests strip image parts while retaining their text, and the
original image parts remain available for later VLM requests. The UI sends up to
31 recent messages while keeping the full visible history. The model's existing
input budget may trim older context independently of the visible history.

### WebSocket and MQTT

WebSocket request (`Authorization` header on the handshake):

```json
{"id":"camera-1","op":"chat","payload":{"messages":[{"role":"user","content":"Hallo"}]}}
```

MQTT: configure `HAILO_MQTT_HOST`, username/password and optionally
`HAILO_MQTT_TLS=true` with port 8883. Publish to `hailo10h/request/chat`:

```json
{"id":"camera-1","payload":{"messages":[{"role":"user","content":"Hallo"}]}}
```

For either protocol, transcription payload:

```json
{"audio_base64":"BASE64_WAV_BYTES","language":"de"}
```

WebSocket returns on the same connection; MQTT publishes to
`hailo10h/response/camera-1`. Response:

```json
{"id":"camera-1","ok":true,"result":{"text":"Antwort"}}
```

Errors use `ok:false,error:"..."`. MQTT IDs: 1..64 letters/digits/`_`/`-`.
Use unique IDs and subscribe to the response before publishing. MQTT requests
and responses use QoS 0 (no deduplication); never retain requests. Retained
requests are ignored. `hailo10h/status` is retained `online` with an `offline`
last will. A shutdown disconnect may leave that status stale; `/health` is the
readiness source of truth. The bridge reconnects after broker outages and handles
one MQTT request at a time. MQTT JSON includes base64 overhead; adjust broker
packet limits too. MQTT/WebSocket requests return complete results; token
streaming is HTTP SSE only. `health` is also supported as an operation.

### Home Assistant virtual model

Home Assistant should request `model: "HA-Assist"`. This model uses the same
`/v1/chat/completions` and WebSocket chat operation, tools, history, sampling
parameters and request limits as the physical models. Its responses always name
`HA-Assist`; `metrics.ha_route.backend_model` identifies the actual target.

```json
{
  "model": "HA-Assist",
  "messages": [{"role": "user", "content": "Mach das Licht im Wohnzimmer aus"}],
  "max_tokens": 256,
  "max_input_tokens": 4096
}
```

For device control the client additionally supplies its normal system/entity
context and function tools. The service does not invent tools or execute actions.
Configure `settings.ha_assist_text_model` and `settings.ha_assist_vision_model`
with enabled local model IDs. ENV equivalents are
`HAILO_HA_ASSIST_TEXT_MODEL` and `HAILO_HA_ASSIST_VISION_MODEL`.
`HAILO_HA_ASSIST_ENABLED=false` hides/disables the virtual model.

Routing is fixed for each request: any image content, including retained image
history, selects the VLM; text selects the LLM. Deterministic HA answers/calls
skip generative inference. Otherwise only the selected backend runs; unavailable
targets, generation errors and token overflow never trigger another model.
MiniLM may rank HA context/tools during preparation; it is not a second
answer-generating backend. Physical IDs and omitted-model requests retain
ordinary backend selection and bypass all HA-specific processing, even when HA
function names or `Static Context:` appear in the input.

For text requests to `HA-Assist`, history is filtered before routing. A house
request keeps the current user turn and its active tool calls/results. Completed
house turns are omitted from later backend prompts. General conversation keeps
previous non-house user/assistant turns, so a follow-up such as “Which sights are
there?” retains the preceding geography question and answer. The client may
still display/store the complete conversation: filtering changes only the
request sent to the backend. `metrics.ha_history` and the debug event
`ha_history` report the policy and message counts. Explicit absolute brightness
and cover percentages can produce validated tool calls without LLM inference;
relative, conditional, composite and ambiguous commands use the normal routing.
When inference is needed for a house request, the prompt compiler supplies the
task-specific rules and schemas instead of the long HA examples.

HA preparation now records `metrics.ha_plan`: the original/canonical text,
catalogue spelling corrections, resolved area/domain and absolute percentage.
Unambiguous spelling aliases and fuzzy catalogue slots feed the same direct
action and prompt compilation paths. Unresolved area commands ask for a target
instead of guessing an individual device. Explicit terminal values such as
“set the light in the living room to 70” can also use the absolute percentage
path. Negations, relative adjustments and composite commands retain the normal
guards. Known tool failures return a deterministic failure message without a
second inference. Active call/result dependencies remain intact during prompt
compaction. Generated actions are validated against both JSON Schema and the
resolved target/value; rejected actions appear in `metrics.ha_validation`.

Static catalogue parsing uses bounded, content-keyed caches (64 entries), and
each backend embedding cache retains at most 512 entries. Changed catalogue
content automatically selects a new entry. Cached catalogue results are copied
before use. Live states, tool results and answers are never cached. Cache hits
are visible in `metrics.ha_plan.catalogue_cache_hit`.

LiteRT text requests without tools receive a brief-answer default (up to three
sentences unless the user explicitly asks for detail). Existing conversation
content and sampling/output limits are retained. This default applies to plain
Gemma too and does not enable HA processing. Native tool constrained decoding is
enabled only when the installed Python engine explicitly exposes
`enable_constrained_decoding`; `metrics.constrained_decoding` reports support.
Schema/HA validation remains active when native decoding is unavailable.

Exact HassIL matching uses the packaged official `home-assistant-intents` grammar
for the request language (de/en/ru). Only declared HA intent tools are eligible.
Ambiguous/unsupported matches defer to the text target. Fuzzy repair only changes
one known entity/area slot with a unique score and reparses with HassIL; it does
not rewrite action words. Configure `ha_assist_fuzzy_enabled`,
`ha_assist_fuzzy_threshold` (default 90) and `ha_assist_fuzzy_margin` (default 8).
Every direct tool call respects the original schema and `tool_choice`.

`metrics.ha_intent` reports candidates, exact/fuzzy results, slots and match time;
`metrics.ha_route` reports route, target, language and preparation time. A direct
response reports zero model input/output tokens with `deterministic` sources and
zero inference time. Model answers retain native metrics when available.
HA streaming buffers the answer/tool calls in one SSE completion; native-start
wait messages remain optional. No extra model request is made for a wait message.

### Home Assistant device control through function tools

Chat requests accept optional `max_input_tokens` (1..131072) and `top_p` (0..1), including the
`top_p: 1.0` sent by Home Assistant's llama.cpp integration. The value is passed
to LiteRT's `SamplerConfig` for Gemma and Hailo's generation parameters for Qwen.
If omitted or null, the backend's existing sampling default is preserved.

LiteRT's implicit 4096-token context is too small for full Home Assistant function
schemas and a system prompt (a reported two-message request used 7385 input
tokens). The gateway now explicitly sets `Engine(max_num_tokens=16384)` by
default. Configure `HAILO_LITERT_MAX_NUM_TOKENS` in
`/etc/hailo-10h-services.env` and restart to change it. The limit includes input,
history, tool schemas and generated output; `max_tokens` in the chat request
only limits the generated response and does not enlarge the context.

For Gemma, `max_input_tokens` is a per-request limit for prompt tokens. It
is measured with the loaded LiteRT-LM tokenizer after rendering the actual
Gemma template and Home Assistant tools. The requested `max_tokens` and one
start-token slot are reserved inside `HAILO_LITERT_MAX_NUM_TOKENS`, so the
effective input cap is the lower of `max_input_tokens`, the configured Gemma
ceiling (4096 by default) and the remaining context. When needed, the service removes complete older user turns (including
their assistant/tool-call/tool-result messages), while keeping system messages
and the complete current user/tool turn. Tool calls remain enabled. If the
required system prompt, tools and current turn alone exceed the cap, the API
returns `input_token_limit_exceeded` with the measured size instead of damaging
the prompt.

Qwen2/Qwen3-VL also accept `max_input_tokens`. Their compiled context limit is
**2048 tokens**, shared by prompt and response. The service renders the loaded
VLM's template and uses **its native tokenizer**, with conservative headroom for
native bookkeeping and image tokens. The usable input ceiling is the minimum of
`models.vlm.max_input_tokens`, the request limit, and native context capacity
minus output reserve. Qwen3 images are resized to **512×288** (one image per
request); Qwen2 defaults to the **v5.1.1 HEF**, with **336×336** frames.

Native Hailo LLMs have a separate text adapter: their own HEF prompt template
and tokenizer measure the actual string prompt sent to `LLM.generate` without
vision placeholders or frames. The effective input limit is the minimum of
`models.hailo_llm.max_input_tokens`, the request limit, and native/catalogue context
capacity minus the requested output reserve and one start-token slot. Current
catalogue LLMs have 2048 context tokens; template headroom is reserved as well.
Tool calling uses the compact contract and strict schema validation before
returning OpenAI `tool_calls`; native `<tool_call>` JSON wrappers are accepted.
Thinking is disabled in the rendered template by default.

Model-specific template behaviour is recorded in `model_catalog.yaml`.
For `Llama3.2-1B-Instruct`, `prompt_template.empty_tool_calls: omit` prevents
ordinary messages from being mistaken for tool-call turns. Qwen retains the
default `include` behaviour for optional template fields.
Llama's `tool_calling.parallel_calls: false` also forces a single-call prompt
contract and rejects multiple generated calls, even when the client requests
`parallel_tool_calls: true`. Tool calls and matched results in history continue
to use the compact text representation; they are not silently dropped.
The service returns calls to the client and does not execute or sequence them.

Omit `model` for automatic routing: text uses ready Gemma, then an enabled Hailo
LLM, then the resident VLM; images require an enabled VLM. Explicit model IDs remain authoritative.
For `HA-Assist`, the incoming JSON may be much larger than the model budget:
tool/entity retrieval and HA prompt compilation run first. Direct physical-model
requests retain the original context/tools and may therefore exceed the budget. The final model-bound prompt, including
selected schemas, template and tool history, must fit. `tool_choice: "none"`
excludes tool schemas from inference. `/health` and `/ui/config` expose
`model_limits` and `default_text_model`; `/ui/config` also exposes `hailo_llm_model`
and `vision_models`. Debug logs include `llm_input_budget` / `final_llm_request`
or `vlm_input_budget` / `final_vlm_request`. Both Hailo chat types report tokenizer
input/output counts and first-text-chunk TTFT; VLM logs include frame dimensions.

With the Home Assistant **Local OpenAI LLM** conversation integration, choose
model **HA-Assist** and server type **Generic OpenAI-Compatible**. In the Conversation Agent options,
open **Request Body Parameters** and add `max_input_tokens` with value `4096`.
The integration sends it as a top-level request parameter. It also has **Max
Message History**; that caps the number of messages before the service applies
its exact token budget. Don't choose server type `llama.cpp` for this service,
because that mode adds llama.cpp-specific request parameters.

The startup log and `/health` → `litert_lm.max_num_tokens` show the configured
context. More context increases RAM requirements, and a particular model export
may impose its own limit. No system prompt, tools or current user/tool turn is
silently removed. Native
inference failures now return a JSON error with HTTP 502; consult the preceding
native log for the specific cause rather than assuming every failure is a
context overflow. Start with 16384 for the reported Home Assistant request;
larger histories may require a larger context or a shorter conversation.

The HTTP `/v1/chat/completions` endpoint accepts `user`, `tools`, `tool_choice`
(`auto`, `none`, `required`, or a named function), and `parallel_tool_calls`.
Tool calling uses **Gemma through LiteRT-LM** with its native function parser,
or **native Hailo LLMs and Qwen2/Qwen3-VL** (including image/tool requests). The VLM adapter places the
offered schemas and choice instructions into a compact JSON-call contract.
Generated function names and arguments undergo the same schema validation as
Gemma, including required/named choice and parallel-call rules. Model tool-call
quality must be tested with your exposed HA devices; it is not a native Hailo
function-calling capability.

For Gemma, the server passes the selected function schemas to LiteRT-LM with
`automatic_tool_calling=False`. It returns OpenAI-compatible `tool_calls` with
unique IDs, JSON string arguments and `finish_reason: "tool_calls"`.
**Home Assistant executes the actions** using its own permissions and exposed
entities. The gateway needs no Home Assistant access token and never executes
the advertised functions itself.

The next request may include assistant messages with `content: null` and
`tool_calls`, followed by `role: "tool"` messages with matching `tool_call_id`s.
The adapter restores these as native LiteRT tool calls and tool responses so
Gemma can produce a spoken answer or request another function. Consecutive tool
results are grouped into one native tool message. Ordinary chat still streams
text as before. Tool-enabled streaming buffers one model response before emitting
validated tool-call deltas. Recognized HA requests can send an earlier localized wait delta at native inference start; see [pipelines](pipelines.md).

Generated functions must be among the offered tools, and their arguments must
validate against the corresponding JSON schema. Invalid or incomplete calls are
reported as errors instead of actions. Named choices restrict the offered tools;
`required` and named choices reject responses without a function call.
`parallel_tool_calls=false` rejects multiple generated calls. JSON schemas may
use local references but cannot retrieve external references. This implements
the protocol; actual tool selection and reliable device identification still
depend on the model and the Home Assistant prompt.

Update an existing installation from the branch containing this change:

```bash
git fetch origin
git switch feat/home-assistant-tool-calling
git pull --ff-only
sudo /opt/hailo-10h-services/venv/bin/pip install --upgrade 'jsonschema>=4.23,<5'
sudo /opt/hailo-10h-services/venv/bin/pip install --no-deps --force-reinstall .
sudo systemctl restart hailo-10h-services.service
```

The installed `litert_lm` package must export `Tool` and support
`create_conversation(tools=..., automatic_tool_calling=False)`.
If the API reports missing Tool support, upgrade `litert-lm` in the service venv.
The installer also checks for `Engine`/`Tool` when a LiteRT model is configured.
No Hailo driver, HEF model or SHARED device settings change.

To check function generation without executing any device action, send a request
with one tool (add your configured bearer API key):

```bash
curl -sS http://localhost:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${HAILO_API_KEY}" \
  -d '{"model":"gemma-4-E2B-it","messages":[{"role":"user","content":"Schalte die Lampe im Wohnzimmer ein."}],"tools":[{"type":"function","function":{"name":"intent__HassTurnOn","description":"Turns on a light","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}}}],"tool_choice":"required","max_tokens":256}'
```

Expect a tool call containing `intent__HassTurnOn` and a JSON argument object
with the light name, rather than just a textual claim that the light is on.
Target Pi/model testing is still required; protocol tests use a fake LiteRT
engine and do not establish actual model accuracy or device execution.

Sources used for the implementation:

- [Hailo GenAI examples](https://github.com/hailo-ai/hailo_model_zoo_genai)
- [Hailo shared-device usage and KV-cache limitation](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/USAGE.rst)
- [Official Wyoming protocol](https://github.com/OHF-Voice/wyoming)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
