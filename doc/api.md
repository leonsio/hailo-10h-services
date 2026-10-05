# API and integrations

## Endpoints

| Protocol | Endpoint / port | VLM | Whisper |
|---|---|---|---|
| OpenAI-style HTTP | `:8090/v1/chat/completions` | Text, images, SSE | — |
| OpenAI-style HTTP | `:8090/v1/audio/transcriptions` | — | File upload |
| Models / readiness | `/v1/models`, `/health` | Model status | Model status |
| WebSocket | `ws://HOST:8090/ws` | `chat` | `transcribe` |
| MCP Streamable HTTP | `http://HOST:8090/mcp/` | `analyze_image`, `chat_text` | `transcribe_audio` |
| MQTT (optional) | `hailo10h/request/chat`, `…/transcribe` | JSON requests | Base64 audio |
| Wyoming TCP | `HOST:10300` | — | Home Assistant Assist STT |

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

This is an inference gateway, not an HA conversation agent that executes tools.
The built-in OpenAI Conversation integration is not automatically redirected to
this server just by installing it; use an integration/client that supports a
custom OpenAI base URL (`http://HOST:8090/v1`) for text/image calls. For HA home
control, a separate agent must handle permitted HA actions. Qwen does not receive
HA entity access from this service. MCP exposes inference tools to clients;
it does not make Qwen itself a tool-calling agent.

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

Images are decoded to RGB uint8 and resized to the model's 336×336 input, at most
four per request. Only inline base64/data URLs are accepted; snapshot HTTP URLs
are not fetched. `stream:true` enables native-token SSE ending in `[DONE]`.
`max_tokens` is 1..1024; `temperature` is 0..1. Unsupported OpenAI parameters,
including `tools`, `tool_choice` and JSON-schema response formats, are rejected;
this implements a documented subset, not the entire OpenAI API. Token usage is
not fabricated. Text-only Qwen requests pass `frames=[]` and need target-device
validation alongside image requests.

Whisper accepts WAV/FLAC/OGG formats supported by libsndfile, averages channels,
resamples to 16 kHz and submits float32 PCM. `response_format=json|text`.
`whisper-1` is an alias for this local Whisper Base, not an external provider.
Default language `de`; explicit two-letter codes supported. No TTS, wake-word
engine, automatic language detection or audio translation is implemented.

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

For Gemma only, `max_input_tokens` is a per-request limit for prompt tokens. It
is measured with the loaded LiteRT-LM tokenizer after rendering the actual
Gemma template and Home Assistant tools. The requested `max_tokens` and one
start-token slot are reserved inside `HAILO_LITERT_MAX_NUM_TOKENS`, so the
effective input cap is the lower of `max_input_tokens`, 4096 and the remaining
context. When needed, the service removes complete older user turns (including
their assistant/tool-call/tool-result messages), while keeping system messages
and the complete current user/tool turn. Tool calls remain enabled. If the
required system prompt, tools and current turn alone exceed the cap, the API
returns `input_token_limit_exceeded` with the measured size instead of damaging
the prompt. Qwen image requests do not accept this parameter because the Gemma
tokenizer cannot count Qwen's image tokens.

With the Home Assistant **Local OpenAI LLM** conversation integration, choose
server type **Generic OpenAI-Compatible**. In the Conversation Agent options,
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
Tool calling uses **Gemma through LiteRT-LM**, including its native model chat
template and function parser. Qwen's Hailo VLM path remains available for text
and images, but rejects tool requests with a clear error.

The server passes the supplied function schemas to LiteRT-LM with
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
