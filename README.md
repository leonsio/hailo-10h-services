# Hailo-10H Services

Resident **Qwen2-VL-2B-Instruct** and multilingual **Whisper Base** for Home Assistant
and other applications. Native HailoRT GenAI inference; no CPU inference fallback.
Every accelerator connection uses **`VDevice group_id="SHARED"`**, hardcoded.

## Status page and browser playground

Open **`http://<cm5-IP>:8090/`** to view service readiness, the effective SHARED
group, pending jobs (including active inference), loaded models, model paths and
MQTT connectivity. Status refreshes every five seconds. All page assets ship with
the Python package; no external CDN, Node build or extra service is needed.

The page includes:

- **VLM chat:** text questions, optional JPEG/PNG/WebP images, conversation history,
  token limit and response time. Start a new chat after 16 turns or four images.
- **Whisper:** start/stop a microphone recording (up to 30 seconds, or the configured
  lower audio/body limit), local playback, language selection and transcription.
  Alternatively upload WAV, FLAC or OGG. Audio is sent only when clicking
  **Transkribieren**. Browser recordings use mono PCM16 WAV, decoded/resampled by
  the existing audio API; no WebM/AAC decoder or ffmpeg is needed.
- **API key:** enter `HAILO_API_KEY` from `/etc/hailo-10h-services.env` if enabled.
  The key is kept only in the open page's memory, never in localStorage or a URL.
  Chat history, selected media and transcripts are also kept only in the page.

Browsers permit microphone capture only in a **secure context**, such as HTTPS
or localhost. A plain HTTP LAN address permits status/chat/uploads but cannot
capture the microphone. Use an HTTPS reverse proxy with a browser-trusted
certificate, or on your desktop forward the service through SSH:

```bash
ssh -N -L 18090:127.0.0.1:8090 leonsio@<cm5-IP>
# Open http://localhost:18090/ in the browser on that same desktop.
```

On a phone, use HTTPS for microphone capture. Recording requires a modern browser
with AudioWorklet support and microphone permission. The page explains when
capture is unavailable. Capturing stops when the page moves to the background.
See [MDN's microphone secure-context requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

Only the page's exact read-only asset routes, `/ui/config` (non-secret limits/model
names) and the existing `/health` are public. Inference endpoints retain their
Bearer authentication and body limits. The page becomes available after normal
service startup, when both models have loaded successfully.

To update an existing installation from the branch containing this change:

```bash
git pull --ff-only
sudo /opt/hailo-10h-services/venv/bin/pip install --no-deps --force-reinstall .
sudo systemctl restart hailo-10h-services
```

## HTTPS with a local self-signed CA

On an existing installation, enable the additional HTTPS listener with:

```bash
cd ~/hailo-10h-services
git switch main
git pull --ff-only
sudo /opt/hailo-10h-services/venv/bin/pip install --no-deps --force-reinstall .
sudo systemctl restart hailo-10h-services
sudo bash scripts/enable-https.sh
```

Open **`https://<cm5-IP>:8443/`**. HTTP on port 8090 remains available for existing
clients. nginx terminates TLS and forwards requests to the same resident service;
it does not start another Hailo process. The setup installs nginx/OpenSSL if needed,
adds one nginx server block and preserves existing sites. SSE is unbuffered and
WebSocket upgrades are forwarded. API authentication and limits remain enforced.

Certificates are generated under `/etc/hailo-10h-services/tls`. A self-signed local
root CA signs a 365-day server certificate with Subject Alternative Names for
localhost, the machine hostname, its `.local` name and detected interface IPs.
Add other names or addresses, or choose port 443, explicitly:

```bash
sudo bash scripts/enable-https.sh --dns cm5.example.lan --ip 192.168.1.42 --port 443
```

Re-running the script reuses the CA and valid matching certificates. It renews
the server certificate when names/IPs change or less than 30 days remain; there
is no automatic renewal timer. Keep the CA private key on the server. Both private
keys are root-readable only. The public CA can be downloaded from
`https://<cm5-IP>:8443/hailo-ca.crt`; its SHA-256 fingerprint is printed during setup.

**Trust is required for reliable microphone capture.** A browser warning is
expected until the local CA is installed and trusted. Simply clicking through a
certificate warning may still prevent microphone access.

For an iPhone/iPad:

1. In Safari open `https://<cm5-IP>:8443/hailo-ca.mobileconfig` and download the
   certificate profile (initially acknowledge the certificate warning).
2. Install **Hailo-10H Local CA** via Settings → General → VPN & Device Management
   (or the **Profile Downloaded** entry).
3. Under Settings → General → About → Certificate Trust Settings, enable full
   trust for **Hailo-10H Local CA**. Compare the certificate fingerprint with the
   value printed on your server before trusting it.
4. Reload the HTTPS page in Safari and allow microphone access.

On a desktop, import `hailo-ca.crt` into the trusted root certificate store used
by the browser. Remove this trust/profile when the local CA is no longer needed.
See [Apple's manual certificate-trust instructions](https://support.apple.com/102390).

Verify from the server without bypassing certificate validation:

```bash
curl --cacert /etc/hailo-10h-services/tls/hailo-ca.crt https://localhost:8443/health
sudo nginx -t
systemctl status nginx
```

If a firewall is enabled, allow the chosen HTTPS port on the local network. After
an IP change, run setup again and use a name/address included in the certificate.

## Behavior and architecture

One systemd service, one process, one native owner thread, one SHARED VDevice,
two persistent model objects. The HTTP, WebSocket, MCP, MQTT and Wyoming adapters
all call this same runtime; no extra inter-process transport is needed. Models
are constructed once at startup and released only on shutdown. No per-request
load/unload and no idle eviction. VLM KV context is cleared between requests so
clients cannot inherit another client's conversation. Supply history in `messages`.

Startup resolves/downloads both compiled HEFs through the **official hailo-apps
resource resolver** before opening the listeners. Defaults:

- `/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef`
- `/usr/local/hailo/resources/models/hailo10h/Whisper-Base.hef`

Existing files are reused. First start needs internet and enough disk space for
both HEFs plus download temporary files. Later starts work offline with cached
models. Raw Hugging Face weights cannot be substituted for compiled Hailo HEFs.
Absolute HEF paths can be configured for version-matched files already installed.

The optional Home Assistant/Gemma path also downloads
`minilm-l6-ruvector.hef` from the public `cstr/all-MiniLM-L6-v2-hailo10h` model
repo on first start to
`/usr/local/hailo/resources/models/hailo10h/minilm-l6-ruvector.hef`. An existing
non-empty file is reused. The tokenizer and original MiniLM weights are cached
beside the HEF on first start (about 90 MB for the weights). The service extracts
the CPU embedding and LayerNorm tensors and keeps the encoder configured on the
same `SHARED` Hailo device as Qwen and Whisper. Install the new Python dependencies
before restarting an existing installation:

```bash
sudo /opt/hailo-10h-services/venv/bin/pip install 'safetensors>=0.4,<1' 'tokenizers>=0.20,<1'
sudo /opt/hailo-10h-services/venv/bin/pip install --no-deps --force-reinstall .
sudo systemctl restart hailo-10h-services
```

### Home Assistant tool-context reduction

For Gemma requests, MiniLM first reduces Home Assistant's generated
`Static Context`: the complete incoming entity catalogue is inspected, but only
the entities relevant to the latest user request are forwarded to Gemma (up to
eight candidates). The remaining system instructions are preserved verbatim.

Tool retrieval then ranks the incoming function definitions with lexical matching
plus MiniLM cosine similarity. Normally at most four relevant tool definitions
are forwarded, while schemas already referenced by an ongoing tool round are
kept even when they fall outside that shortlist. Large entity enums are reduced
when names clearly match. Entity/tool embeddings are cached across requests so
stable Home Assistant metadata does not have to be recomputed every turn.

Every Gemma request is capped at 4096 input tokens, even when
the client does not send `max_input_tokens`. Old complete chat rounds are removed
first. The rendered message text and selected tool JSON are tokenized, with a
256-token margin for LiteRT's template overhead, while requested output tokens
are reserved from the configured context. If the required system/current-turn
content still exceeds the budget, the API returns `input_token_limit_exceeded`
without calling Gemma; it never silently truncates the active request. You may
configure a lower cap with `HAILO_LITERT_MAX_INPUT_TOKENS`.

All Hailo model handles stay open for the service lifetime. The runtime does not
control firmware paging or guarantee physical allocation of every byte at all
times; constructor success and `/health` mean the native model instances loaded.
Insufficient accelerator memory or incompatible HEFs cause startup to **fail**,
without silently falling back, switching models, or unloading Qwen to run Whisper.
Actual coexistence/residency must be validated on the target Hailo-10H.

Requests are serialized through a bounded queue (8 including active by default).
HTTP overload returns 503, timeout 504. A timeout/cancel does not release the slot
while native inference is still running. SSE uses native VLM generation chunks.
Streaming failures are reported inside the SSE stream after headers were sent.

Other Hailo applications must also use `group_id="SHARED"`. This applies to
Frigate and the meter reader too. `SHARED` permits sharing but does not reserve
memory or throughput. Another resident LLM/VLM cannot coexist due to Hailo's
exclusive KV cache; detection models and Whisper still consume shared resources.
Do not run multiple uvicorn workers, reload mode or duplicate service instances.
On Hailo-10H this follows the official GenAI sharing pattern; it does not blindly
enable the Hailo-8-specific `multi_process_service` flag.

## Install on Debian / Raspberry Pi / Proxmox LXC

Prerequisites: working Hailo-10H kernel driver, firmware, matching **HailoRT 5.x
GenAI Python wheel** for your architecture and Python version. `hailo_platform`
with `VLM` and `Speech2Text` must already import. The installer does not replace
your kernel driver, HailoRT or firmware. In LXC, `/dev/h1x-0` must first be passed
through by the Proxmox host; a service inside LXC cannot grant itself that device.
For sharing with host/other containers, they need compatible HailoRT libraries
and matching group IDs too; validate sharing across those boundaries on hardware.

```bash
git clone https://github.com/leonsio/hailo-10h-services.git
cd hailo-10h-services
sudo bash scripts/install.sh
# If the Hailo wheel lives in another Python environment:
# sudo HAILO_PYTHON=/path/to/hailo/venv/bin/python bash scripts/install.sh
journalctl -u hailo-10h-services -f
curl http://127.0.0.1:8090/health
```

The installer creates a dedicated `hailo-services` user, a virtual environment
with system packages visible, device/resource ACLs, an env file and systemd unit.
It installs a pinned official hailo-apps revision with only its resource helper
dependencies, avoiding unrelated camera/TTS/PyTorch packages. Native Hailo wheel
imports are checked again as the service user. For a wheel installed in a custom
venv, the installer adds that environment’s vendor package directory to the
service Python path when necessary. The Python ABI must match. Protected home
paths are not visible to the systemd service; keep vendor environments under
`/opt` or install the matching vendor wheel directly into the service venv.

First installation generates an API key in `/etc/hailo-10h-services.env`.
Configuration is preserved on reinstall. The account's home and working directory
are `/var/lib/hailo-10h-services`, managed by systemd `StateDirectory`. Hailo can
write `$HOME/.hailo` and cwd log files there. `ProtectSystem=full` stays enabled
with a `ReadWritePaths` exception for the shared model directory. ACLs grant the
service user access to that exception. A startup preflight checks real temporary
file creation and rename in these locations inside the systemd sandbox. Reinstall
also migrates the original `/nonexistent` account home and clears failed-start
rate limiting. Read/edit this file as root, then:

```bash
sudo systemctl restart hailo-10h-services
sudo systemctl status hailo-10h-services
```

`/health` is the readiness check, not merely `systemctl is-active`: while HEFs are
being downloaded/loaded there is no listening HTTP socket. It includes loaded
model names, paths, mandatory SHARED group, pending work and MQTT connection state.

Hailo's pinned downloader knows model releases v5.1.0/v5.2.0/v5.3.0. This service
explicitly selects the newest known release not newer than the loaded HailoRT Python binding, and
logs that choice. Version detection reads `hailo_platform.__version__` without
opening the device through `hailortcli`. For HailoRT 5.4 it selects v5.3.0 instead of the upstream silent
v5.1 fallback. This is a candidate release, **not a claim of tested hardware
compatibility**; HailoRT validates the HEFs when constructing the models. You can
set `model_zoo_version=v5.2.0` in the env file for a known matching release, or use
explicit HEFs. Set `hailort_version=5.4.0` only if detection fails and that is your
actual library version. Existing HEFs are reused even when the release setting
changes; move stale/incompatible files out of the store before re-downloading.

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
Base as speech-to-text in your Assist pipeline, language German (`de`). The service
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

## Limits, verification and target-device checks

Default total request size 16 MiB, audio duration 120 seconds, images 50 million
pixels. JPEGs are downsampled during decoding before conversion to the VLM's
336×336 input, which supports typical 48 MP phone photos without allocating the
full RGB image. Larger images are rejected before conversion. Wyoming accepts PCM16 mono/stereo at 8–192 kHz, bounded audio chunks and
up to 32 connections. Large uploads/durations and incomplete PCM are rejected.
Timeouts cannot forcibly interrupt a stuck native driver call; systemd's stop
limit eventually terminates the process if native shutdown cannot finish.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/ruff check .
.venv/bin/pytest -q
bash -n scripts/install.sh
```

CI exercises adapters using a fake native backend: HTTP, SSE, WebSocket, MCP
JSON-RPC, real Wyoming TCP framing, audio normalization, queue timeout/cancel
semantics, SHARED VDevice construction and partial-startup release order. MQTT
dispatch is tested; a live broker and actual Hailo inference require deployment
checks. The installer/systemd unit has static checks, not a target installation
test. No Hailo hardware is available in the development environment.

On your machine, verify both model paths in `/health`, make an image request and
a German recording request, repeat them while the meter reader/Frigate runs,
and inspect logs for sharing/memory errors. Check first-start downloads and an
offline restart. Report HailoRT/device logs if either model fails to initialize;
the service intentionally does not hide that by unloading the other model.

### Protocol debug logging

Set `HAILO_DEBUG_LOG=true` in `/etc/hailo-10h-services.env`, then restart the
service to log request start/end, transport (`http`, `websocket`, `wyoming`,
`mqtt`, or `mcp`), operation, request ID, status and duration. Whisper diagnostics
include model/language and audio container, codec, sample rate, channels, duration
and byte count. Wyoming additionally logs PCM encoding and input sample rate.
Chat logs include model, message/image counts and token limit. Prompts, transcripts,
API keys and raw audio/image payloads are never logged. Debug logging is off by
default; disable it with `HAILO_DEBUG_LOG=false`.

```bash
sudoedit /etc/hailo-10h-services.env
sudo systemctl restart hailo-10h-services
sudo journalctl -u hailo-10h-services -f
```

### Optional Gemma 4 E2B through LiteRT-LM

The service can also keep a LiteRT-LM `Engine` resident and route text-only
OpenAI-compatible chat requests to it. LiteRT-LM runs explicitly on CPU in its
own single-worker queue; Qwen2-VL and Whisper continue to use the Hailo `SHARED`
device. This uses LiteRT-LM's Python API rather than starting a second HTTP
server. The model is loaded once at service startup and each API request gets a
fresh conversation populated from the supplied message history.

Configure `/etc/hailo-10h-services.env`:

```bash
HAILO_LITERT_MODEL_PATH=/home/leonsio/.litert-lm/models/gemma-4-E2B-it.litertlm
```

Then run the installer/update script so the service virtual environment has
`litert-lm` and the service account can read the model. Restart and check
`/health`; the `litert_lm.ready` field and `gemma-4-E2B-it` entry in `/v1/models`
should be present. If LiteRT-LM cannot load, Hailo service startup still
completes and the reason appears in `litert_lm.error` and the service log.
Leave the path empty to disable Gemma.

Chat requests select the backend using the model field:

```json
{"model":"gemma-4-E2B-it","messages":[{"role":"user","content":"Hallo!"}],"max_tokens":128}
```

Use `Qwen2-VL-2B-Instruct` for text and image analysis. Gemma is text-only.
Home Assistant or another OpenAI-compatible client can point to the same service
base URL (`https://<raspberry-pi>:8443/v1` with the default HTTPS setup or
`http://<raspberry-pi>:8090/v1` without HTTPS), use the service API key, and
select `gemma-4-E2B-it` or `Qwen2-VL-2B-Instruct`. For a self-signed certificate,
the client must trust that certificate. LiteRT-LM and
Gemma share system RAM with the service; verify memory headroom on the Pi before
raising the request queue size.

For Home Assistant, use the **LiteLLM** conversation integration with the service
URL, API key and model `gemma-4-E2B-it`. Enable Home Assistant control in the
conversation agent options and expose the devices you want Assist to control.

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
validated tool-call deltas; this adds latency before the first chunk.

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

- [Hailo GenAI examples](https://github.com/hailo-ai/hailo-apps/tree/main/hailo_apps/python/gen_ai_apps)
- [Hailo shared-device usage and KV-cache limitation](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/USAGE.rst)
- [Official Wyoming protocol](https://github.com/OHF-Voice/wyoming)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## Whisper backend choice and DMA startup failures

[The hailocs/hailo-whisper repository](https://github.com/hailocs/hailo-whisper)
provides Whisper export, conversion and evaluation, including separate encoder
and decoder graphs and host embedding/tokenization assets. Its documented Base
conversion uses five-second inputs and requires DFC 5.x for Hailo-10H. Those
compiled models and host-side routines are not drop-in replacements for the
single GenAI `Whisper-Base.hef` consumed by `Speech2Text`. This gateway follows
[Hailo's native Speech2Text example](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/gen_ai_apps/simple_whisper_chat/simple_whisper_chat.py).
A separate low-level encoder/decoder backend would require its own implementation
and hardware validation; it is not enabled by this comparison.

A failure in `VDevice(...)` or `VLM(...)` occurs before Whisper initialization.
`HAILO_TIMEOUT(4)` is a native device/communication timeout; increasing the HTTP
request timeout will not repair startup. For `HAILO_VDMA_ENABLE_CHANNELS` errno
22, the published driver rejects activation of channels already enabled; confirm
the actual kernel reason with `dmesg`. Potential competing device users, stale
channel state and mismatched kernel/runtime components need target investigation.
Do not change `group_id` to bypass sharing or assume a HEF swap fixes driver I/O.
Stop the gateway's restart loop and collect these commands on the Proxmox host
if the kernel/device is owned by the host:

```bash
sudo systemctl stop hailo-10h-services
sudo dmesg -T | grep -Ei 'hailo|h1x|vdma' | tail -100
sudo fuser -v /dev/h1x-0
modinfo hailo1x | grep -E '^(version|filename|vermagic):'
```

`fuser` reports candidates, not proof that another process is incorrectly sharing.
Verify those clients also use SHARED and compatible HailoRT before restarting.
Do not unload the kernel driver or reset the device while other clients use it.
