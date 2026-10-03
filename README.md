# Hailo-10H Services

Resident **Qwen2-VL-2B-Instruct** and multilingual **Whisper Base** for Home Assistant
and other applications. Native HailoRT GenAI inference; no CPU inference fallback.
Every accelerator connection uses **`VDevice group_id="SHARED"`**, hardcoded.

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

Both model handles stay open for the service lifetime. The runtime does not
control firmware paging or guarantee physical allocation of every byte at all
times; constructor success and `/health` mean both native model instances loaded.
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
explicitly selects the newest known release not newer than detected HailoRT, and
logs that choice. For HailoRT 5.4 it selects v5.3.0 instead of the upstream silent
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

HTTP, MCP and WebSocket require `Authorization: Bearer API_KEY` when configured.
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

Default total request size 16 MiB, audio duration 120 seconds, images 20 million
pixels. Wyoming accepts PCM16 mono/stereo at 8–192 kHz, bounded audio chunks and
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

Sources used for the implementation:

- [Hailo GenAI examples](https://github.com/hailo-ai/hailo-apps/tree/main/hailo_apps/python/gen_ai_apps)
- [Hailo shared-device usage and KV-cache limitation](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/USAGE.rst)
- [Official Wyoming protocol](https://github.com/OHF-Voice/wyoming)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
