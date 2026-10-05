# Hailo-10H Services

One resident gateway for **Qwen2-VL / Qwen3-VL**, **Whisper Tiny/Base/Small**,
**MiniLM** and optional **Gemma 4 E2B on CPU**.
Hailo HEF LLM execution is disabled pending hardware tests; its catalogue links
remain available for future support.
Qwen2-VL/Qwen3-VL handle images and text, Whisper speech, and MiniLM context
retrieval. Gemma optionally handles text/tool reasoning on CPU; otherwise the
resident VLM can handle text and validated function calls. Hailo models use `VDevice group_id="SHARED"`; Gemma runs
through LiteRT-LM on the CPU with its own serialized queue. Models stay loaded.

## Install and update

Requires working HailoRT GenAI, matching HEFs and access to `/dev/h1x-0`.
Select models in `/etc/hailo-10h-services.yaml`. Enabled models and MiniLM host
assets download automatically at startup; valid cached files are reused. All links
are centralized in `src/hailo_services/model_catalog.yaml`; no helper repository
is installed or imported. Qwen2-VL defaults to the smaller **v5.1.1 HEF**. Qwen3-VL and Whisper use
runtime-matched releases; VLM preprocessing reads the loaded model shape
(Qwen2: 336×336; Qwen3: 512×288, one image per request).

```bash
git clone https://github.com/leonsio/hailo-10h-services.git
cd hailo-10h-services
sudo bash scripts/install.sh
```

For an existing installation:

```bash
git pull --ff-only
sudo bash scripts/install.sh
```

Configure `/etc/hailo-10h-services.yaml`; an existing ENV file remains an optional
override. Gemma's default input ceiling remains
**4096 tokens**; Qwen2/Qwen3-VL use **2048 context tokens**, shared by input and
output. Set `max_input_tokens` beside each model in YAML. A client may lower,
but cannot raise, the configured ceiling. Larger HTTP requests with tools are
retrieved/compiled first; only the final model prompt must fit. `/health` reports
readiness, model limits and the default text model. When `model` is omitted,
text uses ready Gemma or the resident VLM; image requests use the VLM. An explicit
request for unavailable Gemma returns an error.

### Docker Compose

Build a Debian 13 image with the matching local HailoRT DEB/Wheel and deploy
with `docker compose up -d --build`. Models and runtime state use persistent
volumes; HTTP and Wyoming ports are published. Optional LiteRT-LM is included.
See [Docker setup and package placement](doc/installation.md#docker-compose).

### Proxmox LXC (Raspberry Pi 5 / CM5, ARM64)

Run `scripts/install-proxmox-lxc.sh` **on the Proxmox host** to create a Debian 13
container, pass through the Hailo device and install HailoRT plus this service.
Place the HailoRT DEB and matching Python wheel under `/root` on the host.
See [LXC setup and command example](doc/installation.md#proxmox-lxc-on-arm64).

## Usage

- Browser playground: `http://<host>:8090/` (choose **Text only** or text with
  an optional image; model input limits, speech and status).
- OpenAI clients/HA: `http://<host>:8090/v1`, model `gemma-4-E2B-it` for text and
  device control, or the configured Qwen2/Qwen3-VL for text/tools and images. Use the configured API key.
- Wyoming STT: port **10300**, the selected multilingual Whisper model.
- MCP `/mcp`, WebSocket `/ws`, MQTT and HTTPS are supported.

### API endpoint overview

| Protocol | Endpoint / port | VLM | LLM | Whisper |
|---|---|---|---|---|
| OpenAI-style HTTP | `:8090/v1/chat/completions` | Text, images, SSE, function tools | Text, SSE, function tools | — |
| OpenAI-style HTTP | `:8090/v1/audio/transcriptions` | — | — | File upload |
| Models / readiness | `/v1/models`, `/health` | Model status | Model status | Model status |
| WebSocket | `ws://HOST:8090/ws` | `chat` | `chat` | `transcribe` |
| MCP Streamable HTTP | `http://HOST:8090/mcp/` | `analyze_image`, `chat_text` | `chat_text` | `transcribe_audio` |
| MQTT (optional) | `hailo10h/request/chat`, `…/transcribe` | JSON requests | JSON requests | Base64 audio |
| Wyoming TCP | `HOST:10300` | — | — | Home Assistant Assist STT |

See [APIs and Home Assistant integration](doc/api.md) for authentication,
request formats, Home Assistant tool calling and protocol-specific details.

The UI uses the first supported browser language, with a persistent manual
language selector and the service default as fallback. German, English and
Russian resources live in `src/hailo_services/locales/`. `HAILO_SERVICE_LANGUAGE`
sets the general reply/UI fallback; `HAILO_LANGUAGE` sets Whisper's default.
HA requests detect their input language or accept an explicit `language` field.
Entity names and API identifiers retain the values supplied by HA.

Recognized HA requests use conservative deterministic paths first; ambiguous
requests use MiniLM and the minimal prompt compiler before model inference.
General VLM tool requests also retrieve relevant tools before budgeting. Older
complete turns may be removed; system messages and the active tool round remain.
Streaming HA requests can receive a varying localized wait sentence at inference
start. Tool output remains buffered until validated. Spoken early playback also
requires streaming support in the HA agent and TTS provider.

## Documentation

- [Installation, Gemma provisioning and HTTPS](doc/installation.md)
- [APIs and Home Assistant integration](doc/api.md)
- [Routing and deterministic pipelines](doc/pipelines.md)
- [Languages, vocabulary and Wyoming/HA language selection](doc/languages.md)
- [Diagnostics, memory and hardware checks](doc/troubleshooting.md)

## Verification

```bash
pip install -e '.[test]'
ruff check .
pytest -q
node --test tests/test_web.cjs
bash -n scripts/install.sh scripts/enable-https.sh
```

Protocol/routing tests use simulated backends; accelerator initialization,
measured inference times and audible Assist streaming require target hardware.
