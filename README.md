# Hailo-10H Services

One resident gateway for **Qwen2-VL / Qwen3-VL**, **Whisper Tiny/Base/Small**,
**MiniLM**, native **Hailo HEF LLMs** (such as Qwen2.5-1.5B or Qwen3-1.7B),
and optional **Gemma 4 E2B on CPU**. Hailo LLMs run directly through
`hailo_platform.genai.LLM`; no Ollama server is required.
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
text uses ready Gemma, then an enabled Hailo LLM, then the resident VLM; images
require an enabled VLM. Explicit model IDs select that backend and never fall back.
Gemma can remain enabled alongside either accelerator model. The service does not
enforce VLM/HEF-LLM mutual exclusion; choose the enabled models for your hardware.

To use a native Hailo text model alongside Gemma, edit the existing YAML:

```yaml
models:
  vlm:
    enabled: false
  hailo_llm:
    enabled: true
    model: Qwen2.5-1.5B-Instruct # Or Qwen3-1.7B-Instruct, etc. from the catalogue.
    max_input_tokens: 2048
  gemma:
    enabled: true
    max_input_tokens: 4096
```

After restarting, `/v1/models` and the playground list both text models. Send
`"model": "Qwen2.5-1.5B-Instruct"` or `"model": "gemma-4-E2B-it"` to select one.
Hailo inference uses the existing SHARED owner thread; Gemma keeps its own CPU
engine and queue. Hailo LLM input and output share the compiled 2048-token context.

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
  device control, a configured Hailo LLM such as `Qwen2.5-1.5B-Instruct` for
  text/tools, or Qwen2/Qwen3-VL for text/tools and images. Use the configured API key.
- Wyoming STT: port **10300**, the selected multilingual Whisper model.
- MCP `/mcp`, WebSocket `/ws`, MQTT and HTTPS are supported.

The playground keeps the visible chat and its text context when switching models
or chat modes. Gemma and Hailo LLMs receive text without image attachments; images stay in the
visible history and remain available to the VLM. **New chat** clears the history.
Every chat/transcription request shows timestamps with milliseconds, a live timer
and its final browser/server duration. Available model metrics include request and
inference duration, TTFT where measurable, and tokenizer-based input/output token
counts. Missing values are marked unavailable and their source is labeled. See
[metric definitions](doc/api.md#request-metrics).

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
General Hailo LLM/VLM tool requests also retrieve relevant tools before budgeting. Older
complete turns may be removed; system messages and the active tool round remain.
Streaming HA requests can receive a varying localized wait sentence at inference
start. Tool output remains buffered until validated. Spoken early playback also
requires streaming support in the HA agent and TTS provider.

## Documentation

- [Installation, Gemma provisioning and HTTPS](doc/installation.md)
- [APIs and Home Assistant integration](doc/api.md)
- [Routing and deterministic pipelines](doc/pipelines.md)
- [Languages, vocabulary and Wyoming/HA language selection](doc/languages.md)
- [Model evaluation on Raspberry Pi 5 + Hailo-10H](doc/model-benchmark-evaluation.md)
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
