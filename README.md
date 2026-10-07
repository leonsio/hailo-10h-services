# Hailo-10H Services

A resident inference gateway for **Hailo-10H** that exposes local VLM, LLM, speech,
embedding and object-detection models through a small set of reusable APIs.

The service supports **Qwen VLMs**, native **Hailo HEF LLMs**, **Whisper**, **MiniLM**
and resident **YOLO** detection. Optional **Gemma 4 E2B** runs independently on the
CPU through LiteRT-LM. Hailo accelerator workloads use `VDevice group_id="SHARED"` so
they can coexist with compatible Hailo applications without per-request model reloads.

> **HailoRT / GenAI 5.4.0 limitation:** one Hailo GenAI language model may be resident
> at a time. Configure **either a VLM or a native Hailo LLM**, not both. The service is
> architecturally prepared to expose both backends in parallel when Hailo supports that
> combination, but it does not unload/swap VLM and LLM models per request. Gemma runs on
> the CPU and can therefore be used together with a Hailo VLM today.

## What it provides

| Capability | Models / backend | Main interfaces |
|---|---|---|
| Vision-language | Qwen2-VL-2B-Instruct, Qwen3-VL-2B-Instruct | `/v1/chat/completions`, MCP, WebSocket, MQTT |
| Native Hailo LLM | Qwen, Llama, DeepSeek and other HEF LLMs from the model catalog | `/v1/chat/completions`, WebSocket, MQTT |
| CPU LLM | Gemma 4 E2B through LiteRT-LM | `/v1/chat/completions`, WebSocket, MQTT |
| Text-to-speech | Piper voices (CPU, configurable language) | `/v1/audio/speech`, Wyoming |
| Speech-to-text | Whisper Tiny / Base / Small | `/v1/audio/transcriptions`, Wyoming, MCP |
| Object detection | YOLOv8, YOLO11, YOLO26 | `/v1/vision/detect`, Frigate ZMQ |
| HA / Frigate routing | [HA](#home-assistant-virtual-model)/[Frigate-Assist](doc/frigate-assist.md), deterministic tools, LLM/VLM routing | OpenAI-compatible chat; Frigate experimental |

Native Hailo LLMs run directly through `hailo_platform.genai.LLM`; **Ollama is not
required**. The service owns model loading, input budgeting, queues and validated tool
calling, while clients such as Home Assistant remain responsible for executing their
own actions.

## Architecture

```text
                                            HTTP / OpenAI-compatible
                                     +---- /v1/chat/completions
                                     |     /v1/audio/transcriptions
                                     |     /v1/vision/detect
                                     |
                   Home Assistant ---+---- Wyoming / MCP
                   Frigate ----------+---- ZMQ detector
                   Other clients ----+---- WebSocket / MQTT
                                     |
                                     v
        +----------------------------------------------------------+
        |                    Hailo-10H-Services                    |
        |                                                          |
        | +----------------------+  +----------------------------+ |
        | |         CPU          |  |         Hailo-10H          | |
        | |                      |  |                            | |
        | |  HA/Frigate-Assist   |  |   Qwen VLM or Hailo LLM    | |
        | |  Gemma / LiteRT-LM   |  |  Whisper / MiniLM / YOLO   | |
        | |        Piper         |  |      (VDevice SHARED)      | |
        | +----------------------+  +----------------------------+ |
        +----------------------------+-----------------------------+
                                     |
                             Raspberry Pi 5/CM5
```

The **VLM/LLM choice above applies only to Hailo GenAI models**. The software contains
separate VLM and Hailo-LLM adapters, routing, queues and model selection and is ready
for simultaneous operation once the Hailo runtime supports it. With HailoRT 5.4.0,
set one of `models.vlm.enabled` and `models.hailo_llm.enabled` to `false`.

HTTP and ZMQ object-detection clients use the **same resident YOLO runtime**. Frigate
does not need to open the Hailo device or load its own detector HEF when it uses the
ZMQ bridge.

## Supported model families

The bundled `src/hailo_services/model_catalog.yaml` is the single source for supported
downloads, filenames, releases and model metadata. Enabled catalog models are downloaded
on first start and valid cached files are reused.

### Hailo LLMs

The current catalog includes, among others:

- `Qwen3-1.7B-Instruct`
- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-Coder-1.5B-Instruct`
- `Qwen2-1.5B-Instruct`
- `Qwen2-1.5B-Instruct-Function-Calling-v1`
- `Llama3.2-1B-Instruct`
- `DeepSeek-R1-Distill-Qwen-1.5B`

These models use their own HEF prompt template and tokenizer. Current catalog Hailo
LLMs have a compiled **2048-token context** shared by prompt and generated output.
Tool requests are converted to the model-specific compact contract and validated again
before OpenAI-compatible `tool_calls` are returned.

With HailoRT 5.4.0, enabling a native Hailo LLM requires the VLM to be disabled. This
is a Hailo GenAI runtime limitation, not a limitation of the service's routing design.

### VLM, speech and detection

- VLM: `Qwen2-VL-2B-Instruct`, `Qwen3-VL-2B-Instruct`
- STT: `Whisper-Tiny`, `Whisper-Base`, `Whisper-Small`
- YOLO: `yolov8n/s/m`, `yolov11n/s/m`, `yolo26n/s/m`

Qwen2-VL defaults to the smaller v5.1.1 HEF. Qwen3-VL and the other catalog models use
the configured/runtime-matched release where available. YOLO catalog models use COCO-80
labels and 640×640 input.

## Install and update

Requires working HailoRT/HailoRT GenAI matching the selected models and access to
`/dev/h1x-0`.

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

The primary configuration file is `/etc/hailo-10h-services.yaml`. Existing
`HAILO_<SETTING>` environment variables remain optional overrides.

### Docker Compose

Build a Debian 13 image with the matching local HailoRT DEB/Wheel and deploy with
`docker compose up -d --build`. Models and runtime state use persistent volumes.
See [Docker setup and package placement](doc/installation.md#docker-compose).

### Proxmox LXC / Raspberry Pi 5 / CM5

Run `scripts/install-proxmox-lxc.sh` on the Proxmox host to create a Debian 13 ARM64
container, pass through the Hailo device and install HailoRT plus this service. Place
the matching HailoRT DEB and Python wheel under `/root` on the host first.
See [LXC setup and command example](doc/installation.md#proxmox-lxc-on-arm64).

## Configuration example

This profile keeps the **VLM on Hailo** and uses **Gemma on CPU for text**, so
`HA-Assist` can handle both text and images without violating the HailoRT 5.4.0
VLM/LLM restriction.

```yaml
settings:
  host: 0.0.0.0
  port: 8090
  api_key: ""
  ha_assist_enabled: true
  ha_assist_text_model: gemma-4-E2B-it
  ha_assist_vision_model: Qwen2-VL-2B-Instruct
  ha_assist_verify_attempts: 2
  ha_assist_verify_delay: 0.5
  litert_max_num_tokens: 16384

models:
  vlm:
    enabled: true
    model: Qwen2-VL-2B-Instruct
    max_input_tokens: 2048

  # HailoRT 5.4.0: do not enable this while the VLM above is enabled.
  hailo_llm:
    enabled: false
    model: Qwen3-1.7B-Instruct
    release: auto
    max_input_tokens: 2048

  gemma:
    enabled: true
    max_input_tokens: 4096

  whisper:
    enabled: true
    model: Whisper-Base

  minilm:
    enabled: true

vision:
  enabled: true
  model: yolov11m
  release: auto
  confidence: 0.4
  max_detections: 20
  queue_size: 16
  scheduler_priority: 1
  zmq:
    enabled: true
    endpoint: tcp://127.0.0.1:5555
```

For a **native Hailo LLM text-only profile**, disable `models.vlm`, enable
`models.hailo_llm`, and point `ha_assist_text_model` at that LLM. Image requests then
have no HA-Assist VLM backend until the configuration is switched or a future Hailo
runtime supports both GenAI model types concurrently.

`vision.path` may be used instead of `vision.model`/`release` for an explicit HEF.
Do not configure both at once. ZMQ has no API-key authentication; bind it to loopback
or a trusted/firewalled network when Frigate runs elsewhere.

## LLM selection and routing

`GET /v1/models` lists the models that are currently available through the service.
For `/v1/chat/completions`, an explicit `model` ID always selects that backend and
never silently falls back to another model.

If `model` is omitted, text routing prefers:

1. ready Gemma/LiteRT-LM,
2. an enabled native Hailo LLM,
3. a resident VLM as the final text-capable backend.

The second and third entries cannot both be Hailo-resident under HailoRT 5.4.0. The
ordering is retained because the service is already structured for future parallel
Hailo VLM/LLM support.

Image requests require the configured VLM. `HA-Assist` is different: it is a virtual
model that first attempts deterministic Home Assistant handling and only then routes
text to `settings.ha_assist_text_model` or image requests to
`settings.ha_assist_vision_model`.

Gemma's service input ceiling defaults to **4096 tokens** while its LiteRT engine uses
a larger total context allocation (`litert_max_num_tokens`, default 16384). Hailo LLM
and VLM requests are bounded by their compiled 2048-token contexts. A request may lower
`max_input_tokens`, but cannot raise the configured/model limit.

## API overview

| Protocol | Endpoint / port | Chat / VLM / LLM | STT | YOLO detection |
|---|---|---|---|---|
| HTTP | `:8090/v1/chat/completions` | Text, images, SSE, tools | — | — |
| HTTP | `POST :8090/v1/ha-assist/diagnose` | HA candidates, proposed calls and prepared prompt; no generation/execution | — | — |
| HTTP | `:8090/v1/audio/transcriptions` | — | File upload | — |
| HTTP | `POST :8090/v1/audio/speech` | — | Text → Piper CPU speech (WAV/PCM) | — |
| HTTP | `:8090/v1/vision/detect` | — | — | Base64/data-URL image |
| HTTP | `/v1/models`, `/health` | Models/readiness | Models/readiness | Model/readiness |
| Frigate ZMQ | configured `vision.zmq.endpoint` | — | — | Frigate detector protocol |
| WebSocket | `ws://HOST:8090/ws` | `chat` | `transcribe` | — |
| MCP | `http://HOST:8090/mcp/` | `analyze_image`, `chat_text` | `transcribe_audio` | — |
| MQTT | `hailo10h/request/...` | `chat` | `transcribe` | — |
| Wyoming | `HOST:10300` | — | Home Assistant Assist STT (Whisper) + TTS (Piper CPU) | — |

See [API and integrations](doc/api.md) for complete schemas, authentication,
Frigate configuration, tool calling, metrics and Home Assistant details.

## YOLO and Frigate

For optional GenAI chat and descriptions, select the virtual **`Frigate-Assist`**
model. It removes Frigate's generic prompt overhead, selects read tools, handles
recognized requests deterministically, and routes text to a selected LLM (default: Gemma/CPU) and images to
the resident VLM. Both LLM/VLM targets are configurable, as with HA-Assist.
**Chat is experimental and only conditionally usable.** Frigate
can use separate providers for descriptions and chat, or both roles can be tried
through this proxy. See [Frigate-Assist configuration and limitations](doc/frigate-assist.md).

The selected YOLO model is configured once and kept resident. `/v1/vision/detect` and
the Frigate ZMQ bridge share that runtime and its queue.

Example service configuration:

```yaml
vision:
  enabled: true
  model: yolov11m
  zmq:
    enabled: true
    endpoint: tcp://0.0.0.0:5555
```

Frigate 0.17+ can use its built-in ZMQ detector. The model path is used by Frigate to
derive the model name for the startup handshake; when the configured resident model
matches, the service reports it as already loaded and no model upload is required.

```yaml
detectors:
  hailo10h:
    type: zmq
    endpoint: tcp://HAILO_SERVICE_HOST:5555
    request_timeout_ms: 5000
    linger_ms: 0

model:
  model_type: yolo-generic
  width: 640
  height: 640
  input_tensor: nhwc
  input_pixel_format: rgb
  input_dtype: int
  path: /config/models/yolov11m.hef
  labelmap_path: /labelmap/coco-80.txt
```

Use `tcp://HAILO_SERVICE_HOST:5555` for a remote service; replace the host placeholder
with its address (for example, `tcp://192.168.1.9:5555`). `ipc://` uses a local
socket path and cannot connect to the TCP listener.

| Frigate parameter | Example value | Meaning |
| --- | --- | --- |
| `request_timeout_ms` | `5000` | Allow up to 5 seconds for a request; use this initial value for diagnosis, then tune against measured inference times. |
| `linger_ms` | `0` | Discard pending unsent messages when the socket closes instead of waiting. |
| `model.path` | `/config/models/yolov11m.hef` | Use a path without leading whitespace; its basename must identify the resident model. |

These timeout and linger options belong to the **Frigate detector configuration**,
not the service-side `vision.zmq` settings.


The basename in `model.path` must match the selected service model, for example
`yolov11m.hef`. The file is only needed by Frigate if the remote detector reports that
the model is unavailable; this service intentionally rejects remote HEF uploads and
keeps model lifecycle under server configuration.

## Home Assistant virtual model

Select **`HA-Assist`** in an OpenAI-compatible Home Assistant conversation agent.
`HA-Assist` has no weights of its own. It performs conservative deterministic routing,
HassIL matching, MiniLM retrieval/context reduction and action validation before using
one configured text or vision backend when inference is necessary.

For HailoRT 5.4.0, the recommended full text+image profile is **Gemma on CPU for text**
plus **Qwen VLM on Hailo for images**:

```yaml
settings:
  ha_assist_enabled: true
  ha_assist_text_model: gemma-4-E2B-it
  ha_assist_vision_model: Qwen2-VL-2B-Instruct
  ha_assist_verify_attempts: 2
  ha_assist_verify_delay: 0.5

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

A native Hailo LLM may instead be used as the HA-Assist text backend, but with HailoRT
5.4.0 the VLM must then be disabled. The code paths are already separated so both Hailo
backends can be enabled together once Hailo removes the current runtime limitation.

A deterministic request performs **zero generative calls**. For generated actions the
service returns validated function calls; Home Assistant executes them using its own
permissions. The gateway never receives or needs a Home Assistant access token.

For successful `light`/`switch` on/off actions, HA-Assist may verify the reported live
state with `GetLiveContext`. `ha_assist_verify_attempts` controls the number of state
reads (`2` by default, `0` disables verification) and `ha_assist_verify_delay` controls
the delay before each read (`0.5` seconds by default). A stale state causes another
**verification read**, not another `HassTurnOn`/`HassTurnOff` action. If Home Assistant
returns explicit failed targets, that is reported as an action failure immediately;
if the action was accepted but the state still differs after all reads, the response
states that the action completed while the requested state could not yet be confirmed.

See [How HA-Assist works](doc/ha-assist.md), [request pipelines](doc/pipelines.md) and
[Home Assistant API configuration](doc/api.md#home-assistant-virtual-model).

## Browser playground and metrics

The browser UI keeps visible history when switching chat models. It shows request and
response timestamps with milliseconds, browser/server processing duration and model
metrics where available. These include inference time, TTFT, tokenizer-derived input
and output counts and input-budget measurements. Missing native metrics remain
unavailable rather than being estimated.

## Documentation

- [Technical architecture and file responsibilities](doc/architecture.md)
- [Docstring conventions and IDE support](doc/docstring-style.md)

- [How HA-Assist works and why it does not send everything to an LLM](doc/ha-assist.md)
- [Frigate-Assist: experimental chat and descriptions](doc/frigate-assist.md)
- [APIs, LLM/VLM, YOLO/Frigate and Home Assistant](doc/api.md)
- [Installation, Gemma provisioning and HTTPS](doc/installation.md)
- [Routing and deterministic pipelines](doc/pipelines.md)
- [Languages and Wyoming/HA language selection](doc/languages.md)
- [Model evaluation on Raspberry Pi 5 + Hailo-10H](doc/model-benchmark-evaluation.md)
- [Diagnostics, memory and hardware checks](doc/troubleshooting.md)

## Verification

```bash
pip install -e '.[test]'
ruff check .
pytest -q
node --test tests/test_web.cjs
bash -n scripts/install.sh scripts/update.sh scripts/enable-https.sh
```

Protocol/routing tests use simulated backends. Accelerator initialization, actual model
quality, measured inference performance and Frigate end-to-end latency require target
hardware testing.

### Optional CPU speech output (Piper)

Enable `settings.piper_enabled` after provisioning a local voice. The default
is German `de_DE-thorsten-medium`; language follows the installed voice model.
See [Piper installation](doc/installation.md#piper-cpu-text-to-speech) and the
[`/v1/audio/speech` API](doc/api.md#post-v1audiospeech).

The Playground includes a Piper TTS test with installed voice selection, speed,
WAV playback/download and retained request timings. Piper must be enabled and ready.

## License and third-party components

The **hailo-10h-services source code** is licensed under the
[Apache License 2.0](LICENSE) (`Apache-2.0`).

The project does **not** ship model weights, HEFs, Piper voice models or Hailo vendor
runtime binaries as part of its source distribution. Required third-party software is
installed separately through its upstream package source, and configured model assets
are downloaded at install/runtime from external provider URLs when needed.

Those external components keep their own licenses and are not relicensed by this
project. The bundled model catalog contains identifiers, metadata and download locations;
it is not a license grant for the referenced assets. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details and redistribution notes.
