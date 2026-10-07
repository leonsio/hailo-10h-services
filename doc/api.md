# API and integrations

Hailo-10H-Services exposes local inference through an OpenAI-compatible chat surface
plus dedicated speech, object-detection, Home Assistant, MCP, Wyoming, WebSocket,
MQTT and Frigate interfaces.

## Endpoint overview

| Protocol | Endpoint / port | VLM / LLM | Whisper | YOLO |
|---|---|---|---|---|
| HTTP | `POST :8090/v1/chat/completions` | Text, images, SSE, function tools | — | — |
| HTTP | `POST :8090/v1/ha-assist/diagnose` | HA preparation diagnosis; no generative calls or tool execution | — | — |
| HTTP | `POST :8090/v1/audio/transcriptions` | — | File upload | — |
| HTTP | `POST :8090/v1/audio/speech` | — | Text → Piper CPU speech (WAV/PCM) | — |
| HTTP | `POST :8090/v1/vision/detect` | — | — | Object detection |
| HTTP | `GET :8090/v1/models` | Ready model IDs | Ready model ID | Ready detector ID |
| HTTP | `GET :8090/health` | Backend/readiness status | Status | Status/ZMQ |
| Frigate ZMQ | `vision.zmq.endpoint` | — | — | `(20,6)` detector protocol |
| WebSocket | `ws://HOST:8090/ws` | `chat` | `transcribe` | — |
| MCP | `http://HOST:8090/mcp/` | `analyze_image`, `chat_text` | `transcribe_audio` | — |
| MQTT | `hailo10h/request/...` | `chat` | `transcribe` | — |
| Wyoming | `HOST:10300` | — | Home Assistant Assist STT (Whisper) + TTS (Piper CPU) | — |

The chat endpoint implements the subset of the OpenAI Chat Completions API required by
the supported clients. `/v1/vision/detect` is a service-specific OpenAI-style `/v1`
extension for object detection; it is not an OpenAI public API endpoint.

## HailoRT 5.4.0: VLM and native LLM are mutually exclusive

Hailo's GenAI documentation for the current 5.4.0 stack states that multiple LLM or
VLM models cannot run simultaneously on the same device. For this service that means:

- `models.vlm.enabled: true` and `models.hailo_llm.enabled: true` must **not** be used
  together on HailoRT 5.4.0;
- choose **one Hailo VLM** or **one native Hailo LLM** as the resident GenAI language
  model;
- Gemma/LiteRT-LM runs on the CPU and may be enabled together with either choice;
- the software already has independent VLM and Hailo-LLM adapters, model IDs, routing,
  queues and token-budget paths, so it is prepared for concurrent VLM+LLM operation
  when a future Hailo runtime supports that configuration;
- the service does not unload one GenAI model and load the other on demand.

The configuration parser intentionally remains future-ready and does not turn this
vendor limitation into a permanent schema rule. **For HailoRT 5.4.0 deployments, the
administrator must keep one of the two Hailo GenAI roles disabled.**

A useful full Home Assistant configuration today is therefore **Qwen VLM on Hailo +
Gemma on CPU**. A native Hailo LLM can alternatively be used for text-only operation
with the VLM disabled.

## Authentication and network exposure

When `settings.api_key` is configured, HTTP and WebSocket requests require:

```http
Authorization: Bearer API_KEY
```

`/health` and the browser UI remain public according to the service middleware rules.
MCP may omit the API key for configured peer networks in
`settings.mcp_no_auth_networks`; `settings.mcp_hosts` separately protects allowed Host
headers.

Wyoming and the Frigate ZMQ listener do not use the HTTP bearer token. Keep them on a
trusted network. In particular, **ZMQ is unauthenticated**: bind
`vision.zmq.endpoint` to `127.0.0.1` unless Frigate runs on another trusted host or
container and the port is protected by firewall/network policy.

MQTT uses its own broker credentials/TLS/ACLs.

## Model system

`GET /v1/models` lists models that are actually available through the running service.
An explicit `model` field in `/v1/chat/completions` is authoritative: an unknown,
disabled or unavailable model returns an error and is not silently replaced.

### Native Hailo LLM support

Native text LLMs run directly through `hailo_platform.genai.LLM`; no Ollama process is
required. The bundled model catalog currently contains Hailo LLM entries including:

- `Qwen3-1.7B-Instruct`
- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-Coder-1.5B-Instruct`
- `Qwen2-1.5B-Instruct`
- `Qwen2-1.5B-Instruct-Function-Calling-v1`
- `Llama3.2-1B-Instruct`
- `DeepSeek-R1-Distill-Qwen-1.5B`

Select one under `models.hailo_llm` and disable the VLM on HailoRT 5.4.0:

```yaml
models:
  vlm:
    enabled: false
  hailo_llm:
    enabled: true
    model: Qwen3-1.7B-Instruct
    release: auto
    max_input_tokens: 2048
```

The current catalog Hailo LLMs use a compiled **2048-token context**, shared by prompt
and generated output. The runtime renders the model's own prompt template and counts
the actual prompt with the model tokenizer. The effective input limit is the minimum
of the configured `max_input_tokens`, an optional lower request limit and the remaining
compiled context after output/template reserves.

Model-specific prompt behavior lives in `model_catalog.yaml`. For example,
`Llama3.2-1B-Instruct` omits empty tool-call fields and uses a single-call tool contract;
Qwen models use their configured template behavior. Thinking is disabled in native
rendered templates where the adapter supports that distinction.

### VLM support

The configured VLM is selected under `models.vlm`. On HailoRT 5.4.0 the native Hailo
LLM must remain disabled:

```yaml
models:
  vlm:
    enabled: true
    model: Qwen2-VL-2B-Instruct
    release: auto
    max_input_tokens: 2048
  hailo_llm:
    enabled: false
```

Supported catalog VLMs currently include `Qwen2-VL-2B-Instruct` and
`Qwen3-VL-2B-Instruct`. Their compiled context is also 2048 tokens. Qwen2 defaults to
the smaller v5.1.1 HEF and uses 336×336 frames; Qwen3 uses the model-provided frame
shape currently recorded in the model catalog.

### Gemma on CPU

Optional `gemma-4-E2B-it` runs through LiteRT-LM on the CPU and therefore does not
consume the Hailo GenAI VLM/LLM slot:

```yaml
settings:
  litert_max_num_tokens: 16384
models:
  gemma:
    enabled: true
    max_input_tokens: 4096
```

The service input ceiling defaults to 4096 prompt tokens. `litert_max_num_tokens`
controls the larger LiteRT total context allocation that also has to contain history,
tool schemas and generated output. Increasing it can materially increase RAM use.

### Default text routing

When a chat request omits `model`, text requests prefer the first ready backend in this
order:

1. Gemma/LiteRT-LM,
2. configured native Hailo LLM,
3. resident VLM as a text-capable fallback.

The order is intentionally generic and future-ready. With HailoRT 5.4.0, entries 2 and
3 cannot both be resident at once. Requests containing images require the configured
VLM. This ordinary default routing is separate from `HA-Assist`.

## `POST /v1/chat/completions`

Minimal text request:

```bash
curl -sS http://HOST:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${HAILO_API_KEY}" \
  -d '{
    "model":"gemma-4-E2B-it",
    "messages":[{"role":"user","content":"Was ist die Hauptstadt von Frankreich?"}],
    "max_tokens":128
  }'
```

Important request fields:

| Field | Notes |
|---|---|
| `model` | Exact backend model ID or `HA-Assist`; may be omitted for ordinary default routing |
| `messages` | `system`, `user`, `assistant`, `tool` history |
| `max_tokens` | 1..1024 generated tokens |
| `max_input_tokens` | Optional lower per-request input ceiling; cannot raise backend/config limit |
| `temperature` | 0..1 in API; native Hailo generation requires `> 0` |
| `top_p` | Optional 0..1 |
| `stream` | SSE when `true` |
| `tools` | OpenAI-style function definitions |
| `tool_choice` | `auto`, `none`, `required`, or named function |
| `parallel_tool_calls` | Whether multiple calls are accepted when the selected model contract allows them |
| `language` | Optional `de`, `en`, `ru` family hint for HA processing |

A native Hailo request with `temperature: 0` is rejected with HTTP 400 instead of
passing an invalid sampling configuration into HailoRT.

### Images in chat

Image content is supplied inline as base64/data URL:

```json
{
  "model": "Qwen2-VL-2B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Beschreibe das Bild."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BASE64"}}
      ]
    }
  ]
}
```

The service does not fetch arbitrary remote image URLs. Decoded images are converted to
writable contiguous RGB `uint8` and resized for the loaded VLM.

### Streaming

`stream: true` returns Server-Sent Events using OpenAI-style
`chat.completion.chunk` objects and terminates with:

```text
data: [DONE]
```

Plain text can stream natively. Tool-enabled responses are buffered long enough to
validate generated function calls before they are exposed as actions. `HA-Assist` may
also emit a localized wait text when configured, without making an additional model
request.

## Function/tool calling

Gemma, native Hailo LLMs and Qwen VLMs can participate in the service's
OpenAI-compatible function-tool protocol when that backend is the configured/available
one.

- Gemma uses the LiteRT-LM tool representation/parser.
- Native Hailo LLMs use their model-specific prompt template plus the compact tool
  contract.
- Qwen VLM requests use the validated compact JSON-call adapter, including image/tool
  requests.

Generated function names and arguments are validated against the functions actually
offered by the client and their JSON Schemas. `required`, named choices and
`parallel_tool_calls` constraints are enforced by the gateway. Invalid/incomplete calls
are returned as errors rather than executed.

The service **never executes client functions itself**. Home Assistant or another
client receives the returned `tool_calls`, executes them with its own permissions and
may send the assistant call plus matching `role: "tool"` result back in the next
request.

Actual tool-selection quality is model-dependent and should be benchmarked with the
schemas/devices exposed by the client.

## `POST /v1/vision/detect`

The object-detection endpoint uses the single resident YOLO model selected in the
top-level `vision:` configuration block.

```yaml
vision:
  enabled: true
  model: yolov11m
  release: auto
  confidence: 0.4
  iou_threshold: 0.45
  max_detections: 20
  queue_size: 16
  scheduler_priority: 1
  zmq:
    enabled: true
    endpoint: tcp://127.0.0.1:5555
```

Catalog choices currently include:

- YOLOv8: `yolov8n`, `yolov8s`, `yolov8m`
- YOLO11: `yolov11n`, `yolov11s`, `yolov11m`
- YOLO26: `yolo26n`, `yolo26s`, `yolo26m`

All bundled detector entries use COCO-80 labels and 640×640 input. YOLOv8/YOLO11
catalog HEFs use Hailo NMS output. YOLO26 uses the service's anchor-free/NMS-free
postprocessing path and converts results to the same common detection format.

An explicit HEF can be selected with `vision.path` instead of `model`/`release`.

### Detection request

```bash
curl -sS http://HOST:8090/v1/vision/detect \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${HAILO_API_KEY}" \
  -d '{
    "model":"yolov11m",
    "image":"data:image/jpeg;base64,BASE64",
    "confidence":0.4,
    "max_detections":20
  }'
```

Schema:

| Field | Required | Description |
|---|---:|---|
| `image` | yes | Base64 or data-URL encoded image |
| `model` | no | Must match the configured resident vision model when supplied |
| `confidence` | no | Per-request threshold 0..1 |
| `max_detections` | no | 1..100 |

Example response:

```json
{
  "id": "vision-...",
  "object": "vision.detection",
  "created": 1791290000,
  "model": "yolov11m",
  "image": {"width": 1920, "height": 1080},
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.91,
      "box": {"x_min": 0.20, "y_min": 0.10, "x_max": 0.70, "y_max": 0.80},
      "box_pixels": {"x_min": 384, "y_min": 108, "x_max": 1344, "y_max": 864}
    }
  ],
  "metrics": {}
}
```

Boxes are returned in source-image coordinates after reversing the detector letterbox
transformation. Normalized coordinates are 0..1; `box_pixels` uses the original image
size.

HTTP detection and Frigate ZMQ requests share the same `VisionRuntime`, serialized
accelerator owner thread and queue. The HEF is not configured again per request.

## Frigate ZMQ detector

With service debug logging enabled before startup, the listener logs transport events
(`connected`, `handshake_complete`, `disconnected`) independently of model and
inference requests. The monitor endpoint identifies the server socket, not the
remote client's IP address.

Each received request has a generated `request_id`. Model probes log the requested
model and availability; inference logs input shape, dtype, byte count and pending
queue depth, followed by inference time, total request time and nonzero detection
rows (`class, score, ymin, xmin, ymax, xmax`). Binary image/model payloads are not
logged. A sent response confirms server-side processing, not client receipt.

Follow these messages with:

```bash
sudo journalctl -u hailo-10h-services -f -o short-precise | grep --line-buffered 'protocol=zmq'
```

Frigate normally selects detection regions using its own motion detection and sends
those image tensors to the detector. Tracking and stationary-object checks can also
produce requests without new visible motion. The service does not receive motion
events or perform Frigate's camera-side motion detection.


Frigate 0.17+ includes a built-in `type: zmq` detector. The service implements its
REQ/REP model handshake and inference protocol:

1. Frigate sends `model_request` with the basename of `model.path`.
2. The service reports the configured resident YOLO as available/loaded when the name
   matches.
3. Frigate sends the detector tensor as multipart JSON header + raw bytes.
4. The service returns a `(20,6)` `float32` result in Frigate order:
   `[class_id, confidence, ymin, xmin, ymax, xmax]`.

The service deliberately rejects model uploads over ZMQ. Model lifecycle remains owned
by `/etc/hailo-10h-services.yaml` and the model catalog.

Example service side:

```yaml
vision:
  enabled: true
  model: yolov11m
  zmq:
    enabled: true
    endpoint: tcp://0.0.0.0:5555
```

Example Frigate side:

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


The **basename** of Frigate's `model.path` must match the configured service model
(e.g. `yolov11m.hef`). Frigate uses that path for model identification and as its
transfer fallback, so keeping a readable local file at that path is the safest
configuration. When the service reports the matching model as already loaded, no HEF
transfer is required.

If the ZMQ startup handshake fails, Frigate can remain in a no-detection state until its
detector is reinitialized/restarted depending on the Frigate version. Check both
Frigate logs and `/health` after service or network restarts.

## `POST /v1/audio/transcriptions`

Multipart example:

```bash
curl -sS http://HOST:8090/v1/audio/transcriptions \
  -H "Authorization: Bearer ${HAILO_API_KEY}" \
  -F file=@voice.wav \
  -F model=whisper-base \
  -F language=de
```

Whisper accepts WAV/FLAC/OGG formats supported by libsndfile. Audio is mixed to mono,
resampled to 16 kHz and submitted as float32 PCM. `response_format` may be `json` or
`text`. `whisper-1` is only a local compatibility alias; no external OpenAI service is
contacted.

## `GET /v1/models`

Response is an OpenAI-style model list. Depending on configuration/readiness it can
contain:

- configured VLM **or** configured native Hailo LLM on HailoRT 5.4.0,
- Whisper model,
- resident YOLO model,
- `gemma-4-E2B-it`,
- virtual `HA-Assist`.

Use this endpoint rather than assuming a configured model successfully initialized.

## `GET /health`

`/health` is the readiness source of truth. It includes the regular inference runtime
plus the `vision` block and reports information such as selected models, readiness,
model limits/default text model, pending queues and configured ZMQ endpoint.

`model_limits` contains only initialized, enabled chat backends. Disabled Hailo
models and failed/unavailable LiteRT models are omitted. The Playground receives
the same filtered limits via `/ui/config`; no limits are advertised before startup
or after runtime shutdown.

HTTP status is 503 when a required enabled runtime failed to become ready.

## Home Assistant virtual model

Home Assistant should use `model: "HA-Assist"` when HA-specific deterministic routing,
context reduction and validation are desired. `HA-Assist` has no weights of its own.

For HailoRT 5.4.0 the recommended text+image profile is Gemma on CPU plus a Hailo VLM:

```yaml
settings:
  ha_assist_enabled: true
  ha_assist_text_model: gemma-4-E2B-it
  ha_assist_vision_model: Qwen2-VL-2B-Instruct
  ha_assist_fuzzy_enabled: true
  ha_assist_sentence_fuzzy_enabled: true
  ha_assist_sentence_threshold: 94.0

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

Example request:

```json
{
  "model": "HA-Assist",
  "messages": [{"role": "user", "content": "Mach das Licht im Wohnzimmer aus"}],
  "max_tokens": 256,
  "max_input_tokens": 4096
}
```

The virtual model processes requests in stages rather than blindly forwarding the full
Home Assistant request to an LLM:

- exact/deterministic HA intent, state, measurement and action-result paths are tried
  first;
- conservative fuzzy slot correction and protected official-template recovery are validated with HassIL and the client schemas;
- irrelevant entities/tools/history are removed before generation;
- MiniLM may rank relevant HA context but does not generate the answer;
- only ambiguous/general requests reach the configured text backend;
- image requests go to the configured VLM;
- generated tool calls are validated before Home Assistant receives them.

A deterministic answer or tool call performs **zero generative model requests**.
Physical model IDs bypass these HA-specific steps. Home Assistant supplies tool schemas
and executes returned calls; the gateway never needs an HA access token.

A native Hailo LLM can be selected as `ha_assist_text_model`, but on HailoRT 5.4.0 the
VLM must then be disabled. The service does not dynamically unload/reload models to
alternate between those two Hailo GenAI backends.

See [How HA-Assist works](ha-assist.md) for the complete processing sequence and the
rationale for deterministic routing, retrieval and validation. See also
[request pipelines](pipelines.md).

### Context reduction and token limits

For `HA-Assist`, the inbound request may be much larger than the target model context.
Tool/entity retrieval and prompt compilation occur before the final model-bound prompt
is budgeted. Completed house-control turns can be omitted while the current turn and
active tool call/result dependencies remain intact.

For direct physical-model requests the service does not apply the full HA retrieval
pipeline. Those requests must fit the selected backend context after its normal history
compaction rules.

With Home Assistant's **Local OpenAI LLM** integration, use the generic
OpenAI-compatible server mode and select `HA-Assist`. A request body parameter
`max_input_tokens` may be supplied; it can lower but cannot raise the configured model
limit. Do not choose a llama.cpp-specific server mode solely for this service because
that may add parameters not part of this gateway's supported request schema.

## Home Assistant Wyoming STT and TTS

Add the **Wyoming Protocol** integration in Home Assistant and point it to port 10300.
The service advertises the configured Whisper model and supported language metadata.
Home Assistant supplies speech-end/VAD; the service buffers audio until `audio-stop`.

When Piper is enabled and ready, the same listener also advertises an installed
`hailo-piper` TTS program with the configured default voice and additional locally
installed voices/languages. No second Wyoming port is needed. Disabled or failed
Piper backends are not advertised as available TTS services.

After updating and restarting the service, reload the existing **Wyoming Protocol**
integration in Home Assistant so it discovers the new TTS capability. In your
Assist voice assistant settings, select **hailo-piper** for text-to-speech and the
desired installed voice. Whisper remains the speech-to-text provider.

| Wyoming request | Result |
|---|---|
| `describe` | `info` containing Whisper ASR and ready Piper TTS voices. |
| `transcribe` + audio events | Whisper `transcript`. |
| `synthesize` with `text` and optional `voice.name` or `voice.language` | `audio-start`, PCM `audio-chunk` events, then `audio-stop`. |

Piper audio uses the voice's native sample rate, mono signed PCM16 little-endian.
The server first generates a bounded WAV through the same independent CPU queue
as `/v1/audio/speech`, then sends PCM in bounded chunks. This is buffered synthesis,
not incremental text synthesis: `supports_synthesize_streaming` is false.
SSML, speaker overrides and `synthesize-start/chunk/stop` are not supported.
Unknown voices/languages and synthesis failures return a Wyoming `error` event
with code `synthesis_failed`. `piper_max_input_chars`, `max_audio_seconds`,
`queue_size` and `request_timeout` also apply to Wyoming TTS. A missing voice uses
the configured default; language-only selection chooses a matching installed
voice, preferring the default. Explicit voice name takes precedence over language.

Wyoming uses no HTTP API key; use the service's trusted LAN listener for Home
Assistant. Setting `wyoming_port: 0` disables both ASR and TTS on this transport.

## MCP

The MCP streamable HTTP endpoint is:

```text
http://HOST:8090/mcp/
```

Current inference tools include:

- `chat_text`
- `analyze_image`
- `transcribe_audio`

Example LAN settings:

```yaml
settings:
  mcp_hosts: "localhost:*,127.0.0.1:*,hailo.local:*"
  mcp_no_auth_networks: "127.0.0.0/8,::1/128,192.168.0.0/16"
```

The peer-network exemption applies only to MCP. Normal HTTP/WebSocket bearer auth is
unchanged.

## WebSocket and MQTT

WebSocket request:

```json
{"id":"request-1","op":"chat","payload":{"messages":[{"role":"user","content":"Hallo"}]}}
```

Transcription payload:

```json
{"audio_base64":"BASE64_WAV_BYTES","language":"de"}
```

MQTT publishes the same logical payloads under `hailo10h/request/chat` or
`hailo10h/request/transcribe` and receives results under
`hailo10h/response/<request-id>`. MQTT uses QoS 0; use unique request IDs and do not
retain request messages.

HTTP SSE is the token-streaming interface; WebSocket/MQTT return complete results.

## Request metrics

Successful non-streaming chat and vision responses include `metrics`; JSON Whisper
responses include request timing. Available fields depend on backend support.

| Metric | Meaning |
|---|---|
| `request_id` | ID also used in service logs |
| `requested_at`, `responded_at` | Server timestamps with milliseconds |
| `processing_ms` | End-to-end server processing for the request |
| `inference_ms` | Backend inference/generation duration where measurable |
| `ttft_ms`, `ttft_source` | Generation start to first nonempty text chunk |
| `input_tokens`, `input_tokens_source` | Tokenizer count for accepted rendered prompt |
| `output_tokens`, `output_tokens_source` | Retokenized/generated output count where available |
| `input_budget_tokens` | Conservative budget used for admission/compaction |

`HA-Assist` additionally records routing/history/intent/validation details such as
`ha_route`, `ha_history`, `ha_plan`, `ha_intent` and `ha_validation` when applicable.
A deterministic path reports zero generative input/output/inference work rather than
inventing model usage.

Unavailable metrics are omitted or displayed as unavailable; complete HTTP duration is
not substituted for TTFT.

## Configuration summary

For HailoRT 5.4.0, use one of the two Hailo GenAI profiles below.

### Full HA text + image profile: VLM on Hailo, Gemma on CPU

```yaml
settings:
  ha_assist_enabled: true
  ha_assist_text_model: gemma-4-E2B-it
  ha_assist_vision_model: Qwen2-VL-2B-Instruct

models:
  vlm:
    enabled: true
    model: Qwen2-VL-2B-Instruct
    max_input_tokens: 2048
  hailo_llm:
    enabled: false
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
  zmq:
    enabled: true
    endpoint: tcp://127.0.0.1:5555
```

### Native Hailo LLM text profile

```yaml
settings:
  ha_assist_enabled: true
  ha_assist_text_model: Qwen3-1.7B-Instruct

models:
  vlm:
    enabled: false
  hailo_llm:
    enabled: true
    model: Qwen3-1.7B-Instruct
    max_input_tokens: 2048
```

In the second profile, HA-Assist image requests are unavailable unless the deployment
is reconfigured. The service intentionally does not swap VLM/LLM HEFs per request.

Object detection keeps its dedicated top-level `vision:` block. The legacy
`models.vision` form is parsed for backwards compatibility, but new configurations
should use `vision:` and should not configure both forms at the same time.

## Updating

Use the normal main-branch update path:

```bash
cd /path/to/hailo-10h-services
git pull --ff-only
sudo bash scripts/install.sh
sudo systemctl restart hailo-10h-services.service
```

## Related documentation

- [README / feature overview](../README.md)
- [How HA-Assist works](ha-assist.md)
- [Installation and deployment](installation.md)
- [Request pipelines and HA routing](pipelines.md)
- [Language behavior](languages.md)
- [Model benchmark/evaluation](model-benchmark-evaluation.md)
- [Troubleshooting](troubleshooting.md)

External references:

- [Hailo GenAI usage](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/USAGE.rst)
- [Frigate object detectors](https://docs.frigate.video/configuration/object_detectors/)
- [Wyoming protocol](https://github.com/OHF-Voice/wyoming)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## `POST /v1/audio/speech`

CPU text-to-speech using Piper. Protected by the same Bearer API key and body
limit as other `/v1` endpoints. Enable and provision Piper as described in
[installation](installation.md#piper-cpu-text-to-speech).

```bash
curl --fail-with-body http://HOST:8090/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"piper","input":"Hallo! Das Licht im Wohnzimmer ist eingeschaltet.","voice":"de_DE-thorsten-medium","response_format":"wav","speed":1.0}' \
  --output speech.wav
```

| Field | Default | Meaning |
|---|---|---|
| `model` | `piper` | Only `piper` is currently supported. |
| `input` | Required | Nonblank text; default limit 4096 characters, configurable via `piper_max_input_chars`. |
| `voice` | Configured `piper_voice` | Installed voice ID without `.onnx`, e.g. `de_DE-thorsten-medium`. Client paths/download URLs are rejected. |
| `language` | Selected voice language | Optional validation of the voice language, e.g. `de`, `de_DE`, `en_US`. Does not translate text or change the voice's language. |
| `response_format` | `wav` | `wav`: PCM16 mono WAV at native voice rate. `pcm`: headerless signed PCM16 little-endian, mono, resampled to 24 kHz. |
| `speed` | `1.0` | 0.25–4.0; higher values speak faster (`length_scale = 1 / speed`). |

The response contains binary audio (`audio/wav` or `audio/pcm`), with
`X-Audio-Sample-Rate` and `X-Inference-Ms` headers. Timing includes queue wait
and synthesis; token counts and LLM TTFT do not apply. MP3, Opus, FLAC, AAC,
SSE and streaming responses are not currently supported; unsupported request
fields/formats are rejected with HTTP 422.

Piper has its own bounded CPU queue (`queue_size`) and request deadline
(`request_timeout`). A timed-out/disconnected request retains its queue slot
until native work completes. Generated audio is limited by `max_audio_seconds`;
split longer text across requests. HTTP 400 indicates invalid voice/language or
configured input/audio limits; 503 means disabled/unavailable/full queue;
504 means timeout; 502 means synthesis failure.

`GET /health` includes a `piper` block with enabled/ready, CPU device, default
voice/language, pending count and startup error. Piper startup failure does not
stop other backends. `GET /v1/models` includes `piper` when ready. The Wyoming endpoint also exposes Piper TTS for Home Assistant; see
[Wyoming STT and TTS](#home-assistant-wyoming-stt-and-tts).

The Playground TTS panel uses `/v1/audio/speech` with WAV output. It lists locally
installed voices, allows speed adjustment, retains generated audio and displays
request/response timestamps, total duration, server processing time, sample rate
and file size. Generated files can be played or downloaded; clear the TTS history
to release them. The API key entered in the Playground also authenticates TTS.


## `POST /v1/ha-assist/diagnose`

Accepts the same request as chat, with `model: "HA-Assist"` and `stream: false`.
Uses the existing authentication and owner-thread queue. It may perform MiniLM
embedding retrieval, but never invokes LLM/VLM generation or executes a tool.
`proposed_response` contains the deterministic text or proposed tool calls, or
`null` for a request that would need inference. `metrics.ha_route` reports actual
`inference_calls: 0` plus hypothetical `would_inference_calls`. `ha_plan`,
`ha_intent`, and `ha_stages_ms` explain targets, sentence candidates, protection
rules and preparation timings. Original requests remain unmodified.

```bash
curl -sS http://HOST:8090/v1/ha-assist/diagnose \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  --data-binary @ha-request.json
```

`prepared_request` contains the complete messages and tool schemas **after HA
preparation, before the native chat template and input-token budget**. It is not a
claim about the final native prompt or token count. Existing debug events
`final_gemma_request` / native Hailo prompt events remain authoritative for the
prompt actually used by production inference.

## Optional structured exposed catalogue

HA-Assist clients may add `ha_context` to chat or diagnosis requests:

```json
{
  "version": "catalogue-revision-17",
  "entities": [{
    "entity_id": "light.reading",
    "name": "Reading light",
    "domain": "light",
    "area": "Library",
    "aliases": ["Sofa light"],
    "area_aliases": ["Reading room"],
    "floor": "First floor",
    "device_class": "light",
    "capabilities": ["brightness"]
  }]
}
```

Only include entities exposed to this conversation. The structured catalogue
replaces the legacy static catalogue for this request, even if its version string
is unchanged; legacy clients continue to work without it. Aliases resolve to
canonical client target names. IDs, floor and device class remain rich catalogue
metadata; this does not add floor-target tools absent from the client schema.
`capabilities` describes supported writable properties: omitted/null means
unknown; `[]` means no property-setting capability. Known unsupported adjustments
are rejected before inference. An explicitly empty catalogue permits no device
control. Live state fields are deliberately not accepted in this metadata cache.
Physical-model requests do not apply this metadata to routing or prompts.

`GET /health` now includes `cache`: bounded static tool/template cache counts,
MiniLM embedding counts, observed Hailo context snapshot methods and the current
backend isolation policy. No state snapshot methods are called by this inspection.
See [HA recognition and caching](ha-recognition.md) for policy and SDK findings.
