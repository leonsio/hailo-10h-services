# Process isolation

The production CLI runs inference in process-isolated mode by default.

```text
hailo-10h-services (gateway)
├─ HTTP / OpenAI API / Web UI
├─ MCP / Wyoming / MQTT / Frigate ZMQ
├─ HA-Assist / Frigate-Assist deterministic routing
│
├─ Hailo worker process
│  ├─ VLM / native Hailo LLM
│  ├─ Whisper
│  ├─ MiniLM
│  └─ YOLO detector
│
├─ LiteRT-LM worker process
│  └─ Gemma
│
└─ Piper worker process
   └─ TTS / ONNX Runtime
```

## Goals

The split prevents a long CPU inference or native-library failure from blocking the gateway interpreter or unrelated inference backends. It also lets Linux schedule CPU-backed work on separate Raspberry Pi 5 cores while keeping all Hailo-backed models in one process.

The gateway exposes the public API and ports. Bounded queues, request deadlines, streaming, cancellation and model routing remain in the gateway runtimes; their owner threads perform IPC instead of native inference.

The Hailo process has separate owner threads for generative/STT/MiniLM work and object detection. This keeps one Hailo process while allowing the HailoRT scheduler to arbitrate concurrent configured models and retain the configured detector scheduler priority.

## Startup order

The gateway schedules resident initialization in the background and immediately allows the ASGI server to serve the browser UI, `/monitor`, `/settings`, `/health` and API routes. Requests that require a backend which is not ready yet return the normal unavailable/readiness response.

Resident initialization is ordered so accelerator allocation happens before CPU model allocation:

1. Start the Hailo chat/STT/MiniLM worker and load enabled Hailo models.
2. Start the resident YOLO vision runtime.
3. Start the independent LiteRT-LM worker.
4. Start the independent Piper worker.
5. Start the Frigate ZMQ listener.
6. Start Wyoming when enabled.
7. Start MQTT after the initialization phases when a broker is configured.

`/v1/monitor` exposes the current `startup.state`, `startup.phase` and collected startup errors while this sequence runs. `/health` can therefore be reached during initialization and returns HTTP 503 until the required inference runtimes are ready. This keeps startup failures visible without delaying access to diagnostics.

## Health

`/health` adds a `process_mode` flag and a `workers` object. The worker status contains the gateway/worker PID, restart count, active user count and the last startup error when present.

Example:

```json
{
  "process_mode": true,
  "workers": {
    "gateway": {"status": "ready", "pid": 1200, "restarts": 0},
    "hailo": {"status": "ready", "pid": 1201, "restarts": 0},
    "llm": {"status": "ready", "pid": 1202, "restarts": 0},
    "piper": {"status": "ready", "pid": 1203, "restarts": 0}
  }
}
```

If a worker exits unexpectedly, pending requests fail instead of hanging. A subsequent request to an acquired worker recreates the process and reloads its resident backend.

## Threaded fallback mode

Set:

```bash
HAILO_PROCESS_MODE=0
```

to use resident backends on owner threads instead of child-process IPC. This mode is intended for native driver/runtime troubleshooting rather than normal production operation.

## CPU usage

Process isolation does not imply one CPU core per process. LiteRT-LM and ONNX Runtime can create native worker threads of their own. On a four-core Raspberry Pi 5, avoid configuring every native runtime to use four threads simultaneously; oversubscription can reduce throughput and increase latency.

The purpose of this process model is isolation and schedulability. CPU affinity or backend thread-count tuning should be based on measurements rather than fixed core pinning.

## Why the application remains Python

A full C/C++ rewrite is not required to obtain multicore execution. The expensive inference paths already execute primarily inside native runtimes (HailoRT, LiteRT and ONNX Runtime/Piper). Python mainly coordinates protocols, request validation, deterministic routing, prompt/tool processing and lifecycle management.

Moving those orchestration layers to C++ would increase implementation and maintenance cost substantially while leaving most model execution time unchanged. Native C/C++ components remain appropriate for measured hot paths or bindings where profiling shows Python itself is material, but process isolation addresses the current blocking and fault-isolation problem without replacing the application stack.