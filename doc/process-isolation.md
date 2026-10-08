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

The public API and ports do not change. Existing bounded queues, request deadlines, streaming, cancellation and model routing remain in the gateway runtimes; their owner threads perform IPC instead of native inference.

The Hailo process has separate owner threads for generative/STT/MiniLM work and object detection. This keeps one Hailo process while allowing the HailoRT scheduler to arbitrate concurrent configured models and retain the configured detector scheduler priority.

## Startup order

Startup keeps the existing two phases:

1. Start the Hailo worker and load all enabled Hailo models, including the detector.
2. Start the independent LiteRT-LM and Piper CPU workers.
3. Start protocol listeners.

This prevents CPU model allocation from changing Hailo initialization order.

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

## Compatibility fallback

Set:

```bash
HAILO_PROCESS_MODE=0
```

to use the previous single-process / owner-thread architecture. This is intended as a rollback switch for native driver/runtime troubleshooting, not as the preferred normal mode.

## CPU usage

Process isolation does not imply one CPU core per process. LiteRT-LM and ONNX Runtime can create native worker threads of their own. On a four-core Raspberry Pi 5, avoid configuring every native runtime to use four threads simultaneously; oversubscription can reduce throughput and increase latency.

The purpose of this process model is isolation and schedulability. CPU affinity or backend thread-count tuning should be based on measurements rather than fixed core pinning.

## Why the application remains Python

A full C/C++ rewrite is not required to obtain multicore execution. The expensive inference paths already execute primarily inside native runtimes (HailoRT, LiteRT and ONNX Runtime/Piper). Python mainly coordinates protocols, request validation, deterministic routing, prompt/tool processing and lifecycle management.

Moving those orchestration layers to C++ would increase implementation and maintenance cost substantially while leaving most model execution time unchanged. Native C/C++ components remain appropriate for measured hot paths or bindings where profiling shows Python itself is material, but process isolation addresses the current blocking and fault-isolation problem without replacing the application stack.
