# Diagnostics and target-device checks

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
