"""Bounded, process-local service monitoring without external dependencies."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from hailo_services.config import FRIGATE_ASSIST_MODEL, HA_ASSIST_MODEL, LLM_MODEL
from hailo_services.diagnostics.logging_utils import redact_inline_data
from hailo_services.shared.tool_calling import has_tool_context


def _now():
    """Return the current UTC time.

    Returns:
        str: ISO formatted timestamp.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RecentLogs(logging.Handler):
    """Keep a small, redacted copy of application logs in the gateway process."""

    def __init__(self):
        """Install a bounded INFO-level capture buffer.

        Returns:
            None: Initializes the log handler.
        """
        super().__init__(logging.INFO)
        self.entries = deque(maxlen=100)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record):
        """Capture a bounded line without recursive logging on formatting errors.

        Returns:
            None: Adds a log entry when applicable.
        """
        try:
            if record.name.startswith("hailo_services"):
                line = redact_inline_data(self.format(record))
                with self.lock:
                    self.entries.append(line[:1000])
        except Exception:
            pass

    def snapshot(self):
        """Return the most recent application log lines.

        Returns:
            list[str]: Recent redacted log entries.
        """
        with self.lock:
            return list(self.entries)[-50:]


class Monitor:
    """Aggregate each service's last hour in bounded per-second buckets."""

    SERVICES = ("VLM", "LLM", "Hailo LLM", "HA-Assist", "Frigate-Assist", "Whisper", "Piper", "YOLO")

    def __init__(self, settings):
        """Create independent rolling measurements for one gateway.

        Returns:
            None: Initializes the monitor.
        """
        self.settings = settings
        self.lock = threading.Lock()
        self.buckets = {name: {} for name in self.SERVICES}
        self.last = {name: None for name in self.SERVICES}
        self.active = {name: 0 for name in self.SERVICES}
        self.started = _now()
        self.logs = RecentLogs()
        logging.getLogger().addHandler(self.logs)
        self._cpu = None
        self._cgroup_cpu = None

    def close(self):
        """Detach the capture handler when its owning application stops.

        Returns:
            None: Removes the handler.
        """
        logging.getLogger().removeHandler(self.logs)

    def begin(self, name):
        """Mark an incoming inference request and return its monotonic start.

        Returns:
            float: Monotonic start time.
        """
        with self.lock:
            self.last[name] = _now()
            self.active[name] += 1
        return time.perf_counter()

    def end(self, name, started, *, failed=False, metrics=None):
        """Record completion and timing in a second-wide rolling bucket.

        Returns:
            None: Updates aggregate statistics.
        """
        metrics = metrics or {}
        duration = (time.perf_counter() - started) * 1000
        ttft = metrics.get("ttft_ms")
        if not isinstance(ttft, (int, float)) or not 0 <= ttft < float("inf"):
            ttft = None
        second = int(time.time())
        with self.lock:
            self.active[name] = max(0, self.active[name] - 1)
            bucket = self.buckets[name].setdefault(second, [0, 0, 0.0, 0.0, 0, 0.0])
            bucket[0] += 1
            bucket[1] += int(failed)
            bucket[2] += duration
            bucket[3] = max(bucket[3], duration)
            if ttft is not None:
                bucket[4] += 1
                bucket[5] += ttft
            self._prune(self.buckets[name], second)

    @staticmethod
    def _prune(buckets, now):
        """Remove expired buckets.

        Returns:
            None: Mutates the supplied buckets.
        """
        for second in list(buckets):
            if second <= now - 3600:
                del buckets[second]

    def snapshot(self):
        """Return current rolling counts, latencies and last request timestamps.

        Returns:
            dict: Per-service aggregates.
        """
        now = int(time.time())
        result = {}
        with self.lock:
            for name, buckets in self.buckets.items():
                self._prune(buckets, now)
                count = sum(b[0] for b in buckets.values())
                errors = sum(b[1] for b in buckets.values())
                ttft_count = sum(b[4] for b in buckets.values())
                result[name] = {
                    "requests_1h": count,
                    "errors_1h": errors,
                    "active": self.active[name],
                    "last_request_at": self.last[name],
                    "average_ms": round(sum(b[2] for b in buckets.values()) / count, 1) if count else None,
                    "maximum_ms": round(max((b[3] for b in buckets.values()), default=0), 1) if count else None,
                    "average_ttft_ms": round(sum(b[5] for b in buckets.values()) / ttft_count, 1) if ttft_count else None,
                    "ttft_samples_1h": ttft_count,
                }
        return result

    def resources(self):
        """Read Linux process, host and cgroup consumption without psutil.

        Returns:
            dict: Available resource measurements and nullable unsupported fields.
        """
        data = {"cpu_cores": os.cpu_count(), "load_average": None, "host_memory": None,
                "process_rss_bytes": None, "process_cpu_percent": None, "cgroup_memory_bytes": None,
                "cgroup_memory_limit_bytes": None, "cgroup_cpu_percent": None}
        try:
            data["load_average"] = round(os.getloadavg()[0], 2)
            data["process_rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            mem = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in {"MemTotal", "MemAvailable"}:
                    mem[key] = int(value.strip().split()[0]) * 1024
            data["host_memory"] = {"total_bytes": mem.get("MemTotal"), "available_bytes": mem.get("MemAvailable")}
            cgroup = Path("/proc/self/cgroup").read_text().splitlines()
            group = next((row.split(":", 2)[2].lstrip("/") for row in cgroup if row.startswith("0::")), None)
            if group is not None:
                root = Path("/sys/fs/cgroup") / group
                data["cgroup_memory_bytes"] = int((root / "memory.current").read_text())
                limit = (root / "memory.max").read_text().strip()
                data["cgroup_memory_limit_bytes"] = int(limit) if limit != "max" else None
                cpu_line = next((line for line in (root / "cpu.stat").read_text().splitlines() if line.startswith("usage_usec ")), None)
                if cpu_line:
                    elapsed, used = time.monotonic(), int(cpu_line.split()[1]) / 1_000_000
                    with self.lock:
                        if self._cgroup_cpu is not None and elapsed > self._cgroup_cpu[0]:
                            data["cgroup_cpu_percent"] = round(100 * (used - self._cgroup_cpu[1]) / (elapsed - self._cgroup_cpu[0]), 1)
                        self._cgroup_cpu = elapsed, used
        except (OSError, ValueError, IndexError):
            pass
        try:
            elapsed = time.monotonic()
            cpu = time.process_time()
            with self.lock:
                if self._cpu is not None:
                    previous_elapsed, previous_cpu = self._cpu
                    if elapsed > previous_elapsed:
                        data["process_cpu_percent"] = round(100 * (cpu - previous_cpu) / (elapsed - previous_elapsed), 1)
                self._cpu = elapsed, cpu
        except (OSError, ValueError):
            pass
        return data


def instrument(runtime, vision, speech, monitor):
    """Observe shared runtime entry points across all supported protocols.

    Returns:
        None: Wraps the owned runtime instances.
    """
    chat = runtime.chat
    stream = runtime.stream
    transcribe = runtime.transcribe
    detect = vision.detect_array
    synthesize = speech.synthesize

    def chat_name(request):
        """Resolve a public chat request to a monitoring service.

        Returns:
            str: Service label.
        """
        model = runtime.default_chat_request(request).model
        if model == HA_ASSIST_MODEL:
            return "HA-Assist"
        if model == FRIGATE_ASSIST_MODEL:
            return "Frigate-Assist"
        if model == LLM_MODEL:
            return "LLM"
        if model == runtime.settings.hailo_llm_model_id:
            return "Hailo LLM"
        return "VLM" if model == runtime.settings.vlm_model else "LLM"

    async def measured_chat(request, *args, **kwargs):
        """Observe a complete chat call.

        Returns:
            Any: Original chat result.
        """
        name = chat_name(request)
        started = monitor.begin(name)
        failed = True
        try:
            result = await chat(request, *args, **kwargs)
            failed = False
            return result
        finally:
            monitor.end(name, started, failed=failed, metrics=getattr(request, "_metrics", None))

    async def measured_stream(request, *args, **kwargs):
        """Observe a stream until completion or cancellation.

        Yields:
            Any: Original chunks.
        """
        # Runtime.stream delegates these requests to chat; count exactly once.
        if request.model in {HA_ASSIST_MODEL, FRIGATE_ASSIST_MODEL} or has_tool_context(request):
            async for item in stream(request, *args, **kwargs):
                yield item
            return
        name = chat_name(request)
        started = monitor.begin(name)
        failed = True
        try:
            async for item in stream(request, *args, **kwargs):
                yield item
            failed = False
        finally:
            monitor.end(name, started, failed=failed, metrics=getattr(request, "_metrics", None))

    async def measured_transcribe(*args, **kwargs):
        """Observe a transcription call.

        Returns:
            Any: Original transcript.
        """
        started = monitor.begin("Whisper")
        failed = True
        try:
            result = await transcribe(*args, **kwargs)
            failed = False
            return result
        finally:
            monitor.end("Whisper", started, failed=failed)

    async def measured_detect(*args, **kwargs):
        """Observe a detector inference call.

        Returns:
            Any: Original detections.
        """
        started = monitor.begin("YOLO")
        failed = True
        try:
            result = await detect(*args, **kwargs)
            failed = False
            return result
        finally:
            monitor.end("YOLO", started, failed=failed)

    async def measured_synthesize(*args, **kwargs):
        """Observe a speech synthesis call.

        Returns:
            Any: Original audio and metadata.
        """
        started = monitor.begin("Piper")
        failed = True
        try:
            result = await synthesize(*args, **kwargs)
            failed = False
            return result
        finally:
            monitor.end("Piper", started, failed=failed)

    runtime.chat = measured_chat
    runtime.stream = measured_stream
    runtime.transcribe = measured_transcribe
    vision.detect_array = measured_detect
    speech.synthesize = measured_synthesize
