"""Process-isolated resident inference backends used by the production gateway.

The gateway keeps FastAPI, MCP, Wyoming, MQTT, ZMQ framing and deterministic
routing. Native resident inference is delegated to three long-lived processes:
Hailo (VLM/LLM/Whisper/MiniLM/YOLO), LiteRT-LM and Piper. Existing runtime
classes still provide bounded queues and request deadlines; their owner threads
now block only on IPC instead of native inference.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .backend_hailo import HailoBackend
from .backend_litert import LiteRTLMBackend
from .errors import BusyError, LiteRTInferenceError
from .input_budget import InputBudgetError
from .models import ModelManager
from .runtime import Runtime
from .speech_piper import PiperBackend
from .speech_runtime import SpeechRuntime
from .vision import HailoVisionBackend, VisionRuntime

_LOG = logging.getLogger(__name__)


def _worker_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if debug:
        logging.getLogger("hailo_services").setLevel(logging.DEBUG)


def _error_payload(exc: BaseException) -> dict[str, Any]:
    attributes = {
        key: value
        for key, value in vars(exc).items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    return {
        "module": type(exc).__module__,
        "type": type(exc).__name__,
        "message": str(exc),
        "attributes": attributes,
    }


def _raise_remote(payload: dict[str, Any]) -> None:
    """Recreate exceptions whose concrete class controls API error semantics."""
    name = payload.get("type", "RuntimeError")
    message = payload.get("message", "Remote worker failed")
    attributes = payload.get("attributes") or {}
    if name == "ValueError":
        raise ValueError(message)
    if name == "FileNotFoundError":
        raise FileNotFoundError(message)
    if name == "BusyError":
        raise BusyError(message)
    if name == "LiteRTInferenceError":
        raise LiteRTInferenceError(message)
    if name == "InputBudgetError":
        # Preserve the structured HTTP handler without depending on constructor
        # arguments that only the remote model-budget calculation knows.
        tokens = attributes.get("tokens")
        limit = attributes.get("limit")
        if tokens is None or limit is None:
            match = re.search(r"require (\d+) input tokens; .* limit is (\d+)", message)
            if match:
                tokens, limit = map(int, match.groups())
        exc = InputBudgetError.__new__(InputBudgetError)
        ValueError.__init__(exc, message)
        exc.tokens = int(tokens or 0)
        exc.limit = int(limit or 0)
        raise exc
    raise RuntimeError(f"{name}: {message}")


def _metrics(args: tuple[Any, ...]) -> dict[str, Any] | None:
    if not args:
        return None
    value = getattr(args[0], "_metrics", None)
    return dict(value) if isinstance(value, dict) else None


def _finish(response_queue, request_id: str, future, args: tuple[Any, ...]) -> None:
    try:
        response_queue.put(
            {
                "kind": "result",
                "id": request_id,
                "value": future.result(),
                "metrics": _metrics(args),
            }
        )
    except BaseException as exc:  # Native bindings sometimes raise extension exceptions.
        response_queue.put(
            {
                "kind": "error",
                "id": request_id,
                "error": _error_payload(exc),
                "metrics": _metrics(args),
            }
        )


def _hailo_worker(settings, request_queue, response_queue) -> None:
    """Own every Hailo-backed model in one process."""
    _worker_logging(settings.debug_log)
    chat = HailoBackend(settings)
    vision = HailoVisionBackend(settings) if settings.vision_enabled else None
    chat_owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hailo-chat-owner")
    vision_owner = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="hailo-vision-owner")
        if vision is not None
        else None
    )
    cancellations: dict[str, threading.Event] = {}
    ready = False
    try:
        chat_owner.submit(chat.start).result()
        if vision is not None:
            vision_owner.submit(vision.start).result()
        ready = True
        response_queue.put(
            {
                "kind": "ready",
                "metadata": {
                    "pid": os.getpid(),
                    "paths": getattr(chat, "paths", {}),
                    "artifact_paths": getattr(chat, "artifact_paths", {}),
                    "minilm_ready": bool(getattr(chat, "minilm", None)),
                    "vision_path": getattr(vision, "path", None),
                    "vision_input_shape": getattr(vision, "input_shape", None),
                },
            }
        )
        while True:
            message = request_queue.get()
            command = message.get("cmd")
            if command == "stop":
                break
            if command == "cancel":
                event = cancellations.get(message.get("id"))
                if event is not None:
                    event.set()
                continue
            if command != "call":
                continue
            request_id = message["id"]
            op = message["op"]
            args = tuple(message.get("args", ()))
            kwargs = dict(message.get("kwargs", {}))
            stream = bool(message.get("stream"))
            cancelled = threading.Event()
            cancellations[request_id] = cancelled

            def invoke(
                op=op,
                args=args,
                kwargs=kwargs,
                request_id=request_id,
                stream=stream,
                cancelled=cancelled,
            ):
                def emit(chunk):
                    response_queue.put({"kind": "chunk", "id": request_id, "value": chunk})

                try:
                    if op == "chat":
                        return chat.chat(args[0], emit if stream else None, cancelled)
                    if op == "transcribe":
                        return chat.transcribe(*args, **kwargs)
                    if op == "select_tools":
                        return chat.select_tools(*args, **kwargs)
                    if op == "retrieve_context":
                        return chat.retrieve_context(*args, **kwargs)
                    if op == "vision_detect":
                        if vision is None:
                            raise RuntimeError("Vision model is disabled")
                        return vision.detect(*args, **kwargs)
                    raise ValueError(f"Unknown Hailo worker operation: {op}")
                finally:
                    cancellations.pop(request_id, None)

            executor = vision_owner if op == "vision_detect" and vision_owner else chat_owner
            future = executor.submit(invoke)
            future.add_done_callback(
                lambda completed, rid=request_id, call_args=args: _finish(
                    response_queue, rid, completed, call_args
                )
            )
    except BaseException as exc:
        if not ready:
            response_queue.put({"kind": "startup_error", "error": _error_payload(exc)})
        else:
            _LOG.exception("Hailo worker control loop failed")
    finally:
        try:
            chat_owner.submit(chat.close).result()
        except BaseException:
            _LOG.exception("Hailo chat worker cleanup failed")
        if vision is not None and vision_owner is not None:
            try:
                vision_owner.submit(vision.close).result()
            except BaseException:
                _LOG.exception("Hailo vision worker cleanup failed")
        chat_owner.shutdown(wait=True, cancel_futures=False)
        if vision_owner is not None:
            vision_owner.shutdown(wait=True, cancel_futures=False)


def _litert_worker(settings, request_queue, response_queue) -> None:
    """Own Gemma/LiteRT-LM in an independent CPU process."""
    _worker_logging(settings.debug_log)
    model_path = settings.litert_model_path or str(
        Path(settings.model_store) / "gemma-4-E2B-it.litertlm"
    )
    backend = LiteRTLMBackend(
        model_path,
        settings.litert_max_num_tokens,
        settings.litert_max_input_tokens,
        settings.debug_log,
        ModelManager(settings),
    )
    owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="litert-owner")
    cancellations: dict[str, threading.Event] = {}
    ready = False
    try:
        owner.submit(backend.start).result()
        ready = True
        response_queue.put(
            {
                "kind": "ready",
                "metadata": {
                    "pid": os.getpid(),
                    "model_path": model_path,
                    "max_num_tokens": settings.litert_max_num_tokens,
                    "max_input_tokens": settings.litert_max_input_tokens,
                },
            }
        )
        while True:
            message = request_queue.get()
            command = message.get("cmd")
            if command == "stop":
                break
            if command == "cancel":
                event = cancellations.get(message.get("id"))
                if event is not None:
                    event.set()
                continue
            if command != "call":
                continue
            request_id = message["id"]
            args = tuple(message.get("args", ()))
            kwargs = dict(message.get("kwargs", {}))
            stream = bool(message.get("stream"))
            cancelled = threading.Event()
            cancellations[request_id] = cancelled

            def invoke(
                args=args,
                kwargs=kwargs,
                request_id=request_id,
                stream=stream,
                cancelled=cancelled,
            ):
                def emit(chunk):
                    response_queue.put({"kind": "chunk", "id": request_id, "value": chunk})

                try:
                    return backend.chat(
                        args[0],
                        emit if stream else None,
                        cancelled,
                        kwargs.get("tools_prepared", False),
                    )
                finally:
                    cancellations.pop(request_id, None)

            future = owner.submit(invoke)
            future.add_done_callback(
                lambda completed, rid=request_id, call_args=args: _finish(
                    response_queue, rid, completed, call_args
                )
            )
    except BaseException as exc:
        if not ready:
            response_queue.put({"kind": "startup_error", "error": _error_payload(exc)})
        else:
            _LOG.exception("LiteRT worker control loop failed")
    finally:
        try:
            owner.submit(backend.close).result()
        except BaseException:
            _LOG.exception("LiteRT worker cleanup failed")
        owner.shutdown(wait=True, cancel_futures=False)


def _piper_worker(settings, request_queue, response_queue) -> None:
    """Own Piper/ONNX inference in an independent CPU process."""
    _worker_logging(settings.debug_log)
    backend = PiperBackend(settings)
    owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-owner")
    ready = False
    try:
        owner.submit(backend.start).result()
        ready = True
        response_queue.put(
            {"kind": "ready", "metadata": {"pid": os.getpid(), "voice": settings.piper_voice}}
        )
        while True:
            message = request_queue.get()
            if message.get("cmd") == "stop":
                break
            if message.get("cmd") != "call":
                continue
            request_id = message["id"]
            args = tuple(message.get("args", ()))
            future = owner.submit(backend.synthesize, *args)
            future.add_done_callback(
                lambda completed, rid=request_id, call_args=args: _finish(
                    response_queue, rid, completed, call_args
                )
            )
    except BaseException as exc:
        if not ready:
            response_queue.put({"kind": "startup_error", "error": _error_payload(exc)})
        else:
            _LOG.exception("Piper worker control loop failed")
    finally:
        try:
            owner.submit(backend.close).result()
        except BaseException:
            _LOG.exception("Piper worker cleanup failed")
        owner.shutdown(wait=True, cancel_futures=False)


_TARGETS = {"hailo": _hailo_worker, "litert": _litert_worker, "piper": _piper_worker}


class _WorkerClient:
    """Synchronous IPC endpoint called only by the existing bounded owner threads."""

    def __init__(self, kind: str, settings):
        self.kind = kind
        self.settings = settings
        self.context = mp.get_context("spawn")
        self.process = None
        self.request_queue = None
        self.response_queue = None
        self.listener = None
        self.metadata: dict[str, Any] = {}
        self._pending: dict[str, tuple[queue.Queue, Callable | None]] = {}
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._users = 0
        self.restarts = 0
        self.last_error = None

    def acquire(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self.process is None or not self.process.is_alive():
                self._start_locked()
            self._users += 1
            return dict(self.metadata)

    def _start_locked(self) -> None:
        if self.process is not None:
            self._shutdown_locked(force=True)
            self.restarts += 1
        self.request_queue = self.context.Queue(maxsize=max(4, self.settings.queue_size + 2))
        self.response_queue = self.context.Queue(maxsize=max(32, self.settings.queue_size * 8))
        self.process = self.context.Process(
            target=_TARGETS[self.kind],
            args=(self.settings, self.request_queue, self.response_queue),
            name=f"hailo-services-{self.kind}",
            daemon=False,
        )
        self.process.start()
        startup_timeout = max(180.0, float(self.settings.request_timeout) * 3.0)
        try:
            message = self.response_queue.get(timeout=startup_timeout)
        except queue.Empty as exc:
            self.last_error = "startup timeout"
            self._shutdown_locked(force=True)
            raise RuntimeError(f"{self.kind} worker startup timed out") from exc
        if message.get("kind") != "ready":
            error = message.get("error", {"type": "RuntimeError", "message": "startup failed"})
            self.last_error = f"{error.get('type')}: {error.get('message')}"
            self._shutdown_locked(force=True)
            _raise_remote(error)
        self.metadata = dict(message.get("metadata", {}))
        self.last_error = None
        self.listener = threading.Thread(
            target=self._listen, name=f"{self.kind}-worker-responses", daemon=True
        )
        self.listener.start()
        _LOG.info(
            "worker=%s event=ready pid=%s process_mode=spawn",
            self.kind,
            self.metadata.get("pid", self.process.pid),
        )

    def _listen(self) -> None:
        process = self.process
        response_queue = self.response_queue
        while process is not None and response_queue is not None:
            try:
                message = response_queue.get(timeout=0.25)
            except queue.Empty:
                if not process.is_alive():
                    self._fail_pending("worker process exited")
                    return
                continue
            request_id = message.get("id")
            with self._lock:
                pending = self._pending.get(request_id)
            if pending is None:
                continue
            result_queue, emit = pending
            if message.get("kind") == "chunk":
                if emit is not None:
                    try:
                        emit(message.get("value"))
                    except BaseException:
                        _LOG.exception("worker=%s stream callback failed", self.kind)
                continue
            result_queue.put(message)

    def _fail_pending(self, message: str) -> None:
        with self._lock:
            pending = list(self._pending.values())
        failure = {
            "kind": "error",
            "error": {"type": "RuntimeError", "message": message, "attributes": {}},
        }
        for result_queue, _ in pending:
            try:
                result_queue.put_nowait(failure)
            except queue.Full:
                pass

    def _ensure_running(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        with self._lifecycle_lock:
            if self._users <= 0:
                raise RuntimeError(f"{self.kind} worker is not acquired")
            if self.process is None or not self.process.is_alive():
                self._start_locked()

    def call(
        self,
        op: str,
        *args,
        emit: Callable | None = None,
        cancelled: threading.Event | None = None,
        **kwargs,
    ):
        self._ensure_running()
        request_id = uuid.uuid4().hex
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[request_id] = (result_queue, emit)
        self.request_queue.put(
            {
                "cmd": "call",
                "id": request_id,
                "op": op,
                "args": args,
                "kwargs": kwargs,
                "stream": emit is not None,
            }
        )
        cancel_sent = False
        try:
            while True:
                try:
                    message = result_queue.get(timeout=0.05)
                    break
                except queue.Empty:
                    if cancelled is not None and cancelled.is_set() and not cancel_sent:
                        self.request_queue.put({"cmd": "cancel", "id": request_id})
                        cancel_sent = True
                    if self.process is None or not self.process.is_alive():
                        raise RuntimeError(f"{self.kind} worker process exited")
            returned_metrics = message.get("metrics")
            if args and isinstance(returned_metrics, dict):
                local_metrics = getattr(args[0], "_metrics", None)
                if isinstance(local_metrics, dict):
                    local_metrics.update(returned_metrics)
            if message.get("kind") == "error":
                _raise_remote(message["error"])
            return message.get("value")
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def release(self) -> None:
        with self._lifecycle_lock:
            self._users = max(0, self._users - 1)
            if self._users == 0:
                self._shutdown_locked(force=False)

    def _shutdown_locked(self, force: bool) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive() and not force:
            try:
                self.request_queue.put({"cmd": "stop"})
                process.join(timeout=30)
            except BaseException:
                _LOG.exception("worker=%s graceful shutdown failed", self.kind)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        self._fail_pending("worker stopped")
        self.process = None
        self.listener = None
        self.request_queue = None
        self.response_queue = None

    def status(self) -> dict[str, Any]:
        process = self.process
        return {
            "status": "ready" if process is not None and process.is_alive() else "stopped",
            "pid": process.pid if process is not None else self.metadata.get("pid"),
            "restarts": self.restarts,
            "users": self._users,
            "error": self.last_error,
        }


class WorkerSupervisor:
    """Own the three workers and share Hailo between chat/STT and detector proxies."""

    def __init__(self, settings):
        self.hailo = _WorkerClient("hailo", settings)
        self.litert = _WorkerClient("litert", settings)
        self.piper = _WorkerClient("piper", settings)

    def status(self) -> dict[str, Any]:
        return {
            "gateway": {"status": "ready", "pid": os.getpid(), "restarts": 0},
            "hailo": self.hailo.status(),
            "llm": self.litert.status(),
            "piper": self.piper.status(),
        }


_SUPERVISORS: dict[int, WorkerSupervisor] = {}
_SUPERVISORS_LOCK = threading.Lock()


def _supervisor(settings) -> WorkerSupervisor:
    key = id(settings)
    with _SUPERVISORS_LOCK:
        supervisor = _SUPERVISORS.get(key)
        if supervisor is None:
            supervisor = WorkerSupervisor(settings)
            _SUPERVISORS[key] = supervisor
        return supervisor


class HailoProcessBackend(HailoBackend):
    """HailoBackend-compatible IPC proxy without parent-process Hailo state."""

    def __init__(self, settings, supervisor: WorkerSupervisor):
        self.settings = settings
        self.worker = supervisor.hailo
        self.paths = {}
        self.artifact_paths = {}
        self.minilm = None
        self.llm = None
        self.vlm = None
        self._retrieval_embedding_cache = {}
        self._acquired = False

    def start(self):
        metadata = self.worker.acquire()
        self._acquired = True
        self.paths = metadata.get("paths", {})
        self.artifact_paths = metadata.get("artifact_paths", {})
        self.minilm = object() if metadata.get("minilm_ready") else None

    def chat(self, request, emit=None, cancelled=None):
        callback = getattr(request, "_on_inference", None)
        if callable(callback):
            callback(getattr(request, "_response_language", "de"))
            object.__setattr__(request, "_on_inference", None)
        return self.worker.call("chat", request, emit=emit, cancelled=cancelled)

    def transcribe(self, audio, language=None):
        return self.worker.call("transcribe", audio, language)

    def select_tools(self, request):
        return self.worker.call("select_tools", request)

    def retrieve_context(self, request):
        return self.worker.call("retrieve_context", request)

    def close(self):
        if self._acquired:
            self._acquired = False
            self.worker.release()


class LiteRTProcessBackend(LiteRTLMBackend):
    """LiteRTLMBackend-compatible proxy keeping Gemma out of the gateway."""

    def __init__(self, settings, supervisor: WorkerSupervisor):
        self.settings = settings
        self.worker = supervisor.litert
        self.model_path = settings.litert_model_path or str(
            Path(settings.model_store) / "gemma-4-E2B-it.litertlm"
        )
        self.max_num_tokens = settings.litert_max_num_tokens
        self.max_input_tokens = settings.litert_max_input_tokens
        self.debug_log = settings.debug_log
        self.engine = None
        self._acquired = False

    def start(self):
        self.worker.acquire()
        self._acquired = True

    def chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        callback = getattr(request, "_on_inference", None)
        if callable(callback):
            callback(getattr(request, "_response_language", "de"))
            object.__setattr__(request, "_on_inference", None)
        return self.worker.call(
            "chat",
            request,
            emit=emit,
            cancelled=cancelled,
            tools_prepared=tools_prepared,
        )

    def close(self):
        if self._acquired:
            self._acquired = False
            self.worker.release()


class PiperProcessBackend(PiperBackend):
    """PiperBackend-compatible CPU worker proxy."""

    def __init__(self, settings, supervisor: WorkerSupervisor):
        self.settings = settings
        self.worker = supervisor.piper
        self._acquired = False

    def start(self):
        self.worker.acquire()
        self._acquired = True

    def synthesize(self, request):
        return self.worker.call("synthesize", request)

    def close(self):
        if self._acquired:
            self._acquired = False
            self.worker.release()


class HailoVisionProcessBackend(HailoVisionBackend):
    """Detector proxy sharing the central Hailo process with VLM/STT/MiniLM."""

    def __init__(self, settings, supervisor: WorkerSupervisor):
        self.settings = settings
        self.worker = supervisor.hailo
        self.path = None
        self.input_shape = None
        self._acquired = False

    def start(self):
        metadata = self.worker.acquire()
        self._acquired = True
        self.path = metadata.get("vision_path")
        shape = metadata.get("vision_input_shape")
        self.input_shape = tuple(shape) if shape else None

    def detect(self, frame, confidence, maximum):
        return self.worker.call("vision_detect", frame, confidence, maximum)

    def close(self):
        if self._acquired:
            self._acquired = False
            self.worker.release()


class ProcessRuntime(Runtime):
    """Chat/STT runtime backed by isolated Hailo and LiteRT worker processes."""

    def __init__(self, settings, backend=None, litert_backend=None):
        supervisor = _supervisor(settings)
        if backend is None:
            backend = HailoProcessBackend(settings, supervisor)
        if litert_backend is None and (settings.litert_enabled or settings.litert_model_path):
            litert_backend = LiteRTProcessBackend(settings, supervisor)
        self.worker_supervisor = supervisor
        super().__init__(settings, backend, litert_backend)

    def status(self):
        status = super().status()
        status["workers"] = self.worker_supervisor.status()
        status["process_mode"] = True
        return status


class ProcessSpeechRuntime(SpeechRuntime):
    """Speech runtime whose resident Piper backend lives in its own process."""

    def __init__(self, settings, backend=None):
        supervisor = _supervisor(settings)
        super().__init__(
            settings,
            backend if backend is not None else PiperProcessBackend(settings, supervisor),
        )


class ProcessVisionRuntime(VisionRuntime):
    """Vision runtime sharing the central Hailo process with the chat/STT proxy."""

    def __init__(self, settings, backend=None):
        supervisor = _supervisor(settings)
        super().__init__(
            settings,
            backend if backend is not None else HailoVisionProcessBackend(settings, supervisor),
        )
