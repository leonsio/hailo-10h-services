"""Asynchronous scheduling, routing and lifecycle for resident backends."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hailo_services.chat.backend_hailo import HailoBackend, _validate_hailo_temperature
from hailo_services.chat.backend_litert import LiteRTLMBackend
from hailo_services.config import FRIGATE_ASSIST_MODEL, HA_ASSIST_MODEL, LLM_MODEL, Settings
from hailo_services.diagnostics.metrics import record
from hailo_services.interfaces import ChatBackend, ChatResult
from hailo_services.runtime.models import ModelManager
from hailo_services.schemas import ChatRequest
from hailo_services.shared.errors import BusyError, LiteRTInferenceError
from hailo_services.shared.tool_calling import has_tool_context

__all__ = ["BusyError", "HailoBackend", "LiteRTInferenceError", "LiteRTLMBackend", "Runtime"]
_LOG = logging.getLogger(__name__)


class Runtime:
    """Schedule resident chat and speech backends on separate bounded owner threads."""

    def __init__(
        self,
        settings: Settings,
        backend: ChatBackend | None = None,
        litert_backend: ChatBackend | None = None,
    ):
        """Initialize Runtime configuration and owned dependencies.

        Args:
            settings (Settings): Validated service settings controlling enabled models and limits.
            backend (ChatBackend | None): Resident backend used for generation or context preparation.
            litert_backend (ChatBackend | None): Injected LiteRT backend; None creates one when enabled.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.settings = settings
        self.backend = backend if backend is not None else HailoBackend(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hailo-owner")
        self.pending = 0
        self.ready = False
        self.litert_backend = litert_backend
        if self.litert_backend is None and (settings.litert_enabled or settings.litert_model_path):
            self.litert_backend = LiteRTLMBackend(
                settings.litert_model_path
                or str(Path(settings.model_store) / "gemma-4-E2B-it.litertlm"),
                settings.litert_max_num_tokens,
                settings.litert_max_input_tokens,
                settings.debug_log,
                ModelManager(settings),
            )
        if self.litert_backend is not None:
            self.litert_backend.debug_log = settings.debug_log
        self.litert_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="litert-lm-owner")
            if self.litert_backend is not None
            else None
        )
        self.litert_pending = 0
        self.litert_ready = False
        self.litert_error = None

    async def start_hailo(self):
        """Initialize the resident Hailo backends without starting CPU models.

        Returns:
            None: Marks the Hailo runtime ready after successful initialization.

        Notes:
            This phase intentionally excludes LiteRT-LM so callers can load every
            Hailo-10H resident model before allocating CPU model resources.
        """
        await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.start)
        self.ready = True

    async def start_cpu(self):
        """Initialize optional CPU-backed models after Hailo startup completes.

        Returns:
            None: Marks LiteRT-LM ready when configured and successfully initialized.

        Notes:
            LiteRT-LM startup failures remain non-fatal so initialized Hailo models
            stay available exactly as they did before startup was split into phases.
        """
        if self.litert_backend is not None:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    self.litert_executor, self.litert_backend.start
                )
                self.litert_ready = True
            except Exception as exc:
                self.litert_error = f"{type(exc).__name__}: {exc}"
                _LOG.exception("LiteRT-LM startup failed; Hailo models remain available")

    async def start(self):
        """Initialize Hailo resources followed by optional CPU resources.

        Returns:
            None: Marks configured resident resources ready after initialization.

        Notes:
            This compatibility entry point preserves the previous public lifecycle
            while exposing separate startup phases to the application orchestrator.
        """
        await self.start_hailo()
        await self.start_cpu()

    def submit(self, function, *args, executor=None, litert=False):
        """Schedule native work without releasing queue capacity on client timeout.

        Args:
            function (Callable[..., Any]): Synchronous work executed on the backend owner thread.
            executor (Executor | None): Optional owner executor override.
            litert (bool): Whether to charge the independent LiteRT queue.
            *args (Any): Positional arguments forwarded to the native operation.

        Returns:
            asyncio.Future[Any]: Future completing when owner-thread work actually finishes.

        Raises:
            BusyError: Runtime is not ready.
        """
        if not self.ready:
            raise BusyError("Runtime is not ready")
        pending = self.litert_pending if litert else self.pending
        if pending >= self.settings.queue_size:
            raise BusyError("Inference queue is full")
        if litert:
            self.litert_pending += 1
        else:
            self.pending += 1
        future = asyncio.get_running_loop().run_in_executor(
            executor or self.executor, function, *args
        )
        # A timed out/disconnected client must NOT release capacity before native work ends.
        future.add_done_callback(lambda completed: self._completed(completed, litert))
        return future

    def _completed(self, future, litert=False):
        """Release queue capacity and observe late worker exceptions.

        Args:
            future (asyncio.Future[Any]): Completed future for native owner-thread work.
            litert (bool): Whether to charge the independent LiteRT queue.

        Returns:
            None: Updates pending counters after native work completes.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if litert:
            self.litert_pending -= 1
        else:
            self.pending -= 1
        if not future.cancelled():
            future.exception()  # Observe late exceptions after client cancellation/timeout.

    async def call(self, function, *args):
        """Await Hailo owner-thread work with a shielded request deadline.

        Args:
            function (Callable[..., Any]): Synchronous work executed on the backend owner thread.
            *args (Any): Positional arguments forwarded to the native operation.

        Returns:
            Any: Result returned by the submitted native callable.

        Raises:
            BusyError: The runtime is unavailable or its owner queue is full.
            asyncio.TimeoutError: The client deadline expires; native work remains shielded.
        """
        future = self.submit(function, *args)
        return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout)

    def default_chat_request(self, request):
        """Resolve an omitted model and validate native Hailo sampling constraints.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

        Returns:
            ChatRequest: Request with an enabled default text or vision model.

        Raises:
            ValueError: Image requests require an enabled VLM.
        """
        if "model" not in request.model_fields_set:
            images = any(
                part.get("type") == "image_url"
                for m in request.messages
                for part in (m.get("content") if isinstance(m.get("content"), list) else [])
            )
            if images:
                if not self.settings.vlm_enabled:
                    raise ValueError("Image requests require an enabled VLM")
                model = self.settings.vlm_model
            else:
                model = self.default_text_model or LLM_MODEL
            request = request.model_copy(update={"model": model})
        if request.model in self.hailo_chat_models:
            kind = "LLM" if request.model == self.settings.hailo_llm_model_id else "VLM"
            _validate_hailo_temperature(request, kind)
        return request

    async def chat(self, request: ChatRequest, on_inference=None) -> ChatResult:
        """Route one chat request through deterministic HA preparation or a native backend.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
            on_inference (Callable[[str], None] | None): Optional callback notified with response language before generation.

        Returns:
            ChatResult: Deterministic response or generated text/tool message.

        Raises:
            ValueError: The selected model or HA-Assist route is invalid.
            BusyError: A required backend is unavailable or its queue is full.
            InputBudgetError: Required input exceeds the effective budget.
            LiteRTInferenceError: Native LiteRT generation fails.
            asyncio.TimeoutError: The configured request deadline expires.
        """
        request = self.default_chat_request(request)
        if request.model == FRIGATE_ASSIST_MODEL:
            from hailo_services.assistants.frigate.frigate_assist import prepare
            from hailo_services.assistants.frigate.frigate_routing import validate_result

            original = request
            request, direct = prepare(self.settings, request)
            if direct is not None:
                record(
                    request._metrics,
                    inference_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    input_tokens_source="deterministic",
                    output_tokens_source="deterministic",
                )
                return direct
            request = self.default_chat_request(request)
            if request.model == LLM_MODEL:
                if not self.litert_ready:
                    raise BusyError(
                        f"Frigate-Assist text backend is unavailable: {self.litert_error or 'Gemma is not ready'}"
                    )
                if isinstance(self.litert_backend, LiteRTLMBackend):
                    result = await self.call_litert(
                        self.litert_backend.chat, request, None, None, True
                    )
                else:
                    result = await self.call_litert(self.litert_backend.chat, request)
            else:
                result = await self.call(self.backend.chat, request)
            return validate_result(result, request, original)
        if request.model == HA_ASSIST_MODEL:
            from hailo_services.assistants.ha.ha_assist import prepare, target_model

            target = target_model(self.settings, request)
            request = request.model_copy(update={"model": target})
            request, direct = await self.call(prepare, self.backend, request)
            if direct is not None:
                record(
                    request._metrics,
                    inference_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    input_tokens_source="deterministic",
                    output_tokens_source="deterministic",
                )
                return direct
            if on_inference is not None:
                object.__setattr__(request, "_on_inference", on_inference)
            # Validate the selected backend once, never reroute on failure.
            request = self.default_chat_request(request)
        if request.model == LLM_MODEL:
            if not self.litert_ready:
                detail = self.litert_error or "LiteRT-LM model is not configured"
                raise BusyError(f"{LLM_MODEL} is unavailable: {detail}")
            if isinstance(self.litert_backend, LiteRTLMBackend):
                return await self.call_litert(self.litert_backend.chat, request, None, None, True)
            return await self.call_litert(self.litert_backend.chat, request)
        if request.model not in self.hailo_chat_models:
            raise ValueError(f"Unknown model: {request.model}")
        return await self.call(self.backend.chat, request)

    async def call_litert(self, function, *args):
        """Await LiteRT owner-thread work with independent bounded capacity.

        Args:
            function (Callable[..., Any]): Synchronous work executed on the backend owner thread.
            *args (Any): Positional arguments forwarded to the submitted LiteRT callable.

        Returns:
            Any: Result returned by the submitted LiteRT callable.

        Raises:
            BusyError: The LiteRT backend is unavailable or its owner queue is full.
            asyncio.TimeoutError: The client deadline expires; native work remains shielded.
        """
        future = self.submit(function, *args, executor=self.litert_executor, litert=True)
        return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout)

    async def stream(self, request):
        """Yield output chunks and record latency when iteration finishes.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

        Yields:
            ChatResult: Text chunks or validated assistant messages.

        Raises:
            ValueError: The requested chat model is unknown or its sampling policy is invalid.
            BusyError: The runtime, model or selected owner queue is unavailable.
        """
        request = self.default_chat_request(request)
        if request.model in {HA_ASSIST_MODEL, FRIGATE_ASSIST_MODEL} or has_tool_context(request):
            yield await self.chat(request)
            return
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()  # Bounded by request.max_tokens <= 1024.
        cancelled = threading.Event()
        litert = request.model == LLM_MODEL
        if request.model not in self.hailo_chat_models and not litert:
            raise ValueError(f"Unknown model: {request.model}")
        if litert and not self.litert_ready:
            detail = self.litert_error or "LiteRT-LM model is not configured"
            raise BusyError(f"{LLM_MODEL} is unavailable: {detail}")
        future = self.submit(
            self.litert_backend.chat if litert else self.backend.chat,
            request,
            lambda chunk: loop.call_soon_threadsafe(queue.put_nowait, chunk),
            cancelled,
            executor=self.litert_executor if litert else self.executor,
            litert=litert,
        )
        future.add_done_callback(lambda _: queue.put_nowait(None))
        deadline = loop.time() + self.settings.request_timeout
        try:
            while True:
                chunk = await asyncio.wait_for(queue.get(), max(0.001, deadline - loop.time()))
                if chunk is None:
                    await asyncio.shield(future)
                    break
                yield chunk
        finally:
            cancelled.set()

    async def transcribe(self, audio, language=None):
        """Transcribe normalized audio through the resident Whisper model.

        Args:
            audio (np.ndarray): Normalized 16 kHz mono float32 samples for Whisper.
            language (str | None): Language code; None uses the configured or detected language.

        Returns:
            str: Recognized speech text.

        Raises:
            BusyError: Whisper is disabled or the runtime queue is unavailable.
            asyncio.TimeoutError: Transcription exceeds the request deadline.
        """
        return await self.call(self.backend.transcribe, audio, language or self.settings.language)

    @property
    def hailo_chat_models(self):
        """List enabled Hailo generative model identifiers.

        Returns:
            list[str]: Configured enabled LLM and VLM identifiers.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return ([self.settings.vlm_model] if self.settings.vlm_enabled else []) + (
            [self.settings.hailo_llm_model_id] if self.settings.hailo_llm_enabled else []
        )

    @property
    def chat_models(self):
        """List enabled chat backends and the optional virtual HA-Assist model.

        Returns:
            list[str]: Public chat model identifiers.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return (
            self.hailo_chat_models
            + ([LLM_MODEL] if self.litert_ready else [])
            + ([HA_ASSIST_MODEL] if self.settings.ha_assist_enabled else [])
            + ([FRIGATE_ASSIST_MODEL] if self.settings.frigate_assist_enabled else [])
        )

    @property
    def default_text_model(self):
        """Choose the configured default available text generation backend.

        Returns:
            str | None: Default text model, or None when no text backend is enabled.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if self.litert_ready:
            return LLM_MODEL
        if self.settings.hailo_llm_enabled:
            return self.settings.hailo_llm_model_id
        return self.settings.vlm_model if self.settings.vlm_enabled else None

    @property
    def model_limits(self):
        """Expose input limits only for initialized, enabled chat backends.

        Returns:
            dict[str, Any]: Limits for ready chat models; empty before startup or after shutdown.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if not self.ready:
            return {}
        limits = {}
        if self.settings.vlm_enabled:
            limits[self.settings.vlm_model] = {
                "max_input_tokens": self.settings.vlm_max_input_tokens,
                "context_length": 2048,
            }
        if self.settings.hailo_llm_enabled:
            limits[self.settings.hailo_llm_model_id] = {
                "max_input_tokens": self.settings.hailo_llm_max_input_tokens,
                "context_length": 2048,
            }
        if self.litert_ready:
            limits[LLM_MODEL] = {
                "max_input_tokens": self.settings.litert_max_input_tokens,
                "context_length": self.settings.litert_max_num_tokens,
            }
        return limits

    def status(self):
        """Summarize resident readiness, model paths and pending requests.

        Returns:
            dict[str, Any]: Serializable health and configuration details.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        from hailo_services.diagnostics.diagnostics import cache_status

        return {
            "ready": self.ready,
            "group_id": "SHARED",
            "cache": cache_status(self.backend, self.litert_backend),
            "pending": self.pending,
            "model_limits": self.model_limits,
            "default_text_model": self.default_text_model,
            "ha_assist": {
                "enabled": self.settings.ha_assist_enabled,
                "model": HA_ASSIST_MODEL,
                "text_model": self.settings.ha_assist_text_model,
                "vision_model": self.settings.ha_assist_vision_model,
                "text_ready": bool(
                    self.settings.ha_assist_enabled
                    and self.ready
                    and (
                        self.settings.ha_assist_text_model == LLM_MODEL
                        and self.litert_ready
                        or self.settings.hailo_llm_enabled
                        and self.settings.ha_assist_text_model == self.settings.hailo_llm_model_id
                    )
                ),
                "vision_ready": bool(
                    self.settings.ha_assist_enabled
                    and self.ready
                    and self.settings.vlm_enabled
                    and self.settings.ha_assist_vision_model == self.settings.vlm_model
                ),
            },
            "frigate_assist": {
                "enabled": self.settings.frigate_assist_enabled,
                "experimental": True,
                "model": FRIGATE_ASSIST_MODEL,
                "text_model": self.settings.frigate_text_model,
                "vision_model": self.settings.frigate_vision_model,
                "text_ready": bool(
                    self.settings.frigate_assist_enabled
                    and (
                        self.settings.frigate_text_model == LLM_MODEL
                        and self.litert_ready
                        or self.settings.frigate_text_model == self.settings.hailo_llm_model_id
                        and self.settings.hailo_llm_enabled
                        and self.ready
                        or self.settings.frigate_text_model == self.settings.vlm_model
                        and self.settings.vlm_enabled
                        and self.ready
                    )
                ),
                "vision_ready": bool(
                    self.settings.frigate_assist_enabled
                    and self.ready
                    and self.settings.vlm_enabled
                    and self.settings.frigate_vision_model == self.settings.vlm_model
                ),
            },
            "litert_lm": {
                "ready": self.litert_ready,
                "model": LLM_MODEL if self.litert_backend else None,
                "model_path": getattr(self.litert_backend, "model_path", None),
                "max_num_tokens": getattr(self.litert_backend, "max_num_tokens", None),
                "max_input_tokens": getattr(self.litert_backend, "max_input_tokens", None),
                "pending": self.litert_pending,
                "error": self.litert_error,
            },
            "models": (
                self.hailo_chat_models
                + ([self.settings.stt_model] if self.settings.whisper_enabled else [])
                if self.ready
                else []
            )
            + ([LLM_MODEL] if self.litert_ready else [])
            + ([HA_ASSIST_MODEL] if self.settings.ha_assist_enabled else [])
            + ([FRIGATE_ASSIST_MODEL] if self.settings.frigate_assist_enabled else []),
            "model_paths": self.backend.paths,
            "artifact_paths": getattr(self.backend, "artifact_paths", {}),
            "minilm_ready": bool(self.ready and getattr(self.backend, "minilm", None)),
        }

    async def close(self):
        """Release resources owned by this service or native context.

        Returns:
            None: Closes native resources, connections or owner executors.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.ready = False
        try:
            # Queued work completes before releasing persistent models.
            await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.close)
            if self.litert_backend is not None:
                await asyncio.get_running_loop().run_in_executor(
                    self.litert_executor, self.litert_backend.close
                )
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            if self.litert_executor is not None:
                self.litert_executor.shutdown(wait=True, cancel_futures=True)
