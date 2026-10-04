"""One owner thread. No per-request VDevice/model creation or idle unloading."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import LLM_MODEL, STT_MODEL, VLM_MODEL, Settings
from .media import image_frame
from .models import prepare_model_version
from .tool_calling import (
    has_tool_context,
    native_messages,
    native_tools,
    response_message,
    selected_tools,
)

_LOG = logging.getLogger(__name__)


class BusyError(RuntimeError):
    pass


class HailoBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = self.vlm = self.whisper = None
        self.paths = {}

    def start(self):
        from hailo_apps.python.core.common.core import resolve_hef_path
        from hailo_platform import VDevice
        from hailo_platform.genai import VLM, Speech2Text

        prepare_model_version()
        # Resolve/download BEFORE allocating accelerator resources.
        for key, model, group in (
            ("vlm", self.settings.vlm_hef, "vlm_chat"),
            ("whisper", self.settings.whisper_hef, "whisper_chat"),
        ):
            path = resolve_hef_path(model, app_name=group, arch="hailo10h")
            if path is None or not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Could not resolve/download {model}")
            self.paths[key] = str(path)
        params = VDevice.create_params()
        params.group_id = "SHARED"  # Mandatory, intentionally not configurable.
        try:
            if params.group_id != "SHARED":
                raise RuntimeError("Hailo binding did not preserve mandatory group_id=SHARED")
            _LOG.info("Creating Hailo VDevice with effective group_id=%s", params.group_id)
            self.device = VDevice(params)
            _LOG.info("Loading resident Qwen2-VL from %s", self.paths["vlm"])
            self.vlm = VLM(self.device, self.paths["vlm"])
            _LOG.info("Loading resident Whisper Base from %s", self.paths["whisper"])
            self.whisper = Speech2Text(self.device, self.paths["whisper"])
            _LOG.info("Both models initialized; VDevice group_id=SHARED; paths=%s", self.paths)
        except BaseException:
            self.close()
            raise

    def chat(self, request, emit=None, cancelled=None):
        prompt, frames = [], []
        for message in request.messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            converted = []
            for part in content:
                if part["type"] == "image_url":
                    if len(frames) >= 4:
                        raise ValueError("At most four images per request")
                    frames.append(image_frame(part["image_url"]["url"], self.settings.max_body))
                    converted.append({"type": "image"})
                else:
                    converted.append({"type": "text", "text": part["text"]})
            prompt.append({"role": message["role"], "content": converted})
        output = []
        # Clear only KV context, never unload the model weights.
        try:
            self.vlm.clear_context()
            with self.vlm.generate(
                prompt=prompt,
                frames=frames,
                temperature=request.temperature,
                seed=request.seed,
                max_generated_tokens=request.max_tokens,
            ) as generation:
                for chunk in generation:
                    if cancelled is not None and cancelled.is_set():
                        break
                    chunk = chunk.replace("<|im_end|>", "")
                    if chunk:
                        output.append(chunk)
                        if emit:
                            emit(chunk)
            return "".join(output).strip()
        finally:
            self.vlm.clear_context()

    def transcribe(self, audio, language):
        from hailo_platform.genai import Speech2TextTask

        segments = self.whisper.generate_all_segments(
            audio_data=audio,
            task=Speech2TextTask.TRANSCRIBE,
            language=language,
            timeout_ms=int(self.settings.request_timeout * 1000),
        )
        return "".join(segment.text for segment in segments).strip()

    def close(self):
        # Release models before the device, including after partial startup.
        for name in ("whisper", "vlm", "device"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    _LOG.exception("Error releasing %s", name)
                finally:
                    setattr(self, name, None)


class LiteRTLMBackend:
    """Resident LiteRT-LM Python Engine for text-only Gemma requests."""

    def __init__(self, model_path):
        self.model_path = str(Path(model_path).expanduser())
        self.engine_context = None
        self.engine = None
        self.litert_lm = None

    def start(self):
        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"LiteRT-LM model not found: {path}")
        import litert_lm

        self.litert_lm = litert_lm
        self.engine_context = litert_lm.Engine(
            str(path), backend=litert_lm.Backend.CPU()
        )
        self.engine = self.engine_context.__enter__()
        _LOG.info("Loaded LiteRT-LM model %s on CPU", path)

    @staticmethod
    def _messages(request):
        return native_messages(request.messages)

    @staticmethod
    def _chunk_text(chunk):
        if hasattr(chunk, "to_json"):
            chunk = chunk.to_json()
        if isinstance(chunk, dict) and isinstance(chunk.get("content"), str):
            return chunk["content"]
        return "".join(
            item.get("text", "")
            for item in chunk.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ) if isinstance(chunk, dict) else ""

    def chat(self, request, emit=None, cancelled=None):
        if self.engine is None:
            raise RuntimeError("LiteRT-LM is not ready")
        messages = self._messages(request)
        tool_options = {}
        if has_tool_context(request):
            if not hasattr(self.litert_lm, "Tool"):
                raise ValueError("Installed LiteRT-LM lacks Tool support; upgrade litert-lm")
            tool_options = {
                "tools": native_tools(self.litert_lm, selected_tools(request)),
                "automatic_tool_calling": False,
            }
            instructions = []
            if request.tool_choice == "required" or isinstance(request.tool_choice, dict):
                instructions.append("Return a function call using one of the available tools.")
            if not request.parallel_tool_calls:
                instructions.append("Return at most one function call in this response.")
            if instructions:
                instruction = "\n".join(instructions)
                if messages[0]["role"] == "system":
                    messages[0]["content"] += "\n" + instruction
                else:
                    messages.insert(0, {"role": "system", "content": instruction})
        with self.engine.create_conversation(
            messages=messages[:-1],
            sampler_config=self.litert_lm.SamplerConfig(
                temperature=request.temperature, seed=request.seed
            ),
            max_output_tokens=request.max_tokens,
            **tool_options,
        ) as conversation:
            prompt = messages[-1] if messages[-1]["role"] == "tool" else messages[-1]["content"]
            if emit is None or has_tool_context(request):
                response = conversation.send_message(
                    prompt, max_output_tokens=request.max_tokens
                )
                result = response_message(response, request, self._chunk_text(response).strip())
                if emit:
                    emit(result)
                return result
            output = []
            for chunk in conversation.send_message_async(
                prompt, max_output_tokens=request.max_tokens
            ):
                if cancelled is not None and cancelled.is_set():
                    break
                text = self._chunk_text(chunk)
                if text:
                    output.append(text)
                    emit(text)
            return "".join(output).strip()

    def close(self):
        if self.engine_context is not None:
            try:
                self.engine_context.__exit__(None, None, None)
            finally:
                self.engine = self.engine_context = self.litert_lm = None


class Runtime:
    def __init__(self, settings: Settings, backend=None, litert_backend=None):
        self.settings = settings
        self.backend = backend if backend is not None else HailoBackend(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hailo-owner")
        self.pending = 0
        self.ready = False
        self.litert_backend = litert_backend
        if self.litert_backend is None and settings.litert_model_path:
            self.litert_backend = LiteRTLMBackend(settings.litert_model_path)
        self.litert_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="litert-lm-owner")
            if self.litert_backend is not None else None
        )
        self.litert_pending = 0
        self.litert_ready = False
        self.litert_error = None

    async def start(self):
        await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.start)
        self.ready = True
        if self.litert_backend is not None:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    self.litert_executor, self.litert_backend.start
                )
                self.litert_ready = True
            except Exception as exc:
                self.litert_error = f"{type(exc).__name__}: {exc}"
                _LOG.exception("LiteRT-LM startup failed; Hailo models remain available")

    def submit(self, function, *args, executor=None, litert=False):
        if not self.ready:
            raise BusyError("Runtime is not ready")
        pending = self.litert_pending if litert else self.pending
        if pending >= self.settings.queue_size:
            raise BusyError("Inference queue is full")
        if litert:
            self.litert_pending += 1
        else:
            self.pending += 1
        future = asyncio.get_running_loop().run_in_executor(executor or self.executor, function, *args)
        # A timed out/disconnected client must NOT release capacity before native work ends.
        future.add_done_callback(lambda completed: self._completed(completed, litert))
        return future

    def _completed(self, future, litert=False):
        if litert:
            self.litert_pending -= 1
        else:
            self.pending -= 1
        if not future.cancelled():
            future.exception()  # Observe late exceptions after client cancellation/timeout.

    async def call(self, function, *args):
        future = self.submit(function, *args)
        return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout)

    async def chat(self, request):
        if has_tool_context(request) and request.model != LLM_MODEL:
            raise ValueError(f"Tool calling requires model {LLM_MODEL}")
        if request.model == LLM_MODEL:
            if not self.litert_ready:
                detail = self.litert_error or "LiteRT-LM model is not configured"
                raise BusyError(f"{LLM_MODEL} is unavailable: {detail}")
            return await self.call_litert(self.litert_backend.chat, request)
        if request.model != VLM_MODEL:
            raise ValueError(f"Unknown model: {request.model}")
        return await self.call(self.backend.chat, request)

    async def call_litert(self, function, *args):
        future = self.submit(
            function, *args, executor=self.litert_executor, litert=True
        )
        return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout)

    async def stream(self, request):
        if has_tool_context(request):
            yield await self.chat(request)
            return
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()  # Bounded by request.max_tokens <= 1024.
        cancelled = threading.Event()
        litert = request.model == LLM_MODEL
        if request.model != VLM_MODEL and not litert:
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
        return await self.call(self.backend.transcribe, audio, language or self.settings.language)

    def status(self):
        return {
            "ready": self.ready,
            "group_id": "SHARED",
            "pending": self.pending,
            "litert_lm": {
                "ready": self.litert_ready,
                "model": LLM_MODEL if self.litert_backend else None,
                "model_path": getattr(self.litert_backend, "model_path", None),
                "pending": self.litert_pending,
                "error": self.litert_error,
            },
            "models": ([VLM_MODEL, STT_MODEL] if self.ready else [])
            + ([LLM_MODEL] if self.litert_ready else []),
            "model_paths": self.backend.paths,
        }

    async def close(self):
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
