"""One owner thread. No per-request VDevice/model creation or idle unloading."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from .config import STT_MODEL, VLM_MODEL, Settings
from .media import image_frame
from .models import prepare_model_version

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
            self.device = VDevice(params)
            self.vlm = VLM(self.device, self.paths["vlm"])
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


class Runtime:
    def __init__(self, settings: Settings, backend=None):
        self.settings = settings
        self.backend = backend if backend is not None else HailoBackend(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hailo-owner")
        self.pending = 0
        self.ready = False

    async def start(self):
        await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.start)
        self.ready = True

    def submit(self, function, *args):
        if not self.ready:
            raise BusyError("Runtime is not ready")
        if self.pending >= self.settings.queue_size:
            raise BusyError("Inference queue is full")
        self.pending += 1
        future = asyncio.get_running_loop().run_in_executor(self.executor, function, *args)
        # A timed out/disconnected client must NOT release capacity before native work ends.
        future.add_done_callback(self._completed)
        return future

    def _completed(self, future):
        self.pending -= 1
        if not future.cancelled():
            future.exception()  # Observe late exceptions after client cancellation/timeout.

    async def call(self, function, *args):
        future = self.submit(function, *args)
        return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout)

    async def chat(self, request):
        return await self.call(self.backend.chat, request)

    async def stream(self, request):
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()  # Bounded by request.max_tokens <= 1024.
        cancelled = threading.Event()
        future = self.submit(
            self.backend.chat,
            request,
            lambda chunk: loop.call_soon_threadsafe(queue.put_nowait, chunk),
            cancelled,
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
            "models": [VLM_MODEL, STT_MODEL] if self.ready else [],
            "model_paths": self.backend.paths,
        }

    async def close(self):
        self.ready = False
        try:
            # Queued work completes before releasing persistent models.
            await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.close)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
