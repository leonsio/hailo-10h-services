"""One owner thread. No per-request VDevice/model creation or idle unloading."""

import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import LLM_MODEL, MINILM_HEF_URL, STT_MODEL, VLM_MODEL, Settings
from .input_budget import InputBudgetError, history_candidates
from .media import image_frame
from .minilm import MiniLM
from .models import ensure_minilm_hef, prepare_model_version
from .tool_calling import (
    has_tool_context,
    native_messages,
    native_tools,
    response_message,
    selected_tools,
)
from .tool_retrieval import compact_static_context, retrieve_tools

_LOG = logging.getLogger(__name__)
_INPUT_TOKEN_SAFETY_MARGIN = 256


class BusyError(RuntimeError):
    pass


class LiteRTInferenceError(RuntimeError):
    pass


class HailoBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = self.vlm = self.whisper = self.minilm = None
        self.paths = {}
        self.artifact_paths = {}
        self._retrieval_embedding_cache = {}

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
        # Cache the HEF before allocating the accelerator.
        self.artifact_paths["minilm_hef"] = str(ensure_minilm_hef(
            self.settings.minilm_hef_path, MINILM_HEF_URL
        ))
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
            self.minilm = MiniLM(self.device, self.artifact_paths["minilm_hef"])
            self.artifact_paths.update(self.minilm.artifacts)
            _LOG.info("All models initialized; VDevice group_id=SHARED; paths=%s", self.paths)
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
                **({"top_p": request.top_p} if request.top_p is not None else {}),
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

    def select_tools(self, request):
        compact_messages, context_stats = compact_static_context(
            request.messages,
            encoder=self.minilm,
            embedding_cache=self._retrieval_embedding_cache,
        )
        request = request.model_copy(update={"messages": compact_messages})
        if context_stats["system_prompts_compacted"]:
            _LOG.info(
                "MiniLM entity retrieval: %d -> %d entities; %d chars removed",
                context_stats["entities_before"],
                context_stats["entities_after"],
                context_stats["characters_removed"],
            )
        if not request.tools:
            return request
        source = request.tools
        if isinstance(request.tool_choice, dict):
            name = request.tool_choice["function"]["name"]
            source = [tool for tool in source if tool["function"]["name"] == name]
        selected, stats = retrieve_tools(
            request.messages,
            source,
            encoder=self.minilm,
            embedding_cache=self._retrieval_embedding_cache,
        )
        required_names = set()
        for message in request.messages:
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    name = call.get("function", {}).get("name")
                    if isinstance(name, str):
                        required_names.add(name)
        if required_names:
            selected_names = {
                tool.get("function", {}).get("name") for tool in selected
            }
            required = [
                tool for tool in source
                if tool.get("function", {}).get("name") in required_names
                and tool.get("function", {}).get("name") not in selected_names
            ]
            selected = required + selected
            stats["tools_after"] = len(selected)
        _LOG.info(
            "MiniLM tool retrieval: %d -> %d tools; %d enum values removed",
            stats["tools_before"], stats["tools_after"], stats["enum_values_removed"],
        )
        return request.model_copy(update={"tools": selected})

    def close(self):
        # Release models before the device, including after partial startup.
        for name in ("minilm", "whisper", "vlm", "device"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    if name == "minilm":
                        resource.close()
                    else:
                        resource.release()
                except Exception:
                    _LOG.exception("Error releasing %s", name)
                finally:
                    setattr(self, name, None)


class LiteRTLMBackend:
    """Resident LiteRT-LM Python Engine for text-only Gemma requests."""

    def __init__(self, model_path, max_num_tokens=16384, max_input_tokens=4096):
        self.model_path = str(Path(model_path).expanduser())
        self.max_num_tokens = max_num_tokens
        self.max_input_tokens = max_input_tokens
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
            str(path), backend=litert_lm.Backend.CPU(), max_num_tokens=self.max_num_tokens
        )
        self.engine = self.engine_context.__enter__()
        _LOG.info("Loaded LiteRT-LM model %s on CPU; max_num_tokens=%d", path, self.max_num_tokens)

    @staticmethod
    def _messages(request):
        messages = native_messages(request.messages)
        if has_tool_context(request):
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
        return messages

    @staticmethod
    def _prompt(messages):
        return messages[-1] if messages[-1]["role"] == "tool" else messages[-1]["content"]

    def _limit_input(self, request, tool_options):
        if request.max_input_tokens is None:
            return request
        if not callable(getattr(self.engine, "tokenize", None)):
            raise ValueError("max_input_tokens requires LiteRT-LM Engine.tokenize; upgrade litert-lm")
        # This limit applies only after tool retrieval, immediately before Gemma.
        # The incoming OpenAI request may be much larger because MiniLM processes
        # its tool catalogue first. Native prefill still must fit the configured
        # Gemma context, with the requested output budget reserved.
        limit = min(request.max_input_tokens, self.max_num_tokens - request.max_tokens - 1)
        if limit < 1:
            raise ValueError("The configured LiteRT context leaves no room for input and output")
        # Validate all original call/result dependencies before dropping history.
        self._messages(request)
        for candidate in history_candidates(request.messages):
            trimmed = request.model_copy(update={"messages": candidate})
            messages = self._messages(trimmed)
            # A fresh native conversation supplies the same tools and prefix
            # messages without running prefill or decoding. Release it before
            # creating the next probe or the inference conversation.
            with self.engine.create_conversation(
                messages=messages[:-1], max_output_tokens=request.max_tokens, **tool_options,
            ) as probe:
                render = getattr(probe, "render_message_to_string", None)
                if not callable(render):
                    raise ValueError(
                        "max_input_tokens requires LiteRT-LM Conversation.render_message_to_string; "
                        "upgrade litert-lm"
                    )
                # render_message_to_string renders one message, not the whole
                # conversation preface. Count every message and the tool JSON,
                # then reserve headroom for LiteRT's system/tool template tokens.
                rendered = "\n".join(render(message) for message in messages)
                rendered_tools = json.dumps(
                    selected_tools(request), ensure_ascii=False, separators=(",", ":")
                ) if has_tool_context(request) else ""
                tokens = len(self.engine.tokenize(rendered + "\n" + rendered_tools))
                tokens += _INPUT_TOKEN_SAFETY_MARGIN + 8 * max(0, len(messages) - 1)
                tokens += 16 * len(selected_tools(request))
            if tokens <= limit:
                _LOG.info(
                    "input_budget model=%s input_tokens=%d limit=%d max_input_tokens=%d "
                    "removed_messages=%d output_reserved=%d",
                    request.model, tokens, limit, request.max_input_tokens,
                    len(request.messages) - len(candidate), request.max_tokens,
                )
                return trimmed
        raise InputBudgetError(
            tokens, limit, request.max_input_tokens, self.max_num_tokens, request.max_tokens
        )

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

    def chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        try:
            return self._chat(request, emit, cancelled, tools_prepared)
        except RuntimeError as exc:
            _LOG.exception("LiteRT inference failed; model=%s context_tokens=%d", request.model, self.max_num_tokens)
            raise LiteRTInferenceError(
                f"LiteRT-LM inference failed (configured context: {self.max_num_tokens} tokens). "
                "Check the preceding native log for the cause. If the input exceeds the context, "
                "increase HAILO_LITERT_MAX_NUM_TOKENS and restart, or reduce the Home Assistant "
                "prompt/history. Larger contexts require more RAM."
            ) from exc

    def _chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        if self.engine is None:
            raise RuntimeError("LiteRT-LM is not ready")
        if request.max_input_tokens is None:
            request = request.model_copy(update={"max_input_tokens": self.max_input_tokens})
        if not tools_prepared and request.tools and not any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in request.messages
        ):
            source_tools = request.tools
            if isinstance(request.tool_choice, dict):
                forced_name = request.tool_choice["function"]["name"]
                source_tools = [
                    tool for tool in request.tools
                    if tool["function"]["name"] == forced_name
                ]
            compact_tools, stats = retrieve_tools(request.messages, source_tools)
            if stats["tools_after"] < stats["tools_before"] or stats["enum_values_removed"]:
                request = request.model_copy(update={"tools": compact_tools})
                _LOG.info(
                    "tool_retrieval model=%s tools=%d->%d enum_values_removed=%d",
                    request.model, stats["tools_before"], stats["tools_after"],
                    stats["enum_values_removed"],
                )
        tool_options = {}
        if has_tool_context(request):
            if not hasattr(self.litert_lm, "Tool"):
                raise ValueError("Installed LiteRT-LM lacks Tool support; upgrade litert-lm")
            tool_options = {
                "tools": native_tools(self.litert_lm, selected_tools(request)),
                "automatic_tool_calling": False,
            }
        request = self._limit_input(request, tool_options)
        messages = self._messages(request)
        with self.engine.create_conversation(
            messages=messages[:-1],
            sampler_config=self.litert_lm.SamplerConfig(
                temperature=request.temperature, seed=request.seed,
                **({"top_p": request.top_p} if request.top_p is not None else {}),
            ),
            max_output_tokens=request.max_tokens,
            **tool_options,
        ) as conversation:
            prompt = self._prompt(messages)
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
            self.litert_backend = LiteRTLMBackend(
                settings.litert_model_path, settings.litert_max_num_tokens,
                settings.litert_max_input_tokens,
            )
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
        if request.max_input_tokens is not None and request.model != LLM_MODEL:
            raise ValueError(f"max_input_tokens requires model {LLM_MODEL}; Qwen image tokens cannot be counted here")
        if has_tool_context(request) and request.model != LLM_MODEL:
            raise ValueError(f"Tool calling requires model {LLM_MODEL}")
        if request.model == LLM_MODEL:
            if not self.litert_ready:
                detail = self.litert_error or "LiteRT-LM model is not configured"
                raise BusyError(f"{LLM_MODEL} is unavailable: {detail}")
            if isinstance(self.backend, HailoBackend):
                request = await self.call(self.backend.select_tools, request)
                return await self.call_litert(
                    self.litert_backend.chat, request, None, None, bool(request.tools)
                )
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
        if request.max_input_tokens is not None and request.model != LLM_MODEL:
            raise ValueError(f"max_input_tokens requires model {LLM_MODEL}; Qwen image tokens cannot be counted here")
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
                "max_num_tokens": getattr(self.litert_backend, "max_num_tokens", None),
                "max_input_tokens": getattr(self.litert_backend, "max_input_tokens", None),
                "pending": self.litert_pending,
                "error": self.litert_error,
            },
            "models": ([VLM_MODEL, STT_MODEL] if self.ready else [])
            + ([LLM_MODEL] if self.litert_ready else []),
            "model_paths": self.backend.paths,
            "artifact_paths": getattr(self.backend, "artifact_paths", {}),
            "minilm_ready": bool(self.ready and getattr(self.backend, "minilm", None)),
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
