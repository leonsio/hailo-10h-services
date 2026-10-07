"""Resident SHARED Hailo backend for GenAI chat, Whisper and MiniLM retrieval."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Event

from .chat_hailo_llm import limit_request as limit_llm_request
from .chat_hailo_llm import tool_response as llm_tool_response
from .chat_hailo_vlm import limit_request, tool_response
from .config import Settings
from .diagnostics import debug_json
from .errors import BusyError
from .interfaces import ChatEmitter, ChatResult
from .media import image_frame
from .metrics import count_output, record
from .minilm import MiniLM
from .models import ModelManager, prepare_model_version
from .schemas import ChatRequest
from .speech_hailo_whisper import create_model as create_whisper_model
from .speech_hailo_whisper import transcribe as transcribe_whisper
from .tool_calling import (
    has_tool_context,
)
from .tool_retrieval import compact_static_context, retrieve_tools

_LOG = logging.getLogger(__name__)


def _validate_hailo_temperature(request, kind="VLM"):
    """Reject zero temperature before entering the Hailo GenAI API.

    Args:
        request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
        kind (str): Model role, measurement category or environment query kind.

    Returns:
        None: Accepts strictly positive sampling temperatures.

    Raises:
        ValueError: Temperature is zero or negative; Hailo GenAI requires a positive value.
    """
    if request.temperature <= 0:
        raise ValueError(
            f"Hailo {kind} requires temperature > 0; use temperature=0.1. "
            "Gemma accepts temperature=0, but HailoRT does not."
        )


class HailoBackend:
    """Own resident SHARED Hailo GenAI models, speech and semantic retrieval."""

    def __init__(self, settings: Settings):
        """Initialize HailoBackend configuration and owned dependencies.

        Args:
            settings (Settings): Validated service settings controlling enabled models and limits.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.settings = settings
        self.device = self.vlm = self.llm = self.whisper = self.minilm = None
        self.paths = {}
        self.artifact_paths = {}
        self._retrieval_embedding_cache = {}

    def start(self):
        """Load enabled GenAI and MiniLM models once on a SHARED Hailo device.

        Returns:
            None: Marks the service ready after successful initialization.

        Raises:
            RuntimeError: Native model initialization fails.
            OSError: A required model artifact cannot be read or downloaded.
        """
        if not any(
            (
                self.settings.vlm_enabled,
                self.settings.hailo_llm_enabled,
                self.settings.whisper_enabled,
                self.settings.minilm_enabled,
            )
        ):
            return
        manager = ModelManager(self.settings, prepare_model_version())
        for key, model, kind, enabled in (
            ("vlm", self.settings.vlm_hef, "vlm", self.settings.vlm_enabled),
            ("whisper", self.settings.whisper_hef, "whisper", self.settings.whisper_enabled),
            ("llm", self.settings.hailo_llm_model, "llm", self.settings.hailo_llm_enabled),
        ):
            if enabled:
                self.paths[key] = str(manager.resolve(model, kind))
        if self.settings.minilm_enabled:
            self.artifact_paths["minilm_hef"] = str(
                manager.resolve("minilm-l6-ruvector", "embedding", self.settings.minilm_hef_path)
            )
            # Host assets are also prepared before accelerator allocation.
            for name in ("minilm-tokenizer", "minilm-weights"):
                manager.resolve(
                    name,
                    "asset",
                    Path(self.artifact_paths["minilm_hef"]).parent
                    / manager.entry(name)["filename"],
                )
        from hailo_platform import HailoSchedulingAlgorithm, VDevice

        params = VDevice.create_params()
        params.group_id = "SHARED"  # Mandatory, intentionally not configurable.
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        try:
            if params.group_id != "SHARED":
                raise RuntimeError("Hailo binding did not preserve mandatory group_id=SHARED")
            _LOG.info("Creating Hailo VDevice group_id=%s scheduling=ROUND_ROBIN", params.group_id)
            if self.settings.vision_enabled and self.settings.vision_scheduler_priority <= 16:
                _LOG.warning(
                    "vision_scheduler_priority=%d does not outrank Hailo normal priority 16; "
                    "use 31 for latency-sensitive Frigate detection during GenAI inference",
                    self.settings.vision_scheduler_priority,
                )
            self.device = VDevice(params)
            if self.settings.vlm_enabled:
                from hailo_platform.genai import VLM

                _LOG.info("Loading resident VLM %s", self.paths["vlm"])
                self.vlm = VLM(self.device, self.paths["vlm"])
            if self.settings.hailo_llm_enabled:
                from hailo_platform.genai import LLM

                _LOG.info("Loading resident Hailo LLM %s", self.paths["llm"])
                self.llm = LLM(self.device, self.paths["llm"])
            if self.settings.whisper_enabled:
                _LOG.info("Loading resident Whisper %s", self.paths["whisper"])
                self.whisper = create_whisper_model(self.device, self.paths["whisper"])
            if self.settings.minilm_enabled:
                self.minilm = MiniLM(self.device, self.artifact_paths["minilm_hef"], manager)
                self.artifact_paths.update(self.minilm.artifacts)
            _LOG.info(
                "All models initialized; VDevice group_id=SHARED scheduling=ROUND_ROBIN; paths=%s",
                self.paths,
            )
        except BaseException:
            self.close()
            raise

    def chat(
        self, request: ChatRequest, emit: ChatEmitter | None = None, cancelled: Event | None = None
    ) -> ChatResult:
        """Generate a completion or stream model output for a chat request.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
            emit (ChatEmitter | None): Optional callback receiving text chunks or validated assistant messages.
            cancelled (Event | None): Optional cancellation event checked between generated chunks.

        Returns:
            ChatResult: Final text or a validated assistant tool-call message.

        Raises:
            ValueError: Model selection, sampling, image format or generated calls are invalid.
            BusyError: The selected native model is unavailable.
            InputBudgetError: The mandatory active context exceeds the input limit.
            RuntimeError: Native Hailo generation fails.
        """
        from .ha_action_verification import action_verification_response
        from .ha_assist import has_images
        from .i18n import using_language

        with using_language(
            getattr(
                request, "_response_language", request.language or self.settings.service_language
            )
        ):
            decision = (
                action_verification_response(request)
                if getattr(request, "_ha_assist", False) and not has_images(request)
                else None
            )
        direct = (
            decision["response"]
            if decision
            else next(
                (
                    getattr(request, name)
                    for name in (
                        "_direct_ha_response",
                        "_direct_ha_state_response",
                        "_direct_weather_response",
                    )
                    if getattr(request, name, None) is not None
                ),
                None,
            )
        )
        if direct is not None:
            if emit:
                emit(direct)
            return direct
        if "model" not in request.model_fields_set:
            request = request.model_copy(
                update={
                    "model": (
                        self.settings.hailo_llm_model_id
                        if self.settings.hailo_llm_enabled
                        else self.settings.vlm_model
                    )
                }
            )
        if request.model == self.settings.vlm_model and self.settings.vlm_enabled:
            model, kind = self.vlm, "vlm"
            configured_limit = self.settings.vlm_max_input_tokens
            budget, parse_response = limit_request, tool_response
        elif request.model == self.settings.hailo_llm_model_id and self.settings.hailo_llm_enabled:
            model, kind = self.llm, "llm"
            configured_limit = self.settings.hailo_llm_max_input_tokens
            budget, parse_response = limit_llm_request, llm_tool_response
        else:
            raise ValueError(f"Unknown or disabled Hailo model: {request.model}")
        if model is None:
            raise BusyError(f"{request.model} is disabled")
        _validate_hailo_temperature(request, kind.upper())
        entry = ModelManager(self.settings).entries.get(request.model, {})
        if entry.get("tool_calling", {}).get("parallel_calls") is False:
            # Apply to both the compact prompt contract and output validation.
            # A caller cannot override a model's single-call restriction.
            request = request.model_copy(update={"parallel_tool_calls": False})
        request, prompt = budget(
            model,
            request,
            configured_limit,
            entry.get("context_length", 2048),
            debug=self.settings.debug_log,
            template_options=entry.get("prompt_template"),
        )
        size = tuple(entry.get("frame_size", [336, 336])) if kind == "vlm" else None
        if kind == "vlm" and callable(getattr(model, "input_frame_shape", None)):
            height, width, channels = model.input_frame_shape()
            if channels != 3 or height < 1 or width < 1:
                raise ValueError("VLM requires an unsupported input frame format")
            size = (width, height)
        frames = []
        max_images = entry.get("max_images", 1)
        for message in request.messages:
            for part in message["content"] if isinstance(message.get("content"), list) else []:
                if part["type"] == "image_url":
                    if len(frames) >= max_images:
                        raise ValueError(
                            f"{request.model} supports at most {max_images} image(s) per request"
                        )
                    frames.append(
                        image_frame(part["image_url"]["url"], self.settings.max_body, size)
                    )
        debug_json(
            _LOG,
            self.settings.debug_log,
            f"final_{kind}_request",
            {
                "prompt": prompt,
                "frame_size": size,
                "images": len(frames),
                "model": request.model,
                "max_input_tokens": configured_limit,
                "parallel_tool_calls": request.parallel_tool_calls,
                "prompt_template": entry.get("prompt_template", {}),
            },
            request_id=request._request_id,
        )
        output = []
        started = time.perf_counter()
        first_chunk_ms = None
        # Clear only KV context, never unload the model weights.
        try:
            model.clear_context()
            on_inference = getattr(request, "_on_inference", None)
            if on_inference is not None:
                on_inference(getattr(request, "_response_language", self.settings.service_language))
            with model.generate(
                prompt=prompt,
                **({"frames": frames} if kind == "vlm" else {}),
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
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - started) * 1000
                        output.append(chunk)
                        if emit and not has_tool_context(request):
                            emit(chunk)
            raw_output = "".join(output).strip()
            inference_ms = (time.perf_counter() - started) * 1000
            record(
                request._metrics,
                inference_ms=inference_ms,
                ttft_ms=first_chunk_ms,
                ttft_source="first_text_chunk" if first_chunk_ms is not None else None,
            )
            count_output(request._metrics, getattr(model, "tokenize", None), raw_output)
            result = parse_response(raw_output, request)
            debug_json(
                _LOG,
                self.settings.debug_log,
                f"final_{kind}_response",
                {
                    "model": request.model,
                    "inference_ms": round(inference_ms, 1),
                    "ttft_ms": round(first_chunk_ms, 1) if first_chunk_ms is not None else None,
                    "input_tokens": request._metrics.get("input_tokens"),
                    "output_tokens": request._metrics.get("output_tokens"),
                    "response": result,
                },
                request_id=request._request_id,
            )
            if emit and has_tool_context(request):
                emit(result)
            return result
        except Exception as exc:
            debug_json(
                _LOG,
                self.settings.debug_log,
                f"final_{kind}_error",
                {
                    "model": request.model,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                request_id=request._request_id,
            )
            raise
        finally:
            model.clear_context()

    def transcribe(self, audio, language):
        """Transcribe normalized audio through the resident Whisper model.

        Args:
            audio (np.ndarray): Normalized 16 kHz mono float32 samples for Whisper.
            language (str | None): Language code; None uses the configured or detected language.

        Returns:
            str: Recognized speech text.

        Raises:
            BusyError: Whisper is disabled.
        """
        return transcribe_whisper(self.whisper, audio, language, self.settings.request_timeout)

    def select_tools(self, request):
        """Prepare HA context, deterministic routes and selected tool schemas.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

        Returns:
            ChatRequest: Request carrying compact context or a direct response.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        from .ha_pipeline import prepare_request

        return prepare_request(self, request)

    def retrieve_context(self, request):
        """Retrieve relevant entities and tools while retaining active call dependencies.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

        Returns:
            ChatRequest: Copy with compact messages and selected schemas.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        request_id = getattr(request, "_request_id", "-")
        entity_trace = {} if self.settings.debug_log else None
        debug_json(
            _LOG,
            self.settings.debug_log,
            "retrieval_input",
            {
                "model": request.model,
                "messages": request.messages,
                "tool_names": [
                    tool.get("function", {}).get("name") for tool in (request.tools or [])
                ],
            },
            request_id=request_id,
        )
        compact_messages, context_stats = compact_static_context(
            request.messages,
            encoder=self.minilm,
            embedding_cache=self._retrieval_embedding_cache,
            trace=entity_trace,
        )
        request = request.model_copy(update={"messages": compact_messages})
        if context_stats["system_prompts_compacted"]:
            _LOG.info(
                "MiniLM entity retrieval: %d -> %d entities; %d chars removed",
                context_stats["entities_before"],
                context_stats["entities_after"],
                context_stats["characters_removed"],
            )
        debug_json(
            _LOG,
            self.settings.debug_log,
            "entity_retrieval_trace",
            entity_trace or {},
            request_id=request_id,
        )
        debug_json(
            _LOG,
            self.settings.debug_log,
            "after_entity_retrieval",
            {
                "stats": context_stats,
                "messages": compact_messages,
            },
            request_id=request_id,
        )
        if not request.tools:
            return request
        source = request.tools
        if isinstance(request.tool_choice, dict):
            name = request.tool_choice["function"]["name"]
            source = [tool for tool in source if tool["function"]["name"] == name]
        tool_trace = {} if self.settings.debug_log else None
        selected, stats = retrieve_tools(
            request.messages,
            source,
            encoder=self.minilm,
            embedding_cache=self._retrieval_embedding_cache,
            trace=tool_trace,
        )
        required_names = set()
        for message in request.messages:
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    name = call.get("function", {}).get("name")
                    if isinstance(name, str):
                        required_names.add(name)
        if required_names:
            selected_names = {tool.get("function", {}).get("name") for tool in selected}
            required = [
                tool
                for tool in source
                if tool.get("function", {}).get("name") in required_names
                and tool.get("function", {}).get("name") not in selected_names
            ]
            selected = required + selected
            stats["tools_after"] = len(selected)
        _LOG.info(
            "MiniLM tool retrieval: %d -> %d tools; %d enum values removed",
            stats["tools_before"],
            stats["tools_after"],
            stats["enum_values_removed"],
        )
        debug_json(
            _LOG,
            self.settings.debug_log,
            "tool_retrieval_trace",
            tool_trace or {},
            request_id=request_id,
        )
        debug_json(
            _LOG,
            self.settings.debug_log,
            "after_tool_retrieval",
            {
                "stats": stats,
                "required_tool_names": sorted(required_names),
                "selected_tool_names": [tool.get("function", {}).get("name") for tool in selected],
                "selected_tools": selected,
                "messages": request.messages,
            },
            request_id=request_id,
        )
        return request.model_copy(update={"tools": selected})

    def close(self):
        # Release models before the device, including after partial startup.
        """Release resources owned by this service or native context.

        Returns:
            None: Closes native resources, connections or owner executors.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        for name in ("minilm", "whisper", "llm", "vlm", "device"):
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
