"""Resident CPU LiteRT-LM backend for Gemma text generation and function calling."""

from __future__ import annotations

import inspect
import json
import logging
import time
from pathlib import Path
from threading import Event

from hailo_services.config import LLM_MODEL
from hailo_services.diagnostics.diagnostics import debug_json
from hailo_services.diagnostics.metrics import count_output, record
from hailo_services.interfaces import ChatEmitter, ChatResult
from hailo_services.schemas import ChatRequest
from hailo_services.shared.errors import LiteRTInferenceError
from hailo_services.shared.input_budget import InputBudgetError, history_candidates
from hailo_services.shared.tool_calling import (
    has_tool_context,
    native_messages,
    native_tools,
    response_message,
    selected_tools,
)
from hailo_services.shared.tool_retrieval import retrieve_tools

_LOG = logging.getLogger(__name__)
_INPUT_TOKEN_SAFETY_MARGIN = 256


class LiteRTLMBackend:
    """Resident LiteRT-LM Python Engine for text-only Gemma requests."""

    def __init__(
        self,
        model_path,
        max_num_tokens=16384,
        max_input_tokens=4096,
        debug_log=False,
        model_manager=None,
    ):
        """Initialize LiteRTLMBackend configuration and owned dependencies.

        Args:
            model_path (str | Path): Path to the resident LiteRT model artifact.
            max_num_tokens (int): Native LiteRT context capacity in tokens.
            max_input_tokens (int): Configured maximum input-token count before generation.
            debug_log (bool): Whether verbose request diagnostics should be logged.
            model_manager (ModelManager | None): Optional catalogue resolver for the configured model file.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.model_manager = model_manager
        self.model_path = str(Path(model_path).expanduser())
        self.max_num_tokens = max_num_tokens
        self.max_input_tokens = max_input_tokens
        self.debug_log = debug_log
        self.engine_context = None
        self.engine = None
        self.litert_lm = None

    def start(self):
        """Resolve the Gemma artifact and initialize the resident CPU LiteRT engine.

        Returns:
            None: Marks the service ready after successful initialization.

        Raises:
            FileNotFoundError: The configured LiteRT model artifact does not exist.
            OSError: A required model artifact cannot be read or downloaded.
            RuntimeError: Native model initialization fails.
        """
        path = Path(self.model_path)
        if self.model_manager is not None:
            path = self.model_manager.resolve(LLM_MODEL, "litert", path)
        if not path.is_file():
            raise FileNotFoundError(f"LiteRT-LM model not found: {path}")
        import litert_lm

        self.litert_lm = litert_lm
        self.engine_context = litert_lm.Engine(
            str(path), backend=litert_lm.Backend.CPU(), max_num_tokens=self.max_num_tokens
        )
        self.engine = self.engine_context.__enter__()
        from hailo_services.diagnostics.diagnostics_litert import instrument_engine

        instrument_engine(self)
        _LOG.info("Loaded LiteRT-LM model %s on CPU; max_num_tokens=%d", path, self.max_num_tokens)

    @staticmethod
    def _messages(request):
        """Translate OpenAI history and add required tool-call instructions.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.

        Returns:
            list[dict[str, Any]]: Native LiteRT conversation messages.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
        """Extract the final LiteRT message without losing tool-result structure.

        Args:
            messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

        Returns:
            str | dict[str, Any]: Final textual prompt or native tool message.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return messages[-1] if messages[-1]["role"] == "tool" else messages[-1]["content"]

    def _limit_input(self, request, tool_options):
        """Count the rendered LiteRT prompt and remove complete old turns to fit.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
            tool_options (dict[str, Any]): Native LiteRT tool declarations and constrained-decoding settings.

        Returns:
            ChatRequest: Request whose required context fits the native token budget.

        Raises:
            InputBudgetError: System, tools and the active turn exceed the effective input limit.
            ValueError: max_input_tokens requires LiteRT-LM Engine.tokenize; upgrade litert-lm.
        """
        if request.max_input_tokens is None:
            return request
        if not callable(getattr(self.engine, "tokenize", None)):
            raise ValueError(
                "max_input_tokens requires LiteRT-LM Engine.tokenize; upgrade litert-lm"
            )
        # This limit applies only after tool retrieval, immediately before Gemma.
        # The incoming OpenAI request may be much larger because MiniLM processes
        # its tool catalogue first. Native prefill still must fit the configured
        # Gemma context, with the requested output budget reserved.
        limit = min(
            request.max_input_tokens,
            self.max_input_tokens,
            self.max_num_tokens - request.max_tokens - 1,
        )
        if limit < 1:
            raise ValueError("The configured LiteRT context leaves no room for input and output")
        # Validate all original call/result dependencies before dropping history.
        self._messages(request)
        for candidate_index, candidate in enumerate(history_candidates(request.messages)):
            trimmed = request.model_copy(update={"messages": candidate})
            messages = self._messages(trimmed)
            # A fresh native conversation supplies the same tools and prefix
            # messages without running prefill or decoding. Release it before
            # creating the next probe or the inference conversation.
            with self.engine.create_conversation(
                messages=messages[:-1],
                max_output_tokens=request.max_tokens,
                **tool_options,
            ) as probe:
                render = getattr(probe, "render_message_to_string", None)
                if not callable(render):
                    raise ValueError(
                        "max_input_tokens requires LiteRT-LM Conversation.render_message_to_string; "
                        "upgrade litert-lm"
                    )
                # LiteRT's Conversation already owns messages[:-1] and tool
                # declarations. Rendering only the final message yields the
                # complete prompt for Gemma. Rendering every message separately
                # repeats the whole conversation prefix and tools per message.
                rendered = render(messages[-1])
                rendered_tools = (
                    json.dumps(selected_tools(request), ensure_ascii=False, separators=(",", ":"))
                    if has_tool_context(request)
                    else ""
                )
                raw_tokens = len(self.engine.tokenize(rendered))
                # Keep conservative headroom for native bookkeeping/special tokens.
                # Tools are already present in rendered; do not count their JSON
                # payload a second time.
                template_margin = _INPUT_TOKEN_SAFETY_MARGIN
                tool_margin = 8 * len(selected_tools(request))
                tokens = raw_tokens + template_margin + tool_margin
            debug_json(
                _LOG,
                self.debug_log,
                "input_budget_candidate",
                {
                    "candidate_index": candidate_index,
                    "messages_before": len(request.messages),
                    "messages_after": len(candidate),
                    "removed_messages": len(request.messages) - len(candidate),
                    "raw_tokens": raw_tokens,
                    "template_margin": template_margin,
                    "tool_margin": tool_margin,
                    "input_tokens": tokens,
                    "input_limit": limit,
                    "output_reserved": request.max_tokens,
                    "accepted": tokens <= limit,
                    "messages": messages,
                    "tools": selected_tools(request),
                    "render_strategy": "final_message_full_conversation",
                    "rendered_prompt": rendered,
                    "rendered_messages": rendered,
                    "rendered_tools_reference": rendered_tools,
                },
                request_id=getattr(request, "_request_id", "-"),
            )
            if tokens <= limit:
                record(
                    trimmed._metrics,
                    input_tokens=raw_tokens,
                    input_tokens_source="tokenizer",
                    input_budget_tokens=tokens,
                    removed_messages=len(request.messages) - len(candidate),
                )
                _LOG.info(
                    "input_budget model=%s input_tokens=%d limit=%d max_input_tokens=%d "
                    "removed_messages=%d output_reserved=%d",
                    request.model,
                    tokens,
                    limit,
                    request.max_input_tokens,
                    len(request.messages) - len(candidate),
                    request.max_tokens,
                )
                debug_json(
                    _LOG,
                    self.debug_log,
                    "input_budget_selected",
                    {
                        "input_tokens": tokens,
                        "input_limit": limit,
                        "removed_messages": len(request.messages) - len(candidate),
                        "request": trimmed.model_dump(mode="json", exclude_none=False),
                    },
                    request_id=getattr(request, "_request_id", "-"),
                )
                return trimmed
        debug_json(
            _LOG,
            self.debug_log,
            "input_budget_failed",
            {
                "input_tokens": tokens,
                "input_limit": limit,
                "request": request.model_dump(mode="json", exclude_none=False),
            },
            request_id=getattr(request, "_request_id", "-"),
        )
        raise InputBudgetError(
            tokens, limit, request.max_input_tokens, self.max_num_tokens, request.max_tokens
        )

    @staticmethod
    def _chunk_text(chunk):
        """Extract text from a LiteRT response or streaming chunk.

        Args:
            chunk (Any): Native LiteRT response or streamed chunk.

        Returns:
            str: Text content, or an empty string for non-text output.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if hasattr(chunk, "to_json"):
            chunk = chunk.to_json()
        if isinstance(chunk, dict) and isinstance(chunk.get("content"), str):
            return chunk["content"]
        return (
            "".join(
                item.get("text", "")
                for item in chunk.get("content", [])
                if isinstance(item, dict) and item.get("type") == "text"
            )
            if isinstance(chunk, dict)
            else ""
        )

    def chat(
        self,
        request: ChatRequest,
        emit: ChatEmitter | None = None,
        cancelled: Event | None = None,
        tools_prepared: bool = False,
    ) -> ChatResult:
        """Generate a completion or stream model output for a chat request.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
            emit (ChatEmitter | None): Optional callback receiving text chunks or validated assistant messages.
            cancelled (Event | None): Optional cancellation event checked between generated chunks.
            tools_prepared (bool): Whether context/tool preparation has already run for this request.

        Returns:
            ChatResult: Final text or a validated assistant tool-call message.

        Raises:
            ValueError: Input or generated function calls are invalid.
            InputBudgetError: Required context cannot fit the configured token budget.
            LiteRTInferenceError: Native LiteRT inference fails.
        """
        from hailo_services.assistants.ha.ha_action_verification import (
            _VERIFY_SETTLE_SECONDS,
            action_verification_response,
        )
        from hailo_services.chat.chat_litert import run_chat
        from hailo_services.shared.i18n import using_language

        if not getattr(request, "_ha_assist", False):
            return run_chat(self, request, self._generate, emit, cancelled, True)
        with using_language(getattr(request, "_response_language", request.language or "de")):
            # Preserve established priority: weather, verification, state, action.
            direct = getattr(request, "_direct_weather_response", None)
            if direct is None:
                decision = action_verification_response(request)
                if decision is not None:
                    if decision["kind"] == "verify":
                        time.sleep(_VERIFY_SETTLE_SECONDS)
                    direct = decision["response"]
            if direct is None:
                direct = getattr(request, "_direct_ha_state_response", None)
            if direct is None:
                direct = getattr(request, "_direct_ha_response", None)
            if direct is not None:
                if emit:
                    emit(direct)
                return direct
            return run_chat(self, request, self._generate, emit, cancelled, True)

    def _generate(self, request, emit=None, cancelled=None, tools_prepared=False):
        """Run LiteRT generation and translate native runtime failures.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
            emit (ChatEmitter | None): Optional callback receiving text chunks or validated assistant messages.
            cancelled (threading.Event | None): Optional cancellation event checked between generated chunks.
            tools_prepared (bool): Whether context/tool preparation has already run for this request.

        Returns:
            ChatResult: Final model text or validated tool-call message.

        Raises:
            LiteRTInferenceError: Native LiteRT reports a runtime failure during generation.
        """
        try:
            return self._chat(request, emit, cancelled, tools_prepared)
        except RuntimeError as exc:
            _LOG.exception(
                "LiteRT inference failed; model=%s context_tokens=%d",
                request.model,
                self.max_num_tokens,
            )
            raise LiteRTInferenceError(
                f"LiteRT-LM inference failed (configured context: {self.max_num_tokens} tokens). "
                "Check the preceding native log for the cause. If the input exceeds the context, "
                "increase HAILO_LITERT_MAX_NUM_TOKENS and restart, or reduce the Home Assistant "
                "prompt/history. Larger contexts require more RAM."
            ) from exc

    def _chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        """Render, budget and generate a LiteRT conversation on CPU.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
            emit (ChatEmitter | None): Optional callback receiving text chunks or validated assistant messages.
            cancelled (threading.Event | None): Optional cancellation event checked between generated chunks.
            tools_prepared (bool): Whether context/tool preparation has already run for this request.

        Returns:
            ChatResult: Final model text or validated tool-call message.

        Raises:
            RuntimeError: LiteRT-LM is not ready.
            ValueError: Installed LiteRT-LM lacks Tool support; upgrade litert-lm.
        """
        if self.engine is None:
            raise RuntimeError("LiteRT-LM is not ready")
        if not has_tool_context(request):
            from hailo_services.shared.i18n import detect_language, t, using_language
            from hailo_services.shared.tool_retrieval import latest_user_text

            with using_language(
                request.language or detect_language(latest_user_text(request.messages), "de")
            ):
                # Explicit user/system requests for detail override the default.
                request = request.model_copy(
                    update={
                        "messages": [
                            {"role": "system", "content": t("gemma.concise")},
                            *request.messages,
                        ]
                    }
                )
        if request.max_input_tokens is None:
            request = request.model_copy(update={"max_input_tokens": self.max_input_tokens})
        if (
            getattr(request, "_ha_assist", False)
            and not tools_prepared
            and request.tools
            and not any(
                message.get("role") == "tool" or message.get("tool_calls")
                for message in request.messages
            )
        ):
            source_tools = request.tools
            if isinstance(request.tool_choice, dict):
                forced_name = request.tool_choice["function"]["name"]
                source_tools = [
                    tool for tool in request.tools if tool["function"]["name"] == forced_name
                ]
            compact_tools, stats = retrieve_tools(request.messages, source_tools)
            if stats["tools_after"] < stats["tools_before"] or stats["enum_values_removed"]:
                request = request.model_copy(update={"tools": compact_tools})
                _LOG.info(
                    "tool_retrieval model=%s tools=%d->%d enum_values_removed=%d",
                    request.model,
                    stats["tools_before"],
                    stats["tools_after"],
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
            # The Python API varies by LiteRT version. Enable native constrained
            # decoding only when explicitly exposed, never assume **kwargs means support.
            engine = getattr(self.engine, "_engine", self.engine)
            try:
                supported = (
                    "enable_constrained_decoding"
                    in inspect.signature(engine.create_conversation).parameters
                )
            except (ValueError, TypeError):
                supported = False
            if supported:
                tool_options["enable_constrained_decoding"] = True
            request._metrics["constrained_decoding"] = {
                "enabled": supported,
                "validation": "schema_and_ha_target",
            }
        debug_json(
            _LOG,
            self.debug_log,
            "before_input_budget",
            request.model_dump(mode="json", exclude_none=False),
            request_id=getattr(request, "_request_id", "-"),
        )
        request = self._limit_input(request, tool_options)
        messages = self._messages(request)
        debug_json(
            _LOG,
            self.debug_log,
            "final_gemma_request",
            {
                "request": request.model_dump(mode="json", exclude_none=False),
                "native_messages": messages,
                "selected_tools": selected_tools(request),
                "max_num_tokens": self.max_num_tokens,
                "max_input_tokens": request.max_input_tokens,
                "max_output_tokens": request.max_tokens,
            },
            request_id=getattr(request, "_request_id", "-"),
        )
        with self.engine.create_conversation(
            messages=messages[:-1],
            sampler_config=self.litert_lm.SamplerConfig(
                temperature=request.temperature,
                seed=request.seed,
                **({"top_p": request.top_p} if request.top_p is not None else {}),
            ),
            max_output_tokens=request.max_tokens,
            **tool_options,
        ) as conversation:
            prompt = self._prompt(messages)
            on_inference = getattr(request, "_on_inference", None)
            if on_inference is not None:
                on_inference(getattr(request, "_response_language", "de"))
            if emit is None or has_tool_context(request):
                response = conversation.send_message(prompt, max_output_tokens=request.max_tokens)
                count_output(
                    request._metrics,
                    getattr(self.engine, "tokenize", None),
                    self._chunk_text(response),
                )
                result = response_message(response, request, self._chunk_text(response).strip())
                debug_json(
                    _LOG,
                    self.debug_log,
                    "gemma_response",
                    {
                        "prompt": prompt,
                        "raw_response": (
                            response.to_json() if hasattr(response, "to_json") else response
                        ),
                        "parsed_response": result,
                    },
                    request_id=getattr(request, "_request_id", "-"),
                )
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
            result = "".join(output).strip()
            count_output(request._metrics, getattr(self.engine, "tokenize", None), result)
            debug_json(
                _LOG,
                self.debug_log,
                "gemma_stream_response",
                {"prompt": prompt, "response": result, "chunks": output},
                request_id=getattr(request, "_request_id", "-"),
            )
            return result

    def close(self):
        """Release resources owned by this service or native context.

        Returns:
            None: Closes native resources, connections or owner executors.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if self.engine_context is not None:
            try:
                self.engine_context.__exit__(None, None, None)
            finally:
                self.engine = self.engine_context = self.litert_lm = None
