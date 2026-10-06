"""Low-latency LiteRT helpers and request timing diagnostics.

This module is installed once from :mod:`hailo_services.__init__`. It keeps the
main runtime implementation small while providing production behavior only:

* Successful Home Assistant action results can be acknowledged without a second
  Gemma inference round.
* LiteRT conversations expose wall-clock inference and first-text-chunk timing.
* In debug mode, the exact rendered prompt sent to Gemma is logged after the
  LiteRT chat template and tool declarations have been applied.

No native LiteRT benchmark mode is enabled and no benchmark-only engine options
are injected into production startup.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from functools import wraps

from .i18n import t
from .metrics import record

_LOG = logging.getLogger(__name__)
_REQUEST = threading.local()


def _json_object(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        value = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _speech(result):
    speech = result.get("speech")
    if not isinstance(speech, dict):
        return None
    plain = speech.get("plain")
    if isinstance(plain, dict):
        value = plain.get("speech")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def successful_action_followup(request):
    """Return a deterministic acknowledgement for a completed HA action turn."""
    from .ha_pipeline import is_home_assistant_request

    if not is_home_assistant_request(request):
        return None
    messages = request.messages
    if not messages or messages[-1].get("role") != "tool":
        return None

    first_tool = len(messages) - 1
    while first_tool > 0 and messages[first_tool - 1].get("role") == "tool":
        first_tool -= 1
    assistant_index = first_tool - 1
    if assistant_index < 0:
        return None
    assistant = messages[assistant_index]
    calls = assistant.get("tool_calls") if assistant.get("role") == "assistant" else None
    if not isinstance(calls, list) or not calls:
        return None

    pending = {}
    for call in calls:
        if not isinstance(call, dict):
            return None
        identifier = call.get("id")
        function = call.get("function")
        if not isinstance(identifier, str) or not isinstance(function, dict):
            return None
        name = function.get("name")
        if not isinstance(name, str):
            return None
        pending[identifier] = name

    results = {}
    speeches = []
    successes = []
    for message in messages[first_tool:]:
        identifier = message.get("tool_call_id")
        if not isinstance(identifier, str) or identifier not in pending or identifier in results:
            return None
        payload = _json_object(message.get("content"))
        if payload is None or payload.get("response_type") != "action_done":
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        if data.get("failed"):
            return None
        success = data.get("success")
        spoken = _speech(payload)
        if not (isinstance(success, list) and success) and not spoken:
            return None
        results[identifier] = payload
        if spoken:
            speeches.append(spoken)
        if isinstance(success, list):
            successes.extend(item for item in success if isinstance(item, dict))

    if set(results) != set(pending):
        return None

    text = " ".join(dict.fromkeys(speeches)) if speeches else t("litert_optimizations.122")
    return {
        "text": text,
        "tool_names": [pending[call["id"]] for call in calls],
        "successes": successes,
        "tool_results": [results[call["id"]] for call in calls],
    }


def _log_timing(backend, *, wall_ms, create_call_ms, enter_ms, first_chunk_ms=None):
    request_id = getattr(_REQUEST, "request_id", "-")
    metrics = getattr(_REQUEST, "metrics", None)
    if metrics is not None:
        record(
            metrics,
            inference_ms=wall_ms,
            conversation_create_ms=create_call_ms,
            conversation_enter_ms=enter_ms,
        )
        if first_chunk_ms is not None:
            record(metrics, ttft_ms=first_chunk_ms, ttft_source="first_text_chunk")

    _LOG.info(
        "gemma_timing request_id=%s wall_ms=%.1f ttft_ms=%s "
        "conversation_create_ms=%.1f conversation_enter_ms=%.1f",
        request_id,
        wall_ms,
        first_chunk_ms,
        create_call_ms,
        enter_ms,
    )
    if getattr(backend, "debug_log", False):
        _LOG.debug(
            "event=gemma_timing request_id=%s json=%s",
            request_id,
            json.dumps(
                {
                    "wall_inference_ms": wall_ms,
                    "conversation_create_call_ms": create_call_ms,
                    "conversation_enter_ms": enter_ms,
                    "first_stream_chunk_ms": first_chunk_ms,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ),
        )


class _ConversationProxy:
    def __init__(self, conversation, backend, create_call_ms, enter_ms):
        self._conversation = conversation
        self._backend = backend
        self._create_call_ms = create_call_ms
        self._enter_ms = enter_ms

    def __getattr__(self, name):
        return getattr(self._conversation, name)

    def render_message_to_string(self, *args, **kwargs):
        rendered = self._conversation.render_message_to_string(*args, **kwargs)
        if getattr(self._backend, "debug_log", False):
            token_count = None
            try:
                token_count = len(self._backend.engine.tokenize(rendered))
            except Exception:
                pass
            source_message = args[0] if args else kwargs.get("message")
            _LOG.debug(
                "event=gemma_rendered_prompt request_id=%s json=%s",
                getattr(_REQUEST, "request_id", "-"),
                json.dumps(
                    {
                        "characters": len(rendered),
                        "raw_tokens": token_count,
                        "source_message": source_message,
                        "prompt": rendered,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
            )
        return rendered

    def send_message(self, *args, **kwargs):
        started = time.perf_counter()
        response = self._conversation.send_message(*args, **kwargs)
        wall_ms = (time.perf_counter() - started) * 1000.0
        _log_timing(
            self._backend,
            wall_ms=wall_ms,
            create_call_ms=self._create_call_ms,
            enter_ms=self._enter_ms,
        )
        return response

    def send_message_async(self, *args, **kwargs):
        started = time.perf_counter()
        iterator = self._conversation.send_message_async(*args, **kwargs)

        def stream():
            first_chunk_ms = None
            try:
                for chunk in iterator:
                    text = self._backend._chunk_text(chunk)
                    if first_chunk_ms is None and text:
                        first_chunk_ms = (time.perf_counter() - started) * 1000.0
                    yield chunk
            finally:
                wall_ms = (time.perf_counter() - started) * 1000.0
                _log_timing(
                    self._backend,
                    wall_ms=wall_ms,
                    create_call_ms=self._create_call_ms,
                    enter_ms=self._enter_ms,
                    first_chunk_ms=first_chunk_ms,
                )

        return stream()


class _ConversationContextProxy:
    def __init__(self, context, backend, create_call_ms):
        self._context = context
        self._backend = backend
        self._create_call_ms = create_call_ms
        self._conversation = None

    def __enter__(self):
        started = time.perf_counter()
        conversation = self._context.__enter__()
        enter_ms = (time.perf_counter() - started) * 1000.0
        self._conversation = _ConversationProxy(
            conversation, self._backend, self._create_call_ms, enter_ms
        )
        return self._conversation

    def __exit__(self, exc_type, exc, tb):
        return self._context.__exit__(exc_type, exc, tb)


class _EngineProxy:
    def __init__(self, engine, backend):
        self._engine = engine
        self._backend = backend
        self._hailo_metrics_proxy = True

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def create_conversation(self, *args, **kwargs):
        started = time.perf_counter()
        context = self._engine.create_conversation(*args, **kwargs)
        create_call_ms = (time.perf_counter() - started) * 1000.0
        return _ConversationContextProxy(context, self._backend, create_call_ms)


def instrument_engine(backend):
    """Wrap an initialized LiteRT engine with per-conversation wall timings."""
    engine = getattr(backend, "engine", None)
    if engine is None or getattr(engine, "_hailo_metrics_proxy", False):
        return
    backend.engine = _EngineProxy(engine, backend)


def install():
    """Install production fast paths and timing hooks on ``LiteRTLMBackend`` once."""
    from . import runtime

    cls = runtime.LiteRTLMBackend
    if getattr(cls, "_hailo_optimizations_installed", False):
        return

    original_start = cls.start
    original_chat = cls.chat

    @wraps(original_start)
    def start(self, *args, **kwargs):
        result = original_start(self, *args, **kwargs)
        instrument_engine(self)
        return result

    @wraps(original_chat)
    def chat(self, request, emit=None, cancelled=None, tools_prepared=False):
        request_id = getattr(request, "_request_id", "-")
        fast = successful_action_followup(request)
        if fast is not None:
            _LOG.info(
                "tool_followup_fast_path request_id=%s tools=%s successes=%d skipped_gemma=true",
                request_id,
                ",".join(fast["tool_names"]),
                len(fast["successes"]),
            )
            if getattr(self, "debug_log", False):
                _LOG.debug(
                    "event=tool_followup_fast_path request_id=%s json=%s",
                    request_id,
                    json.dumps(fast, ensure_ascii=False, separators=(",", ":"), default=str),
                )
            if emit:
                emit(fast["text"])
            return fast["text"]

        previous = getattr(_REQUEST, "request_id", None)
        previous_metrics = getattr(_REQUEST, "metrics", None)
        _REQUEST.request_id = request_id
        _REQUEST.metrics = getattr(request, "_metrics", None)
        try:
            return original_chat(self, request, emit, cancelled, tools_prepared)
        finally:
            _REQUEST.metrics = previous_metrics
            if previous is None:
                try:
                    del _REQUEST.request_id
                except AttributeError:
                    pass
            else:
                _REQUEST.request_id = previous

    cls.start = start
    cls.chat = chat
    cls._hailo_optimizations_installed = True
