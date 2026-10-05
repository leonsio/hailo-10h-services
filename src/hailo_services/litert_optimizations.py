"""Low-latency LiteRT helpers for Home Assistant agent requests.

This module is installed once from :mod:`hailo_services.__init__`. It keeps the
main runtime implementation small while providing production optimizations and
diagnostics:

* Successful Home Assistant action results can be acknowledged without a second
  Gemma inference round.
* LiteRT conversations expose native benchmark information, which is logged
  together with wall-clock timings for prefill/decode diagnostics.
* In debug mode, the exact rendered prompt sent to Gemma is logged after the
  LiteRT chat template and tool declarations have been applied.
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
    """Return a deterministic acknowledgement for a completed HA action turn.

    The fast path is deliberately conservative. The current turn must end in
    tool results that exactly cover the preceding assistant tool calls. Every
    result must be a Home Assistant ``action_done`` payload with no failures and
    either at least one successful target or an explicit speech response.

    Data/query tools therefore continue through Gemma, as do failed or partial
    actions where model interpretation can still be useful.
    """
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
        failed = data.get("failed")
        if failed:
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

    # Prefer Home Assistant's own localized speech when available. Otherwise a
    # short generic acknowledgement is safer than regenerating a description of
    # an action that Home Assistant has already confirmed.
    text = " ".join(dict.fromkeys(speeches)) if speeches else t('litert_optimizations.122')
    return {
        "text": text,
        "tool_names": [pending[call["id"]] for call in calls],
        "successes": successes,
        "tool_results": [results[call["id"]] for call in calls],
    }


def _value(info, name, default=None):
    if isinstance(info, dict):
        return info.get(name, default)
    return getattr(info, name, default)


def _duration_ms(tokens, tokens_per_second):
    try:
        tokens = float(tokens)
        tokens_per_second = float(tokens_per_second)
    except (TypeError, ValueError):
        return None
    if tokens < 0 or tokens_per_second <= 0:
        return None
    return tokens / tokens_per_second * 1000.0


def _benchmark(conversation, *, enabled):
    if not enabled:
        return {"available": False, "disabled": True}
    getter = getattr(conversation, "get_benchmark_info", None)
    if not callable(getter):
        return {"available": False, "error": "get_benchmark_info unavailable"}
    try:
        info = getter()
    except Exception as exc:  # Native metrics are diagnostic, never request-fatal.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    prefill_tokens = _value(info, "last_prefill_token_count")
    prefill_tps = _value(info, "last_prefill_tokens_per_second")
    decode_tokens = _value(info, "last_decode_token_count")
    decode_tps = _value(info, "last_decode_tokens_per_second")
    ttft = _value(info, "time_to_first_token_in_second")
    init_time = _value(info, "init_time_in_second")
    return {
        "available": True,
        "init_ms": float(init_time) * 1000.0 if isinstance(init_time, (int, float)) else None,
        "time_to_first_token_ms": float(ttft) * 1000.0 if isinstance(ttft, (int, float)) else None,
        "prefill_tokens": prefill_tokens,
        "prefill_tokens_per_second": prefill_tps,
        "prefill_ms_estimate": _duration_ms(prefill_tokens, prefill_tps),
        "decode_tokens": decode_tokens,
        "decode_tokens_per_second": decode_tps,
        "decode_ms_estimate": _duration_ms(decode_tokens, decode_tps),
    }


def _log_timing(backend, conversation, *, wall_ms, create_call_ms, enter_ms, first_chunk_ms=None):
    benchmark = _benchmark(conversation, enabled=True)
    prefill_ms = benchmark.get("prefill_ms_estimate")
    decode_ms = benchmark.get("decode_ms_estimate")
    accounted = sum(value for value in (prefill_ms, decode_ms) if isinstance(value, (int, float)))
    payload = {
        "wall_inference_ms": wall_ms,
        "conversation_create_call_ms": create_call_ms,
        "conversation_enter_ms": enter_ms,
        "first_stream_chunk_ms": first_chunk_ms,
        "native": benchmark,
        "native_prefill_decode_ms": accounted if benchmark.get("available") else None,
        "wall_minus_native_ms": max(0.0, wall_ms - accounted) if benchmark.get("available") else None,
    }
    request_id = getattr(_REQUEST, "request_id", "-")
    metrics = getattr(_REQUEST, "metrics", None)
    if metrics is not None:
        record(metrics, inference_ms=wall_ms,
               conversation_create_ms=create_call_ms, conversation_enter_ms=enter_ms,
               ttft_ms=benchmark.get("time_to_first_token_ms"),
               prefill_tokens_per_second=benchmark.get("prefill_tokens_per_second"),
               decode_tokens_per_second=benchmark.get("decode_tokens_per_second"),
               prefill_ms_estimate=prefill_ms, decode_ms_estimate=decode_ms)
        if benchmark.get("time_to_first_token_ms") is not None:
            metrics["ttft_source"] = "native"
        elif first_chunk_ms is not None:
            record(metrics, ttft_ms=first_chunk_ms, ttft_source="first_text_chunk")
        for target, source in (("input_tokens", "prefill_tokens"), ("output_tokens", "decode_tokens")):
            value = benchmark.get(source)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                record(metrics, **{target: value, target + "_source": "native"})
    _LOG.info(
        "gemma_timing request_id=%s wall_ms=%.1f ttft_ms=%s prefill_tokens=%s "
        "prefill_tps=%s prefill_ms=%s decode_tokens=%s decode_tps=%s decode_ms=%s "
        "conversation_create_ms=%.1f conversation_enter_ms=%.1f",
        request_id,
        wall_ms,
        benchmark.get("time_to_first_token_ms"),
        benchmark.get("prefill_tokens"),
        benchmark.get("prefill_tokens_per_second"),
        benchmark.get("prefill_ms_estimate"),
        benchmark.get("decode_tokens"),
        benchmark.get("decode_tokens_per_second"),
        benchmark.get("decode_ms_estimate"),
        create_call_ms,
        enter_ms,
    )
    if getattr(backend, "debug_log", False):
        _LOG.debug(
            "event=gemma_timing request_id=%s json=%s",
            request_id,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
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
            self._conversation,
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
                    self._conversation,
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
    """Wrap an initialized LiteRT engine with per-conversation timing metrics."""
    engine = getattr(backend, "engine", None)
    if engine is None or getattr(engine, "_hailo_metrics_proxy", False):
        return
    backend.engine = _EngineProxy(engine, backend)


def _start_with_benchmark(self, original_start, *args, **kwargs):
    """Enable LiteRT native benchmark collection for API and WebGUI diagnostics.

    ``LiteRTLMBackend.start`` owns engine construction in :mod:`runtime`. Keep
    that implementation as the single source of truth and temporarily decorate
    the public ``litert_lm.Engine`` constructor so the native benchmark option
    is injected with a fallback for older bindings. Startup is serialized before request
    handling begins, and the constructor is restored immediately afterwards.
    """
    import litert_lm

    original_engine = litert_lm.Engine

    @wraps(original_engine)
    def engine_with_diagnostics(*engine_args, **engine_kwargs):
        engine_kwargs.setdefault("enable_benchmark", True)
        try:
            return original_engine(*engine_args, **engine_kwargs)
        except TypeError as exc:
            if "enable_benchmark" not in str(exc):
                raise
            engine_kwargs.pop("enable_benchmark", None)
            return original_engine(*engine_args, **engine_kwargs)

    litert_lm.Engine = engine_with_diagnostics
    try:
        result = original_start(self, *args, **kwargs)
    finally:
        litert_lm.Engine = original_engine
    if getattr(self, "debug_log", False):
        _LOG.info("LiteRT native benchmark diagnostics enabled")
    return result


def install():
    """Install the fast path and metric hooks on ``LiteRTLMBackend`` once."""
    from . import runtime

    cls = runtime.LiteRTLMBackend
    if getattr(cls, "_hailo_optimizations_installed", False):
        return

    original_start = cls.start
    original_chat = cls.chat

    @wraps(original_start)
    def start(self, *args, **kwargs):
        result = _start_with_benchmark(self, original_start, *args, **kwargs)
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
        _REQUEST.metrics = request._metrics
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
