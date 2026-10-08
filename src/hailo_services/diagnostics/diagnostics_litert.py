"""Native LiteRT conversation proxies and request-scoped timing diagnostics."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager

from hailo_services.diagnostics.metrics import record

_LOG = logging.getLogger(__name__)
_REQUEST = threading.local()


def _log_timing(backend, *, wall_ms, create_call_ms, enter_ms, first_chunk_ms=None):
    """Record measured LiteRT latency and first-text-chunk timing.

    Args:
        backend (ChatBackend): Resident backend used for generation or context preparation.
        wall_ms (float): Measured inference wall-clock duration in milliseconds.
        create_call_ms (float): Native conversation factory duration in milliseconds.
        enter_ms (float): Conversation context initialization duration in milliseconds.
        first_chunk_ms (float | None): Time to first non-empty text chunk, or None when unavailable.

    Returns:
        None: Updates request metrics and logs timing data.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
    """Forward native conversation operations and measure prompt/generation timing."""

    def __init__(self, conversation, backend, create_call_ms, enter_ms):
        """Initialize _ConversationProxy configuration and owned dependencies.

        Args:
            conversation (Any): Native LiteRT conversation wrapped for diagnostics.
            backend (ChatBackend): Resident backend used for generation or context preparation.
            create_call_ms (float): Native conversation factory duration in milliseconds.
            enter_ms (float): Conversation context initialization duration in milliseconds.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self._conversation = conversation
        self._backend = backend
        self._create_call_ms = create_call_ms
        self._enter_ms = enter_ms

    def __getattr__(self, name):
        """Forward attributes to the wrapped native LiteRT object.

        Args:
            name (str): Function, attribute, device or model identifier.

        Returns:
            Any: Attribute from the underlying object.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return getattr(self._conversation, name)

    def render_message_to_string(self, *args, **kwargs):
        """Render the native conversation prompt and optionally log its token count.

        Args:
            *args (Any): Positional arguments forwarded to the native operation.
            **kwargs (Any): Keyword arguments forwarded to the native operation.

        Returns:
            str: Native rendered prompt.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
        """Run synchronous native generation while recording wall-clock latency.

        Args:
            *args (Any): Positional arguments forwarded to the native operation.
            **kwargs (Any): Keyword arguments forwarded to the native operation.

        Returns:
            Any: Native LiteRT response.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
        """Wrap native chunk iteration with latency and first-chunk measurements.

        Args:
            *args (Any): Positional arguments forwarded to the native operation.
            **kwargs (Any): Keyword arguments forwarded to the native operation.

        Returns:
            Iterator[Any]: Iterator yielding original native chunks.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        started = time.perf_counter()
        iterator = self._conversation.send_message_async(*args, **kwargs)

        def stream():
            """Yield output chunks and record latency when iteration finishes.

            Yields:
                ChatResult: Text chunks or validated assistant messages.

            Notes:
                No application-specific exceptions are raised for valid inputs.
            """
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
    """Measure context initialization and attach conversation diagnostics."""

    def __init__(self, context, backend, create_call_ms):
        """Initialize _ConversationContextProxy configuration and owned dependencies.

        Args:
            context (Any): Native context manager or total model context-token capacity.
            backend (ChatBackend): Resident backend used for generation or context preparation.
            create_call_ms (float): Native conversation factory duration in milliseconds.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self._context = context
        self._backend = backend
        self._create_call_ms = create_call_ms
        self._conversation = None

    def __enter__(self):
        """Enter the native conversation context and attach timing diagnostics.

        Returns:
            _ConversationProxy: Wrapped initialized native conversation.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        started = time.perf_counter()
        conversation = self._context.__enter__()
        enter_ms = (time.perf_counter() - started) * 1000.0
        self._conversation = _ConversationProxy(
            conversation, self._backend, self._create_call_ms, enter_ms
        )
        return self._conversation

    def __exit__(self, exc_type, exc, tb):
        """Release the wrapped native context with the current exception details.

        Args:
            exc_type (type[BaseException] | None): Active exception type, or None on normal exit.
            exc (BaseException | None): Exception being translated or passed to context cleanup.
            tb (TracebackType | None): Active exception traceback, or None.

        Returns:
            bool | None: Native context exception-suppression result.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return self._context.__exit__(exc_type, exc, tb)


class _EngineProxy:
    """Forward native engine operations while instrumenting conversation creation."""

    def __init__(self, engine, backend):
        """Initialize _EngineProxy configuration and owned dependencies.

        Args:
            engine (Any): Initialized native LiteRT engine.
            backend (ChatBackend): Resident backend used for generation or context preparation.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self._engine = engine
        self._backend = backend
        self._hailo_metrics_proxy = True

    def __getattr__(self, name):
        """Forward attributes to the wrapped native LiteRT object.

        Args:
            name (str): Function, attribute, device or model identifier.

        Returns:
            Any: Attribute from the underlying object.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return getattr(self._engine, name)

    def create_conversation(self, *args, **kwargs):
        """Wrap native conversations with request timing diagnostics.

        Args:
            *args (Any): Positional arguments forwarded to the native operation.
            **kwargs (Any): Keyword arguments forwarded to the native operation.

        Returns:
            _ConversationContextProxy: Context manager forwarding to the native conversation.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        started = time.perf_counter()
        context = self._engine.create_conversation(*args, **kwargs)
        create_call_ms = (time.perf_counter() - started) * 1000.0
        return _ConversationContextProxy(context, self._backend, create_call_ms)


def instrument_engine(backend):
    """Wrap an initialized LiteRT engine with per-conversation wall timings.

    Args:
        backend (ChatBackend): Resident backend used for generation or context preparation.

    Returns:
        None: Wraps the engine once; updates backend.engine in place.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    engine = getattr(backend, "engine", None)
    if engine is None or getattr(engine, "_hailo_metrics_proxy", False):
        return
    backend.engine = _EngineProxy(engine, backend)


@contextmanager
def request_diagnostics(request):
    """Scope native timing records to the active owner-thread request.

    Args:
        request (ChatRequest): Request carrying correlation ID and mutable metrics.

    Yields:
        None: Runs generation with request-local timing correlation.

    Notes:
        Previous thread-local values are restored even when generation fails.
        Exceptions from the context body propagate unchanged.
    """
    previous = getattr(_REQUEST, "request_id", None)
    previous_metrics = getattr(_REQUEST, "metrics", None)
    _REQUEST.request_id = request._request_id
    _REQUEST.metrics = request._metrics
    try:
        yield
    finally:
        _REQUEST.metrics = previous_metrics
        if previous is None:
            try:
                del _REQUEST.request_id
            except AttributeError:
                pass
        else:
            _REQUEST.request_id = previous
