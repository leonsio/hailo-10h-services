import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from .config import LLM_MODEL, STT_MODEL, VLM_MODEL, Settings
from .media import audio_file, audio_metadata, decode_base64
from .protocols import MQTTBridge, WyomingServer, dispatch
from .runtime import BusyError, Runtime
from .schemas import ChatRequest, TranscribeRequest
from .tool_calling import has_tool_context

_LOG = logging.getLogger(__name__)
_WEB = Path(__file__).with_name("web")
_WEB_FILES = {
    "/": "index.html",
    "/ui/app.js": "app.js",
    "/ui/style.css": "style.css",
    "/ui/recorder-worklet.js": "recorder-worklet.js",
}


def _debug(settings, message, *args):
    if settings.debug_log:
        _LOG.debug(message, *args)


class AccessAndSizeLimit:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex[:12]
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        headers = dict(scope.get("headers", []))
        path = scope.get("path", "-")
        method = scope.get("method", "WEBSOCKET")
        peer = scope.get("client") or ("unknown", 0)
        is_websocket = scope["type"] == "websocket"
        protocol = (
            "websocket" if is_websocket else
            "mcp" if path == "/mcp" or path.startswith("/mcp/") else "http"
        )
        transport = "websocket" if is_websocket else "http"
        _debug(self.settings,
               "protocol=%s transport=%s event=request_start request_id=%s method=%s path=%s peer=%s content_type=%s content_length=%s",
               protocol, transport, request_id,
               method, path, peer[0], headers.get(b"content-type", b"-").decode("latin1"),
               headers.get(b"content-length", b"-").decode("latin1"))
        response_status = "accepted"

        async def debug_send(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "websocket.close":
                response_status = "closed:" + str(message.get("code", ""))
            await send(message)

        async def call_app(current_scope, current_receive):
            try:
                await self.app(current_scope, current_receive, debug_send)
            finally:
                _debug(self.settings,
                       "protocol=%s transport=%s event=request_end request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
                       protocol, transport, request_id, method, path, response_status,
                       (time.perf_counter() - started) * 1000)
        expected = f"Bearer {self.settings.api_key}".encode()
        public = scope["path"] == "/health" or (
            scope["type"] == "http"
            and scope.get("method") in {"GET", "HEAD"}
            and scope["path"] in {*_WEB_FILES, "/ui/config"}
        )
        if self.settings.api_key and not public:
            if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await JSONResponse({"error": "Unauthorized"}, status_code=401)(
                        scope, receive, debug_send
                    )
                return
        if scope["type"] == "websocket":
            return await call_app(scope, receive)
        # Buffer bounded bodies before handing them to multipart/MCP/JSON parsers.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.settings.max_body:
                await JSONResponse({"error": "Request body too large"}, status_code=413)(
                    scope, receive, debug_send
                )
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await call_app(scope, bounded_receive)


def completion(text, identifier, created, model):
    message = text if isinstance(text, dict) else {"role": "assistant", "content": text}
    return {
        "id": identifier,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "message": message,
             "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}
        ],
    }


def create_app(settings=None, backend=None, litert_backend=None):
    settings = settings or Settings.from_env()
    runtime = Runtime(settings, backend, litert_backend)
    wyoming = WyomingServer(runtime, settings)
    mqtt = MQTTBridge(runtime, settings)
    mcp = MCPServer("Hailo-10H", version="0.1.0")

    @mcp.tool()
    async def analyze_image(image_base64: str, prompt: str, max_tokens: int = 256) -> str:
        """Analyze a base64-encoded camera snapshot using Qwen2-VL-2B-Instruct."""
        request = ChatRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        return await runtime.chat(request)

    @mcp.tool()
    async def transcribe_audio(audio_base64: str, language: str | None = None) -> str:
        """Transcribe a base64 WAV/FLAC/OGG recording with multilingual Whisper Base."""
        request = TranscribeRequest(audio_base64=audio_base64, language=language)
        data = decode_base64(request.audio_base64, settings.max_body)
        _debug(settings,
               "protocol=mcp operation=whisper_transcribe request_id=%s model=%s language=%s audio=%s bytes=%d",
               uuid.uuid4().hex[:12], STT_MODEL, request.language or settings.language,
               audio_metadata(data), len(data))
        return await runtime.transcribe(
            audio_file(data, settings.max_audio_seconds), request.language
        )

    @mcp.tool()
    async def chat_text(prompt: str, max_tokens: int = 256) -> str:
        """Ask Qwen2-VL a text question. Does not execute Home Assistant actions."""
        return await runtime.chat(
            ChatRequest(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        )

    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        max_request_body_size=settings.max_body,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[host.strip() for host in settings.mcp_hosts.split(",") if host.strip()],
        ),
    )

    @asynccontextmanager
    async def lifespan(app):
        mqtt_task = None
        try:
            await runtime.start()  # Failure prevents all listeners from becoming ready.
            if settings.wyoming_port:
                await wyoming.start()
            if settings.mqtt_host:
                mqtt_task = asyncio.create_task(mqtt.run(), name="mqtt-bridge")
            async with mcp.session_manager.run():
                yield
        finally:
            await wyoming.close()
            if mqtt_task:
                mqtt_task.cancel()
                with suppress(asyncio.CancelledError):
                    await mqtt_task
            await runtime.close()

    app = FastAPI(title="Hailo-10H Services", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.mqtt = mqtt
    app.add_middleware(AccessAndSizeLimit, settings=settings)

    async def web_file(request):
        return FileResponse(
            _WEB / _WEB_FILES[request.url.path],
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data: blob:; media-src 'self' blob:; "
                    "connect-src 'self'; worker-src 'self'; object-src 'none'; "
                    "base-uri 'none'; frame-ancestors 'none'"
                ),
            },
        )

    for path in _WEB_FILES:
        app.add_route(path, web_file, methods=["GET", "HEAD"], include_in_schema=False)

    @app.get("/ui/config", include_in_schema=False)
    async def web_config():
        return {
            "auth_required": bool(settings.api_key),
            "vlm_model": VLM_MODEL,
            "llm_model": LLM_MODEL,
            "chat_models": [VLM_MODEL] + ([LLM_MODEL] if runtime.litert_ready else []),
            "whisper_model": STT_MODEL,
            "language": settings.language,
            "max_body": settings.max_body,
            "max_audio_seconds": settings.max_audio_seconds,
            "recording_seconds": min(30, settings.max_audio_seconds),
        }

    @app.exception_handler(BusyError)
    async def busy_handler(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=503, headers={"Retry-After": "5"})

    @app.exception_handler(ValueError)
    async def value_handler(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(asyncio.TimeoutError)
    async def timeout_handler(request, exc):
        return JSONResponse(
            {"error": "Inference timed out; native work may still be completing"}, status_code=504
        )

    @app.get("/health")
    async def health():
        return JSONResponse(
            {**runtime.status(), "mqtt_connected": mqtt.connected},
            status_code=200 if runtime.ready else 503,
        )

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": "hailo"}
                for model in (VLM_MODEL, STT_MODEL)
            ] + ([{"id": LLM_MODEL, "object": "model", "owned_by": "litert-lm"}]
                 if runtime.litert_ready else []),
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest, http_request: Request):
        identifier, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        image_count = sum(
            part.get("type") == "image_url"
            for message in request.messages
            for part in (message["content"] if isinstance(message.get("content"), list) else [])
        )
        _debug(settings,
               "protocol=http operation=chat_completion request_id=%s model=%s messages=%d images=%d max_tokens=%d stream=%s",
               http_request.scope.get("state", {}).get("request_id", "-"),
               request.model, len(request.messages), image_count,
               request.max_tokens, request.stream)
        if not request.stream:
            return completion(await runtime.chat(request), identifier, created, request.model)

        async def events():
            def event(delta, finish=None):
                return (
                    "data: "
                    + json.dumps(
                        {
                            "id": identifier,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                        }
                    )
                    + "\n\n"
                )

            try:
                yield event({"role": "assistant", "content": ""})
                if has_tool_context(request):
                    # Buffer native tool output so invalid/incomplete calls are never streamed
                    # as actions or spoken as text by the voice assistant.
                    result = await runtime.chat(request)
                    if isinstance(result, dict):
                        if result.get("content"):
                            yield event({"content": result["content"]})
                        yield event({"tool_calls": [
                            {"index": index, **call}
                            for index, call in enumerate(result["tool_calls"])
                        ]})
                        yield event({}, "tool_calls")
                    else:
                        yield event({"content": result})
                        yield event({}, "stop")
                else:
                    async for chunk in runtime.stream(request):
                        yield event({"content": chunk})
                    yield event({}, "stop")
            except Exception as exc:
                _LOG.exception("Streaming inference failed")
                yield (
                    "data: "
                    + json.dumps({"error": {"message": str(exc), "type": "inference_error"}})
                    + "\n\n"
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = STT_MODEL,
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
    ):
        if model not in {STT_MODEL, "Whisper-Base", "whisper-1"}:
            raise HTTPException(400, "Unknown transcription model")
        if response_format not in {"json", "text"}:
            raise HTTPException(400, "response_format must be json or text")
        if language:
            TranscribeRequest(audio_base64="", language=language)
        try:
            data = await file.read(settings.max_body + 1)
        finally:
            await file.close()
        if len(data) > settings.max_body:
            raise HTTPException(413, "Audio upload too large")
        metadata = audio_metadata(data)
        _debug(settings,
               "protocol=http operation=whisper_transcribe request_id=%s model=%s language=%s audio=%s bytes=%d",
               request.scope.get("state", {}).get("request_id", "-"), model,
               language or settings.language, metadata, len(data))
        text = await runtime.transcribe(audio_file(data, settings.max_audio_seconds), language)
        return PlainTextResponse(text) if response_format == "text" else {"text": text}

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                if len(raw.encode()) > settings.max_body:
                    await ws.close(code=1009)
                    return
                identifier = None
                operation = "unknown"
                started = time.perf_counter()
                try:
                    envelope = json.loads(raw)
                    identifier = envelope.get("id")
                    if not isinstance(identifier, str) or len(identifier) > 64:
                        raise ValueError("Provide a string id of at most 64 characters")
                    operation = str(envelope.get("op", "unknown"))
                    _debug(settings,
                           "protocol=websocket event=request_start request_id=%s operation=%s",
                           identifier, operation)
                    result = await dispatch(
                        runtime, settings, envelope["op"], envelope.get("payload", {})
                    )
                    await ws.send_json({"id": identifier, "ok": True, "result": result})
                    _debug(settings,
                           "protocol=websocket event=request_end request_id=%s operation=%s duration_ms=%.1f ok=true",
                           identifier, operation, (time.perf_counter() - started) * 1000)
                except Exception as exc:
                    _debug(settings,
                           "protocol=websocket event=request_error request_id=%s operation=%s duration_ms=%.1f error_type=%s",
                           identifier or "-", operation, (time.perf_counter() - started) * 1000,
                           type(exc).__name__)
                    await ws.send_json({"id": identifier, "ok": False, "error": str(exc)})
        except WebSocketDisconnect:
            pass

    app.mount("/mcp", mcp_app)
    return app
