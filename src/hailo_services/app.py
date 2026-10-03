import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from .config import STT_MODEL, VLM_MODEL, Settings
from .media import audio_file, decode_base64
from .protocols import MQTTBridge, WyomingServer, dispatch
from .runtime import BusyError, Runtime
from .schemas import ChatRequest, TranscribeRequest

_LOG = logging.getLogger(__name__)


class AccessAndSizeLimit:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        expected = f"Bearer {self.settings.api_key}".encode()
        if self.settings.api_key and scope["path"] != "/health":
            if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await JSONResponse({"error": "Unauthorized"}, status_code=401)(
                        scope, receive, send
                    )
                return
        if scope["type"] == "websocket":
            return await self.app(scope, receive, send)
        # Buffer bounded bodies before handing them to multipart/MCP/JSON parsers.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.settings.max_body:
                await JSONResponse({"error": "Request body too large"}, status_code=413)(
                    scope, receive, send
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

        await self.app(scope, bounded_receive, send)


def completion(text, identifier, created):
    return {
        "id": identifier,
        "object": "chat.completion",
        "created": created,
        "model": VLM_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }


def create_app(settings=None, backend=None):
    settings = settings or Settings.from_env()
    runtime = Runtime(settings, backend)
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

    @app.exception_handler(BusyError)
    async def busy_handler(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=503, headers={"Retry-After": "5"})

    @app.exception_handler(ValueError)
    async def value_handler(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(TimeoutError)
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
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest):
        identifier, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        if not request.stream:
            return completion(await runtime.chat(request), identifier, created)

        async def events():
            def event(delta, finish=None):
                return (
                    "data: "
                    + json.dumps(
                        {
                            "id": identifier,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": VLM_MODEL,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                        }
                    )
                    + "\n\n"
                )

            try:
                yield event({"role": "assistant", "content": ""})
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
                try:
                    envelope = json.loads(raw)
                    identifier = envelope.get("id")
                    if not isinstance(identifier, str) or len(identifier) > 64:
                        raise ValueError("Provide a string id of at most 64 characters")
                    result = await dispatch(
                        runtime, settings, envelope["op"], envelope.get("payload", {})
                    )
                    await ws.send_json({"id": identifier, "ok": True, "result": result})
                except Exception as exc:
                    await ws.send_json({"id": identifier, "ok": False, "error": str(exc)})
        except WebSocketDisconnect:
            pass

    app.mount("/mcp", mcp_app)
    return app
