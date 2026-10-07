"""HTTP/MCP application composition, authentication, UI assets and response envelopes."""

import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from ipaddress import ip_address, ip_network
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
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from .config import FRIGATE_ASSIST_MODEL, HA_ASSIST_MODEL, LLM_MODEL, STT_MODEL, Settings
from .errors import BusyError, LiteRTInferenceError
from .i18n import SUPPORTED_LANGUAGES, catalogue, wait_sentence
from .input_budget import InputBudgetError
from .media import audio_file, audio_metadata, decode_base64
from .metrics import response_metrics, timestamp
from .models import ModelManager
from .protocols import LANGUAGES, MQTTBridge, WyomingServer, dispatch
from .runtime import Runtime
from .schemas import ChatRequest, SpeechRequest, TranscribeRequest, VisionDetectRequest
from .speech_piper import installed_voices
from .speech_runtime import SpeechRuntime
from .tool_calling import has_tool_context
from .vision import COCO80, VisionRuntime
from .vision_zmq import FrigateZmqServer

_LOG = logging.getLogger(__name__)
_WEB = Path(__file__).with_name("web")
_WEB_FILES = {
    "/": "index.html",
    "/ui/app.js": "app.js",
    "/ui/style.css": "style.css",
    "/ui/recorder-worklet.js": "recorder-worklet.js",
}


def _debug(settings, message, *args):
    """Emit a formatted diagnostic when debug logging is enabled.

    Args:
        settings (Settings): Validated service settings controlling enabled models and limits.
        message (str | dict[str, Any]): Formatted diagnostic text or ASGI message.
        *args (Any): Positional arguments forwarded to the native operation.

    Returns:
        None: Writes a debug record when enabled.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if settings.debug_log:
        _LOG.debug(message, *args)


class AccessAndSizeLimit:
    """Enforce ASGI authentication, request size limits and request diagnostics."""

    def __init__(self, app, settings):
        """Initialize AccessAndSizeLimit configuration and owned dependencies.

        Args:
            app (ASGIApp): Application whose lifecycle or ASGI events are wrapped.
            settings (Settings): Validated service settings controlling enabled models and limits.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.app, self.settings = app, settings
        self.mcp_no_auth_networks = tuple(
            ip_network(network.strip())
            for network in settings.mcp_no_auth_networks.split(",")
            if network.strip()
        )

    async def __call__(self, scope, receive, send):
        """Enforce access/body policy before dispatching an ASGI request.

        Args:
            scope (dict[str, Any]): ASGI connection scope including peer, path and request state.
            receive (Callable): ASGI receive callback yielding incoming events.
            send (Callable): ASGI send callback forwarding response events.

        Returns:
            None: Sends the application response or rejects invalid/unauthorized input.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex[:12]
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        scope["state"].update(request_started=started, requested_at=timestamp())
        headers = dict(scope.get("headers", []))
        path = scope.get("path", "-")
        method = scope.get("method", "WEBSOCKET")
        peer = scope.get("client") or ("unknown", 0)
        is_websocket = scope["type"] == "websocket"
        protocol = (
            "websocket"
            if is_websocket
            else "mcp"
            if path == "/mcp" or path.startswith("/mcp/")
            else "http"
        )
        transport = "websocket" if is_websocket else "http"
        _debug(
            self.settings,
            "protocol=%s transport=%s event=request_start request_id=%s method=%s path=%s peer=%s content_type=%s content_length=%s",
            protocol,
            transport,
            request_id,
            method,
            path,
            peer[0],
            headers.get(b"content-type", b"-").decode("latin1"),
            headers.get(b"content-length", b"-").decode("latin1"),
        )
        response_status = "accepted"

        async def debug_send(message):
            """Record response status before forwarding an ASGI message.

            Args:
                message (str | dict[str, Any]): Formatted diagnostic text or ASGI message.

            Returns:
                None: Forwards the message to the original send callback.

            Notes:
                No application-specific exceptions are raised for valid inputs.
            """
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "websocket.close":
                response_status = "closed:" + str(message.get("code", ""))
            await send(message)

        async def call_app(current_scope, current_receive):
            """Run the wrapped application and log request duration and outcome.

            Args:
                current_scope (dict[str, Any]): Scope passed to the wrapped ASGI application.
                current_receive (Callable): Bounded ASGI receive callback.

            Returns:
                None: Completes the wrapped ASGI request.

            Notes:
                No application-specific exceptions are raised for valid inputs.
            """
            nonlocal response_status
            try:
                await self.app(current_scope, current_receive, debug_send)
            except Exception:
                if response_status == "accepted":
                    response_status = "failed" if is_websocket else 500
                raise
            finally:
                _debug(
                    self.settings,
                    "protocol=%s transport=%s event=request_end request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
                    protocol,
                    transport,
                    request_id,
                    method,
                    path,
                    response_status,
                    (time.perf_counter() - started) * 1000,
                )

        expected = f"Bearer {self.settings.api_key}".encode()
        public = scope["path"] == "/health" or (
            scope["type"] == "http"
            and scope.get("method") in {"GET", "HEAD"}
            and (
                scope["path"] in {*_WEB_FILES, "/ui/config"}
                or scope["path"] in {f"/ui/locales/{lang}.json" for lang in SUPPORTED_LANGUAGES}
            )
        )
        # Trust the socket peer only, never client-supplied forwarding headers.
        if protocol == "mcp" and not is_websocket:
            try:
                address = ip_address(peer[0])
                if address.version == 6 and address.ipv4_mapped:
                    address = address.ipv4_mapped
                public = any(address in network for network in self.mcp_no_auth_networks)
            except ValueError:
                pass
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
        if self.settings.debug_log:
            # Log the complete inbound HTTP request after the bounded body has
            # been collected. Keep credentials out of logs while preserving all
            # other headers and the exact JSON/body sent by the client.
            debug_headers = {
                key.decode("latin1"): (
                    "<redacted>"
                    if key.lower()
                    in {
                        b"authorization",
                        b"proxy-authorization",
                        b"cookie",
                        b"set-cookie",
                        b"x-api-key",
                        b"api-key",
                    }
                    else value.decode("latin1")
                )
                for key, value in scope.get("headers", [])
            }
            content_type = headers.get(b"content-type", b"").decode("latin1").lower()
            if "application/json" in content_type:
                try:
                    debug_body = json.dumps(
                        json.loads(bytes(body)), ensure_ascii=False, separators=(",", ":")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    debug_body = bytes(body).decode("utf-8", errors="replace")
            elif content_type.startswith("text/") or not body:
                debug_body = bytes(body).decode("utf-8", errors="replace")
            else:
                debug_body = f"<binary body: {len(body)} bytes>"
            _LOG.debug(
                "protocol=%s transport=%s event=request_payload request_id=%s "
                "method=%s path=%s headers=%s body=%s",
                protocol,
                transport,
                request_id,
                method,
                path,
                json.dumps(debug_headers, ensure_ascii=False, separators=(",", ":")),
                debug_body,
            )

        delivered = False

        async def bounded_receive():
            """Reject streamed request bodies that exceed the configured limit.

            Returns:
                dict[str, Any]: Next ASGI receive event.

            Notes:
                No application-specific exceptions are raised for valid inputs.
            """
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await call_app(scope, bounded_receive)


def completion(text, identifier, created, model):
    """Build an OpenAI completion envelope from validated chat output.

    Args:
        text (str): Text to parse, normalize, match or render.
        identifier (str): Completion or call identifier.
        created (int): Unix timestamp in whole seconds.
        model (str): Public model identifier or configured HEF path.

    Returns:
        dict[str, Any]: Completion ID, model, choices and finish reason.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    message = text if isinstance(text, dict) else {"role": "assistant", "content": text}
    return {
        "id": identifier,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
    }


def create_app(
    settings=None, backend=None, litert_backend=None, vision_backend=None, speech_backend=None
):
    """Compose HTTP, MCP, Wyoming, MQTT and resident inference services.

    Args:
        settings (Settings): Validated service settings controlling enabled models and limits.
        backend (ChatBackend): Resident backend used for generation or context preparation.
        litert_backend (ChatBackend | None): Injected LiteRT backend; None creates one when enabled.
        speech_backend (PiperBackend | None): Injectable CPU speech adapter.
        vision_backend (HailoVisionBackend | None): Injected detector backend; None uses the resident Hailo detector.

    Returns:
        FastAPI: Application with backend lifecycle and request validation.

    Raises:
        ValueError: Service settings or protocol configuration are invalid.
    """
    settings = settings or Settings.from_env()
    runtime = Runtime(settings, backend, litert_backend)
    speech = SpeechRuntime(settings, speech_backend)
    vision = VisionRuntime(settings, vision_backend)
    frigate_zmq = FrigateZmqServer(vision, settings)
    wyoming = WyomingServer(runtime, settings, speech)
    mqtt = MQTTBridge(runtime, settings)
    mcp = MCPServer("Hailo-10H", version="0.1.0")

    @mcp.tool()
    async def analyze_image(image_base64: str, prompt: str, max_tokens: int = 256) -> str:
        """Analyze a base64-encoded camera snapshot using the configured resident VLM.

        Args:
            image_base64 (str): Base64-encoded image or image data URL.
            prompt (str): Question text or prepared native prompt messages.
            max_tokens (int): Maximum generated tokens reserved for the response.

        Returns:
            str: Model analysis of the supplied image.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
        """Transcribe a base64 WAV/FLAC/OGG recording with the configured multilingual Whisper model.

        Args:
            audio_base64 (str): Base64-encoded recording or audio data URL.
            language (str | None): Language code; None uses the configured or detected language.

        Returns:
            str: Recognized speech text.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        request = TranscribeRequest(audio_base64=audio_base64, language=language)
        data = decode_base64(request.audio_base64, settings.max_body)
        _debug(
            settings,
            "protocol=mcp operation=whisper_transcribe request_id=%s model=%s language=%s audio=%s bytes=%d",
            uuid.uuid4().hex[:12],
            settings.stt_model,
            request.language or settings.language,
            audio_metadata(data),
            len(data),
        )
        return await runtime.transcribe(
            audio_file(data, settings.max_audio_seconds), request.language
        )

    @mcp.tool()
    async def chat_text(prompt: str, max_tokens: int = 256) -> str:
        """Ask the configured chat model a text question. Does not execute Home Assistant actions.

        Args:
            prompt (str): Question text or prepared native prompt messages.
            max_tokens (int): Maximum generated tokens reserved for the response.

        Returns:
            str: Generated text or a validated assistant message.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
        """Start resident services and release them when the application stops.

        Args:
            app (ASGIApp): Application whose lifecycle or ASGI events are wrapped.

        Yields:
            None: Keeps services active within the application lifespan.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        mqtt_task = None
        try:
            _LOG.info("Startup phase 1/2: loading resident Hailo-10H models")
            await runtime.start_hailo()  # Failure prevents all listeners from becoming ready.
            await vision.start()
            _LOG.info("Startup phase 1/2 complete: resident Hailo-10H models initialized")

            _LOG.info("Startup phase 2/2: loading CPU models")
            await runtime.start_cpu()
            await speech.start()
            _LOG.info("Startup phase 2/2 complete: CPU model initialization finished")

            await frigate_zmq.start()
            if settings.wyoming_port:
                await wyoming.start()
            if settings.mqtt_host:
                mqtt_task = asyncio.create_task(mqtt.run(), name="mqtt-bridge")
            async with mcp.session_manager.run():
                yield
        finally:
            await frigate_zmq.close()
            await vision.close()
            await wyoming.close()
            if mqtt_task:
                mqtt_task.cancel()
                with suppress(asyncio.CancelledError):
                    await mqtt_task
            await speech.close()
            await runtime.close()

    app = FastAPI(title="Hailo-10H Services", version="0.1.0", lifespan=lifespan)
    app.state.speech = speech
    app.state.runtime = runtime
    app.state.vision = vision
    app.state.frigate_zmq = frigate_zmq
    app.state.mqtt = mqtt
    app.add_middleware(AccessAndSizeLimit, settings=settings)

    async def web_file(request):
        """Serve a bundled UI asset with restrictive browser security headers.

        Args:
            request (Request): HTTP request associated with this operation.

        Returns:
            FileResponse: Requested static asset.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
        """Expose enabled models, limits and language settings to the browser.

        Returns:
            dict[str, Any]: Public UI configuration without credentials.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return {
            "auth_required": bool(settings.api_key),
            "vlm_model": settings.vlm_model,
            "llm_model": LLM_MODEL,
            "hailo_llm_model": settings.hailo_llm_model_id,
            "default_text_model": runtime.default_text_model,
            "vision_models": ([settings.vlm_model] if settings.vlm_enabled else [])
            + (
                [HA_ASSIST_MODEL]
                if settings.ha_assist_enabled
                and settings.vlm_enabled
                and settings.ha_assist_vision_model == settings.vlm_model
                else []
            )
            + (
                [FRIGATE_ASSIST_MODEL]
                if settings.frigate_assist_enabled
                and settings.vlm_enabled
                and settings.frigate_assist_vision_model == settings.vlm_model
                else []
            ),
            "object_detection": vision.status(),
            "ha_assist_model": HA_ASSIST_MODEL,
            "ha_assist": runtime.status()["ha_assist"],
            "frigate_assist": runtime.status()["frigate_assist"],
            "model_limits": runtime.model_limits,
            "vlm_max_images": ModelManager(settings)
            .entries.get(settings.vlm_model, {})
            .get("max_images", 1),
            "chat_models": runtime.chat_models,
            "piper": {
                "enabled": settings.piper_enabled,
                "default_voice": Path(settings.piper_voice).stem,
                "language": settings.piper_language,
                "max_input_chars": settings.piper_max_input_chars,
                "voices": installed_voices(settings),
            },
            "whisper_model": settings.stt_model,
            "language": settings.language,
            "service_language": settings.service_language,
            "ui_languages": list(SUPPORTED_LANGUAGES),
            "stt_languages": LANGUAGES,
            "max_body": settings.max_body,
            "max_audio_seconds": settings.max_audio_seconds,
            "recording_seconds": min(30, settings.max_audio_seconds),
        }

    @app.get("/ui/locales/{language}.json", include_in_schema=False)
    async def web_locale(language: str):
        """Return browser translations for a supported UI language.

        Args:
            language (str): Language code; None uses the configured or detected language.

        Returns:
            dict[str, Any] | JSONResponse: Translation catalogue or a 404 response.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if language not in SUPPORTED_LANGUAGES:
            return JSONResponse({"error": "Unsupported language"}, status_code=404)
        data = catalogue(language)
        return {
            "ui": data["ui"],
            "language_name": data["language_name"],
            "language_names": data["language_names"],
        }

    @app.exception_handler(BusyError)
    async def busy_handler(request, exc):
        """Map unavailable models and full queues to HTTP 503.

        Args:
            request (Request): HTTP request associated with this operation.
            exc (BaseException | None): Exception being translated or passed to context cleanup.

        Returns:
            JSONResponse: Error response with a retry hint.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return JSONResponse({"error": str(exc)}, status_code=503, headers={"Retry-After": "5"})

    @app.exception_handler(ValueError)
    async def value_handler(request, exc):
        """Map invalid service input to HTTP 400.

        Args:
            request (Request): HTTP request associated with this operation.
            exc (BaseException | None): Exception being translated or passed to context cleanup.

        Returns:
            JSONResponse: Client error message.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(InputBudgetError)
    async def input_budget_handler(request, exc):
        """Expose required input tokens and the effective limit as HTTP 400.

        Args:
            request (Request): HTTP request associated with this operation.
            exc (BaseException | None): Exception being translated or passed to context cleanup.

        Returns:
            JSONResponse: Structured input_token_limit_exceeded error.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return JSONResponse(
            {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": "input_token_limit_exceeded",
                    "input_tokens": exc.tokens,
                    "input_limit": exc.limit,
                }
            },
            status_code=400,
        )

    @app.exception_handler(LiteRTInferenceError)
    async def litert_error_handler(request, exc):
        """Map native LiteRT inference failure to HTTP 502.

        Args:
            request (Request): HTTP request associated with this operation.
            exc (BaseException | None): Exception being translated or passed to context cleanup.

        Returns:
            JSONResponse: Structured inference error.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return JSONResponse(
            {"error": {"message": str(exc), "type": "inference_error"}}, status_code=502
        )

    @app.exception_handler(asyncio.TimeoutError)
    async def timeout_handler(request, exc):
        """Map inference deadline expiry to HTTP 504.

        Args:
            request (Request): HTTP request associated with this operation.
            exc (BaseException | None): Exception being translated or passed to context cleanup.

        Returns:
            JSONResponse: Timeout response; native work may still be running.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return JSONResponse(
            {"error": "Inference timed out; native work may still be completing"}, status_code=504
        )

    @app.get("/health")
    async def health():
        """Return readiness and status for inference and transport services.

        Returns:
            JSONResponse: HTTP 200 when required services are ready, otherwise 503.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return JSONResponse(
            {
                **runtime.status(),
                "vision": vision.status(),
                "piper": speech.status(),
                "mqtt_connected": mqtt.connected,
            },
            status_code=200
            if runtime.ready and (not settings.vision_enabled or vision.ready)
            else 503,
        )

    @app.get("/v1/models")
    async def models():
        """List currently enabled or ready models in OpenAI format.

        Returns:
            dict[str, Any]: Model identifiers and owning backend names.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": "hailo"}
                for model in runtime.hailo_chat_models
                + ([settings.stt_model] if settings.whisper_enabled else [])
            ]
            + (
                [{"id": settings.vision_model_id, "object": "model", "owned_by": "hailo"}]
                if vision.ready
                else []
            )
            + (
                [{"id": LLM_MODEL, "object": "model", "owned_by": "litert-lm"}]
                if runtime.litert_ready
                else []
            )
            + (
                [{"id": "piper", "object": "model", "owned_by": "piper-cpu"}]
                if speech.ready
                else []
            )
            + (
                [{"id": HA_ASSIST_MODEL, "object": "model", "owned_by": "hailo-services"}]
                if settings.ha_assist_enabled
                else []
            )
            + (
                [{"id": FRIGATE_ASSIST_MODEL, "object": "model", "owned_by": "hailo-services"}]
                if settings.frigate_assist_enabled
                else []
            ),
        }

    @app.post("/v1/vision/detect")
    async def vision_detect(request: VisionDetectRequest, http_request: Request):
        """Detect objects and report normalized and pixel-space boxes.

        Args:
            request (VisionDetectRequest): Validated chat request, history, tool policy and request-local metadata.
            http_request (Request): HTTP request providing correlation and timing state.

        Returns:
            dict[str, Any]: Detections, image dimensions and request metrics.

        Raises:
            HTTPException: The detector is unavailable (503) or the requested model is unknown (400).
        """
        if not vision.ready:
            raise HTTPException(503, "Vision model is disabled or not ready")
        if request.model and request.model not in {
            settings.vision_model,
            settings.vision_model_id,
            Path(settings.vision_model).name,
        }:
            raise HTTPException(400, f"Unknown vision model: {request.model}")
        started = time.perf_counter()
        width, height, raw = await vision.detect_base64(
            request.image, request.confidence, request.max_detections
        )
        detections = []
        for row in raw:
            class_id = int(row[0])
            confidence = float(row[1])
            if confidence <= 0:
                continue
            ymin, xmin, ymax, xmax = (float(value) for value in row[2:6])
            detections.append(
                {
                    "class_id": class_id,
                    "label": COCO80[class_id] if 0 <= class_id < len(COCO80) else str(class_id),
                    "confidence": confidence,
                    "box": {"x_min": xmin, "y_min": ymin, "x_max": xmax, "y_max": ymax},
                    "box_pixels": {
                        "x_min": max(0, min(width, round(xmin * width))),
                        "y_min": max(0, min(height, round(ymin * height))),
                        "x_max": max(0, min(width, round(xmax * width))),
                        "y_max": max(0, min(height, round(ymax * height))),
                    },
                }
            )
        request_id = http_request.scope.get("state", {}).get("request_id", "-")
        _debug(
            settings,
            "protocol=http operation=vision_detect request_id=%s model=%s detections=%d inference_ms=%.1f",
            request_id,
            settings.vision_model_id,
            len(detections),
            (time.perf_counter() - started) * 1000,
        )
        return {
            "id": "vision-" + uuid.uuid4().hex,
            "object": "vision.detection",
            "created": int(time.time()),
            "model": settings.vision_model_id,
            "image": {"width": width, "height": height},
            "detections": detections,
            "metrics": response_metrics({}, http_request.scope["state"]),
        }

    @app.post("/v1/ha-assist/diagnose")
    async def ha_diagnose(request: ChatRequest, http_request: Request):
        """Explain preparation without invoking a generative backend or executing tools.

        Args:
            request: HA-Assist request with stream disabled.
            http_request: HTTP request carrying the diagnostic correlation ID.

        Returns:
            dict: Candidate scores, proposed response and prepared messages/tools.

        Raises:
            ValueError: Invalid diagnostic model or tool/history input.
            BusyError: The owner-thread preparation queue is unavailable.
        """
        from .ha_diagnostics import diagnose

        request._request_id = http_request.scope.get("state", {}).get("request_id", "-")
        return await runtime.call(diagnose, runtime.backend, request)

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest, http_request: Request):
        """Return an OpenAI completion or SSE stream with request metrics.

        Args:
            request (ChatRequest): Validated chat request, history, tool policy and request-local metadata.
            http_request (Request): HTTP request providing correlation and timing state.

        Returns:
            dict[str, Any] | StreamingResponse: Completion envelope or streaming HTTP response.

        Raises:
            ValueError: Input, selected model or generated function calls are invalid.
            BusyError: A required backend is unavailable or its queue is full.
            InputBudgetError: Required context exceeds the effective input budget.
            LiteRTInferenceError: Native CPU generation fails.
            asyncio.TimeoutError: Inference exceeds the configured request deadline.
        """
        request = runtime.default_chat_request(request)
        identifier, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        request_id = http_request.scope.get("state", {}).get("request_id", "-")
        request._request_id = request_id
        image_count = sum(
            part.get("type") == "image_url"
            for message in request.messages
            for part in (message["content"] if isinstance(message.get("content"), list) else [])
        )
        _debug(
            settings,
            "protocol=http operation=chat_completion request_id=%s model=%s messages=%d images=%d max_tokens=%d max_input_tokens=%s stream=%s",
            request_id,
            request.model,
            len(request.messages),
            image_count,
            request.max_tokens,
            request.max_input_tokens,
            request.stream,
        )
        if settings.debug_log:
            _LOG.debug(
                "protocol=http operation=chat_completion event=validated_request "
                "request_id=%s json=%s",
                request_id,
                request.model_dump_json(exclude_none=False),
            )
        if not request.stream:
            result = completion(await runtime.chat(request), identifier, created, request.model)
            result["metrics"] = response_metrics(request._metrics, http_request.scope["state"])
            counts = request._metrics
            if "input_tokens" in counts and "output_tokens" in counts:
                result["usage"] = {
                    "prompt_tokens": counts["input_tokens"],
                    "completion_tokens": counts["output_tokens"],
                    "total_tokens": counts["input_tokens"] + counts["output_tokens"],
                }
            return result

        async def events():
            """Stream OpenAI completion chunks, timing metadata and terminal markers.

            Yields:
                str: SSE event containing a completion chunk or error.

            Notes:
                No application-specific exceptions are raised for valid inputs.
            """

            stream_finish = None
            stream_status = "completed"

            def event(delta, finish=None):
                """Serialize an OpenAI completion chunk as a server-sent event.

                Args:
                    delta (dict[str, Any]): OpenAI streaming message delta.
                    finish (str | None): Finish reason, or None while generation continues.

                Returns:
                    str: SSE data frame.

                Notes:
                    No application-specific exceptions are raised for valid inputs.
                """
                nonlocal stream_finish
                if finish:
                    stream_finish = finish
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
                if request.model in {HA_ASSIST_MODEL, FRIGATE_ASSIST_MODEL} or has_tool_context(
                    request
                ):
                    # Buffer native tool output so invalid/incomplete calls are never streamed
                    # as actions or spoken as text by the voice assistant.
                    if (
                        request.model == HA_ASSIST_MODEL
                        and settings.ha_wait_messages
                        and isinstance(runtime, Runtime)
                    ):
                        loop = asyncio.get_running_loop()
                        started = asyncio.Event()
                        response_language = [settings.service_language]

                        def on_inference(language):
                            """Emit a localized wait notification when native inference starts.

                            Args:
                                language (str | None): Language code; None uses the configured or detected language.

                            Returns:
                                None: Queues a wait notification for streaming clients.

                            Notes:
                                No application-specific exceptions are raised for valid inputs.
                            """
                            response_language[0] = language
                            loop.call_soon_threadsafe(started.set)

                        inference = asyncio.create_task(runtime.chat(request, on_inference))
                        notification = asyncio.create_task(started.wait())
                        try:
                            await asyncio.wait(
                                {inference, notification}, return_when=asyncio.FIRST_COMPLETED
                            )
                            if started.is_set() and not inference.done():
                                yield event({"content": wait_sentence(response_language[0]) + "\n"})
                            result = await inference
                        finally:
                            notification.cancel()
                            if not inference.done():
                                inference.cancel()
                            await asyncio.gather(notification, inference, return_exceptions=True)
                    else:
                        result = await runtime.chat(request)
                    if isinstance(result, dict):
                        if result.get("content"):
                            yield event({"content": result["content"]})
                        if result.get("tool_calls"):
                            yield event(
                                {
                                    "tool_calls": [
                                        {"index": index, **call}
                                        for index, call in enumerate(result["tool_calls"])
                                    ]
                                }
                            )
                        yield event({}, "tool_calls" if result.get("tool_calls") else "stop")
                    else:
                        yield event({"content": result})
                        yield event({}, "stop")
                else:
                    async for chunk in runtime.stream(request):
                        yield event({"content": chunk})
                    yield event({}, "stop")
                if request.stream_options and request.stream_options.include_usage:
                    counts = request._metrics
                    usage = None
                    if "input_tokens" in counts and "output_tokens" in counts:
                        usage = {
                            "prompt_tokens": counts["input_tokens"],
                            "completion_tokens": counts["output_tokens"],
                            "total_tokens": counts["input_tokens"] + counts["output_tokens"],
                        }
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": identifier,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [],
                                "usage": usage,
                                "metrics": response_metrics(
                                    request._metrics, http_request.scope["state"]
                                ),
                            }
                        )
                        + "\n\n"
                    )
            except (asyncio.CancelledError, GeneratorExit):
                if request.model == FRIGATE_ASSIST_MODEL:
                    _LOG.info(
                        "event=frigate_stream_end request_id=%s status=cancelled finish_reason=%s done_emitted=false",
                        request_id,
                        stream_finish,
                    )
                raise
            except Exception as exc:
                stream_status = "error"
                _LOG.exception("Streaming inference failed request_id=%s", request_id)
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": {
                                "message": str(exc),
                                "type": "invalid_request_error"
                                if isinstance(exc, InputBudgetError)
                                else "inference_error",
                                **(
                                    {
                                        "code": "input_token_limit_exceeded",
                                        "input_tokens": exc.tokens,
                                        "input_limit": exc.limit,
                                    }
                                    if isinstance(exc, InputBudgetError)
                                    else {}
                                ),
                            }
                        }
                    )
                    + "\n\n"
                )
            yield "data: [DONE]\n\n"
            if request.model == FRIGATE_ASSIST_MODEL:
                _LOG.info(
                    "event=frigate_stream_end request_id=%s status=%s finish_reason=%s done_emitted=true",
                    request_id,
                    stream_status,
                    stream_finish,
                )

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/audio/speech")
    async def audio_speech(body: SpeechRequest):
        """Generate CPU speech with an installed Piper voice.

        Args:
            body (SpeechRequest): Text, local voice, language, speed and output format.

        Returns:
            Response: Binary WAV or headerless mono PCM16LE at 24 kHz.

        Raises:
            HTTPException: Native speech synthesis fails.
        """
        started = time.perf_counter()
        try:
            data, content_type, rate = await speech.synthesize(body)
        except (ValueError, BusyError, asyncio.TimeoutError):
            raise
        except Exception as exc:
            _LOG.exception("Piper synthesis failed")
            raise HTTPException(status_code=502, detail="Piper synthesis failed") from exc
        return Response(
            data,
            media_type=content_type,
            headers={
                "X-Audio-Sample-Rate": str(rate),
                "X-Inference-Ms": f"{(time.perf_counter() - started) * 1000:.3f}",
            },
        )

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = STT_MODEL,
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
    ):
        """Decode an uploaded recording and return a Whisper transcription.

        Args:
            request (Request): HTTP request associated with this operation.
            file (Annotated[UploadFile, File()]): Uploaded supported audio recording.
            model (str): Public model identifier or configured HEF path.
            language (Annotated[str | None, Form()]): Language code; None uses the configured or detected language.
            response_format (Annotated[str, Form()]): Requested transcription output format: json or text.

        Returns:
            dict | PlainTextResponse: Transcript with request metrics, or plain text when requested.

        Raises:
            HTTPException: The model or response format is unsupported, or the upload is oversized.
        """
        if model not in {settings.stt_model, settings.whisper_hef, "whisper-1", STT_MODEL}:
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
        _debug(
            settings,
            "protocol=http operation=whisper_transcribe request_id=%s model=%s language=%s audio=%s bytes=%d",
            request.scope.get("state", {}).get("request_id", "-"),
            model,
            language or settings.language,
            metadata,
            len(data),
        )
        text = await runtime.transcribe(audio_file(data, settings.max_audio_seconds), language)
        return (
            PlainTextResponse(text)
            if response_format == "text"
            else {
                "text": text,
                "metrics": response_metrics({}, request.scope["state"]),
            }
        )

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        """Dispatch JSON chat, transcription and health operations over WebSocket.

        Args:
            ws (WebSocket): Connected client socket.

        Returns:
            None: Sends operation results or structured errors until disconnect.

        Raises:
            ValueError: Provide a string id of at most 64 characters.
        """
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
                    _debug(
                        settings,
                        "protocol=websocket event=request_start request_id=%s operation=%s",
                        identifier,
                        operation,
                    )
                    result = await dispatch(
                        runtime, settings, envelope["op"], envelope.get("payload", {})
                    )
                    await ws.send_json({"id": identifier, "ok": True, "result": result})
                    _debug(
                        settings,
                        "protocol=websocket event=request_end request_id=%s operation=%s duration_ms=%.1f ok=true",
                        identifier,
                        operation,
                        (time.perf_counter() - started) * 1000,
                    )
                except Exception as exc:
                    _debug(
                        settings,
                        "protocol=websocket event=request_error request_id=%s operation=%s duration_ms=%.1f error_type=%s",
                        identifier or "-",
                        operation,
                        (time.perf_counter() - started) * 1000,
                        type(exc).__name__,
                    )
                    await ws.send_json({"id": identifier, "ok": False, "error": str(exc)})
        except WebSocketDisconnect:
            pass

    app.mount("/mcp", mcp_app)
    return app
