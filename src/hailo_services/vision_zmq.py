"""Frigate-compatible ZeroMQ REQ/REP detector bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import numpy as np

_LOG = logging.getLogger(__name__)


class FrigateZmqServer:
    """Serve the protocol implemented by Frigate's ``type: zmq`` detector."""

    def __init__(self, vision, settings):
        """Initialize FrigateZmqServer configuration and owned dependencies.

        Args:
            vision (VisionRuntime): Resident detector scheduler used by the bridge.
            settings (Settings): Validated service settings controlling enabled models and limits.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.vision = vision
        self.settings = settings
        self.context = self.socket = self.task = None
        self.monitor_socket = self.monitor_task = None

    async def start(self):
        """Initialize resident resources or start the configured transport listener.

        Returns:
            None: Marks the service ready after successful initialization.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if not (self.settings.vision_enabled and self.settings.vision_zmq_enabled):
            return
        import zmq
        import zmq.asyncio

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        if _LOG.isEnabledFor(logging.DEBUG):
            self.monitor_socket = self.socket.get_monitor_socket(
                events=zmq.EVENT_ACCEPTED | zmq.EVENT_HANDSHAKE_SUCCEEDED | zmq.EVENT_DISCONNECTED
            )
            self.monitor_task = asyncio.create_task(self._monitor(), name="frigate-zmq-monitor")
        self.socket.bind(self.settings.vision_zmq_endpoint)
        self.task = asyncio.create_task(self._run(), name="frigate-zmq-detector")
        _LOG.info("Frigate ZMQ detector listening on %s", self.settings.vision_zmq_endpoint)

    async def _monitor(self):
        """Log transport connections independently of detector requests."""
        import zmq
        from zmq.utils.monitor import parse_monitor_message

        names = {
            zmq.EVENT_ACCEPTED: "connected",
            zmq.EVENT_HANDSHAKE_SUCCEEDED: "handshake_complete",
            zmq.EVENT_DISCONNECTED: "disconnected",
        }
        while True:
            try:
                event = parse_monitor_message(await self.monitor_socket.recv_multipart())
                endpoint = event["endpoint"].decode("utf-8", errors="replace")
                _LOG.debug(
                    "protocol=zmq event=%s endpoint=%s",
                    names.get(event["event"], str(event["event"])),
                    endpoint,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("ZMQ transport monitor failed")
                return

    async def _reply_json(self, payload):
        """Send a JSON-only reply while preserving ZeroMQ REP state.

        Args:
            payload (Any): Decoded protocol data or structured diagnostic payload.

        Returns:
            None: Sends one multipart JSON frame.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        await self.socket.send_multipart(
            [json.dumps(payload, separators=(",", ":")).encode("utf-8")]
        )

    async def _run(self):
        """Serve detector model probes and tensor inference on the ZeroMQ REP socket.

        Returns:
            None: Processes requests until the server task is cancelled.

        Raises:
            ValueError: Invalid or oversized detector tensor.
        """
        while True:
            try:
                frames = await self.socket.recv_multipart()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("ZMQ receive failed")
                await asyncio.sleep(0.1)
                continue
            request_id = uuid.uuid4().hex[:12]
            started = time.perf_counter()
            _LOG.debug(
                "protocol=zmq event=request_received request_id=%s frames=%d bytes=%d",
                request_id,
                len(frames),
                sum(len(frame) for frame in frames),
            )
            try:
                if not frames:
                    raise ValueError("Empty ZMQ detector request")
                header = json.loads(frames[0].decode("utf-8"))
                if header.get("model_request"):
                    model_name = str(header.get("model_name", ""))
                    available = self.vision.ready and self.vision.accepts_model(model_name)
                    await self._reply_json(
                        {
                            "model_available": available,
                            "model_loaded": available,
                            "model_name": self.settings.vision_model_id,
                        }
                    )
                    _LOG.debug(
                        "protocol=zmq event=model_response request_id=%s model=%s available=%s duration_ms=%.1f",
                        request_id,
                        model_name,
                        available,
                        (time.perf_counter() - started) * 1000,
                    )
                    continue
                if header.get("model_data"):
                    # Model lifecycle is deliberately server-owned. Accepting arbitrary
                    # HEFs over an unauthenticated detector socket would also be unsafe.
                    await self._reply_json(
                        {
                            "model_saved": False,
                            "model_loaded": False,
                            "error": "Model uploads are disabled; configure models.vision.model on the server",
                        }
                    )
                    _LOG.debug(
                        "protocol=zmq event=model_upload_rejected request_id=%s duration_ms=%.1f",
                        request_id,
                        (time.perf_counter() - started) * 1000,
                    )
                    continue
                if len(frames) != 2:
                    raise ValueError("Inference requires header and tensor frames")
                shape = tuple(int(value) for value in header.get("shape", []))
                if len(shape) not in {3, 4} or any(value < 1 for value in shape):
                    raise ValueError("Invalid detector tensor shape")
                if len(shape) == 4 and shape[0] != 1:
                    raise ValueError("Only batch size 1 is supported")
                dtype = np.dtype(header.get("dtype", "uint8"))
                if dtype not in {np.dtype("uint8"), np.dtype("float32")}:
                    raise ValueError("Only uint8 and float32 detector tensors are supported")
                expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if expected != len(frames[1]) or expected > self.settings.max_body:
                    raise ValueError("Invalid or oversized detector tensor")
                tensor = np.frombuffer(frames[1], dtype=dtype).reshape(shape)
                # Frigate's detector shared-memory contract is exactly (20, 6).
                _LOG.debug(
                    "protocol=zmq event=inference_start request_id=%s model=%s shape=%s dtype=%s tensor_bytes=%d pending=%s",
                    request_id,
                    self.settings.vision_model_id,
                    shape,
                    dtype,
                    len(frames[1]),
                    self.vision.pending,
                )
                inference_started = time.perf_counter()
                detections = await self.vision.detect_array(tensor, maximum=20)
                inference_ms = (time.perf_counter() - inference_started) * 1000
                detections = np.ascontiguousarray(detections, dtype=np.float32).reshape((20, 6))
                response = {"shape": [20, 6], "dtype": "float32"}
                await self.socket.send_multipart(
                    [
                        json.dumps(response, separators=(",", ":")).encode("utf-8"),
                        detections.tobytes(order="C"),
                    ]
                )
                if _LOG.isEnabledFor(logging.DEBUG):
                    active = detections[detections[:, 1] > 0]
                    _LOG.debug(
                        "protocol=zmq event=inference_response request_id=%s detections=%d inference_ms=%.1f duration_ms=%.1f response_bytes=%d results=%s",
                        request_id,
                        len(active),
                        inference_ms,
                        (time.perf_counter() - started) * 1000,
                        detections.nbytes,
                        active.tolist(),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Frigate treats a correctly shaped zero result as "no detections";
                # keeping REP state intact is more useful than breaking the detector loop.
                _LOG.exception(
                    "protocol=zmq event=request_failed request_id=%s duration_ms=%.1f",
                    request_id,
                    (time.perf_counter() - started) * 1000,
                )
                zeros = np.zeros((20, 6), dtype=np.float32)
                try:
                    await self.socket.send_multipart(
                        [
                            json.dumps(
                                {"shape": [20, 6], "dtype": "float32", "error": str(exc)},
                                separators=(",", ":"),
                            ).encode("utf-8"),
                            zeros.tobytes(order="C"),
                        ]
                    )
                except Exception:
                    _LOG.exception("ZMQ error response failed")

    async def close(self):
        """Release resources owned by this service or native context.

        Returns:
            None: Closes native resources, connections or owner executors.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        if self.monitor_task is not None:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        if self.socket is not None:
            if self.monitor_socket is not None:
                self.socket.disable_monitor()
            self.socket.close(linger=0)
        if self.monitor_socket is not None:
            self.monitor_socket.close(linger=0)
            self.monitor_socket = None
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None
