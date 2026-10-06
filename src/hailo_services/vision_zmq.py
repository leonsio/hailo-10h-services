"""Frigate-compatible ZeroMQ REQ/REP detector bridge."""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np

_LOG = logging.getLogger(__name__)


class FrigateZmqServer:
    """Serve the protocol implemented by Frigate's ``type: zmq`` detector."""

    def __init__(self, vision, settings):
        self.vision = vision
        self.settings = settings
        self.context = self.socket = self.task = None

    async def start(self):
        if not (self.settings.vision_enabled and self.settings.vision_zmq_enabled):
            return
        import zmq
        import zmq.asyncio

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.settings.vision_zmq_endpoint)
        self.task = asyncio.create_task(self._run(), name="frigate-zmq-detector")
        _LOG.info("Frigate ZMQ detector listening on %s", self.settings.vision_zmq_endpoint)

    async def _reply_json(self, payload):
        await self.socket.send_multipart([
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ])

    async def _run(self):
        while True:
            try:
                frames = await self.socket.recv_multipart()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("ZMQ receive failed")
                await asyncio.sleep(0.1)
                continue
            try:
                if not frames:
                    raise ValueError("Empty ZMQ detector request")
                header = json.loads(frames[0].decode("utf-8"))
                if header.get("model_request"):
                    model_name = str(header.get("model_name", ""))
                    available = self.vision.ready and self.vision.accepts_model(model_name)
                    await self._reply_json({
                        "model_available": available,
                        "model_loaded": available,
                        "model_name": self.settings.vision_model_id,
                    })
                    continue
                if header.get("model_data"):
                    # Model lifecycle is deliberately server-owned.  Accepting arbitrary
                    # HEFs over an unauthenticated detector socket would also be unsafe.
                    await self._reply_json({
                        "model_saved": False,
                        "model_loaded": False,
                        "error": "Model uploads are disabled; configure models.vision.model on the server",
                    })
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
                detections = await self.vision.detect_array(tensor)
                detections = np.ascontiguousarray(detections, dtype=np.float32)
                response = {"shape": list(detections.shape), "dtype": "float32"}
                await self.socket.send_multipart([
                    json.dumps(response, separators=(",", ":")).encode("utf-8"),
                    detections.tobytes(order="C"),
                ])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Frigate treats a correctly shaped zero result as "no detections";
                # keeping REP state intact is more useful than breaking the detector loop.
                _LOG.exception("ZMQ detector request failed")
                zeros = np.zeros((self.settings.vision_max_detections, 6), dtype=np.float32)
                try:
                    await self.socket.send_multipart([
                        json.dumps({
                            "shape": list(zeros.shape), "dtype": "float32", "error": str(exc)
                        }, separators=(",", ":")).encode("utf-8"),
                        zeros.tobytes(order="C"),
                    ])
                except Exception:
                    _LOG.exception("ZMQ error response failed")

    async def close(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None
