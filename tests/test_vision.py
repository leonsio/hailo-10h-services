import asyncio
import base64
import errno
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hailo_services.app import create_app
from hailo_services.config import Settings
from hailo_services.vision import decode_hailo_nms, decode_yolo26, letterbox
from hailo_services.vision_zmq import FrigateZmqServer


class FakeVisionBackend:
    def __init__(self):
        self.path = "/models/yolov11m.hef"
        self.input_shape = (640, 640, 3)
        self.closed = False

    def start(self):
        pass

    def detect(self, frame, confidence, maximum):
        result = np.zeros((maximum, 6), dtype=np.float32)
        result[0] = [0, 0.91, 0.1, 0.2, 0.8, 0.7]
        return result

    def close(self):
        self.closed = True


def _settings(**changes):
    values = dict(
        vlm_enabled=False,
        whisper_enabled=False,
        minilm_enabled=False,
        hailo_llm_enabled=False,
        litert_enabled=False,
        ha_assist_enabled=False,
        vision_enabled=True,
        vision_model="yolov11m",
        vision_zmq_enabled=False,
    )
    values.update(changes)
    return Settings(**values)


def _png_data_url(width=100, height=50):
    image = Image.new("RGB", (width, height), (255, 255, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def test_letterbox_preserves_expected_shape():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    output = letterbox(frame, 640, 640)
    assert output.shape == (640, 640, 3)
    assert output.dtype == np.uint8
    assert np.all(output[:100, :100] == 114)


def test_hailo_nms_is_converted_to_frigate_contract():
    output = [
        np.array([[0.1, 0.2, 0.8, 0.7, 0.91]], dtype=np.float32),
        np.array([[0.2, 0.3, 0.4, 0.5, 0.20]], dtype=np.float32),
    ]
    detections = decode_hailo_nms(output, threshold=0.4, maximum=20)
    assert detections.shape == (20, 6)
    assert detections.dtype == np.float32
    assert detections[0, 0] == 0
    assert detections[0, 1] > 0.9
    assert not detections[1:].any()


def test_yolo26_decoder_returns_fixed_frigate_contract():
    bbox = np.zeros((1, 80, 80, 4), dtype=np.float32)
    scores = np.full((1, 80, 80, 80), -20, dtype=np.float32)
    bbox[0, 10, 10] = [1, 1, 1, 1]
    scores[0, 10, 10, 0] = 10
    detections = decode_yolo26({"bbox": bbox, "classes": scores}, threshold=0.4, maximum=20)
    assert detections.shape == (20, 6)
    assert detections[0, 0] == 0
    assert detections[0, 1] > 0.99


def test_vision_http_endpoint_and_model_listing():
    settings = _settings()
    backend = FakeVisionBackend()
    app = create_app(settings=settings, vision_backend=backend)
    with TestClient(app) as client:
        models = client.get("/v1/models").json()["data"]
        assert any(model["id"] == "yolov11m" for model in models)
        response = client.post(
            "/v1/vision/detect",
            json={"model": "yolov11m", "image": _png_data_url()},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "vision.detection"
        assert payload["model"] == "yolov11m"
        assert payload["image"] == {"width": 100, "height": 50}
        assert payload["detections"][0]["label"] == "person"
        assert payload["detections"][0]["confidence"] > 0.9
    assert backend.closed


def test_frigate_zmq_model_handshake_and_fixed_output(tmp_path):
    async def scenario():
        import zmq
        import zmq.asyncio

        endpoint = f"ipc://{tmp_path}/vision.sock"
        settings = _settings(vision_zmq_enabled=True, vision_zmq_endpoint=endpoint)
        backend = FakeVisionBackend()

        class FakeVision:
            ready = True

            def accepts_model(self, name):
                return name in {"yolov11m", "yolov11m.hef"}

            async def detect_array(self, tensor, confidence=None, maximum=None):
                return backend.detect(tensor, confidence or 0.4, maximum or 20)

        server = FrigateZmqServer(FakeVision(), settings)
        try:
            await server.start()
        except zmq.ZMQError as exc:
            await server.close()
            if exc.errno == errno.EPERM:
                pytest.skip("Execution environment denies ZeroMQ IPC socket binding")
            raise
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(endpoint)
        try:
            await socket.send_multipart(
                [json.dumps({"model_request": True, "model_name": "yolov11m.hef"}).encode()]
            )
            handshake = json.loads((await socket.recv_multipart())[0])
            assert handshake["model_available"] is True
            assert handshake["model_loaded"] is True

            tensor = np.zeros((1, 640, 640, 3), dtype=np.uint8)
            await socket.send_multipart(
                [
                    json.dumps(
                        {
                            "shape": list(tensor.shape),
                            "dtype": "uint8",
                            "model_type": "yolo-generic",
                        }
                    ).encode(),
                    tensor.tobytes(),
                ]
            )
            frames = await socket.recv_multipart()
            header = json.loads(frames[0])
            result = np.frombuffer(frames[1], dtype=np.float32).reshape(header["shape"])
            assert header == {"shape": [20, 6], "dtype": "float32"}
            assert result.shape == (20, 6)
            assert result[0, 1] > 0.9
        finally:
            socket.close(linger=0)
            context.term()
            await server.close()

    asyncio.run(scenario())
