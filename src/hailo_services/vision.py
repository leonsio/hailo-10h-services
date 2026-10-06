"""Resident object-detection runtime for Hailo-10H.

The selected HEF is configured once and kept resident.  HTTP and ZMQ clients share
this runtime; no per-request VDevice or model configuration is performed.
"""

from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .media import decode_base64
from .models import ModelManager, prepare_model_version

_LOG = logging.getLogger(__name__)

COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


def _strip_batch(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Vision input must have shape HxWx3 or 1xHxWx3")
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize without distortion and pad using the conventional YOLO value 114."""
    frame = _strip_batch(frame)
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    image = Image.fromarray(frame, mode="RGB").resize((dst_w, dst_h), Image.Resampling.BILINEAR)
    output = np.full((height, width, 3), 114, dtype=np.uint8)
    x = (width - dst_w) // 2
    y = (height - dst_h) // 2
    output[y : y + dst_h, x : x + dst_w] = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(output)


def _class_aware_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.int64)
    keep: list[int] = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_current = max(0.0, (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1]))
            areas = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
            overlap = inter / (area_current + areas - inter + 1e-6)
            order = rest[overlap <= iou]
    return np.asarray(sorted(keep, key=lambda idx: scores[idx], reverse=True), dtype=np.int64)


def _frigate_array(rows: list[list[float]], maximum: int) -> np.ndarray:
    result = np.zeros((maximum, 6), dtype=np.float32)
    if rows:
        ordered = sorted(rows, key=lambda row: row[1], reverse=True)[:maximum]
        result[: len(ordered)] = np.asarray(ordered, dtype=np.float32)
    return result


def decode_hailo_nms(output, threshold: float, maximum: int = 20) -> np.ndarray:
    """Convert Hailo NMS output to Frigate [class,score,ymin,xmin,ymax,xmax]."""
    if isinstance(output, np.ndarray) and output.ndim >= 1 and output.shape[0] == 1:
        output = output[0]
    rows: list[list[float]] = []
    for class_id, detections in enumerate(output):
        array = np.asarray(detections)
        if array.size == 0:
            continue
        array = np.atleast_2d(array)
        for detection in array:
            if detection.size < 5:
                continue
            score = float(detection[4])
            if score >= threshold:
                rows.append([
                    float(class_id), score, float(detection[0]), float(detection[1]),
                    float(detection[2]), float(detection[3]),
                ])
    return _frigate_array(rows, maximum)


def decode_yolo26(outputs: dict[str, np.ndarray], threshold: float, maximum: int = 20,
                  iou_threshold: float = 0.45, input_size: int = 640) -> np.ndarray:
    """Decode Hailo Model-Zoo YOLO26 anchor-free tensors and apply class-aware NMS."""
    bbox_tensors: dict[int, np.ndarray] = {}
    class_tensors: dict[int, np.ndarray] = {}
    for tensor in outputs.values():
        array = np.asarray(tensor)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            continue
        grid, channels = array.shape[0], array.shape[-1]
        if channels == 4:
            bbox_tensors[grid] = array.astype(np.float32, copy=False)
        else:
            class_tensors[grid] = array.astype(np.float32, copy=False)

    boxes_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    classes_all: list[np.ndarray] = []
    for grid in sorted(bbox_tensors, reverse=True):
        if grid not in class_tensors:
            continue
        bbox = bbox_tensors[grid]
        logits = class_tensors[grid]
        height, width = bbox.shape[:2]
        stride = input_size / height
        gy, gx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride
        x1 = np.clip(cx - bbox[..., 0] * stride, 0, input_size) / input_size
        y1 = np.clip(cy - bbox[..., 1] * stride, 0, input_size) / input_size
        x2 = np.clip(cx + bbox[..., 2] * stride, 0, input_size) / input_size
        y2 = np.clip(cy + bbox[..., 3] * stride, 0, input_size) / input_size
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
        scores = probabilities.max(axis=-1)
        classes = probabilities.argmax(axis=-1)
        mask = scores >= threshold
        if mask.any():
            boxes_all.append(np.stack((x1[mask], y1[mask], x2[mask], y2[mask]), axis=-1))
            scores_all.append(scores[mask])
            classes_all.append(classes[mask])

    if not boxes_all:
        return np.zeros((maximum, 6), dtype=np.float32)
    boxes = np.concatenate(boxes_all).astype(np.float32, copy=False)
    scores = np.concatenate(scores_all).astype(np.float32, copy=False)
    classes = np.concatenate(classes_all).astype(np.int32, copy=False)
    keep = _class_aware_nms(boxes, scores, classes, iou_threshold)[:maximum]
    rows = [
        [float(classes[index]), float(scores[index]), float(boxes[index, 1]),
         float(boxes[index, 0]), float(boxes[index, 3]), float(boxes[index, 2])]
        for index in keep
    ]
    return _frigate_array(rows, maximum)


class HailoVisionBackend:
    """One persistent HailoRT configured model for object detection."""

    def __init__(self, settings):
        self.settings = settings
        self.device = self.config_context = self.configured_model = None
        self.infer_model = self.hef = None
        self.path: str | None = None
        self.entry: dict = {}
        self.input_shape: tuple[int, ...] | None = None
        self.output_types: dict[str, str] = {}

    def start(self):
        manager = ModelManager(self.settings, prepare_model_version())
        selection = self.settings.vision_model
        self.path = str(manager.resolve(selection, "vision"))
        if selection in manager.entries:
            self.entry = manager.entry(selection, "vision")
        else:
            self.entry = {"postprocess": "hailo_nms", "labels": "coco80"}

        from hailo_platform import HEF, FormatType, HailoSchedulingAlgorithm, VDevice

        params = VDevice.create_params()
        params.group_id = "SHARED"
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        _LOG.info("Creating resident vision VDevice group_id=SHARED model=%s", self.path)
        self.device = VDevice(params)
        self.hef = HEF(self.path)
        self.infer_model = self.device.create_infer_model(self.path)
        self.infer_model.set_batch_size(1)
        infos = self.hef.get_output_vstream_infos()
        # Raw multi-output models need deterministic host float tensors.  Hailo NMS
        # output is left in its native NMS representation.
        if len(infos) > 1:
            for info in infos:
                self.infer_model.output(info.name).set_format_type(FormatType.FLOAT32)
                self.output_types[info.name] = "float32"
        else:
            for info in infos:
                self.output_types[info.name] = str(info.format.type).split(".")[-1].lower()
        self.config_context = self.infer_model.configure()
        self.configured_model = self.config_context.__enter__()
        self.configured_model.set_scheduler_priority(self.settings.vision_scheduler_priority)
        self.input_shape = tuple(self.hef.get_input_vstream_infos()[0].shape)
        _LOG.info("Resident vision model ready model=%s input_shape=%s postprocess=%s",
                  self.settings.vision_model_id, self.input_shape, self.entry.get("postprocess"))

    def _bindings(self, frame: np.ndarray):
        buffers = {
            name: np.empty(self.infer_model.output(name).shape, dtype=np.dtype(dtype))
            for name, dtype in self.output_types.items()
        }
        bindings = self.configured_model.create_bindings(output_buffers=buffers)
        bindings.input().set_buffer(frame)
        return bindings

    def detect(self, frame: np.ndarray, confidence: float, maximum: int) -> np.ndarray:
        if self.configured_model is None or self.input_shape is None:
            raise RuntimeError("Vision model is not ready")
        height, width = self.input_shape[:2]
        frame = letterbox(frame, width, height)
        bindings = self._bindings(frame)
        self.configured_model.wait_for_async_ready(timeout_ms=int(self.settings.request_timeout * 1000))
        completion_error: list[BaseException] = []

        def completed(info):
            if info.exception:
                completion_error.append(info.exception)

        job = self.configured_model.run_async([bindings], completed)
        job.wait(int(self.settings.request_timeout * 1000))
        if completion_error:
            raise RuntimeError(f"Vision inference failed: {completion_error[0]}")
        names = list(self.output_types)
        if len(names) == 1:
            output = bindings.output(names[0]).get_buffer()
        else:
            output = {name: np.expand_dims(bindings.output(name).get_buffer(), 0) for name in names}
        if self.entry.get("postprocess") == "yolo26_anchor_free" or isinstance(output, dict):
            return decode_yolo26(output, confidence, maximum, self.settings.vision_iou_threshold,
                                 max(height, width))
        return decode_hailo_nms(output, confidence, maximum)

    def close(self):
        if self.config_context is not None:
            try:
                self.config_context.__exit__(None, None, None)
            finally:
                self.config_context = self.configured_model = None
        if self.device is not None:
            try:
                self.device.release()
            finally:
                self.device = None


class VisionRuntime:
    def __init__(self, settings, backend=None):
        self.settings = settings
        self.backend = backend if backend is not None else HailoVisionBackend(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hailo-vision-owner")
        self.ready = False
        self.pending = 0

    async def start(self):
        if not self.settings.vision_enabled:
            return
        await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.start)
        self.ready = True

    async def detect_array(self, frame: np.ndarray, confidence: float | None = None,
                           maximum: int | None = None) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Vision model is disabled or not ready")
        if self.pending >= self.settings.vision_queue_size:
            raise RuntimeError("Vision inference queue is full")
        confidence = self.settings.vision_confidence if confidence is None else confidence
        maximum = self.settings.vision_max_detections if maximum is None else maximum
        self.pending += 1
        future = asyncio.get_running_loop().run_in_executor(
            self.executor, self.backend.detect, frame, confidence, maximum
        )
        future.add_done_callback(lambda _: setattr(self, "pending", max(0, self.pending - 1)))
        return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout)

    async def detect_base64(self, value: str, confidence: float | None = None,
                            maximum: int | None = None):
        try:
            data = decode_base64(value, self.settings.max_body)
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                frame = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid or oversized image") from exc
        result = await self.detect_array(frame, confidence, maximum)
        return frame.shape[1], frame.shape[0], result

    def accepts_model(self, name: str) -> bool:
        configured = self.settings.vision_model
        candidates = {
            configured,
            Path(configured).name,
            Path(configured).stem,
            self.settings.vision_model_id,
            self.settings.vision_model_id + ".hef",
        }
        path = getattr(self.backend, "path", None)
        if path:
            candidates.update({Path(path).name, Path(path).stem})
        return name in candidates or Path(name).name in candidates or Path(name).stem in candidates

    def status(self):
        return {
            "enabled": self.settings.vision_enabled,
            "ready": self.ready,
            "model": self.settings.vision_model_id if self.settings.vision_enabled else None,
            "model_path": getattr(self.backend, "path", None),
            "pending": self.pending,
            "zmq_enabled": bool(self.settings.vision_enabled and self.settings.vision_zmq_enabled),
            "zmq_endpoint": self.settings.vision_zmq_endpoint if self.settings.vision_zmq_enabled else None,
        }

    async def close(self):
        self.ready = False
        try:
            if self.settings.vision_enabled:
                await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.close)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
