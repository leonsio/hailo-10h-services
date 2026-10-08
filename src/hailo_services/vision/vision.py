"""Resident object-detection runtime for Hailo-10H.

The selected HEF is configured once and kept resident. HTTP and ZMQ clients share
this runtime; no per-request VDevice or model configuration is performed.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from hailo_services.runtime.models import ModelManager, prepare_model_version
from hailo_services.shared.errors import BusyError
from hailo_services.shared.media import decode_base64

_LOG = logging.getLogger(__name__)
_VISION_REQUEST_ID = ContextVar("vision_request_id", default="-")

COCO80 = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _strip_batch(frame: np.ndarray) -> np.ndarray:
    """Validate detector RGB shape and make uint8 input contiguous.

    Args:
        frame (np.ndarray): RGB image tensor in HxWx3 or supported singleton-batch form.

    Returns:
        np.ndarray: Unbatched HxWx3 uint8 frame.

    Raises:
        ValueError: Vision input must have shape HxWx3 or 1xHxWx3.
    """
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
    """Resize without distortion and pad using the conventional YOLO value 114.

    Args:
        frame (np.ndarray): RGB image tensor in HxWx3 or supported singleton-batch form.
        width (int): Target width in pixels.
        height (int): Target height in pixels.

    Returns:
        np.ndarray: Contiguous uint8 RGB image at target size, padded with 114.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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


def unletterbox_detections(
    detections: np.ndarray,
    source_width: int,
    source_height: int,
    model_width: int,
    model_height: int,
) -> np.ndarray:
    """Map normalized boxes from the letterboxed model plane back to the source image.

    Args:
        detections (np.ndarray): Detection rows in normalized letterbox coordinates.
        source_width (int): Original image width in pixels.
        source_height (int): Original image height in pixels.
        model_width (int): Letterboxed model input width in pixels.
        model_height (int): Letterboxed model input height in pixels.

    Returns:
        np.ndarray: Copied detection rows normalized to the original source image.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    if detections.size == 0:
        return detections
    result = np.array(detections, dtype=np.float32, copy=True)
    scale = min(model_width / source_width, model_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    pad_x = (model_width - resized_width) // 2
    pad_y = (model_height - resized_height) // 2

    active = result[:, 1] > 0
    if not active.any():
        return result
    rows = result[active]
    rows[:, [3, 5]] = (rows[:, [3, 5]] * model_width - pad_x) / resized_width
    rows[:, [2, 4]] = (rows[:, [2, 4]] * model_height - pad_y) / resized_height
    rows[:, 2:6] = np.clip(rows[:, 2:6], 0.0, 1.0)
    result[active] = rows
    return result


def _frigate_array(rows: list[list[float]], maximum: int) -> np.ndarray:
    """Sort and pad detector rows into Frigate fixed-size output.

    Args:
        rows (list[list[float]]): Detection rows before sorting and padding.
        maximum (int): Maximum detection rows or accepted artifact bytes.

    Returns:
        np.ndarray: Float32 array of shape (maximum, 6).

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    result = np.zeros((maximum, 6), dtype=np.float32)
    if rows:
        ordered = sorted(rows, key=lambda row: row[1], reverse=True)[:maximum]
        result[: len(ordered)] = np.asarray(ordered, dtype=np.float32)
    return result


def decode_hailo_nms(output, threshold: float, maximum: int = 20) -> np.ndarray:
    """Convert Hailo NMS output to Frigate [class,score,ymin,xmin,ymax,xmax].

    Args:
        output (Any): Native detector output or reserved output-token count.
        threshold (float): Minimum fuzzy score or detection confidence for acceptance.
        maximum (int): Maximum detection rows or accepted artifact bytes.

    Returns:
        np.ndarray: Float32 (maximum, 6) Frigate detection tensor.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
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
                rows.append(
                    [
                        float(class_id),
                        score,
                        float(detection[0]),
                        float(detection[1]),
                        float(detection[2]),
                        float(detection[3]),
                    ]
                )
    return _frigate_array(rows, maximum)


def decode_yolo26(
    outputs: dict[str, np.ndarray],
    threshold: float,
    maximum: int = 20,
    input_size: int = 640,
) -> np.ndarray:
    """Decode YOLO26 using the official NMS-free two-stage top-k selection.

    Hailo Model Zoo exposes six raw tensors: three 4-channel LTRB distance maps and
    three class-logit maps. YOLO26 deliberately has no NMS. Its reference postprocess
    first chooses top-k anchors by their best class score, then top-k anchor/class
    pairs from that subset. Multiple classes may therefore refer to the same anchor.

    Args:
        outputs (dict[str, np.ndarray]): Raw native YOLO output tensors keyed by stream name.
        threshold (float): Minimum fuzzy score or detection confidence for acceptance.
        maximum (int): Maximum detection rows or accepted artifact bytes.
        input_size (int): YOLO model input size in pixels.

    Returns:
        np.ndarray: Float32 (maximum, 6) NMS-free top-k detections.

    Raises:
        ValueError: Raw tensors are missing, mispaired or incompatible with YOLO26 decoding.
    """
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
    for grid in sorted(bbox_tensors, reverse=True):
        if grid not in class_tensors:
            continue
        bbox = bbox_tensors[grid]
        logits = class_tensors[grid]
        height, width = bbox.shape[:2]
        if height != width:
            raise ValueError(f"Unsupported non-square YOLO26 output grid: {bbox.shape}")
        stride = input_size / height
        gy, gx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride
        # _yolo6_decode used by Hailo's YOLO26 implementation: the first two
        # channels are left/top distances and the last two are right/bottom.
        x1 = np.clip(cx - bbox[..., 0] * stride, 0, input_size) / input_size
        y1 = np.clip(cy - bbox[..., 1] * stride, 0, input_size) / input_size
        x2 = np.clip(cx + bbox[..., 2] * stride, 0, input_size) / input_size
        y2 = np.clip(cy + bbox[..., 3] * stride, 0, input_size) / input_size
        boxes_all.append(np.stack((y1, x1, y2, x2), axis=-1).reshape(-1, 4))
        scores_all.append(
            (1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))).reshape(-1, logits.shape[-1])
        )

    if not boxes_all:
        return np.zeros((maximum, 6), dtype=np.float32)

    boxes = np.concatenate(boxes_all, axis=0).astype(np.float32, copy=False)
    scores = np.concatenate(scores_all, axis=0).astype(np.float32, copy=False)
    if scores.shape[0] != boxes.shape[0]:
        raise ValueError("YOLO26 box/class output shapes do not match")

    # Ultralytics/Hailo stage 1: select k anchors by the maximum class score.
    k = min(maximum, boxes.shape[0])
    anchor_best = scores.max(axis=1)
    if k < anchor_best.size:
        anchor_indices = np.argpartition(anchor_best, -k)[-k:]
    else:
        anchor_indices = np.arange(anchor_best.size)

    # Stage 2: select k anchor/class pairs from k * class_count scores.
    candidate_scores = scores[anchor_indices]
    flat = candidate_scores.reshape(-1)
    pair_k = min(k, flat.size)
    if pair_k < flat.size:
        pair_indices = np.argpartition(flat, -pair_k)[-pair_k:]
    else:
        pair_indices = np.arange(flat.size)
    pair_indices = pair_indices[np.argsort(flat[pair_indices])[::-1]]
    class_count = scores.shape[1]

    rows: list[list[float]] = []
    for pair_index in pair_indices:
        score = float(flat[pair_index])
        if score <= threshold:
            continue
        anchor_in_topk = int(pair_index // class_count)
        class_id = int(pair_index % class_count)
        box = boxes[int(anchor_indices[anchor_in_topk])]
        rows.append([class_id, score, float(box[0]), float(box[1]), float(box[2]), float(box[3])])
    return _frigate_array(rows, maximum)


class HailoVisionBackend:
    """One persistent HailoRT configured model for object detection."""

    def __init__(self, settings):
        """Initialize HailoVisionBackend configuration and owned dependencies.

        Args:
            settings (Settings): Validated service settings controlling enabled models and limits.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.settings = settings
        self.device = self.config_context = self.configured_model = None
        self.infer_model = self.hef = None
        self.path: str | None = None
        self.entry: dict = {}
        self.input_shape: tuple[int, ...] | None = None
        self.output_types: dict[str, str] = {}

    def start(self):
        """Initialize resident resources or start the configured transport listener.

        Returns:
            None: Marks the service ready after successful initialization.

        Raises:
            OSError: A required model artifact cannot be read or downloaded.
            RuntimeError: Native model initialization fails.
        """
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
        # Raw multi-output models need deterministic host float tensors. Hailo NMS
        # output is left in its native representation.
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
        _LOG.info(
            "Resident vision model ready model=%s input_shape=%s postprocess=%s scheduler_priority=%d batch_size=1 group_id=SHARED",
            self.settings.vision_model_id,
            self.input_shape,
            self.entry.get("postprocess"),
            self.settings.vision_scheduler_priority,
        )

    def _bindings(self, frame: np.ndarray):
        """Allocate native detector outputs and attach the input frame.

        Args:
            frame (np.ndarray): RGB image tensor in HxWx3 or supported singleton-batch form.

        Returns:
            Any: HailoRT inference bindings owning input and output buffers.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        buffers = {
            name: np.empty(self.infer_model.output(name).shape, dtype=np.dtype(dtype))
            for name, dtype in self.output_types.items()
        }
        bindings = self.configured_model.create_bindings(output_buffers=buffers)
        bindings.input().set_buffer(frame)
        return bindings

    def detect(self, frame: np.ndarray, confidence: float, maximum: int) -> np.ndarray:
        """Run resident detector inference and decode its native outputs.

        Args:
            frame (np.ndarray): RGB image tensor in HxWx3 or supported singleton-batch form.
            confidence (float): Detection score threshold; None uses service settings.
            maximum (int): Maximum detection rows or accepted artifact bytes.

        Returns:
            np.ndarray: Rows [class, score, ymin, xmin, ymax, xmax].

        Raises:
            RuntimeError: Vision model is not ready.
        """
        if self.configured_model is None or self.input_shape is None:
            raise RuntimeError("Vision model is not ready")
        started = time.perf_counter()
        height, width = self.input_shape[:2]
        frame = letterbox(frame, width, height)
        bindings = self._bindings(frame)
        prepared_at = time.perf_counter()
        self.configured_model.wait_for_async_ready(
            timeout_ms=int(self.settings.request_timeout * 1000)
        )
        ready_at = time.perf_counter()
        completion_error: list[BaseException] = []

        def completed(completion_info):
            """Capture native detector completion failures for the owner thread.

            Args:
                completion_info (Any): Native Hailo completion metadata including an optional exception.

            Returns:
                None: Appends a native error when the callback reports one.

            Notes:
                No application-specific exceptions are raised for valid inputs.
            """
            if completion_info.exception:
                completion_error.append(completion_info.exception)

        job = self.configured_model.run_async([bindings], completed)
        submitted_at = time.perf_counter()
        job.wait(int(self.settings.request_timeout * 1000))
        completed_at = time.perf_counter()
        _LOG.debug(
            "event=vision_native_phases request_id=%s model=%s scheduler_priority=%d prepare_ms=%.1f async_ready_wait_ms=%.1f submit_ms=%.1f completion_wait_ms=%.1f",
            _VISION_REQUEST_ID.get(),
            self.settings.vision_model_id,
            self.settings.vision_scheduler_priority,
            (prepared_at - started) * 1000,
            (ready_at - prepared_at) * 1000,
            (submitted_at - ready_at) * 1000,
            (completed_at - submitted_at) * 1000,
        )
        if completion_error:
            raise RuntimeError(f"Vision inference failed: {completion_error[0]}")
        names = list(self.output_types)
        if len(names) == 1:
            output = bindings.output(names[0]).get_buffer()
        else:
            output = {name: np.expand_dims(bindings.output(name).get_buffer(), 0) for name in names}
        if self.entry.get("postprocess") == "yolo26_anchor_free" or isinstance(output, dict):
            return decode_yolo26(output, confidence, maximum, max(height, width))
        return decode_hailo_nms(output, confidence, maximum)

    def close(self):
        """Release resources owned by this service or native context.

        Returns:
            None: Closes native resources, connections or owner executors.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
    """Schedule a resident object detector on a dedicated bounded owner thread."""

    def __init__(self, settings, backend=None):
        """Initialize VisionRuntime configuration and owned dependencies.

        Args:
            settings (Settings): Validated service settings controlling enabled models and limits.
            backend (ChatBackend): Resident backend used for generation or context preparation.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.settings = settings
        self.backend = backend if backend is not None else HailoVisionBackend(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hailo-vision-owner")
        self.ready = False
        self.pending = 0

    async def start(self):
        """Initialize resident resources or start the configured transport listener.

        Returns:
            None: Marks the service ready after successful initialization.

        Raises:
            OSError: A required model artifact cannot be read or downloaded.
            RuntimeError: Native model initialization fails.
        """
        if not self.settings.vision_enabled:
            return
        await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.start)
        self.ready = True

    async def detect_array(
        self,
        frame: np.ndarray,
        confidence: float | None = None,
        maximum: int | None = None,
        request_id: str | None = None,
    ) -> np.ndarray:
        """Schedule bounded detector inference on its dedicated owner thread.

        Args:
            frame (np.ndarray): RGB image tensor in HxWx3 or supported singleton-batch form.
            confidence (float | None): Detection score threshold; None uses service settings.
            maximum (int | None): Maximum detection rows or accepted artifact bytes.
            request_id (str | None): Optional transport ID for correlated phase logs.

        Returns:
            np.ndarray: Fixed-size detection tensor in model coordinates.

        Raises:
            RuntimeError: The detector is unavailable or native inference fails.
            BusyError: The detector queue is full.
            asyncio.TimeoutError: Detection exceeds the configured deadline.
        """
        if not self.ready:
            raise RuntimeError("Vision model is disabled or not ready")
        if self.pending >= self.settings.vision_queue_size:
            raise BusyError("Vision inference queue is full")
        confidence = self.settings.vision_confidence if confidence is None else confidence
        maximum = self.settings.vision_max_detections if maximum is None else maximum
        self.pending += 1
        queued_at = time.perf_counter()
        identifier = request_id or uuid.uuid4().hex[:12]

        def detect():
            """Measure detector thread scheduling separately from native execution.

            Returns:
                np.ndarray: Unchanged detector results or the original exception.
            """
            token = _VISION_REQUEST_ID.set(identifier)
            entered = time.perf_counter()
            status = "completed"
            try:
                return self.backend.detect(frame, confidence, maximum)
            except BaseException:
                status = "error"
                raise
            finally:
                elapsed = (time.perf_counter() - entered) * 1000
                _LOG.log(
                    logging.WARNING if elapsed > 500 else logging.DEBUG,
                    "event=vision_execution request_id=%s model=%s scheduler_priority=%d status=%s queue_wait_ms=%.1f execution_ms=%.1f",
                    identifier,
                    self.settings.vision_model_id,
                    self.settings.vision_scheduler_priority,
                    status,
                    (entered - queued_at) * 1000,
                    elapsed,
                )
                _VISION_REQUEST_ID.reset(token)

        future = asyncio.get_running_loop().run_in_executor(self.executor, detect)
        future.add_done_callback(lambda _: setattr(self, "pending", max(0, self.pending - 1)))
        return await asyncio.wait_for(asyncio.shield(future), self.settings.request_timeout)

    async def detect_base64(
        self,
        value: str,
        confidence: float | None = None,
        maximum: int | None = None,
    ):
        """Decode an image, detect objects and map boxes to source coordinates.

        Args:
            value (str): Input value inspected or normalized by this helper.
            confidence (float | None): Detection score threshold; None uses service settings.
            maximum (int | None): Maximum detection rows or accepted artifact bytes.

        Returns:
            tuple[int, int, np.ndarray]: Source width, source height and normalized detections.

        Raises:
            ValueError: The image encoding or pixel data cannot be decoded.
            BusyError: The detector queue is full.
            RuntimeError: The detector is disabled, unready or fails inference.
            asyncio.TimeoutError: Detection exceeds the configured deadline.
        """
        try:
            data = decode_base64(value, self.settings.max_body)
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                frame = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid or oversized image") from exc
        result = await self.detect_array(frame, confidence, maximum)
        source_height, source_width = frame.shape[:2]
        input_shape = getattr(self.backend, "input_shape", None)
        if input_shape and len(input_shape) >= 2:
            result = unletterbox_detections(
                result,
                source_width,
                source_height,
                int(input_shape[1]),
                int(input_shape[0]),
            )
        return source_width, source_height, result

    def accepts_model(self, name: str) -> bool:
        """Match detector IDs, filenames or HEF stems against the configured model.

        Args:
            name (str): Function, attribute, device or model identifier.

        Returns:
            bool: Whether the requested model identifies the resident detector.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
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
        """Summarize resident readiness, model paths and pending requests.

        Returns:
            dict[str, Any]: Serializable health and configuration details.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        return {
            "enabled": self.settings.vision_enabled,
            "ready": self.ready,
            "model": self.settings.vision_model_id if self.settings.vision_enabled else None,
            "model_path": getattr(self.backend, "path", None),
            "pending": self.pending,
            "zmq_enabled": bool(self.settings.vision_enabled and self.settings.vision_zmq_enabled),
            "zmq_endpoint": (
                self.settings.vision_zmq_endpoint if self.settings.vision_zmq_enabled else None
            ),
        }

    async def close(self):
        """Release resources owned by this service or native context.

        Returns:
            None: Closes native resources, connections or owner executors.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.ready = False
        try:
            if self.settings.vision_enabled:
                await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.close)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
