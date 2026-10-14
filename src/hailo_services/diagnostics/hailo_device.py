"""Cached HailoRT device measurements, collected independently from inference."""

import logging
import math
import threading
from datetime import datetime, timezone

_LOG = logging.getLogger(__name__)
_FIELDS = ("nnc_percent", "cpu_percent", "ram_used_bytes", "ram_total_bytes", "temperature_c", "throttling")


def empty_telemetry(state="unavailable"):
    """Create a hardware snapshot without inventing missing measurements.

    Returns:
        dict: Nullable hardware metrics and collection metadata.
    """
    return {**dict.fromkeys(_FIELDS), "state": state, "sampled_at": None, "errors": {}}


def _number(value):
    """Normalize HailoRT's unavailable sentinel and nonfinite values.

    Returns:
        float | None: Finite nonnegative measurement, or None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _query(name, call, errors, unavailable):
    """Read one API family and remember unsupported methods for this session.

    Returns:
        object | None: API result, or None when the query failed.
    """
    if name in unavailable:
        errors[name] = unavailable[name]
        return None
    try:
        return call()
    except Exception as exc:
        errors[name] = str(exc)
        if isinstance(exc, AttributeError) or any(code in str(exc) for code in ("HAILO_INVALID_OPERATION", "HAILO_NOT_IMPLEMENTED", "HAILO_UNSUPPORTED_OPCODE", "HAILO_UNSUPPORTED_FW_VERSION")):
            unavailable[name] = str(exc)
        return None


def _throttling(control):
    """Distinguish enabled protection from an actual active throttling level.

    Returns:
        bool | None: Actual throttling, or None when only protection is known.
    """
    if control.get_throttling_state() is False:
        return False
    health = control._get_health_information()
    level = getattr(health, "current_temperature_throttling_level", None)
    zone = getattr(health, "current_overcurrent_zone", None)
    if isinstance(level, int) and isinstance(zone, int):
        return level >= 0 or zone > 0
    return None


def read_device(control, unavailable=None):
    """Query supported performance and control APIs on a physical device.

    Returns:
        dict: Device percentages, RAM in bytes, temperature and throttling status.
    """
    result = empty_telemetry()
    unavailable = unavailable if unavailable is not None else {}
    stats = _query("performance", lambda: control.query_performance_stats(sampling_period_ms=100), result["errors"], unavailable)
    if stats is not None:
        for target, name in (("nnc_percent", "nnc_utilization"), ("cpu_percent", "cpu_utilization")):
            value = _number(getattr(stats, name, None))
            result[target] = value if value is not None and value <= 100 else None
        for target, name in (("ram_used_bytes", "ram_size_used"), ("ram_total_bytes", "ram_size_total")):
            value = _number(getattr(stats, name, None))
            result[target] = int(value * 1024) if value is not None else None  # HailoRT returns KiB.
    temperatures = _query("temperature", lambda: control.get_chip_temperature(), result["errors"], unavailable)
    if temperatures is not None:
        values = [_number(getattr(temperatures, name, None)) for name in ("ts0_temperature", "ts1_temperature")]
        values = [value for value in values if value is not None]
        result["temperature_c"] = max(values) if values else None
    result["throttling"] = _query("throttling", lambda: _throttling(control), result["errors"], unavailable)
    result["sampled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    measured = sum(result[name] is not None for name in _FIELDS)
    result["state"] = "ready" if measured == len(_FIELDS) else "partial" if measured else "unavailable"
    return result


def resident_models(chat, vision, settings):
    """List native model objects successfully loaded by this service.

    Returns:
        list: Resident model names, roles and loaded states from live backend objects.
    """
    models = []
    for role, attribute, name in (
        ("VLM", "vlm", settings.vlm_model),
        ("LLM", "llm", settings.hailo_llm_model_id),
        ("Whisper", "whisper", settings.whisper_hef.rsplit("/", 1)[-1].removesuffix(".hef")),
        ("MiniLM", "minilm", "MiniLM-L6"),
    ):
        if getattr(chat, attribute, None) is not None:
            models.append({"name": name, "role": role, "state": "loaded"})
    if getattr(vision, "configured_model", None) is not None:
        models.append({"name": settings.vision_model_id, "role": "YOLO", "state": "loaded"})
    return models


class HailoTelemetry:
    """Sample physical Hailo devices with a separate read-only control handle."""

    def __init__(self, publish=None, interval=5):
        """Initialize a lightweight background collector.

        Returns:
            None: Initializes the last snapshot and owned thread.
        """
        self.publish = publish
        self.interval = interval
        self.stop = threading.Event()
        self.thread = None
        self.latest = empty_telemetry()

    def start(self, vdevice):
        """Start sampling the physical device underlying the resident VDevice.

        Returns:
            None: Starts one collector thread when a device exists.
        """
        if self.thread is not None:
            return
        if vdevice is None:
            self.latest = empty_telemetry()
            if self.publish is not None:
                self.publish(self.latest)
            return
        self.latest = empty_telemetry("pending")
        self.thread = threading.Thread(target=self._run, args=(vdevice,), name="hailo-device-telemetry", daemon=True)
        self.thread.start()

    def _run(self, vdevice):
        """Own, sample and release the read-only physical device handle.

        Returns:
            None: Collects bounded snapshots until shutdown.
        """
        devices = []
        try:
            devices = vdevice.get_physical_devices()
            if not devices:
                raise RuntimeError("No physical Hailo device for telemetry")
            control = devices[0].control
            unavailable = {}
            while not self.stop.is_set():
                self.latest = read_device(control, unavailable)
                if self.publish is not None:
                    self.publish(self.latest)
                self.stop.wait(self.interval)
        except Exception as exc:
            self.latest = {**empty_telemetry(), "errors": {"device": str(exc)}}
            if self.publish is not None:
                self.publish(self.latest)
            _LOG.warning("Hailo device telemetry unavailable: %s", exc)
        finally:
            for device in devices:
                try:
                    device.release()
                except Exception:
                    _LOG.debug("Hailo telemetry handle cleanup failed", exc_info=True)

    def snapshot(self):
        """Return the latest immutable-by-replacement snapshot without native calls.

        Returns:
            dict: Last cached hardware measurement.
        """
        return dict(self.latest)

    def close(self):
        """Stop background sampling before resident device resources are released.

        Returns:
            None: Requests shutdown and briefly joins the collector.
        """
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
