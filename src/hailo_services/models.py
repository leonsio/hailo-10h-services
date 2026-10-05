"""Independent catalogue and atomic model downloads; no accelerator probes."""

import logging
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import yaml

_LOG = logging.getLogger(__name__)


def ensure_minilm_hef(path, url):
    """Reuse or atomically download the MiniLM HEF into the shared Hailo store."""
    return ensure_model_file(path, url, 1024 * 1024, 100 * 1024 * 1024)


def ensure_model_file(path, url, minimum, maximum, expected_size=None):
    """Download a bounded model asset atomically and reuse a complete local copy."""
    destination = Path(path).expanduser()
    if (destination.is_file() and minimum <= destination.stat().st_size <= maximum
            and (expected_size is None or destination.stat().st_size == expected_size)):
        _LOG.info("Reusing model asset at %s", destination)
        return destination

    _LOG.info("Downloading model asset url=%s destination=%s expected_bytes=%s", url, destination, expected_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "hailo-10h-services/0.1"})
        with os.fdopen(fd, "wb") as output:
            with urllib.request.urlopen(request, timeout=120) as response:
                length = response.headers.get("Content-Length")
                expected = int(length) if length and length.isdecimal() else None
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > maximum:
                        raise RuntimeError(f"Model download exceeds {maximum} bytes")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if (size < minimum or (expected is not None and size != expected)
                or (expected_size is not None and size != expected_size)):
            raise RuntimeError(
                f"Incomplete MiniLM HEF/model download ({size} bytes; expected {expected or minimum})"
            )
        os.replace(temporary_path, destination)
        _LOG.info("Downloaded model asset to %s (%d bytes)", destination, size)
        return destination
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def release_tuple(version):
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Invalid Hailo version: {version}")
    return tuple(int(value) for value in match.groups())


def select_release(runtime_version, available, override=None):
    if override:
        if override not in available:
            raise ValueError(
                f"model_zoo_version={override} is not present in the model catalogue: {available}"
            )
        if release_tuple(override) > release_tuple(runtime_version):
            raise ValueError("Model release is newer than HailoRT")
        return override
    runtime = release_tuple(runtime_version)
    compatible = [
        version
        for version in available
        if release_tuple(version)[0] == runtime[0] and release_tuple(version) <= runtime
    ]
    if not compatible:
        raise ValueError(f"No known model release for HailoRT {runtime_version}")
    return max(compatible, key=release_tuple)


def prepare_model_version():
    import hailo_platform

    version = os.getenv("hailort_version") or getattr(hailo_platform, "__version__", None)
    if not version:
        raise RuntimeError("Cannot detect HailoRT version; set hailort_version")
    release_tuple(version)
    return version


class ModelManager:
    def __init__(self, settings, runtime_version=None):
        self.settings = settings
        path = Path(settings.model_catalog) if settings.model_catalog else Path(__file__).with_name("model_catalog.yaml")
        with path.open(encoding="utf-8") as stream:
            self.catalog = yaml.safe_load(stream)
        if not isinstance(self.catalog, dict) or self.catalog.get("schema_version") != 1:
            raise ValueError("Unsupported model catalogue schema")
        self.entries = self.catalog["models"]
        self.runtime_version = runtime_version

    def entry(self, model, kind=None):
        if model not in self.entries:
            raise ValueError(f"Unknown catalogue model: {model}")
        entry = self.entries[model]
        if kind and entry["kind"] != kind:
            raise ValueError(f"{model} is not a {kind} model")
        return entry

    def url(self, model):
        entry = self.entry(model)
        if "url" in entry:
            return entry["url"]
        version = self.runtime_version or prepare_model_version()
        override = self.settings.model_release
        if override == "auto":
            override = os.getenv("model_zoo_version")
        # Use a documented release for the same HailoRT minor; never silently
        # substitute a newer/older HEF ABI for unknown runtime versions.
        available = [r for r in entry["releases"]
                     if release_tuple(r)[:2] == release_tuple(version)[:2]]
        selected = select_release(version, available, override)
        _LOG.info("Model=%s HailoRT=%s release=%s", model, version, selected)
        return entry["releases"][selected]

    def resolve(self, model, kind=None, path=None):
        candidate = Path(model).expanduser()
        if candidate.suffix == ".hef" or candidate.is_absolute():
            if not candidate.is_file() or candidate.stat().st_size == 0:
                raise FileNotFoundError(f"Configured HEF not found: {candidate}")
            return candidate
        entry = self.entry(model, kind)
        destination = Path(path) if path else Path(self.settings.model_store) / entry["filename"]
        # Resolve compatibility even when a cached named model exists.
        url = self.url(model)
        expected_size = entry.get("sizes", {}).get(next((release for release, link in entry.get("releases", {}).items() if link == url), ""))
        return ensure_model_file(destination, url, entry["minimum_bytes"], entry["maximum_bytes"], expected_size)
