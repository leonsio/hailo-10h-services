"""Keep official Hailo storage/downloader, explicitly select its model release."""

import logging
import os
import re
import tempfile
import urllib.request
from pathlib import Path

_LOG = logging.getLogger(__name__)


def ensure_minilm_hef(path, url):
    """Reuse or atomically download the MiniLM HEF into the shared Hailo store."""
    return ensure_model_file(path, url, 1024 * 1024, 100 * 1024 * 1024)


def ensure_model_file(path, url, minimum, maximum):
    """Download a bounded model asset atomically and reuse a complete local copy."""
    destination = Path(path).expanduser()
    if destination.is_file() and minimum <= destination.stat().st_size <= maximum:
        _LOG.info("Reusing model asset at %s", destination)
        return destination

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
        if size < minimum or (expected is not None and size != expected):
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
                f"model_zoo_version={override} is not supported by installed hailo-apps: {available}"
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
    from hailo_apps.python.core.common.defines import (
        HAILORT_VERSION_KEY,
        MODEL_ZOO_VERSION_KEY,
        VALID_H10_MODEL_ZOO_VERSION,
    )

    # Read the loaded binding's version without opening the accelerator through
    # hailortcli fw-control identify. That probe can time out with active clients.
    version = os.getenv(HAILORT_VERSION_KEY) or getattr(hailo_platform, "__version__", None)
    if not version:
        raise RuntimeError(
            "Cannot detect HailoRT version; set hailort_version in service environment"
        )
    selected = select_release(
        version, VALID_H10_MODEL_ZOO_VERSION, os.getenv(MODEL_ZOO_VERSION_KEY)
    )
    os.environ[MODEL_ZOO_VERSION_KEY] = selected
    _LOG.info("HailoRT=%s; official model release=%s (existing HEFs reused)", version, selected)
    if release_tuple(selected)[:2] != release_tuple(version)[:2]:
        _LOG.warning(
            "No same-minor model release is known to hailo-apps; selected latest older release %s. HEF compatibility will be checked by HailoRT at load time.",
            selected,
        )
