"""Keep official Hailo storage/downloader, explicitly select its model release."""

import logging
import os
import re

_LOG = logging.getLogger(__name__)


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
