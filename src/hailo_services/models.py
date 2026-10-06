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
    """Reuse or atomically download the MiniLM HEF into the shared Hailo store.

    Args:
        path (str | Path): Filesystem destination or diagnostic schema path.
        url (str): Artifact download URL.

    Returns:
        Path: Complete local MiniLM HEF path.

    Raises:
        OSError: The HEF cannot be downloaded or written.
        RuntimeError: The downloaded artifact is incomplete or oversized.
    """
    return ensure_model_file(path, url, 1024 * 1024, 100 * 1024 * 1024)


def ensure_model_file(path, url, minimum, maximum, expected_size=None):
    """Download a bounded model asset atomically and reuse a complete local copy.

    Args:
        path (str | Path): Filesystem destination or diagnostic schema path.
        url (str): Artifact download URL.
        minimum (int): Minimum accepted model artifact size in bytes.
        maximum (int): Maximum detection rows or accepted artifact bytes.
        expected_size (int | None): Exact catalogue file size in bytes when known.

    Returns:
        Path: Complete cached or downloaded artifact path.

    Raises:
        OSError: The artifact cannot be downloaded or written.
        RuntimeError: Download size is invalid, incomplete or exceeds the bound.
    """
    destination = Path(path).expanduser()
    if (
        destination.is_file()
        and minimum <= destination.stat().st_size <= maximum
        and (expected_size is None or destination.stat().st_size == expected_size)
    ):
        _LOG.info("Reusing model asset at %s", destination)
        return destination

    _LOG.info(
        "Downloading model asset url=%s destination=%s expected_bytes=%s",
        url,
        destination,
        expected_size,
    )
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
        if (
            size < minimum
            or (expected is not None and size != expected)
            or (expected_size is not None and size != expected_size)
        ):
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
    """Parse a semantic HailoRT release string.

    Args:
        version (str): Hailo release in major.minor.patch form.

    Returns:
        tuple[int, int, int]: Major, minor and patch version numbers.

    Raises:
        ValueError: The version is not a major.minor.patch release string.
    """
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Invalid Hailo version: {version}")
    return tuple(int(value) for value in match.groups())


def select_release(runtime_version, available, override=None):
    """Select the newest compatible model release or validate an override.

    Args:
        runtime_version (str | None): Detected HailoRT release; None defers detection.
        available (Iterable[str]): Known model release identifiers.
        override (str | None): Explicit release selection, or None for compatible automatic selection.

    Returns:
        str: Compatible model-zoo release identifier.

    Raises:
        ValueError: Model release is newer than HailoRT.
    """
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
    """Detect HailoRT version before selecting compatible model artifacts.

    Returns:
        str: Validated runtime release string.

    Raises:
        RuntimeError: Cannot detect HailoRT version; set hailort_version.
    """
    import hailo_platform

    version = os.getenv("hailort_version") or getattr(hailo_platform, "__version__", None)
    if not version:
        raise RuntimeError("Cannot detect HailoRT version; set hailort_version")
    release_tuple(version)
    return version


class ModelManager:
    """Resolve catalogue metadata, compatible releases and complete local artifacts."""

    def __init__(self, settings, runtime_version=None):
        """Initialize ModelManager configuration and owned dependencies.

        Args:
            settings (Settings): Validated service settings controlling enabled models and limits.
            runtime_version (str | None): Detected HailoRT release; None defers detection.

        Returns:
            None: Creates the object without running inference.

        Raises:
            OSError: The configured catalogue cannot be read.
            ValueError: The catalogue schema is unsupported.
        """
        self.settings = settings
        path = (
            Path(settings.model_catalog)
            if settings.model_catalog
            else Path(__file__).with_name("model_catalog.yaml")
        )
        with path.open(encoding="utf-8") as stream:
            self.catalog = yaml.safe_load(stream)
        if not isinstance(self.catalog, dict) or self.catalog.get("schema_version") != 1:
            raise ValueError("Unsupported model catalogue schema")
        self.entries = self.catalog["models"]
        self.runtime_version = runtime_version

    def entry(self, model, kind=None):
        """Find a model catalogue entry and validate its expected role.

        Args:
            model (str): Public model identifier or configured HEF path.
            kind (str): Model role, measurement category or environment query kind.

        Returns:
            dict[str, Any]: Model metadata including kind, artifact and releases.

        Raises:
            ValueError: The model is absent from the catalogue or has a different role.
        """
        if model not in self.entries:
            raise ValueError(f"Unknown catalogue model: {model}")
        entry = self.entries[model]
        if kind and entry["kind"] != kind:
            raise ValueError(f"{model} is not a {kind} model")
        return entry

    def url(self, model):
        """Resolve the artifact URL matching configured role and runtime release.

        Args:
            model (str): Public model identifier or configured HEF path.

        Returns:
            str: Compatible download URL.

        Raises:
            ValueError: The model or runtime/release combination is unsupported.
            RuntimeError: HailoRT version cannot be detected.
        """
        entry = self.entry(model)
        if "url" in entry:
            return entry["url"]
        version = self.runtime_version or prepare_model_version()
        role_release = {
            "vlm": self.settings.vlm_release,
            "llm": self.settings.hailo_llm_release,
            "vision": self.settings.vision_release,
        }.get(entry["kind"], "auto")
        override = role_release if role_release != "auto" else self.settings.model_release
        if override == "auto":
            override = os.getenv("model_zoo_version")
        # Use a documented release for the same HailoRT minor; never silently
        # substitute a newer/older HEF ABI for unknown runtime versions.
        available = [
            r for r in entry["releases"] if release_tuple(r)[:2] == release_tuple(version)[:2]
        ]
        if not available:
            raise ValueError(f"No known model release for HailoRT {version}")
        if not override and entry.get("preferred_release"):
            override = entry["preferred_release"]
        selected = select_release(version, entry["releases"] if override else available, override)
        _LOG.info("Model=%s HailoRT=%s release=%s", model, version, selected)
        return entry["releases"][selected]

    def resolve(self, model, kind=None, path=None):
        """Reuse a configured file or download a compatible named model artifact.

        Args:
            model (str): Public model identifier or configured HEF path.
            kind (str): Model role, measurement category or environment query kind.
            path (str | Path): Filesystem destination or diagnostic schema path.

        Returns:
            Path: Existing complete model artifact path.

        Raises:
            ValueError: The catalogue model, role or compatible release is invalid.
            FileNotFoundError: An explicitly configured model file is missing or empty.
            OSError: Artifact download or filesystem access fails.
            RuntimeError: The downloaded artifact fails size validation.
        """
        candidate = Path(model).expanduser()
        if candidate.suffix == ".hef" or candidate.is_absolute():
            if not candidate.is_file() or candidate.stat().st_size == 0:
                raise FileNotFoundError(f"Configured HEF not found: {candidate}")
            return candidate
        entry = self.entry(model, kind)
        destination = Path(path) if path else Path(self.settings.model_store) / entry["filename"]
        # Resolve compatibility even when a cached named model exists.
        url = self.url(model)
        if entry.get("preferred_release") and path is None:
            # Separate releases so selecting the smaller Qwen2 never overwrites
            # a user's existing HEF or accidentally reuses a larger release.
            release = next(r for r, link in entry["releases"].items() if link == url)
            destination = Path(self.settings.model_store) / release / entry["filename"]
        expected_size = entry.get("sizes", {}).get(
            next(
                (release for release, link in entry.get("releases", {}).items() if link == url), ""
            )
        )
        return ensure_model_file(
            destination, url, entry["minimum_bytes"], entry["maximum_bytes"], expected_size
        )
