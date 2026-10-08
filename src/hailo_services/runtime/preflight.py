"""Check actual writes inside the systemd mount namespace, before Hailo imports."""

import os
import tempfile
from pathlib import Path


def check_writable_directory(path: Path):
    """Verify file creation and atomic rename in a service directory.

    Args:
        path (Path): Filesystem destination or diagnostic schema path.

    Returns:
        None: Removes temporary probes after the write check.

    Raises:
        RuntimeError: The service user cannot create or atomically rename a probe file.
    """
    original = renamed = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path, prefix=".hailo-preflight-", delete=False
        ) as probe:
            original = Path(probe.name)
            probe.write(b"write check")
        renamed = original.with_suffix(".renamed")
        original.rename(renamed)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot write/rename in {path}: {exc}. "
            "Check service-user permissions and systemd ProtectSystem/ReadWritePaths."
        ) from exc
    finally:
        for candidate in (original, renamed):
            if candidate is not None:
                candidate.unlink(missing_ok=True)


def main():
    """Start the command-line entry point for this module.

    Returns:
        None: Runs the configured command until completion.

    Raises:
        RuntimeError: The Hailo home directory cannot be created.
    """
    home = Path(os.path.expanduser("~"))
    try:
        hailo_home = home / ".hailo"
        hailo_home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create Hailo home {home / '.hailo'}: {exc}. Set a writable service HOME."
        ) from exc
    from hailo_services.config import Settings

    settings = Settings.from_env()
    model_store = Path(settings.model_store)
    model_store.mkdir(parents=True, exist_ok=True)
    # No HailoRT import here: it attempts to open its own log on import.
    for directory in (hailo_home, Path.cwd(), model_store):
        check_writable_directory(directory)
    print("Preflight OK: Hailo home, working directory and model store are writable", flush=True)


if __name__ == "__main__":
    main()
