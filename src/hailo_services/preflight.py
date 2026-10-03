"""Check actual writes inside the systemd mount namespace, before Hailo imports."""

import os
import tempfile
from pathlib import Path


def check_writable_directory(path: Path):
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
    home = Path(os.path.expanduser("~"))
    try:
        hailo_home = home / ".hailo"
        hailo_home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create Hailo home {home / '.hailo'}: {exc}. Set a writable service HOME."
        ) from exc
    # No HailoRT import here: it attempts to open its own log on import.
    for directory in (hailo_home, Path.cwd(), Path("/usr/local/hailo/resources/models/hailo10h")):
        check_writable_directory(directory)
    print("Preflight OK: Hailo home, working directory and model store are writable", flush=True)


if __name__ == "__main__":
    main()
