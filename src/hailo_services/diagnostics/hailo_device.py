"""Read-only, non-invasive Hailo-10H hardware status for the dashboard."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read(path):
    """Read one optional sysfs attribute.

    Returns:
        str | None: Trimmed value, or None when inaccessible.
    """
    try:
        return path.read_text(encoding="ascii").strip() or None
    except (OSError, UnicodeError):
        return None


def hailo_device_status(sys_root=Path("/sys/bus/pci/devices"), device_root=Path("/dev")):
    """Inspect a Hailo-10H PCI function without opening the active accelerator.

    Returns:
        dict: Available PCI, driver and hwmon attributes; missing values remain null.
    """
    result = {
        "device_node": next((str(path) for path in (device_root / "h1x-0", device_root / "hailo0")
                             if path.exists()), None),
        "pci_address": None,
        "driver": None,
        "link_speed": None,
        "link_width": None,
        "temperature_c": None,
        "hailort_version": None,
    }
    try:
        result["hailort_version"] = version("hailort")
    except PackageNotFoundError:
        pass
    try:
        devices = sorted(sys_root.iterdir())
    except OSError:
        devices = ()
    for device in devices:
        # Hailo PCI vendor 1e60, Hailo-10H product 45c4. Do not report an
        # unrelated Hailo-8 as the Hailo-10H if both are installed.
        if (_read(device / "vendor"), _read(device / "device")) != ("0x1e60", "0x45c4"):
            continue
        result["pci_address"] = device.name
        try:
            result["driver"] = (device / "driver").resolve(strict=True).name
        except OSError:
            pass
        result["link_speed"] = _read(device / "current_link_speed")
        result["link_width"] = _read(device / "current_link_width")
        try:
            sensors = sorted((device / "hwmon").glob("hwmon*/temp*_input"))
        except OSError:
            sensors = ()
        for sensor in sensors:
            raw = _read(sensor)
            try:
                result["temperature_c"] = round(int(raw) / 1000, 1)
                break
            except (ValueError, TypeError):
                continue
        break
    return result
