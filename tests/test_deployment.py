"""Exercise host installer orchestration without creating containers/devices."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def host_command(tmp_path, *, version="5.4.0", check=False):
    if os.geteuid() != 0:
        pytest.skip("Installer requires root, even with mocked Proxmox commands")
    commands = tmp_path / "bin"
    commands.mkdir()
    template = tmp_path / "debian-13-standard_arm64.tar.zst"
    template.touch()
    deb = tmp_path / "runtime.deb"
    deb.touch()
    wheel = tmp_path / "hailort-5.4.0-cp313-cp313-linux_aarch64.whl"
    wheel.touch()
    log = tmp_path / "pct.log"
    mocks = {
        "uname": "echo aarch64",
        "ip": "exit 0",
        "dpkg-deb": (
            'case "$3" in Architecture) echo arm64;; Package) echo hailort;; '
            f'Version) echo {version};; esac'
        ),
        "pvesm": f'if [ "$1" = path ]; then echo "{template}"; fi',
        "pct": f'echo "$*" >> "{log}"\nif [ "$1" = exec ]; then cat >/dev/null; fi',
    }
    for name, body in mocks.items():
        path = commands / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    args = ["bash", str(ROOT / "scripts/install-proxmox-lxc.sh"),
            "--ctid", "999999", "--template", "local:vztmpl/debian-13-standard_arm64.tar.zst",
            "--deb", str(deb), "--wheel", str(wheel), "--device", "/dev/null"]
    if check:
        args.append("--check")
    result = subprocess.run(args, env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
                            text=True, capture_output=True, timeout=10)
    return result, log


def test_lxc_check_does_not_mutate(tmp_path):
    result, log = host_command(tmp_path, check=True)
    assert result.returncode == 0, result.stderr
    assert not log.exists()


def test_lxc_rejects_mismatched_runtime_before_create(tmp_path):
    result, log = host_command(tmp_path, version="5.3.0")
    assert result.returncode != 0
    assert "matching HailoRT 5.4.0" in result.stderr
    assert not log.exists()


def test_lxc_creates_unprivileged_device_before_start(tmp_path):
    result, log = host_command(tmp_path)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert calls[0].startswith("create 999999 ")
    assert "--unprivileged 1" in calls[0]
    assert calls[1] == "set 999999 --dev0 path=/dev/null,uid=0,gid=0,mode=0660"
    assert calls[2] == "start 999999"
    assert any("runtime.deb" in call and call.startswith("push ") for call in calls)
    assert calls[-1] == "exec 999999 -- bash -s -- main /dev/null"
