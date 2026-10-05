#!/usr/bin/env bash
# Run on the Proxmox host; creates a NEW Debian 13 ARM64 container.
set -euo pipefail
usage() {
  cat <<'HELP'
Usage: bash scripts/install-proxmox-lxc.sh --ctid ID --template STORAGE:vztmpl/FILE [options]
  --storage NAME       Rootfs storage (default: local-lvm)
  --bridge NAME        Network bridge (default: vmbr0; DHCP)
  --device PATH        Explicit Hailo character device (otherwise auto-detected)
  --deb PATH           Default: /root/hailort_5.4.0_arm64.deb
  --wheel PATH         Default: /root/hailort-5.4.0-cp313-cp313-linux_aarch64.whl
  --memory MB          Default: 4096 (increase for CPU Gemma)
  --cores N            Default: 4
  --disk GB            Default: 32
  --hostname NAME      Default: hailo-services
  --ref REF            Repository ref (default: main)
  --check              Validate host inputs without creating a container
HELP
}
CTID='' TEMPLATE='' STORAGE=local-lvm BRIDGE=vmbr0 DEVICE=''
DEB=/root/hailort_5.4.0_arm64.deb
WHEEL=/root/hailort-5.4.0-cp313-cp313-linux_aarch64.whl
MEMORY=4096 CORES=4 DISK=32 HOSTNAME_CT=hailo-services REF=main CHECK=0
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --check) CHECK=1; shift; continue ;;
    --ctid|--template|--storage|--bridge|--device|--deb|--wheel|--memory|--cores|--disk|--hostname|--ref)
      (($# >= 2)) || { echo "Missing value: $1" >&2; exit 1; }
      case "$1" in
        --ctid) CTID=$2;; --template) TEMPLATE=$2;; --storage) STORAGE=$2;;
        --bridge) BRIDGE=$2;; --device) DEVICE=$2;; --deb) DEB=$2;;
        --wheel) WHEEL=$2;; --memory) MEMORY=$2;; --cores) CORES=$2;;
        --disk) DISK=$2;; --hostname) HOSTNAME_CT=$2;; --ref) REF=$2;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
[[ $EUID == 0 ]] || { echo 'Run on the Proxmox host as root' >&2; exit 1; }
[[ $(uname -m) == aarch64 ]] || { echo 'This installer requires an ARM64 Proxmox host' >&2; exit 1; }
for command in pct pvesm ip; do command -v "$command" >/dev/null; done
[[ $CTID =~ ^[1-9][0-9]{2,}$ && -n $TEMPLATE ]] || { usage; exit 1; }
for number in "$MEMORY" "$CORES" "$DISK"; do
  [[ $number =~ ^[1-9][0-9]*$ ]] || { echo 'Resources must be positive integers' >&2; exit 1; }
done
[[ $REF != -* ]] || { echo 'Invalid repository ref' >&2; exit 1; }
[[ ! -e /etc/pve/lxc/$CTID.conf && ! -e /etc/pve/qemu-server/$CTID.conf ]] || { echo "CT/VM $CTID already exists; refusing to overwrite" >&2; exit 1; }
[[ -f $DEB && -f $WHEEL && $WHEEL == *cp313*linux_aarch64.whl ]] || { echo 'Missing runtime DEB or CPython 3.13 ARM64 wheel' >&2; exit 1; }
# Fail before creation for driver packages or incompatible runtime.
[[ $(dpkg-deb -f "$DEB" Architecture) == arm64 ]] || { echo 'DEB must be ARM64' >&2; exit 1; }
PACKAGE=$(dpkg-deb -f "$DEB" Package)
[[ $PACKAGE == hailort || $PACKAGE == h10-hailort ]] || { echo 'Supply HailoRT userspace DEB, not a driver/DKMS package' >&2; exit 1; }
[[ $(dpkg-deb -f "$DEB" Version) == 5.4.0* && $WHEEL == *5.4.0* ]] || { echo 'This recipe requires matching HailoRT 5.4.0 packages' >&2; exit 1; }
TEMPLATE_PATH=$(pvesm path "$TEMPLATE")
[[ -f $TEMPLATE_PATH && $TEMPLATE == *debian-13*arm64* ]] || { echo 'Supply an existing Debian 13 ARM64 template volume' >&2; exit 1; }
ip link show "$BRIDGE" >/dev/null
pvesm status --storage "$STORAGE"
if [[ -z $DEVICE ]]; then
  shopt -s nullglob
  DEVICES=()
  for candidate in /dev/h1x-* /dev/hailo[0-9]*; do
    [[ ! -c $candidate ]] || DEVICES+=("$candidate")
  done
  ((${#DEVICES[@]} == 1)) || { echo 'Specify --device: no unique Hailo device found' >&2; exit 1; }
  DEVICE=${DEVICES[0]}
fi
[[ -c $DEVICE ]] || { echo 'Hailo device is not a character device' >&2; exit 1; }
echo "Validated: CT $CTID, $DEVICE, Debian 13 ARM64, HailoRT 5.4.0"
((CHECK == 0)) || exit 0
# Keep a failed container for diagnostics; never destroy existing data automatically.
trap 'echo "Installation failed. Inspect CT $CTID with pct status/exec; it has not been deleted." >&2' ERR
pct create "$CTID" "$TEMPLATE" --hostname "$HOSTNAME_CT" \
  --rootfs "$STORAGE:$DISK" --memory "$MEMORY" --swap 512 --cores "$CORES" \
  --unprivileged 1 --net0 "name=eth0,bridge=$BRIDGE,ip=dhcp" --onboot 1
# Proxmox manages ownership/device cgroup permissions for the unprivileged CT.
pct set "$CTID" --dev0 "path=$DEVICE,uid=0,gid=0,mode=0660"
pct start "$CTID"
pct exec "$CTID" -- mkdir -p /root/hailo-packages
pct push "$CTID" "$DEB" /root/hailo-packages/runtime.deb
pct push "$CTID" "$WHEEL" "/root/hailo-packages/$(basename "$WHEEL")"
pct exec "$CTID" -- bash -s -- "$REF" "$DEVICE" <<'CONTAINER'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# DHCP/network may not be ready immediately after pct start.
for attempt in {1..30}; do
  if getent hosts deb.debian.org >/dev/null; then break; fi
  sleep 2
done
apt-get update
apt-get install -y git ca-certificates python3 python3-venv python3-pip \
  libsndfile1 libgomp1 acl curl /root/hailo-packages/runtime.deb
ldconfig
python3 -m venv /opt/hailort-venv
/opt/hailort-venv/bin/pip install /root/hailo-packages/*.whl
/opt/hailort-venv/bin/python -c 'from hailo_platform.genai import VLM, Speech2Text'
hailortcli --version
hailortcli fw-control identify
git clone https://github.com/leonsio/hailo-10h-services.git /opt/hailo-10h-services-source
git -C /opt/hailo-10h-services-source checkout "$1"
# Also support vendor device names other than h1x; grant the service account access.
id hailo-services >/dev/null 2>&1 || useradd --system --user-group --home-dir /var/lib/hailo-10h-services --shell /usr/sbin/nologin hailo-services
setfacl -m u:hailo-services:rw "$2"
HAILO_PYTHON=/opt/hailort-venv/bin/python bash /opt/hailo-10h-services-source/scripts/install.sh
CONTAINER
echo "Installed in CT $CTID. Configuration: /etc/hailo-10h-services.yaml"
echo "Logs: pct exec $CTID -- journalctl -u hailo-10h-services -f"
echo "Readiness: pct exec $CTID -- curl http://127.0.0.1:8090/health"
