#!/usr/bin/env bash
set -euo pipefail
if [[ ${EUID} -ne 0 ]]; then
  echo 'Run with sudo bash scripts/install.sh' >&2
  exit 1
fi
SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SERVICE_DIR=/opt/hailo-10h-services
SERVICE_STATE=/var/lib/hailo-10h-services
# Select the Python that can already import your vendor HailoRT wheel.
SERVICE_PYTHON=${HAILO_PYTHON:-python3}
TOOLS_REVISION=891ce701c2ebe239a5d277759eb75a30f76678a9
"${SERVICE_PYTHON}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit('Python >=3.10 required')
from hailo_platform import VDevice
from hailo_platform.genai import VLM, Speech2Text
print('HailoRT GenAI Python API available')
PY
# Do not replace the kernel driver, firmware or vendor HailoRT installation.
apt-get update
apt-get install -y python3-venv git libsndfile1 acl
if systemctl cat hailo-10h-services.service >/dev/null 2>&1; then
  systemctl stop hailo-10h-services.service
fi
id hailo-services >/dev/null 2>&1 || useradd --system --user-group --home-dir "${SERVICE_STATE}" --shell /usr/sbin/nologin hailo-services
# Migrate users created by the original installer with /nonexistent as home.
usermod --home "${SERVICE_STATE}" hailo-services
install -d -o hailo-services -g hailo-services -m 0750 "${SERVICE_STATE}"
install -d -o hailo-services -g hailo-services -m 0750 "${SERVICE_STATE}/.hailo"
for DEVICE_GROUP in hailo video render; do
  if getent group "${DEVICE_GROUP}" >/dev/null; then
    usermod -aG "${DEVICE_GROUP}" hailo-services
  fi
done
install -d -m 0755 "${SERVICE_DIR}"
"${SERVICE_PYTHON}" -m venv --system-site-packages "${SERVICE_DIR}/venv"
# A venv created from another venv sees base-system packages, not necessarily its
# parent's site-packages. Expose the selected vendor wheel if it is otherwise absent.
if ! "${SERVICE_DIR}/venv/bin/python" -c 'import hailo_platform.genai' 2>/dev/null; then
  VENDOR_SITE=$("${SERVICE_PYTHON}" -c 'import pathlib, hailo_platform; print(pathlib.Path(hailo_platform.__file__).resolve().parent.parent)')
  SERVICE_SITE=$("${SERVICE_DIR}/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  printf '%s\n' "${VENDOR_SITE}" > "${SERVICE_SITE}/hailort-vendor.pth"
fi
"${SERVICE_DIR}/venv/bin/pip" install --upgrade pip setuptools wheel
"${SERVICE_DIR}/venv/bin/pip" install "${SOURCE_DIR}" 'PyYAML>=6,<7' 'python-dotenv>=1,<2' 'opencv-python-headless==4.10.0.84'
if [[ ! -f /etc/hailo-10h-services.env ]]; then
  install -m 0600 "${SOURCE_DIR}/deploy/hailo-10h-services.env.example" /etc/hailo-10h-services.env
  SERVICE_KEY=$("${SERVICE_DIR}/venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')
  sed -i "s/^HAILO_API_KEY=$/HAILO_API_KEY=${SERVICE_KEY}/" /etc/hailo-10h-services.env
fi
# LiteRT-LM is optional. Install it into the service's interpreter when a model
# path is configured; a package installed only in another user's venv is not visible.
LITERT_MODEL_PATH=$(sed -n 's/^HAILO_LITERT_MODEL_PATH=//p' /etc/hailo-10h-services.env | tail -n 1)
if [[ -n ${LITERT_MODEL_PATH} && -f ${LITERT_MODEL_PATH} ]]; then
  if ! "${SERVICE_DIR}/venv/bin/python" -c 'from litert_lm import Engine, Tool' 2>/dev/null; then
    "${SERVICE_DIR}/venv/bin/pip" install --upgrade litert-lm
  fi
fi
# Install only the official downloader/config helpers, avoiding unrelated camera capture,
# audio capture, TTS, GStreamer and PyTorch dependency bundles.
if [[ ! -d ${SERVICE_DIR}/hailo-apps/.git ]]; then
  git clone https://github.com/hailo-ai/hailo-apps.git "${SERVICE_DIR}/hailo-apps"
fi
git -C "${SERVICE_DIR}/hailo-apps" fetch origin "${TOOLS_REVISION}"
git -C "${SERVICE_DIR}/hailo-apps" checkout --detach "${TOOLS_REVISION}"
"${SERVICE_DIR}/venv/bin/pip" install --no-deps --no-build-isolation "${SERVICE_DIR}/hailo-apps"
# Grant access to the common store without changing other applications' ownership.
install -d -m 0755 /usr/local/hailo/resources/models/hailo10h
setfacl -m u:hailo-services:rx /usr/local/hailo /usr/local/hailo/resources /usr/local/hailo/resources/models
setfacl -R -m u:hailo-services:rwX /usr/local/hailo/resources/models/hailo10h
setfacl -m d:u:hailo-services:rwx /usr/local/hailo/resources/models/hailo10h
if [[ -n ${LITERT_MODEL_PATH} && -f ${LITERT_MODEL_PATH} ]]; then
  LITERT_DIR=$(dirname -- "${LITERT_MODEL_PATH}")
  while [[ ${LITERT_DIR} != / && ${LITERT_DIR} != /home ]]; do
    setfacl -m u:hailo-services:x "${LITERT_DIR}"
    LITERT_DIR=$(dirname -- "${LITERT_DIR}")
  done
  setfacl -m u:hailo-services:r "${LITERT_MODEL_PATH}"
fi
# Typical Hailo-10H character device name. Preserve existing device groups/modes.
install -d -m 0755 /etc/udev/rules.d
cat > /etc/udev/rules.d/71-hailo-10h-services.rules <<'RULE'
KERNEL=="h1x*", TAG+="uaccess", RUN+="/usr/bin/setfacl -m u:hailo-services:rw /dev/%k"
RULE
# Give access immediately, including hosts/containers without an active udev daemon.
for DEVICE_PATH in /dev/h1x*; do
  if [[ -c ${DEVICE_PATH} ]]; then
    setfacl -m u:hailo-services:rw "${DEVICE_PATH}"
  fi
done
if command -v udevadm >/dev/null; then
  udevadm control --reload-rules || true
fi
# Root reads the env file on behalf of the service; secrets need not be user-readable.
install -m 0644 "${SOURCE_DIR}/deploy/hailo-10h-services.service" /etc/systemd/system/hailo-10h-services.service
# Native Hailo logging can use both HOME/.hailo and the process cwd.
cd -- "${SERVICE_STATE}"
runuser -u hailo-services -- env HOME="${SERVICE_STATE}" "${SERVICE_DIR}/venv/bin/python" - <<'PY'
from hailo_platform import VDevice
from hailo_platform.genai import VLM, Speech2Text
from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.installation.download_resources import download_resources
from hailo_services.app import create_app
from hailo_services.preflight import main
main()
print('Service user imports and resource downloader OK')
PY
systemctl daemon-reload
systemctl reset-failed hailo-10h-services.service || true
systemctl enable --now hailo-10h-services.service
echo 'Installed. Follow startup/downloads with: journalctl -u hailo-10h-services -f'
echo 'Configuration and API key: /etc/hailo-10h-services.env'
echo 'Check readiness: curl http://127.0.0.1:8090/health'
