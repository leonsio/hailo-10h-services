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
apt-get install -y python3-venv libsndfile1 acl
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
"${SERVICE_DIR}/venv/bin/pip" install "${SOURCE_DIR}"
if [[ ! -f /etc/hailo-10h-services.yaml ]]; then
  install -o root -g hailo-services -m 0640 "${SOURCE_DIR}/deploy/hailo-10h-services.yaml.example" /etc/hailo-10h-services.yaml
  SERVICE_KEY=$("${SERVICE_DIR}/venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')
  sed -i "s/api_key: \"\"/api_key: \"${SERVICE_KEY}\"/" /etc/hailo-10h-services.yaml
fi
chown root:hailo-services /etc/hailo-10h-services.yaml
chmod 0640 /etc/hailo-10h-services.yaml
# Existing ENV files remain optional overrides; new installations use YAML.
# The small catalogue is package data; never clone/install a helper repository.
LITERT_MODEL_PATH=$(HAILO_CONFIG=/etc/hailo-10h-services.yaml "${SERVICE_DIR}/venv/bin/python" - <<'CONFIGPY'
import os
from hailo_services.config import Settings
legacy = '/etc/hailo-10h-services.env'
if os.path.isfile(legacy):
    for line in open(legacy):
        key, sep, value = line.strip().partition('=')
        if sep and key.startswith('HAILO_'):
            os.environ[key] = value.strip('"').strip("'")
s = Settings.from_env()
print(s.litert_model_path)
if s.litert_enabled or s.litert_model_path:
    print('enabled')
CONFIGPY
)
if [[ ${LITERT_MODEL_PATH} == *enabled ]]; then
  "${SERVICE_DIR}/venv/bin/pip" install --upgrade litert-lm
  LITERT_MODEL_PATH=${LITERT_MODEL_PATH%$'\nenabled'}
  [[ ${LITERT_MODEL_PATH} != enabled ]] || LITERT_MODEL_PATH=''
fi
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
runuser -u hailo-services -- env HOME="${SERVICE_STATE}" HAILO_CONFIG=/etc/hailo-10h-services.yaml "${SERVICE_DIR}/venv/bin/python" - <<'PY'
from hailo_platform import VDevice
from hailo_platform.genai import VLM, Speech2Text
from hailo_services.models import ModelManager
from hailo_services.app import create_app
from hailo_services.preflight import main
main()
print('Service user imports and model manager OK')
PY
systemctl daemon-reload
systemctl reset-failed hailo-10h-services.service || true
systemctl enable --now hailo-10h-services.service
echo 'Installed. Follow startup/downloads with: journalctl -u hailo-10h-services -f'
echo 'Configuration and API key: /etc/hailo-10h-services.yaml (legacy ENV overrides remain supported)'
echo 'Check readiness: curl http://127.0.0.1:8090/health'
