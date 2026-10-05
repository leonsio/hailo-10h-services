#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo 'Run with sudo bash scripts/update.sh' >&2
  exit 1
fi

SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SERVICE_DIR=/opt/hailo-10h-services
VENV=${SERVICE_DIR}/venv
SERVICE=hailo-10h-services.service

if [[ ! -x ${VENV}/bin/python || ! -x ${VENV}/bin/pip ]]; then
  echo "Existing installation not found in ${SERVICE_DIR}. Run scripts/install.sh first." >&2
  exit 1
fi

if command -v git >/dev/null 2>&1 && git -C "${SOURCE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  COMMIT=$(git -C "${SOURCE_DIR}" rev-parse HEAD)
  DIRTY=$(git -C "${SOURCE_DIR}" status --porcelain)
  echo "Updating from repository commit: ${COMMIT}"
  if [[ -n ${DIRTY} ]]; then
    echo 'Warning: the working tree contains local changes; those changes will be installed too.' >&2
  fi
else
  COMMIT=unknown
  echo 'Updating from a source tree without Git metadata.'
fi

if systemctl cat "${SERVICE}" >/dev/null 2>&1; then
  systemctl stop "${SERVICE}"
fi

# Reinstall the application package even when its Python package version did not
# change. --no-deps deliberately leaves the known-good runtime dependency set
# untouched while replacing all hailo_services package files from this checkout.
"${VENV}/bin/pip" install --upgrade --force-reinstall --no-deps "${SOURCE_DIR}"

# Keep the unit file in sync with the repository as part of every update.
install -m 0644 "${SOURCE_DIR}/deploy/hailo-10h-services.service" \
  /etc/systemd/system/hailo-10h-services.service
systemctl daemon-reload

verify_module() {
  local source_file=$1
  local module_name=$2
  local installed_file
  installed_file=$("${VENV}/bin/python" -c \
    "import importlib; print(importlib.import_module('${module_name}').__file__)")
  if ! cmp -s "${source_file}" "${installed_file}"; then
    echo "Installed module differs from repository: ${module_name}" >&2
    echo "  repository: ${source_file}" >&2
    echo "  installed:  ${installed_file}" >&2
    exit 1
  fi
  echo "Verified ${module_name}: ${installed_file}"
}

verify_module "${SOURCE_DIR}/src/hailo_services/ha_routing.py" hailo_services.ha_routing
verify_module "${SOURCE_DIR}/src/hailo_services/ha_prompt_compiler.py" hailo_services.ha_prompt_compiler

systemctl reset-failed "${SERVICE}" || true
systemctl start "${SERVICE}"
systemctl is-active --quiet "${SERVICE}"

echo "Hailo-10H-Services updated successfully (${COMMIT})."
echo "Check startup with: journalctl -u ${SERVICE} -n 80 --no-pager"
echo "Check readiness with: curl http://127.0.0.1:8090/health"
