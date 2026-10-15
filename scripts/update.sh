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

git_repo() {
  # When the updater was started through sudo, perform Git operations as the
  # invoking user. This avoids changing checkout ownership and keeps that
  # user's Git credentials/configuration available. A root shell falls back to
  # root and explicitly trusts only this checkout for the current command.
  if [[ -n ${SUDO_USER:-} && ${SUDO_USER} != root ]] && command -v sudo >/dev/null 2>&1; then
    sudo -u "${SUDO_USER}" -H -- git -C "${SOURCE_DIR}" "$@"
  else
    git -c "safe.directory=${SOURCE_DIR}" -C "${SOURCE_DIR}" "$@"
  fi
}

if command -v git >/dev/null 2>&1 && git_repo rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  BEFORE_COMMIT=$(git_repo rev-parse HEAD)
  DIRTY=$(git_repo status --porcelain)
  BRANCH=$(git_repo symbolic-ref --quiet --short HEAD || true)
  UPSTREAM=$(git_repo rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)

  if [[ -z ${BRANCH} || -z ${UPSTREAM} ]]; then
    echo 'The source checkout has no active branch with an upstream; automatic git pull is not possible.' >&2
    echo 'Switch to a tracking branch or update this checkout manually, then run the updater again.' >&2
    exit 1
  fi

  echo "Updating source repository: ${BRANCH} <- ${UPSTREAM}"
  if [[ -n ${DIRTY} ]]; then
    echo 'Warning: the working tree contains local changes. Git will not overwrite conflicting changes;' >&2
    echo 'non-conflicting local changes remain in place and will be installed too.' >&2
  fi

  # Fast-forward only prevents the updater from creating an implicit merge or
  # rewriting local commits. Pull failures abort before the installed service is
  # changed, so repository problems can be resolved safely first.
  git_repo pull --ff-only

  COMMIT=$(git_repo rev-parse HEAD)
  if [[ ${BEFORE_COMMIT} == "${COMMIT}" ]]; then
    echo "Repository already up to date at ${COMMIT}."
  else
    echo "Repository updated: ${BEFORE_COMMIT} -> ${COMMIT}"

    # If the pull changed the updater itself, restart it once so the remainder
    # of this update uses the freshly pulled implementation as well.
    if [[ ${HAILO_UPDATE_REEXEC:-0} != 1 ]] && \
       ! git_repo diff --quiet "${BEFORE_COMMIT}" "${COMMIT}" -- scripts/update.sh; then
      echo 'The updater changed; restarting with the updated script.'
      export HAILO_UPDATE_REEXEC=1
      exec bash "${SOURCE_DIR}/scripts/update.sh"
    fi
  fi
else
  COMMIT=unknown
  echo 'Updating from a source tree without Git metadata; repository synchronization is skipped.'
fi

if systemctl cat "${SERVICE}" >/dev/null 2>&1; then
  systemctl stop "${SERVICE}"
fi

# Migrate older /etc configurations before activating the writable state path.
install -d -o hailo-services -g hailo-services -m 0750 /var/lib/hailo-10h-services
if [[ ! -L /etc/hailo-10h-services.yaml ]]; then
  install -o hailo-services -g hailo-services -m 0600 /etc/hailo-10h-services.yaml /var/lib/hailo-10h-services/config.yaml
fi
ln -sfn /var/lib/hailo-10h-services/config.yaml /etc/hailo-10h-services.yaml

# Reinstall the application package even when its Python package version did not
# change. --no-deps deliberately leaves the known-good runtime dependency set
# untouched while replacing all hailo_services package files from this checkout.
"${VENV}/bin/pip" install --upgrade --force-reinstall --no-deps "${SOURCE_DIR}"

if HAILO_CONFIG=/etc/hailo-10h-services.yaml "${VENV}/bin/python" -c \
  'from hailo_services.config import Settings; raise SystemExit(0 if Settings.from_env().piper_enabled else 1)'; then
  "${VENV}/bin/pip" install 'piper-tts>=1.3,<2'
fi

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

verify_module "${SOURCE_DIR}/src/hailo_services/assistants/ha/ha_routing.py" hailo_services.assistants.ha.ha_routing
verify_module "${SOURCE_DIR}/src/hailo_services/assistants/ha/ha_prompt_compiler.py" hailo_services.assistants.ha.ha_prompt_compiler
verify_module "${SOURCE_DIR}/src/hailo_services/assistants/ha/ha_action_verification.py" hailo_services.assistants.ha.ha_action_verification

systemctl reset-failed "${SERVICE}" || true
systemctl start "${SERVICE}"
systemctl is-active --quiet "${SERVICE}"

echo "Hailo-10H-Services updated successfully (${COMMIT})."
echo "Check startup with: journalctl -u ${SERVICE} -n 80 --no-pager"
echo "Check readiness with: curl http://127.0.0.1:8090/health"
