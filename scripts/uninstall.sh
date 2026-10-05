#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo 'Run with sudo bash scripts/uninstall.sh [--remove-nginx]' >&2
  exit 1
fi

REMOVE_NGINX=false
for ARG in "$@"; do
  case "${ARG}" in
    --remove-nginx)
      REMOVE_NGINX=true
      ;;
    -h|--help)
      cat <<'EOF'
Usage: sudo bash scripts/uninstall.sh [--remove-nginx]

Removes hailo-10h-services completely while keeping HEF/model files and external
LiteRT-LM installations/models.

Options:
  --remove-nginx  Also purge all installed nginx packages and remove /etc/nginx.
                  Use this only if nginx is not needed by another application.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: ${ARG}" >&2
      echo 'Usage: sudo bash scripts/uninstall.sh [--remove-nginx]' >&2
      exit 2
      ;;
  esac
done

SERVICE=hailo-10h-services.service
SERVICE_DIR=/opt/hailo-10h-services
SERVICE_STATE=/var/lib/hailo-10h-services
CONFIG=/etc/hailo-10h-services.yaml
LEGACY_ENV=/etc/hailo-10h-services.env
SYSTEMD_UNIT=/etc/systemd/system/${SERVICE}
UDEV_RULE=/etc/udev/rules.d/71-hailo-10h-services.rules
TLS_DIR=/etc/hailo-10h-services/tls
NGINX_SITE=/etc/nginx/sites-available/hailo-10h-https.conf
NGINX_ENABLED=/etc/nginx/sites-enabled/hailo-10h-https.conf
MODEL_DIR=/usr/local/hailo/resources/models/hailo10h
SERVICE_USER=hailo-services

printf '%s\n' 'Uninstalling hailo-10h-services ...'

# Stop and remove the application service.
if systemctl cat "${SERVICE}" >/dev/null 2>&1; then
  systemctl disable --now "${SERVICE}" || true
else
  systemctl stop "${SERVICE}" 2>/dev/null || true
fi
rm -f -- "${SYSTEMD_UNIT}"
systemctl daemon-reload
systemctl reset-failed "${SERVICE}" 2>/dev/null || true

# Remove only the nginx configuration created by scripts/enable-https.sh first.
if [[ -L ${NGINX_ENABLED} ]]; then
  TARGET=$(readlink -- "${NGINX_ENABLED}")
  if [[ ${TARGET} == "${NGINX_SITE}" ]]; then
    rm -f -- "${NGINX_ENABLED}"
  else
    echo "WARNING: ${NGINX_ENABLED} points to ${TARGET}; leaving it untouched." >&2
  fi
elif [[ -e ${NGINX_ENABLED} ]]; then
  echo "WARNING: ${NGINX_ENABLED} is not the expected symlink; leaving it untouched." >&2
fi
rm -f -- "${NGINX_SITE}"

if ! ${REMOVE_NGINX}; then
  if command -v nginx >/dev/null 2>&1 && nginx -t >/dev/null 2>&1; then
    if systemctl is-active --quiet nginx; then
      systemctl reload nginx || true
    fi
  fi
fi

# Remove application-generated TLS material.
rm -rf -- "${TLS_DIR}"

# Remove application files, venv, configuration and runtime state.
# litert-lm installed into this venv disappears with the venv; any system/vendor
# LiteRT-LM installation and model files outside SERVICE_DIR are deliberately retained.
rm -rf -- "${SERVICE_DIR}" "${SERVICE_STATE}"
rm -f -- "${CONFIG}" "${LEGACY_ENV}"

# Remove the application-specific udev rule and ACLs added by install.sh.
rm -f -- "${UDEV_RULE}"
if command -v setfacl >/dev/null 2>&1; then
  for PATH_TO_CLEAN in \
    /usr/local/hailo \
    /usr/local/hailo/resources \
    /usr/local/hailo/resources/models \
    "${MODEL_DIR}"; do
    if [[ -e ${PATH_TO_CLEAN} ]]; then
      setfacl -x "u:${SERVICE_USER}" "${PATH_TO_CLEAN}" 2>/dev/null || true
      setfacl -x "d:u:${SERVICE_USER}" "${PATH_TO_CLEAN}" 2>/dev/null || true
    fi
  done
  for DEVICE_PATH in /dev/h1x*; do
    [[ -e ${DEVICE_PATH} ]] || continue
    setfacl -x "u:${SERVICE_USER}" "${DEVICE_PATH}" 2>/dev/null || true
  done
fi
if command -v udevadm >/dev/null 2>&1; then
  udevadm control --reload-rules || true
fi

# Remove the dedicated system account after its files/ACL references are gone.
if id "${SERVICE_USER}" >/dev/null 2>&1; then
  userdel "${SERVICE_USER}" || true
fi
if getent group "${SERVICE_USER}" >/dev/null 2>&1; then
  groupdel "${SERVICE_USER}" 2>/dev/null || true
fi

# Optional: remove nginx completely. This intentionally affects every nginx site on
# the host, not only hailo-10h-services, therefore it is opt-in.
if ${REMOVE_NGINX}; then
  echo 'Removing nginx because --remove-nginx was specified ...'
  systemctl disable --now nginx 2>/dev/null || true

  if command -v dpkg-query >/dev/null 2>&1; then
    mapfile -t NGINX_PACKAGES < <(
      dpkg-query -W -f='${binary:Package}\t${db:Status-Abbrev}\n' 'nginx*' 2>/dev/null \
        | awk '$2 ~ /^ii/ {print $1}'
    )
  else
    NGINX_PACKAGES=()
  fi

  if (( ${#NGINX_PACKAGES[@]} > 0 )); then
    printf 'Purging nginx packages:'
    printf ' %s' "${NGINX_PACKAGES[@]}"
    printf '\n'
    DEBIAN_FRONTEND=noninteractive apt-get purge -y "${NGINX_PACKAGES[@]}"
    DEBIAN_FRONTEND=noninteractive apt-get autoremove -y
  else
    echo 'No installed nginx packages found.'
  fi

  # Purge removes package-owned configuration; remove remaining locally-created
  # nginx configuration/cache/log directories as part of the explicit full removal.
  rm -rf -- /etc/nginx /var/cache/nginx
  rm -rf -- /var/log/nginx
fi

cat <<EOF

hailo-10h-services has been removed.

INTENTIONALLY NOT REMOVED:
  * HailoRT kernel driver, firmware and vendor HailoRT installation
  * HEF/model files under: ${MODEL_DIR}
  * LiteRT-LM installations/models outside ${SERVICE_DIR}
  * openssl, iproute2, python3-venv, libsndfile1 and acl packages
EOF

if ${REMOVE_NGINX}; then
  echo '  * nginx was REMOVED because --remove-nginx was specified'
else
  echo '  * nginx remains installed (use --remove-nginx to purge it too)'
fi

cat <<EOF

The installer may have installed litert-lm inside ${SERVICE_DIR}/venv. That private
copy was removed together with the application venv; no external/system LiteRT-LM
installation or LiteRT model file was deleted.

Existing HEF/model files:
EOF

if [[ -d ${MODEL_DIR} ]]; then
  find "${MODEL_DIR}" -maxdepth 2 -type f \( -iname '*.hef' -o -iname '*.litertlm' -o -iname '*.bin' \) -print 2>/dev/null || true
else
  echo "  (model directory does not exist: ${MODEL_DIR})"
fi

printf '\nSearch for remaining LiteRT-LM model files/installations if needed:\n'
printf '%s\n' "  find /usr/local /opt /home -iname '*litert*' -o -iname '*.litertlm' 2>/dev/null"
printf '%s\n' "  python3 -m pip show litert-lm 2>/dev/null || true"
printf '\nDone.\n'
