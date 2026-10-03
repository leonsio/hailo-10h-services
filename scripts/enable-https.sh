#!/usr/bin/env bash
# Additional TLS listener; preserves the service's existing HTTP port and runtime.
set -euo pipefail
if [[ ${EUID} -ne 0 ]]; then
  echo 'Run with sudo bash scripts/enable-https.sh [--dns cm5.local] [--ip 192.168.1.10] [--port 8443]' >&2
  exit 1
fi
SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TLS_DIR=/etc/hailo-10h-services/tls
SITE=/etc/nginx/sites-available/hailo-10h-https.conf
ENABLED=/etc/nginx/sites-enabled/hailo-10h-https.conf
if ! command -v nginx >/dev/null || ! command -v openssl >/dev/null || ! command -v ip >/dev/null; then
  apt-get update
  apt-get install -y nginx openssl iproute2
fi
install -d -m 0755 /etc/hailo-10h-services "${TLS_DIR}"
python3 "${SOURCE_DIR}/src/hailo_services/tls.py" "$@" --directory "${TLS_DIR}"
install -d -m 0755 /etc/nginx/sites-available /etc/nginx/sites-enabled
BACKUP=$(mktemp)
HAD_SITE=false
HAD_LINK=false
if [[ -e ${SITE} ]]; then
  cp -p -- "${SITE}" "${BACKUP}"
  HAD_SITE=true
fi
if [[ -e ${ENABLED} || -L ${ENABLED} ]]; then
  if [[ ! -L ${ENABLED} || $(readlink "${ENABLED}") != "${SITE}" ]]; then
    echo "Refusing to replace unrelated ${ENABLED}" >&2
    rm -f -- "${BACKUP}"
    exit 1
  fi
  HAD_LINK=true
fi
install -m 0644 "${TLS_DIR}/nginx-site.conf" "${SITE}"
ln -sfn -- "${SITE}" "${ENABLED}"
if ! nginx -t; then
  if ${HAD_SITE}; then cp -p -- "${BACKUP}" "${SITE}"; else rm -f -- "${SITE}"; fi
  if ! ${HAD_LINK}; then rm -f -- "${ENABLED}"; fi
  rm -f -- "${BACKUP}"
  echo 'nginx validation failed; previous site configuration restored.' >&2
  exit 1
fi
rm -f -- "${BACKUP}"
systemctl enable nginx
if systemctl is-active --quiet nginx; then systemctl reload nginx; else systemctl start nginx; fi
PORT=$(python3 - "${TLS_DIR}/nginx-site.conf" <<'PY'
import re, sys
print(re.search(r'listen (\d+) ssl;', open(sys.argv[1]).read()).group(1))
PY
)
echo "HTTPS ready: https://$(hostname):${PORT}/ (or use a certificate-listed IP)"
echo "iPhone profile: https://$(hostname):${PORT}/hailo-ca.mobileconfig"
echo "CA certificate: ${TLS_DIR}/hailo-ca.crt (public); never copy ca.key or server.key."
echo 'iPhone: install profile, then Settings > General > About > Certificate Trust Settings > enable Hailo-10H Local CA.'
echo 'HTTPS proxy logs: journalctl -u nginx; inference logs: journalctl -u hailo-10h-services'
