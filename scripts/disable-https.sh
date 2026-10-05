#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo 'Run with sudo bash scripts/disable-https.sh [--purge-tls]' >&2
  exit 1
fi

SITE=/etc/nginx/sites-available/hailo-10h-https.conf
ENABLED=/etc/nginx/sites-enabled/hailo-10h-https.conf
TLS_DIR=/etc/hailo-10h-services/tls
PURGE_TLS=false

if [[ ${1:-} == '--purge-tls' ]]; then
  PURGE_TLS=true
elif [[ $# -gt 0 ]]; then
  echo "Unknown option: $1" >&2
  echo 'Usage: sudo bash scripts/disable-https.sh [--purge-tls]' >&2
  exit 2
fi

# Only remove the nginx site created by enable-https.sh.
if [[ -L ${ENABLED} ]]; then
  TARGET=$(readlink -- "${ENABLED}")
  if [[ ${TARGET} == "${SITE}" ]]; then
    rm -f -- "${ENABLED}"
  else
    echo "WARNING: ${ENABLED} points to ${TARGET}; leaving it untouched." >&2
  fi
elif [[ -e ${ENABLED} ]]; then
  echo "WARNING: ${ENABLED} is not the expected symlink; leaving it untouched." >&2
fi

rm -f -- "${SITE}"

if command -v nginx >/dev/null 2>&1; then
  if nginx -t; then
    if systemctl is-active --quiet nginx; then
      systemctl reload nginx
    fi
  else
    echo 'WARNING: nginx configuration test failed after removing the Hailo HTTPS site.' >&2
  fi
fi

if ${PURGE_TLS}; then
  rm -rf -- "${TLS_DIR}"
  echo "Removed TLS certificates/keys generated for hailo-10h-services: ${TLS_DIR}"
else
  echo "TLS files retained in ${TLS_DIR}"
  echo 'Use --purge-tls if you also want to remove the generated Hailo TLS CA/server certificates and keys.'
fi

echo 'HTTPS proxy disabled. The native HTTP service remains unchanged.'
echo 'HTTP health endpoint: http://127.0.0.1:8090/health'
