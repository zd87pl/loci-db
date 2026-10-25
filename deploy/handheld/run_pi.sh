#!/usr/bin/env bash
# Start the LOCI handheld server (uvicorn, one worker, never --reload).
#
# Layout (Pi): /home/ziggy/loci-handheld/{src,.venv,env,data,models,certs}
# This script lives at <src>/deploy/handheld/run_pi.sh; LOCI_HOME defaults to the parent of
# <src> when that directory is called "src", else to $HOME/loci-handheld.
#
# Environment: $LOCI_HOME/env is sourced (set -a) when present; systemd also passes it through
# EnvironmentFile=, which is harmless duplication. PORT (8001) and LOCI_TLS (0) are honoured.
#
#   run_pi.sh            start in the foreground (Ctrl-C stops)
#   run_pi.sh --help     this text
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -z "${LOCI_HOME:-}" ]; then
  if [ "$(basename "$SRC_DIR")" = "src" ]; then
    LOCI_HOME="$(dirname "$SRC_DIR")"
  else
    LOCI_HOME="${HOME}/loci-handheld"
  fi
fi
export LOCI_HOME

ENV_FILE="${LOCI_ENV_FILE:-$LOCI_HOME/env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

VENV="${LOCI_VENV:-$LOCI_HOME/.venv}"
if [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
else
  PY="$(command -v python3)"
  echo "run_pi.sh: no venv at $VENV, using $PY" >&2
fi

PORT="${PORT:-8001}"
HOST="${LOCI_BIND_HOST:-0.0.0.0}"
# short host name without a trailing .local (macOS hostname already carries it)
SHORT_HOST="$(hostname -s 2>/dev/null || hostname)"
SHORT_HOST="${SHORT_HOST%.local}"
ARGS=(demo_spatial.app.main:app --host "$HOST" --port "$PORT" --workers 1)

if [ "${LOCI_TLS:-0}" = "1" ]; then
  CERT_DIR="${LOCI_CERT_DIR:-$LOCI_HOME/certs}"
  if [ ! -f "$CERT_DIR/cert.pem" ] || [ ! -f "$CERT_DIR/key.pem" ]; then
    mkdir -p "$CERT_DIR"
    LOCAL_IP="$("$PY" -c 'import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("10.255.255.255", 1))
    print(s.getsockname()[0])
except OSError:
    print("127.0.0.1")
finally:
    s.close()' 2>/dev/null || echo 127.0.0.1)"
    echo "run_pi.sh: generating a self-signed certificate in $CERT_DIR (SAN: ${SHORT_HOST}.local, $LOCAL_IP)"
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
      -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
      -subj "/CN=LOCI handheld" \
      -addext "subjectAltName=DNS:localhost,DNS:${SHORT_HOST}.local,IP:127.0.0.1,IP:${LOCAL_IP}" \
      >/dev/null 2>&1
    chmod 600 "$CERT_DIR/key.pem"
  fi
  ARGS+=(--ssl-keyfile "$CERT_DIR/key.pem" --ssl-certfile "$CERT_DIR/cert.pem")
  SCHEME=https
else
  SCHEME=http
fi

mkdir -p "${LOCI_DATA_DIR:-$LOCI_HOME/data}" 2>/dev/null || true

cd "$SRC_DIR"
echo "LOCI handheld: source=${LOCI_DETECTION_SOURCE:-none} voice=${LOCI_VOICE:-none} tls=${LOCI_TLS:-0}"
echo "LOCI handheld: phone page ${SCHEME}://${SHORT_HOST}.local:${PORT}/phone  (python: $PY, src: $SRC_DIR)"
exec "$PY" -m uvicorn "${ARGS[@]}"
