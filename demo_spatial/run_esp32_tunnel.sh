#!/usr/bin/env bash
# Put the running ESP32 host behind a Cloudflare Tunnel so the board can reach it from any
# network (phone hotspot, LTE pendant, another site) over HTTPS, without opening a port.
#
# Two ways to run it:
#
#   Quick tunnel (default) — no Cloudflare account. cloudflared prints a random
#   https://<words>.trycloudflare.com URL; it changes on every start and is meant for tests.
#
#   Named tunnel — TUNNEL_NAME=loci-esp32 TUNNEL_HOSTNAME=loci.example.com. Done once by you,
#   in your own browser session (the script never logs in for you):
#       cloudflared tunnel login
#       cloudflared tunnel create loci-esp32
#       cloudflared tunnel route dns loci-esp32 loci.example.com
#   The script then writes an ingress config that forwards only /api/esp32/* and /esp32 and
#   answers 404 for everything else (demo_spatial/cloudflare/esp32-tunnel.yml.example shows it).
#
# The host must be running (demo_spatial/run_esp32_demo.sh) with a device token: the board
# sends it as X-Device-Token, and requests arriving through the tunnel are refused without it
# (LOCI_ESP32_AUTH=auto). The token itself is never printed here; give it to the board with
# its `token <value>` console command or LOCI_DEVICE_TOKEN in secrets.h, then point the board
# at the tunnel with `host https://<url>` (both are stored in the board's NVS).

set -euo pipefail

PORT="${PORT:-8002}"
LOCI_DATA_DIR="${LOCI_DATA_DIR:-$HOME/.local/share/loci-esp32}"
TOKEN_FILE="${LOCI_ESP32_TOKEN_FILE:-$LOCI_DATA_DIR/device_token}"
AUTH_MODE="${LOCI_ESP32_AUTH:-auto}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed (macOS: brew install cloudflared)" >&2
  exit 1
fi
if [[ "$AUTH_MODE" == "off" ]]; then
  echo "refusing to expose the host with LOCI_ESP32_AUTH=off: every route would be public" >&2
  exit 1
fi
if [[ -z "${LOCI_ESP32_TOKEN:-}" && -z "${LOCI_ESP32_TOKENS:-}" && ! -s "$TOKEN_FILE" ]]; then
  echo "no device token: start demo_spatial/run_esp32_demo.sh first (it creates $TOKEN_FILE)" >&2
  exit 1
fi
if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/esp32/status" >/dev/null; then
  echo "the ESP32 host is not answering on port ${PORT}; start demo_spatial/run_esp32_demo.sh" >&2
  exit 1
fi

echo "=== Cloudflare Tunnel for the LOCI ESP32 host on port ${PORT} ==="
echo "Through the tunnel only /api/esp32/* and /esp32 are served, and they need the device token."
echo "Board console once the URL is known:   host https://<url>     (and 'token <value>' once)"
echo

if [[ -n "${TUNNEL_NAME:-}" ]]; then
  : "${TUNNEL_HOSTNAME:?set TUNNEL_HOSTNAME=loci.example.com (routed with: cloudflared tunnel route dns $TUNNEL_NAME <hostname>)}"
  CREDS="$(ls "$HOME"/.cloudflared/*.json 2>/dev/null | head -1 || true)"
  TUNNEL_ID="$(cloudflared tunnel list --name "$TUNNEL_NAME" --output json 2>/dev/null | "${PYTHON:-python3}" -c 'import json,sys; t=json.load(sys.stdin); print(t[0]["id"] if t else "")' || true)"
  if [[ -z "$TUNNEL_ID" ]]; then
    echo "tunnel '$TUNNEL_NAME' not found: run 'cloudflared tunnel login' and 'cloudflared tunnel create $TUNNEL_NAME' first" >&2
    exit 1
  fi
  CONFIG="$(mktemp -t loci-tunnel)"
  cat > "$CONFIG" <<EOF
tunnel: ${TUNNEL_ID}
credentials-file: ${HOME}/.cloudflared/${TUNNEL_ID}.json
ingress:
  - hostname: ${TUNNEL_HOSTNAME}
    path: ^/(api/esp32/|esp32$)
    service: http://127.0.0.1:${PORT}
  - service: http_status:404
EOF
  echo "Named tunnel ${TUNNEL_NAME} -> https://${TUNNEL_HOSTNAME} (config ${CONFIG}${CREDS:+, credentials in ~/.cloudflared})"
  exec cloudflared tunnel --no-autoupdate --config "$CONFIG" run "$TUNNEL_NAME"
fi

# Quick tunnel. cloudflared would otherwise read ~/.cloudflared/config.yml, and the ingress rules
# of whatever tunnel lives there win over --url (their catch-all answers 404 for every request),
# so the ingress is always written here: the board's surface, nothing else.
CONFIG="$(mktemp -t loci-tunnel)"
cat > "$CONFIG" <<EOF
ingress:
  - path: ^/(api/esp32/|esp32$)
    service: http://127.0.0.1:${PORT}
  - service: http_status:404
EOF
echo "Quick tunnel: watch for the https://....trycloudflare.com line below (new URL on every start)."
exec cloudflared tunnel --no-autoupdate --config "$CONFIG" --url "http://127.0.0.1:${PORT}" "$@"
