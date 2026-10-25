#!/usr/bin/env bash
# Start the LOCI spatial-memory demo as the HOST for an ESP32 thin client.
#
# The board streams JPEG frames to this process (POST /api/esp32/frame) and fetches
# spoken answers as raw PCM (GET /api/esp32/audio/next). Everything heavy — YOLO,
# LOCI memory, Whisper, TTS — runs here. See docs/ESP32_THIN_CLIENT_DEMO.md.
#
# Defaults are tuned for a laptop without any cloud key:
#   PORT=8002                 the Mac already runs the browser demo on 8001
#   YOLO_MODEL=yolov8n.pt     COCO model, no CLIP download (YOLO-World needs it)
#   WHISPER_LOCAL=1           faster-whisper on CPU
#   WHISPER_LOCAL_MODEL=tiny  ~1 s for a short question
#   TTS_ENGINE=edge           edge-tts (needs internet); set TTS_ENGINE=piper + PIPER_MODEL_PATH for offline
#   EDGE_TTS_VOICE            en-US-JennyNeural (adult female); e.g. en-GB-SoniaNeural, en-US-GuyNeural
#   LOCI_ESP32_MAX_FPS=4      host-side cap on frames processed per second
#   LOCI_ESP32_MIN_CONF=0.4   YOLO confidence floor for the small camera
#   LOCI_ESP32_WARMUP=1       load YOLO at startup instead of on the first frame
#   LOCI_ESP32_WB=gray        gray-world white balance before detection (none to disable)
#   LOCI_ESP32_IMGSZ=960      YOLO inference size, matched to the P4 camera's 1280x960 frames (640 is faster)
#   LOCI_DATA_DIR             memory snapshot dir (default ~/.local/share/loci-esp32, separate from the handheld)
#   LOCI_ESP32_TOKEN_FILE     device token for the online mode (default $LOCI_DATA_DIR/device_token, created once)
#   LOCI_ESP32_AUTH=auto      token required for requests from outside the LAN (Cloudflare Tunnel); always | off
#
# Online mode: demo_spatial/run_esp32_tunnel.sh puts this host behind a Cloudflare Tunnel (HTTPS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PORT="${PORT:-8002}"
export YOLO_MODEL="${YOLO_MODEL:-yolov8n.pt}"
export WHISPER_LOCAL="${WHISPER_LOCAL:-1}"
export WHISPER_LOCAL_MODEL="${WHISPER_LOCAL_MODEL:-tiny}"
export TTS_ENGINE="${TTS_ENGINE:-edge}"
# edge-tts voice: an adult female voice (the pipeline default, en-US-AnaNeural, is a child voice)
export EDGE_TTS_VOICE="${EDGE_TTS_VOICE:-en-US-JennyNeural}"
export LOCI_ESP32_MAX_FPS="${LOCI_ESP32_MAX_FPS:-4}"
export LOCI_ESP32_MIN_CONF="${LOCI_ESP32_MIN_CONF:-0.4}"
export LOCI_ESP32_WARMUP="${LOCI_ESP32_WARMUP:-1}"
# Gray-world white balance before YOLO: the OV5647 on the P4 arrives with a strong blue cast
export LOCI_ESP32_WB="${LOCI_ESP32_WB:-gray}"
export LOCI_ESP32_IMGSZ="${LOCI_ESP32_IMGSZ:-960}"
# Keep the ESP32 demo's memory apart from the handheld demo's (both default to ~/.local/share/loci-handheld)
export LOCI_DATA_DIR="${LOCI_DATA_DIR:-$HOME/.local/share/loci-esp32}"
# Every 10th frame + detections on disk: the training material for a detector that knows your objects
export LOCI_ESP32_DATASET_DIR="${LOCI_ESP32_DATASET_DIR:-$LOCI_DATA_DIR/dataset}"
# Device token: requests that arrive through the Cloudflare Tunnel must carry it (the LAN needs none
# unless LOCI_ESP32_AUTH=always). Created once with 0600 permissions and never printed.
export LOCI_ESP32_TOKEN_FILE="${LOCI_ESP32_TOKEN_FILE:-$LOCI_DATA_DIR/device_token}"
export LOCI_ESP32_AUTH="${LOCI_ESP32_AUTH:-auto}"
# The OpenAI key in many shells is a placeholder; the ESP32 path never needs it.
unset OPENAI_API_KEY

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in "$REPO_ROOT/venv/bin/python" "$REPO_ROOT/.venv/bin/python" python3; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then PYTHON="$candidate"; break; fi
  done
fi

if [[ -z "${LOCI_ESP32_TOKEN:-}" && ! -s "$LOCI_ESP32_TOKEN_FILE" ]]; then
  mkdir -p "$(dirname "$LOCI_ESP32_TOKEN_FILE")"
  (umask 077; "$PYTHON" -c 'import secrets; print(secrets.token_hex(24))' > "$LOCI_ESP32_TOKEN_FILE")
fi

LAN_IP="$("$PYTHON" - <<'EOF' 2>/dev/null || echo "<this-mac-ip>"
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
EOF
)"

echo "=== LOCI ESP32 thin-client host ==="
echo "Board posts frames to:  http://${LAN_IP}:${PORT}/api/esp32/frame"
echo "Board fetches audio:    http://${LAN_IP}:${PORT}/api/esp32/audio/next"
echo "Status page:            http://localhost:${PORT}/esp32"
echo "Main demo UI:           http://localhost:${PORT}/"
TOKEN_WHERE="$LOCI_ESP32_TOKEN_FILE"
[[ -n "${LOCI_ESP32_TOKEN:-}" ]] && TOKEN_WHERE="(the LOCI_ESP32_TOKEN variable)"
echo "Device token:           ${TOKEN_WHERE}  (auth mode ${LOCI_ESP32_AUTH}; board console: token <value>)"
echo "Online mode:            demo_spatial/run_esp32_tunnel.sh  (Cloudflare Tunnel, HTTPS; board console: host https://<url>)"
echo "YOLO: ${YOLO_MODEL}  STT: faster-whisper ${WHISPER_LOCAL_MODEL}  TTS: ${TTS_ENGINE}"
echo

cd "$REPO_ROOT"
exec "$PYTHON" -m uvicorn demo_spatial.app.main:app --host "${HOST:-0.0.0.0}" --port "$PORT" "$@"
