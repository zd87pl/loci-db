#!/usr/bin/env bash
# Start the LOCI-DB Spatial Memory Assistant backend.
#
# Required env vars:
#   OPENAI_API_KEY — enables Whisper STT, GPT-4o mini VLM, OpenAI TTS
#
# Optional env vars:
#   VLM_PROVIDER      — "openai" (default) | "gemini"
#   GOOGLE_API_KEY    — needed if VLM_PROVIDER=gemini
#   TTS_ENGINE        — "openai" (default) | "piper" | "edge"
#   PIPER_MODEL_PATH  — path to .onnx model file if TTS_ENGINE=piper
#   WHISPER_LOCAL     — "1" to use local faster-whisper
#   PORT              — server port (default: 8001)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${PORT:-8001}"

# Performance defaults: local STT + TTS avoids cloud round-trips (~600 ms vs 7-10 s)
export WHISPER_LOCAL="${WHISPER_LOCAL:-1}"
export WHISPER_LOCAL_MODEL="${WHISPER_LOCAL_MODEL:-tiny}"
export TTS_ENGINE="${TTS_ENGINE:-edge}"
CERTS_DIR="$SCRIPT_DIR/.certs"

# ── Generate self-signed SSL cert (needed for iPhone camera access) ──
if [[ ! -f "$CERTS_DIR/cert.pem" ]]; then
  echo "Generating self-signed SSL certificate for HTTPS..."
  mkdir -p "$CERTS_DIR"
  # Get local IP for SAN (Subject Alternative Name) so iPhone trusts it
  LOCAL_IP=$(python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()" 2>/dev/null || echo "127.0.0.1")
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$CERTS_DIR/key.pem" \
    -out "$CERTS_DIR/cert.pem" \
    -days 365 \
    -subj "/CN=LOCI Spatial Demo" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${LOCAL_IP}" \
    2>/dev/null
  echo "✓ SSL cert generated at $CERTS_DIR/"
  echo ""
fi

echo "=== LOCI-DB Spatial Memory Assistant ==="
echo "Backend URL: https://localhost:${PORT}"
echo "API docs:    https://localhost:${PORT}/docs"
echo ""
echo "📱 iPhone setup:"
echo "   1. Open https://$(python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()" 2>/dev/null || echo "localhost"):${PORT} on your Mac"
echo "   2. Click the teal 'iPhone 3D Scanner' button → scan QR with iPhone"
echo "   3. On iPhone: tap 'Advanced' → 'Accept Risk' for the self-signed cert"
echo ""

echo "STT:  ${WHISPER_LOCAL:+local faster-whisper (${WHISPER_LOCAL_MODEL})}${WHISPER_LOCAL:-cloud Whisper API}"
echo "TTS:  ${TTS_ENGINE}"
echo ""

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "⚠  OPENAI_API_KEY not set — VLM scene analysis will be disabled."
  echo "   Voice queries use rule-based responses (fast, no API key needed)."
  echo ""
fi

# Install dependencies if needed
cd "$SCRIPT_DIR"
if ! python -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

# Install loci package in editable mode if not already installed
if ! python -c "import loci" 2>/dev/null; then
  echo "Installing loci package..."
  pip install -e "$REPO_ROOT"
fi

exec uvicorn demo_spatial.app.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --ssl-keyfile "$CERTS_DIR/key.pem" \
  --ssl-certfile "$CERTS_DIR/cert.pem" \
  --reload \
  --reload-dir "$REPO_ROOT/demo_spatial" \
  --reload-dir "$REPO_ROOT/loci"
