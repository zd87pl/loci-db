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

echo "=== LOCI-DB Spatial Memory Assistant ==="
echo "Backend URL: http://localhost:${PORT}"
echo "API docs:    http://localhost:${PORT}/docs"
echo ""

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "⚠  OPENAI_API_KEY not set — VLM, STT and OpenAI TTS will be disabled."
  echo "   Voice queries will use rule-based responses only."
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
  --reload \
  --reload-dir "$REPO_ROOT/demo_spatial" \
  --reload-dir "$REPO_ROOT/loci"
