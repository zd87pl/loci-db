#!/usr/bin/env bash
# Launch the LOCI OAK-D Lite assistive demo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run scripts/setup_rpi5.sh first."
    exit 1
fi

# Set environment defaults for offline operation
export WHISPER_LOCAL=1
export WHISPER_LOCAL_MODEL="${WHISPER_MODEL:-base}"
export TTS_ENGINE="${TTS_ENGINE:-piper}"
export PIPER_MODEL_PATH="${PIPER_MODEL_PATH:-$PROJECT_DIR/models/en_US-lessac-medium.onnx}"
export EDGE_TTS_VOICE="${EDGE_TTS_VOICE:-en-US-AnaNeural}"

# Ensure LOCI package is importable
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

echo "LOCI OAK-D Lite Demo"
echo "  Whisper: local ($WHISPER_LOCAL_MODEL)"
echo "  TTS: $TTS_ENGINE"
echo "  Piper model: $PIPER_MODEL_PATH"
echo ""

# Parse arguments
MODE="headless"
PORT="8000"
for arg in "$@"; do
    case "$arg" in
        --server) MODE="server" ;;
        --port=*) PORT="${arg#*=}" ;;
    esac
done

if [ "$MODE" = "server" ]; then
    echo "Starting in server mode on port $PORT..."
    python3 -m app.main --server --port "$PORT"
else
    echo "Starting in headless voice mode..."
    echo "Speak into your Bluetooth headphones to query objects."
    echo "Press Ctrl+C to stop."
    echo ""
    python3 -m app.main
fi
