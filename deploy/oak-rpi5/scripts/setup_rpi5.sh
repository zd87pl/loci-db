#!/usr/bin/env bash
# Setup script for LOCI OAK-D Lite demo on Raspberry Pi 5
# Run once after flashing Raspberry Pi OS (64-bit Bookworm)
set -euo pipefail

echo "=== LOCI OAK-D Lite + RPi5 Setup ==="
echo ""

# --- System packages ---
echo "[1/6] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    libopenblas-dev liblapack-dev \
    portaudio19-dev \
    libatlas-base-dev \
    bluetooth bluez pulseaudio-module-bluetooth \
    ffmpeg \
    usbutils \
    libusb-1.0-0-dev

# --- DepthAI udev rules ---
echo ""
echo "[2/6] Setting up OAK-D udev rules..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/80-movidius.rules > /dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger

# --- Python virtual environment ---
echo ""
echo "[3/6] Creating Python virtual environment..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# --- Install Python dependencies ---
echo ""
echo "[4/6] Installing Python packages..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Install loci-db from the repo root
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
pip install -e "$REPO_ROOT"

# --- Download Piper TTS model ---
echo ""
echo "[5/6] Downloading Piper TTS voice model..."
MODELS_DIR="$PROJECT_DIR/models"
mkdir -p "$MODELS_DIR"

PIPER_MODEL="$MODELS_DIR/en_US-lessac-medium.onnx"
PIPER_CONFIG="$MODELS_DIR/en_US-lessac-medium.onnx.json"
if [ ! -f "$PIPER_MODEL" ]; then
    curl -L -o "$PIPER_MODEL" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    curl -L -o "$PIPER_CONFIG" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    echo "  Piper model downloaded to $PIPER_MODEL"
else
    echo "  Piper model already exists at $PIPER_MODEL"
fi

# --- Bluetooth setup ---
echo ""
echo "[6/6] Configuring Bluetooth audio..."
# Enable Bluetooth service
sudo systemctl enable bluetooth
sudo systemctl start bluetooth

# Ensure PulseAudio Bluetooth module loads on boot
if ! grep -q "module-bluetooth-discover" /etc/pulse/default.pa 2>/dev/null; then
    echo "load-module module-bluetooth-discover" | sudo tee -a /etc/pulse/default.pa > /dev/null
    echo "  Added Bluetooth module to PulseAudio config"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Pair Bluetooth headphones: bash scripts/pair_bluetooth.sh"
echo "  2. Convert YOLO model (optional): bash scripts/convert_model.sh"
echo "  3. Run the demo: bash scripts/run.sh"
echo ""
echo "To install as a system service:"
echo "  sudo cp systemd/loci-demo.service /etc/systemd/system/"
echo "  sudo systemctl enable loci-demo"
echo "  sudo systemctl start loci-demo"
