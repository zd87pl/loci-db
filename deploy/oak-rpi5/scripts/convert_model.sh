#!/usr/bin/env bash
# Convert YOLOv8-nano to OpenVINO blob for OAK-D Lite Myriad X VPU
#
# This converts a PyTorch YOLO model to an OpenVINO IR, then compiles
# it into a .blob that runs on the Myriad X neural compute engine.
#
# Prerequisites:
#   pip install ultralytics openvino-dev blobconverter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
VENV_DIR="$PROJECT_DIR/.venv"

# Activate venv if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

mkdir -p "$MODELS_DIR"

echo "=== YOLOv8-nano Model Conversion for OAK-D Lite ==="
echo ""

# Step 1: Export YOLOv8-nano to OpenVINO IR
echo "[1/2] Exporting YOLOv8-nano to OpenVINO format..."
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='openvino', imgsz=416, half=True)
print('OpenVINO IR exported successfully')
"

# The export creates a directory like yolov8n_openvino_model/
OPENVINO_DIR="yolov8n_openvino_model"
if [ ! -d "$OPENVINO_DIR" ]; then
    echo "ERROR: OpenVINO export directory not found"
    exit 1
fi

# Step 2: Compile to .blob for Myriad X
echo ""
echo "[2/2] Compiling OpenVINO IR to Myriad X blob..."
python3 -c "
import blobconverter
blob_path = blobconverter.from_openvino(
    xml='$OPENVINO_DIR/yolov8n.xml',
    bin='$OPENVINO_DIR/yolov8n.bin',
    data_type='FP16',
    shaves=6,
    output_dir='$MODELS_DIR',
)
print(f'Blob compiled: {blob_path}')
"

# Copy/rename to standard location
BLOB_FILE=$(find "$MODELS_DIR" -name "*.blob" -type f | head -1)
if [ -n "$BLOB_FILE" ]; then
    cp "$BLOB_FILE" "$MODELS_DIR/yolov8n.blob"
    echo ""
    echo "Model ready: $MODELS_DIR/yolov8n.blob"
else
    echo ""
    echo "WARNING: .blob file not found in $MODELS_DIR"
    echo "You may need to use blobconverter.luxonis.com for online conversion."
fi

# Cleanup
rm -rf "$OPENVINO_DIR"

echo ""
echo "=== Done ==="
echo "The blob can now be used with: oak_pipeline.py --blob models/yolov8n.blob"
