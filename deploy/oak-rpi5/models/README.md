# Models Directory

This directory holds pre-converted models for the OAK-D Lite demo.

## Required Models

### YOLOv8-nano for Myriad X VPU

The OAK-D Lite's Myriad X neural compute engine runs OpenVINO `.blob` models.

**Option A: Automatic conversion (on RPi5):**
```bash
bash scripts/convert_model.sh
```

**Option B: Online conversion (any machine):**
1. Go to https://blobconverter.luxonis.com
2. Select "OpenVINO Model Zoo" or upload custom model
3. Search for YOLOv8 or upload the OpenVINO IR files
4. Select shaves=6, data type=FP16
5. Download the `.blob` file to this directory as `yolov8n.blob`

**Option C: Use COCO-pretrained blob:**
```bash
python3 -c "
import blobconverter
path = blobconverter.from_zoo(
    name='yolov8n_coco_416x416',
    zoo_type='depthai',
    shaves=6,
)
print(f'Downloaded to: {path}')
"
cp <downloaded-path> models/yolov8n.blob
```

### Piper TTS Voice Model

Downloaded automatically by `scripts/setup_rpi5.sh`. Manual download:

```bash
curl -L -o en_US-lessac-medium.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
curl -L -o en_US-lessac-medium.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
```

### Faster-Whisper STT

Downloaded automatically on first use. The `base` model (~140MB) provides
a good accuracy/speed tradeoff on RPi5 ARM. Use `tiny` (~75MB) for faster
transcription at lower accuracy.
