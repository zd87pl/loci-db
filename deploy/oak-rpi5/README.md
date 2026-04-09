# LOCI OAK-D Lite + Raspberry Pi 5 — Assistive Spatial Memory Demo

A wearable assistive device that helps visually impaired users locate
personal objects using spatial episodic memory. Ask "where did I leave my
keys?" and get a spoken natural-language response.

## Hardware Required

| Component | Notes |
|-----------|-------|
| Raspberry Pi 5 (8GB) | With active cooler |
| Luxonis OAK-D Lite | USB-C, RGB + stereo depth + Myriad X VPU |
| Bluetooth headphones | Any A2DP + HFP profile (mic + speakers) |
| USB-C PD power supply | 27W for RPi5 |

## Quick Start

```bash
# 1. Clone the repo on your RPi5
git clone https://github.com/anthropics/loci-db.git
cd loci-db/deploy/oak-rpi5

# 2. Run the setup script (installs system deps, Python env, models)
bash scripts/setup_rpi5.sh

# 3. Pair Bluetooth headphones
bash scripts/pair_bluetooth.sh

# 4. (Optional) Convert YOLOv8 model for on-device inference
bash scripts/convert_model.sh

# 5. Run the demo
bash scripts/run.sh
```

## How It Works

1. **OAK-D Lite** captures RGB frames + stereo depth at 8 FPS
2. **YOLOv8-nano** runs on the Myriad X VPU for real-time object detection
3. Detections pass through **temporal consensus** (2+ sightings in 1s window)
4. Confirmed objects are stored in **LOCI spatial memory** with 4D coordinates
5. User asks a question via **Bluetooth headphones** microphone
6. **Whisper** transcribes speech locally (faster-whisper, int8 on ARM)
7. Intent parsing extracts the query (where is, history, list objects)
8. **LOCI** searches spatial memory using Hilbert-indexed vector similarity
9. **Piper TTS** speaks the answer through the Bluetooth headphones

## Modes

### Headless (default)
Voice-only operation via Bluetooth headphones. No screen needed.
```bash
bash scripts/run.sh
```

### Server (debug)
FastAPI server with REST endpoints for debugging and visualization.
```bash
bash scripts/run.sh --server --port=8000
```

Endpoints:
- `GET /health` — System status
- `GET /api/objects` — All tracked objects
- `GET /api/objects/{label}/location` — Find an object
- `POST /api/voice/text-query?text=where+are+my+keys` — Text query
- `GET /api/stats` — Detailed pipeline stats

## Configuration

Edit `config.yaml` to tune detection classes, confidence thresholds,
voice model sizes, and audio settings.

## Auto-Start on Boot

```bash
sudo cp systemd/loci-demo.service /etc/systemd/system/
sudo systemctl enable loci-demo
sudo systemctl start loci-demo
```

## Architecture

```
OAK-D Lite (USB-C)
    │
    ├─ RGB camera ──────┐
    ├─ Stereo depth ────┤
    └─ Myriad X VPU ────┤ DepthAI Pipeline
                        │
                        ▼
              Scene Ingestion
              (consensus buffer)
                        │
                        ▼
              LOCI Spatial Memory
              (128-dim, 4D Hilbert)
                        │
                ┌───────┴───────┐
                ▼               ▼
          Voice Query      REST API
          (Whisper STT)    (FastAPI)
                │
                ▼
           Piper TTS
                │
                ▼
        Bluetooth Headphones
```

## Offline Operation

The entire system runs without internet:
- **Detection**: On-device Myriad X VPU (no cloud API)
- **Memory**: In-process LOCI with MemoryStore (no database)
- **STT**: Local faster-whisper (int8 quantized for ARM)
- **TTS**: Local Piper TTS (ONNX model)
