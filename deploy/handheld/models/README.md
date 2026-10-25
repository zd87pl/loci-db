# Handheld demo models

Weights are **not** committed (see `.gitignore` here). `scripts/fetch_models.py` downloads and
verifies them; `install_pi.sh` calls it with `--only-missing` on the Pi.

| Model | Purpose | Source (pinned) | Size |
|---|---|---|---|
| `yolov6-nano` → `YOLOv6_Nano-R2_COCO_512x384.rvc2.tar.xz` | on-camera COCO-80 detector (RVC2 blob, 8 shaves) | Luxonis HubAI zoo via `depthai.getModelFromZoo(NNModelDescription("yolov6-nano", platform="RVC2"))` | 8.1 MB |
| `potion-base-8M` (`model.safetensors`, `tokenizer.json`, `config.json`) | 256-d static label embeddings (`StaticLabelEmbedder`) | `minishlab/potion-base-8M` @ `bf8b056651a2c21b8d2565580b8569da283cab23`, sha256 checked | 31 MB |
| `en_US-lessac-medium.onnx` (+ `.onnx.json`) | Piper TTS voice | `rhasspy/piper-voices` @ `5041a7a49ae5b3a0f30e6837769eb03fdbb231bd`, path `en/en_US/lessac/medium/`, sha256 checked | 63 MB |
| faster-whisper `tiny.en` | local STT (optional, `--whisper`) | `Systran/faster-whisper-tiny.en` through `faster_whisper.WhisperModel` into `~/.cache/huggingface` | 75 MB |

## Layout

Two layouts exist because the Pi already held models before this directory was designed:

```
dev machine (this directory)              Pi (/home/ziggy/loci-handheld/models)
  zoo/<hash>/*.rvc2.tar.xz                  zoo/<hash>/*.rvc2.tar.xz          DEPTHAI_ZOO_CACHE_PATH
  potion-base-8M/{model.safetensors,...}    embeddings/potion-base-8M/{...}   LOCI_EMBED_MODEL_DIR
  piper/en_US-lessac-medium.onnx{,.json}    piper/en_US-lessac-medium.onnx    PIPER_MODEL_PATH
```

`fetch_models.py --layout auto` picks the Pi layout when `<models-dir>/embeddings` exists or the
models dir is `/home/ziggy/loci-handheld/models`, otherwise the dev layout.

## Commands

```bash
# dev machine: everything that can be fetched here (zoo needs `depthai` importable)
python deploy/handheld/scripts/fetch_models.py --list
python deploy/handheld/scripts/fetch_models.py            # zoo + potion + piper into this directory
python deploy/handheld/scripts/fetch_models.py --verify   # re-hash what is present

# Pi (inside the venv so depthai and faster_whisper are importable)
/home/ziggy/loci-handheld/.venv/bin/python /home/ziggy/loci-handheld/src/deploy/handheld/scripts/fetch_models.py \
    --models-dir /home/ziggy/loci-handheld/models --only-missing --whisper tiny.en
```

Without network on the Pi, run the fetch on the Mac and `rsync -a deploy/handheld/models/zoo/
ziggy@loci-pi.local:/home/ziggy/loci-handheld/models/zoo/` (same for `potion-base-8M` →
`embeddings/potion-base-8M` and `piper/`).

The Piper voice may also be copied from an older checkout: `install_pi.sh` copies
`~/loci-db/deploy/oak-rpi5/models/en_US-lessac-medium.onnx{,.json}` into `models/piper/` when present.

## Licences

Check each model card before redistributing: YOLOv6 (Luxonis HubAI model page), potion-base-8M
(MIT, minishlab), the Piper lessac voice (`MODEL_CARD` next to the files on the Hub), and
faster-whisper weights (MIT conversion of OpenAI Whisper).
