# LOCI spatial memory on an ESP32 thin client

**Device:** an ESP32 board with a small camera module and a speaker (optionally a microphone and a
button), on Wi-Fi.
**Host:** this Mac today; the Raspberry Pi 5 or a cloud endpoint later. The host runs object
detection, the LOCI spatial memory and the voice pipeline. The board only moves pixels and sound.
**Promise:** point the board around the room, then ask "where is my cup" (typed, or spoken into
the board's microphone) and hear the answer from the board's speaker.

This is the first step of the IDD-64 form-factor plan: a thin wearable client whose compute lives
on a host. The same firmware works against a laptop, a Pi or a cloud URL; only `LOCI_HOST` changes.

Status legend: ✅ done · 🔧 in progress · ⬜ planned

## 1. Architecture

```
ESP32 (camera + I2S speaker [+ PDM mic + button])          host (Mac / Pi / cloud)
 ┌────────────────────────────────────────────┐             ┌────────────────────────────────────────┐
 │ capture JPEG (VGA, q12) at LOCI_FPS        │ ──POST──►   │ /api/esp32/frame                       │
 │                                            │   JPEG      │   Esp32Hub: rate cap → YOLO (yolov8n)  │
 │                                            │ ◄──JSON──   │   → Detection(source="esp32")          │
 │   {"labels":[...],"audio_pending":1,...}   │             │   → SceneIngestion consensus           │
 │                                            │             │   → SpatialMemory (LOCI, image frame)  │
 │ if audio_pending: GET audio/next           │ ──GET───►   │ /api/esp32/audio/next  (long poll)     │
 │   stream PCM16 16 kHz mono → I2S speaker   │ ◄──L16───   │   per-device audio queue               │
 │                                            │             │                                        │
 │ hold button: record PCM16 → release        │ ──POST──►   │ /api/esp32/voice                       │
 │                                            │   PCM16     │   faster-whisper → intent → LOCI query │
 │ then GET audio/next → speaker              │             │   → phrasing → TTS → PCM → queue       │
 └────────────────────────────────────────────┘             │ /api/esp32/ask, /say, /status, /esp32  │
                                                            └────────────────────────────────────────┘
```

Host modules (all new, nothing in the shared pipeline modified):

| Module | Role |
|---|---|
| `demo_spatial/app/esp32_source.py` | `Esp32Hub`: per-device state, host-side frame rate cap, `HostYoloDetector` (ultralytics, lazy) → `Detection` objects → `SceneIngestion`; per-device **audio queue**; `Esp32DetectionSource` adapter for the handheld `DetectionSource` protocol. |
| `demo_spatial/app/esp32_voice.py` | `Esp32VoiceBridge`: board PCM → faster-whisper (float32 straight in, no container), `VoicePipeline.handle_text_query` for intent + answer + TTS, TTS output → PCM. |
| `demo_spatial/app/pcm_audio.py` | PCM16 helpers: decode MP3/WAV/WebM → 16 kHz mono (PyAV), WAV fast path, resample, gain, RMS. |
| `demo_spatial/app/esp32_api.py` | The `/api/esp32/*` routes and the `/esp32` page; `install_esp32_routes(app, ...)` is called once at the end of `main.py`. |
| `demo_spatial/static/esp32.html` | Debug page: live MJPEG of the board's frames, typed questions, objects in memory, status JSON. |
| `demo_spatial/tools/esp32_fake_board.py` | A laptop pretending to be the board (image/webcam → frames, plays answers with `afplay`). |
| `demo_spatial/run_esp32_demo.sh` | Starts the host on port 8002 with offline-friendly defaults. |

Firmware, two flavours speaking the same protocol:

| Project | Boards | Stack |
|---|---|---|
| `firmware/esp32-p4-client/` | **Waveshare ESP32-P4-NANO** (the board on the desk): MIPI-CSI OV5647, ES8311 codec + NS4150B speaker, onboard mic, Wi-Fi via an ESP32-C6 over SDIO | ESP-IDF ≥ 5.4: `esp_video` (ISP + hardware JPEG), `esp_codec_dev`, `esp_wifi_remote`/`esp_hosted`, `esp_http_client` |
| `firmware/esp32-cam-client/` | XIAO ESP32S3 Sense, AI-Thinker ESP32-CAM, Freenove ESP32-S3 CAM (DVP OV2640, I2S amplifier, optional PDM mic) | PlatformIO, Arduino-ESP32, `esp32-camera`, legacy I2S driver |

Each README has the pin map, build and flash steps.

### Where observations live

The ESP32 camera has no depth, so observations are stored in the **image frame** exactly like the
browser demo: `cx`, `cy` normalised to [0, 1], `depth_m = None`, `source = "esp32"`. They go through
the same `SceneIngestion` temporal consensus (two agreeing frames within one second) and the same
`SpatialMemory.observe` storage gate (confidence ≥ 0.55), so the objects list, `/api/objects`,
`/api/query` and the voice answers are shared with the phone and the handheld.

A monocular depth estimate is **optional and low confidence**. `Esp32Hub` accepts a
`depth_estimator(jpeg, detections) -> list[float | None]` and attaches the result as `depth_m` when
`LOCI_ESP32_DEPTH=mono`. Nothing ships enabled: a small Depth Anything model is the natural plug-in,
but a relative depth map gives ordering, not metres, and the handheld's stereo numbers must not be
mixed with it. Treat any ESP32 depth as "nearer/farther", never as a measurement.

### Plugging into the handheld runtime later

`Esp32DetectionSource(hub)` implements the `DetectionSource` protocol from
`docs/HANDHELD_DEMO_DESIGN.md` §4.2: `start/stop/events/status`, `name="esp32"`, `events()` yields
`DetectionFrame`s of `SpatialDetection`s with `xyz_cam_m=None`. It imports `app.detection_source`
lazily, so it is inert on trees where the handheld modules have not landed. Once the branches are
merged, `HandheldRuntime(Esp32DetectionSource(esp32_hub), localizer, ingestion, memory, cfg)` gives
the ESP32 the same room anchoring path as the OAK (tags would have to be detected on the host from
the JPEG; the ESP32 sends no `TagFrame`s today).

## 2. Protocol between board and host

All calls are plain HTTP/1.1 over the LAN. No TLS, no auth: this is a desk demo. `device` is a short
id the board chooses (`LOCI_DEVICE_ID`, default `esp32`); the host keeps one audio queue per device.

### `POST /api/esp32/frame?device=<id>&seq=<n>`

Body: a JPEG (`Content-Type: image/jpeg`); multipart with a `file` field also works. Optional headers
`X-Device-Id`, `X-Seq`, `X-Device-Ms` (board `millis()`), `X-Rssi` (dBm) are shown on the status page.

Response `200`:

```json
{"ok": true, "device": "esp32", "seq": 42, "dropped": false, "reason": null,
 "n": 2, "labels": ["cup", "laptop"],
 "detections": [{"label": "cup", "cx": 0.61, "cy": 0.52, "width": 0.12, "height": 0.2,
                 "confidence": 0.78, "source": "esp32"}],
 "audio_pending": 0, "next_ms": 250, "infer_ms": 38.5, "width": 640, "height": 480}
```

* `dropped: true, reason: "rate_limited"` — the frame arrived faster than `LOCI_ESP32_MAX_FPS`; it was
  kept for the preview but not run through YOLO. The board should wait `next_ms` between frames.
* `audio_pending` — clips waiting in this device's queue. The board fetches them right after the frame.
* The firmware parses this with `strstr`, so the keys stay flat and stable.

`400` for an empty body or anything that is not a JPEG (`FF D8` magic).

### `GET /api/esp32/audio/next?device=<id>&wait=<seconds>`

Long-polls up to `wait` seconds (0–25). `200` with raw **PCM16 little-endian, mono, 16 kHz** and
`Content-Type: audio/L16; rate=16000; channels=1`, plus `X-Audio-Rate`, `X-Audio-Seconds`,
`X-Audio-Queued` (clips still waiting). `204` when nothing is queued. The board writes the body to
I2S as it arrives; a 5 s answer is 160 kB.

### `POST /api/esp32/voice?device=<id>&rate=16000&language=en&speak=1`

Body: one push-to-talk utterance as raw PCM16 mono at `rate` (resampled to 16 kHz on the host), or
any WAV/MP3/WebM. The host runs STT → intent → LOCI → answer → TTS and **queues** the spoken answer
for the device, then returns:

```json
{"transcription": "where is my cup", "intent": {"kind": "where_is", "object_name": "cup"},
 "answer": "Your cup was last seen just now, in the center of view.",
 "has_audio": true, "audio_seconds": 3.1, "tts_error": null,
 "latency_ms": 1420.3, "stt_ms": 610.2, "device": "esp32", "seconds_in": 1.8,
 "audio_queued": true, "queue_len": 1}
```

Silence (RMS < 0.003), clips under 0.25 s, or an STT failure give `transcription: ""` and the spoken
hint *"Sorry, I couldn't hear that. Hold the button and ask again."*

### `POST /api/esp32/ask` `{"text": "where is my cup", "device": "esp32", "speak": true}`

Typed question (status page, curl, or the firmware's serial console / button without a mic). Same
response shape as `/voice`, the answer is queued for the device.

### `POST /api/esp32/say` `{"text": "hello", "device": "esp32"}`

Speak arbitrary text on the board. `503` when the TTS engine is unavailable.

### Local storage on the board (microSD)

Every frame reply also carries `now_ms` (the host's Unix clock, from which the board derives
timestamps for frames it keeps while offline) and `memory_rev`, a short digest of the remembered
objects that changes only when something appears or moves.

* **Store-and-forward.** While the host or the Wi-Fi is unreachable the board keeps capturing to
  `/sdcard/loci/backlog/<captured_ms>.jpg` and, once back, uploads oldest-first with
  `X-Backlog: 1` and `X-Captured-Ms: <unix ms>`. The host stores those observations at their capture
  time (`SceneIngestion.process_detections(..., timestamp_ms=captured)`), skips the rate limiter and
  the live WebSocket broadcast for them, and counts them under `backlog_frames` in the status. A
  capture time that is implausible (zero, in the future, older than 30 days) falls back to arrival.
* **Cached memory and offline answers.** `GET /api/esp32/memory` returns `{rev, count, objects:
  [{label, where, room, last_seen_ms, age_s}], summary}`; `GET /api/esp32/memory.pcm` is the summary
  sentence as board PCM; `GET /api/esp32/phrase.pcm?text=…` any short phrase (both cached on the
  host). The board refreshes `/sdcard/loci/memory.json` and `memory.pcm` when `memory_rev` changes
  (at most every 10 s) and, when the button is pressed with no host in reach, plays that summary:
  *"I can't reach the host right now. The last things I remember: cup in the center…"*.
* **Dataset.** Every Nth live frame (`LOCI_SD_KEEP_EVERY`, default 10) is saved to
  `/sdcard/loci/frames/<ms>.jpg` with the host's full reply as `<ms>.json` (labels and boxes), the raw
  material for training a detector on the user's own objects later; push-to-talk clips go to
  `/sdcard/loci/voice/<ms>.wav`. Writing stops below `LOCI_SD_MIN_FREE_MB` free.

### Debugging

`GET /api/esp32/status` (devices, fps, dropped and backlog frames, inference time, detector state,
voice engines, tracked labels, memory revision), `GET /api/esp32/frame.jpg` (`?corrected=1` shows
the frame as the detector sees it, rotated and white-balanced), `GET /api/esp32/stream.mjpg?fps=5`,
and the page `GET /esp32`. Processed live frames are also broadcast on the existing `/ws` feed as
`{"type": "detections", "source": "esp32", ...}`, so the main page draws them.

## 3. Running the Mac end to end

The browser demo already occupies port 8001 on this Mac; the ESP32 host uses **8002**.

```bash
# 1. host (repo root; uses venv/ with ultralytics, faster-whisper, edge-tts, PyAV installed)
YOLO_MODEL=yolov8n.pt demo_spatial/run_esp32_demo.sh
# prints:  Board posts frames to: http://<mac-ip>:8002/api/esp32/frame
```

Defaults set by the script: `WHISPER_LOCAL=1`, `WHISPER_LOCAL_MODEL=tiny`, `TTS_ENGINE=edge`
(needs internet; use `TTS_ENGINE=piper PIPER_MODEL_PATH=…onnx` for offline),
`EDGE_TTS_VOICE=en-US-JennyNeural` (an adult female voice; the pipeline's own default is the child
voice Ana), `LOCI_DATA_DIR=~/.local/share/loci-esp32` (memory kept apart from the handheld demo's),
`LOCI_ESP32_MAX_FPS=4`, `LOCI_ESP32_MIN_CONF=0.4`, `LOCI_ESP32_WARMUP=1`. `OPENAI_API_KEY` is unset on purpose: the ESP32
path never calls OpenAI. `yolov8n.pt` is the plain COCO model (no CLIP download, unlike YOLO-World);
it knows cup, bottle, laptop, cell phone, banana, book, keyboard, mouse, chair… but **not keys**.

```bash
# 2a. the board on the desk: Waveshare ESP32-P4-NANO (ESP-IDF; see firmware/esp32-p4-client/README.md)
cd firmware/esp32-p4-client
cp main/secrets.h.example main/secrets.h          # fill in Wi-Fi + LOCI_HOST=http://<mac-ip>:8002
. ~/esp/esp-idf/export.sh && idf.py set-target esp32p4 && idf.py build
idf.py -p /dev/cu.usbmodem5ABA0768111 flash monitor

# 2b. an S3 / classic ESP32 camera board instead (PlatformIO; see firmware/esp32-cam-client/README.md)
cd firmware/esp32-cam-client
cp include/secrets.h.example include/secrets.h
pio run -e xiao_esp32s3_sense -t upload -t monitor
```

The serial monitor shows the camera sensor, Wi-Fi IP, and one line per frame with the labels the
host saw. Two rising beeps mean the speaker works; a third means Wi-Fi is up.

```bash
# 3. look and ask
open http://localhost:8002/esp32          # live frames, objects, typed questions
curl -s -X POST localhost:8002/api/esp32/ask -H 'content-type: application/json' \
     -d '{"text":"where is my cup"}' | jq .answer
```

Put a cup and a laptop in front of the board, wait two seconds, type "where is my cup": the answer
plays on the board. Hold the button and ask by voice if the board has a microphone.

```bash
# no board yet? pretend to be one (plays the answer on the Mac with afplay)
venv/bin/python demo_spatial/tools/esp32_fake_board.py --host http://localhost:8002 \
    --image venv/lib/python3.14/site-packages/ultralytics/assets/bus.jpg --fps 2 \
    --frames 8 --ask "what do you see"
```

### Environment variables (host)

| Variable | Default | Meaning |
|---|---|---|
| `LOCI_ESP32_MAX_FPS` | 5 | Frames per second the host will process per device; faster frames are dropped |
| `LOCI_ESP32_MIN_CONF` | 0.40 | YOLO confidence floor for the small camera (storage still needs ≥ 0.55 via consensus max) |
| `LOCI_ESP32_ROTATE` | 0 | 0/90/180/270 clockwise rotation applied host-side before detection |
| `LOCI_ESP32_YOLO_MODEL` | `$YOLO_MODEL` or `yolov8n.pt` | Weights for the ESP32 path |
| `LOCI_ESP32_AUDIO_RATE` | 16000 | PCM rate the board plays/records |
| `LOCI_ESP32_AUDIO_GAIN` | 1.0 | Linear gain applied to answers before queueing |
| `LOCI_ESP32_AUDIO_QUEUE_MAX` | 8 | Oldest clip dropped when a device falls this far behind |
| `LOCI_ESP32_STALE_S` | 10 | Device shown offline after this many silent seconds |
| `LOCI_ESP32_DEVICE` | `esp32` | Device id used when the board sends none |
| `LOCI_ESP32_WARMUP` | 0 | `1` loads YOLO in a thread at startup |
| `LOCI_ESP32_DEPTH` | `none` | `mono` enables a plugged-in monocular depth estimator (low confidence) |
| `LOCI_ESP32_WB` | `none` (`gray` in the run script) | Gray-world white balance before detection; fixes the P4 camera's blue cast |
| `LOCI_ESP32_IMGSZ` | 640 | YOLO inference size; 960 for 1280×960 frames |

Voice uses the existing `WHISPER_LOCAL`, `WHISPER_LOCAL_MODEL`, `TTS_ENGINE`, `PIPER_MODEL_PATH`,
`EDGE_TTS_VOICE`.

## 4. Latency budget (Mac M-series, Wi-Fi)

| Step | Measured / expected |
|---|---|
| Capture + JPEG (VGA q12) on the board | ~60–120 ms |
| POST over Wi-Fi (30–60 kB) | ~50–150 ms |
| YOLOv8n on the host (640 px) | ~30–80 ms CPU, less on MPS |
| Consensus → first stored observation | second agreeing frame, ≤ 1 s |
| STT (faster-whisper tiny, 2 s clip) | ~0.5–1 s |
| TTS (edge-tts, one sentence) → PCM | ~0.5–1.5 s |
| Audio fetch and playback start | < 100 ms |

Ask-to-answer is typically 2–3 s for a typed question and 3–4 s spoken.

## 5. Tests

`tests/test_pcm_audio.py`, `tests/test_esp32_source.py`, `tests/test_esp32_api.py` — fakes only:
Pillow-made JPEGs for the camera, a callable for YOLO, a coroutine for STT, a WAV tone for TTS, a
`MagicMock` for `SpatialMemory`. The `Esp32DetectionSource` tests skip on trees without
`app.detection_source`. `ruff check demo_spatial/ tests/` must stay clean.

## 6. Status and next steps

- ✅ Host: hub, voice bridge, PCM helpers, routes, status page, fake board, tests, run script.
- ✅ Firmware: PlatformIO project for XIAO ESP32S3 Sense, AI-Thinker ESP32-CAM, Freenove ESP32-S3 CAM.
- ✅ Verified on the Mac without hardware: the fake board's frames become observations, typed
  and spoken questions ("where is the bus?" synthesized, transcribed by faster-whisper tiny in
  ~0.6 s) come back as 16 kHz PCM; ask-to-audio 2.4–3.5 s with edge-tts.
- ✅ Board on the desk (2026-10-25): it is a **Waveshare ESP32-P4-NANO**, flashed with
  `firmware/esp32-p4-client`. Measured: Wi-Fi via the C6 at −40 to −50 dBm; 800×800 JPEGs of
  45–90 kB at 2.0 fps with 180–340 ms per POST; YOLOv8n 30–40 ms per frame on the Mac; typed
  "where is my cup" answered from the board's own observations and the 12 s reply played on the
  board's speaker; push-to-talk through the onboard mic transcribed "Where is my cup?" twice
  (3.8 s and 3.3 s clips), the second answer starting 1.3 s after the button was released.
- ✅ microSD on the P4 (64 GB, formatted FAT32 from the board): memory cache and spoken summary
  downloaded within 2 s of mounting (453 B of JSON, 9.6 s of speech) and played from the card by the
  `offline` command; 6 frames captured while the host was stopped were uploaded on reconnect and
  counted by the host as `backlog_frames`; the dataset reached 24 frames in the first minutes.
- ✅ Picture: with the ISP pipeline controller exposure is right and focus is sharp; the remaining
  blue cast is removed on the host by gray-world white balance (mean RGB 102/152/167 → 139/140/140).
  The camera on the desk was pointing at the ceiling; it has to be propped to look horizontally.
- ✅ Camera mode moved to 1280×960 with 2×2 binning (more light, more detail) and the host inference
  size to 960: frames of 135–145 kB, still 2.0 fps with 300–500 ms per POST, YOLOv8n 74 ms per frame
  on the Mac; the same desk view went from mostly empty labels to chair, microwave, refrigerator,
  dining table and person per frame.
- ✅ Rebased on `feat/handheld-oak-voice-demo`; the hub feeds `SceneIngestion.process_detections`.
  ⬜ Drop the legacy `_update_consensus` fallback once that branch is on `main`.
- ⬜ Push-to-talk over WebSocket (streaming STT) instead of one POST per utterance.
- ⬜ Optional monocular depth plug-in behind `LOCI_ESP32_DEPTH=mono`.
- ⬜ Pi as host: same script, `LOCI_HOST=http://loci-pi.local:8002` in `secrets.h`.
