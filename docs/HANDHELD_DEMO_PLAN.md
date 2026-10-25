# Handheld Spatial Memory Demo — Plan

**Device:** Raspberry Pi 5 + OAK-D Lite (USB) + Bluetooth headset, carried by hand.
**Promise:** walk through your home; later ask, by voice, where things are and what changed, and hear the answer, offline, including after a restart.
**Why it matters:** the same loop (perceive with depth, remember in a room frame, answer in natural language) is what a blind or low-vision user needs from an assistant, and what a home robot needs from its memory layer.

Status legend: ✅ done · 🔧 in progress · ⬜ planned

## 1. The experience we are building

> "Where did I leave my laptop today?"
> "Your laptop was last seen 25 minutes ago in the living room, about two metres left of the door, at table height. Earlier today it was in the bedroom at 9:10."

> "Where are the bananas?"
> "Bananas: in the kitchen, 40 minutes ago, on the counter to the right of the fridge."

> "What changed in the kitchen?"
> "Since your last visit the mug moved from the table to the counter, and I no longer see the bananas."

Interaction: press the headset button (push-to-talk), speak, release; the answer is spoken back within about three seconds. No phone is required for the voice loop; a phone page remains available for sighted helpers and for debugging.

## 2. Architecture

```
OAK-D Lite ── USB ──> Raspberry Pi 5 (Python 3.13, DepthAI v3)
 on camera: YOLOv6-nano (80 COCO classes) + stereo depth
            -> label, box, XYZ in metres per object, 10 fps
 on Pi:     AprilTag room anchoring (OpenCV aruco 36h11, camera calibration from the device)
            -> camera pose in the room frame; objects stored in room metres
            SceneIngestion consensus buffer -> SpatialMemory (LOCI, in-process, snapshotted to disk)
            label embeddings: model2vec potion-base-8M (256-d, offline)
            voice: headset mic -> faster-whisper (local) -> intent -> LOCI query -> answer text -> Piper TTS -> headset
            FastAPI: /api/* queries, WebSocket feed, phone page
```

Coordinate conventions (single source of truth in code): DepthAI spatial coordinates are +X right, +Y down, +Z forward (OpenCV); AprilTag pose from solvePnP gives the camera pose in the tag frame; each room is a box in metres with its origin at the room's anchor tag; room metres map to LOCI's [0, 1] cube per axis.

## 3. Verified so far (2026-10-24)

- ✅ Pi 5 (16 GB, Raspberry Pi OS trixie, Python 3.13.5) reachable over SSH; venv `~/loci-handheld/.venv` with DepthAI, OpenCV, FastAPI, model2vec; YOLO and embedding models staged offline on the Pi.
- ✅ Camera identified: OAK-D-LITE-AF with a BMI270 IMU; calibration read from the device.
- ✅ Build-order step 1 on the Mac against the real camera: on-camera YOLOv6-nano + stereo depth streaming labels and XYZ in metres at 10 fps (stereo preset FAST_DENSITY with 2 post-processing shaves; HIGH_DETAIL/DEFAULT do not fit next to the 8-shave blob).
- ✅ Embedding quality: cup/mug 0.47, keys/key chain 0.51, couch/sofa 0.68 vs cup/dog 0.14.
- 🔧 Camera on the Pi: not yet enumerating; the camera and a USB-C to USB-C cable work on the Mac, so the USB-C to USB-A cable is the suspect. The Pi's supply negotiates 3 A, so USB is capped at 600 mA; a 27 W PSU or a powered hub may be needed in addition to a data cable.

## 4. Build order and status

| # | Deliverable | Status |
|---|---|---|
| 1 | DepthAI script: live detections with XYZ printed to the console | ✅ on Mac, ⬜ on Pi |
| 2 | Adapter into SceneIngestion/SpatialMemory with metric coordinates | ⬜ |
| 3 | AprilTag room anchoring; observations stored in room coordinates | ⬜ |
| 4 | Phone page + typed queries; **headset voice loop** (push-to-talk, STT, TTS) | ⬜ |
| 5 | Battery walkaround test; tune consensus thresholds | ⬜ needs Pi + cable |
| 0 | Real label embeddings; fix per-observation query in `_try_merge`; persistence across restarts | ⬜ |

## 5. Voice track (headset)

- **Audio I/O:** one cross-platform module on PortAudio (`sounddevice`), so the same code records and plays on the Mac during development and on the Pi (PipeWire/ALSA) in the field. Device selection by name substring (e.g. the headset's Bluetooth name) with sensible fallbacks.
- **Trigger:** push-to-talk. On the Pi the headset's media button arrives as a Linux input event; a keyboard key or the phone button serve as the equivalent on the Mac. A wake word is a later option.
- **STT:** faster-whisper, `tiny` or `base` int8, English first; measured on the Pi before choosing.
- **Understanding:** reuse `voice_pipeline.parse_intent` and extend it with time expressions ("today", "this morning", "yesterday", "since lunch"), room names, plural/synonym normalisation through the embedder ("bananas" finds "banana").
- **Answers:** room first, then landmark-relative position (objects the camera also saw, like the fridge or the door tag), then time; confidence hedging ("I think") below a threshold; history on request; "what changed in <room>" from LOCI's time windows.
- **TTS:** Piper (offline); edge-tts only when online and explicitly enabled.
- **Latency budget:** STT 1–2 s, LOCI query < 50 ms, TTS 0.5–1 s.

## 6. How we try it end to end

**Stage A — on the Mac (available now).** Camera on the Mac over USB-C, headset paired to the Mac, LOCI and the voice loop running on the Mac. Desk test: put a laptop, a cup and a banana in view, hold the button, ask, hear the answer; move the cup, ask again; restart the process, ask again. This exercises every component except the Pi hardware path.

**Stage B — on the Pi (after a data cable).** `deploy.sh` rsyncs the repo to `~/loci-handheld`, installs pins into the venv, installs the service. Pair the headset, print and mount two AprilTags (one per room). Walk the two-room story: put a mug down, walk away, ask where it is, move it, ask again, restart the Pi, ask again.

## 7. Acceptance criteria for the demo

- Object seen once for ≥ 1 s while a tag is visible is answered correctly in room terms, with position error under ~0.5 m at ≤ 3 m range.
- "Where is X" answers in ≤ 4 s from button release, offline.
- Memory survives a process restart and a Pi reboot.
- "What changed in <room>" reports moved and disappeared objects within the asked window.
- Unknown or never-seen labels get an honest answer ("I haven't seen a wallet") rather than a guess.

## 8. Risks and open questions

- COCO vocabulary: laptop, banana, cup, bottle, cell phone, backpack, book, remote, chair are covered; keys, wallet, glasses, medicine are not (custom model or optional VLM fallback later).
- Pose hold between tag sightings: start with a timed hold of the last pose; the IMU is an upgrade path.
- Depth noise grows with range (≈ ±1 m at 6 m); clamp the usable range to ~5 m and rely on the consensus buffer.
- Pi power: 3 A supplies cap USB at 600 mA; the camera may need `usb_max_current_enable=1` or a powered hub.
- Headset audio on Linux (HFP microphone profile vs A2DP playback) can be finicky; test the chosen headset early.
- Identical objects share a label; answers give the latest sighting and history on request.

## 9. After the demo

- ROS 2 node exposing recall and novelty as services, LeRobot loader, and the world-model memory benchmark (already on the roadmap) reuse the same memory API; the handheld is the lab platform named in the IDD-64 form-factor plan.
