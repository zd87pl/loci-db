# Hardware verification plan — LOCI handheld (Pi 5 + OAK-D Lite + Bluetooth headset)

This is design section 14 (`docs/HANDHELD_DEMO_DESIGN.md`) expanded into a checkbox plan with the
real commands, flags, environment keys, endpoints and file names of the implementation. Verified
camera facts (`docs/HANDHELD_HARDWARE_FACTS.md`) override the design wherever they differ: stereo
preset `FAST_DENSITY` with `setPostProcessingHardwareResources(2, 2)` is the only preset that fits
next to the 8-shave YOLO blob; detections without depth come back as exact-zero
`spatialCoordinates` and are dropped; spatial axes are +x right, +y down, +z forward (RDF); a
device must be back in `X_LINK_UNBOOTED` before it can be reopened.

Tick every box in order. When an item fails, apply its **If it fails** line, re-run the item, and
record what you changed in the results record (Appendix B). Never skip a failed item silently:
later steps assume the earlier ones.

## Conventions

| Prompt | Where | Preparation |
|---|---|---|
| `mac$` | the Mac, repo root `/Users/0x7d0/git/loci-db`, venv active (Stage A.1) | — |
| `pi$` | `ssh ziggy@loci-pi.local` | `source /home/ziggy/loci-handheld/.venv/bin/activate; set -a; source /home/ziggy/loci-handheld/env; set +a` |
| `phone` | a phone browser on the same Wi-Fi (or the hotspot) | — |

Paths on the Pi: `/home/ziggy/loci-handheld/{src,.venv,env,rooms.json,data,models,certs}`;
scripts live under `/home/ziggy/loci-handheld/src/deploy/handheld/`. In the commands below
`$S` is that scripts directory:

```bash
pi$  S=/home/ziggy/loci-handheld/src/deploy/handheld
mac$ S=deploy/handheld
```

JSON helper (no `jq` needed; prints dotted paths from stdin):

```bash
j() { python3 -c 'import json, sys
d = json.load(sys.stdin)
for p in sys.argv[1:]:
    v = d
    for k in p.split("."):
        v = v.get(k) if isinstance(v, dict) else (v[int(k)] if isinstance(v, list) and k.lstrip("-").isdigit() else None)
        if v is None:
            break
    print(f"{p} = {json.dumps(v)}")' "$@"; }
```

Reading the console's 1 Hz summary line (`oak_console.py`):

```
fps=10.1 det/s=4.0 tags/s=5.0 labels={cup:3,bottle:1} state=running usb=HIGH reconnects=0
 │        │         │          │                       │             │        └ supervisor re-opens so far
 │        │         │          │                       │             └ HIGH (USB 2) is fine; SUPER = USB 3
 │        │         │          │                       └ starting | running | waiting_for_camera | stopped | error | done
 │        │         │          └ labels in the last second (spaces -> underscores)
 │        │         └ AprilTag RGB stream frames/s (expect ~5.0 with LOCI_TAG_STREAM=1280x800@5; 0.0 = stream not running)
 │        └ detections per second (objects, not frames)
 └ detection frames per second from the camera (expect 8-10)
```

Endpoint names and status keys were taken from `demo_spatial/app/main.py` and
`demo_spatial/app/handheld.py` (T7) and `deploy/handheld/smoke_test.sh` (T8). On the first
server start confirm them once anyway: `curl -s localhost:8001/openapi.json | j paths` must list
`/api/query`, `/api/handheld/status`, `/api/anchor`, `/api/rooms`, `/api/objects/changes`,
`/api/handheld/snapshot`, `/api/memory/clear` and `/phone`. `/api/handheld/status` returns
`{config, runtime, memory, restore, started_at_ms, voice, errors, anchor, ws_subscribers}`;
`errors` is a dict of startup failures (`restore`, `voice`, ...) and is the first thing to read
when a check below comes back `null`.

## Fallback ladders (read before you start)

| Symptom | First | Then | Last resort |
|---|---|---|---|
| camera not in `lsusb` / `system_profiler` | data-capable cable (many USB-C to USB-A cables are charge-only) | blue USB-A port (the Pi's USB-C is power only); on macOS 26 Privacy & Security > "Allow accessories to connect" = Always | `usb_max_current_enable=1` + reboot; powered hub or OAK Y-adapter |
| pipeline boot fails `Blob compiled for 8 shaves, but only 7 are available` | keep `LOCI_OAK_STEREO_PRESET=FAST_DENSITY` and `LOCI_OAK_PP_SHAVES=2` (defaults) | `LOCI_OAK_PP_SHAVES=3` | `LOCI_TAG_STREAM=off` (or `--no-tags`); `LOCI_OAK_USB2=1` (`--usb2`) |
| `X_LINK_DEVICE_NOT_FOUND` | cable/port as above | udev rule `80-movidius.rules` (step 0.5) | — |
| `X_LINK_INSUFFICIENT_PERMISSIONS` / `X_LINK_ERROR` right after a failure | wait: the source polls for `X_LINK_UNBOOTED` (up to ~10 s) | udev rule | replug |
| `tags/s=0.0` while `state=running` | `LOCI_TAG_STREAM=640x400@10` and 24 cm tags (`make_tag_sheet.py --size-mm 240`) | `LOCI_TAG_UNDISTORT=0` (default; on-device undistortion costs a shave) | `LOCI_TAG_STREAM=off`: camera-frame-only demo |
| `--check-y-sign` says `Y appears UP` | `LOCI_OAK_AXES=ruf` in `env`, re-run | tilt check (camera level, cup clearly above the lens centre) | — |
| tag pose off by > 3 % in z, or mirrored | re-measure the printed black square -> `rooms.json tag_size_m` | `LOCI_TAG_UNDISTORT=1` | `LOCI_TAG_UPRIGHT_FIX=1` only when tags were printed in the AprilRobotics canonical orientation |
| static embedder fails at startup (`LOCI_LABEL_EMBEDDER=static`) | `fetch_models.py --verify` / `--only-missing` | `LOCI_LABEL_EMBEDDER=auto` | `hash` |
| headset plays but records silence | HFP profile: `wpctl set-profile <card> headset-head-unit` (or `headset-head-unit-msbc`) | `LOCI_AUDIO_BACKEND=cli` (pw-record/pw-play) | wired USB headset |
| headset button ignored | `LOCI_PTT_DEVICE` substring from `/proc/bus/input/devices`, user in group `input` | `LOCI_PTT_TOGGLE=1` (click-only headsets) | `LOCI_PTT=vad` (hands-free) |
| `OSError: PortAudio library not found` | `sudo apt install libportaudio2` (Pi) / `brew install portaudio` (Mac) | `LOCI_AUDIO_BACKEND=cli` | — |
| camera drops on battery, `get_throttled` != `0x0` | `usb_max_current_enable=1` | 27 W supply / powered hub | OAK Y-adapter |

---

## Stage A — everything on the Mac (camera on USB-C, headset paired to the Mac)

Stage A exercises every component except the Pi hardware path (design plan section 6). The
server, the camera supervisor, persistence, the phone page and the headset loop all run on the Mac.

- [ ] **A.1 Mac venv with the Pi pins.** The Homebrew Python 3.14 has no depthai/cv2 wheels; use
  3.11 or 3.12. Put the venv where `run_pi.sh` finds it by default (`$HOME/loci-handheld/.venv`).
  ```bash
  mac$ brew install portaudio                       # PortAudio for python-sounddevice (headset I/O)
  mac$ python3.11 -m venv "$HOME/loci-handheld/.venv"
  mac$ "$HOME/loci-handheld/.venv/bin/pip" install -e . -r deploy/handheld/requirements-dev.txt
  mac$ source "$HOME/loci-handheld/.venv/bin/activate"
  mac$ python -c "import depthai, cv2, pyapriltags, tokenizers, safetensors, sounddevice, fastapi; print('ok', depthai.__version__)"
  ```
  Expect: `ok 3.11.0`.
  If it fails: `piper-tts` or `faster-whisper` wheels failing on macOS are not blocking: comment
  them out of a copy of `requirements-pi.txt` (the block between `# --- voice` and `# --- end voice`)
  and keep `sounddevice`; later use `--no-tts` / `--loopback-text` in the voice items and
  `TTS_ENGINE=openai` (needs `OPENAI_API_KEY`) if you want spoken answers on the Mac. The session
  scratch venv `/private/tmp/claude-501/-Users-0x7d0-git-loci-db/8139ebf0-1ba6-436c-8885-212cc37c9445/scratchpad/oakenv`
  already has depthai 3.11.0 and cv2; point `LOCI_VENV` at it instead if you prefer.

- [ ] **A.2 Models on the Mac.**
  ```bash
  mac$ python deploy/handheld/scripts/fetch_models.py --list
  mac$ python deploy/handheld/scripts/fetch_models.py --only-missing --whisper tiny.en   # zoo via depthai, potion + piper from the Hub
  mac$ python deploy/handheld/scripts/fetch_models.py --verify
  ```
  Expect: a status table with `ok` for `zoo`, `embed/model.safetensors`, `embed/tokenizer.json`,
  `embed/config.json`, `piper/en_US-lessac-medium.onnx`, `piper/en_US-lessac-medium.onnx.json`
  under `deploy/handheld/models/` (repo layout: `models/zoo`, `models/potion-base-8M`,
  `models/piper`); exit code 0.
  If it fails: no network -> copy `models/` from a machine that has them; `zoo missing` with
  `depthai` importable means the HubAI download failed, retry or follow the printed recipe;
  `--whisper` needs `faster_whisper` importable (skip it and use `--loopback-text` later).

- [ ] **A.3 Camera enumerates on the Mac.** Connect the OAK-D Lite with the USB-C to USB-C cable
  that worked before (USB 2 is enough).
  ```bash
  mac$ system_profiler SPUSBDataType 2>/dev/null | grep -i -B2 -A6 "movidius\|luxonis"
  ```
  Expect: `Movidius MyriadX` with Vendor ID `0x03e7`, Product ID `0x2485` (unbooted). After a
  pipeline ran it re-enumerates as `Luxonis Device` (`0xf63b`); both are fine.
  If it fails: macOS 26 Privacy & Security > "Allow accessories to connect" must be `Always`;
  try the other USB-C port; a stale `Device ... has crashed. Crash dump logs` line on the next open
  is harmless.

- [ ] **A.4 Simulator sanity (no camera).**
  ```bash
  mac$ python deploy/handheld/scripts/oak_console.py --sim --seconds 5; echo exit=$?
  ```
  Expect: banner `source=sim scenario=mug_two_rooms fps=10 frames_total=180 realtime=True seed=0`,
  detection lines like `14:51:58.837  cup          0.85  x=+0.99m y=+0.57m z=1.48m  box=(0.93,0.89,1.00,1.00)`,
  1 Hz summaries with `state=running`, a final `done: frames=... detections=... tag_frames=... labels={bottle:..,cup:..} elapsed=5.0s state=stopped` and `exit=0`.
  If it fails: a traceback here is a software problem (numpy/import), not hardware; run
  `/opt/homebrew/bin/pytest tests/test_oak_console.py tests/test_detection_source.py -q -p no:cacheprovider`.

- [ ] **A.5 Live camera for 30 s.**
  ```bash
  mac$ export DEPTHAI_ZOO_CACHE_PATH="$PWD/deploy/handheld/models/zoo" DEPTHAI_ZOO_INTERNET_CHECK=0
  mac$ python deploy/handheld/scripts/oak_console.py --seconds 0            # dry run: prints the resolved config, opens nothing
  mac$ python deploy/handheld/scripts/oak_console.py --seconds 30; echo exit=$?
  ```
  Expect (dry run): `source=oak config={"axes": "rdf", "bbox_scale": 0.5, ..., "model": "yolov6-nano", "pp_shaves": 2, ..., "stereo_preset": "FAST_DENSITY", "tag_stream": [1280, 800, 5.0], "tag_undistort": false, "usb2": false}`
  followed by `dry run: source constructed, device not opened (omit --seconds to stream)`, exit 0.
  Expect (live): first detection line within ~2 s of start (pipeline start ~0.6 s, first
  detections ~0.9 s later); summaries `fps=9.5..10.2 ... state=running usb=HIGH reconnects=0`
  and `tags/s≈5.0`; `exit=0`. Boxes without depth print `(no depth)` instead of `x= y= z=`.
  If it fails: follow the "pipeline boot fails" and "tags/s=0.0" ladders; record the exact error
  text from the summary's `error="..."` field in Appendix B. `tags/s=0.0` with
  `state=running` means the second 1280x800 NV12 output did not start next to the detector:
  `LOCI_TAG_STREAM=640x400@10 python deploy/handheld/scripts/oak_console.py --seconds 30`.

- [ ] **A.6 Y sign.** Hold a cup clearly above the lens centre (camera level) for the whole run.
  ```bash
  mac$ python deploy/handheld/scripts/oak_console.py --seconds 20 --check-y-sign
  ```
  Expect: `y-sign check: n_above=.. mean_y_above=-0.xx n_below=.. mean_y_below=+0.xx` and
  `verdict: RDF confirmed (above-centre objects have y<0)`.
  If it fails: `verdict: Y appears UP: set LOCI_OAK_AXES=ruf` -> set `LOCI_OAK_AXES=ruf` and
  re-run (`--axes ruf` tests it without editing env; expect `RDF confirmed` then);
  `Y sign inconclusive` -> hold the cup higher/lower so boxes land at `cy<0.4` or `cy>0.6`.

- [ ] **A.7 Range and lateral sign.** Tape at 1.00 m and 2.00 m on the optical axis; cup on the tape.
  ```bash
  mac$ python deploy/handheld/scripts/oak_console.py --seconds 15 --rate 1
  ```
  Expect: `z=` within ±5 % (0.95-1.05 m, 1.90-2.10 m); cup moved to the camera's right ->
  `x=+..`; depth jitter at 1-2 m well under ±0.05 m.
  If it fails: a constant scale error means the depth alignment/calibration is off -> read
  `device.readCalibration()` (not a config knob; report it); noisy z at < 0.4 m is the MinZ limit
  (try `LOCI_OAK_STEREO_EXTENDED=1`, keep it only if the pipeline still boots).

- [ ] **A.8 Reconnect supervisor.** Start a 60 s run, unplug the camera after 10 s, replug after 10 s.
  ```bash
  mac$ python deploy/handheld/scripts/oak_console.py --seconds 60
  ```
  Expect: summaries go `state=running` -> `state=waiting_for_camera ... reconnects=1 error="..."`
  -> `state=running ... reconnects=1`; the process never exits by itself; `exit=0`.
  If it fails: a stuck `waiting_for_camera` for > 60 s after replug means the device never
  returned to `X_LINK_UNBOOTED` (macOS accessory prompt, or the device needs a power cycle);
  the supervisor keeps polling, Ctrl-C once stops cleanly.

- [ ] **A.9 Record and replay.**
  ```bash
  mac$ mkdir -p "$HOME/loci-handheld/data"
  mac$ python deploy/handheld/scripts/oak_console.py --seconds 60 --record "$HOME/loci-handheld/data/mac_step1.jsonl"
  mac$ python deploy/handheld/scripts/oak_console.py --replay "$HOME/loci-handheld/data/mac_step1.jsonl" --no-realtime; echo exit=$?
  ```
  Expect: `recording -> .../mac_step1.jsonl`, `recorded N lines -> ...` (N ≈ 600 det + 300 tag
  lines); the replay prints `source=replay path=... realtime=False` and ends with `state=done`,
  `exit=0`. Keep the file: it is hardware-free regression data (`LOCI_DETECTION_SOURCE=replay`
  + `LOCI_REPLAY_PATH=...` runs the whole server against it).

- [ ] **A.10 Server env file on the Mac.** `run_pi.sh` uses `$HOME/loci-handheld` as `LOCI_HOME`
  when the checkout is not called `src`, sources `$LOCI_HOME/env`, and runs `$LOCI_HOME/.venv`.
  ```bash
  mac$ REPO="$PWD"; mkdir -p "$HOME/loci-handheld/data"
  mac$ sed -e "s#/home/ziggy/loci-handheld/models/embeddings/potion-base-8M#$REPO/deploy/handheld/models/potion-base-8M#" \
           -e "s#/home/ziggy/loci-handheld/models/zoo#$REPO/deploy/handheld/models/zoo#" \
           -e "s#/home/ziggy/loci-handheld/models/piper#$REPO/deploy/handheld/models/piper#" \
           -e "s#/home/ziggy/loci-handheld/rooms.json#$REPO/deploy/handheld/config/rooms.example.json#" \
           -e "s#/home/ziggy/loci-handheld#$HOME/loci-handheld#g" \
           -e "s#^LOCI_LABEL_EMBEDDER=static#LOCI_LABEL_EMBEDDER=auto#" \
           deploy/handheld/env.example > "$HOME/loci-handheld/env"
  mac$ grep -E '^(LOCI_DETECTION_SOURCE|LOCI_DATA_DIR|LOCI_EMBED_MODEL_DIR|LOCI_ROOMS_FILE|LOCI_VOICE|PIPER_MODEL_PATH|DEPTHAI_ZOO_CACHE_PATH)=' "$HOME/loci-handheld/env"
  ```
  Expect: `LOCI_DETECTION_SOURCE=oak`, `LOCI_VOICE=none` (voice comes in A.16), absolute Mac
  paths, no `/home/ziggy` left. `LOCI_LABEL_EMBEDDER=auto` falls back to `hash` if the potion
  files are missing instead of refusing to start.

- [ ] **A.11 Server on the Mac.**
  ```bash
  mac$ deploy/handheld/run_pi.sh                                  # foreground; Ctrl-C stops
  # second terminal:
  mac$ curl -s localhost:8001/health | j status pipeline.stt_available pipeline.tts_engine
  mac$ curl -s localhost:8001/api/handheld/status | j runtime.source.state runtime.source.fps runtime.source.usb_speed memory.embedder restore.source
  mac$ curl -s localhost:8001/openapi.json | j paths | tr ',' '\n' | grep -c '/api/'
  ```
  Expect: `run_pi.sh` prints `LOCI handheld: source=oak voice=none tls=0` and the phone URL,
  uvicorn logs `Application startup complete`; `status = "ok"`; after ~3 s
  `runtime.source.state = "running"`, `fps ≈ 10`, `usb_speed = "HIGH"`,
  `memory.embedder = "static:potion-base-8M"` (or `hash:128` with the fallback),
  `restore.source = "empty"` on the first start.
  If it fails: `ModuleNotFoundError` -> the venv in `$HOME/loci-handheld/.venv` is not the one
  with the pins (or set `LOCI_VENV`); `state = "waiting_for_camera"` -> the camera is still held
  by a previous console run (wait 10 s) or A.3; `static` embedder error -> A.2 / ladder.

- [ ] **A.12 Camera-frame objects and typed queries.** Put a cup on the desk 1-2 m away for 5 s.
  ```bash
  mac$ curl -s localhost:8001/api/objects | j count objects.0.label objects.0.frame objects.0.xyz_cam_m objects.0.anchored
  mac$ curl -s -X POST localhost:8001/api/query -H 'content-type: application/json' -d '{"text":"where is my mug"}' | j resolution.label resolution.how latency_ms answer
  mac$ curl -s -X POST localhost:8001/api/query -H 'content-type: application/json' -d '{"text":"where are my keys"}' | j resolution.label answer
  mac$ deploy/handheld/smoke_test.sh --expect-object cup --snapshot
  ```
  Expect: `cup` with `frame = "camera"`, `xyz_cam_m = [x, y, z]`, `anchored = false`;
  `resolution.label = "cup"`, `resolution.how = "alias"`, `latency_ms < 300`, an answer with
  "in front of the camera" and the suffix "roughly — I wasn't sure of my position"; the keys
  query returns `resolution.label = null` and "I can't recognise 'keys' with this camera ...";
  `smoke_test.sh` ends with `all N checks passed`.
  If it fails: no `cup` after 5 s -> check `/api/handheld/status | j runtime.ingestion`
  (`fires` must grow; `suppressed_by_holdoff` grows with `LOCI_CONSENSUS_REFIRE_MS=2000`), and
  that the console in A.5 saw `cup` at all (lighting, 0.5 confidence threshold `LOCI_OAK_CONF`).

- [ ] **A.13 Phone page from the Mac and a phone.**
  ```bash
  mac$ open http://localhost:8001/phone
  mac$ ipconfig getifaddr en0          # phone opens http://<this ip>:8001/phone
  ```
  Expect: status strip with source state and fps, "seeing now" updating within 1 s; typing
  "Where is my mug?" answers in < 1 s and the answer sits in the `aria-live` region (VoiceOver
  reads it); "Speak" uses browser speech.
  If it fails: the phone cannot reach the Mac -> same Wi-Fi, macOS firewall prompt for Python.

- [ ] **A.14 Restart persistence on the Mac.** Ctrl-C the server, start it again, wait 5 s.
  ```bash
  mac$ ls -la "$HOME/loci-handheld/data"
  mac$ curl -s localhost:8001/api/handheld/status | j restore.records restore.source restore.bad_lines memory.persistence.journal_lines
  mac$ curl -s -X POST localhost:8001/api/query -H 'content-type: application/json' -d '{"text":"where is my mug"}' | j answer
  mac$ deploy/handheld/smoke_test.sh --restored --expect-state any
  ```
  Expect: `memory.snapshot.json`, `memory.snapshot.bak.json` (after the second snapshot),
  `memory.journal.jsonl` in `data/`; `restore.records > 0`, `restore.source = "snapshot"` or
  `"snapshot+journal"`, `bad_lines = 0`; the answer ends with "That was before I restarted."
  If it fails: `restore.source = "corrupt"` -> a `memory.snapshot.corrupt-<ts>.json` was
  quarantined and `.bak` used (report it); `"empty"` -> `LOCI_PERSIST` or `LOCI_DATA_DIR` differ
  between the two runs.

- [ ] **A.15 Headset paired to the Mac.** System Settings > Bluetooth: pair the headset; System
  Settings > Sound: select it for **both** output and input (input switches the headset to the
  HFP profile; lower playback quality while the mic is open is expected).
  ```bash
  mac$ python deploy/handheld/scripts/voice_console.py --list-devices
  ```
  Expect: `platform: darwin; sounddevice: True`, an `audio devices (sounddevice)` list with the
  headset name tagged `(in/out ...)`, and `input (button) devices ...: (none; not Linux ...)`
  (the headset button is not readable on macOS; use the keyboard or VAD here).
  If it fails: `sounddevice: False` -> A.1 (`brew install portaudio`, `pip install sounddevice`);
  headset listed as `(out)` only -> select it as the input device in Sound settings or pass
  `--audio-in "<name substring>"`.

- [ ] **A.16 Loopback through the running server (no mic).** Server from A.11 still running.
  ```bash
  mac$ python deploy/handheld/scripts/voice_console.py --loopback-text "where is my mug" --tts-out "$HOME/loci-handheld/data/answer.wav"; echo exit=$?
  mac$ afplay "$HOME/loci-handheld/data/answer.wav"
  ```
  Expect: event lines `[   x.x ms] transcript  where is my mug`, `answer  Your mug (I track it
  as a cup) ...`, `tts`, `done`, then one JSON line with `"answer"` and `"tts_out"`; `exit=0`;
  `afplay` speaks the answer.
  If it fails: `server ... unreachable ... answering from the canned path` -> the server is
  not on `127.0.0.1:8001` (`--host/--port`); `tts_unavailable` -> `TTS_ENGINE`/`PIPER_MODEL_PATH`
  in the shell env (the console reads the process env, not `$HOME/loci-handheld/env`:
  `set -a; source "$HOME/loci-handheld/env"; set +a` first) or add `--no-tts` to test the rest.

- [ ] **A.17 One real utterance, keyboard push-to-talk.** Needs `WHISPER_LOCAL=1` in the shell
  env (faster-whisper `tiny.en`; first run downloads it unless A.2 cached it).
  ```bash
  mac$ set -a; source "$HOME/loci-handheld/env"; set +a
  mac$ python deploy/handheld/scripts/voice_console.py --ptt keyboard --audio sounddevice --turns 1; echo exit=$?
  ```
  Press Enter, say "where is my mug", press Enter.
  Expect: events in order `started`, `press`, `earcon` (short 880 Hz beep in the headset),
  `recorded`, `transcript  where is my mug` (1-2 s), `answer ...`, `tts`, `played`, `done`;
  `exit=0`. Total `done` latency ≤ 4 s.
  If it fails: `KeyboardPTT needs an interactive terminal` -> run it in a real terminal, not
  through a pipe; empty transcript ("Sorry, I didn't catch that") -> the input device is not the
  headset (`--audio-in`), or speak after the earcon; `recorded` but silence -> HFP not active
  (re-select the headset as input in Sound settings).

- [ ] **A.18 Hands-free loop inside the server (VAD).** Stop the console; edit
  `$HOME/loci-handheld/env`: `LOCI_VOICE=headset`, `LOCI_PTT=vad`, `LOCI_AUDIO_BACKEND=sounddevice`
  (optionally `LOCI_AUDIO_IN=<headset substring>`, `LOCI_AUDIO_OUT=<headset substring>`);
  restart `run_pi.sh`.
  ```bash
  mac$ curl -s localhost:8001/api/handheld/status | j voice.running voice.ptt voice.audio voice.turns voice.errors
  # say "where is my mug", wait for the spoken answer, then:
  mac$ curl -s localhost:8001/api/handheld/status | j voice.turns voice.last_events
  mac$ deploy/handheld/smoke_test.sh --voice --expect-state any
  ```
  Expect: `voice.running = true`, `voice.ptt = "vad"`, `voice.audio = "sounddevice"`; after the
  question `voice.turns = 1` and `last_events` containing `transcript`, `answer`, `played`,
  `done` with `latency_ms`; the answer is spoken in the headset.
  If it fails: VAD never triggers -> lower `LOCI_VAD_THRESHOLD` (500 -> 200; it is RMS on the
  int16 scale) or raise it if it triggers on room noise; it cuts you off -> raise
  `LOCI_VAD_SILENCE_S` (1.2 -> 2.0); the answer re-triggers the VAD -> use the headset (not
  speakers) for output, or go back to `LOCI_VOICE=none` and the console with `--ptt keyboard`.

- [ ] **A.19 Desk story (acceptance of the Mac stage).** Laptop, cup and banana in view for 5 s;
  ask "where is my mug", "where is the banana", "what do you see"; move the cup ≥ 1.5 m, wait
  5 s, ask again; ask "what changed"; restart the server; ask again.
  Expect: each answer names the right object, camera-frame wording with the "roughly" suffix
  (no tag on the Mac desk), the moved cup gets a new distance, "what changed" reports
  `the cup moved about N metres` only once a room frame exists (camera-frame sightings yield
  `new`/`seen` only, so without a tag expect "Nothing changed ..." or a `new` entry: this is by
  design), and after the restart the answer ends with "That was before I restarted."

- [ ] **A.20 Optional: one tag on the Mac desk wall.** `A.1`'s venv has cv2 and pyapriltags.
  ```bash
  mac$ python deploy/handheld/scripts/make_tag_sheet.py --ids 0 --size-mm 160 --out deploy/handheld/tags
  mac$ open deploy/handheld/tags/tag36h11_000_160mm.pdf        # print at 100 %, measure, tape at ~1.4 m
  mac$ curl -s localhost:8001/api/anchor | j room frame anchored anchor_confidence camera_xyz_m heading_deg tag_ids age_s
  mac$ curl -s localhost:8001/api/handheld/status | j runtime.localizer.detector runtime.localizer.poses_accepted runtime.localizer.poses_rejected
  ```
  Expect: facing the tag from 2 m: `room = "kitchen"`, `frame = "room"`, `anchored = true`,
  `camera_xyz_m[2] ≈ 2.0`, `heading_deg ≈ 0`, `tag_ids = [0]`; `detector = "pyapriltags"` (or
  `aruco`), `poses_accepted` growing. This proves the tag chain before any Pi work (step 3 repeats
  it with a tape measure).

---

## Step 0 — Pi preconditions (once)

- [ ] **0.1 Cable and port.** Camera on a **blue USB-A** port with a data-capable cable.
  ```bash
  pi$ lsusb | grep -i 03e7
  pi$ sudo dmesg -w          # plug the camera in while this runs; Ctrl-C after
  ```
  Expect: `Bus 00x Device 00y: ID 03e7:2485 Intel Movidius MyriadX`; `dmesg` shows
  `new high-speed USB device` (or SuperSpeed) and no `device descriptor read error`.
  If it fails: nothing in `lsusb` and no attach event at all = charge-only cable (the earlier
  USB-C to USB-A cable did exactly this); the USB-C socket of the Pi is power only.

- [ ] **0.2 USB current.**
  ```bash
  pi$ vcgencmd get_config usb_max_current_enable
  pi$ od -An -tx4 /proc/device-tree/chosen/power/max_current
  ```
  Expect: `usb_max_current_enable=1` and ` 00001388` (5000 mA budget).
  If it fails (`=0`): `sudo sh -c 'echo usb_max_current_enable=1 >> /boot/firmware/config.txt' && sudo reboot`
  (the one mandatory sudo step), then re-check.

- [ ] **0.3 Power and thermal baseline.**
  ```bash
  pi$ vcgencmd get_throttled; vcgencmd measure_temp
  ```
  Expect: `throttled=0x0`, `temp` < 70 °C.
  If it fails: a non-zero `throttled` on the bench PSU means the supply is too weak before the
  camera is even drawing; use the 27 W supply.

- [ ] **0.4 Deploy and install.**
  ```bash
  mac$ deploy/handheld/deploy_from_mac.sh --no-restart
  pi$  bash /home/ziggy/loci-handheld/src/deploy/handheld/install_pi.sh
  ```
  Expect: rsync stats, then install sections `Layout`, `Steps that need sudo`, `Python venv`
  (`depthai 3.11.0`, `cv2 5.0.0`, `pyapriltags 3.4.3.1`, `tokenizers 0.23.3`, `safetensors 0.8.0`),
  `Models` (`ok models complete`), `Configuration` (`env` / `rooms.json` written or present),
  `Self-test` (`import OK: depthai 3.11.0 cv2 5.0.0`, `ok voice imports` or a WARN naming the
  missing wheel), `systemd user service` (`installed` or `unit up to date`), and `Done` with the
  warning list (acceptable warnings: pending sudo steps, voice stack incomplete).
  If it fails: a voice wheel (sounddevice / faster-whisper / piper-tts) failing on cp313/aarch64
  makes the installer retry without the voice block and warn; typed queries still work and the
  voice items (4V) will need the missing wheel. `systemctl --user` unavailable over ssh ->
  `export XDG_RUNTIME_DIR=/run/user/$(id -u)` and re-run with `--skip-pip --no-models`.

- [ ] **0.5 Permissions and groups.**
  ```bash
  pi$ cat /etc/udev/rules.d/80-movidius.rules
  pi$ id -nG ziggy
  pi$ loginctl show-user ziggy | grep Linger
  ```
  Expect: the rule `SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"`; groups include
  `plugdev video audio input` (`input` is needed for the headset button in 4V.3); `Linger=yes`.
  If it fails: `install_pi.sh` printed the exact sudo lines; add `sudo usermod -aG input ziggy`
  and log out/in (a new ssh session) before 4V; `sudo loginctl enable-linger ziggy`.

- [ ] **0.6 Env and rooms, dry run of the OAK config.**
  ```bash
  pi$ diff <(grep -oE '^#?[A-Z][A-Z0-9_]*=' $S/env.example | sort -u) <(grep -oE '^#?[A-Z][A-Z0-9_]*=' /home/ziggy/loci-handheld/env | sort -u)
  pi$ python3 -m json.tool /home/ziggy/loci-handheld/rooms.json | head -5
  pi$ python $S/scripts/oak_console.py --seconds 0; echo exit=$?
  ```
  Expect: an empty diff (same key set), `"format": 1` and `"tag_size_m": 0.16` in `rooms.json`,
  and `source=oak config={"axes": "rdf", "bbox_scale": 0.5, ..., "model": "yolov6-nano", "pp_shaves": 2, ..., "stereo_preset": "FAST_DENSITY", "tag_stream": [1280, 800, 5.0], "tag_undistort": false, "usb2": false}`
  plus `dry run: source constructed, device not opened (omit --seconds to stream)` with `exit=0`.
  If it fails: `error: ...` on stderr with exit 2 names the bad `LOCI_OAK_*` value in `env`.

- [ ] **0.7 Models on the Pi.**
  ```bash
  pi$ python $S/scripts/fetch_models.py --list --models-dir /home/ziggy/loci-handheld/models --layout pi
  pi$ ls /home/ziggy/loci-handheld/models/zoo/*/ /home/ziggy/loci-handheld/models/embeddings/potion-base-8M /home/ziggy/loci-handheld/models/piper
  ```
  Expect: every row `ok`; `YOLOv6_Nano-R2_COCO_512x384.rvc2.tar.xz` under `zoo/<hash>/`,
  `model.safetensors tokenizer.json config.json`, `en_US-lessac-medium.onnx{,.json}`.
  If it fails: `fetch_models.py --only-missing --models-dir /home/ziggy/loci-handheld/models --layout pi`
  (network) or `rsync` the directories from the Mac (`models/README.md`). A potion directory
  written earlier by model2vec may fail `--verify` while still loading; presence is what matters.

---

## Step 1 — camera console on the Pi (`oak_console.py`, no server)

- [ ] **1.1 Simulator sanity.**
  ```bash
  pi$ python $S/scripts/oak_console.py --sim --seconds 5; echo exit=$?
  ```
  Expect: as A.4 (`source=sim scenario=mug_two_rooms ...`, detection lines, `exit=0`).

- [ ] **1.2 First camera contact, 30 s.**
  ```bash
  pi$ python $S/scripts/oak_console.py --seconds 30; echo exit=$?
  ```
  Expect: `source=oak config={...}`, detection lines within ~2 s, summaries with
  `fps≥8 ... state=running usb=HIGH|SUPER reconnects=0` and `tags/s≈5.0`, `exit=0`. Note the
  `usb=` value in Appendix B (HIGH is fine).
  If it fails:
  1. boot error with `8 shaves` in the text -> `LOCI_OAK_PP_SHAVES=3 python $S/scripts/oak_console.py --seconds 30`;
  2. still failing -> `python $S/scripts/oak_console.py --seconds 30 --no-tags` (= `LOCI_TAG_STREAM=off`);
  3. still failing -> add `--usb2`;
  4. `X_LINK_DEVICE_NOT_FOUND` -> 0.1 / 0.5; `X_LINK_INSUFFICIENT_PERMISSIONS` -> udev rule, else wait 10 s (the source waits for `X_LINK_UNBOOTED`);
  5. `tags/s=0.0` while running -> `LOCI_TAG_STREAM=640x400@10 python $S/scripts/oak_console.py --seconds 30` and plan on 24 cm tags (step 3.7).
  Whatever made it boot goes into `/home/ziggy/loci-handheld/env` and Appendix B.

- [ ] **1.3 Geometry and Y sign.** Tape at 1.00 m and 2.00 m; cup on the tape; then cup to the
  right; then cup above the lens centre for the y-sign run.
  ```bash
  pi$ python $S/scripts/oak_console.py --seconds 15 --rate 1
  pi$ python $S/scripts/oak_console.py --seconds 20 --check-y-sign
  ```
  Expect: `z=` within ±5 %, `x=+..` for the cup on the right, `(no depth)` for boxes without
  depth (never `z=0.00m`), and `verdict: RDF confirmed (above-centre objects have y<0)`.
  If it fails: `verdict: Y appears UP: set LOCI_OAK_AXES=ruf` -> set it in `env`, repeat and
  expect `RDF confirmed` (with `--axes ruf` the opposite message
  `Y appears UP with LOCI_OAK_AXES=ruf: remove the override` means the first run was wrong, keep
  `rdf`); `Y sign inconclusive` -> hold the cup clearly above the lens centre.

- [ ] **1.4 Optional stereo flags.** One at a time, 30 s each; keep a flag only if the pipeline
  still boots and `fps` stays ≥ 8.
  ```bash
  pi$ LOCI_OAK_STEREO_LRC=1 python $S/scripts/oak_console.py --seconds 30
  pi$ LOCI_OAK_STEREO_EXTENDED=1 python $S/scripts/oak_console.py --seconds 30     # MinZ ~40 cm -> ~20 cm
  ```
  If it fails (`8 shaves` error): leave both at `0` in `env` (the verified default).

- [ ] **1.5 Resilience and CPU.** 90 s run; unplug at 20 s, replug at 30 s; in a second ssh
  session watch the thread load.
  ```bash
  pi$ python $S/scripts/oak_console.py --seconds 90
  pi$ top -H -b -n 1 -p "$(pgrep -f oak_console.py)" | head -20        # second session, while running
  ```
  Expect: `state=waiting_for_camera ... reconnects=1`, then `state=running ... reconnects=1`,
  process alive to the end, `exit=0`; the busiest thread < 25 % of one core.
  If it fails: stuck in `waiting_for_camera` after replug -> `lsusb` must show `03e7:2485`
  again (power/cable); a second Ctrl-C aborts if needed.

- [ ] **1.6 Record 60 s for replay tests.**
  ```bash
  pi$  python $S/scripts/oak_console.py --seconds 60 --record /home/ziggy/loci-handheld/data/step1.jsonl
  mac$ scp ziggy@loci-pi.local:/home/ziggy/loci-handheld/data/step1.jsonl /private/tmp/claude-501/-Users-0x7d0-git-loci-db/8139ebf0-1ba6-436c-8885-212cc37c9445/scratchpad/step1.jsonl
  mac$ python deploy/handheld/scripts/oak_console.py --replay /private/tmp/claude-501/-Users-0x7d0-git-loci-db/8139ebf0-1ba6-436c-8885-212cc37c9445/scratchpad/step1.jsonl --no-realtime; echo exit=$?
  ```
  Expect: `recorded N lines -> /home/ziggy/loci-handheld/data/step1.jsonl`; the Mac replay ends
  with `state=done`, `exit=0`.

---

## Step 2 — adapter and persistence (server on the Pi)

- [ ] **2.1 Service up.**
  ```bash
  pi$ systemctl --user restart loci-handheld && sleep 5
  pi$ journalctl --user -u loci-handheld -n 30 --no-pager
  pi$ curl -s localhost:8001/health | j status observation_count pipeline.stt_available pipeline.tts_engine
  pi$ curl -s localhost:8001/api/handheld/status | j runtime.source.state runtime.source.fps runtime.source.usb_speed runtime.source.reconnects runtime.tag_frames runtime.dropped_no_depth memory.embedder restore.source config.detection_source errors
  pi$ $S/smoke_test.sh
  ```
  Expect: journal shows `LOCI handheld: source=oak voice=none tls=0`, the phone URL and
  `Application startup complete`; `status = "ok"`; `runtime.source.state = "running"`,
  `fps ≥ 8`, `runtime.tag_frames` growing by ~5/s (the tag stream is alive inside the server
  too), `memory.embedder = "static:potion-base-8M"`, `errors = {}`; `smoke_test.sh` prints `ok`
  lines and `all N checks passed`.
  If it fails: `systemctl --user` errors over ssh -> `export XDG_RUNTIME_DIR=/run/user/$(id -u)`;
  the unit flapping (`Restart=on-failure`) -> `journalctl --user -u loci-handheld -f` for the
  traceback: a `static` embedder error means 0.7 (or `LOCI_LABEL_EMBEDDER=auto`), a camera error
  means step 1's flags are missing from `env`; `smoke_test.sh` stops at the first `FAIL` line
  naming the endpoint.

- [ ] **2.2 Camera-frame sightings and the hold-off.** Cup on a table 1-2 m away for 5 s.
  ```bash
  pi$ curl -s localhost:8001/api/objects | j count objects.0.label objects.0.frame objects.0.xyz_cam_m objects.0.anchored objects.0.scene_id
  pi$ for i in 1 2 3 4 5 6; do curl -s localhost:8001/health | j observation_count; sleep 2; done
  pi$ curl -s localhost:8001/api/handheld/status | j runtime.ingestion memory.frames
  ```
  Expect: `label = "cup"`, `frame = "camera"`, `xyz_cam_m = [..]`, `anchored = false`,
  `scene_id = "unanchored/cup"`; `observation_count` grows by at most 1 per 2 s per static object
  (`LOCI_CONSENSUS_REFIRE_MS=2000`), `runtime.ingestion.suppressed_by_holdoff` grows,
  `memory.frames.camera > 0`.
  If it fails: count growing ~5/s -> `LOCI_CONSENSUS_REFIRE_MS` not applied (check `env` and
  `config.consensus_refire_ms` in the status); no objects -> `runtime.source.state`, lighting,
  `LOCI_OAK_CONF`.

- [ ] **2.3 Typed query latency and wording.**
  ```bash
  pi$ curl -s -X POST localhost:8001/api/query -H 'content-type: application/json' -d '{"text":"where is my mug"}' | j resolution.label resolution.how latency_ms answer
  pi$ curl -s -X POST localhost:8001/api/query -H 'content-type: application/json' -d '{"text":"where are my keys"}' | j resolution.label resolution.suggestions answer
  pi$ $S/smoke_test.sh --expect-object cup
  ```
  Expect: `resolution.label = "cup"`, `how = "alias"`, `latency_ms < 300`, answer "Your mug (I
  track it as a cup) was last seen about N metres in front of the camera ..., roughly — I wasn't
  sure of my position. That was ..."; keys -> `label = null` and "I can't recognise 'keys' with
  this camera. ... I can look for ..."; smoke test passes.
  If it fails: `latency_ms` ≥ 300 -> note the value; it should be single-digit ms (the budget is
  the `where_is` window query); a wrong label for "mug" means the alias table did not load
  (`resolution.how`).

- [ ] **2.4 Restart, reboot, kill -9.**
  ```bash
  pi$ systemctl --user restart loci-handheld && sleep 10
  pi$ curl -s localhost:8001/api/handheld/status | j restore.records restore.rooms restore.source restore.bad_lines
  pi$ curl -s -X POST localhost:8001/api/query -H 'content-type: application/json' -d '{"text":"where is my mug"}' | j answer
  pi$ $S/smoke_test.sh --restored
  pi$ sudo reboot      # then ssh back in after ~60 s and repeat the three lines above
  # journal durability: walk the cup around for 10 s, then
  pi$ kill -9 "$(pgrep -f 'uvicorn demo_spatial')"; sleep 8        # systemd restarts it
  pi$ curl -s localhost:8001/api/handheld/status | j restore.records restore.source restore.journal_ops restore.bad_lines
  ```
  Expect: `restore.records > 0`, `source = "snapshot"` or `"snapshot+journal"`, `bad_lines = 0`,
  and the answer ends with "That was before I restarted."; after the reboot the same (user unit
  autostarts thanks to linger); after `kill -9`, `source = "snapshot+journal"` with
  `journal_ops > 0`, at most ~1 s of sightings missing (`LOCI_JOURNAL_FSYNC_S=1`), `bad_lines`
  0 or 1 (a torn last line is tolerated and counted).
  If it fails: `records = 0` after a restart -> `LOCI_DATA_DIR` in `env` vs
  `memory.persistence.data_dir` in the status, `memory.persistence.errors` (disk full /
  permissions); the service not back after reboot -> `loginctl enable-linger ziggy`.

- [ ] **2.5 Snapshot on demand and files.**
  ```bash
  pi$ curl -s -X POST localhost:8001/api/handheld/snapshot | j status persistence.journal_lines persistence.snapshots persistence.errors
  pi$ ls -la /home/ziggy/loci-handheld/data/
  ```
  Expect: `status = "ok"`, `journal_lines = 0` right after the snapshot, `errors = 0`;
  `memory.snapshot.json`, `memory.snapshot.bak.json`, `memory.journal.jsonl` present, no
  leftover `*.tmp`.

---

## Step 3 — AprilTag room anchoring

- [ ] **3.1 Print, measure, mount.**
  ```bash
  mac$ python deploy/handheld/scripts/make_tag_sheet.py --ids 0 1 2 3 --size-mm 160 --out deploy/handheld/tags
  mac$ ls deploy/handheld/tags/
  ```
  Expect: `tag36h11_000_160mm.png` / `.pdf` ... `tag36h11_003_160mm.*` (2360 px wide, 300 dpi,
  black square 1888 px = 159.85 mm). Print at 100 % (no "fit to page"), **measure the black
  square with a ruler** and put the value in metres into `/home/ziggy/loci-handheld/rooms.json`
  (`tag_size_m`) or set `LOCI_TAG_SIZE_M` in `env`; restart the service. Mount tag 0 (kitchen)
  and tag 1 (living room) at ~1.4 m beside each door facing into the room, flat and matte,
  arrow up; optionally tag 2 on an adjacent kitchen wall.
  ```bash
  pi$ systemctl --user restart loci-handheld && sleep 5
  pi$ curl -s localhost:8001/api/rooms | j tag_size_m rooms.0.name rooms.0.anchor_tag rooms.0.anchor_height_m rooms.1.name rooms.1.anchor_tag
  ```
  Expect: your measured `tag_size_m`, `kitchen`/`0` and `living_room`/`1`.

- [ ] **3.2 Anchor at 2.00 m.** Stand on tape 2.00 m in front of tag 0, camera level, facing it.
  ```bash
  pi$ curl -s localhost:8001/api/anchor | j room frame anchored anchor_confidence camera_xyz_m heading_deg tag_ids age_s
  pi$ curl -s localhost:8001/api/handheld/status | j runtime.localizer.detector runtime.localizer.tags_visible runtime.localizer.poses_accepted runtime.localizer.poses_rejected runtime.localizer.pnp_available
  ```
  Expect: `room = "kitchen"`, `frame = "room"`, `anchored = true`, `anchor_confidence ≥ 0.8`,
  `camera_xyz_m[2]` within 2.00 ± 0.06, `camera_xyz_m[1]` ≈ the height you hold the camera at,
  `heading_deg ≈ 0` (looking at the anchor wall), `tag_ids = [0]`, `age_s < 1`;
  `detector = "pyapriltags"`, `tags_visible = [0]`, `poses_accepted` growing, `pnp_available = true`.
  If it fails: `poses_rejected` growing with `poses_accepted` flat -> too oblique/far/small
  (`TagFilter`: side ≥ 30 px, view ≤ 60°, range ≤ 6 m), move closer and frontal; z off by > 3 %
  -> note the ratio (EEPROM intrinsics check), re-measure the tag, then `LOCI_TAG_UNDISTORT=1`
  and restart (`LOCI_TAG_FX_SCALE` is **not** a feature); `detector = null` -> pyapriltags/cv2
  missing in the venv (0.4); mirrored `heading_deg`/x sign -> `LOCI_TAG_UPRIGHT_FIX=1` only if
  the tags were printed in the canonical orientation (not from `make_tag_sheet.py`).

- [ ] **3.3 Object in room metres.** Mark a floor point 1.0 m to the **left** of the tag when
  facing it (room x = −1.0) and 2.0 m out from the wall (z = 2.0); put the cup there at table
  height (~0.75 m), look at it with the tag still in view for 5 s.
  ```bash
  pi$ curl -s localhost:8001/api/objects/cup/location | j best_match.room best_match.frame best_match.x_m best_match.y_m best_match.z_m best_match.anchored best_match.scene_id
  pi$ curl -s -X POST localhost:8001/api/query -H 'content-type: application/json' -d '{"text":"where is my mug"}' | j answer
  ```
  Expect: `room = "kitchen"`, `frame = "room"`, `x_m ≈ −1.0 ± 0.15`, `y_m ≈ 0.7-0.8`,
  `z_m ≈ 2.0 ± 0.15`, `anchored = true`, `scene_id = "kitchen/cup"`; the answer says "in the
  kitchen, about 2 metres into the room and 1 metre to your right as you walk in from the door tag,
  at table height" (walk-in perspective: room −x is the walker's right) or "near the table" when
  within 1.2 m of the `rooms.json` landmark `(1.0, 0.75, 1.2)`.
  If it fails: x sign inverted -> the tag is mounted mirrored/upside down (arrow up?) or
  `LOCI_TAG_UPRIGHT_FIX`; z short by a constant factor -> `tag_size_m`.

- [ ] **3.4 Pose hold and staleness.** Turn away from every tag and keep looking at the cup.
  ```bash
  pi$ sleep 15; curl -s localhost:8001/api/anchor | j anchored anchor_confidence frame age_s
  pi$ sleep 50; curl -s localhost:8001/api/anchor | j anchored frame age_s
  # look back at tag 0
  pi$ sleep 1;  curl -s localhost:8001/api/anchor | j anchored anchor_confidence age_s
  ```
  Expect: after 15 s `anchored = false`, `anchor_confidence = 0`, `frame = "room"` (held pose,
  `LOCI_POSE_HOLD_S=10`); after 65 s `frame = "camera"` (`LOCI_POSE_STALE_S=60`), new sightings
  go to `unanchored/cup`; looking back re-anchors within 1 s (`age_s < 1`).

- [ ] **3.5 Second tag self-registers.** Tags 0 and 2 both in view for ≥ 3 s (10 co-visible frames
  at 5 fps).
  ```bash
  pi$ curl -s localhost:8001/api/rooms | j rooms.0.tags_registered
  pi$ curl -s localhost:8001/api/handheld/status | j runtime.localizer.tags_registered runtime.localizer.tags_pending runtime.localizer.multi_tag_solves
  pi$ python3 -m json.tool /home/ziggy/loci-handheld/data/tags.json | head -20
  ```
  Expect: `tags_registered` for kitchen = `[0, 2]`, `multi_tag_solves` growing while both are
  visible, `tags.json` with `"format": 1` and the kitchen map containing tag 2.
  If it fails: tag 2 stays in `tags_pending` -> quads must be ≥ 40 px for registration (move
  closer); `unknown_tags` lists ids not in `rooms.json`.

- [ ] **3.6 Two-room walk.** Kitchen -> living room -> kitchen, looking at each door tag on the way.
  ```bash
  pi$ watch -n 1 'curl -s localhost:8001/api/anchor | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[\"room\"], d[\"anchored\"], d[\"camera_xyz_m\"])"'
  pi$ curl -s localhost:8001/api/objects/cup/history | j count observations.0.room observations.0.x_m observations.0.z_m observations.1.room
  pi$ curl -s 'localhost:8001/api/objects?room=kitchen' | j count
  ```
  Expect: `room` switches at each door tag; the cup's position before and after the loop differs
  by < 0.3 m (a second sighting merges into the same record when within `LOCI_DEDUP_RADIUS_M=0.35`,
  so `history` may show one record with a larger `merge_count` instead of two); `?room=kitchen`
  lists only kitchen objects.

- [ ] **3.7 Tag-stream ladder (only if 1.2 needed it).** With `LOCI_TAG_STREAM=640x400@10` in
  `env`, print 24 cm tags and repeat 3.2:
  ```bash
  mac$ python deploy/handheld/scripts/make_tag_sheet.py --ids 0 1 --size-mm 240 --out deploy/handheld/tags
  ```
  Expect: `tag36h11_000_240mm.*`; usable range ≈ 1.5 m oblique / 2 m frontal (record the
  measured range in Appendix B).

---

## Step 4 — phone page and typed queries

- [ ] **4.1 Open the page.**
  ```bash
  pi$ curl -s localhost:8001/api/local-ip
  phone  http://loci-pi.local:8001/phone      # Android: http://<ip>:8001/phone, or scan /api/scanner-qr
  ```
  Expect: status strip with the room name, a green dot while a tag is in view (amber held, grey
  camera-only), source state and fps; "seeing now" updates within 1 s.
  If it fails: `.local` does not resolve on Android -> the IP; nothing loads -> same Wi-Fi, or
  the hotspot (4.6).

- [ ] **4.2 Typed question, screen reader.** Type "Where is my mug?" -> Send.
  Expect: the kitchen answer in < 1 s inside the `aria-live` region (VoiceOver/TalkBack reads it
  unprompted); "Speak" plays it with browser speech.

- [ ] **4.3 Moved and gone.** Move the cup ≥ 1.5 m, wait 5 s, ask again; then "What changed in the
  kitchen?"; then carry the cup out of the kitchen and stay anchored in the kitchen (tag in view)
  for ≥ 10 s; ask again.
  ```bash
  pi$ curl -s 'localhost:8001/api/objects/changes?room=kitchen&window_seconds=600' | j count changes.0.kind changes.0.label changes.0.delta_m
  ```
  Expect: a new position; "In the kitchen in the last 10 minutes: the cup moved about N metres"
  (`kind = "moved"`, `delta_m ≥ 0.5` = `LOCI_MOVED_THRESHOLD_M`); after carrying it out: "I no
  longer see the cup" (`kind = "gone"`, requires ≥ 5 s of anchored presence in the room after the
  last sighting).
  If it fails: no `gone` -> you were not anchored (green dot) long enough after the cup left; no
  `moved` -> the move was < 0.5 m or the new sighting merged (move further).

- [ ] **4.4 WebSocket reconnect and restore banner.** With the page open:
  ```bash
  pi$ systemctl --user restart loci-handheld
  ```
  Expect: the page reconnects by itself and shows the restore banner ("Restored N sightings in M
  rooms, newest ...").

- [ ] **4.5 Optional: phone microphone over HTTPS.** `LOCI_TLS=1` in `env`, restart; the first
  start generates `/home/ziggy/loci-handheld/certs/{cert.pem,key.pem}`.
  ```bash
  pi$ journalctl --user -u loci-handheld -n 5 --no-pager | grep -i 'https\|certificate'
  phone  https://loci-pi.local:8001/phone     # accept the self-signed certificate
  pi$ curl -sk -X POST https://localhost:8001/api/voice/tts -H 'content-type: application/json' -d '{"text":"hello"}' -o /dev/null -w '%{content_type}\n'
  ```
  Expect: the mic button appears (`isSecureContext`), hold it and ask; with `TTS_ENGINE=piper`
  the reply plays and `/api/voice/tts` answers `audio/wav`.

- [ ] **4.6 Optional: hotspot.**
  ```bash
  pi$ sudo nmcli device wifi hotspot ifname wlan0 ssid loci-pi password <password>
  phone  http://10.42.0.1:8001/phone
  ```

---

## Step 4V — headset voice on the Pi (no phone)

The headset loop needs the **user** unit (PipeWire is per user); the system unit cannot reach it.

- [ ] **4V.1 Pair, trust, connect.** Headset in pairing mode.
  ```bash
  pi$ bluetoothctl
  [bluetooth]# power on
  [bluetooth]# agent on
  [bluetooth]# default-agent
  [bluetooth]# scan on                 # wait for [NEW] Device AA:BB:CC:DD:EE:FF <name>
  [bluetooth]# pair AA:BB:CC:DD:EE:FF
  [bluetooth]# trust AA:BB:CC:DD:EE:FF
  [bluetooth]# connect AA:BB:CC:DD:EE:FF
  [bluetooth]# scan off
  [bluetooth]# exit
  pi$ bluetoothctl info AA:BB:CC:DD:EE:FF | grep -E 'Name|Paired|Trusted|Connected|UUID: (Headset|Handsfree|Audio Sink|A/V Remote)'
  ```
  Expect: `Paired: yes`, `Trusted: yes`, `Connected: yes`, UUIDs for `Handsfree` (HFP) and
  `A/V Remote Control` (the button).
  If it fails: `Failed to pair` -> `remove AA:..` and pair again; no `Handsfree` UUID -> the
  headset has no HFP microphone and cannot be used for input.

- [ ] **4V.2 HFP profile, defaults, mic test.**
  ```bash
  pi$ wpctl status | sed -n '/Audio/,/Video/p'
  pi$ wpctl set-profile <bluez_card id> headset-head-unit        # or headset-head-unit-msbc when listed
  pi$ wpctl set-default <sink id>; wpctl set-default <source id>
  pi$ pw-record --rate 16000 --channels 1 /home/ziggy/loci-handheld/data/mic_test.wav   # say "where is my mug", Ctrl-C after ~3 s
  pi$ pw-play /home/ziggy/loci-handheld/data/mic_test.wav
  pi$ python3 -c "import wave; w=wave.open('/home/ziggy/loci-handheld/data/mic_test.wav'); print(w.getframerate(), w.getnchannels(), w.getnframes()/w.getframerate(), 's')"
  ```
  Expect: a `bluez_card.*` with the headset name, a `bluez_output...` sink and `bluez_input...`
  source marked `*` (default) after `set-default`; the recording plays back intelligibly
  (`16000 1 ~3.0 s`).
  If it fails: no `bluez_input` source -> profile still A2DP (`set-profile ... headset-head-unit`,
  some headsets need `bluetoothctl disconnect` + `connect` after the switch); silence -> the
  headset's own mute, or `wpctl set-volume <source id> 1.0`.

- [ ] **4V.3 Button device and permissions.**
  ```bash
  pi$ grep -B1 -A6 'AVRCP' /proc/bus/input/devices          # N: Name="<Headset> (AVRCP)" ... H: Handlers=kbd eventN
  pi$ id -nG | tr ' ' '\n' | grep -x input
  pi$ timeout 10 od -An -tx1 -w24 /dev/input/eventN          # press the headset button once within 10 s
  pi$ python $S/scripts/voice_console.py --list-devices
  ```
  Expect: the AVRCP device with `Handlers=... eventN`; `input` printed; two to four 24-byte rows
  per press (`EV_KEY` value 1 then 0, plus `EV_SYN` rows); `--list-devices` shows the headset
  under `audio devices (sounddevice)` (`in/out`) or the CLI backend's `pipewire` entries, and
  `/dev/input/eventN  <Headset> (AVRCP)` under `input (button) devices`.
  If it fails: no AVRCP line -> the headset is not connected (4V.1) or exposes no AVRCP;
  `od: Permission denied` -> `sudo usermod -aG input ziggy` and a new login; rows only on
  every second press -> the headset sends click events only: `LOCI_PTT_TOGGLE=1` (`--toggle`);
  `sounddevice: False` -> `sudo apt install libportaudio2` or `--audio cli`.

- [ ] **4V.4 Loopback text -> Piper (no mic, server running).**
  ```bash
  pi$ python $S/scripts/voice_console.py --loopback-text "where is my mug" --tts-out /home/ziggy/loci-handheld/data/answer.wav; echo exit=$?
  pi$ pw-play /home/ziggy/loci-handheld/data/answer.wav
  ```
  Expect: `transcript`, `answer  Your mug (I track it as a cup) ...`, `tts` (Piper, ~0.5-1 s),
  `done`, a JSON line with `"tts_out"`, `exit=0`; the headset speaks the answer.
  If it fails: `tts_unavailable` -> `PIPER_MODEL_PATH`/`TTS_ENGINE=piper` in the shell env
  (`set -a; source /home/ziggy/loci-handheld/env; set +a`) or the piper wheel (0.4); the
  `piper >= 1.3` API (`synthesize_wav`) is handled, an older piper falls back to `synthesize`.

- [ ] **4V.5 Loopback WAV -> Whisper.** Uses the 4V.2 recording; `WHISPER_LOCAL=1`,
  `WHISPER_LOCAL_MODEL=tiny.en`, `HF_HUB_OFFLINE=1` from `env` (model cached by `install_pi.sh`).
  ```bash
  pi$ python $S/scripts/voice_console.py --loopback-wav /home/ziggy/loci-handheld/data/mic_test.wav --no-tts; echo exit=$?
  ```
  Expect: `transcript  where is my mug` with `latency_ms` 1-2 s on the Pi for `tiny.en`, an
  answer, `exit=0`.
  If it fails: a Hub download error -> `fetch_models.py --whisper tiny.en` with network once, or
  `HF_HUB_OFFLINE=0`; a wrong transcript -> try `WHISPER_LOCAL_MODEL=base.en` and note the latency.

- [ ] **4V.6 One live turn with the headset button.** Server running (`LOCI_VOICE=none` in the
  service for now, so only the console owns the microphone).
  ```bash
  pi$ python $S/scripts/voice_console.py --ptt evdev --turns 1; echo exit=$?
  ```
  Press the button, wait for the earcon, say "where is my mug", release.
  Expect: `started ptt=evdev audio=...`, `press`, `earcon`, `recorded`, `transcript  where is my
  mug`, `answer ...`, `tts`, `played`, `done` with the total `latency_ms` (target ≤ 4000 from
  release to the end of playback), `exit=0`.
  If it fails: nothing after `started` -> wrong device (`--ptt-device "<substring>"` or
  `--ptt-device /dev/input/eventN`), permission (4V.3), or click-only headset (`--toggle`);
  `recorded` with an empty transcript -> HFP (4V.2) or `--audio cli`; the loop keeps running after
  an error (`error` event + the spoken fallback) by design, Ctrl-C exits with 130.
  Alternative check (hands-free): `python $S/scripts/voice_console.py --ptt vad --turns 1`.

- [ ] **4V.7 Voice loop inside the service.** In `/home/ziggy/loci-handheld/env` set
  `LOCI_VOICE=headset`, `LOCI_PTT=evdev` (`LOCI_PTT_DEVICE=AVRCP` is the default substring;
  set it only if 4V.3 showed a different name), `LOCI_PTT_TOGGLE` per 4V.3; keep
  `LOCI_AUDIO_BACKEND=auto`.
  ```bash
  pi$ systemctl --user restart loci-handheld && sleep 8
  pi$ journalctl --user -u loci-handheld -n 20 --no-pager | grep -i 'voice\|push-to-talk\|headset'
  pi$ curl -s localhost:8001/api/handheld/status | j voice.running voice.ptt voice.audio voice.turns voice.errors
  # press the button, ask "what changed in the kitchen", release, listen; then
  pi$ curl -s localhost:8001/api/handheld/status | j voice.turns voice.errors voice.last_events
  pi$ $S/smoke_test.sh --voice --expect-object cup
  ```
  Expect: journal `LOCI handheld: source=oak voice=headset tls=0` and `push-to-talk button:
  /dev/input/eventN`; `voice.running = true`, `voice.ptt = "evdev"`; after the question
  `voice.turns = 1`, `voice.errors = 0`, `last_events` ending in `played`/`done`; the answer
  spoken in the headset; smoke test passes.
  If it fails: `voice = null` -> `LOCI_VOICE` not `headset` in `env` or the service is the system
  unit; `voice = {"running": false, "error": "..."}` -> the loop refused to start, the error
  names the cause (`LOCI_AUDIO_BACKEND=sounddevice` without the package, unknown `LOCI_PTT`);
  `voice.errors` growing with `error` events naming `sounddevice`/`PortAudio` ->
  `LOCI_AUDIO_BACKEND=cli`; `no input device matching 'AVRCP'` in the journal -> headset not
  connected or `LOCI_PTT_DEVICE`. `curl -s localhost:8001/health | j handheld.voice` is the
  one-field version of the same check.

- [ ] **4V.8 Headset power cycle.** Switch the headset off for 20 s and on again.
  Expect: BlueZ reconnects (trusted), the journal shows `push-to-talk button: /dev/input/eventN`
  again (the device node may get a new N), the next press works without a service restart.
  If it fails: `systemctl --user restart loci-handheld` is the workaround; note it.

---

## Step 5 — battery walkaround and tuning

- [ ] **5.1 On the power bank.**
  ```bash
  pi$ vcgencmd get_config usb_max_current_enable; vcgencmd get_throttled; vcgencmd measure_temp
  pi$ curl -s localhost:8001/api/handheld/status | j runtime.source.state runtime.source.usb_speed runtime.source.reconnects
  ```
  Expect: `=1`, `throttled=0x0`, `state = "running"`, `reconnects` unchanged after 5 minutes.
  If it fails: `throttled` bits set or the camera dropping (`reconnects` growing) -> the bank
  negotiates < 5 A: powered hub or OAK Y-adapter.

- [ ] **5.2 20-minute two-room walk, story twice.** Optionally set `LOCI_RECORD_PATH=/home/ziggy/loci-handheld/data/walk.jsonl`
  in `env` first (replay data for later tuning). In a second ssh session:
  ```bash
  pi$ journalctl --user -u loci-handheld -f | grep -iE 'x_link|traceback|error|reconnect'
  pi$ watch -n 10 'curl -s localhost:8001/api/handheld/status | python3 -c "import json,sys; d=json.load(sys.stdin); s=d[\"runtime\"][\"source\"]; m=d[\"memory\"]; print(s[\"state\"], s[\"fps\"], \"reconnects\", s[\"reconnects\"], \"frames\", m[\"frames\"], \"obs\", m[\"observation_count\"], \"voice\", (d.get(\"voice\") or {}).get(\"turns\"))"; vcgencmd measure_temp'
  ```
  Story (each pass): put the cup down in the kitchen, walk to the living room, ask "where is my
  mug" (headset), move the cup, ask "what changed in the kitchen", ask "where are my keys",
  power-cycle the Pi, ask again.
  Expect: no `X_LINK` errors or tracebacks, `reconnects` stable, anchored ratio
  `memory.frames.room / (room + camera)` ≥ 0.7 while walking slowly past tags, `measure_temp`
  < 75 °C, every answer within 4 s of button release, "before I restarted" after the power cycle.
  With `LOCI_RECORD_PATH` set, `curl -s localhost:8001/api/handheld/status | j runtime.recorder`
  shows `{"path": ".../walk.jsonl", "lines": N}` growing; copy the file to the Mac afterwards
  (`scp`, as in 1.6) for replay-based tuning.

- [ ] **5.3 Tune.** Edit `env` (and `rooms.json`), restart, repeat the relevant item:
  `LOCI_CONSENSUS_REFIRE_MS` (1000-3000), `LOCI_CONSENSUS_RADIUS_M` (0.3), `LOCI_DEDUP_RADIUS_M`
  (0.25-0.5), `LOCI_MOVED_THRESHOLD_M` (0.5), `LOCI_POSE_HOLD_S` (5-15), `LOCI_POSE_STALE_S` (60),
  `LOCI_OAK_CONF` (0.5 -> 0.6 against false positives), `LOCI_OAK_DEPTH_MAX_MM` (6000 -> 5000
  against far-range jitter). Targets: a static cup produces one record per 2 s window, a moved cup
  is reported within 3 s, position error < 0.5 m at ≤ 3 m.

- [ ] **5.4 Cold boot.**
  ```bash
  pi$ sudo reboot
  # after ~60 s
  pi$ systemctl --user is-active loci-handheld; curl -s localhost:8001/api/handheld/status | j runtime.source.state restore.records voice.running
  ```
  Expect: `active`, `state = "running"`, `restore.records > 0`, `voice.running = true`; the
  headset answers within 60 s of power-on.

---

## Appendix A — what this plan relies on

CLI flags used: `oak_console.py` `--seconds --sim --scenario --replay --no-realtime --seed --usb2
--model --no-tags --axes --fps --record --check-y-sign --rate -v`; `voice_console.py`
`--list-devices --ptt --audio --audio-in --audio-out --ptt-device --toggle --vad-threshold
--vad-silence --max-seconds --no-earcon --loopback-wav --loopback-text --in-process --canned
--host --port --timeout --tts-out --no-tts --play --turns -v`; `fetch_models.py` `--models-dir
--layout --only-missing --force --verify --list --whisper`; `make_tag_sheet.py` `--ids --size-mm
--out`; `smoke_test.sh` `--expect-state --min-fps --expect-object --query --label --max-latency-ms
--voice --snapshot --restored`; `deploy_from_mac.sh` `--no-restart --install --logs --dry-run`;
`install_pi.sh` `--skip-pip --no-models --no-service --start`; `run_pi.sh` honours `LOCI_HOME`,
`LOCI_ENV_FILE`, `LOCI_VENV`, `LOCI_BIND_HOST`, `PORT`, `LOCI_TLS`.

Environment keys touched: `LOCI_DETECTION_SOURCE LOCI_REPLAY_PATH LOCI_RECORD_PATH LOCI_OAK_AXES
LOCI_OAK_PP_SHAVES LOCI_OAK_STEREO_PRESET LOCI_OAK_STEREO_LRC LOCI_OAK_STEREO_EXTENDED LOCI_OAK_USB2
LOCI_OAK_CONF LOCI_OAK_DEPTH_MAX_MM LOCI_TAG_STREAM LOCI_TAG_UNDISTORT LOCI_TAG_SIZE_M
LOCI_TAG_UPRIGHT_FIX LOCI_POSE_HOLD_S LOCI_POSE_STALE_S LOCI_LABEL_EMBEDDER LOCI_EMBED_MODEL_DIR
LOCI_DATA_DIR LOCI_PERSIST LOCI_JOURNAL_FSYNC_S LOCI_CONSENSUS_REFIRE_MS LOCI_CONSENSUS_RADIUS_M
LOCI_DEDUP_RADIUS_M LOCI_MOVED_THRESHOLD_M LOCI_TLS LOCI_VOICE LOCI_PTT LOCI_PTT_DEVICE
LOCI_PTT_TOGGLE LOCI_AUDIO_BACKEND LOCI_AUDIO_IN LOCI_AUDIO_OUT LOCI_VAD_THRESHOLD
LOCI_VAD_SILENCE_S TTS_ENGINE PIPER_MODEL_PATH WHISPER_LOCAL WHISPER_LOCAL_MODEL HF_HUB_OFFLINE
DEPTHAI_ZOO_CACHE_PATH DEPTHAI_ZOO_INTERNET_CHECK`.

Endpoints: `GET /health`, `GET /openapi.json`, `GET /api/handheld/status`, `POST
/api/handheld/snapshot`, `GET /api/anchor`, `GET /api/rooms`, `GET /api/objects[?room=]`, `GET
/api/objects/{label}/location`, `GET /api/objects/{label}/history`, `GET
/api/objects/changes?room=&window_seconds=`, `POST /api/query`, `POST /api/voice/tts`, `GET
/api/local-ip`, `GET /api/scanner-qr`, `GET /phone`, `WS /ws`.

Files: `/home/ziggy/loci-handheld/env`, `/home/ziggy/loci-handheld/rooms.json`,
`/home/ziggy/loci-handheld/data/{memory.snapshot.json,memory.snapshot.bak.json,memory.journal.jsonl,tags.json,step1.jsonl,walk.jsonl,mic_test.wav,answer.wav}`,
`/home/ziggy/loci-handheld/models/{zoo,embeddings/potion-base-8M,piper}`,
`/home/ziggy/loci-handheld/certs/{cert.pem,key.pem}`, `/home/ziggy/.config/systemd/user/loci-handheld.service`,
`deploy/handheld/tags/tag36h11_<id>_<mm>mm.{png,pdf}`; on the Mac `$HOME/loci-handheld/{env,.venv,data}`.

## Appendix B — results record

Copy this block into `/home/ziggy/loci-handheld/data/verification-<YYYY-MM-DD>.md` (or the Mac
scratchpad) and fill one line per item; the `env` diff at the end is the deliverable that goes
back into `env.example` defaults if a fallback became permanent.

```
date / tester / camera id / Pi OS / depthai version:
item   result   measured value or output line                     change made (env key=value)
A.5    ok       fps=10.1 usb=HIGH tags/s=5.0                       -
A.6    ok       verdict: RDF confirmed                              -
1.2    FAIL->ok 8-shave error with default; boots with ...           LOCI_TAG_STREAM=640x400@10
...
Open problems:
Final env diff vs env.example:  diff deploy/handheld/env.example /home/ziggy/loci-handheld/env
```
