# LOCI handheld demo on the Raspberry Pi 5 (OAK-D Lite + Bluetooth headset)

Walk through your home with the Pi in one hand and the camera in the other; later ask, by voice
through a Bluetooth headset, where things are and what changed, and hear the answer. Offline, and
it survives a restart. Design: `docs/HANDHELD_DEMO_DESIGN.md`; verified camera facts:
`docs/HANDHELD_HARDWARE_FACTS.md`; product plan: `docs/HANDHELD_DEMO_PLAN.md`; step-by-step
hardware checklist: `deploy/handheld/HARDWARE_VERIFICATION.md`.

## What is in this directory

| File | Purpose |
|---|---|
| `install_pi.sh` | idempotent installer (venv pins, editable repo, models, env, rooms.json, self-test, user unit) |
| `run_pi.sh` | starts uvicorn (one worker, never `--reload`); used by both systemd units |
| `deploy_from_mac.sh` | rsync the checkout to the Pi and restart the service (`--no-restart`, `--install`, `--logs`) |
| `smoke_test.sh` | the step-2 checklist as a script against a running server |
| `loci-handheld.user.service` / `loci-handheld.service` | systemd user unit (recommended) / system unit |
| `requirements-pi.txt` / `requirements-dev.txt` | exact pins for the Pi / the Mac scratch venv |
| `env.example` | every `LOCI_*` key with its Pi value; copied to `/home/ziggy/loci-handheld/env` |
| `config/rooms.example.json` | two-room example (kitchen tag 0, living room tag 1) |
| `scripts/oak_console.py` | build-order step 1: live detections with XYZ in the console (`--sim` without a camera) |
| `scripts/fetch_models.py` | downloads and sha256-verifies the models (`models/README.md`) |
| `scripts/make_tag_sheet.py` | printable AprilTag 36h11 sheets with the measured size in the caption |
| `scripts/voice_console.py` | headset voice loop CLI: `--list-devices`, `--loopback-text`, `--loopback-wav` |
| `models/` | weights (git-ignored) on a dev machine |

Pi layout (user `ziggy`, host `loci-pi.local`):

```
/home/ziggy/loci-handheld/
  src/        this repository (rsync target; ~/loci-db is never touched)
  .venv/      Python 3.13 venv with requirements-pi.txt
  env         environment file (from env.example)
  rooms.json  room / tag configuration (from config/rooms.example.json)
  data/       memory.snapshot.json, memory.snapshot.bak.json, memory.journal.jsonl, tags.json, logs
  models/     zoo/  embeddings/potion-base-8M/  piper/
  certs/      only with LOCI_TLS=1
```

## Bring-up

### 0. Hardware

* Camera on a **blue USB 3 (USB-A) port** with a **data-capable** cable. The USB-C socket of the
  Pi is power only, and many USB-C to USB-A cables are charge-only: `lsusb | grep -i 03e7` must
  list `03e7:2485` (unbooted Movidius). If nothing appears, swap the cable first.
* Power: the camera can draw up to 900 mA at boot; a 3 A supply caps USB at 600 mA unless
  `usb_max_current_enable=1` is set (this Pi already has it). Check:
  `vcgencmd get_config usb_max_current_enable` → `=1`; on a power bank also
  `vcgencmd get_throttled` → `0x0`. If the camera drops out on battery, use a powered hub or the
  OAK Y-adapter.
* Bluetooth headset with a microphone and a media button (push-to-talk). Pair it in step 5.

### 1. One-time sudo steps (printed by `install_pi.sh`, which never calls sudo itself)

```bash
# mandatory once (already done on this Pi) -- more than 600 mA on USB
sudo sh -c 'echo usb_max_current_enable=1 >> /boot/firmware/config.txt' && sudo reboot
# let non-root open the camera
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
# PortAudio for the headset voice loop (python-sounddevice)
sudo apt install -y libportaudio2
# keep the user service manager alive without a login, so the user unit starts at boot
sudo loginctl enable-linger ziggy
```

### 2. Deploy from the Mac and install

```bash
mac$ deploy/handheld/deploy_from_mac.sh --no-restart          # rsync to /home/ziggy/loci-handheld/src
mac$ ssh ziggy@loci-pi.local bash /home/ziggy/loci-handheld/src/deploy/handheld/install_pi.sh
```

`install_pi.sh` upgrades `~/loci-handheld/.venv` to the pins (depthai 3.5.0 → 3.11.0), installs
the repo editable, copies the Piper voice from `~/loci-db/deploy/oak-rpi5/models/` when present,
fetches whatever model is still missing, writes `env` and `rooms.json` from the examples when they
do not exist, runs an import self-test (the camera is not opened) and installs the user unit. It is
safe to re-run after every deploy (`deploy_from_mac.sh --install` does both).

Later deploys: `mac$ deploy/handheld/deploy_from_mac.sh` (sync + restart), `--logs` to follow the
journal.

### 3. Step 1: camera console (no server)

```bash
source /home/ziggy/loci-handheld/.venv/bin/activate
set -a; source /home/ziggy/loci-handheld/env; set +a
python /home/ziggy/loci-handheld/src/deploy/handheld/scripts/oak_console.py --sim --seconds 5   # software sanity
python /home/ziggy/loci-handheld/src/deploy/handheld/scripts/oak_console.py --seconds 30        # real camera
python /home/ziggy/loci-handheld/src/deploy/handheld/scripts/oak_console.py --seconds 20 --check-y-sign
```

Expect `state=running fps≈10` and lines like `cup 0.81 x=+0.12m y=-0.05m z=1.84m`. Hold a cup
above the lens centre for `--check-y-sign`: `RDF confirmed` means `LOCI_OAK_AXES=rdf` (the
default); otherwise set `ruf` in `env`.

### 4. Server, phone page, smoke test

```bash
systemctl --user restart loci-handheld          # or: /home/ziggy/loci-handheld/src/deploy/handheld/run_pi.sh
journalctl --user -u loci-handheld -f
curl -s localhost:8001/api/handheld/status | python3 -m json.tool
/home/ziggy/loci-handheld/src/deploy/handheld/smoke_test.sh                 # step-2 checklist
/home/ziggy/loci-handheld/src/deploy/handheld/smoke_test.sh --expect-object cup --snapshot
```

Phone on the same Wi-Fi: `http://loci-pi.local:8001/phone` (Android may not resolve `.local`;
use the IP printed at startup or `curl -s localhost:8001/api/local-ip`). Type "Where is my mug?".

### 5. Headset voice (no phone needed)

The loop lives in `demo_spatial/app/headset_voice.py`: press the headset button → short earcon →
record until release (or `LOCI_VOICE_MAX_S`) → faster-whisper → `/api/query` answer → Piper →
play back. On Raspberry Pi OS trixie audio goes through **PipeWire + WirePlumber**, Bluetooth
through **BlueZ**.

Pair, trust and connect (headset in pairing mode):

```bash
bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# default-agent
[bluetooth]# scan on                      # wait for: [NEW] Device AA:BB:CC:DD:EE:FF <headset name>
[bluetooth]# pair AA:BB:CC:DD:EE:FF
[bluetooth]# trust AA:BB:CC:DD:EE:FF      # auto-reconnect after reboots
[bluetooth]# connect AA:BB:CC:DD:EE:FF
[bluetooth]# scan off
[bluetooth]# exit
```

Select the **HFP** profile (the microphone only works in HFP/HSP; A2DP is playback only) and make
the headset the default sink and source:

```bash
wpctl status                                        # note the bluez_card id and the Audio Sink / Source ids
wpctl set-profile <card-id> headset-head-unit       # HFP; use headset-head-unit-msbc for mSBC wideband if listed
wpctl set-default <sink-id>                         # playback -> headset
wpctl set-default <source-id>                       # microphone -> headset
pw-record --rate 16000 --channels 1 /home/ziggy/loci-handheld/data/mic_test.wav   # speak, Ctrl-C
pw-play /home/ziggy/loci-handheld/data/mic_test.wav
```

Find the button device and test the loop before enabling it in the service:

```bash
grep -B1 -A4 'AVRCP\|Headset\|<headset name>' /proc/bus/input/devices   # N: Name="... (AVRCP)" -> LOCI_PTT_DEVICE substring
source /home/ziggy/loci-handheld/.venv/bin/activate
python /home/ziggy/loci-handheld/src/deploy/handheld/scripts/voice_console.py --list-devices
python /home/ziggy/loci-handheld/src/deploy/handheld/scripts/voice_console.py --loopback-text "where is my mug" --tts-out /home/ziggy/loci-handheld/data/answer.wav
python /home/ziggy/loci-handheld/src/deploy/handheld/scripts/voice_console.py --ptt evdev      # live: press the button, ask, release
```

Then in `/home/ziggy/loci-handheld/env`:

```
LOCI_VOICE=headset          # none | headset
LOCI_PTT=evdev              # evdev (headset button) | vad (hands-free) | keyboard (terminal)
LOCI_PTT_DEVICE=AVRCP       # substring of the input device name (case-insensitive)
LOCI_PTT_TOGGLE=0           # 1 if the headset only sends click events
LOCI_AUDIO_BACKEND=auto     # auto | sounddevice | cli (pw-record/pw-play, no PortAudio)
#LOCI_AUDIO_IN= / #LOCI_AUDIO_OUT=   keep unset: the PipeWire defaults set with wpctl are used
LOCI_VOICE_MAX_S=10  LOCI_EARCON=1  LOCI_VAD_THRESHOLD=500  LOCI_VAD_SILENCE_S=1.2
```

`systemctl --user restart loci-handheld`, then `curl -s localhost:8001/api/handheld/status`
shows the last voice events under `"voice"` (`smoke_test.sh --voice`). The headset loop needs the
**user** unit: PipeWire is a per-user service that a system unit cannot reach.

### 6. AprilTags (room anchoring)

```bash
mac$ python deploy/handheld/scripts/make_tag_sheet.py --ids 0 1 2 3 --size-mm 160 --out deploy/handheld/tags
```

Print at 100 % (no "fit to page"), **measure the black square** with a ruler and put the value in
metres into `rooms.json` `tag_size_m` (the caption shows the nominal 0.1599). Mount tag 0 (kitchen)
and tag 1 (living room) at about 1.4 m beside each door, facing into the room, flat and matte;
optionally a second tag per room on an adjacent wall (it self-registers when both are visible).
`curl -s localhost:8001/api/anchor` reports `room`, `anchored`, `camera_xyz_m` while a tag is in
view. If the 1280x800 tag stream does not start next to the detector (shave budget), set
`LOCI_TAG_STREAM=640x400@10` and print 24 cm tags (`--size-mm 240`), or `off` for a camera-frame
only demo.

### 7. Hotspot (no LAN)

```bash
sudo nmcli device wifi hotspot ifname wlan0 ssid loci-pi password <password>
```

The phone joins `loci-pi` and opens `http://10.42.0.1:8001/phone`. The headset loop does not
need any network.

## Service

* User unit (recommended): installed by `install_pi.sh` into
  `~/.config/systemd/user/loci-handheld.service`; `systemctl --user {status,restart,stop} loci-handheld`,
  `journalctl --user -u loci-handheld -f`. Starts at boot once `loginctl enable-linger ziggy` is set.
* System unit: `sudo cp loci-handheld.service /etc/systemd/system/ && sudo systemctl enable --now
  loci-handheld` -- typed queries and the phone page only (no PipeWire session).
* Both: `Restart=on-failure`, `EnvironmentFile=/home/ziggy/loci-handheld/env`, absolute paths.
  The in-process camera supervisor reconnects after unplug/replug; systemd only restarts the
  process on a crash.

## Models

See `models/README.md`. `install_pi.sh` fetches only what is missing; offline, copy the
directories from a machine that ran `scripts/fetch_models.py`. The faster-whisper `tiny.en` model
is cached under `~/.cache/huggingface` (`fetch_models.py --whisper tiny.en`); `HF_HUB_OFFLINE=1`
in `env` keeps startup from waiting on the network.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `X_LINK_DEVICE_NOT_FOUND`, `lsusb` shows nothing | charge-only cable or USB-C socket; use a data cable on a blue USB-A port |
| `X_LINK_INSUFFICIENT_PERMISSIONS` | udev rule missing (step 1) or the device is still resetting after a failed pipeline; the source waits for `X_LINK_UNBOOTED` and retries |
| `Blob compiled for 8 shaves, but only 7 are available` | stereo preset other than `FAST_DENSITY` or `LOCI_OAK_PP_SHAVES` ≠ 2; keep the defaults, drop `LOCI_TAG_STREAM` to `640x400@10` or `off` |
| camera drops out on battery, `get_throttled` ≠ 0 | 600 mA cap or a weak bank: `usb_max_current_enable=1`, 27 W supply or powered hub |
| `fps` ≈ 10 but `state=waiting_for_camera` loops | unplugged camera or `usb=UNKNOWN`; replug, check the cable |
| `sounddevice` raises `OSError: PortAudio library not found` | `sudo apt install libportaudio2` or `LOCI_AUDIO_BACKEND=cli` |
| headset plays but records silence | profile is A2DP: `wpctl set-profile <card> headset-head-unit`; some headsets need re-connect after the switch |
| button presses are ignored | wrong `LOCI_PTT_DEVICE` substring (`/proc/bus/input/devices`), user not in group `input`, or the headset sends only clicks (`LOCI_PTT_TOGGLE=1`) |
| `systemctl --user` fails over ssh | `export XDG_RUNTIME_DIR=/run/user/$(id -u)`; enable linger |
| Android cannot open `loci-pi.local` | use the IP (`/api/local-ip`) or the hotspot address `10.42.0.1` |
| answers say "roughly -- I wasn't sure of my position" | no tag seen for > 10 s; look at a door tag, or add a second tag per room |
| "I can't recognise 'keys' with this camera" | expected: COCO-80 has no keys/wallet/glasses; the demo uses a cup ("mug" is an alias) |

## Demo story (two rooms, one cup)

1. Stand in the kitchen door facing tag 0; put the cup on the table; wait 3 s.
2. Walk to the living room (tag 1), look around, come back.
3. Press the headset button: **"Where is my mug?"** → "Your mug (I track it as a cup) was last seen in
   the kitchen, about 1 m into the kitchen and 1 m to your left as you walk in from the door tag, at
   table height. That was 40 seconds ago."
4. Move the cup ≥ 1.5 m, wait 5 s, ask again; then **"What changed in the kitchen?"** → "the cup moved
   about 2 m".
5. **"Where are my keys?"** → the honest refusal with suggestions.
6. `systemctl --user restart loci-handheld` (or power-cycle); ask again → "... That was before I
   restarted."

## Updating

`mac$ deploy/handheld/deploy_from_mac.sh` after every change; `--install` when pins or the env
contract changed (then compare `/home/ziggy/loci-handheld/env` with `env.example`; `install_pi.sh`
lists keys that are missing). Memory data lives outside `src/`, so deploys never delete sightings.
