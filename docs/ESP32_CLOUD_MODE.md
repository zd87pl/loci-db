# ESP32 boards on LOCI Cloud (cloud mode)

**Boards:**
- the Waveshare ESP32-P4-NANO (`firmware/esp32-p4-client`);
- the Meshnology W11 ESP32-S3 (`firmware/esp32-s3-client`).

**What changes:** there is no Mac host in the boards' path. A board does everything it can **locally first**:
- its own object detector;
- MultiNet speech commands;
- the microSD cache.

It then uses **LOCI Cloud** (`cloud/workers`, the board routes in `cloud/workers/README.md`) for three things:
- the database: it writes its own sightings and reads everyone's;
- spoken answers to free-form questions;
- the content of its card cache.

The host protocol of `docs/ESP32_THIN_CLIENT_DEMO.md` still works as `mode host`, unchanged.

```
camera ──► on-board YOLO11n (ESP-DL) ──► sightings (merged per object) ──► POST spaces/<s>/observations (upsert)
                                                      └─ offline: /sdcard/loci/sightings.jsonl, uploaded later
button + mic ──► MultiNet on the board ──► "where is my <cached object>": answer from the card
                         └─ anything else ──► POST board/voice (Whisper → answer → speech) ──► speaker
                                    └─ no network: the card's summary, or the offline prompt
card cache ◄── GET board/memory, board/memory.pcm, board/phrase.pcm (every 10 min, or after own writes)
```

## Setting up a board

1. Create a claim code for the board. It is valid for 30 minutes and works once. `ENV` is the file `bootstrap.py` printed, e.g. `~/.loci/cloud.env`:

   ```bash
   python3 cloud/workers/scripts/devices.py --env-file ~/.loci/cloud.env create --name "P4 desk board"
   ```

2. On the board's serial console, type the line it prints:

   ```
   cloud claim https://loci-cloud.<acct>.workers.dev/p/<ref>/v1 home ABCD-EFGH-JKLM
   location kitchen
   cloud status
   ```

   - The device id and secret are stored in NVS (`c_url`, `c_space`, `c_dev`, `c_sec`) before the board reports success. A code works only once, so a lost reply means a new code.
   - The secret is never printed.
   - The board then switches to `mode cloud` and stays in it across reboots.
   - `location` names the room its sightings are filed under.

3. `mode host` goes back to the Mac. `cloud forget` erases the credentials. `devices.py revoke <id>` locks a board out within a minute.

## Console

| Command | What it does |
|---|---|
| `cloud claim <project_url> <space> <CODE>` | Claim the board (above) |
| `cloud status` | URL, space, device id, room, how long the token has left, last HTTP status, clock; internal free RAM and largest block (and on the P4 a warning when retention RAM runs low) |
| `cloud ask <question>` | Ask the cloud directly, skipping the card |
| `cloud cache` | Refresh the card cache from the cloud now, in either mode |
| `cloud forget` | Erase the credentials, back to `mode host` |
| `mode cloud` · `mode host` | Switch (cloud needs a claim first) |
| `location <room>` · `location clear` | The room sent with this board's sightings |
| `ask <question>` | In cloud mode: the card first ("where is my cup"), then the cloud |
| `detect [n]` | Run the on-board detector on `n` live frames and print labels and timing |
| `wifi off` · `wifi on` | Test aid: drop Wi-Fi and stay off (the offline paths and the card queue), then reconnect |

## What the board does in cloud mode

**Sightings**
- Every 2 s (`LOCI_DETECT_INTERVAL_MS`) a frame goes through the detector (`firmware/common/loci_detect`, ESP-DL COCO YOLO11n 320×320 int8). On the P4 a 1280×960 frame takes 51 ms to decode (hardware JPEG) and about 645 ms to detect.
- **Merging:** detections with a score of at least 0.5 are merged per object. The same label seen again within 60 s and within 0.15 of the frame of its last position keeps its id (`d_<device>_<boot>_<nonce>_<n>`) and is re-sent as an upsert at most every 30 s. Anything else starts a new sighting.
- **Not sent:** people are left out. Nothing is recorded until SNTP has set the clock (`pool.ntp.org`), so no sighting lands in 1970.
- **Batches** of up to 50 go to `POST spaces/<s>/observations` with `"upsert": true` and the image frame, x and y at the box centre and `room` from `location`.
- **Offline**, the same JSON lines are appended to `/sdcard/loci/sightings.jsonl` (at most 512 kB) and uploaded oldest first when the cloud is back. Upserts make a re-sent line harmless.
- Every 10th frame also goes to the card's dataset, with the board's detections as the side-car.

**Questions**
- **Push-to-talk:** MultiNet runs first. "Where is my *X*", with *X* on the card, is answered from the card in under a second.
- **Everything else** goes to `board/voice`: "what do you see", other objects, free-form questions.
  - Silence or Whisper's stock phrases give a short low tone.
  - With no cloud, the board plays the card's summary or the offline prompt.
- **Typed `ask`** does the same: the card first, then `board/ask`, then the card's summary.

**Cache**
- `board/memory` returns the newest sighting per label in the space, from this board, other boards and the Mac's mirror.
- It is polled every 10 minutes, and a minute after the board's own sightings changed something.
- When its `rev` changes, the board fetches the spoken summary and one phrase per pass, so the button is never kept waiting. It then refreshes the MultiNet command list from the new labels.

**Network**
- One keep-alive TLS client, User-Agent `loci-board/<fw>`.
- The 1-hour device token is renewed 5 minutes before expiry, timed with `esp_timer`, so it works before the clock is set. After a 401 the token is renewed once.
- A revoked board reports "needs a new claim".
- After a failed request the cloud is left alone for a minute.

## Memory and flash

**P4**
- **Static internal RAM:** cloud mode and the detector add about 400 B in all, against a retention-RAM margin of about 10.8 kB at boot (`firmware/esp32-p4-client/README.md`, Troubleshooting).
  - Large buffers are allocated in PSRAM.
  - ESP-DL's, the detector's and `loci_detect`'s `.bss` go to PSRAM through `main/linker.lf`.
  - The P4 decodes frames with its hardware JPEG codec, so the software decoder (~3 kB of `.data`) is never linked.
- **TLS:** the P4 has no `MBEDTLS_EXTERNAL_MEM_ALLOC`, so TLS sessions use internal heap. The board skips the cloud while the largest internal block is under 40 kB (`LOCI_CLOUD_MIN_INTERNAL_BLOCK`).
- **New flash layout:**

  | Partition | Size | Notes |
  |---|---|---|
  | app | 6 MB | was 4 MB; ESP-DL with the detector's full operator set makes the image 4.4 MB |
  | `model` (ESP-SR) | 3.5 MB | was 6 MB; `srmodels.bin` is 2.8 MB |
  | `coco_det` | 4 MB | new; the P4 model is 3 MB |

  - The next `idf.py flash` writes all of it.
  - NVS is at the same offset, so Wi-Fi, host, token and cloud settings survive.

**S3**
- TLS runs in PSRAM.
- The image is 3.5 MB of its 4 MB app partition.
- Inference takes about 1.4 s per frame, so detection keeps CPU1 busy about half the time.

## Status

- ✅ Both boards build (IDF 6.1).
- ✅ **P4 tested against staging on 2026-10-30.** Flashed with main (`3fac7d9`) plus `wifi off|on`, and claimed as `dev_4b582c9cb3b20ec3`, room `office`, in staging project `qzjibr9prr5h`.
  - **Boot:** the new partition layout boots. Free retention RAM at `app_main` is 9,136 B, above the 8 kB floor. At runtime, TLS sessions use it all up (`cloud status` reports "retention RAM LOW"), but that only matters at boot.
  - **Claim:** `cloud claim` stored the credentials, fetched a token and switched to `mode cloud`. The claim survived several reboots.
  - **Cache:** the card cache filled from `board/memory` (15 objects, 15 s spoken summary) and was refreshed after the board's own sightings.
  - **`detect 10`:** decode 51 ms, detect 647 ms on average.
  - **Sightings:** live sightings (laptop, tv, chair) reached staging filed under `office`, one row per object.
  - **Push-to-talk:** two questions. MultiNet missed both: one was matched to a word not on the card, the other recognised nothing. So both went to `board/voice`. Whisper heard "Where is my laptop?" and "Where is my TV?". The answers took 1.5 s and 2.6 s; the second was "Your tv was last seen in the office, just now."
    - The laptop question came 4 s after the laptop was first detected, before the 5 s flush had sent it, so the cloud had not seen it yet.
  - **Typed `ask`:** "where is my cup" was answered from the card in about 1 s. "what changed in the kitchen" was answered by the cloud in about 2 s.
  - **`wifi off`:**
    - Card answers still play, and `offline` plays the card summary.
    - Sightings queue on the card. One run queued 2.6 kB, which was uploaded after a reboot as 13 upserts that became 2 rows, with their original times and ids.
    - `wifi on` reconnects in about 4 s, and the next cloud answer included the chair queued while offline.
- ⬜ **Not tested yet:**
  - the S3 against staging;
  - a recognised "where is my X" on push-to-talk being answered from the card;
  - typed `ask` falling back to the card summary while offline (fixed after the test run, not flashed yet).
- ⬜ **Production:** the `board/*` routes and upsert are on staging only; production still runs #80.
- **Findings:**
  - The board power-cycled each time it was picked up. USB dropped and the board booted again several times within a few seconds. It looks like the cable or the power supply, not the firmware: a `wifi off` with nobody touching the board did not reset it.
  - The detector reports detections below the 0.5 score that sightings need, so a faint TV is printed but not tracked. COCO also gives some false positives (a "traffic light" on a desk).
