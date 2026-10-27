# Meshnology W11 (ESP32-S3) vs Waveshare ESP32-P4-NANO as the LOCI pendant base

*Desk evaluation, 2026-10-26. The board was handed over as a "Meshology ESP32"; the vendor is
**Meshnology** and the product is the **W11 "ESP32S3 Mini Module"** with its camera/mic/microSD
expansion board. The host-side device id is `meshology`.*

**Question.** Is the W11 a better base than the ESP32-P4-NANO for the LOCI thin client and, more
importantly, for the pendant we want? That means detection on the board, a local hot window of the
last 1–2 h of observations, local speech recognition and assembled answers, the phone as a BLE
uplink, and LOCI behind Cloudflare for older history.

**Short answer.** The W11 is the better **wearable** base and the weaker **compute** base:

* **Wearable:** the board is thumb-sized, one chip carries Wi-Fi and BLE natively, it has a Li-ion
  charger, an IMU and µA deep sleep, and it needs none of the P4's ESP-Hosted workarounds.
* **Compute:** measured on this board, COCO YOLO11n-320 takes **1.4 s per frame with one core
  saturated**. That is about 4× the P4, and with Wi-Fi, ESP-SR and the detector loaded the free
  internal RAM dips to 17 kB.
* **No speaker:** answers have to reach the user through the phone (BLE, which is the plan) or an
  add-on amplifier.
* **As a thin client** (frames to the host, as the P4 does today) it works as well as the P4:
  **2.0 fps, POST median 109 ms**, once its antenna was attached and three board problems
  found on the desk were fixed (§2).

Recommendation: build the **Stage 1 BLE-tethered pendant prototype on the S3 class (W11 or a
XIAO ESP32S3 Sense, same pins and same firmware)**. It should run detection at about 0.5 Hz, or
faster with smaller single-purpose models, and send observations. Keep the **P4-NANO as the bench
platform** for on-device detection and memory work. Treat the S3's compute as the limit to design
around, and re-evaluate the production pendant on a faster single-chip part (the ESP32-S31 is
listed at ~1.8× the S3 in Espressif's tables) before committing to a PCB.

## Decision table

| Criterion (pendant goal) | Meshnology W11 (ESP32-S3) | ESP32-P4-NANO | Edge |
|---|---|---|---|
| On-board detection (ESP-DL YOLO11n-320, int8) | **1.38–1.49 s/frame measured** (0.7 fps, one core at 100 %); 640² is ~4 s (vendor) | 0.32 s at 400 MHz (vendor; ~0.36 s on the NANO's 360 MHz v1.3 silicon) | **P4, ~4×** |
| Smaller single-purpose detectors (people, faces) | Pico 224²: 125 ms (vendor) | 50 ms (vendor) | P4, but both are usable |
| Local speech commands (ESP-SR MultiNet7, 4 s clip) | runs: **1.5–1.7 s per 4 s clip measured**; recognition accuracy not verified (see §2) | 0.72–0.73 s, recognised "where is my cup" at p = 0.81 | P4 (S3 is fine for button-triggered use) |
| RAM for detector + speech + hot window | 8 MB PSRAM (2.4–2.9 MB free with everything loaded), internal heap minimum 17.5 kB | 32 MB PSRAM | **P4** |
| Camera | OV3660 3 MP DVP, JPEG on the sensor, no ISP, fixed focus; its 20 MHz clock disturbed Wi-Fi and the SD lines, runs at 10 MHz | OV5647 5 MP MIPI, P4 ISP + hardware JPEG, fixed focus | P4 for image pipeline; S3 is simpler |
| Thin-client streaming to the host | **2.0 fps, POST median 109 ms, p90 205 ms**, 1024×768 frames | 2.0 fps, 145–220 ms per POST, 1280×960 frames | even |
| Wi-Fi | native in the S3, −31 to −56 dBm with the u.FL antenna (unusable without it) | C6 over SDIO via ESP-Hosted: retention-RAM boot loops, shared SDMMC, pinned versions, link timeouts at 40 MHz (2 resets in 10 min; none at 20 MHz) | **W11** (one chip, no Hosted) |
| BLE phone tether | native BLE 5 in the S3 | BLE on the C6, through ESP-Hosted's HCI | **W11** |
| Battery and power | Li-ion charger + BAT pads on the board; deep sleep ≤ 8 µA (vendor); one chip | no charger; P4 + C6 always on | **W11** |
| Size and weight | module 24.5 × 17.8 × 4.5 mm, 2.4 g, plus a stacked expansion board | 50 × 50 mm dev board with RJ45 | **W11** |
| Speaking answers | no speaker; mic only | ES8311 + amplifier + speaker header | P4 on the board; for the pendant, the phone does it either way |
| Motion sensing | QMI8658C IMU on the board (gate capture, detect falls, steadiness) | none | **W11** |
| Firmware effort today | this branch: same loop as the P4 (shared `loci_client`), streaming on the board | runs (PR #71) | even |

## 1. What the W11 is (found on the desk, not assumed)

| | Meshnology W11 (on the desk) | Waveshare ESP32-P4-NANO (yesterday) |
|---|---|---|
| SoC | ESP32-S3R8 rev v0.2, 2× Xtensa LX7 @ 240 MHz with 128-bit SIMD (PIE); no NPU despite the vendor's wording | ESP32-P4 rev v1.3, 2× RISC-V @ 360 MHz with AI/SIMD extensions, plus ISP, H.264 and JPEG codecs, PPA |
| RAM | 512 kB SRAM + **8 MB** octal PSRAM | 768 kB SRAM + **32 MB** hex PSRAM |
| Flash | 16 MB quad (Puya) | 16 MB quad |
| Camera | **OV3660** 3 MP on 8-bit DVP (PID 0x3660 read over SCCB), JPEG encoded on the sensor | OV5647 5 MP on 2-lane MIPI-CSI, raw → P4 ISP → hardware JPEG |
| Microphone | PDM MEMS mic (expansion board) | ES8311 codec + onboard mic |
| Speaker | **none** | ES8311 + NS4150B amp, 2-pin speaker header |
| Wi-Fi | native 802.11 b/g/n, **u.FL antenna connector only** | ESP32-C6 companion (Wi-Fi 6) over SDIO with ESP-Hosted |
| BLE | native BLE 5 | BLE 5 on the C6 |
| microSD | SPI, 1-bit, 20 MHz | SDMMC, 4-bit |
| Other | QMI8658C IMU, WS2812 LED (shares GPIO48 with camera D7), BOOT + RESET | Ethernet, optional PoE, MIPI-DSI, USB OTG HS |
| Battery | BAT+/BAT− pads, on-board Li-ion charger up to 1 A | none on the board (USB-C or PoE) |
| Size | module 24.5 × 17.8 × 4.5 mm, 2.4 g; plus expansion board and camera flex | 50 × 50 mm |
| USB | native USB-Serial/JTAG | CH343 USB-UART |
| LoRa / GNSS / display | none (the brand also sells LoRa boards; this one has no radio besides the S3's) | none (DSI connector, no panel) |

Discovery trail:
1. `ioreg` showed an Espressif USB-Serial/JTAG device (303A:1001).
2. esptool reported an ESP32-S3 (QFN56) rev v0.2 with 8 MB embedded PSRAM and 16 MB flash.
3. The factory firmware ("W11 Factory Test + Data Output", Arduino-ESP32 3.3.3) self-tested the
   IMU, digital mic, SD card, camera, Wi-Fi AP, BLE and ADC. Its "OV5640" line is a hard-coded
   string; the sensor that answers is an **OV3660** (the Amazon listing says "OV3660 Sense Kit").
4. The vendor wiki (wiki.meshnology.com, W11) gave the pin map, which is the Seeed XIAO ESP32S3
   Sense one.

The full 16 MB factory image was read back before flashing (kept outside the repo, sha256
`8d120121…7a193`). Pins and the original banner are in `firmware/esp32-s3-client/README.md`.

## 2. Measured on the desk (W11, firmware `0.5.0-s3`, this branch)

| Measurement | Result |
|---|---|
| Boot to `app_main` | 0.86 s; 8 MB PSRAM found, 80 MHz octal |
| Camera | OV3660 detected, warm-up 0.55 s; JPEG straight from the sensor, a capture is ready immediately ("grab latest"). Sizes at quality 12: VGA 24 kB, XGA 61 kB, SXGA 190–214 kB, UXGA 64–312 kB depending on the scene |
| microSD (the P4's card, nothing deleted) | SanDisk 64 GB FAT32 over SPI, 1-bit (20 MHz at first, 10 MHz now); the P4's data was readable: 3,231 dataset frames, cached memory rev `7ef24afc` (the same rev the host reports), spoken summary present; 0 backlog frames left from the P4 |
| Store-and-forward without a host | frames go to `/sdcard/loci/backlog/` at 1 fps (500 frames in ~8 min); the live loop probes the host every 5 s |
| ESP-SR | MultiNet7 English loads from the `model` partition; 54 command phrases built from the cached memory's labels; a 4 s clip is processed in **1.50–1.67 s** |
| ESP-SR recognition | **not verified**: the test clips from the Mac speaker (44 % volume, about a metre away) reached the PDM mic at peak 3–5 k over a 300–650 noise floor and nothing was recognised. Yesterday's P4 test needed the clip right next to the board too. Push-to-talk with a voice close to the mic is the next check |
| **On-board detection (ESP-DL COCO YOLO11n-320 int8)** | model load 0.40 s; **1.38–1.49 s per frame**, up to 1.8 s while the upload loop runs on the other core; the default 32 kB data cache gave 1.97 s, ESP-DL's 64 kB / 64 B line setting gave 1.38 s. JPEG → RGB565 decode: 41 ms for VGA, ~230 ms for UXGA or 1280×960 downscaled to ≤ 640 px |
| Board vs host on the same frames | 15 of the P4's 1280×960 frames from the card (`detect sd 15`): the board's 320-px YOLO11n and the host's YOLOv8n at 960 px agree on large objects (person 0.90, refrigerator 0.62–0.88, chair) and disagree on small or partial ones (host-only dining table/oven, board-only microwave). Live, the board found the person at the desk at 0.88–0.94 |
| Memory with Wi-Fi + ESP-SR + detector loaded | PSRAM 2.4–2.9 MB free of 8 MB; internal heap 59 kB free, **17.5 kB minimum** |
| IMU | QMI8658C at 0x6B, WHO_AM_I 0x05; reads 1 g with the board at rest, 42 °C die temperature |
| Wi-Fi without the antenna | −81 to −86 dBm where the P4 read −40 to −50 at the same desk; pings lost 75 % (56 B) to 100 % (1,400 B) and frame uploads stalled. The W11 has only a u.FL connector |
| Wi-Fi with the antenna | −31 to −56 dBm; joins in under 2 s; 450 pings of 1,000 B: 0 % loss, 9 ms average |
| **Thin-client streaming** (firmware `0b7f87c`, 150 s, card mounted) | **293 live frames, 1.96 fps** on the board and at the host; **POST 109 ms median, 205 ms p90, 345 ms max**; 0 failures, 0 frames dropped by the host; host YOLOv8n 63–82 ms per frame; 1024×768 JPEGs of 20–23 kB (ceiling view) to 60–100 kB (lit room) |
| Without the card (75 s) | 150 frames, 2.0 fps, POST 107 ms median, 234 ms p90, 323 ms max |
| Backlog | the ~980 frames captured to the card while the board had no antenna uploaded interleaved with the live stream until the backlog was empty. While they drained, the host's 4 fps limit dropped many live frames (it counts the backlog frames it processes) |
| Typed question | `ask what do you see` answered by the host in 2.2 s, printed on the board; `offline` printed the 11 remembered objects with their places from the card |

For comparison, yesterday's P4-NANO on the same desk and host: 1280×960 frames of 135–145 kB at
2.0 fps, 145–220 ms per POST with keep-alive, host YOLOv8n 74 ms at 960 px; push-to-talk answer
1.3 s after the button was released.

Problems the desk test found, and what fixed them:

1. **The camera disturbed the radio and the card.** With the camera clock (XCLK) at the usual
   20 MHz, and worse after switching to XGA (faster pixel clock), pings went to 0.2–8 s with up to
   87 % loss at −49 dBm, and the microSD stopped initialising (CRC errors). At 10 MHz both are clean.
   The camera flex, the SD slot and the antenna share one small expansion board; a pendant PCB
   needs this layout checked.
2. **Frames overflowed the camera buffer.** At UXGA, quality 12, a bright scene produced 370 kB
   JPEGs against a 384 kB buffer. The driver dropped them ("FB-OVF"), so the camera looked dead.
   The default is now XGA, which is all the host's YOLO uses at 960 px, and the camera compresses
   harder by itself after repeated empty captures.
3. **SD dataset saves stalled the stream.** Every 10th frame is saved for training. In the flat
   `frames/` folder (~6,600 files with similar long names) FatFs needed ~18 s per save, because it
   makes up a unique 8.3 alias for each new name. Nothing was sent meanwhile, and the host closed
   the idle keep-alive connection. New saves go into folders of 250 and take ~0.6 s. This is shared
   code, so the P4 benefits too.
4. **Host: dead uploads used up its file descriptors (fixed on the host).** Each upload cut off by
   a board reset or a dead link left a connection open on the host. uvicorn times out a connection
   only between requests, and a vanished board sends no FIN. After ~20 resets the host held 241 of
   them, hit the launchd job's 256-file limit and reset every new connection, including the P4 and
   the Cloudflare hostname, until it was restarted. The host now runs uvicorn with
   `demo_spatial/app/esp32_http.py`, which adds TCP keepalive on every connection and a timeout for
   the first request, and raises the open-file limit. Reproduced with the W11 leaving 6 dead
   connections: plain uvicorn kept all 6 for good, while the fixed host had closed them within
   20 s. Streaming through it was unchanged (1.90 fps, 96 ms median POST).

Backlog frames captured before the board first hears from the host this boot carry only a
monotonic stamp, so the host files them at arrival time rather than capture time. A per-boot id in
the file name would let the board convert them once the clock is known.

### The P4-NANO again, on the shared code and the cloud link

After the W11, the P4-NANO went back on the desk with the same microSD card. It was flashed with
`0.5.0-p4`, the first P4 run of the shared `loci_client`, and kept streaming to the cloud host
(`https://loci.0x7d0.com`, host URL and token from NVS):
- **Boot:** same retention-RAM margin as before (10,800 B); camera, codec, card and ESP-SR came up.
- **Stream:** 1280×960 frames through the tunnel, POST 400–420 ms median.
- **Speaker:** the answers to questions asked through the tunnel played (7.8–28.7 s clips).

With the C6's SDIO link at its 40 MHz maximum the P4 reset twice in 10 minutes, on link timeouts
("Unrecoverable host sdio state"). At 20 MHz, with the card at 20 MHz too, the next 10 minutes had no
reset. The board was on a different USB port and cable that day, because the Mac's USB-C ports
would not power it, so the power supply is the other suspect if it comes back. The W11 has no
second chip to lose a link to; this is part of the ESP-Hosted cost in the decision table.

## 3. Vendor numbers for on-device detection (ESP-DL model zoo, int8)

Espressif's own benchmarks for the models we would run on the pendant (S3 at 240 MHz, P4 at
400 MHz; the P4-NANO's v1.3 silicon tops out at 360 MHz, so its numbers are about 10 % worse):

| Model (COCO unless noted) | Input | ESP32-S3 | ESP32-P4 | P4 advantage |
|---|---|---|---|---|
| YOLO11n `coco_detect` | 640² | 4.0 s | 1.40 s | 2.9× |
| YOLO11n `coco_detect` | 320² | 0.99 s (1.38 s measured here) | 0.32 s | 3.1× (4.3× vs our measurement) |
| YOLO26n `yolo26` | 512² | 2.41 s | 0.95 s | 2.5× |
| Pico `pedestrian_detect` (people only) | 224² | 125 ms | 50 ms | 2.5× |

Sources: the READMEs of `esp-dl/models/coco_detect`, `models/yolo26` and
`models/pedestrian_detect`. The same tables list an ESP32-S31 at 320 MHz at roughly 1.8× the S3.
Our 1.38 s is 40 % slower than Espressif's 0.99 s. Our run includes the letterbox preprocessing
and shares PSRAM with the camera DMA and Wi-Fi; the model weights stay in flash because the board
has less than 9 MB of PSRAM.

## 4. Answers without a speaker

The W11 has a microphone but nothing to speak with. Ways to get an answer back to a blind or
low-vision user, from cheapest to most work:

| Option | Hardware | What it gives | Cost / effort |
|---|---|---|---|
| **BLE to the phone** (phone speaks) | none | full sentences in the phone's own TTS voice, through whatever the user already wears (earbuds, hearing aids paired to the phone) | firmware: a GATT service that notifies the answer text; a small phone app or shortcut. This is the uplink we want anyway |
| **I2S amplifier add-on** | MAX98357A breakout + 1–2 W speaker on free header pins (BCLK GPIO1, LRC GPIO2, DIN GPIO3 on I2S1; I2S0 is the mic) | the existing host speech path and the cached per-object clips play on the board, as on the P4 | ~US$5 and a few grams; `audio.c` needs a TX channel (about 40 lines). Speaker size and loudness work against a discreet pendant |
| **Haptics** | LRA/ERM motor + transistor, or a DRV2605L on the IMU's I2C bus | yes/no, "found it", obstacle proximity by pattern | cheap, silent, good as a second channel; not a replacement for speech |
| **Piezo buzzer** | buzzer on one GPIO (LEDC) | tones only | cheapest; acknowledgements only |
| **Display** | SPI panel | nothing for the target users | no |

The S3 has **no Bluetooth Classic** (and neither has the P4's C6), so it cannot stream audio to
BT earbuds itself (no A2DP/HFP; LE Audio is not available on these chips in ESP-IDF). Speech to
the ear goes through the phone in every design without a speaker on the pendant. That fits the
plan to use the phone as the BLE uplink.

What the firmware does today without a speaker:
* Host answers (`ask`, push-to-talk) are printed on the console and shown on the host page
  `/esp32?device=meshology`.
* Offline, the per-object answers are assembled **on the board** from the cached memory
  ("Your cup was last seen in the center, in the middle area.") and kept as text on the card,
  with no host round trip and no fetched speech. A recognised "where is my <label>" prints that
  sentence.

Sending the same text over a BLE characteristic is the next step.

## 5. What this means for the pendant plan

* **Detection on the board.** On the S3, YOLO11n-320 at about 0.5 Hz (one frame every 2 s,
  leaving CPU for Wi-Fi/BLE and speech) is enough for "where did I leave my cup". It is not enough
  for walking-speed obstacle warnings, which need the 125 ms people/obstacle class of models or a
  depth sensor. The IMU can gate capture (no detection while the wearer is still and the scene
  unchanged) to save power. On the P4 the same model runs at about 3 Hz.
* **Hot window of 1–2 h.** Observations are small (label, box, confidence, time: about 24 B). At
  1 Hz with 10 objects, 2 h is 72 k records, about 1.7 MB: too much for the S3's free PSRAM next
  to the detector, easy on the microSD with a small per-label index in RAM. The card layout from
  the P4 build (`/sdcard/loci/…`) already works on the S3 unchanged.
* **Local speech.** ESP-SR command recognition runs on the S3, the chip ESP-SR was built for.
  A 4 s clip takes 1.6 s, which is fine for push-to-talk. Recognition accuracy on the W11's mic
  still has to be shown with a real voice.
* **Phone uplink.** The S3's native BLE removes the ESP-Hosted layer the P4 would need for the
  same tether.
* **Antenna and layout.** A pendant needs an antenna that works against the body. The W11's u.FL
  connector lets us choose one; without it the link was unusable. The camera clock interfering with
  Wi-Fi and the SD card on the W11's small expansion board is the layout lesson for our own PCB.
* **Abrupt power loss is normal for a wearable.** The host must drop connections from a board
  that vanished (finding 4, now done with TCP keepalive). The card must survive resets mid-write: a
  soft reset left it not answering until the init was retried, and the firmware now retries.

## 6. Not verified on this board yet

* Push-to-talk (needs someone to hold BOOT) and ESP-SR recognition accuracy on the PDM mic with a
  real voice.
* Battery operation: no cell attached, no current measured. The vendor figures above are not ours.
* HTTPS / Cloudflare online mode on the S3: the same code as the P4, built and configured, not yet
  exercised.
