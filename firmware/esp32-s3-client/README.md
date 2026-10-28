# LOCI thin client for ESP32-S3 camera boards (Meshnology W11, XIAO ESP32S3 Sense)

ESP-IDF firmware for ESP32-S3 boards with a DVP camera, a PDM microphone and a microSD slot. The
board on the desk is a **Meshnology W11 "ESP32S3 Mini Module"** with its expansion board; its pin
map is the Seeed XIAO ESP32S3 Sense one, so a XIAO Sense runs the same build. The host protocol is
the same as for the P4 (`docs/ESP32_THIN_CLIENT_DEMO.md`, section 2), and so is most of the code:
the main loop, host link, microSD memory and offline commands are shared with
`../esp32-p4-client` through `../common/loci_client`. This project only has the S3 parts:

* `main/camera.c`: esp32-camera on the S3's LCD_CAM DVP port at a 10 MHz XCLK. The sensor encodes
  JPEG itself (the S3 has no JPEG encoder). Streaming is at 1024×768; two UXGA-sized PSRAM frame
  buffers in "grab latest" mode, and the JPEG quality step rises by itself if frames overflow them. Continuous
  autofocus is enabled when the sensor supports it (OV5640 modules with a focus motor).
* `main/audio.c`: the PDM microphone through I2S0 PDM-RX (hardware PDM→PCM, 16 kHz), with a DC
  blocker and a fixed gain. There is **no speaker**, so `audio_speaker_ready()` is false.
* `main/storage_board.c`: the microSD slot in SPI mode on SPI2.
* `main/app_main.c`: bring-up, the QMI8658C IMU, and the board's extra console commands.

Why this is not the PlatformIO sketch in `../esp32-cam-client`: that sketch has no microSD
store-and-forward, device token, HTTPS keep-alive or ESP-SR. The direction we want (detection and
speech on the board) also needs ESP-DL and ESP-SR, which are ESP-IDF components.
`docs/MESHOLOGY_EVALUATION.md` compares this board with the P4-NANO.

## Board facts (discovered 2026-10-26)

| Item | Value |
|---|---|
| USB | native **USB-Serial/JTAG** (VID 0x303A PID 0x1001, serial `84:C7:BB:72:03:84`) → `/dev/cu.usbmodem2101`; no USB-UART bridge |
| Chip | **ESP32-S3** (QFN56) rev **v0.2**, dual LX7 at 240 MHz + ULP, 40 MHz crystal, MAC `84:c7:bb:72:03:84` |
| Memory | 16 MB quad flash (Puya, JEDEC 0x85/0x2018), **8 MB octal PSRAM** in the package (ESP32-S3R8, AP 3.3 V), 512 kB SRAM |
| Wi-Fi / BT | native 2.4 GHz 802.11 b/g/n + BLE 5 on the S3 (u.FL antenna connector; the factory test ran an AP) |
| Camera | **OV3660** (3 MP, 2048×1536, fixed focus, PID 0x3660 read over SCCB) on the expansion board's 24-pin DVP FPC. The factory test prints "OV5640", but that string is hard-coded; the vendor sells kits with OV5640 or OV3660. Pins: XCLK 10, SIOD 40, SIOC 39, D0–D7 = 15 17 18 16 14 12 11 48, VSYNC 38, HREF 47, PCLK 13, no PWDN/RESET |
| Microphone | PDM on the expansion board: CLK GPIO42, DATA GPIO41 |
| microSD | SPI: SCK 7, MISO 8, MOSI 9, CS 21 (CS shared with the active-low USER LED) |
| IMU | QMI8658C 6-axis, I2C SDA 5 / SCL 4 |
| LEDs | WS2812-type RGB LED on **GPIO48, shared with camera D7**: unusable while the camera runs; USER LED on the SD CS line |
| Buttons | BOOT (GPIO0, used as push-to-talk), RESET |
| Power | USB-C 5 V; BAT+/BAT− pads for a 3.7 V Li-ion cell (300–2000 mAh) with an on-board charger (up to 1 A, charge LED); vendor figures: active ≤120 mA, light sleep ≤15 mA, deep sleep ≤8 µA |
| Size | module 24.5 × 17.8 × 4.5 mm, 2.4 g (plus the stacked expansion board and camera flex) |
| Not on the board | speaker/amplifier, display, LoRa radio, GNSS |
| Factory firmware | "W11 Factory Test + Data Output" (Arduino-ESP32 3.3.3, sketch `W11_Test`): IMU, mic, SD, camera, Wi-Fi AP, BLE, ADC self-test, then IMU/ADC/SD readings once per second. A full 16 MB backup was read before the first flash (sha256 `8d120121…7a193`, kept outside the repo) |

Original banner (captured before flashing; the full 16 MB flash image was backed up first):

```
ESP-ROM:esp32s3-20210327
rst:0x15 (USB_UART_CHIP_RESET),boot:0x8 (SPI_FAST_FLASH_BOOT)
==========================================================
W11 Factory Test + Data Output
==========================================================
[1/7] Testing QMI8658C IMU...OK (self-test pass)
[2/7] Testing digital mic...OK (peak=11363, mean=9567, range=6439, nonzero=256)
[3/7] Testing SD card...OK (mounted), size=60906 MB
[4/7] Testing OV5640 camera...E (2207) gdma: gdma_disconnect(309): no peripheral is connected to the channel
OK (init success)
[5/7] Testing WiFi...OK (AP mode), IP=192.168.4.1
[6/7] Testing BLE...OK (init ok)
[7/7] Testing ADC...OK (avg=2700, min=0, max=3403)
==================== Data Output Mode ====================
IMU ACCEL: x->0.42, y->0.11, z->0.81  GYRO: x->4.34, y->-0.34, z->-0.08  Temperature:36.16°C
ADC: raw=3364, voltage=5.422V
SD: card=60906 MB, total=60902 MB, used=751 MB
```

The vendor's ADC example reads GPIO15, which is camera D0 on this pin map, so it is not a usable
battery gauge while the camera runs. Battery sensing needs a free ADC pin (GPIO1–9 on the
header) and a divider.

## Build and flash

Requirements: ESP-IDF **5.4 or newer** with the `esp32s3` target (this Mac has v6.1 at
`~/esp/esp-idf`). Components come from the registry on the first build: `esp32-camera` 2.1.x,
`esp-sr` 2.6.x (with `esp-dl`).

```bash
cd firmware/esp32-s3-client
cp main/secrets.h.example main/secrets.h       # WIFI_SSID, WIFI_PASS, LOCI_HOST, LOCI_DEVICE_ID "meshology"
. ~/esp/esp-idf/export.sh
idf.py build
idf.py -p /dev/cu.usbmodem2101 flash monitor   # also writes the ESP-SR models to the "model" partition
```

`main/secrets.h` is git-ignored. Never commit it. The board's live view is
`http://<host>:8002/esp32?device=meshology`.

Flashing needs no button presses: the USB-Serial/JTAG port resets the chip into the bootloader.
If a deep-sleep test leaves the port missing, hold BOOT while plugging the cable in.

## Console

On the USB port at any baud rate: `status | sd | objects | cache | offline | recog | snap | ask <q>
| say <t> | fps <n> | host <url>|default | token <value>|clear`, which are shared with the P4. The S3
adds `imu` (accelerometer and temperature), `mic` (2 s microphone level), `af` / `af now`
(autofocus status, one focus cycle) and `res <qvga|vga|svga|xga|hd|sxga|uxga|fhd|qxga>`
(camera resolution at runtime). Opening the port does not reset the board when DTR and RTS stay
asserted (the scratchpad `serial_cmd.py` helper does that).

## On-board detection (`detect`)

The firmware carries ESP-DL's COCO detector (YOLO11n, 320×320, int8; `coco_detect` component,
model in the 4 MB `coco_det` partition that `idf.py flash` writes). In `mode host` it is a
measuring tool: the frame loop uploads JPEGs and the host decides what is remembered. In
`mode cloud` (`docs/ESP32_CLOUD_MODE.md`) its detections are the board's own sightings in LOCI
Cloud. The code is shared with the P4 in `../common/loci_detect`.

* `detect [n]` runs it on `n` live frames. Each frame is decoded at reduced size (JPEG → RGB565,
  scaled by ½, ¼ or ⅛ until it is at most 640 px wide) and the labels are printed with the timing.
* `detect sd [n]` runs it on every 100th frame of the card's dataset and prints the host's
  YOLOv8n labels for that frame next to the board's.

Measured on the W11 (2026-10-26): model load 0.40 s, **1.38–1.49 s per frame** (up to 1.8 s while
the upload loop competes), decode 41 ms for VGA and ~230 ms for UXGA. Results are in
`docs/MESHOLOGY_EVALUATION.md`.

Three build details this needed:

* **ESP-DL op stripping.** ESP-DL 3.3 compiles only the operators and Conv kernels named in
  registered `compile_req` files. esp-sr registers its own list and `coco_detect` registers none,
  so the detector failed at load with `dl::Model: Do not support Swish` and then crashed.
  The project `CMakeLists.txt` clears `ESPDL_COMPILE_REQS` after `project()`, before ESP-DL's
  deferred finalize step, which builds the full operator set. The app grows to 3.4 MB of the 4 MB
  partition.
* **Data cache.** `CONFIG_ESP32S3_DATA_CACHE_64KB` with 64-byte lines, as in ESP-DL's examples,
  took inference from 1.97 s to 1.38 s per frame. The weights stay in flash because ESP-DL copies
  them to PSRAM only on boards with 9 MB or more.
* **Watchdog.** One inference keeps CPU1 busy for over a second, so the IDLE1 task-watchdog check
  is off.

## Measured on the desk (2026-10-26)

| | |
|---|---|
| Boot | `app_main` at 0.86 s; Wi-Fi joins in under 2 s (4–13 s without the antenna); SD, cached memory and ESP-SR ready ~3 s later |
| Camera | OV3660, JPEG at quality 12: VGA 24 kB, XGA 61 kB, SXGA 190–214 kB, UXGA 64–312 kB (scene-dependent); capture returns immediately (grab latest) |
| microSD | the P4's 64 GB FAT32 card read as is over SPI (now 10 MHz): 3,231 dataset frames, cached memory, no P4 backlog. Offline capture to `backlog/` at 1 fps; a dataset save (every 10th frame) takes ~0.6 s |
| ESP-SR | MultiNet7 English, 54 phrases from the cached memory; 1.5–1.7 s for a 4 s clip. Recognition not yet verified with a real voice |
| IMU | QMI8658C at 0x6B |
| Heap with Wi-Fi + ESP-SR + detector | PSRAM 2.4–2.9 MB free, internal 59 kB free (17.5 kB minimum) |
| Wi-Fi | −31 to −56 dBm with the kit's antenna on the u.FL connector, 0 % loss and 9 ms average for 1,000-byte pings. Without it: −81 to −86 dBm, 75–100 % loss, uploads stall and everything goes to the backlog |
| **Streaming to the host** | 150 s at 1024×768: **1.96 fps, POST 109 ms median / 205 ms p90 / 345 ms max**, 0 failures, host YOLOv8n 63–82 ms per frame |
| `ask` | answered by the host in 2.2 s and printed on the console |

## Without a speaker

The shared code checks `audio_speaker_ready()`:

* Host answers to `ask` and push-to-talk are logged on the console (`heard: …`, `answer: …`) and
  shown on the host page. The host still queues speech for the device, capped at
  `LOCI_ESP32_AUDIO_QUEUE_MAX` clips, but the board never downloads it.
* The microSD cache keeps `memory.json` but fetches no speech. The per-object answers are kept as
  text (`obj_<label>.pcm.txt`, "Your cup was last seen …"), assembled on the board from the cached
  memory with no host round trip. An offline "where is my cup" recognised by ESP-SR prints that
  sentence. With no object recognised, `offline` and the button print every remembered object
  with its place.
* Beeps are silent, and the RGB LED cannot stand in for them because GPIO48 is also camera D7.

## Troubleshooting

* **RSSI around −85 dBm, pings losing most packets, frames failing with `Poll timeout or error,
  errno=Connection already in progress` / `write failed after 0 bytes`.** The antenna is not
  plugged into the u.FL socket; the module has no PCB antenna. The errno text is a stale
  `EINPROGRESS` from the non-blocking connect, not the cause. Small requests (`ask`) may still get
  through.
* **The camera reports OV3660 although the box and the factory test say OV5640.** The factory test
  string is hard-coded; trust the PID in the boot log. `CONFIG_OV3660_SUPPORT`,
  `CONFIG_OV5640_SUPPORT` and `CONFIG_OV2640_SUPPORT` are all on, so any of them works. Autofocus
  is only attempted on an OV5640.
* **Pings of 0.2–8 s, heavy loss at a good RSSI, SD card init failing with CRC or `send_if_cond`
  errors.** The camera clock disturbs the radio and the SD lines on the expansion board. It was
  bad at a 20 MHz XCLK, worse at XGA; the default is 10 MHz (`BOARD_CAM_XCLK_HZ`) with the SD bus at
  10 MHz.
* **`cam_hal: FB-OVF`, then `Failed to get frame: timeout`, and the camera looks dead.** The JPEGs
  are larger than the frame buffer (UXGA at quality 12 in a bright, detailed scene reached 370 kB of
  384 kB). Use a smaller `res` or a higher `quality` step. The firmware raises the step by itself
  after repeated empty captures.
* **One frame in ten takes many seconds, and the host closes the keep-alive connection.** SD saves
  into a huge flat folder: FatFs scans the whole folder several times to make up an 8.3 alias for
  each long name (~18 s with 6,600 files over SPI). Dataset frames now go to `dataset/<NNNN>/`
  (250 per folder). The frame log shows the POST time and warns separately when a save is slow.
* **The card is not found after a soft reset.** A reset does not power-cycle the card, so one that
  was mid-write may not answer at first. The firmware retries the init up to four times; unplugging
  USB always clears it.
* **The host stops accepting connections after many board resets.** Hosts started before
  `demo_spatial/app/esp32_http.py` existed kept every upload cut off by a reset open (uvicorn times
  out only idle connections between requests), until the launchd job's 256 open files ran out.
  `run_esp32_demo.sh` now starts uvicorn with that protocol (TCP keepalive, a timeout for the first
  request) and a higher open-file limit. On an old host, restart it with
  `launchctl kickstart -k gui/$(id -u)/com.loci.esp32-host`.
* **Reading the whole flash back is slow.** `esptool read-flash 0 ALL` over the USB-Serial/JTAG
  port runs at about 12 kB/s, so 16 MB takes about 24 minutes. Writing is about 1.3 Mbit/s.
* **Opening the serial port reboots the board.** Leave DTR and RTS asserted as `idf.py monitor`
  does. Dropping DTR while RTS is high resets the S3 through its USB-Serial/JTAG logic.
