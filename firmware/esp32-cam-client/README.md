# LOCI ESP32 thin-client firmware

Camera frames out, spoken answers in. The board joins Wi-Fi, posts JPEG frames to the LOCI host
at a few frames per second, plays the host's 16 kHz PCM answers on an I2S speaker, and, when it has
a microphone, records push-to-talk questions while a button is held. All intelligence (YOLO, LOCI
memory, Whisper, TTS) runs on the host; see `docs/ESP32_THIN_CLIENT_DEMO.md` for the protocol.

## Supported boards

| PlatformIO env | Board | Camera | Mic | Speaker default pins (BCLK / LRC / DIN) | Button |
|---|---|---|---|---|---|
| `xiao_esp32s3_sense` | Seeed XIAO ESP32S3 Sense | OV2640 (on the Sense board) | PDM, GPIO42 CLK / GPIO41 DATA | GPIO1 (D0) / GPIO2 (D1) / GPIO3 (D2) | GPIO4 (D3) |
| `esp32cam` | AI-Thinker ESP32-CAM | OV2640 | none | GPIO14 / GPIO15 / GPIO2 | GPIO13 |
| `freenove_esp32s3` | Freenove ESP32-S3 WROOM CAM | OV2640 / OV3660 | none | GPIO41 / GPIO42 / GPIO1 | GPIO21 |

The camera pins come from the board vendors' reference designs (`include/board_config.h`). Speaker
and button pins are *our* choice of free GPIOs; override any of them with `-D` flags, for example
`-DLOCI_I2S_DOUT=7` in `platformio.ini` or on the command line:

```bash
PLATFORMIO_BUILD_FLAGS="-DLOCI_I2S_BCLK=5 -DLOCI_I2S_LRC=6 -DLOCI_I2S_DOUT=7" pio run -e xiao_esp32s3_sense
```

### Speaker wiring (MAX98357A or any I2S DAC)

| Amp pin | ESP32 | Note |
|---|---|---|
| VIN | 3V3 or 5V | 5V gives more volume; the XIAO's 5V pin is USB power |
| GND | GND | |
| BCLK | `LOCI_I2S_BCLK` | bit clock |
| LRC / WS | `LOCI_I2S_LRC` | word select |
| DIN | `LOCI_I2S_DOUT` | data from the ESP32 |
| SD | leave floating or 3V3 | floating = (L+R)/2; we send the same sample on both channels |
| GAIN | leave floating (9 dB) | tie to GND for 12 dB, to VIN for 6 dB |
| + / − | speaker | 4 Ω or 8 Ω, 3 W max |

A bare speaker on a DAC pin is **not** supported on these boards: the AI-Thinker's DAC pins
(GPIO25/26) are taken by the camera and the ESP32-S3 has no DAC. Use an I2S amplifier.

### Push-to-talk button

Momentary switch between `LOCI_BTN_PTT` and GND; the pin uses the internal pull-up. With a
microphone: hold, speak, release, and the answer plays. Without one: a press asks the host
"what do you see". Set `-DLOCI_BTN_PTT=-1` if there is no button.

## Build and flash

Requirements: PlatformIO Core (`uv tool install platformio` or `pip install platformio`); the
`espressif32` platform downloads on the first build (a few hundred MB).

```bash
cd firmware/esp32-cam-client
cp include/secrets.h.example include/secrets.h
# edit include/secrets.h: WIFI_SSID, WIFI_PASS, LOCI_HOST="http://<host-ip>:8002", LOCI_DEVICE_ID
pio run -e xiao_esp32s3_sense            # compile
pio run -e xiao_esp32s3_sense -t upload  # flash (auto-detects the serial port)
pio device monitor -b 115200             # watch it
```

`include/secrets.h` is git-ignored. Never commit it.

Board notes:

* **XIAO ESP32S3 Sense** — native USB: if it does not enumerate, hold **B** (BOOT) while plugging
  in, or press B and tap R (RESET). The Sense expansion board must be seated for the camera and mic.
  Serial goes over USB CDC (`ARDUINO_USB_CDC_ON_BOOT=1`), so unplugging during a run drops the
  console but not the firmware.
* **AI-Thinker ESP32-CAM** — no USB. Wire a 3.3 V USB-serial adapter (TX→U0R, RX→U0T, GND, 5V),
  tie **IO0 to GND**, press RESET, flash, remove the IO0 jumper, press RESET. `-mfix-esp32-psram-cache-issue`
  is already set. GPIO12 must stay free (flash-voltage strap). The flash LED on GPIO4 is unused.
* **Freenove ESP32-S3 CAM** — native USB on the right-hand port; the left port is a UART bridge
  which also works for flashing.

### Build flags you may want to change (`platformio.ini` `[env] build_flags`)

| Flag | Default | Meaning |
|---|---|---|
| `LOCI_FPS` | 2 | Frames per second the board aims for; the host may slow it down via `next_ms` |
| `LOCI_FRAME_SIZE` | `FRAMESIZE_VGA` | `FRAMESIZE_QVGA` (320×240) is lighter, `FRAMESIZE_SVGA` sharper |
| `LOCI_JPEG_QUALITY` | 12 | 10 best … 63 worst |
| `LOCI_CAM_VFLIP` / `LOCI_CAM_HMIRROR` | 0 | Flip/mirror if the image arrives upside down |
| `LOCI_SPK_GAIN` | 1.0f | Software gain on playback |
| `LOCI_MIC_GAIN` | 6 | PDM mics are quiet; integer gain before upload |
| `LOCI_PTT_MAX_S` | 8 | Longest recording, buffered in PSRAM |

## What you should see on the serial console

```
=== LOCI ESP32 thin client v0.1.0 — Seeed XIAO ESP32S3 Sense ===
chip ESP32-S3 rev 0, 8 MB flash, PSRAM yes (8386279 B free), device id "esp32"
camera XCLK=10 SIOD=40 SIOC=39 ...  speaker BCLK=1 LRC=2 DIN=3  mic CLK=42 DATA=41  button=4
[cam] sensor PID 0x0026 (OV2640)
[spk] I2S1 ready: BCLK=1 LRC=2 DIN=3, 16000 Hz
[mic] PDM on I2S0 ready: CLK=42 DATA=41
[wifi] connecting to <ssid>....
[wifi] 192.168.1.42  rssi -58 dBm  host http://192.168.1.23:8002
[frame] #1 640x480 38112 B -> "cup","laptop"
[frame] #2 640x480 37990 B -> "cup","laptop"  [audio waiting]
[audio] played 99200 bytes (3.1 s)
```

Two rising beeps at boot mean the speaker works; a third beep means Wi-Fi connected. Serial
commands: `status`, `beep`, `audio`, `ask where is my cup`, `say hello`, `fps 3`.

## Other boards

Define `LOCI_BOARD_CUSTOM` and pass every camera pin:

```ini
[env:mycam]
board = esp32-s3-devkitc-1
build_flags = ${env.build_flags} -DLOCI_BOARD_CUSTOM -DBOARD_HAS_PSRAM
  -DLOCI_CAM_XCLK=15 -DLOCI_CAM_SIOD=4 -DLOCI_CAM_SIOC=5 -DLOCI_CAM_Y9=16 -DLOCI_CAM_Y8=17
  -DLOCI_CAM_Y7=18 -DLOCI_CAM_Y6=12 -DLOCI_CAM_Y5=10 -DLOCI_CAM_Y4=8 -DLOCI_CAM_Y3=9 -DLOCI_CAM_Y2=11
  -DLOCI_CAM_VSYNC=6 -DLOCI_CAM_HREF=7 -DLOCI_CAM_PCLK=13
  -DLOCI_I2S_BCLK=41 -DLOCI_I2S_LRC=42 -DLOCI_I2S_DOUT=1 -DLOCI_BTN_PTT=21
```

On the classic ESP32 the camera driver owns I2S0, so the speaker is always on I2S1; on the S3 the
PDM microphone uses I2S0 and the speaker I2S1.

## Board on the desk

Resolved on 2026-10-25: the board is a **Waveshare ESP32-P4-NANO** (ESP32-P4 rev v1.3, CH343
USB-UART, MIPI-CSI camera, ES8311 codec, ESP32-C6 for Wi-Fi), which this Arduino project cannot
target. Its firmware lives in [`../esp32-p4-client`](../esp32-p4-client/README.md), together with
the captured original banner and the verified pin map. The checklist below is what found that out
and still applies to any new board:

```bash
ls /dev/cu.usb*                                   # a data cable shows cu.usbmodem* (native USB) or cu.usbserial* (CP210x/CH34x)
system_profiler SPUSBDataType | grep -iA8 -E "espressif|cp210|ch34|ftdi|usb jtag"
venv/bin/python -m esptool chip_id                # chip family, MAC, features (hold BOOT while plugging an S3 if it is silent)
venv/bin/python -m esptool flash_id               # flash size → partition table choice
pio device monitor -b 115200                      # press RESET: capture the ORIGINAL firmware banner before flashing over it
```

Then: match the camera pins to the board's silkscreen or vendor pinout (ESP32-S3 + OV2640 on a
XIAO Sense, a Freenove or a generic "ESP32-S3-CAM" all differ), confirm the speaker is driven by
an I2S amplifier and note its BCLK/LRC/DIN wiring, pick the matching `[env:…]` (or
`LOCI_BOARD_CUSTOM`), fill `include/secrets.h`, flash, and check the first frames on
`http://<host>:8002/esp32` (set `LOCI_CAM_VFLIP`/`LOCI_CAM_HMIRROR` if the image is upside down).
Record the banner and the final pin map here.
