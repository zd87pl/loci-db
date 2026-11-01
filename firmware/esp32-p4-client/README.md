# LOCI thin client for the Waveshare ESP32-P4-NANO

The board on the desk turned out to be an **ESP32-P4**, not an ESP32-S3: the MIPI-CSI camera, the
ES8311 codec and the Wi-Fi-over-SDIO companion chip need ESP-IDF rather than the Arduino sketch in
`../esp32-cam-client` (which stays for XIAO / ESP32-CAM / Freenove boards). This project does the
same job with the same host protocol (`docs/ESP32_THIN_CLIENT_DEMO.md`, section 2):

* camera (OV5647, MIPI-CSI 2-lane) → ISP → **hardware JPEG** → `POST /api/esp32/frame`
* `GET /api/esp32/audio/next` → 16 kHz mono PCM16 → **ES8311 + NS4150B** speaker
* BOOT button held → onboard mic → `POST /api/esp32/voice` → answer plays back
* serial console: `status | beep | audio | ask <q> | say <text> | fps <n> | vol <0-100>` (`help` lists all)

Code layout: this project holds only the P4-specific parts (`main/camera.c` esp_video + hardware
JPEG, `main/audio.c` ES8311, `main/storage_board.c` the SDMMC slot, `main/app_main.c` bring-up).
The main loop, the host link, the microSD memory and the offline commands live in
`../common/loci_client` and are shared with the ESP32-S3 client in `../esp32-s3-client`; the
interface a board implements is `../common/loci_client/include/loci_hw.h`.

**Cloud mode** (`mode cloud`, after `cloud claim`): no host. The board runs its own detector
(`../common/loci_detect`, ESP-DL YOLO11n on the P4's hardware JPEG decode), answers locally first
and uses LOCI Cloud for the rest; see `docs/ESP32_CLOUD_MODE.md`. That added a `coco_det`
partition and grew the app partition to 6 MB (the `model` partition is now 3.5 MB); NVS keeps its
offset, so settings survive the reflash.

## On the desk with the shared code (2026-10-26)

Firmware `0.5.0-p4` (this project plus `../common/loci_client`, the first run of the shared loop on
the P4) flashed over yesterday's `0.4.0-p4`. The host URL `https://loci.0x7d0.com` and the device
token stayed in NVS. The boot keeps the same retention-RAM margin (10,800 B free at `app_main`).
ES8311, OV5647 + hardware JPEG at 1280×960, the C6 link, the 4-bit card and ESP-SR (54 phrases)
came up as before.

Through the Cloudflare tunnel, with the camera facing a dark surface (noisy 280–390 kB frames):

| | 40 MHz SDIO (IDF default here) | 20 MHz SDIO + 20 MHz card (now) |
|---|---|---|
| 10-minute soak | 1,004 frames, POST 400 ms median / 502 ms p90, **2 resets** (C6 link timeouts) | 855 frames, POST 420 ms median / 545 ms p90, **0 resets**, 0 failed uploads |
| Spoken answers | 12.9 s and 28.7 s clips played | 6 clips played (7.8–28.7 s); the frame loop pauses while a clip plays |

## Board facts (discovered 2026-10-25)

| Item | Value |
|---|---|
| USB | CH343 USB-UART bridge (`USB Single Serial`, VID 0x1A86 PID 0x55D3) → `/dev/cu.usbmodem5ABA0768111`; native USB Serial/JTAG is not available on this chip revision |
| Chip | ESP32-P4 rev **v1.3**, dual core + LP core, 40 MHz crystal, MAC `80:f1:b2:d0:83:d5` |
| Memory | 16 MB QIO flash at 80 MHz, 32 MB hex PSRAM at 200 MHz |
| Wi-Fi / BT | ESP32-C6-MINI-1 over SDIO (slot 1: CLK 18, CMD 19, D0 14, D1 15, D2 16, D3 17), slave reset GPIO54, ESP-Hosted |
| Audio | ES8311 at I2C 0x18; I2S0 MCLK 13 / BCLK 12 / WS 10 / DOUT 9 / DIN 11; NS4150B PA enable GPIO53 (active high); MX1.25 2-pin speaker header (8 Ω 2 W); onboard microphone |
| I2C | SDA GPIO7, SCL GPIO8 (shared by codec, camera SCCB, touch) |
| Camera | MIPI-CSI, 2 lanes; OV5647 default format 800×800 RAW8 @ 50 fps (menuconfig) |
| Button | BOOT on strapping GPIO35 (used as push-to-talk) |
| Original firmware | `esp_brookesia_demo` v1.0.1, ESP-IDF v5.4, compiled 2025-05-27, with a Waveshare 10.1" DSI panel driver (JD9365) that failed on I2C because no display is attached |

Original banner excerpt (captured before flashing):

```
ESP-ROM:esp32p4-eco2-20240710
rst:0x1 (POWERON),boot:0x30f (SPI_FAST_FLASH_BOOT)
I (25) boot: ESP-IDF v5.4 2nd stage bootloader
I (27) boot: chip revision: v1.3
I (44) boot.esp32p4: SPI Flash Size : 16MB
I esp_psram: Found 32MB PSRAM device
I (1628) app_init: Project name:     esp_brookesia_demo
I (1633) app_init: App version:      1.0.1
I (1745) H_API: ESP-Hosted starting. ...
I (1904) ES8311: Work in Slave mode
I (1984) ESP32_P4_EV: Install Waveshare 10.1-DSI-TOUCH-A or 101M-8001280-IPS-CT-K LCD control panel
E (2003) i2c.master: I2C transaction unexpected nack detected
```

## Build and flash

Requirements: ESP-IDF **5.4 or newer** with the `esp32p4` target installed (this Mac has v6.1 at
`~/esp/esp-idf`). Components come from the registry on the first build (`esp_video`,
`esp_cam_sensor`, `esp_wifi_remote`, `esp_hosted` 2.12.x, `esp_codec_dev`).

```bash
cd firmware/esp32-p4-client
cp main/secrets.h.example main/secrets.h       # WIFI_SSID, WIFI_PASS, LOCI_HOST=http://<host-ip>:8002
. ~/esp/esp-idf/export.sh
idf.py set-target esp32p4
idf.py build
idf.py -p /dev/cu.usbmodem5ABA0768111 flash monitor   # ctrl-] leaves the monitor
```

`main/secrets.h` is git-ignored. Never commit it.

`esp_hosted` is pinned to 2.12.x and `esp_video` to 2.1.0 on purpose: the newer lines expect
Wi-Fi and camera-controller headers newer than the IDF 6.1 snapshot on this Mac, and the C6 on the
board shipped with a 2.x hosted slave.

`sdkconfig.defaults` selects **ESP32-P4 revisions below 3.0** (`CONFIG_ESP32P4_SELECTS_REV_LESS_V3=y`,
minimum v1.0). IDF 6 builds for v3.x silicon by default and its bootloader then refuses to flash a
v1.3 board ("requires chip revision in range [v3.1 - v3.99]"). Boards with v3.x silicon need the
opposite setting; the two are mutually exclusive.

## What you should see (real log from the desk test, 2026-10-25)

```
I (1597) loci: === LOCI thin client v0.2.0-p4 on Waveshare ESP32-P4-NANO (ESP32-P4 rev v1.3, 2 cores) ===
I (1700) audio: ES8311 ready: 16000 Hz mono, volume 75%, mic gain 36 dB, PA on GPIO53
I (1877) ov5647: Detected Camera sensor PID=0x5647
I (1936) camera: /dev/video0: driver MIPI-CSI, card MIPI-CSI, version 2.1.0
I (1937) camera: /dev/video10: driver JPEG, card JPEG, version 2.1.0
I (1938) camera: capture 800x800, pixel format RGBP, JPEG quality 80
I (9455) wifi: got ip 172.20.1.183
I (9780) loci: [frame] #1 800x800 20758 B in 239 ms -> -
I (24833) loci: [frame] #31 800x800 81376 B in 240 ms -> "person"
I (478311) loci: [ptt] recording... release the button to ask
I (482137) loci: [ptt] 3.8 s captured
I (487758) host: heard: "Where is my cup?"
I (487758) host: answer: Your cup was last seen in the center, in the middle area. That was 3 minutes ago, at 10:19 AM. It's to the left of the scissors.
I (499689) host: played 383232 bytes (12.0 s) in 11904 ms
```

Two rising beeps at boot mean the codec and amplifier work; a third beep means Wi-Fi is up. A
short high beep after releasing BOOT means the clip was captured; a low beep means the press was
too short. `snap` on the console captures and encodes one frame without a host (800×800 → ~20–90 kB
in ~36 ms).

## microSD card

A FAT32 card in the slot (SDMMC slot 0: CLK 43, CMD 44, D0–D3 39–42, 4-bit at 20 MHz, powered by LDO channel 4)
is mounted at `/sdcard` after Wi-Fi comes up and used for:

| Path | Purpose |
|---|---|
| `/sdcard/loci/backlog/<captured_ms>.jpg` | frames captured while the host or Wi-Fi was unreachable (1 fps); uploaded oldest-first with `X-Backlog`/`X-Captured-Ms` once the host answers again, so the memory has no gap |
| `/sdcard/loci/memory.json`, `memory.pcm`, `memory.rev` | the host's remembered objects and a spoken summary, refreshed when the host's `memory_rev` changes; the button plays the summary when the host cannot be reached |
| `/sdcard/loci/offline.pcm` | fixed prompt fetched once from the host |
| `/sdcard/loci/frames/<ms>.jpg` + `.json` | every 10th live frame with the host's detections (dataset for a custom detector) |
| `/sdcard/loci/voice/<ms>.wav` | push-to-talk recordings |

Two things the P4 needs for this to work, both in `sdkconfig.defaults`: `CONFIG_VFS_MAX_COUNT=15`
(esp_video registers every `/dev/video*` node with the VFS and the default table of 8 is full, so the
card's `esp_vfs_fat_register` fails with `ESP_ERR_NO_MEM`), and the card is attached by hand to the
SDMMC controller that ESP-Hosted already created for the C6 on slot 1 (`main/storage_board.c`): a second
`sdmmc_host_init()` returns `ESP_ERR_NOT_FOUND` on IDF 6.1 and the all-in-one mount helper gives up.

A 64 GB card arrives formatted as exFAT, which IDF's FatFs cannot read. The firmware reports
"card has no FAT filesystem"; build with `-DLOCI_SD_FORMAT_IF_NEEDED=1` once (it formats the card as
FAT32, erasing it) or format it on a computer. Console: `sd` (card info), `objects` (cached memory),
`offline` (play the cached summary), `requeue` (put backlog frames the host refused, kept in
`backlog_bad/`, back in the upload queue, one every 250 ms while the board streams to the host),
`reboot` (soft reset: the clock taken from the host survives it, a power cut does not; `status` shows
the boot number and the clock's source). After a panic or watchdog reset the kept clock is off by
the RTC slow clock's drift over the previous boot (about 0.1-1 %, so up to minutes after hours of
uptime) until the host answers again.

Backlog and voice files are named `<ms>_<boot>_<seq>`, so frames recorded before the clock is known
(their `<ms>` counts from boot) no longer overwrite each other across resets.

## Offline questions (ESP-SR on the board)

When the host cannot be reached, a button press no longer just plays the generic summary: the
recorded clip is run through ESP-SR's audio front end (WebRTC noise suppression) and **MultiNet7
English** (`../common/loci_client/voice_local.c`), with no wake word because the button is the trigger. The command list
is rebuilt whenever the cached memory changes: "what do you see" plus "where is my / where are my
<label>" for every remembered label and for a fixed set of household objects. A recognised object
plays `/sdcard/loci/obj_<label>.pcm` ("Your cup was last seen in the center, in the middle area."),
which the board fetches from the host's `phrase.pcm` endpoint once per boot and whenever an object's
position changes (a `.txt` stamp next to each clip holds the sentence it was made from, so unchanged
objects are not refetched); anything else falls back to the cached summary.

Models live in the 3.5 MB `model` partition (`CONFIG_MODEL_IN_FLASH=y`, `CONFIG_SR_MN_EN_MULTINET7_QUANT=y`,
`CONFIG_SR_WN_NONE=y`); `idf.py flash` writes them with the app. The esp-sr component ships a build
for exactly this silicon (`include/esp32p4_less_v3_idf6`). Console: `recog` records 4 s from the
microphone, prints the recognised command and plays the cached answer, which makes the feature
testable without taking the host down; `cache` fetches any per-object answers the card lacks right
away.

## Online mode (Cloudflare Tunnel, HTTPS, device token)

The board talks plain HTTP to a host on the LAN. To reach the same host from any network (a phone
hotspot, the LTE pendant later) without opening a port, put the host behind a Cloudflare Tunnel:

1. Start the host as usual (`demo_spatial/run_esp32_demo.sh`). On first start it creates a device
   token in `~/.local/share/loci-esp32/device_token` (mode 0600, never printed) and shows the path.
2. Start the tunnel: `demo_spatial/run_esp32_tunnel.sh`. Without a Cloudflare account it opens a
   *quick tunnel* and prints a random `https://….trycloudflare.com` URL (new on every start; for
   tests). For a stable hostname on your own domain, once, in your own terminal:

   ```bash
   cloudflared tunnel login                          # only if ~/.cloudflared/cert.pem is missing
   cloudflared tunnel create loci-esp32
   TUNNEL_NAME=loci-esp32 TUNNEL_HOSTNAME=loci.example.com demo_spatial/run_esp32_tunnel.sh
   cloudflared tunnel --config ~/.cloudflared/loci-esp32.yml route dns loci-esp32 loci.example.com
   ```

   The script writes `~/.cloudflared/loci-esp32.yml` (the ingress in
   `demo_spatial/cloudflare/esp32-tunnel.yml.example`: only the board's routes, 404 for the rest) and
   never touches `~/.cloudflared/config.yml`, so another tunnel on the same machine keeps running.
   Pass that config to the `route dns` step as well: when `~/.cloudflared/config.yml` names another
   tunnel, cloudflared resolves a tunnel *name* to that one (`cloudflared tunnel info loci-esp32`
   shows the other tunnel) and the CNAME lands on the wrong tunnel, whose catch-all then answers an
   empty 404. `--overwrite-dns` replaces such a record.
   Two launchd jobs keep the whole chain up across logouts and reboots on a Mac:
   `demo_spatial/launchd/com.loci.esp32-host.plist.example` for the host (port 8002) and
   `demo_spatial/cloudflare/com.loci.esp32-tunnel.plist.example` for the connector; both log to
   `~/Library/Logs/`.
3. On the board's console, once: `token <contents of the token file>`, then `host https://<url>`.
   Both live in NVS, so neither needs a reflash or a change to `secrets.h`; `host default` returns
   to the compiled-in LAN URL and `token clear` removes the token. `status` shows the host in use
   and whether a token is set.

On the wire every request now carries `X-Device-Token`; `https://` hosts are verified against the
IDF certificate bundle (Cloudflare's edge certificates are in it); the keep-alive frame connection
is reused across TLS, so only the first frame pays the handshake. On the host, requests that arrive
from outside the LAN see only `/api/esp32/*` and the `/esp32` page (everything else answers 404)
and need the token, while the LAN keeps working with none (`LOCI_ESP32_AUTH=auto`; `always`
requires it on the LAN too). The offline path is untouched: when the tunnel or the host goes
away, the board stores frames on the card, answers from the cached memory and recognises
questions on-device as before.

Known gap: `CONFIG_MBEDTLS_HAVE_TIME_DATE` is off (the IDF default), so certificate validity dates
are not checked; the board has no clock before its first host reply. Turning it on needs SNTP
before the first HTTPS request, a follow-up for a production pendant.

## Tuning (`main/board_p4_nano.h` and `../common/loci_client/include/loci_config.h`, overridable with `-D`)

| Macro | Default | Meaning |
|---|---|---|
| `LOCI_FPS` | 2 | Frames per second the board aims for; the host's `next_ms` can slow it |
| `LOCI_JPEG_QUALITY` | 80 | Hardware encoder quality (1–100) |
| `LOCI_SPK_VOLUME` | 75 | Codec output volume in percent |
| `LOCI_MIC_GAIN_DB` | 36 | Microphone analog gain |
| `LOCI_PTT_MAX_S` | 8 | Longest push-to-talk clip (PSRAM) |
| camera format | 1280×960 RAW10, 2×2 binning | two lines in `sdkconfig.defaults`: enable the format (`CONFIG_CAMERA_OV5647_MIPI_<fmt>=y`) *and* pick it as default (`..._MIPI_DEFAULT_FMT_<fmt>=y`); the default-format choice only lists enabled formats. Others: 800×640, 800×800, 800×1280 RAW8, 1920×1080 RAW10 |

## Troubleshooting

* **Boot loop with `sleep_clock: failed to allocate memory for system sleep clock ICG` then
  `cpu_start: init function ... has failed (0x101), aborting`** — the P4 heap hands ordinary
  internal allocations out of its 131 KiB `RETENT_RAM` region first, and ESP-Hosted allocates its
  SDIO buffers from a C constructor that runs before the system's sleep-clock init, which needs
  RETENTION memory from that same region. With the default 20+20 queue entries the region was
  within a few hundred bytes of full on this board, so any `.bss` growth tipped it into a boot loop.
  `sdkconfig.defaults` sets `CONFIG_ESP_HOSTED_SDIO_TX_Q_SIZE`/`RX_Q_SIZE` to 8 to leave headroom;
  if it comes back after adding code, lower them further or move more static data to PSRAM.
* **Boot loop with `assert failed: sleep_retention_entries_do_destroy sleep_retention.c:498`
  right after `sleep_gpio: Enable automatic switching ...`** — the same shortage, seen when ESP-SR
  was added (its libraries bring ~37 kB of `.data`/`.bss` into that region): the sleep-clock init
  runs out of RETENTION memory and this IDF build asserts on the way out instead of logging. The
  boot banner prints the free retention RAM at the start of `app_main`
  (`internal N B (retention M B)`, 11 kB on this board with the current build); that figure is the
  margin the system init had left, so keep it clearly positive. The steady-state value in `status`
  sits near zero by design, because the allocator hands ordinary internal allocations out of that
  region first. Levers, all in place now: speech-library `.bss` in PSRAM (`main/linker.lf`
  with `CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y`), ESP-Hosted task stacks in PSRAM
  (`CONFIG_ESP_HOSTED_DFLT_TASK_FROM_SPIRAM=y`), heap code in flash
  (`CONFIG_HEAP_PLACE_FUNCTION_INTO_FLASH=y`), the 8+8 SDIO queue entries, and the firmware's own
  big buffers (HTTP reply, clip playback, push-to-talk) allocated from PSRAM.
* **`HTTP_CLIENT: No transport found … enable HTTPS at menuconfig`** after `host https://…` — the
  HTTP client was built without TLS; `CONFIG_ESP_HTTP_CLIENT_ENABLE_HTTPS=y` is in
  `sdkconfig.defaults` now (delete `sdkconfig` to re-apply defaults). Until the reflash the board
  simply runs its offline path.
* **`esp-x509-crt-bundle: Failed to verify certificate` right after
  `psa_crypto_driver_esp_ecdsa: ECDSA peripheral not supported on this chip revision`** — IDF
  enables hardware ECDSA verification for every P4, but silicon before v3.0 has no such peripheral
  and the PSA driver fails instead of falling back, so any ECDSA-signed chain (Cloudflare, Let's
  Encrypt E-series) is rejected. `CONFIG_MBEDTLS_HARDWARE_ECDSA_VERIFY=n` keeps verification in
  software.
* **`esp-x509-crt-bundle: No matching trusted root certificate found`** on a Cloudflare hostname —
  Cloudflare's edge sends `GTS Root R4` cross-signed by `GlobalSign Root CA`, the bundle verifier
  only looks up the issuer of the *last* certificate it receives, and that 1998 root is no longer in
  Mozilla's list (so not in the IDF bundle) although it is valid until 2028 and still shipped by
  Apple and Microsoft. `../common/certs/extra_roots.pem` adds it through
  `CONFIG_MBEDTLS_CUSTOM_CERTIFICATE_BUNDLE_PATH`; remove it once Cloudflare drops the cross-sign.
* **`esp_video_init failed`** — camera ribbon not seated or a sensor other than OV5647: enable it
  under `Component config → Camera sensor configuration` (SC2336 is the other one Waveshare ships).
* **`ES8311 not found on I2C 0x18`** — I2C pull-ups or a hung bus; power-cycle the board.
* **`sdmmc_send_cmd returned 0x107`, `H_SDIO_DRV: Unrecoverable host sdio state`, then a reboot** —
  the SDIO link to the C6 timed out twice in a row and ESP-Hosted restarted the P4. At the 40 MHz
  maximum this happened twice in 10 minutes on 2026-10-26, once 4.7 s after boot before the card was
  mounted, so it was not the card. The P4 was streaming 300 kB frames through the Cloudflare tunnel
  from a different USB port than the day before. With `CONFIG_ESP_HOSTED_SDIO_CLOCK_FREQ_KHZ=20000`,
  and the card at 20 MHz so the shared controller never re-clocks, the next 10 minutes had none. If
  it comes back, suspect the power supply first: Wi-Fi TX bursts plus the speaker amplifier on a weak
  USB port.
* **No IP / `esp_wifi_init` error** — the C6 slave firmware must speak ESP-Hosted 2.x; the log prints
  the slave's version at start. Reset GPIO54 is pulsed on every boot (`HOST_CP_RESET_STRATEGY_ALWAYS`).
* **Image upside down** — rotate on the host with `LOCI_ESP32_ROTATE=180` (no sensor flip is wired yet).
* **Flat blue-grey frames, nothing detected** — the fixed-focus OV5647 is looking at a blank surface
  or is out of focus: prop the board so the lens faces the desk from 0.5–1 m and check the live
  view at `http://<host>:8002/esp32?device=p4-nano`; the lens barrel can be turned a little to
  focus. The blue cast is white balance (the ISP pipeline controller runs AE/AWB from
  `esp_ipa`; tuning is a follow-up) and does not stop YOLO.
* **Serial tools reboot the board** — opening the CH343 port with RTS asserted then dropping DTR
  pulses EN. `idf.py monitor` handles it; for scripts, leave DTR and RTS asserted (see the
  `serial_cmd.py` helper in the notes) or accept the reset.
