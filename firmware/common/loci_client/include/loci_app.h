#pragma once

#include <stdbool.h>

// The thin client's main loop, shared by every board project. The board's app_main() brings up its
// camera and audio path (loci_hw.h), configures the button GPIO and then calls loci_app_run(), which
// never returns:
//
//   camera ──► POST /api/esp32/frame ──► host runs YOLO + LOCI memory; replies with labels
//   speaker ◄── GET /api/esp32/audio/next (16 kHz mono PCM16), when the board has one
//   mic + button ──► POST /api/esp32/voice (push-to-talk); offline: ESP-SR command + answer from the card
//   microSD ── frames kept while offline and uploaded later at their capture time, a cached copy of the
//              host's memory for offline answers, and a dataset of frames + detections
//
// Serial console (115200 on UART0, or the USB-Serial/JTAG port): type `help` for the commands.

typedef struct {
  const char *fw_version;  // shown in the banner and `status`
  const char *board_name;
  bool camera_ok;  // camera_init() succeeded; without it questions still work
  int button_gpio;  // push-to-talk button to GND (configured as input by the board), -1 = none
  // Network and identity (the board's git-ignored main/secrets.h)
  const char *wifi_ssid;
  const char *wifi_pass;
  const char *host_url;
  const char *device_id;
  const char *device_token;
  // Optional board-specific console commands: return true when the line was handled.
  bool (*console_extra)(const char *line);
  const char *console_extra_help;  // appended to the help line, e.g. "imu | led <0|1>"
} loci_app_config_t;

void loci_app_run(const loci_app_config_t *cfg);
