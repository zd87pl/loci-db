// LOCI thin client for the Waveshare ESP32-P4-NANO.
//
//   camera (MIPI-CSI, hardware JPEG) ──► POST /api/esp32/frame ──► host runs YOLO + LOCI memory
//   speaker (ES8311 + NS4150B)       ◄── GET  /api/esp32/audio/next (16 kHz mono PCM16)
//   mic (ES8311) + BOOT button       ──► POST /api/esp32/voice (push-to-talk), answer plays back
//   microSD (SDMMC 4-bit)            ──  store-and-forward, cached memory, dataset (loci_app.h)
//   ESP-DL COCO YOLO11n (320, int8)  ──  `mode cloud`: the board's own sightings, no host (loci_app.h)
//
// This file brings up the P4-specific hardware; the loop, the host link and the microSD logic are
// shared with the S3 client in firmware/common/loci_client.

#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "audio.h"
#include "board_p4_nano.h"
#include "camera.h"
#include "detect_local.h"
#include "loci_app.h"
#include "loci_hw.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Create main/secrets.h from main/secrets.h.example (WIFI_SSID, WIFI_PASS, LOCI_HOST)."
#endif

#ifndef LOCI_DEVICE_TOKEN
#define LOCI_DEVICE_TOKEN ""  // older secrets.h without a token: LAN-only unless `token <value>` is used
#endif

#define FW_VERSION "0.5.0-p4"

static const char *TAG = "loci";

// ---- on-board detection benchmark -----------------------------------------------------------
static void run_detect(int runs) {
  if (!detect_local_init()) return;
  int n = 0, dec_sum = 0, inf_sum = 0;
  for (int i = 0; i < runs; i++) {
    uint8_t *jpeg = NULL;
    size_t len = 0;
    if (camera_capture_jpeg(&jpeg, &len, NULL, NULL) != ESP_OK) continue;
    detect_local_result_t r;
    bool ok = detect_local_run(jpeg, len, &r);
    camera_release();
    if (!ok) continue;
    n++;
    dec_sum += r.decode_ms;
    inf_sum += r.infer_ms;
    ESP_LOGI(TAG, "[detect] %dx%d: decode %d ms, detect %d ms -> %s", r.width, r.height, r.decode_ms, r.infer_ms,
             r.count ? r.labels : "-");
  }
  if (n) ESP_LOGI(TAG, "[detect] %d runs: decode avg %d ms, detect avg %d ms", n, dec_sum / n, inf_sum / n);
}

static bool console_extra(const char *line) {
  if (!strcmp(line, "detect") || !strncmp(line, "detect ", 7)) {
    int runs = atoi(line + 6);
    run_detect(runs > 0 && runs <= 50 ? runs : 1);
    return true;
  }
  return false;
}

void app_main(void) {
  esp_chip_info_t chip;
  esp_chip_info(&chip);
  ESP_LOGI(TAG, "=== LOCI thin client v%s on %s (ESP32-P4 rev v%d.%d, %d cores) ===", FW_VERSION, BOARD_NAME,
           chip.revision / 100, chip.revision % 100, chip.cores);
  ESP_LOGI(TAG, "device id \"%s\", host %s, %d fps target, PSRAM %u B free, internal %u B (retention %u B)",
           LOCI_DEVICE_ID, LOCI_HOST, LOCI_FPS, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_RETENTION));

#if BOARD_BTN_PTT >= 0
  gpio_config_t btn = {
      .pin_bit_mask = 1ULL << BOARD_BTN_PTT,
      .mode = GPIO_MODE_INPUT,
      .pull_up_en = GPIO_PULLUP_ENABLE,
  };
  gpio_config(&btn);
#endif

  // One I2C bus shared by the codec and the camera's SCCB
  i2c_master_bus_handle_t i2c = NULL;
  i2c_master_bus_config_t bus_cfg = {
      .clk_source = I2C_CLK_SRC_DEFAULT,
      .i2c_port = BOARD_I2C_PORT,
      .sda_io_num = BOARD_I2C_SDA,
      .scl_io_num = BOARD_I2C_SCL,
      .glitch_ignore_cnt = 7,
      .flags.enable_internal_pullup = true,
  };
  ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &i2c));

  if (audio_init(i2c, LOCI_SPK_VOLUME) == ESP_OK) {
    audio_beep(523.25f, 90);
    audio_beep(659.25f, 90);
  } else {
    ESP_LOGE(TAG, "audio unavailable");
  }

  static loci_app_config_t cfg = {
      .fw_version = FW_VERSION,
      .board_name = BOARD_NAME,
      .button_gpio = BOARD_BTN_PTT,
      .wifi_ssid = WIFI_SSID,
      .wifi_pass = WIFI_PASS,
      .host_url = LOCI_HOST,
      .device_id = LOCI_DEVICE_ID,
      .device_token = LOCI_DEVICE_TOKEN,
      .detect = detect_local_boxes,  // cloud mode: the board's own sightings
      .console_extra = console_extra,
      .console_extra_help = "detect [n]",
  };
  cfg.camera_ok = camera_init(i2c, LOCI_JPEG_QUALITY) == ESP_OK;
  loci_app_run(&cfg);  // never returns
}
