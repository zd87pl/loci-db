// LOCI thin client for ESP32-S3 camera boards: the Meshnology W11 ("ESP32S3 Mini Module" with its
// camera/mic/microSD expansion board) and, with the same pins, the Seeed XIAO ESP32S3 Sense.
//
//   camera (DVP, JPEG from the sensor) ──► POST /api/esp32/frame ──► host runs YOLO + LOCI memory
//   PDM mic + BOOT button              ──► POST /api/esp32/voice (push-to-talk); answer in the log
//                                          and on the host page (no speaker on this board)
//   microSD (SPI)                      ──  store-and-forward, cached memory, dataset (loci_app.h)
//   QMI8658C IMU                       ──  `imu` console command (motion gating is future work)
//   ESP-DL COCO YOLO11n (320, int8)    ──  `detect` console command: detection on the board itself
//
// This file brings up the S3-specific hardware; the loop, the host link and the microSD logic are
// shared with the P4 client in firmware/common/loci_client.

#include <dirent.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "audio.h"
#include "board_w11.h"
#include "camera.h"
#include "detect_local.h"
#include "local_memory.h"
#include "loci_app.h"
#include "storage.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Create main/secrets.h from main/secrets.h.example (WIFI_SSID, WIFI_PASS, LOCI_HOST)."
#endif

#ifndef LOCI_DEVICE_TOKEN
#define LOCI_DEVICE_TOKEN ""
#endif

#define FW_VERSION "0.5.0-s3"

static const char *TAG = "loci";

// ---- QMI8658C IMU (I2C 0x6B or 0x6A): accelerometer + temperature on demand ----------------------
#define QMI_WHO_AM_I 0x00
#define QMI_CTRL1 0x02
#define QMI_CTRL2 0x03
#define QMI_CTRL7 0x08
#define QMI_TEMP_L 0x33
static i2c_master_dev_handle_t s_imu;

static bool imu_init(void) {
  i2c_master_bus_handle_t bus = NULL;
  i2c_master_bus_config_t bus_cfg = {
      .clk_source = I2C_CLK_SRC_DEFAULT,
      .i2c_port = I2C_NUM_0,  // esp32-camera's SCCB takes port 1 (CONFIG_SCCB_HARDWARE_I2C_PORT1)
      .sda_io_num = BOARD_IMU_SDA,
      .scl_io_num = BOARD_IMU_SCL,
      .glitch_ignore_cnt = 7,
      .flags.enable_internal_pullup = true,
  };
  if (i2c_new_master_bus(&bus_cfg, &bus) != ESP_OK) return false;
  const uint8_t addrs[] = {0x6B, 0x6A};
  for (size_t i = 0; i < sizeof(addrs); i++) {
    if (i2c_master_probe(bus, addrs[i], 50) != ESP_OK) continue;
    i2c_device_config_t dev = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = addrs[i], .scl_speed_hz = 400000};
    if (i2c_master_bus_add_device(bus, &dev, &s_imu) != ESP_OK) return false;
    uint8_t reg = QMI_WHO_AM_I, who = 0;
    i2c_master_transmit_receive(s_imu, &reg, 1, &who, 1, 50);
    // auto-increment addresses, accelerometer +-4 g at ~250 Hz, accelerometer on
    const uint8_t init[][2] = {{QMI_CTRL1, 0x40}, {QMI_CTRL2, 0x15}, {QMI_CTRL7, 0x01}};
    for (size_t k = 0; k < sizeof(init) / sizeof(init[0]); k++) i2c_master_transmit(s_imu, init[k], 2, 50);
    ESP_LOGI(TAG, "QMI8658 IMU at 0x%02x (WHO_AM_I 0x%02x)", addrs[i], who);
    return true;
  }
  ESP_LOGW(TAG, "no QMI8658 on I2C SDA %d SCL %d", BOARD_IMU_SDA, BOARD_IMU_SCL);
  return false;
}

static void imu_print(void) {
  if (!s_imu) {
    ESP_LOGW(TAG, "[imu] not found");
    return;
  }
  uint8_t reg = QMI_TEMP_L, raw[8] = {0};
  if (i2c_master_transmit_receive(s_imu, &reg, 1, raw, sizeof(raw), 50) != ESP_OK) {
    ESP_LOGW(TAG, "[imu] read failed");
    return;
  }
  float temp = (int16_t)(raw[1] << 8 | raw[0]) / 256.0f;
  float ax = (int16_t)(raw[3] << 8 | raw[2]) / 8192.0f;
  float ay = (int16_t)(raw[5] << 8 | raw[4]) / 8192.0f;
  float az = (int16_t)(raw[7] << 8 | raw[6]) / 8192.0f;
  ESP_LOGI(TAG, "[imu] accel x %.2f y %.2f z %.2f g, %.1f C", (double)ax, (double)ay, (double)az, (double)temp);
}

// ---- on-board detection benchmark -----------------------------------------------------------
// Runs the ESP-DL detector on `runs` live frames (decoded at a quarter size: 1600x1200 -> 400x300).
static void run_detect(int runs) {
  int64_t t0 = esp_timer_get_time();
  if (!detect_local_init()) return;
  int load_ms = (int)((esp_timer_get_time() - t0) / 1000);
  int n = 0, dec_sum = 0, inf_sum = 0, inf_min = 1 << 30, inf_max = 0;
  for (int i = 0; i < runs; i++) {
    camera_fb_t *fb = esp_camera_fb_get();  // its own buffer: the upload loop may hold the other one
    if (!fb) continue;
    detect_local_result_t r;
    bool ok = fb->format == PIXFORMAT_JPEG && detect_local_run(fb->buf, fb->len, &r);
    esp_camera_fb_return(fb);
    if (!ok) continue;
    n++;
    dec_sum += r.decode_ms;
    inf_sum += r.infer_ms;
    if (r.infer_ms < inf_min) inf_min = r.infer_ms;
    if (r.infer_ms > inf_max) inf_max = r.infer_ms;
    ESP_LOGI(TAG, "[detect] %dx%d: decode %d ms, detect %d ms -> %s", r.width, r.height, r.decode_ms, r.infer_ms,
             r.count ? r.labels : "-");
  }
  if (n) {
    ESP_LOGI(TAG, "[detect] %d runs: decode avg %d ms, detect avg %d ms (min %d, max %d); model load %d ms", n,
             dec_sum / n, inf_sum / n, inf_min, inf_max, load_ms);
  }
}

// The same detector on frames from the card's dataset (every 100th, so they span the recording),
// printed next to the labels the host's YOLOv8n gave for that frame (the .json side-car).
static void run_detect_sd(int runs) {
  char dir[64];
  if (!storage_ready() || !detect_local_init()) return;
  storage_dir("frames", dir, sizeof(dir));
  DIR *d = opendir(dir);
  if (!d) return;
  int seen = 0, n = 0, inf_sum = 0, dec_sum = 0;
  struct dirent *e;
  while (n < runs && (e = readdir(d)) != NULL) {
    size_t len = strlen(e->d_name);
    if (len < 5 || strcmp(e->d_name + len - 4, ".jpg") != 0 || (seen++ % 100) != 0) continue;
    char path[128];
    snprintf(path, sizeof(path), "%s/%.60s", dir, e->d_name);
    size_t jpeg_len = 0;
    uint8_t *jpeg = local_read_file(path, &jpeg_len);
    if (!jpeg) continue;
    detect_local_result_t r;
    bool ok = detect_local_run(jpeg, jpeg_len, &r);
    free(jpeg);
    if (!ok) continue;
    n++;
    inf_sum += r.infer_ms;
    dec_sum += r.decode_ms;
    char host_labels[160] = "";
    snprintf(path + strlen(path) - 4, 6, ".json");
    uint8_t *json = local_read_file(path, &jpeg_len);
    if (json) {
      const char *p = strstr((const char *)json, "\"labels\":[");
      const char *q = p ? strchr(p, ']') : NULL;
      if (p && q) snprintf(host_labels, sizeof(host_labels), "%.*s", (int)(q - p - 10), p + 10);
      free(json);
    }
    ESP_LOGI(TAG, "[detect sd] %s %dx%d: decode %d ms, detect %d ms | board: %s | host: %s", e->d_name, r.width,
             r.height, r.decode_ms, r.infer_ms, r.count ? r.labels : "-", host_labels[0] ? host_labels : "-");
  }
  closedir(d);
  if (n) ESP_LOGI(TAG, "[detect sd] %d frames: decode avg %d ms, detect avg %d ms", n, dec_sum / n, inf_sum / n);
}

// ---- board console commands -----------------------------------------------------------------
static bool console_extra(const char *line) {
  if (!strncmp(line, "detect sd", 9)) {
    int runs = line[9] ? atoi(line + 10) : 5;
    run_detect_sd(runs > 0 && runs <= 50 ? runs : 5);
    return true;
  }
  if (!strcmp(line, "detect") || !strncmp(line, "detect ", 7)) {
    int runs = line[6] ? atoi(line + 7) : 1;
    run_detect(runs > 0 && runs <= 50 ? runs : 1);
    return true;
  }
  if (!strcmp(line, "imu")) {
    imu_print();
    return true;
  }
  if (!strncmp(line, "res ", 4)) {
    if (camera_set_frame_size(line + 4)) {
      ESP_LOGI(TAG, "[res] camera now %s", line + 4);
    } else {
      ESP_LOGW(TAG, "[res] unknown size or sensor refused: qvga vga svga xga hd sxga uxga fhd qxga");
    }
    return true;
  }
  if (!strcmp(line, "af") || !strcmp(line, "af now")) {
    camera_af_report(line[2] != 0);
    return true;
  }
  if (!strcmp(line, "mic")) {  // two seconds of microphone level, no recognition
    static int16_t buf[LOCI_AUDIO_RATE / 10];
    int peak_max = 0, mean_sum = 0;
    audio_flush_mic();
    for (int i = 0; i < 20; i++) {
      if (audio_record_pcm16(buf, sizeof(buf) / sizeof(buf[0])) != ESP_OK) break;
      int peak, mean;
      audio_last_levels(&peak, &mean);
      if (peak > peak_max) peak_max = peak;
      mean_sum += mean;
    }
    ESP_LOGI(TAG, "[mic] 2 s: peak %d, mean level %d (of 32767)", peak_max, mean_sum / 20);
    return true;
  }
  return false;
}

void app_main(void) {
  esp_chip_info_t chip;
  esp_chip_info(&chip);
  uint32_t flash_bytes = 0;
  esp_flash_get_size(NULL, &flash_bytes);
  ESP_LOGI(TAG, "=== LOCI thin client v%s on %s (ESP32-S3 rev v%d.%d, %d cores, %lu MB flash) ===", FW_VERSION,
           BOARD_NAME, chip.revision / 100, chip.revision % 100, chip.cores, (unsigned long)(flash_bytes >> 20));
  ESP_LOGI(TAG, "device id \"%s\", host %s, %d fps target, PSRAM %u B free, internal %u B", LOCI_DEVICE_ID,
           LOCI_HOST, LOCI_FPS, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

#if BOARD_BTN_PTT >= 0
  gpio_config_t btn = {
      .pin_bit_mask = 1ULL << BOARD_BTN_PTT,
      .mode = GPIO_MODE_INPUT,
      .pull_up_en = GPIO_PULLUP_ENABLE,
  };
  gpio_config(&btn);
#endif

  if (audio_init() != ESP_OK) ESP_LOGE(TAG, "microphone unavailable");
  imu_init();

  static loci_app_config_t cfg = {
      .fw_version = FW_VERSION,
      .board_name = BOARD_NAME,
      .button_gpio = BOARD_BTN_PTT,
      .wifi_ssid = WIFI_SSID,
      .wifi_pass = WIFI_PASS,
      .host_url = LOCI_HOST,
      .device_id = LOCI_DEVICE_ID,
      .device_token = LOCI_DEVICE_TOKEN,
      .console_extra = console_extra,
      .console_extra_help = "detect [n] | detect sd [n] | imu | mic | af [now] | res <qvga|vga|svga|xga|hd|sxga|uxga|fhd|qxga>",
  };
  cfg.camera_ok = camera_init(LOCI_JPEG_QUALITY, LOCI_FRAME_SIZE) == ESP_OK;
  loci_app_run(&cfg);  // never returns
}
