// LOCI thin client for the Waveshare ESP32-P4-NANO.
//
//   camera (MIPI-CSI, hardware JPEG) ──► POST /api/esp32/frame ──► host runs YOLO + LOCI memory
//   speaker (ES8311 + NS4150B)       ◄── GET  /api/esp32/audio/next (16 kHz mono PCM16)
//   mic (ES8311) + BOOT button       ──► POST /api/esp32/voice (push-to-talk), answer plays back
//
// Everything heavy runs on the host; this file is the loop that moves pixels and sound.
// Serial console (115200): status | beep | audio | ask <question> | say <text> | fps <n> | vol <0-100>

#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "audio.h"
#include "board_p4_nano.h"
#include "camera.h"
#include "host_link.h"
#include "wifi_link.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Create main/secrets.h from main/secrets.h.example (WIFI_SSID, WIFI_PASS, LOCI_HOST)."
#endif

static const char *TAG = "loci";
static const char *FW_VERSION = "0.2.0-p4";

static bool s_cam_ok;
static uint32_t s_interval_ms = 1000 / LOCI_FPS;
static uint32_t s_seq, s_sent, s_failed;
static char s_last_labels[96];
static int16_t *s_ptt_buf;
static size_t s_ptt_cap_samples;

static bool button_down(void) {
#if BOARD_BTN_PTT >= 0
  return gpio_get_level(BOARD_BTN_PTT) == 0;
#else
  return false;
#endif
}

static void print_status(void) {
  ESP_LOGI(TAG, "fw %s | %s | cam %d %lux%lu | audio %d | wifi %d ip %s rssi %d", FW_VERSION, BOARD_NAME,
           s_cam_ok, (unsigned long)camera_width(), (unsigned long)camera_height(), audio_ready(),
           wifi_link_connected(), wifi_link_ip(), wifi_link_rssi());
  ESP_LOGI(TAG, "frames sent %lu failed %lu interval %lu ms | clips %lu | heap %u psram %u | labels: %s",
           (unsigned long)s_sent, (unsigned long)s_failed, (unsigned long)s_interval_ms,
           (unsigned long)host_audio_clips_played(), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), s_last_labels[0] ? s_last_labels : "-");
}

// ---- push-to-talk --------------------------------------------------------------------------
static void record_and_ask(void) {
  vTaskDelay(pdMS_TO_TICKS(30));  // debounce
  if (!button_down() || !audio_ready() || !s_ptt_buf) return;
  ESP_LOGI(TAG, "[ptt] recording... release the button to ask");
  audio_flush_mic();
  const size_t chunk = LOCI_AUDIO_RATE / 50;  // 20 ms
  size_t samples = 0;
  int64_t t0 = esp_timer_get_time();
  while (button_down() && samples + chunk <= s_ptt_cap_samples) {
    if (audio_record_pcm16(s_ptt_buf + samples, chunk) != ESP_OK) break;
    samples += chunk;
    if (esp_timer_get_time() - t0 > (int64_t)LOCI_PTT_MAX_S * 1000000) break;
  }
  while (button_down()) vTaskDelay(pdMS_TO_TICKS(10));
  float seconds = samples / (float)LOCI_AUDIO_RATE;
  ESP_LOGI(TAG, "[ptt] %.1f s captured", seconds);
  if (seconds < 0.3f) {
    audio_beep(220.0f, 80);
    return;
  }
  audio_beep(880.0f, 60);  // "got it"
  host_post_voice((const uint8_t *)s_ptt_buf, samples * sizeof(int16_t));
}

// ---- serial console ------------------------------------------------------------------------
static void console_task(void *arg) {
  char line[200];
  while (fgets(line, sizeof(line), stdin)) {
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
    if (!n) continue;
    if (!strcmp(line, "status")) {
      print_status();
    } else if (!strcmp(line, "beep")) {
      audio_beep(660.0f, 150);
    } else if (!strcmp(line, "audio")) {
      host_fetch_and_play_audio(2);
    } else if (!strncmp(line, "ask ", 4)) {
      host_ask(line + 4);
    } else if (!strncmp(line, "say ", 4)) {
      host_say(line + 4);
    } else if (!strncmp(line, "fps ", 4)) {
      int fps = atoi(line + 4);
      if (fps > 0 && fps <= 15) s_interval_ms = 1000 / fps;
      ESP_LOGI(TAG, "interval %lu ms", (unsigned long)s_interval_ms);
    } else if (!strncmp(line, "vol ", 4)) {
      audio_set_volume(atoi(line + 4));
    } else {
      ESP_LOGI(TAG, "commands: status | beep | audio | ask <question> | say <text> | fps <n> | vol <0-100>");
    }
  }
  vTaskDelete(NULL);
}

static void console_start(void) {
  // Blocking stdin over UART0 so fgets() works in its own task.
  uart_driver_install(UART_NUM_0, 512, 0, 0, NULL, 0);
  uart_vfs_dev_use_driver(UART_NUM_0);
  xTaskCreate(console_task, "console", 6144, NULL, 3, NULL);
}

// ---- main ----------------------------------------------------------------------------------
void app_main(void) {
  esp_chip_info_t chip;
  esp_chip_info(&chip);
  ESP_LOGI(TAG, "=== LOCI thin client v%s on %s (ESP32-P4 rev v%d.%d, %d cores) ===", FW_VERSION, BOARD_NAME,
           chip.revision / 100, chip.revision % 100, chip.cores);
  ESP_LOGI(TAG, "device id \"%s\", host %s, %d fps target, PSRAM %u B free", LOCI_DEVICE_ID, LOCI_HOST, LOCI_FPS,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    nvs_flash_init();
  }

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

  s_cam_ok = camera_init(i2c, LOCI_JPEG_QUALITY) == ESP_OK;
  if (!s_cam_ok) ESP_LOGE(TAG, "camera unavailable; questions still work from the console/button");

  s_ptt_cap_samples = (size_t)LOCI_AUDIO_RATE * LOCI_PTT_MAX_S;
  s_ptt_buf = heap_caps_malloc(s_ptt_cap_samples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  host_link_init(LOCI_HOST, LOCI_DEVICE_ID);
  console_start();

  if (wifi_link_start(WIFI_SSID, WIFI_PASS) == ESP_OK && wifi_link_wait(20000)) {
    audio_beep(783.99f, 120);  // third beep: online
  }
  print_status();

  int64_t last_frame_ms = 0;
  for (;;) {
    if (!wifi_link_connected()) {
      wifi_link_wait(2000);
      continue;
    }
    if (button_down()) {
      if (audio_ready()) {
        record_and_ask();
      } else {
        while (button_down()) vTaskDelay(pdMS_TO_TICKS(10));
        host_ask("what do you see");
      }
      continue;
    }
    int64_t now_ms = esp_timer_get_time() / 1000;
    if (s_cam_ok && now_ms - last_frame_ms >= (int64_t)s_interval_ms) {
      last_frame_ms = now_ms;
      uint8_t *jpeg = NULL;
      size_t len = 0;
      uint32_t w = 0, h = 0;
      if (camera_capture_jpeg(&jpeg, &len, &w, &h) == ESP_OK) {
        s_seq++;
        host_frame_result_t r;
        int64_t t0 = esp_timer_get_time();
        esp_err_t rc = host_post_frame(jpeg, len, s_seq, &r);
        camera_release();
        if (rc == ESP_OK) {
          s_sent++;
          snprintf(s_last_labels, sizeof(s_last_labels), "%s", r.labels);
          uint32_t own = 1000 / LOCI_FPS;
          s_interval_ms = r.next_ms > (int)own ? (uint32_t)r.next_ms : own;
          ESP_LOGI(TAG, "[frame] #%lu %lux%lu %u B in %lld ms -> %s%s%s", (unsigned long)s_seq, (unsigned long)w,
                   (unsigned long)h, (unsigned)len, (esp_timer_get_time() - t0) / 1000,
                   r.dropped ? "(dropped by host) " : "", r.labels[0] ? r.labels : "-",
                   r.audio_pending ? "  [audio waiting]" : "");
          if (r.audio_pending) host_fetch_and_play_audio(0);
        } else {
          s_failed++;
        }
      } else {
        ESP_LOGW(TAG, "capture failed");
        vTaskDelay(pdMS_TO_TICKS(200));
      }
    }
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}
