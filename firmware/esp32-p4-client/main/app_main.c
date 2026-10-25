// LOCI thin client for the Waveshare ESP32-P4-NANO.
//
//   camera (MIPI-CSI, hardware JPEG) ──► POST /api/esp32/frame ──► host runs YOLO + LOCI memory
//   speaker (ES8311 + NS4150B)       ◄── GET  /api/esp32/audio/next (16 kHz mono PCM16)
//   mic (ES8311) + BOOT button       ──► POST /api/esp32/voice (push-to-talk), answer plays back
//   microSD                          ──  frames kept while offline and uploaded later at their capture
//                                        time, a cached copy of the host's memory with a spoken summary
//                                        for offline answers, and a dataset of frames + detections
//
// Everything heavy runs on the host; this file is the loop that moves pixels and sound.
// Serial console (115200): status | sd | objects | offline | snap | beep | audio | ask <q> | say <t> | fps <n> | vol <n>

#include <stdio.h>
#include <stdlib.h>
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
#include "local_memory.h"
#include "storage.h"
#include "voice_local.h"
#include "wifi_link.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Create main/secrets.h from main/secrets.h.example (WIFI_SSID, WIFI_PASS, LOCI_HOST)."
#endif

static const char *TAG = "loci";
static const char *FW_VERSION = "0.4.0-p4";

static bool s_cam_ok;
static uint32_t s_interval_ms = 1000 / LOCI_FPS;
static uint32_t s_seq, s_sent, s_failed, s_backlog_sent;
#ifndef LOCI_DEVICE_TOKEN
#define LOCI_DEVICE_TOKEN ""  // older secrets.h without a token: LAN-only unless `token <value>` is used
#endif

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
  storage_info_t si = storage_info();
  ESP_LOGI(TAG, "fw %s | %s | cam %d %lux%lu | audio %d | wifi %d ip %s rssi %d | host %s %s%s", FW_VERSION,
           BOARD_NAME, s_cam_ok, (unsigned long)camera_width(), (unsigned long)camera_height(), audio_ready(),
           wifi_link_connected(), wifi_link_ip(), wifi_link_rssi(), host_link_base(),
           host_reachable() ? "reachable" : "OFFLINE", host_link_has_token() ? " (token)" : "");
  ESP_LOGI(TAG,
           "frames sent %lu failed %lu backlog sent %lu interval %lu ms | clips %lu | heap %u (retention %u) psram %u",
           (unsigned long)s_sent, (unsigned long)s_failed, (unsigned long)s_backlog_sent,
           (unsigned long)s_interval_ms, (unsigned long)host_audio_clips_played(),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_RETENTION),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  if (si.mounted) {
    ESP_LOGI(TAG, "card: %.1f GB free, backlog %d frames, dataset %d frames | labels: %s", si.free_bytes / 1e9,
             local_backlog_count(), local_dataset_count(), s_last_labels[0] ? s_last_labels : "-");
  } else {
    ESP_LOGI(TAG, "card: none%s | labels: %s", si.format_needed ? " (present, needs FAT32 format)" : "",
             s_last_labels[0] ? s_last_labels : "-");
  }
}

static void answer_offline(void) {
  if (!local_memory_play_offline()) audio_beep(220.0f, 250);  // nothing cached: a low tone
}

// Offline question: recognise the command on the board and answer from the card.
static void answer_offline_clip(const int16_t *pcm, size_t samples) {
  char label[32] = "";
  int cmd = voice_local_ready() ? voice_local_recognize(pcm, samples, label, sizeof(label)) : VOICE_CMD_NONE;
  if (cmd >= VOICE_CMD_FIRST_OBJECT && label[0]) {
    if (local_memory_play_object(label)) return;  // "Your cup was last seen ..."
  }
  answer_offline();  // the cached summary (or the fixed prompt)
}

// Keep the recogniser's object list in step with the cached memory.
static void refresh_offline_commands(void) {
  if (!voice_local_ready() || !storage_ready()) return;
  static char labels[12][32];
  int n = local_memory_labels(labels, 12);
  voice_local_set_labels(labels, n);
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
  local_voice_save(s_ptt_buf, samples, host_now_ms());
  if (!wifi_link_connected() || !host_reachable()) {
    ESP_LOGW(TAG, "[ptt] host unreachable: recognising on the board, answering from the card");
    answer_offline_clip(s_ptt_buf, samples);
    return;
  }
  host_post_voice((const uint8_t *)s_ptt_buf, samples * sizeof(int16_t));
}

// ---- frames --------------------------------------------------------------------------------
static void capture_to_card(void) {
  uint8_t *jpeg = NULL;
  size_t len = 0;
  if (camera_capture_jpeg(&jpeg, &len, NULL, NULL) != ESP_OK) return;
  local_backlog_push(jpeg, len, host_now_ms());
  camera_release();
}

static void upload_one_backlog_frame(void) {
  char path[96];
  int64_t captured = 0;
  if (!local_backlog_oldest(path, sizeof(path), &captured)) return;
  size_t len = 0;
  uint8_t *buf = local_read_file(path, &len);
  if (!buf) {
    local_backlog_remove(path);  // unreadable: drop it rather than loop on it
    return;
  }
  host_frame_result_t r;
  s_seq++;
  if (host_post_frame_ex(buf, len, s_seq, captured, true, &r) == ESP_OK) {
    local_backlog_remove(path);
    s_backlog_sent++;
    if ((s_backlog_sent % 10) == 1 || local_backlog_count() == 0)
      ESP_LOGI(TAG, "[backlog] uploaded %lu, %d left -> %s", (unsigned long)s_backlog_sent, local_backlog_count(),
               r.labels[0] ? r.labels : "-");
  }
  free(buf);
}

static void live_frame(void) {
  uint8_t *jpeg = NULL;
  size_t len = 0;
  uint32_t w = 0, h = 0;
  if (camera_capture_jpeg(&jpeg, &len, &w, &h) != ESP_OK) {
    ESP_LOGW(TAG, "capture failed");
    vTaskDelay(pdMS_TO_TICKS(200));
    return;
  }
  s_seq++;
  host_frame_result_t r;
  int64_t t0 = esp_timer_get_time();
  esp_err_t rc = host_post_frame(jpeg, len, s_seq, &r);
  if (rc == ESP_OK) {
    s_sent++;
    local_dataset_maybe_save(jpeg, len, host_now_ms(), host_last_reply());
  } else {
    s_failed++;
    local_backlog_push(jpeg, len, host_now_ms());  // keep it; it goes up when the host is back
  }
  camera_release();
  if (rc != ESP_OK) return;
  snprintf(s_last_labels, sizeof(s_last_labels), "%s", r.labels);
  uint32_t own = 1000 / LOCI_FPS;
  s_interval_ms = r.next_ms > (int)own ? (uint32_t)r.next_ms : own;
  ESP_LOGI(TAG, "[frame] #%lu %lux%lu %u B in %lld ms -> %s%s%s", (unsigned long)s_seq, (unsigned long)w,
           (unsigned long)h, (unsigned)len, (esp_timer_get_time() - t0) / 1000,
           r.dropped ? "(dropped by host) " : "", r.labels[0] ? r.labels : "-",
           r.audio_pending ? "  [audio waiting]" : "");
  static char last_rev[24];
  static bool phrases_checked;
  local_memory_refresh(r.memory_rev);
  local_memory_ensure_prompt();
  if (r.memory_rev[0] && strcmp(last_rev, r.memory_rev) != 0 && storage_ready()) {
    snprintf(last_rev, sizeof(last_rev), "%s", r.memory_rev);
    refresh_offline_commands();
  }
  if (!phrases_checked && storage_ready()) {  // once per boot: fill in per-object answers the card lacks
    phrases_checked = true;
    local_memory_cache_object_phrases();
  }
  if (r.audio_pending) host_fetch_and_play_audio(0);
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
    } else if (!strcmp(line, "sd")) {
      storage_info_t si = storage_info();
      if (si.mounted) {
        ESP_LOGI(TAG, "[sd] \"%s\" %.1f GB, %.1f GB free, %d-bit @ %d kHz, dir %s", si.name, si.capacity_bytes / 1e9,
                 si.free_bytes / 1e9, si.bus_width, si.speed_khz, STORAGE_DIR);
      } else {
        ESP_LOGW(TAG, "[sd] not mounted%s", si.format_needed ? " (card present but not FAT: needs formatting)" : "");
      }
    } else if (!strcmp(line, "sdformat yes")) {  // destructive: formats a non-FAT card as FAT32
      if (storage_ready()) {
        ESP_LOGW(TAG, "[sd] already mounted; nothing to format");
      } else if (storage_init(true) == ESP_OK) {
        local_memory_init();
        ESP_LOGI(TAG, "[sd] formatted and mounted");
      } else {
        ESP_LOGE(TAG, "[sd] format failed");
      }
    } else if (!strcmp(line, "objects")) {
      local_memory_print();
    } else if (!strcmp(line, "cache")) {  // fetch the per-object spoken answers the card lacks
      local_memory_cache_object_phrases();
      refresh_offline_commands();
      ESP_LOGI(TAG, "[cache] done");
    } else if (!strncmp(line, "host ", 5)) {  // runtime host URL, kept in NVS; "host default" restores secrets.h
      const char *arg = line + 5;
      while (*arg == ' ') arg++;
      if (host_link_set_host(!strcmp(arg, "default") ? NULL : arg)) ESP_LOGI(TAG, "host is now %s", host_link_base());
    } else if (!strncmp(line, "token ", 6)) {  // device token for the host, kept in NVS and never printed
      const char *arg = line + 6;
      while (*arg == ' ') arg++;
      host_link_set_token(!strcmp(arg, "clear") ? NULL : arg);
      ESP_LOGI(TAG, "device token %s", host_link_has_token() ? "stored" : "cleared");
    } else if (!strcmp(line, "offline")) {
      answer_offline();
    } else if (!strcmp(line, "recog")) {  // self-test: record 4 s from the mic, recognise on the board
      if (!audio_ready() || !s_ptt_buf) {
        ESP_LOGW(TAG, "[recog] no audio");
      } else {
        ESP_LOGI(TAG, "[recog] speak now (4 s)...");
        audio_beep(880.0f, 60);
        audio_flush_mic();
        size_t samples = (size_t)LOCI_AUDIO_RATE * 4;
        if (samples > s_ptt_cap_samples) samples = s_ptt_cap_samples;
        audio_record_pcm16(s_ptt_buf, samples);
        int peak = 0;
        int64_t sum = 0;
        for (size_t i = 0; i < samples; i++) {
          int v = s_ptt_buf[i] < 0 ? -s_ptt_buf[i] : s_ptt_buf[i];
          sum += v;
          if (v > peak) peak = v;
        }
        ESP_LOGI(TAG, "[recog] %.1f s recorded: peak %d, mean level %d (of 32767)", samples / (float)LOCI_AUDIO_RATE,
                 peak, (int)(sum / (int64_t)samples));
        char label[32] = "";
        int cmd = voice_local_recognize(s_ptt_buf, samples, label, sizeof(label));
        ESP_LOGI(TAG, "[recog] result %d %s", cmd, label);
        if (cmd >= VOICE_CMD_FIRST_OBJECT && label[0] && local_memory_play_object(label)) {
          ESP_LOGI(TAG, "[recog] played the cached answer for %s", label);
        }
      }
    } else if (!strcmp(line, "beep")) {
      audio_beep(660.0f, 150);
    } else if (!strcmp(line, "snap")) {  // capture + encode one frame without the host (self-test)
      uint8_t *jpeg = NULL;
      size_t len = 0;
      uint32_t w = 0, h = 0;
      int64_t t0 = esp_timer_get_time();
      if (s_cam_ok && camera_capture_jpeg(&jpeg, &len, &w, &h) == ESP_OK) {
        bool soi = len > 4 && jpeg[0] == 0xFF && jpeg[1] == 0xD8;
        ESP_LOGI(TAG, "[snap] %lux%lu -> %u B JPEG in %lld ms (SOI %s)", (unsigned long)w, (unsigned long)h,
                 (unsigned)len, (esp_timer_get_time() - t0) / 1000, soi ? "ok" : "MISSING");
        camera_release();
      } else {
        ESP_LOGW(TAG, "[snap] capture failed (camera %s)", s_cam_ok ? "ok" : "unavailable");
      }
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
      ESP_LOGI(TAG, "commands: status | sd | sdformat yes | objects | cache | offline | recog | snap | beep | audio "
                    "| ask <question> | say <text> | fps <n> | vol <0-100> | host <url>|default | token <value>|clear");
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
  ESP_LOGI(TAG, "device id \"%s\", host %s, %d fps target, PSRAM %u B free, internal %u B (retention %u B)",
           LOCI_DEVICE_ID, LOCI_HOST, LOCI_FPS, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_RETENTION));

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

  host_link_init(LOCI_HOST, LOCI_DEVICE_ID, LOCI_DEVICE_TOKEN);
  console_start();

  if (wifi_link_start(WIFI_SSID, WIFI_PASS) == ESP_OK && wifi_link_wait(20000)) {
    audio_beep(783.99f, 120);  // third beep: online
  }
  // After the Wi-Fi link: ESP-Hosted has set up the SDMMC controller the card shares (see storage.c)
  storage_init(LOCI_SD_FORMAT_IF_NEEDED != 0);
  local_memory_init();
  if (voice_local_init()) refresh_offline_commands();
  print_status();

  int64_t last_frame_ms = 0, last_backlog_ms = 0, last_probe_ms = 0;
  for (;;) {
    int64_t now_ms = esp_timer_get_time() / 1000;

    if (!wifi_link_connected()) {
      // No network: keep remembering to the card at a slow rate, answer the button from the cache.
      if (button_down()) {
        while (button_down()) vTaskDelay(pdMS_TO_TICKS(10));
        answer_offline();
        continue;
      }
      if (s_cam_ok && storage_ready() && now_ms - last_frame_ms >= 1000 / LOCI_OFFLINE_FPS) {
        last_frame_ms = now_ms;
        capture_to_card();
      }
      wifi_link_wait(200);
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

    bool online = host_reachable();
    host_set_frame_timeout_ms(online ? 8000 : 2500);

    // Drain the backlog between live frames (one file per pass keeps the live stream smooth).
    if (online && local_backlog_count() > 0 && now_ms - last_backlog_ms >= 150) {
      last_backlog_ms = now_ms;
      upload_one_backlog_frame();
    }

    if (s_cam_ok && now_ms - last_frame_ms >= (int64_t)s_interval_ms) {
      if (!online && storage_ready() && now_ms - last_probe_ms < 5000) {
        // Host down but Wi-Fi up: record to the card, probe the host every 5 s with a live frame.
        if (now_ms - last_frame_ms >= 1000 / LOCI_OFFLINE_FPS) {
          last_frame_ms = now_ms;
          capture_to_card();
        }
      } else {
        last_frame_ms = now_ms;
        if (!online) last_probe_ms = now_ms;
        live_frame();
      }
    }
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}
