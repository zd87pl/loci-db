#include "local_memory.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "audio.h"
#include "board_p4_nano.h"
#include "host_link.h"
#include "storage.h"

static const char *TAG = "local";

static char s_dir_backlog[64];
static char s_dir_frames[64];
static char s_dir_voice[64];
static int s_backlog_count = -1;  // -1 = not scanned yet
static int s_dataset_count;
static uint32_t s_live_frames;
static char s_cached_rev[24];
static int64_t s_last_refresh_attempt_ms = -60000;
static bool s_prompt_checked;

#define BACKLOG_DIR "backlog"
#define FRAMES_DIR "frames"
#define VOICE_DIR "voice"
#define MEMORY_JSON STORAGE_DIR "/memory.json"
#define MEMORY_PCM STORAGE_DIR "/memory.pcm"
#define MEMORY_REV STORAGE_DIR "/memory.rev"
#define OFFLINE_PCM STORAGE_DIR "/offline.pcm"
#define OFFLINE_PROMPT "I can't reach the host right now."

static int count_files(const char *dir) {
  DIR *d = opendir(dir);
  if (!d) return 0;
  int n = 0;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (e->d_name[0] != '.') n++;
  }
  closedir(d);
  return n;
}

static bool enough_space(void) {
  static int calls;
  static bool ok = true;
  if ((calls++ % 25) == 0) {
    storage_info_t si = storage_info();
    ok = si.mounted && si.free_bytes > (uint64_t)LOCI_SD_MIN_FREE_MB * 1024 * 1024;
    if (!ok) ESP_LOGW(TAG, "card nearly full (%.1f GB free): not saving more", si.free_bytes / 1e9);
  }
  return ok;
}

static int64_t stamp(int64_t ms) {
  // Unix ms when the host clock is known, else monotonic ms (sorts first, host uses arrival time)
  if (ms > 0) return ms;
  int64_t now = host_now_ms();
  return now > 0 ? now : esp_timer_get_time() / 1000;
}

void local_memory_init(void) {
  if (!storage_ready()) return;
  storage_dir(BACKLOG_DIR, s_dir_backlog, sizeof(s_dir_backlog));
  storage_dir(FRAMES_DIR, s_dir_frames, sizeof(s_dir_frames));
  storage_dir(VOICE_DIR, s_dir_voice, sizeof(s_dir_voice));
  s_backlog_count = count_files(s_dir_backlog);
  s_dataset_count = count_files(s_dir_frames) / 2;
  FILE *f = fopen(MEMORY_REV, "r");
  if (f) {
    if (!fgets(s_cached_rev, sizeof(s_cached_rev), f)) s_cached_rev[0] = 0;
    s_cached_rev[strcspn(s_cached_rev, "\r\n")] = 0;
    fclose(f);
  }
  struct stat st;
  ESP_LOGI(TAG, "card: %d backlog frames, %d dataset frames, cached memory rev \"%s\"%s", s_backlog_count,
           s_dataset_count, s_cached_rev, stat(MEMORY_PCM, &st) == 0 ? ", spoken summary present" : "");
}

// -- backlog ------------------------------------------------------------------------------------

bool local_backlog_push(const uint8_t *jpeg, size_t len, int64_t captured_ms) {
  if (!storage_ready() || s_backlog_count >= LOCI_BACKLOG_MAX_FILES || !enough_space()) return false;
  char path[96];
  snprintf(path, sizeof(path), "%s/%lld.jpg", s_dir_backlog, (long long)stamp(captured_ms));
  if (storage_write_file(path, jpeg, len) != (int)len) return false;
  s_backlog_count++;
  if ((s_backlog_count % 20) == 1) ESP_LOGI(TAG, "backlog: %d frames on the card", s_backlog_count);
  return true;
}

int local_backlog_count(void) { return s_backlog_count < 0 ? 0 : s_backlog_count; }

bool local_backlog_oldest(char *path, size_t cap, int64_t *captured_ms) {
  if (!storage_ready() || local_backlog_count() == 0) return false;
  DIR *d = opendir(s_dir_backlog);
  if (!d) return false;
  long long best = -1;
  char best_name[64] = "";
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (e->d_name[0] == '.') continue;
    long long v = atoll(e->d_name);
    if (v > 0 && (best < 0 || v < best)) {
      best = v;
      snprintf(best_name, sizeof(best_name), "%.63s", e->d_name);
    }
  }
  closedir(d);
  if (best < 0) {
    s_backlog_count = 0;
    return false;
  }
  snprintf(path, cap, "%s/%s", s_dir_backlog, best_name);
  if (captured_ms) *captured_ms = best > 1000000000000LL ? best : 0;  // monotonic stamps are not Unix time
  return true;
}

void local_backlog_remove(const char *path) {
  if (unlink(path) == 0 && s_backlog_count > 0) s_backlog_count--;
}

uint8_t *local_read_file(const char *path, size_t *len_out) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0) {
    fclose(f);
    return NULL;
  }
  uint8_t *buf = heap_caps_malloc((size_t)n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (got != (size_t)n) {
    free(buf);
    return NULL;
  }
  *len_out = got;
  return buf;
}

// -- dataset / voice ----------------------------------------------------------------------------

void local_dataset_maybe_save(const uint8_t *jpeg, size_t len, int64_t ms, const char *host_reply_json) {
  if (!storage_ready() || LOCI_SD_KEEP_EVERY <= 0) return;
  if ((++s_live_frames % LOCI_SD_KEEP_EVERY) != 0 || !enough_space()) return;
  long long t = (long long)stamp(ms);
  char path[96];
  snprintf(path, sizeof(path), "%s/%lld.jpg", s_dir_frames, t);
  if (storage_write_file(path, jpeg, len) != (int)len) return;
  snprintf(path, sizeof(path), "%s/%lld.json", s_dir_frames, t);
  storage_write_file(path, host_reply_json, strlen(host_reply_json));
  s_dataset_count++;
}

int local_dataset_count(void) { return s_dataset_count; }

void local_voice_save(const int16_t *pcm, size_t samples, int64_t ms) {
  if (!storage_ready() || !enough_space()) return;
  char path[96];
  snprintf(path, sizeof(path), "%s/%lld.wav", s_dir_voice, (long long)stamp(ms));
  FILE *f = fopen(path, "wb");
  if (!f) return;
  uint32_t data_bytes = samples * 2, rate = LOCI_AUDIO_RATE, byte_rate = rate * 2, chunk = 36 + data_bytes, fmt_len = 16;
  uint16_t fmt_tag = 1, channels = 1, block_align = 2, bits = 16;
  fwrite("RIFF", 1, 4, f);
  fwrite(&chunk, 4, 1, f);
  fwrite("WAVEfmt ", 1, 8, f);
  fwrite(&fmt_len, 4, 1, f);
  fwrite(&fmt_tag, 2, 1, f);
  fwrite(&channels, 2, 1, f);
  fwrite(&rate, 4, 1, f);
  fwrite(&byte_rate, 4, 1, f);
  fwrite(&block_align, 2, 1, f);
  fwrite(&bits, 2, 1, f);
  fwrite("data", 1, 4, f);
  fwrite(&data_bytes, 4, 1, f);
  fwrite(pcm, 1, data_bytes, f);
  fclose(f);
}

// -- cached memory + offline speech ---------------------------------------------------------------

static void save_rev(const char *rev) {
  storage_write_file(MEMORY_REV, rev, strlen(rev));
  snprintf(s_cached_rev, sizeof(s_cached_rev), "%s", rev);
}

void local_memory_refresh(const char *rev) {
  if (!storage_ready() || !rev || !rev[0] || !strcmp(rev, s_cached_rev)) return;
  int64_t now = esp_timer_get_time() / 1000;
  if (now - s_last_refresh_attempt_ms < 10000) return;  // the host's memory can change every frame
  s_last_refresh_attempt_ms = now;
  size_t n1 = 0, n2 = 0;
  char q[96];
  snprintf(q, sizeof(q), "/api/esp32/memory?device=%s", host_device_id());
  if (host_fetch_file(q, MEMORY_JSON, &n1) != ESP_OK) return;
  snprintf(q, sizeof(q), "/api/esp32/memory.pcm?device=%s", host_device_id());
  if (host_fetch_file(q, MEMORY_PCM, &n2) != ESP_OK) return;
  save_rev(rev);
  ESP_LOGI(TAG, "memory cache refreshed (rev %s): %u B json, %.1f s of speech", rev, (unsigned)n1,
           n2 / (2.0f * LOCI_AUDIO_RATE));
}

void local_memory_ensure_prompt(void) {
  if (!storage_ready() || s_prompt_checked) return;
  struct stat st;
  if (stat(OFFLINE_PCM, &st) == 0 && st.st_size > 1000) {
    s_prompt_checked = true;
    return;
  }
  size_t n = 0;
  if (host_fetch_file("/api/esp32/phrase.pcm?text=" OFFLINE_PROMPT_URL, OFFLINE_PCM, &n) == ESP_OK) {
    s_prompt_checked = true;
    ESP_LOGI(TAG, "offline prompt cached (%.1f s)", n / (2.0f * LOCI_AUDIO_RATE));
  }
}

static bool play_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  static int16_t buf[2048];
  size_t total = 0, n;
  if (strstr(path, ".wav")) fseek(f, 44, SEEK_SET);
  while ((n = fread(buf, 2, sizeof(buf) / sizeof(buf[0]), f)) > 0) {
    audio_play_pcm16(buf, n);
    total += n;
  }
  fclose(f);
  ESP_LOGI(TAG, "played %s (%.1f s) from the card", path, total / (float)LOCI_AUDIO_RATE);
  return total > 0;
}

bool local_memory_play_offline(void) {
  if (!storage_ready()) return false;
  struct stat st;
  if (stat(MEMORY_PCM, &st) == 0 && st.st_size > 1000) return play_file(MEMORY_PCM);
  if (stat(OFFLINE_PCM, &st) == 0 && st.st_size > 1000) return play_file(OFFLINE_PCM);
  return false;
}

void local_memory_print(void) {
  if (!storage_ready()) {
    ESP_LOGW(TAG, "card not mounted");
    return;
  }
  size_t n = 0;
  uint8_t *json = local_read_file(MEMORY_JSON, &n);
  ESP_LOGI(TAG, "backlog %d, dataset %d, cached rev \"%s\"", local_backlog_count(), s_dataset_count, s_cached_rev);
  if (json) {
    printf("%.*s\n", (int)(n > 1500 ? 1500 : n), (const char *)json);
    free(json);
  } else {
    ESP_LOGI(TAG, "no cached memory.json yet");
  }
}
