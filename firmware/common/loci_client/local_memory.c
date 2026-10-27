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

#include "loci_config.h"
#include "loci_hw.h"
#include "host_link.h"
#include "storage.h"

static const char *TAG = "local";
static bool play_file(const char *path);
#define PLAY_CHUNK 2048  // samples per read while playing a cached clip

static char s_dir_backlog[64];
static char s_dir_backlog_bad[64];
static char s_dir_frames[64];   // legacy flat dataset folder (read and counted, no longer written)
static char s_dir_dataset[64];  // dataset/<NNNN>/, DATASET_PER_DIR frames per folder
static char s_dir_voice[64];
static int s_backlog_count = -1;  // -1 = not scanned yet
static int s_dataset_count;  // legacy + sharded
static int s_dataset_new;    // frames in dataset/<NNNN>/ (decides the next folder)
static uint32_t s_live_frames;
static char s_cached_rev[24];
static int64_t s_last_refresh_attempt_ms = -60000;
static bool s_prompt_checked;

#define BACKLOG_DIR "backlog"
#define BACKLOG_BAD_DIR "backlog_bad"
#define FRAMES_DIR "frames"
// FatFs makes up a unique 8.3 alias for every long file name, and our names (<ms>.jpg) all share
// their first digits, so each create scans the whole directory several times. With ~6,600 entries
// in one folder that took ~18 s per saved frame on the W11 (1-bit SPI) and stalled the frame loop;
// with 250 frames per folder a save (JPEG + side-car) takes about 0.6 s there.
#define DATASET_DIR "dataset"
#define DATASET_PER_DIR 250
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
  storage_dir(BACKLOG_BAD_DIR, s_dir_backlog_bad, sizeof(s_dir_backlog_bad));
  storage_dir(FRAMES_DIR, s_dir_frames, sizeof(s_dir_frames));
  storage_dir(DATASET_DIR, s_dir_dataset, sizeof(s_dir_dataset));
  storage_dir(VOICE_DIR, s_dir_voice, sizeof(s_dir_voice));
  s_backlog_count = count_files(s_dir_backlog);
  s_dataset_new = 0;
  DIR *d = opendir(s_dir_dataset);
  if (d) {
    struct dirent *e;
    char sub[96];
    while ((e = readdir(d)) != NULL) {
      if (e->d_name[0] == '.') continue;
      snprintf(sub, sizeof(sub), "%s/%.24s", s_dir_dataset, e->d_name);
      s_dataset_new += count_files(sub) / 2;
    }
    closedir(d);
  }
  s_dataset_count = count_files(s_dir_frames) / 2 + s_dataset_new;
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

void local_backlog_quarantine(const char *path, const char *why) {
  const char *name = strrchr(path, '/');
  name = name ? name + 1 : path;
  char dest[112];
  snprintf(dest, sizeof(dest), "%s/%.40s", s_dir_backlog_bad, name);
  unlink(dest);  // FatFs rename does not replace an existing file
  if (s_dir_backlog_bad[0] && rename(path, dest) == 0) {
    if (s_backlog_count > 0) s_backlog_count--;
    ESP_LOGW(TAG, "backlog: %s %s, moved to %s/", name, why, BACKLOG_BAD_DIR);
  } else {
    local_backlog_remove(path);
    ESP_LOGW(TAG, "backlog: %s %s, deleted", name, why);
  }
}

int local_backlog_requeue(void) {
  if (!storage_ready() || !s_dir_backlog_bad[0]) return 0;
  DIR *d = opendir(s_dir_backlog_bad);
  if (!d) return 0;
  int n = 0;
  char from[112], to[112];
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (e->d_name[0] == '.') continue;
    snprintf(from, sizeof(from), "%s/%.40s", s_dir_backlog_bad, e->d_name);
    snprintf(to, sizeof(to), "%s/%.40s", s_dir_backlog, e->d_name);
    if (rename(from, to) == 0) n++;
  }
  closedir(d);
  s_backlog_count = count_files(s_dir_backlog);
  return n;
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
  // one spare byte: callers search JSON files with strstr, so the buffer is always NUL-terminated
  uint8_t *buf = heap_caps_malloc((size_t)n + 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
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
  buf[got] = 0;
  *len_out = got;
  return buf;
}

// -- dataset / voice ----------------------------------------------------------------------------

void local_dataset_maybe_save(const uint8_t *jpeg, size_t len, int64_t ms, const char *host_reply_json) {
  if (!storage_ready() || LOCI_SD_KEEP_EVERY <= 0) return;
  if ((++s_live_frames % LOCI_SD_KEEP_EVERY) != 0 || !enough_space()) return;
  long long t = (long long)stamp(ms);
  char dir[80], path[112];
  snprintf(dir, sizeof(dir), "%s/%04d", s_dir_dataset, s_dataset_new / DATASET_PER_DIR);
  struct stat st;
  if ((s_dataset_new % DATASET_PER_DIR) == 0 && stat(dir, &st) != 0) mkdir(dir, 0775);
  snprintf(path, sizeof(path), "%s/%lld.jpg", dir, t);
  if (storage_write_file(path, jpeg, len) != (int)len) return;
  snprintf(path, sizeof(path), "%s/%lld.json", dir, t);
  storage_write_file(path, host_reply_json, strlen(host_reply_json));
  s_dataset_new++;
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
  if (audio_speaker_ready()) {  // the spoken summary is of no use to a board without a speaker
    snprintf(q, sizeof(q), "/api/esp32/memory.pcm?device=%s", host_device_id());
    if (host_fetch_file(q, MEMORY_PCM, &n2) != ESP_OK) return;
  }
  save_rev(rev);
  ESP_LOGI(TAG, "memory cache refreshed (rev %s): %u B json, %.1f s of speech", rev, (unsigned)n1,
           n2 / (2.0f * LOCI_AUDIO_RATE));
  local_memory_cache_object_phrases();
}

// -- per-object spoken answers (for the offline "where is my X") ---------------------------------

static void url_encode(const char *in, char *out, size_t cap) {
  static const char *hex = "0123456789ABCDEF";
  size_t j = 0;
  for (size_t i = 0; in[i] && j + 4 < cap; i++) {
    unsigned char c = (unsigned char)in[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_') {
      out[j++] = (char)c;
    } else {
      out[j++] = '%';
      out[j++] = hex[c >> 4];
      out[j++] = hex[c & 15];
    }
  }
  out[j] = 0;
}

static void safe_name(const char *label, char *out, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; label[i] && j + 1 < cap; i++) {
    char c = label[i];
    out[j++] = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ? c : (c >= 'A' && c <= 'Z') ? (char)(c + 32) : '_';
  }
  out[j] = 0;
}

// Labels the memory may hold that are not things one asks "where is my ..." about.
static const char *k_not_objects[] = {"person"};

static bool is_object_label(const char *label, size_t len) {
  for (size_t i = 0; i < sizeof(k_not_objects) / sizeof(k_not_objects[0]); i++) {
    if (strlen(k_not_objects[i]) == len && !strncmp(label, k_not_objects[i], len)) return false;
  }
  return true;
}

int local_memory_labels(char labels[][32], int max) {
  size_t n = 0;
  uint8_t *json = local_read_file(MEMORY_JSON, &n);
  if (!json) return 0;
  int count = 0;
  const char *p = (const char *)json;
  const char *end = p + n;
  while (count < max && (p = strstr(p, "\"label\":\"")) != NULL && p < end) {
    p += 9;
    const char *e = strchr(p, '"');
    if (!e) break;
    size_t len = (size_t)(e - p);
    if (len > 0 && len < 32 && is_object_label(p, len)) {
      memcpy(labels[count], p, len);
      labels[count][len] = 0;
      count++;
    }
    p = e + 1;
  }
  free(json);
  return count;
}

static bool object_where(const char *label, char *where, size_t cap) {
  size_t n = 0;
  uint8_t *json = local_read_file(MEMORY_JSON, &n);
  if (!json) return false;
  char key[48];
  snprintf(key, sizeof(key), "\"label\":\"%.31s\"", label);
  const char *p = strstr((const char *)json, key);
  bool ok = false;
  if (p) {
    const char *w = strstr(p, "\"where\":\"");
    if (w) {
      w += 9;
      const char *e = strchr(w, '"');
      size_t len = e ? (size_t)(e - w) : 0;
      if (len >= cap) len = cap - 1;
      memcpy(where, w, len);
      where[len] = 0;
      ok = true;
    }
  }
  free(json);
  return ok;
}

void local_memory_cache_object_phrases(void) {
  if (!storage_ready()) return;
  char labels[12][32];
  int n = local_memory_labels(labels, 12);
  for (int i = 0; i < n; i++) {
    char name[40], path[96], where[96] = "", text[200], enc[400], q[460];
    safe_name(labels[i], name, sizeof(name));
    snprintf(path, sizeof(path), "%s/obj_%.39s.pcm", STORAGE_DIR, name);
    object_where(labels[i], where, sizeof(where));
    if (where[0]) {
      snprintf(text, sizeof(text), "Your %.31s was last seen %.95s.", labels[i], where);
    } else {
      snprintf(text, sizeof(text), "I remember a %.31s, but not where.", labels[i]);
    }
    // the phrase changes with the position: a stamp file per object keeps us from refetching
    char stamp_path[100], stamp_old[200] = "";
    snprintf(stamp_path, sizeof(stamp_path), "%.95s.txt", path);
    FILE *f = fopen(stamp_path, "r");
    if (f) {
      if (!fgets(stamp_old, sizeof(stamp_old), f)) stamp_old[0] = 0;
      fclose(f);
    }
    struct stat st;
    if (!audio_speaker_ready()) {  // no speaker: the answer is the text itself, assembled here
      if (strcmp(stamp_old, text) != 0 && storage_write_file(stamp_path, text, strlen(text)) > 0)
        ESP_LOGI(TAG, "cached answer for \"%s\" (text)", labels[i]);
      continue;
    }
    if (!strcmp(stamp_old, text) && stat(path, &st) == 0 && st.st_size > 1000) continue;
    url_encode(text, enc, sizeof(enc));
    snprintf(q, sizeof(q), "/api/esp32/phrase.pcm?text=%.399s", enc);
    size_t bytes = 0;
    if (host_fetch_file(q, path, &bytes) == ESP_OK) {
      storage_write_file(stamp_path, text, strlen(text));
      ESP_LOGI(TAG, "cached answer for \"%s\" (%.1f s)", labels[i], bytes / (2.0f * LOCI_AUDIO_RATE));
    }
  }
}

// The answer as text: the stamp file written next to each clip. On a board without a speaker this
// is the whole answer (serial console today, a phone over BLE later).
static bool print_phrase(const char *pcm_path) {
  char stamp_path[100], text[200] = "";
  snprintf(stamp_path, sizeof(stamp_path), "%.95s.txt", pcm_path);
  FILE *f = fopen(stamp_path, "r");
  if (!f) return false;
  if (!fgets(text, sizeof(text), f)) text[0] = 0;
  fclose(f);
  if (text[0]) ESP_LOGI(TAG, "answer from the card: %s", text);
  return text[0] != 0;
}

bool local_memory_play_object(const char *label) {
  if (!storage_ready() || !label || !label[0]) return false;
  char name[40], path[96];
  safe_name(label, name, sizeof(name));
  snprintf(path, sizeof(path), "%s/obj_%.39s.pcm", STORAGE_DIR, name);
  bool printed = print_phrase(path);
  if (!audio_speaker_ready()) return printed;
  struct stat st;
  if (stat(path, &st) != 0 || st.st_size < 1000) return false;
  return play_file(path);
}

void local_memory_ensure_prompt(void) {
  if (!storage_ready() || s_prompt_checked || !audio_speaker_ready()) return;
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
  if (!audio_speaker_ready()) return false;
  static int16_t *buf;  // 4 kB in PSRAM (allocated once); internal RAM is kept for the system
  if (!buf) buf = heap_caps_malloc(PLAY_CHUNK * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!buf) return false;
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  size_t total = 0, n;
  if (strstr(path, ".wav")) fseek(f, 44, SEEK_SET);
  while ((n = fread(buf, 2, PLAY_CHUNK, f)) > 0) {
    audio_play_pcm16(buf, n);
    total += n;
  }
  fclose(f);
  ESP_LOGI(TAG, "played %s (%.1f s) from the card", path, total / (float)LOCI_AUDIO_RATE);
  return total > 0;
}

bool local_memory_print_summary(void) {
  if (!storage_ready()) return false;
  char labels[12][32], where[96];
  int n = local_memory_labels(labels, 12);
  for (int i = 0; i < n; i++) {
    if (!object_where(labels[i], where, sizeof(where))) where[0] = 0;
    ESP_LOGI(TAG, "remembered: %s%s%s", labels[i], where[0] ? ", last seen " : "", where);
  }
  if (n == 0) ESP_LOGI(TAG, "the cached memory holds no objects yet");
  return n > 0;
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
