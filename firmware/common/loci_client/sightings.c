#include "sightings.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_random.h"

#include "cloud_link.h"
#include "local_memory.h"
#include "storage.h"

static const char *TAG = "sightings";

#define MAX_TRACKS 32
#define MERGE_MS 60000      // same object if seen again within a minute ...
#define MERGE_DIST 0.15f    // ... and this close (fraction of the frame) to where it was
#define RESEND_MS 30000     // a tracked object is re-sent at most this often
#define MIN_SCORE 0.5f
#define BATCH 50            // the cloud's limit per request
#define BODY_CAP (BATCH * 320 + 64)
#define QUEUE_PATH STORAGE_DIR "/sightings.jsonl"
#define QUEUE_MAX_BYTES (512 * 1024)  // ~2,000 lines; read whole into PSRAM to upload

typedef struct {
  bool used, dirty;
  char label[24];
  char id[80];
  float cx, cy, w, h, score;
  int64_t t;       // last seen, Unix ms
  int64_t sent_t;  // last sent or queued, 0 = never
} track_t;

static track_t *s_tracks;  // PSRAM
static char *s_body;       // PSRAM, one request body
static uint32_t s_nonce, s_next;

void sightings_init(void) {
  if (!s_tracks) s_tracks = heap_caps_calloc(MAX_TRACKS, sizeof(track_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!s_body) s_body = heap_caps_malloc(BODY_CAP, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!s_tracks || !s_body) ESP_LOGE(TAG, "no memory for sightings");
  s_nonce = esp_random();  // ids stay unique even if NVS (and with it the boot counter) is erased
}

int sightings_tracked(void) {
  int n = 0;
  for (int i = 0; s_tracks && i < MAX_TRACKS; i++) n += s_tracks[i].used;
  return n;
}

size_t sightings_queue_bytes(void) {
  struct stat st;
  return storage_ready() && stat(QUEUE_PATH, &st) == 0 ? (size_t)st.st_size : 0;
}

static void new_id(char *out, size_t cap) {
  const char *dev = cloud_link_device_id();
  if (!strncmp(dev, "dev_", 4)) dev += 4;
  snprintf(out, cap, "d_%s_%lu_%08lx_%lu", dev[0] ? dev : "board", (unsigned long)local_boot_id(),
           (unsigned long)s_nonce, (unsigned long)++s_next);
}

void sightings_add(const loci_detection_t *dets, int n) {
  int64_t now = cloud_link_now_ms();
  if (!s_tracks || now == 0) return;  // no real time yet: the cloud would file it in 1970
  for (int k = 0; k < n; k++) {
    const loci_detection_t *d = &dets[k];
    if (d->score < MIN_SCORE || !d->label[0] || !strcmp(d->label, "person")) continue;
    track_t *best = NULL;
    float best_dist = MERGE_DIST * 2;
    for (int i = 0; i < MAX_TRACKS; i++) {
      track_t *t = &s_tracks[i];
      if (!t->used || strcmp(t->label, d->label) || now - t->t > MERGE_MS) continue;
      float dx = fabsf(t->cx - d->cx), dy = fabsf(t->cy - d->cy);
      if (dx < MERGE_DIST && dy < MERGE_DIST && dx + dy < best_dist) {
        best = t;
        best_dist = dx + dy;
      }
    }
    if (!best) {  // a new sighting: a free slot, or the one seen longest ago
      best = &s_tracks[0];
      for (int i = 0; i < MAX_TRACKS; i++) {
        if (!s_tracks[i].used) {
          best = &s_tracks[i];
          break;
        }
        if (s_tracks[i].t < best->t) best = &s_tracks[i];
      }
      memset(best, 0, sizeof(*best));
      best->used = true;
      snprintf(best->label, sizeof(best->label), "%s", d->label);
      new_id(best->id, sizeof(best->id));
    }
    best->cx = d->cx;
    best->cy = d->cy;
    best->w = d->w;
    best->h = d->h;
    best->score = d->score;
    best->t = now;
    best->dirty = true;
  }
}

static int obs_json(const track_t *t, char *out, size_t cap) {
  const char *room = cloud_link_room();
  char room_part[80] = "";
  if (room[0] && !strpbrk(room, "\"\\")) snprintf(room_part, sizeof(room_part), ",\"room\":\"%s\"", room);
  return snprintf(out, cap,
                  "{\"id\":\"%s\",\"label\":\"%s\",\"frame\":\"image\",\"x\":%.3f,\"y\":%.3f,\"z\":0.5,\"t\":%lld,"
                  "\"confidence\":%.2f%s,\"metadata\":{\"source\":\"board\",\"w\":%.3f,\"h\":%.3f}}",
                  t->id, t->label, (double)t->cx, (double)t->cy, (long long)t->t, (double)t->score, room_part,
                  (double)t->w, (double)t->h);
}

// One request: {"observations":[<lines>],"upsert":true}. `items` are observation objects.
static int post_batch(const char *const *items, int n) {
  size_t used = (size_t)snprintf(s_body, BODY_CAP, "{\"observations\":[");
  for (int i = 0; i < n; i++) {
    int w = snprintf(s_body + used, BODY_CAP - used, "%s%s", i ? "," : "", items[i]);
    if (w < 0 || used + (size_t)w >= BODY_CAP - 32) break;
    used += (size_t)w;
  }
  used += (size_t)snprintf(s_body + used, BODY_CAP - used, "],\"upsert\":true}");
  int written = 0;
  esp_err_t rc = cloud_link_post_observations(s_body, used, &written);
  if (rc == ESP_ERR_INVALID_RESPONSE) {
    ESP_LOGW(TAG, "the cloud refused a batch of %d; dropping it", n);
    return 0;  // retrying cannot help
  }
  return rc == ESP_OK ? written : -1;
}

static void queue_append(const char *line) {
  if (!storage_ready() || sightings_queue_bytes() > QUEUE_MAX_BYTES) return;
  FILE *f = fopen(QUEUE_PATH, "a");
  if (!f) return;
  fputs(line, f);
  fputc('\n', f);
  fclose(f);
}

// Upload the card queue in batches; the file goes once every batch is in.
static int upload_queue(void) {
  if (!sightings_queue_bytes()) return 0;
  size_t len = 0;
  char *buf = (char *)local_read_file(QUEUE_PATH, &len);
  if (!buf) return 0;
  const char *items[BATCH];
  int n = 0, written = 0;
  bool ok = true;
  for (char *line = strtok(buf, "\n"); line && ok; line = strtok(NULL, "\n")) {
    if (line[0] != '{') continue;
    items[n++] = line;
    if (n == BATCH) {
      int w = post_batch(items, n);
      ok = w >= 0;
      written += w > 0 ? w : 0;
      n = 0;
    }
  }
  if (ok && n) {
    int w = post_batch(items, n);
    ok = w >= 0;
    written += w > 0 ? w : 0;
  }
  free(buf);
  if (ok) {
    unlink(QUEUE_PATH);
    ESP_LOGI(TAG, "card queue uploaded (%d written)", written);
  }
  return written;
}

int sightings_flush(void) {
  int64_t now = cloud_link_now_ms();
  if (!s_tracks || !s_body || now == 0) return 0;
  char *lines[MAX_TRACKS];
  track_t *due[MAX_TRACKS];
  int n = 0;
  for (int i = 0; i < MAX_TRACKS; i++) {
    track_t *t = &s_tracks[i];
    if (!t->used || !t->dirty || (t->sent_t && now - t->sent_t < RESEND_MS)) continue;
    due[n++] = t;
  }
  bool online = cloud_link_usable();
  int written = online ? upload_queue() : 0;
  if (!n) return written;
  static char (*pool)[320];  // MAX_TRACKS observation objects, ~10 kB in PSRAM (never internal RAM)
  if (!pool) pool = heap_caps_malloc(MAX_TRACKS * sizeof(*pool), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!pool) return written;
  for (int i = 0; i < n; i++) {
    obs_json(due[i], pool[i], sizeof(pool[i]));
    lines[i] = pool[i];
  }
  int w = online ? post_batch((const char *const *)lines, n) : -1;
  if (w < 0) {
    for (int i = 0; i < n; i++) queue_append(lines[i]);  // keep them for when the cloud is back
  } else {
    written += w;
  }
  for (int i = 0; i < n; i++) {
    due[i]->dirty = false;
    due[i]->sent_t = now;
  }
  if (n) ESP_LOGI(TAG, "%d sightings %s", n, w >= 0 ? "sent" : "queued on the card");
  return written;
}
