#include "voice_local.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_afe_config.h"
#include "esp_afe_sr_iface.h"
#include "esp_afe_sr_models.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_mn_iface.h"
#include "esp_mn_models.h"
#include "esp_mn_speech_commands.h"
#include "esp_timer.h"
#include "model_path.h"

#include "loci_config.h"

static const char *TAG = "voice";

#define MAX_LABELS 32
#define LABEL_LEN 32

static srmodel_list_t *s_models;
static const esp_afe_sr_iface_t *s_afe;
static esp_afe_sr_data_t *s_afe_data;
static esp_mn_iface_t *s_mn;
static model_iface_data_t *s_mn_data;
static int s_feed_chunk, s_fetch_chunk, s_mn_chunk;
static char s_labels[MAX_LABELS][LABEL_LEN];
static int s_label_count;
static bool s_ready;

// Objects worth recognising even before the memory has anything: all COCO classes a home has.
static const char *k_default_labels[] = {"cup",    "bottle",   "laptop",  "cell phone", "book",  "keys",   "remote",
                                         "backpack", "handbag", "scissors", "mouse",   "keyboard", "chair", "glasses",
                                         "wallet", "bowl",     "banana",  "apple",     "clock", "umbrella"};

static void upper_ascii(const char *in, char *out, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; in[i] && j + 1 < cap; i++) {
    char c = in[i];
    out[j++] = (c >= 'a' && c <= 'z') ? (char)(c - 32) : c;
  }
  out[j] = 0;
}

static void add_label(const char *label) {
  if (!label || !label[0] || s_label_count >= MAX_LABELS) return;
  for (int i = 0; i < s_label_count; i++) {
    if (!strcmp(s_labels[i], label)) return;
  }
  snprintf(s_labels[s_label_count++], LABEL_LEN, "%s", label);
}

static void rebuild_commands(void) {
  if (!s_ready) return;
  esp_mn_commands_clear();
  esp_mn_commands_add(VOICE_CMD_WHAT_DO_YOU_SEE, "WHAT DO YOU SEE");
  esp_mn_commands_add(VOICE_CMD_WHAT_DO_YOU_SEE, "WHAT CAN YOU SEE");
  char up[LABEL_LEN], phrase[LABEL_LEN + 16];
  for (int i = 0; i < s_label_count; i++) {
    upper_ascii(s_labels[i], up, sizeof(up));
    snprintf(phrase, sizeof(phrase), "WHERE IS MY %s", up);
    esp_mn_commands_add(VOICE_CMD_FIRST_OBJECT + i, phrase);
    snprintf(phrase, sizeof(phrase), "WHERE ARE MY %s", up);
    esp_mn_commands_add(VOICE_CMD_FIRST_OBJECT + i, phrase);
  }
  esp_mn_error_t *err = esp_mn_commands_update();
  if (err && err->num > 0) ESP_LOGW(TAG, "%d command phrases were rejected by MultiNet", err->num);
  ESP_LOGI(TAG, "%d objects, %d command phrases active", s_label_count, 2 + 2 * s_label_count);
}

bool voice_local_init(void) {
  if (s_ready) return true;
  s_models = esp_srmodel_init("model");
  if (!s_models) {
    ESP_LOGW(TAG, "no speech models: is the \"model\" partition flashed?");
    return false;
  }
  char *mn_name = esp_srmodel_filter(s_models, ESP_MN_PREFIX, ESP_MN_ENGLISH);
  if (!mn_name) {
    ESP_LOGW(TAG, "no English MultiNet model in the partition");
    return false;
  }
  afe_config_t *cfg = afe_config_init("M", s_models, AFE_TYPE_SR, AFE_MODE_HIGH_PERF);
  if (!cfg) {
    ESP_LOGW(TAG, "afe_config_init failed");
    return false;
  }
  cfg->wakenet_init = false;  // the button is the wake word
  cfg->aec_init = false;      // nothing plays while we record
  cfg->vad_init = false;      // the clip is already trimmed by the button
  s_afe = esp_afe_handle_from_config(cfg);
  s_afe_data = s_afe ? s_afe->create_from_config(cfg) : NULL;
  afe_config_free(cfg);
  if (!s_afe_data) {
    ESP_LOGW(TAG, "AFE creation failed");
    return false;
  }
  s_feed_chunk = s_afe->get_feed_chunksize(s_afe_data);
  s_fetch_chunk = s_afe->get_fetch_chunksize(s_afe_data);

  s_mn = esp_mn_handle_from_name(mn_name);
  s_mn_data = s_mn ? s_mn->create(mn_name, 6000) : NULL;
  if (!s_mn_data) {
    ESP_LOGW(TAG, "MultiNet %s creation failed", mn_name);
    return false;
  }
  s_mn_chunk = s_mn->get_samp_chunksize(s_mn_data);
  esp_mn_commands_alloc(s_mn, s_mn_data);
  s_ready = true;
  for (size_t i = 0; i < sizeof(k_default_labels) / sizeof(k_default_labels[0]); i++) add_label(k_default_labels[i]);
  rebuild_commands();
  ESP_LOGI(TAG, "%s ready: AFE feed %d / fetch %d samples, MultiNet chunk %d, heap %u psram %u", mn_name,
           s_feed_chunk, s_fetch_chunk, s_mn_chunk, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  return true;
}

bool voice_local_ready(void) { return s_ready; }

static int cmp_label(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

void voice_local_set_labels(const char labels[][32], int count) {
  // two scratch tables in PSRAM: as .bss they would come out of the P4's retention RAM (see the
  // P4 README), where every kB counts
  static char (*before)[LABEL_LEN], (*sorted)[LABEL_LEN];
  if (!before) {
    before = heap_caps_calloc(2 * MAX_LABELS, LABEL_LEN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!before) before = heap_caps_calloc(2 * MAX_LABELS, LABEL_LEN, MALLOC_CAP_8BIT);
    if (!before) return;
    sorted = before + MAX_LABELS;
  }
  int before_count = s_label_count;
  memcpy(before, s_labels, sizeof(s_labels));
  // memory.json lists the newest sighting first and that order changes with every refresh; sorted,
  // the list (and with it the command ids) only changes when an object is added or dropped
  if (count > MAX_LABELS) count = MAX_LABELS;
  for (int i = 0; i < count; i++) snprintf(sorted[i], LABEL_LEN, "%s", labels[i]);
  qsort(sorted, count, LABEL_LEN, cmp_label);
  s_label_count = 0;
  memset(s_labels, 0, sizeof(s_labels));  // so the comparison below sees no stale rows
  for (int i = 0; i < count; i++) add_label(sorted[i]);  // remembered objects first: they must fit
  for (size_t i = 0; i < sizeof(k_default_labels) / sizeof(k_default_labels[0]); i++) add_label(k_default_labels[i]);
  // Every memory refresh lands here, but a moved object changes no phrase: MultiNet is only
  // rebuilt (a few hundred ms in the frame loop) when the list of objects itself changed.
  if (s_label_count == before_count && !memcmp(before, s_labels, sizeof(s_labels))) return;
  rebuild_commands();
}

int voice_local_recognize(const int16_t *pcm, size_t samples, char *label_out, size_t label_cap) {
  if (!s_ready || !pcm || samples < (size_t)s_feed_chunk) return VOICE_CMD_NONE;
  int64_t t0 = esp_timer_get_time();
  s_afe->reset_buffer(s_afe_data);
  s_mn->clean(s_mn_data);
  if (label_out && label_cap) label_out[0] = 0;

  // Feed the clip plus half a second of silence so the last word is flushed through the pipeline.
  const size_t tail = LOCI_AUDIO_RATE / 2;
  int16_t *zeros = heap_caps_calloc(s_feed_chunk, sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!zeros) zeros = heap_caps_calloc(s_feed_chunk, sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!zeros) {
    ESP_LOGE(TAG, "no memory to flush the recogniser");
    return VOICE_CMD_NONE;
  }
  int result = VOICE_CMD_NONE;
  float prob = 0.0f;
  size_t fed = 0;
  while (result == VOICE_CMD_NONE && fed < samples + tail) {
    const int16_t *chunk = fed + s_feed_chunk <= samples ? pcm + fed : zeros;
    s_afe->feed(s_afe_data, chunk);
    fed += s_feed_chunk;
    afe_fetch_result_t *res = s_afe->fetch(s_afe_data);
    if (!res || res->ret_value != ESP_OK || !res->data) continue;
    esp_mn_state_t state = s_mn->detect(s_mn_data, res->data);
    if (state == ESP_MN_STATE_DETECTED) {
      esp_mn_results_t *r = s_mn->get_results(s_mn_data);
      if (r && r->num > 0) {
        result = r->command_id[0];
        prob = r->prob[0];
      }
    } else if (state == ESP_MN_STATE_TIMEOUT) {
      break;
    }
  }
  free(zeros);
  int ms = (int)((esp_timer_get_time() - t0) / 1000);
  if (result >= VOICE_CMD_FIRST_OBJECT && result - VOICE_CMD_FIRST_OBJECT < s_label_count) {
    if (label_out) snprintf(label_out, label_cap, "%s", s_labels[result - VOICE_CMD_FIRST_OBJECT]);
    ESP_LOGI(TAG, "recognised \"where is my %s\" (p=%.2f) in %d ms", s_labels[result - VOICE_CMD_FIRST_OBJECT],
             (double)prob, ms);
  } else if (result == VOICE_CMD_WHAT_DO_YOU_SEE) {
    ESP_LOGI(TAG, "recognised \"what do you see\" (p=%.2f) in %d ms", (double)prob, ms);
  } else {
    ESP_LOGI(TAG, "no command recognised (%.1f s of audio, %d ms)", samples / (float)LOCI_AUDIO_RATE, ms);
    result = VOICE_CMD_NONE;
  }
  return result;
}
