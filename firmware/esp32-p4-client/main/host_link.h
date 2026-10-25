#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

// HTTP link to the LOCI host (see docs/ESP32_THIN_CLIENT_DEMO.md, section 2).

typedef struct {
  bool dropped;        // host skipped detection (rate limit) but kept the frame
  int audio_pending;   // clips waiting in this device's speaker queue
  int next_ms;         // interval the host would like between frames
  char labels[96];     // comma-separated labels the host saw, for the console
} host_frame_result_t;

void host_link_init(const char *base_url, const char *device_id);

esp_err_t host_post_frame(const uint8_t *jpeg, size_t len, uint32_t seq, host_frame_result_t *out);
// Download the next queued clip (long-polling up to wait_s) and play it while it streams.
// Returns true when something was played.
bool host_fetch_and_play_audio(int wait_s);
// One push-to-talk utterance as raw PCM16 mono at LOCI_AUDIO_RATE; plays the answer.
esp_err_t host_post_voice(const uint8_t *pcm, size_t len);
// Typed question / arbitrary text; both play the host's speech afterwards.
esp_err_t host_ask(const char *text);
esp_err_t host_say(const char *text);

uint32_t host_audio_clips_played(void);
