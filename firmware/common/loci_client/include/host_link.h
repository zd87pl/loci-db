#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

// HTTP link to the LOCI host (see docs/ESP32_THIN_CLIENT_DEMO.md, section 2).

typedef struct {
  int http_status;     // status of the post, set on failure too (<= 0: no answer from the host)
  bool dropped;        // host skipped detection (rate limit) but kept the frame
  int audio_pending;   // clips waiting in this device's speaker queue
  int next_ms;         // interval the host would like between frames
  char labels[96];     // comma-separated labels the host saw, for the console
  char memory_rev[24]; // digest of the host's remembered objects (refresh the card cache when it changes)
} host_frame_result_t;

// `base_url`, `device_id` and `token` are the compile-time defaults (secrets.h); a host URL or
// token stored in NVS by the console commands `host <url>` / `token <value>` overrides them.
void host_link_init(const char *base_url, const char *device_id, const char *token);

// Runtime host URL (http:// or https://, no trailing slash), kept in NVS; NULL restores the default.
bool host_link_set_host(const char *url);
// Device token sent as X-Device-Token (needed from outside the LAN); NULL clears it. Never logged.
void host_link_set_token(const char *token);
const char *host_link_base(void);
bool host_link_has_token(void);

esp_err_t host_post_frame(const uint8_t *jpeg, size_t len, uint32_t seq, host_frame_result_t *out);
// Same, for a frame kept on the card: captured_ms (Unix, 0 = unknown) and the backlog flag go in headers.
esp_err_t host_post_frame_ex(const uint8_t *jpeg, size_t len, uint32_t seq, int64_t captured_ms, bool backlog,
                             host_frame_result_t *out);
// True for the statuses with which the host refuses one frame (not a JPEG, too large): retrying that
// frame cannot help, and the link itself is fine.
bool host_status_rejects_frame(int status);
// Unix time in ms: from the host's replies, else the system clock when it survived a soft reset,
// else 0 (power-on and no host reply yet).
int64_t host_now_ms(void);
// Where host_now_ms() comes from: "host", "kept across reset" or "unknown".
const char *host_clock_source(void);
// The raw body of the last frame reply (for the dataset side-car files).
const char *host_last_reply(void);
// Device id given to host_link_init (the one the host knows this board by).
const char *host_device_id(void);
// False after two consecutive failed frame posts; true again after a success.
bool host_reachable(void);
// Shorter while offline so a probe does not stall capture.
void host_set_frame_timeout_ms(int ms);
// GET <base><path_and_query> into dest_path (written to a temp file, then renamed). 204 -> ESP_ERR_NOT_FOUND.
esp_err_t host_fetch_file(const char *path_and_query, const char *dest_path, size_t *bytes_out);
// Download the next queued clip (long-polling up to wait_s) and play it while it streams.
// Returns true when something was played.
bool host_fetch_and_play_audio(int wait_s);
// One push-to-talk utterance as raw PCM16 mono at LOCI_AUDIO_RATE; plays the answer.
esp_err_t host_post_voice(const uint8_t *pcm, size_t len);
// Typed question / arbitrary text; both play the host's speech afterwards.
esp_err_t host_ask(const char *text);
esp_err_t host_say(const char *text);

uint32_t host_audio_clips_played(void);
