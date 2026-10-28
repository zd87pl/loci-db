#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

// LOCI Cloud for the boards (cloud/workers, docs: cloud/workers/README.md "Board routes"). A board
// claims itself once with a code from cloud/workers/scripts/devices.py (console: `cloud claim
// <project_url> <space> <CODE>`), then keeps a one-hour device token and calls
//
//   POST spaces/<s>/observations                          its own sightings (upsert)
//   GET  spaces/<s>/board/memory | memory.pcm | phrase.pcm the card cache, in the host's formats
//   POST spaces/<s>/board/ask | board/voice                spoken answers (16 kHz PCM16)
//
// Kept in NVS (namespace "loci"): c_url (project URL, https://<worker>/p/<ref>/v1), c_space,
// c_dev, c_sec (the device secret, never logged) and c_room (the room this board is in, sent with
// its sightings).

void cloud_link_init(const char *fw_version);
// Start SNTP once Wi-Fi is up: sightings need real time stamps (there is no host to ask).
void cloud_link_start_clock(void);
// Unix ms from the system clock, 0 while it is not set.
int64_t cloud_link_now_ms(void);

bool cloud_link_configured(void);
// Configured, Wi-Fi up, outside the 60 s pause after a failed request, and (P4) enough internal RAM
// for a TLS session.
bool cloud_link_usable(void);
// Exchange a one-time claim code for this board's device id and secret (stored before success is
// reported: a code works once, so a lost reply means a new code).
esp_err_t cloud_link_claim(const char *project_url, const char *space, const char *code);
void cloud_link_forget(void);
void cloud_link_print_status(void);
const char *cloud_link_device_id(void);  // "" until claimed
const char *cloud_link_room(void);       // "" when not set
void cloud_link_set_room(const char *room);

// Spoken answers, played while they stream in. ESP_OK when an answer was played (or printed on a
// board without a speaker); from voice, ESP_ERR_NOT_FOUND when the cloud heard nothing.
esp_err_t cloud_link_ask(const char *text);
esp_err_t cloud_link_voice(const uint8_t *pcm, size_t len);

// GET a board route into a file, for local_memory's cache. `host_path` is the host's path
// ("/api/esp32/memory?device=...", "/api/esp32/phrase.pcm?text=..."): the same route name is
// served under spaces/<s>/board/. Signature of local_fetch_fn (local_memory.h).
esp_err_t cloud_link_fetch_file(const char *host_path, const char *dest_path, size_t *bytes_out);

// POST a ready {"observations":[...],"upsert":true} body. `written` = accepted + updated.
esp_err_t cloud_link_post_observations(const char *json, size_t len, int *written);
