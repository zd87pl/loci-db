#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

// What the microSD card is used for (everything is a no-op when the card is not mounted):
//
//   /sdcard/loci/backlog/<ms>_<boot>_<seq>.jpg   frames captured while the host (or Wi-Fi) was unreachable;
//                                            uploaded oldest-first with X-Backlog/X-Captured-Ms once it is back
//   /sdcard/loci/backlog_bad/                backlog files the host refused or that are not JPEGs, kept
//                                            for a look instead of being retried forever
//   /sdcard/loci/memory.json, memory.pcm     the host's remembered objects and a spoken summary, refreshed
//                                            when the host's memory revision changes; played offline on the button
//   /sdcard/loci/offline.pcm                 fixed prompt fetched once ("I'm offline right now...")
//   /sdcard/loci/dataset/<NNNN>/<ms>.jpg + .json   every Nth live frame with the host's detections, 250 per
//                                            folder: a dataset for training a detector on the user's own
//                                            objects (builds before 0.5 wrote them flat into frames/)
//   /sdcard/loci/voice/<ms>_<boot>_<seq>.wav   push-to-talk recordings, for checking the microphone and STT
//
// On a board without a speaker (audio_speaker_ready() false) no speech is fetched: the per-object
// answers are kept as text only (obj_<label>.pcm.txt) and printed instead of played.

void local_memory_init(void);
// Boot counter (NVS), part of every backlog and voice file name.
uint32_t local_boot_id(void);

// -- store-and-forward -------------------------------------------------------------------------
bool local_backlog_push(const uint8_t *jpeg, size_t len, int64_t captured_ms);
int local_backlog_count(void);
// Oldest backlog file; returns false when empty. `captured_ms` is parsed from the file name.
bool local_backlog_oldest(char *path, size_t cap, int64_t *captured_ms);
void local_backlog_remove(const char *path);
// Move a backlog file out of the queue into backlog_bad/ (deleted when that fails); `why` is logged.
void local_backlog_quarantine(const char *path, const char *why);
// Start moving everything in backlog_bad/ back into the queue (after the host is fixed); returns how
// many. The main loop moves them with local_backlog_requeue_step(), one file per call.
int local_backlog_requeue(void);
bool local_backlog_requeue_step(void);  // false when there is nothing (left) to move
// Read a whole file into a PSRAM buffer (caller frees). Returns NULL on failure.
uint8_t *local_read_file(const char *path, size_t *len_out);

// -- dataset / voice ---------------------------------------------------------------------------
void local_dataset_maybe_save(const uint8_t *jpeg, size_t len, int64_t ms, const char *host_reply_json);
void local_voice_save(const int16_t *pcm, size_t samples, int64_t ms);

// -- cached memory + offline speech ------------------------------------------------------------
// Where the cache files come from: the host (host_fetch_file, the default) or LOCI Cloud
// (cloud_link_fetch_file), which serves the same /api/esp32/<route> names under board/.
typedef esp_err_t (*local_fetch_fn)(const char *path_and_query, const char *dest_path, size_t *bytes_out);
void local_memory_set_source(local_fetch_fn fetch);
// Fetch memory.json / memory.pcm from the host when `rev` differs from the cached one (rate limited).
// True when the card now holds a new memory (time to rebuild the offline commands).
bool local_memory_refresh(const char *rev);
// Cloud mode (no frame replies carry a rev): fetch memory.json, and when its rev differs from the
// cached one, the spoken summary too. True when the cache changed (then fetch the phrases).
bool local_memory_poll(void);
// Fetch up to `max` per-object answers the card lacks (or that changed); returns how many it fetched,
// 0 when all are up to date. Lets the main loop spread the fetches and keep the button live.
int local_memory_cache_some_phrases(int max);
// Fetch the fixed offline prompt once.
void local_memory_ensure_prompt(void);
// Speak what the card holds: memory.pcm if present, else offline.pcm. True if something played.
bool local_memory_play_offline(void);
// The same as text, one log line per remembered object (boards without a speaker). True if any.
bool local_memory_print_summary(void);
// Per-object spoken answers ("Your cup was last seen in the center..."), fetched from the host as
// /sdcard/loci/obj_<label>.pcm whenever the cached memory changes; played offline on a recognised
// "where is my <label>".
void local_memory_cache_object_phrases(void);
bool local_memory_play_object(const char *label);
// Labels in the cached memory.json (lower case, up to `max`); returns the count.
int local_memory_labels(char labels[][32], int max);
// Console helpers
void local_memory_print(void);
int local_dataset_count(void);
