#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// What the microSD card is used for (everything is a no-op when the card is not mounted):
//
//   /sdcard/loci/backlog/<captured_ms>.jpg   frames captured while the host (or Wi-Fi) was unreachable;
//                                            uploaded oldest-first with X-Backlog/X-Captured-Ms once it is back
//   /sdcard/loci/memory.json, memory.pcm     the host's remembered objects and a spoken summary, refreshed
//                                            when the host's memory revision changes; played offline on the button
//   /sdcard/loci/offline.pcm                 fixed prompt fetched once ("I can't reach the host...")
//   /sdcard/loci/frames/<ms>.jpg + .json     every Nth live frame with the host's detections: a dataset for
//                                            training a detector on the user's own objects
//   /sdcard/loci/voice/<ms>.wav              push-to-talk recordings, for checking the microphone and STT

void local_memory_init(void);

// -- store-and-forward -------------------------------------------------------------------------
bool local_backlog_push(const uint8_t *jpeg, size_t len, int64_t captured_ms);
int local_backlog_count(void);
// Oldest backlog file; returns false when empty. `captured_ms` is parsed from the file name.
bool local_backlog_oldest(char *path, size_t cap, int64_t *captured_ms);
void local_backlog_remove(const char *path);
// Read a whole file into a PSRAM buffer (caller frees). Returns NULL on failure.
uint8_t *local_read_file(const char *path, size_t *len_out);

// -- dataset / voice ---------------------------------------------------------------------------
void local_dataset_maybe_save(const uint8_t *jpeg, size_t len, int64_t ms, const char *host_reply_json);
void local_voice_save(const int16_t *pcm, size_t samples, int64_t ms);

// -- cached memory + offline speech ------------------------------------------------------------
// Fetch memory.json / memory.pcm from the host when `rev` differs from the cached one (rate limited).
void local_memory_refresh(const char *rev);
// Fetch the fixed offline prompt once.
void local_memory_ensure_prompt(void);
// Speak what the card holds: memory.pcm if present, else offline.pcm. True if something played.
bool local_memory_play_offline(void);
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
