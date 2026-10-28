#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// One detection made on the board, its box normalised to the frame (0-1).
typedef struct {
  char label[24];
  float score;
  float cx, cy, w, h;
} loci_detection_t;

// The board's own sightings for LOCI Cloud (cloud mode). Detections are merged per object: the same
// label seen again within 60 s and within 0.15 of the frame of its last position keeps its id
// (d_<device>_<boot>_<nonce>_<n>) and is re-sent as an upsert at most every 30 s; anything else
// starts a new sighting. Batches go to POST spaces/<s>/observations; while the cloud is out of reach
// they are appended to /sdcard/loci/sightings.jsonl and uploaded, oldest first, once it is back.
// People are not sent. Nothing is recorded until the clock is set (SNTP).

void sightings_init(void);
// Fold one frame's detections in.
void sightings_add(const loci_detection_t *dets, int n);
// Send what changed (after the card queue) when the cloud is usable, else queue it on the card.
// Returns how many observations the cloud wrote.
int sightings_flush(void);
int sightings_tracked(void);
size_t sightings_queue_bytes(void);
