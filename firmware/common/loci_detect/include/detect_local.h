#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sightings.h"

#ifdef __cplusplus
extern "C" {
#endif

// On-board object detection with ESP-DL (COCO YOLO11n, 320x320 int8, the "coco_det" flash partition;
// the model variant for the chip is picked by the coco_detect component). In cloud mode
// (loci_app.h) detect_local_boxes() feeds the board's own sightings; the console command `detect`
// runs it on the current camera frame and prints labels and timing, to compare the board's view
// with the host's YOLO on the same scene.

typedef struct {
  int count;          // detections above the score threshold
  int decode_ms;      // JPEG -> RGB565, downscaled to <= 640 wide (software, esp_new_jpeg)
  int infer_ms;       // preprocess + model + postprocess
  int width, height;  // decoded (downscaled) frame
  char labels[160];   // "cup 0.71, chair 0.55"
} detect_local_result_t;

// Load the model (about 1 s the first time). False when the model partition is missing.
bool detect_local_init(void);
// Decode the JPEG and run the detector on it.
bool detect_local_run(const uint8_t *jpeg, size_t len, detect_local_result_t *out);
// The same, as boxes normalised to the frame (0-1) for the sightings sent to LOCI Cloud (cloud mode);
// returns the count (at most `max`), or -1 when the detector cannot run. Signature of loci_app's detect hook.
int detect_local_boxes(const uint8_t *jpeg, size_t len, loci_detection_t *out, int max);

#ifdef __cplusplus
}
#endif
