#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// On-board object detection with ESP-DL (COCO YOLO11n, 320x320 int8, the "coco_det" flash partition).
// A first step towards sending observations instead of frames: the console command `detect` runs
// it on the current camera frame and prints the labels next to the timing, so the board's own view
// can be compared with the host's YOLOv8n on the same scene.

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

#ifdef __cplusplus
}
#endif
