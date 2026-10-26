#pragma once

#include <stdbool.h>

#include "esp_err.h"
#include "loci_hw.h"  // camera_capture_jpeg() / camera_release() / camera_width() / camera_height()
#include "esp_camera.h"  // framesize_t, camera_fb_t

// DVP sensor through esp32-camera, JPEG straight from the sensor (OV5640/OV3660/OV2640 encode on chip;
// the S3 has no JPEG encoder).
esp_err_t camera_init(int jpeg_quality, framesize_t frame_size);
// Change the resolution at runtime by name: qvga vga svga xga hd sxga uxga fhd qxga. False if unknown.
bool camera_set_frame_size(const char *name);
const char *camera_sensor_name(void);
// Name of the current resolution (as accepted by camera_set_frame_size).
const char *camera_frame_size_name(void);
// Autofocus (OV5640 modules with a focus motor): log the status, or run one focus cycle first.
void camera_af_report(bool trigger);
