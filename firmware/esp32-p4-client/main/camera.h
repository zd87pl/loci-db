#pragma once

#include <stddef.h>
#include <stdint.h>

#include "driver/i2c_master.h"
#include "esp_err.h"

// Bring up the MIPI-CSI sensor through esp_video on an I2C bus the app already owns, open the
// capture device and the hardware JPEG M2M encoder, negotiate a format and start streaming.
esp_err_t camera_init(i2c_master_bus_handle_t i2c_bus, int jpeg_quality);

// Grab one frame and encode it. The returned JPEG lives in the encoder's capture buffer and is
// valid until camera_release() is called; call them strictly in pairs.
esp_err_t camera_capture_jpeg(uint8_t **jpeg, size_t *len, uint32_t *width, uint32_t *height);
void camera_release(void);

uint32_t camera_width(void);
uint32_t camera_height(void);
