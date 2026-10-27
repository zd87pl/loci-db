#pragma once

#include "driver/i2c_master.h"
#include "esp_err.h"
#include "loci_hw.h"  // camera_capture_jpeg() / camera_release() / camera_width() / camera_height()

// Bring up the MIPI-CSI sensor through esp_video on an I2C bus the app already owns, open the
// capture device and the hardware JPEG M2M encoder, negotiate a format and start streaming.
esp_err_t camera_init(i2c_master_bus_handle_t i2c_bus, int jpeg_quality);
