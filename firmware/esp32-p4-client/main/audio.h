#pragma once

#include "driver/i2c_master.h"
#include "esp_err.h"
#include "loci_hw.h"  // the runtime audio API (play, record, beep) the shared code uses

// ES8311 codec on I2S0 + the shared I2C bus; speaker through the NS4150B, onboard microphone.
esp_err_t audio_init(i2c_master_bus_handle_t i2c_bus, int volume_pct);
