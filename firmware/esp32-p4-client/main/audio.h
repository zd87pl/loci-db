#pragma once

#include <stddef.h>
#include <stdint.h>

#include "driver/i2c_master.h"
#include "esp_err.h"

// ES8311 codec on I2S0 + the shared I2C bus; speaker through the NS4150B, onboard microphone.
esp_err_t audio_init(i2c_master_bus_handle_t i2c_bus, int volume_pct);
bool audio_ready(void);

// Blocking: queue `samples` mono PCM16 samples at LOCI_AUDIO_RATE to the speaker.
esp_err_t audio_play_pcm16(const int16_t *mono, size_t samples);
// Blocking: read `samples` mono PCM16 samples from the microphone.
esp_err_t audio_record_pcm16(int16_t *mono, size_t samples);
// Throw away whatever the microphone DMA accumulated while nobody was reading.
void audio_flush_mic(void);

void audio_set_volume(int volume_pct);
void audio_beep(float freq_hz, uint32_t ms);
