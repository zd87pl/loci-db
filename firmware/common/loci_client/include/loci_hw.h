// What a board project provides to the shared thin-client code (firmware/common/loci_client).
//
// Each board project (firmware/esp32-*-client/main) implements these functions for its camera,
// audio path and microSD slot, brings the hardware up in its own app_main() and then hands over to
// loci_app_run(). Initialisation is board-specific and declared in the board project's own headers.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "sdmmc_cmd.h"

// ---- camera ---------------------------------------------------------------------------------------
// Grab one frame as JPEG. The buffer stays valid until camera_release(); call them strictly in pairs.
esp_err_t camera_capture_jpeg(uint8_t **jpeg, size_t *len, uint32_t *width, uint32_t *height);
void camera_release(void);
uint32_t camera_width(void);
uint32_t camera_height(void);

// ---- audio (16 kHz mono PCM16, LOCI_AUDIO_RATE) ---------------------------------------------------
// A board may have a speaker, a microphone, both or neither; the shared code checks before use.
bool audio_speaker_ready(void);
bool audio_mic_ready(void);
// Blocking: queue `samples` mono PCM16 samples to the speaker.
esp_err_t audio_play_pcm16(const int16_t *mono, size_t samples);
// Blocking: read `samples` mono PCM16 samples from the microphone.
esp_err_t audio_record_pcm16(int16_t *mono, size_t samples);
// Throw away whatever the microphone DMA accumulated while nobody was reading.
void audio_flush_mic(void);
void audio_set_volume(int volume_pct);
// Short tone as user feedback; boards without a speaker may signal some other way or do nothing.
void audio_beep(float freq_hz, uint32_t ms);

// ---- microSD slot ---------------------------------------------------------------------------------
// Power and initialise the card in the board's slot (SDMMC or SPI) and return it, ready for FatFs.
// storage.c registers it with FatFs/VFS and mounts it; on failure it calls storage_board_close_card().
esp_err_t storage_board_open_card(sdmmc_card_t **card);
void storage_board_close_card(sdmmc_card_t *card);
