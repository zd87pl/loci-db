#pragma once

#include "esp_err.h"
#include "loci_hw.h"  // the runtime audio API the shared code uses

// PDM microphone on I2S0 (PDM RX, hardware PDM->PCM at 16 kHz). There is no speaker on the W11:
// audio_speaker_ready() is false, playback is refused and beeps are silent.
esp_err_t audio_init(void);
// Peak and mean absolute level of the last audio_record_pcm16() call (after gain), for the console.
void audio_last_levels(int *peak, int *mean);
