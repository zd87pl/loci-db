// PDM microphone through the S3's I2S0 PDM-RX path (hardware PDM->PCM), 16 kHz mono PCM16.
// The raw stream carries a DC offset and is quiet, so every read goes through a one-pole DC blocker
// and a fixed gain with clipping. No speaker on this board: playback calls return
// ESP_ERR_NOT_SUPPORTED and the shared code checks audio_speaker_ready() before trying.

#include "audio.h"

#include "driver/i2s_pdm.h"
#include "esp_check.h"
#include "esp_log.h"

#include "board_w11.h"

static const char *TAG = "audio";
static i2s_chan_handle_t s_rx;
static float s_dc_x, s_dc_y;  // DC blocker state
static int s_last_peak, s_last_mean;

esp_err_t audio_init(void) {
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan_cfg.dma_desc_num = 8;
  chan_cfg.dma_frame_num = 320;  // 20 ms per DMA buffer
  ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &s_rx), TAG, "i2s_new_channel");
  i2s_pdm_rx_config_t pdm_cfg = {
      .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(LOCI_AUDIO_RATE),
      .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
      .gpio_cfg =
          {
              .clk = BOARD_MIC_CLK,
              .din = BOARD_MIC_DATA,
              .invert_flags = {.clk_inv = false},
          },
  };
  esp_err_t err = i2s_channel_init_pdm_rx_mode(s_rx, &pdm_cfg);
  if (err == ESP_OK) err = i2s_channel_enable(s_rx);
  if (err != ESP_OK) {
    i2s_del_channel(s_rx);
    s_rx = NULL;
    ESP_LOGE(TAG, "PDM microphone init failed: %s", esp_err_to_name(err));
    return err;
  }
  ESP_LOGI(TAG, "PDM mic ready: %d Hz mono, CLK GPIO%d DATA GPIO%d, gain x%d; no speaker", LOCI_AUDIO_RATE,
           BOARD_MIC_CLK, BOARD_MIC_DATA, LOCI_MIC_GAIN);
  return ESP_OK;
}

bool audio_speaker_ready(void) { return false; }
bool audio_mic_ready(void) { return s_rx != NULL; }

esp_err_t audio_play_pcm16(const int16_t *mono, size_t samples) { return ESP_ERR_NOT_SUPPORTED; }

esp_err_t audio_record_pcm16(int16_t *mono, size_t samples) {
  if (!s_rx || samples == 0) return ESP_ERR_INVALID_STATE;
  size_t want = samples * sizeof(int16_t), got = 0;
  while (got < want) {
    size_t n = 0;
    esp_err_t err = i2s_channel_read(s_rx, (uint8_t *)mono + got, want - got, &n, 1000);
    if (err != ESP_OK) return err;
    got += n;
  }
  int peak = 0;
  int64_t sum = 0;
  for (size_t i = 0; i < samples; i++) {
    float x = mono[i];
    float y = x - s_dc_x + 0.995f * s_dc_y;  // high-pass at ~13 Hz
    s_dc_x = x;
    s_dc_y = y;
    float v = y * LOCI_MIC_GAIN;
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    mono[i] = (int16_t)v;
    int a = mono[i] < 0 ? -mono[i] : mono[i];
    sum += a;
    if (a > peak) peak = a;
  }
  s_last_peak = peak;
  s_last_mean = (int)(sum / (int64_t)samples);
  return ESP_OK;
}

void audio_flush_mic(void) {
  static int16_t scratch[320];
  for (int i = 0; i < 6; i++) audio_record_pcm16(scratch, sizeof(scratch) / sizeof(scratch[0]));
}

void audio_last_levels(int *peak, int *mean) {
  if (peak) *peak = s_last_peak;
  if (mean) *mean = s_last_mean;
}

void audio_set_volume(int volume_pct) {}

void audio_beep(float freq_hz, uint32_t ms) {
  // No speaker and the RGB LED shares GPIO48 with the camera: feedback is the console (and the
  // host page) until a buzzer, a haptic motor or the phone over BLE takes over.
  ESP_LOGD(TAG, "beep %.0f Hz %lu ms (no speaker)", (double)freq_hz, (unsigned long)ms);
}
