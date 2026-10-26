// ES8311 through esp_codec_dev: 16 kHz mono PCM16 in both directions.
// Configuration values follow the Waveshare ESP32-P4-NANO BSP (bsp_audio_codec_*_init).

#include "audio.h"

#include <math.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "esp_log.h"

#include "board_p4_nano.h"

static const char *TAG = "audio";

static i2s_chan_handle_t s_tx;
static i2s_chan_handle_t s_rx;
static esp_codec_dev_handle_t s_dev;

esp_err_t audio_init(i2c_master_bus_handle_t i2c_bus, int volume_pct) {
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(BOARD_I2S_PORT, I2S_ROLE_MASTER);
  chan_cfg.auto_clear = true;
  ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, &s_tx, &s_rx), TAG, "i2s_new_channel");

  i2s_std_config_t std_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(LOCI_AUDIO_RATE),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
      .gpio_cfg =
          {
              .mclk = BOARD_I2S_MCLK,
              .bclk = BOARD_I2S_BCLK,
              .ws = BOARD_I2S_WS,
              .dout = BOARD_I2S_DOUT,
              .din = BOARD_I2S_DIN,
              .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
          },
  };
  ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_tx, &std_cfg), TAG, "i2s tx init");
  ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx, &std_cfg), TAG, "i2s rx init");
  ESP_RETURN_ON_ERROR(i2s_channel_enable(s_tx), TAG, "i2s tx enable");
  ESP_RETURN_ON_ERROR(i2s_channel_enable(s_rx), TAG, "i2s rx enable");

  audio_codec_i2s_cfg_t i2s_cfg = {.port = BOARD_I2S_PORT, .tx_handle = s_tx, .rx_handle = s_rx};
  const audio_codec_data_if_t *data_if = audio_codec_new_i2s_data(&i2s_cfg);
  ESP_RETURN_ON_FALSE(data_if, ESP_FAIL, TAG, "i2s data interface");

  audio_codec_i2c_cfg_t i2c_cfg = {
      .port = BOARD_I2C_PORT, .addr = ES8311_CODEC_DEFAULT_ADDR, .bus_handle = i2c_bus};
  const audio_codec_ctrl_if_t *ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
  ESP_RETURN_ON_FALSE(ctrl_if, ESP_FAIL, TAG, "i2c control interface");
  const audio_codec_gpio_if_t *gpio_if = audio_codec_new_gpio();

  es8311_codec_cfg_t es_cfg = {
      .ctrl_if = ctrl_if,
      .gpio_if = gpio_if,
      .codec_mode = ESP_CODEC_DEV_TYPE_IN_OUT,
      .pa_pin = BOARD_PA_PIN,
      .pa_reverted = false,
      .master_mode = false,
      .use_mclk = true,
      .digital_mic = false,
      .invert_mclk = false,
      .invert_sclk = false,
      .hw_gain = {.pa_voltage = 5.0, .codec_dac_voltage = 3.3},
  };
  const audio_codec_if_t *codec_if = es8311_codec_new(&es_cfg);
  ESP_RETURN_ON_FALSE(codec_if, ESP_FAIL, TAG, "ES8311 not found on I2C 0x%02x", ES8311_CODEC_DEFAULT_ADDR);

  esp_codec_dev_cfg_t dev_cfg = {
      .dev_type = ESP_CODEC_DEV_TYPE_IN_OUT, .codec_if = codec_if, .data_if = data_if};
  s_dev = esp_codec_dev_new(&dev_cfg);
  ESP_RETURN_ON_FALSE(s_dev, ESP_FAIL, TAG, "esp_codec_dev_new");

  esp_codec_dev_sample_info_t fs = {
      .bits_per_sample = 16,
      .channel = 1,
      .channel_mask = ESP_CODEC_DEV_MAKE_CHANNEL_MASK(0),
      .sample_rate = LOCI_AUDIO_RATE,
  };
  ESP_RETURN_ON_FALSE(esp_codec_dev_open(s_dev, &fs) == ESP_CODEC_DEV_OK, ESP_FAIL, TAG, "codec open");
  esp_codec_dev_set_out_vol(s_dev, volume_pct);
  esp_codec_dev_set_in_gain(s_dev, LOCI_MIC_GAIN_DB);
  ESP_LOGI(TAG, "ES8311 ready: %d Hz mono, volume %d%%, mic gain %.0f dB, PA on GPIO%d",
           LOCI_AUDIO_RATE, volume_pct, (double)LOCI_MIC_GAIN_DB, BOARD_PA_PIN);
  return ESP_OK;
}

bool audio_speaker_ready(void) { return s_dev != NULL; }
bool audio_mic_ready(void) { return s_dev != NULL; }

esp_err_t audio_play_pcm16(const int16_t *mono, size_t samples) {
  if (!s_dev || samples == 0) return ESP_ERR_INVALID_STATE;
  int rc = esp_codec_dev_write(s_dev, (void *)mono, samples * sizeof(int16_t));
  return rc == ESP_CODEC_DEV_OK ? ESP_OK : ESP_FAIL;
}

esp_err_t audio_record_pcm16(int16_t *mono, size_t samples) {
  if (!s_dev || samples == 0) return ESP_ERR_INVALID_STATE;
  int rc = esp_codec_dev_read(s_dev, mono, samples * sizeof(int16_t));
  return rc == ESP_CODEC_DEV_OK ? ESP_OK : ESP_FAIL;
}

void audio_flush_mic(void) {
  static int16_t scratch[320];
  for (int i = 0; i < 6; i++) audio_record_pcm16(scratch, sizeof(scratch) / sizeof(scratch[0]));
}

void audio_set_volume(int volume_pct) {
  if (s_dev) esp_codec_dev_set_out_vol(s_dev, volume_pct);
}

void audio_beep(float freq_hz, uint32_t ms) {
  if (!s_dev) return;
  static int16_t buf[320];
  const size_t total = (size_t)LOCI_AUDIO_RATE * ms / 1000;
  const float step = 2.0f * (float)M_PI * freq_hz / LOCI_AUDIO_RATE;
  float phase = 0.0f;
  for (size_t done = 0; done < total;) {
    size_t n = total - done;
    if (n > sizeof(buf) / sizeof(buf[0])) n = sizeof(buf) / sizeof(buf[0]);
    for (size_t k = 0; k < n; k++) {
      // short fade in/out keeps the amplifier from clicking
      float env = 1.0f;
      size_t pos = done + k;
      if (pos < 160) env = pos / 160.0f;
      if (total - pos < 160) env = (total - pos) / 160.0f;
      buf[k] = (int16_t)(sinf(phase) * 0.3f * env * 32767.0f);
      phase += step;
      if (phase > 2.0f * (float)M_PI) phase -= 2.0f * (float)M_PI;
    }
    audio_play_pcm16(buf, n);
    done += n;
  }
}
