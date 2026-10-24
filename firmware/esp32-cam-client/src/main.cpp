// LOCI ESP32 thin client
//
// The board does three simple things and nothing clever:
//   1. captures JPEG frames and POSTs them to the host (LOCI_HOST/api/esp32/frame) at LOCI_FPS;
//   2. when the host says audio is waiting (or after a question), GETs raw 16 kHz mono PCM16
//      from LOCI_HOST/api/esp32/audio/next and plays it on the I2S speaker while it downloads;
//   3. with a microphone and a push-to-talk button: records PCM16 while the button is held and
//      POSTs the clip to LOCI_HOST/api/esp32/voice, then plays the spoken answer.
//      Without a microphone the button asks the host "what do you see".
//
// Everything heavy (YOLO, LOCI memory, Whisper, TTS) runs on the host. The same firmware talks
// to a laptop, a Raspberry Pi or a cloud endpoint: only LOCI_HOST changes.
//
// Serial console (115200): type  ask where is my cup | say hello | beep | status | fps 3
//
// Built against the Arduino-ESP32 2.x core (ESP-IDF 4.4 legacy I2S driver) through PlatformIO.

#include <Arduino.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiClient.h>

#ifndef CONFIG_I2S_SUPPRESS_DEPRECATE_WARN
#define CONFIG_I2S_SUPPRESS_DEPRECATE_WARN 1
#endif
#include "driver/i2s.h"
#include "esp_camera.h"

#include "board_config.h"

// ESP-IDF 5 renamed the DMA buffer fields; keep both cores compiling without warnings.
#include "esp_idf_version.h"
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#define LOCI_I2S_DMA(cfg, count, len) \
  do {                                \
    (cfg).dma_desc_num = (count);     \
    (cfg).dma_frame_num = (len);      \
  } while (0)
#else
#define LOCI_I2S_DMA(cfg, count, len) \
  do {                                \
    (cfg).dma_buf_count = (count);    \
    (cfg).dma_buf_len = (len);        \
  } while (0)
#endif

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Create include/secrets.h from include/secrets.h.example (WIFI_SSID, WIFI_PASS, LOCI_HOST)."
#endif

#ifndef LOCI_DEVICE_ID
#define LOCI_DEVICE_ID "esp32"
#endif

static const char* FW_VERSION = "0.1.0";

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------
static bool g_cam_ok = false;
static bool g_spk_ok = false;
static bool g_mic_ok = false;
static uint32_t g_seq = 0;
static uint32_t g_frame_interval_ms = 1000 / LOCI_FPS;
static uint32_t g_last_frame_ms = 0;
static uint32_t g_frames_sent = 0;
static uint32_t g_frames_failed = 0;
static uint32_t g_audio_clips = 0;
static String g_last_labels = "";

static String g_url_frame;
static String g_url_audio;
static String g_url_voice;
static String g_url_ask;

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------
static void ledSet(bool on) {
#if LOCI_LED >= 0
  digitalWrite(LOCI_LED, (on ^ (LOCI_LED_INVERTED != 0)) ? HIGH : LOW);
#else
  (void)on;
#endif
}

// Minimal JSON peeking: the host keeps its replies flat so strstr-style parsing is enough.
static int jsonInt(const String& body, const char* key, int fallback = -1) {
  String needle = String("\"") + key + "\":";
  int i = body.indexOf(needle);
  if (i < 0) return fallback;
  int j = i + needle.length();
  while (j < (int)body.length() && body[j] == ' ') j++;
  if (body.startsWith("true", j)) return 1;
  if (body.startsWith("false", j)) return 0;
  if (body.startsWith("null", j)) return fallback;
  return body.substring(j).toInt();
}

static String jsonString(const String& body, const char* key) {
  String needle = String("\"") + key + "\":\"";
  int i = body.indexOf(needle);
  if (i < 0) return "";
  int j = i + needle.length();
  int k = body.indexOf('"', j);
  return k < 0 ? "" : body.substring(j, k);
}

static String jsonArray(const String& body, const char* key) {
  String needle = String("\"") + key + "\":[";
  int i = body.indexOf(needle);
  if (i < 0) return "";
  int j = i + needle.length();
  int k = body.indexOf(']', j);
  return k < 0 ? "" : body.substring(j, k);
}

static const char* sensorName(uint16_t pid) {
  switch (pid) {
    case 0x26: return "OV2640";
    case 0x3660: return "OV3660";
    case 0x5640: return "OV5640";
    case 0x77: return "OV7725";
    case 0x7FA2: return "OV7670";
    default: return "unknown";
  }
}

// ---------------------------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------------------------
static bool cameraInit() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = LOCI_CAM_Y2;
  c.pin_d1 = LOCI_CAM_Y3;
  c.pin_d2 = LOCI_CAM_Y4;
  c.pin_d3 = LOCI_CAM_Y5;
  c.pin_d4 = LOCI_CAM_Y6;
  c.pin_d5 = LOCI_CAM_Y7;
  c.pin_d6 = LOCI_CAM_Y8;
  c.pin_d7 = LOCI_CAM_Y9;
  c.pin_xclk = LOCI_CAM_XCLK;
  c.pin_pclk = LOCI_CAM_PCLK;
  c.pin_vsync = LOCI_CAM_VSYNC;
  c.pin_href = LOCI_CAM_HREF;
  c.pin_sccb_sda = LOCI_CAM_SIOD;
  c.pin_sccb_scl = LOCI_CAM_SIOC;
  c.pin_pwdn = LOCI_CAM_PWDN;
  c.pin_reset = LOCI_CAM_RESET;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = LOCI_FRAME_SIZE;
  c.jpeg_quality = LOCI_JPEG_QUALITY;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (psramFound()) {
    c.fb_count = 2;
    c.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_DRAM;
    c.frame_size = FRAMESIZE_QVGA;  // VGA JPEG does not fit in internal RAM
    Serial.println("[cam] no PSRAM: forcing QVGA, one frame buffer");
  }
  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("[cam] init failed: 0x%x\n", err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("[cam] sensor PID 0x%04x (%s)\n", s->id.PID, sensorName(s->id.PID));
    if (s->id.PID == 0x3660) {  // OV3660 ships upside down on most modules
      s->set_vflip(s, 1);
      s->set_brightness(s, 1);
      s->set_saturation(s, -2);
    }
#if LOCI_CAM_VFLIP
    s->set_vflip(s, 1);
#endif
#if LOCI_CAM_HMIRROR
    s->set_hmirror(s, 1);
#endif
  }
  return true;
}

// ---------------------------------------------------------------------------------------------
// Speaker (I2S DAC / amplifier such as MAX98357A)
// ---------------------------------------------------------------------------------------------
static bool speakerInit() {
#if LOCI_I2S_BCLK < 0 || LOCI_I2S_LRC < 0 || LOCI_I2S_DOUT < 0
  Serial.println("[spk] no I2S pins configured");
  return false;
#else
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = LOCI_AUDIO_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;  // mono is duplicated to both slots in software
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  LOCI_I2S_DMA(cfg, 8, 512);
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = true;
  cfg.fixed_mclk = 0;
  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = LOCI_I2S_BCLK;
  pins.ws_io_num = LOCI_I2S_LRC;
  pins.data_out_num = LOCI_I2S_DOUT;
  pins.data_in_num = I2S_PIN_NO_CHANGE;
  if (i2s_driver_install(LOCI_SPK_I2S_PORT, &cfg, 0, nullptr) != ESP_OK) {
    Serial.println("[spk] i2s_driver_install failed");
    return false;
  }
  if (i2s_set_pin(LOCI_SPK_I2S_PORT, &pins) != ESP_OK) {
    Serial.println("[spk] i2s_set_pin failed");
    return false;
  }
  i2s_zero_dma_buffer(LOCI_SPK_I2S_PORT);
  Serial.printf("[spk] I2S%d ready: BCLK=%d LRC=%d DIN=%d, %d Hz\n", (int)LOCI_SPK_I2S_PORT,
                LOCI_I2S_BCLK, LOCI_I2S_LRC, LOCI_I2S_DOUT, LOCI_AUDIO_RATE);
  return true;
#endif
}

// Play `n` mono PCM16 samples (blocking until queued into DMA).
static void playMono(const int16_t* mono, size_t n) {
  if (!g_spk_ok || n == 0) return;
  static int16_t stereo[1024];
  size_t i = 0;
  while (i < n) {
    size_t chunk = min(n - i, sizeof(stereo) / sizeof(stereo[0]) / 2);
    for (size_t k = 0; k < chunk; k++) {
      int32_t v = mono[i + k];
#if defined(LOCI_SPK_GAIN)
      v = (int32_t)(v * (LOCI_SPK_GAIN));
      if (v > 32767) v = 32767;
      if (v < -32768) v = -32768;
#endif
      stereo[2 * k] = (int16_t)v;
      stereo[2 * k + 1] = (int16_t)v;
    }
    size_t written = 0;
    i2s_write(LOCI_SPK_I2S_PORT, stereo, chunk * 2 * sizeof(int16_t), &written, portMAX_DELAY);
    i += chunk;
  }
}

static void speakerFlush() {
  if (!g_spk_ok) return;
  delay(80);  // let the last DMA buffers play out before silencing them
  i2s_zero_dma_buffer(LOCI_SPK_I2S_PORT);
}

static void beep(float freq_hz, uint32_t ms, float amp = 0.25f) {
  if (!g_spk_ok) return;
  static int16_t buf[320];
  const size_t total = (size_t)LOCI_AUDIO_RATE * ms / 1000;
  float phase = 0.0f;
  const float step = 2.0f * PI * freq_hz / LOCI_AUDIO_RATE;
  for (size_t done = 0; done < total;) {
    size_t n = min(total - done, sizeof(buf) / sizeof(buf[0]));
    for (size_t k = 0; k < n; k++) {
      buf[k] = (int16_t)(sinf(phase) * amp * 32767.0f);
      phase += step;
      if (phase > 2.0f * PI) phase -= 2.0f * PI;
    }
    playMono(buf, n);
    done += n;
  }
  speakerFlush();
}

// ---------------------------------------------------------------------------------------------
// Microphone (PDM, ESP32-S3 boards such as the XIAO Sense)
// ---------------------------------------------------------------------------------------------
#if LOCI_HAS_MIC
static bool micInit() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
  cfg.sample_rate = LOCI_AUDIO_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  LOCI_I2S_DMA(cfg, 8, 512);
  cfg.use_apll = false;
  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = I2S_PIN_NO_CHANGE;
  pins.ws_io_num = LOCI_MIC_CLK;
  pins.data_in_num = LOCI_MIC_DATA;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  if (i2s_driver_install(LOCI_MIC_I2S_PORT, &cfg, 0, nullptr) != ESP_OK) {
    Serial.println("[mic] i2s_driver_install failed");
    return false;
  }
  if (i2s_set_pin(LOCI_MIC_I2S_PORT, &pins) != ESP_OK) {
    Serial.println("[mic] i2s_set_pin failed");
    return false;
  }
  Serial.printf("[mic] PDM on I2S%d ready: CLK=%d DATA=%d\n", (int)LOCI_MIC_I2S_PORT, LOCI_MIC_CLK,
                LOCI_MIC_DATA);
  return true;
}
#endif

// ---------------------------------------------------------------------------------------------
// Wi-Fi
// ---------------------------------------------------------------------------------------------
static bool wifiEnsure(uint32_t timeout_ms = 15000) {
  if (WiFi.status() == WL_CONNECTED) return true;
  Serial.printf("[wifi] connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // lower latency for the frame stream
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < timeout_ms) {
    delay(250);
    Serial.print('.');
    ledSet((millis() / 250) & 1);
  }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[wifi] not connected");
    ledSet(false);
    return false;
  }
  Serial.printf("[wifi] %s  rssi %d dBm  host %s\n", WiFi.localIP().toString().c_str(), WiFi.RSSI(),
                LOCI_HOST);
  ledSet(true);
  return true;
}

// ---------------------------------------------------------------------------------------------
// Host link
// ---------------------------------------------------------------------------------------------

// Download the next queued clip (waiting up to wait_s on the host) and play it while it streams.
static bool fetchAndPlayAudio(int wait_s) {
  if (!g_spk_ok) return false;
  HTTPClient http;
  http.setTimeout((wait_s + 10) * 1000);
  http.begin(g_url_audio + String(wait_s));
  int code = http.GET();
  if (code != 200) {
    if (code != 204) Serial.printf("[audio] GET -> %d\n", code);
    http.end();
    return false;
  }
  int remaining = http.getSize();  // -1 when chunked
  WiFiClient* stream = http.getStreamPtr();
  static uint8_t buf[2048];
  uint8_t carry = 0;
  bool have_carry = false;
  size_t total = 0;
  uint32_t last_data = millis();
  ledSet(false);
  while (http.connected() && (remaining > 0 || remaining == -1)) {
    size_t avail = stream->available();
    if (avail == 0) {
      if (millis() - last_data > 4000) break;
      delay(1);
      continue;
    }
    size_t off = 0;
    if (have_carry) {
      buf[0] = carry;
      off = 1;
      have_carry = false;
    }
    int n = stream->readBytes(buf + off, min(avail, sizeof(buf) - off));
    if (n <= 0) break;
    size_t len = off + n;
    if (len & 1) {
      carry = buf[len - 1];
      have_carry = true;
      len--;
    }
    playMono((const int16_t*)buf, len / 2);
    total += n;
    if (remaining > 0) remaining -= n;
    last_data = millis();
  }
  http.end();
  speakerFlush();
  ledSet(true);
  g_audio_clips++;
  Serial.printf("[audio] played %u bytes (%.1f s)\n", (unsigned)total,
                total / (2.0f * LOCI_AUDIO_RATE));
  return true;
}

// POST one JPEG; returns true on 200 and reports whether audio is waiting on the host.
static bool postFrame(camera_fb_t* fb, bool& audio_pending) {
  HTTPClient http;
  http.setTimeout(8000);
  http.begin(g_url_frame);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Device-Id", LOCI_DEVICE_ID);
  http.addHeader("X-Seq", String(g_seq));
  http.addHeader("X-Device-Ms", String(millis()));
  http.addHeader("X-Rssi", String(WiFi.RSSI()));
  int code = http.POST(fb->buf, fb->len);
  bool ok = false;
  if (code == 200) {
    String body = http.getString();
    audio_pending = jsonInt(body, "audio_pending", 0) > 0;
    int next_ms = jsonInt(body, "next_ms", 0);
    uint32_t own_ms = 1000 / LOCI_FPS;
    g_frame_interval_ms = (next_ms > (int)own_ms) ? (uint32_t)next_ms : own_ms;
    g_last_labels = jsonArray(body, "labels");
    bool dropped = jsonInt(body, "dropped", 0) == 1;
    Serial.printf("[frame] #%u %ux%u %u B -> %s%s%s\n", g_seq, fb->width, fb->height, fb->len,
                  dropped ? "(dropped by host) " : "", g_last_labels.length() ? g_last_labels.c_str() : "-",
                  audio_pending ? "  [audio waiting]" : "");
    ok = true;
  } else {
    Serial.printf("[frame] #%u POST -> %d %s\n", g_seq, code,
                  code < 0 ? http.errorToString(code).c_str() : "");
  }
  http.end();
  return ok;
}

// Send a push-to-talk clip; the host transcribes, answers and queues the speech.
static void postVoice(const uint8_t* pcm, size_t len) {
  HTTPClient http;
  http.setTimeout(45000);
  http.begin(g_url_voice);
  http.addHeader("Content-Type", "audio/L16; rate=16000");
  http.addHeader("X-Device-Id", LOCI_DEVICE_ID);
  int code = http.POST(const_cast<uint8_t*>(pcm), len);
  if (code == 200) {
    String body = http.getString();
    Serial.printf("[voice] heard: \"%s\"\n[voice] answer: %s\n", jsonString(body, "transcription").c_str(),
                  jsonString(body, "answer").c_str());
    bool queued = jsonInt(body, "audio_queued", 0) == 1;
    http.end();
    if (queued) fetchAndPlayAudio(5);
    return;
  }
  Serial.printf("[voice] POST -> %d\n", code);
  http.end();
}

// Typed question (serial console or button without a microphone).
static void askHost(const String& text) {
  HTTPClient http;
  http.setTimeout(30000);
  http.begin(g_url_ask);
  http.addHeader("Content-Type", "application/json");
  String escaped = text;
  escaped.replace("\\", "\\\\");
  escaped.replace("\"", "\\\"");
  String payload = String("{\"text\":\"") + escaped + "\",\"device\":\"" + LOCI_DEVICE_ID + "\"}";
  int code = http.POST(payload);
  if (code == 200) {
    String body = http.getString();
    Serial.printf("[ask] %s -> %s\n", text.c_str(), jsonString(body, "answer").c_str());
    bool queued = jsonInt(body, "audio_queued", 0) == 1;
    http.end();
    if (queued) fetchAndPlayAudio(5);
    return;
  }
  Serial.printf("[ask] POST -> %d\n", code);
  http.end();
}

static void sayHost(const String& text) {
  HTTPClient http;
  http.setTimeout(30000);
  http.begin(String(LOCI_HOST) + "/api/esp32/say");
  http.addHeader("Content-Type", "application/json");
  String escaped = text;
  escaped.replace("\\", "\\\\");
  escaped.replace("\"", "\\\"");
  int code = http.POST(String("{\"text\":\"") + escaped + "\",\"device\":\"" + LOCI_DEVICE_ID + "\"}");
  http.end();
  if (code == 200) fetchAndPlayAudio(5);
  else Serial.printf("[say] POST -> %d\n", code);
}

// ---------------------------------------------------------------------------------------------
// Push-to-talk
// ---------------------------------------------------------------------------------------------
static bool buttonDown() {
#if LOCI_BTN_PTT >= 0
  return digitalRead(LOCI_BTN_PTT) == LOW;
#else
  return false;
#endif
}

#if LOCI_HAS_MIC
static uint8_t* g_ptt_buf = nullptr;
static size_t g_ptt_cap = 0;

static void recordAndAsk() {
  if (!g_mic_ok || g_ptt_buf == nullptr) return;
  delay(30);  // debounce
  if (!buttonDown()) return;
  Serial.println("[ptt] recording... release the button to ask");
  ledSet(false);
  size_t len = 0;
  static int16_t chunk[512];
  size_t got = 0;
  // Drop the first ~60 ms: PDM mics need a moment to settle after the clock starts.
  i2s_read(LOCI_MIC_I2S_PORT, chunk, sizeof(chunk), &got, pdMS_TO_TICKS(100));
  i2s_read(LOCI_MIC_I2S_PORT, chunk, sizeof(chunk), &got, pdMS_TO_TICKS(100));
  uint32_t t0 = millis();
  while (buttonDown() && len + sizeof(chunk) <= g_ptt_cap) {
    if (i2s_read(LOCI_MIC_I2S_PORT, chunk, sizeof(chunk), &got, pdMS_TO_TICKS(200)) != ESP_OK) break;
    size_t n = got / 2;
    int16_t* out = (int16_t*)(g_ptt_buf + len);
    for (size_t k = 0; k < n; k++) {
      int32_t v = (int32_t)chunk[k] * LOCI_MIC_GAIN;
      if (v > 32767) v = 32767;
      if (v < -32768) v = -32768;
      out[k] = (int16_t)v;
    }
    len += n * 2;
    if (millis() - t0 > (uint32_t)LOCI_PTT_MAX_S * 1000) break;
  }
  ledSet(true);
  float seconds = len / (2.0f * LOCI_AUDIO_RATE);
  Serial.printf("[ptt] %.1f s captured\n", seconds);
  while (buttonDown()) delay(10);  // wait for release (max length reached while held)
  if (seconds < 0.3f) {
    beep(220.0f, 80);
    return;
  }
  beep(880.0f, 60);  // "got it"
  postVoice(g_ptt_buf, len);
}
#endif

// ---------------------------------------------------------------------------------------------
// Serial console
// ---------------------------------------------------------------------------------------------
static void printStatus() {
  Serial.printf("[status] fw %s board %s cam %d spk %d mic %d wifi %d rssi %d ip %s\n", FW_VERSION,
                LOCI_BOARD_NAME, g_cam_ok, g_spk_ok, g_mic_ok, WiFi.status() == WL_CONNECTED, WiFi.RSSI(),
                WiFi.localIP().toString().c_str());
  Serial.printf("[status] frames sent %u failed %u interval %u ms audio clips %u heap %u psram %u\n",
                g_frames_sent, g_frames_failed, g_frame_interval_ms, g_audio_clips, ESP.getFreeHeap(),
                ESP.getFreePsram());
  Serial.printf("[status] last labels: %s\n", g_last_labels.length() ? g_last_labels.c_str() : "-");
}

static void handleSerial() {
  static String line;
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch != '\n') {
      line += ch;
      if (line.length() > 200) line = "";
      continue;
    }
    line.trim();
    if (line.length() == 0) continue;
    if (line == "status") {
      printStatus();
    } else if (line == "beep") {
      beep(660.0f, 150);
    } else if (line.startsWith("ask ")) {
      askHost(line.substring(4));
    } else if (line.startsWith("say ")) {
      sayHost(line.substring(4));
    } else if (line.startsWith("fps ")) {
      int fps = line.substring(4).toInt();
      if (fps > 0 && fps <= 15) {
        g_frame_interval_ms = 1000 / fps;
        Serial.printf("[cfg] interval %u ms\n", g_frame_interval_ms);
      }
    } else if (line == "audio") {
      fetchAndPlayAudio(2);
    } else {
      Serial.println("commands: status | beep | audio | ask <question> | say <text> | fps <n>");
    }
    line = "";
  }
}

// ---------------------------------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(400);
  Serial.printf("\n=== LOCI ESP32 thin client v%s — %s ===\n", FW_VERSION, LOCI_BOARD_NAME);
  Serial.printf("chip %s rev %d, %u MB flash, PSRAM %s (%u B free), device id \"%s\"\n",
                ESP.getChipModel(), ESP.getChipRevision(), ESP.getFlashChipSize() / (1024 * 1024),
                psramFound() ? "yes" : "no", ESP.getFreePsram(), LOCI_DEVICE_ID);
  Serial.printf("camera XCLK=%d SIOD=%d SIOC=%d VSYNC=%d HREF=%d PCLK=%d  speaker BCLK=%d LRC=%d DIN=%d  "
                "mic CLK=%d DATA=%d  button=%d\n",
                LOCI_CAM_XCLK, LOCI_CAM_SIOD, LOCI_CAM_SIOC, LOCI_CAM_VSYNC, LOCI_CAM_HREF, LOCI_CAM_PCLK,
                LOCI_I2S_BCLK, LOCI_I2S_LRC, LOCI_I2S_DOUT, LOCI_MIC_CLK, LOCI_MIC_DATA, LOCI_BTN_PTT);

#if LOCI_LED >= 0
  pinMode(LOCI_LED, OUTPUT);
  ledSet(false);
#endif
#if LOCI_BTN_PTT >= 0
  pinMode(LOCI_BTN_PTT, INPUT_PULLUP);
#endif

  g_url_frame = String(LOCI_HOST) + "/api/esp32/frame?device=" + LOCI_DEVICE_ID;
  g_url_audio = String(LOCI_HOST) + "/api/esp32/audio/next?device=" + LOCI_DEVICE_ID + "&wait=";
  g_url_voice = String(LOCI_HOST) + "/api/esp32/voice?device=" + LOCI_DEVICE_ID + "&rate=" + LOCI_AUDIO_RATE;
  g_url_ask = String(LOCI_HOST) + "/api/esp32/ask";

  g_cam_ok = cameraInit();
  g_spk_ok = speakerInit();
#if LOCI_HAS_MIC
  g_mic_ok = micInit();
  if (g_mic_ok) {
    g_ptt_cap = (size_t)LOCI_AUDIO_RATE * 2 * LOCI_PTT_MAX_S;
    g_ptt_buf = (uint8_t*)(psramFound() ? ps_malloc(g_ptt_cap) : malloc(g_ptt_cap));
    if (g_ptt_buf == nullptr) {
      Serial.println("[ptt] could not allocate the recording buffer; push-to-talk disabled");
      g_mic_ok = false;
    }
  }
#endif
  if (g_spk_ok) {
    beep(523.25f, 90);
    beep(659.25f, 90);
  }
  wifiEnsure();
  if (g_spk_ok && WiFi.status() == WL_CONNECTED) beep(783.99f, 120);
  printStatus();
}

void loop() {
  handleSerial();
  if (!wifiEnsure()) {
    delay(1000);
    return;
  }

  if (buttonDown()) {
#if LOCI_HAS_MIC
    if (g_mic_ok) {
      recordAndAsk();
      return;
    }
#endif
    delay(30);
    if (buttonDown()) {
      while (buttonDown()) delay(10);
      askHost("what do you see");
    }
    return;
  }

  uint32_t now = millis();
  if (g_cam_ok && now - g_last_frame_ms >= g_frame_interval_ms) {
    g_last_frame_ms = now;
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb == nullptr) {
      Serial.println("[cam] capture failed");
      g_frames_failed++;
      delay(100);
      return;
    }
    g_seq++;
    bool audio_pending = false;
    if (postFrame(fb, audio_pending)) g_frames_sent++;
    else g_frames_failed++;
    esp_camera_fb_return(fb);
    if (audio_pending) fetchAndPlayAudio(0);
  }
  delay(5);
}
