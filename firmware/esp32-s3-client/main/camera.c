// DVP camera on the ESP32-S3 (LCD_CAM peripheral) through esp32-camera. The sensor produces JPEG
// itself, so a capture is a frame buffer handed out as is; two buffers in PSRAM with "grab latest"
// keep the newest frame ready while the previous one is uploaded.

#include "camera.h"

#include <string.h>

#include "esp_camera.h"
#include "esp_camera_af.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "board_w11.h"

static const char *TAG = "camera";
static camera_fb_t *s_fb;
static uint32_t s_width, s_height;
static const char *s_sensor = "none";
static bool s_af;  // OV5640 autofocus firmware loaded
static const char *s_size_name = "?";

typedef struct {
  const char *name;
  framesize_t size;
} named_size_t;

static const named_size_t k_sizes[] = {
    {"qvga", FRAMESIZE_QVGA}, {"vga", FRAMESIZE_VGA},   {"svga", FRAMESIZE_SVGA}, {"xga", FRAMESIZE_XGA},
    {"hd", FRAMESIZE_HD},     {"sxga", FRAMESIZE_SXGA}, {"uxga", FRAMESIZE_UXGA}, {"fhd", FRAMESIZE_FHD},
    {"qxga", FRAMESIZE_QXGA},
};

esp_err_t camera_init(int jpeg_quality, framesize_t frame_size) {
  camera_config_t cfg = {
      .pin_pwdn = BOARD_CAM_PWDN,
      .pin_reset = BOARD_CAM_RESET,
      .pin_xclk = BOARD_CAM_XCLK,
      .pin_sccb_sda = BOARD_CAM_SIOD,
      .pin_sccb_scl = BOARD_CAM_SIOC,
      .pin_d7 = BOARD_CAM_D7,
      .pin_d6 = BOARD_CAM_D6,
      .pin_d5 = BOARD_CAM_D5,
      .pin_d4 = BOARD_CAM_D4,
      .pin_d3 = BOARD_CAM_D3,
      .pin_d2 = BOARD_CAM_D2,
      .pin_d1 = BOARD_CAM_D1,
      .pin_d0 = BOARD_CAM_D0,
      .pin_vsync = BOARD_CAM_VSYNC,
      .pin_href = BOARD_CAM_HREF,
      .pin_pclk = BOARD_CAM_PCLK,
      .xclk_freq_hz = BOARD_CAM_XCLK_HZ,
      .ledc_timer = LEDC_TIMER_0,
      .ledc_channel = LEDC_CHANNEL_0,
      .pixel_format = PIXFORMAT_JPEG,
      .frame_size = frame_size,
      .jpeg_quality = jpeg_quality,
      .fb_count = 2,
      .fb_location = CAMERA_FB_IN_PSRAM,
      .grab_mode = CAMERA_GRAB_LATEST,
  };
  for (size_t i = 0; i < sizeof(k_sizes) / sizeof(k_sizes[0]); i++) {
    if (k_sizes[i].size == frame_size) s_size_name = k_sizes[i].name;
  }
  esp_err_t err = esp_camera_init(&cfg);
  ESP_RETURN_ON_ERROR(err, TAG, "esp_camera_init (is the camera seated in the expansion board?)");
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    camera_sensor_info_t *info = esp_camera_sensor_get_info(&s->id);
    s_sensor = info ? info->name : "unknown";
    ESP_LOGI(TAG, "%s (PID 0x%04x) on DVP, XCLK %d MHz, JPEG quality %d", s_sensor, s->id.PID,
             BOARD_CAM_XCLK_HZ / 1000000, jpeg_quality);
    if (esp_camera_af_is_supported(s)) {
      // Continuous autofocus: the OV5640 runs it on chip once its AF firmware is loaded. A module
      // without a focus motor accepts the firmware and simply never moves the lens.
      esp_camera_af_config_t af = {.mode = ESP_CAMERA_AF_MODE_AUTO, .timeout_ms = 2000};
      esp_err_t af_err = esp_camera_af_init(s, &af);
      s_af = af_err == ESP_OK;
      ESP_LOGI(TAG, "autofocus %s", s_af ? "running (continuous)" : esp_err_to_name(af_err));
    }
  }
  // Let auto exposure / white balance settle: the first frames come out dark and green.
  int64_t t0 = esp_timer_get_time();
  for (int i = 0; i < 6; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
      s_width = fb->width;
      s_height = fb->height;
      esp_camera_fb_return(fb);
    }
  }
  ESP_LOGI(TAG, "streaming %lux%lu JPEG (warm-up %lld ms)", (unsigned long)s_width, (unsigned long)s_height,
           (esp_timer_get_time() - t0) / 1000);
  return ESP_OK;
}

esp_err_t camera_capture_jpeg(uint8_t **jpeg, size_t *len, uint32_t *width, uint32_t *height) {
  if (s_fb) camera_release();  // a missed release must not leak the buffer
  s_fb = esp_camera_fb_get();
  if (!s_fb) return ESP_FAIL;
  if (s_fb->format != PIXFORMAT_JPEG || s_fb->len < 4) {
    camera_release();
    return ESP_ERR_INVALID_RESPONSE;
  }
  s_width = s_fb->width;
  s_height = s_fb->height;
  *jpeg = s_fb->buf;
  *len = s_fb->len;
  if (width) *width = s_width;
  if (height) *height = s_height;
  return ESP_OK;
}

void camera_release(void) {
  if (s_fb) {
    esp_camera_fb_return(s_fb);
    s_fb = NULL;
  }
}

bool camera_set_frame_size(const char *name) {
  sensor_t *s = esp_camera_sensor_get();
  if (!s || !name) return false;
  for (size_t i = 0; i < sizeof(k_sizes) / sizeof(k_sizes[0]); i++) {
    if (!strcmp(name, k_sizes[i].name)) {
      if (s->set_framesize(s, k_sizes[i].size) != 0) return false;
      s_size_name = k_sizes[i].name;
      // drop the frames already queued at the old size
      for (int k = 0; k < 3; k++) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
      }
      return true;
    }
  }
  return false;
}

void camera_af_report(bool trigger) {
  sensor_t *s = esp_camera_sensor_get();
  if (!s || !s_af) {
    ESP_LOGW(TAG, "[af] autofocus not available on this sensor/module");
    return;
  }
  esp_camera_af_status_t st = {0};
  int64_t t0 = esp_timer_get_time();
  esp_err_t err = trigger ? esp_camera_af_trigger(s) : ESP_OK;
  if (err == ESP_OK) err = trigger ? esp_camera_af_wait(s, 0, &st) : esp_camera_af_get_status(s, &st);
  ESP_LOGI(TAG, "[af] %s: %s, focused %d busy %d raw 0x%02x (%lld ms)", trigger ? "single shot" : "status",
           esp_err_to_name(err), st.focused, st.busy, st.raw, (esp_timer_get_time() - t0) / 1000);
}

uint32_t camera_width(void) { return s_width; }
uint32_t camera_height(void) { return s_height; }
const char *camera_sensor_name(void) { return s_sensor; }
const char *camera_frame_size_name(void) { return s_size_name; }
