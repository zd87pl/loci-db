// MIPI-CSI camera → ISP → hardware JPEG on the ESP32-P4, through esp_video's V4L2-style devices.
// Modelled on esp-video-components/esp_video/examples/image_storage (Apache-2.0, Espressif).

#include "camera.h"

#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "esp_check.h"
#include "esp_log.h"
#include "esp_video_device.h"
#include "esp_video_init.h"
#include "linux/videodev2.h"

#include "board_p4_nano.h"

static const char *TAG = "camera";

#define CAP_BUFFER_COUNT 2
#define SKIP_STARTUP_FRAMES 2

static int s_cap_fd = -1;
static int s_m2m_fd = -1;
static uint8_t *s_cap_buf[CAP_BUFFER_COUNT];
static uint8_t *s_jpeg_buf;
static uint32_t s_width;
static uint32_t s_height;
static bool s_frame_out;

static void log_device(const char *path, const struct v4l2_capability *cap) {
  ESP_LOGI(TAG, "%s: driver %s, card %s, version %u.%u.%u", path, cap->driver, cap->card,
           (unsigned)(cap->version >> 16), (unsigned)((cap->version >> 8) & 0xff),
           (unsigned)(cap->version & 0xff));
}

static esp_err_t set_control(int fd, uint32_t ctrl_class, uint32_t id, int32_t value) {
  struct v4l2_ext_controls controls = {0};
  struct v4l2_ext_control control[1] = {0};
  controls.ctrl_class = ctrl_class;
  controls.count = 1;
  controls.controls = control;
  control[0].id = id;
  control[0].value = value;
  if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &controls) != 0) {
    ESP_LOGW(TAG, "control 0x%08lx not accepted", (unsigned long)id);
    return ESP_FAIL;
  }
  return ESP_OK;
}

esp_err_t camera_init(i2c_master_bus_handle_t i2c_bus, int jpeg_quality) {
  esp_video_init_csi_config_t csi = {
      .sccb_config =
          {
              .init_sccb = false,
              .i2c_handle = i2c_bus,
              .freq = BOARD_CAM_SCCB_FREQ,
          },
      .reset_pin = BOARD_CAM_RESET,
      .pwdn_pin = BOARD_CAM_PWDN,
  };
  esp_video_init_config_t cfg = {.csi = &csi};
  ESP_RETURN_ON_ERROR(esp_video_init(&cfg), TAG, "esp_video_init failed (is the camera plugged in?)");

  struct v4l2_capability cap;
  s_cap_fd = open(ESP_VIDEO_MIPI_CSI_DEVICE_NAME, O_RDONLY);
  ESP_RETURN_ON_FALSE(s_cap_fd >= 0, ESP_FAIL, TAG, "open %s failed", ESP_VIDEO_MIPI_CSI_DEVICE_NAME);
  ESP_RETURN_ON_FALSE(ioctl(s_cap_fd, VIDIOC_QUERYCAP, &cap) == 0, ESP_FAIL, TAG, "QUERYCAP cam");
  log_device(ESP_VIDEO_MIPI_CSI_DEVICE_NAME, &cap);

  s_m2m_fd = open(ESP_VIDEO_JPEG_DEVICE_NAME, O_RDONLY);
  ESP_RETURN_ON_FALSE(s_m2m_fd >= 0, ESP_FAIL, TAG, "open %s failed", ESP_VIDEO_JPEG_DEVICE_NAME);
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_QUERYCAP, &cap) == 0, ESP_FAIL, TAG, "QUERYCAP jpeg");
  log_device(ESP_VIDEO_JPEG_DEVICE_NAME, &cap);
  set_control(s_m2m_fd, V4L2_CID_JPEG_CLASS, V4L2_CID_JPEG_COMPRESSION_QUALITY, jpeg_quality);

  // Sensor resolution comes from the sensor's default format (menuconfig); pick a pixel format
  // the JPEG encoder accepts.
  struct v4l2_format fmt = {0};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ESP_RETURN_ON_FALSE(ioctl(s_cap_fd, VIDIOC_G_FMT, &fmt) == 0, ESP_FAIL, TAG, "G_FMT");
  s_width = fmt.fmt.pix.width;
  s_height = fmt.fmt.pix.height;

  const uint32_t jpeg_inputs[] = {V4L2_PIX_FMT_RGB565, V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_RGB24,
                                  V4L2_PIX_FMT_GREY};
  uint32_t capture_fmt = 0;
  for (int index = 0; !capture_fmt; index++) {
    struct v4l2_fmtdesc desc = {.index = index, .type = V4L2_BUF_TYPE_VIDEO_CAPTURE};
    if (ioctl(s_cap_fd, VIDIOC_ENUM_FMT, &desc) != 0) break;
    for (size_t i = 0; i < sizeof(jpeg_inputs) / sizeof(jpeg_inputs[0]); i++) {
      if (jpeg_inputs[i] == desc.pixelformat) {
        capture_fmt = desc.pixelformat;
        break;
      }
    }
  }
  ESP_RETURN_ON_FALSE(capture_fmt != 0, ESP_ERR_NOT_SUPPORTED, TAG,
                      "sensor offers no pixel format the JPEG encoder accepts");
  ESP_LOGI(TAG, "capture %lux%lu, pixel format %c%c%c%c, JPEG quality %d", (unsigned long)s_width,
           (unsigned long)s_height, (char)(capture_fmt & 0xff), (char)((capture_fmt >> 8) & 0xff),
           (char)((capture_fmt >> 16) & 0xff), (char)((capture_fmt >> 24) & 0xff), jpeg_quality);

  // Camera capture stream (mmap buffers)
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = s_width;
  fmt.fmt.pix.height = s_height;
  fmt.fmt.pix.pixelformat = capture_fmt;
  ESP_RETURN_ON_FALSE(ioctl(s_cap_fd, VIDIOC_S_FMT, &fmt) == 0, ESP_FAIL, TAG, "S_FMT cam");

  struct v4l2_requestbuffers req = {0};
  req.count = CAP_BUFFER_COUNT;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  ESP_RETURN_ON_FALSE(ioctl(s_cap_fd, VIDIOC_REQBUFS, &req) == 0, ESP_FAIL, TAG, "REQBUFS cam");
  for (int i = 0; i < CAP_BUFFER_COUNT; i++) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    ESP_RETURN_ON_FALSE(ioctl(s_cap_fd, VIDIOC_QUERYBUF, &buf) == 0, ESP_FAIL, TAG, "QUERYBUF cam");
    s_cap_buf[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, s_cap_fd, buf.m.offset);
    ESP_RETURN_ON_FALSE(s_cap_buf[i] != NULL, ESP_ERR_NO_MEM, TAG, "mmap cam buffer");
    ESP_RETURN_ON_FALSE(ioctl(s_cap_fd, VIDIOC_QBUF, &buf) == 0, ESP_FAIL, TAG, "QBUF cam");
  }

  // Encoder input (user pointers into the camera buffers) and output (one mmap JPEG buffer)
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt.fmt.pix.width = s_width;
  fmt.fmt.pix.height = s_height;
  fmt.fmt.pix.pixelformat = capture_fmt;
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_S_FMT, &fmt) == 0, ESP_FAIL, TAG, "S_FMT jpeg in");
  memset(&req, 0, sizeof(req));
  req.count = 1;
  req.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  req.memory = V4L2_MEMORY_USERPTR;
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_REQBUFS, &req) == 0, ESP_FAIL, TAG, "REQBUFS jpeg in");

  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = s_width;
  fmt.fmt.pix.height = s_height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_JPEG;
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_S_FMT, &fmt) == 0, ESP_FAIL, TAG, "S_FMT jpeg out");
  memset(&req, 0, sizeof(req));
  req.count = 1;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_REQBUFS, &req) == 0, ESP_FAIL, TAG, "REQBUFS jpeg out");
  struct v4l2_buffer jbuf = {0};
  jbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  jbuf.memory = V4L2_MEMORY_MMAP;
  jbuf.index = 0;
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_QUERYBUF, &jbuf) == 0, ESP_FAIL, TAG, "QUERYBUF jpeg");
  s_jpeg_buf = mmap(NULL, jbuf.length, PROT_READ | PROT_WRITE, MAP_SHARED, s_m2m_fd, jbuf.m.offset);
  ESP_RETURN_ON_FALSE(s_jpeg_buf != NULL, ESP_ERR_NO_MEM, TAG, "mmap jpeg buffer");
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_QBUF, &jbuf) == 0, ESP_FAIL, TAG, "QBUF jpeg");

  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_STREAMON, &type) == 0, ESP_FAIL, TAG, "STREAMON jpeg out");
  type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  ESP_RETURN_ON_FALSE(ioctl(s_m2m_fd, VIDIOC_STREAMON, &type) == 0, ESP_FAIL, TAG, "STREAMON jpeg in");
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ESP_RETURN_ON_FALSE(ioctl(s_cap_fd, VIDIOC_STREAMON, &type) == 0, ESP_FAIL, TAG, "STREAMON cam");

  for (int i = 0; i < SKIP_STARTUP_FRAMES; i++) {  // let exposure settle
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(s_cap_fd, VIDIOC_DQBUF, &buf) == 0) ioctl(s_cap_fd, VIDIOC_QBUF, &buf);
  }
  ESP_LOGI(TAG, "streaming");
  return ESP_OK;
}

esp_err_t camera_capture_jpeg(uint8_t **jpeg, size_t *len, uint32_t *width, uint32_t *height) {
  if (s_frame_out) return ESP_ERR_INVALID_STATE;
  struct v4l2_buffer cap_buf = {0};
  cap_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  cap_buf.memory = V4L2_MEMORY_MMAP;
  ESP_RETURN_ON_FALSE(ioctl(s_cap_fd, VIDIOC_DQBUF, &cap_buf) == 0, ESP_FAIL, TAG, "DQBUF cam");

  struct v4l2_buffer in_buf = {0};
  in_buf.index = 0;
  in_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  in_buf.memory = V4L2_MEMORY_USERPTR;
  in_buf.m.userptr = (unsigned long)s_cap_buf[cap_buf.index];
  in_buf.length = cap_buf.bytesused;
  if (ioctl(s_m2m_fd, VIDIOC_QBUF, &in_buf) != 0) {
    ioctl(s_cap_fd, VIDIOC_QBUF, &cap_buf);
    ESP_LOGE(TAG, "QBUF jpeg in failed");
    return ESP_FAIL;
  }

  struct v4l2_buffer out_buf = {0};
  out_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  out_buf.memory = V4L2_MEMORY_MMAP;
  if (ioctl(s_m2m_fd, VIDIOC_DQBUF, &out_buf) != 0) {
    ioctl(s_cap_fd, VIDIOC_QBUF, &cap_buf);
    ESP_LOGE(TAG, "DQBUF jpeg out failed");
    return ESP_FAIL;
  }
  ioctl(s_cap_fd, VIDIOC_QBUF, &cap_buf);  // camera buffer back to the sensor
  ioctl(s_m2m_fd, VIDIOC_DQBUF, &in_buf);  // encoder is done with the input

  *jpeg = s_jpeg_buf;
  *len = out_buf.bytesused;
  if (width) *width = s_width;
  if (height) *height = s_height;
  s_frame_out = true;
  return ESP_OK;
}

void camera_release(void) {
  if (!s_frame_out) return;
  struct v4l2_buffer out_buf = {0};
  out_buf.index = 0;
  out_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  out_buf.memory = V4L2_MEMORY_MMAP;
  ioctl(s_m2m_fd, VIDIOC_QBUF, &out_buf);
  s_frame_out = false;
}

uint32_t camera_width(void) { return s_width; }
uint32_t camera_height(void) { return s_height; }
