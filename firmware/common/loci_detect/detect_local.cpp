// ESP-DL COCO detector on the board (S3 and P4, see detect_local.h).

#include "detect_local.h"

#include <stdio.h>
#include <string.h>

#include "coco_detect.hpp"
#include "esp_heap_caps.h"
#include "sdkconfig.h"
#if CONFIG_SOC_JPEG_CODEC_SUPPORTED
#include "dl_image_jpeg.hpp"
#else
#include "esp_jpeg_dec.h"
#endif
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "detect";
static COCODetect *s_detect;

static const char *k_coco[] = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",     "bus",          "train",
    "truck",         "boat",         "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",          "cat",          "dog",           "horse",         "sheep",        "cow",          "elephant",
    "bear",          "zebra",        "giraffe",       "backpack",      "umbrella",     "handbag",      "tie",
    "suitcase",      "frisbee",      "skis",          "snowboard",     "sports ball",  "kite",         "baseball bat",
    "baseball glove", "skateboard",  "surfboard",     "tennis racket", "bottle",       "wine glass",   "cup",
    "fork",          "knife",        "spoon",         "bowl",          "banana",       "apple",        "sandwich",
    "orange",        "broccoli",     "carrot",        "hot dog",       "pizza",        "donut",        "cake",
    "chair",         "couch",        "potted plant",  "bed",           "dining table", "toilet",       "tv",
    "laptop",        "mouse",        "remote",        "keyboard",      "cell phone",   "microwave",    "oven",
    "toaster",       "sink",         "refrigerator",  "book",          "clock",        "vase",         "scissors",
    "teddy bear",    "hair drier",   "toothbrush",
};

bool detect_local_init(void) {
  if (s_detect) return true;
  int64_t t0 = esp_timer_get_time();
  s_detect = new COCODetect(COCODetect::YOLO11N_320_S8_V1, false);
  if (!s_detect || !s_detect->get_raw_model()) {
    ESP_LOGW(TAG, "model not loaded: is the coco_det partition flashed?");
    delete s_detect;
    s_detect = nullptr;
    return false;
  }
  ESP_LOGI(TAG, "COCO YOLO11n 320 int8 loaded in %lld ms; psram free %u, internal free %u",
           (esp_timer_get_time() - t0) / 1000, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
  return true;
}

#if CONFIG_SOC_JPEG_CODEC_SUPPORTED
// P4: the JPEG codec decodes the frame in hardware, at full size into PSRAM (the detector scales it
// to 320x320 itself). The software decoder would also bring ~3 kB of .data into internal RAM, which
// the P4 has none to spare of (its retention-RAM margin, see the P4 README).
static bool decode_rgb565(const uint8_t *jpeg, size_t len, dl::image::img_t *img) {
  dl::image::jpeg_img_t j = {.data = (void *)jpeg, .data_len = len};
  *img = dl::image::hw_decode_jpeg(j, dl::image::DL_IMAGE_PIX_TYPE_RGB565LE, 200);
  return img->data != nullptr;
}

static void free_img(dl::image::img_t *img) { heap_caps_free(img->data); }
#else
// JPEG -> RGB565, scaled by 1/2, 1/4 or 1/8 until the width is at most 640 (the detector works at
// 320x320, so more pixels only cost decode time and PSRAM: a 1600x1200 frame becomes 400x300).
static bool decode_rgb565(const uint8_t *jpeg, size_t len, dl::image::img_t *img) {
  jpeg_dec_config_t cfg = DEFAULT_JPEG_DEC_CONFIG();
  cfg.output_type = JPEG_PIXEL_FORMAT_RGB565_LE;
  jpeg_dec_header_info_t info = {};
  jpeg_dec_io_t io = {};
  jpeg_dec_handle_t dec = NULL;
  for (int pass = 0; pass < 2; pass++) {  // pass 0 reads the size, pass 1 decodes at the chosen scale
    if (jpeg_dec_open(&cfg, &dec) != JPEG_ERR_OK) return false;
    io = {};
    io.inbuf = (uint8_t *)jpeg;
    io.inbuf_len = (int)len;
    if (jpeg_dec_parse_header(dec, &io, &info) != JPEG_ERR_OK) {
      jpeg_dec_close(dec);
      return false;
    }
    if (pass == 1) break;
    jpeg_dec_close(dec);
    int k = 1;
    while (info.width / k > 640 && k < 8) k *= 2;
    if (k > 1) {
      cfg.scale.width = (uint16_t)((info.width / k) & ~7);
      cfg.scale.height = (uint16_t)((info.height / k) & ~7);
    }
  }
  int out_len = 0;
  uint8_t *buf = NULL;
  if (jpeg_dec_get_outbuf_len(dec, &out_len) == JPEG_ERR_OK && out_len > 0)
    buf = (uint8_t *)jpeg_calloc_align((size_t)out_len, 16);
  io.outbuf = buf;
  bool ok = buf && jpeg_dec_process(dec, &io) == JPEG_ERR_OK;
  jpeg_dec_close(dec);
  if (!ok) {
    if (buf) jpeg_free_align(buf);
    return false;
  }
  img->data = buf;
  img->width = cfg.scale.width ? cfg.scale.width : info.width;
  img->height = cfg.scale.height ? cfg.scale.height : info.height;
  img->pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565LE;
  return true;
}

static void free_img(dl::image::img_t *img) { jpeg_free_align(img->data); }
#endif

bool detect_local_run(const uint8_t *jpeg, size_t len, detect_local_result_t *out) {
  memset(out, 0, sizeof(*out));
  if (!detect_local_init()) return false;
  int64_t t0 = esp_timer_get_time();
  dl::image::img_t img = {};
  bool decoded = decode_rgb565(jpeg, len, &img);
  int64_t t1 = esp_timer_get_time();
  if (!decoded) {
    ESP_LOGW(TAG, "JPEG decode failed");
    return false;
  }
  std::list<dl::detect::result_t> &res = s_detect->run(img);
  int64_t t2 = esp_timer_get_time();
  out->decode_ms = (int)((t1 - t0) / 1000);
  out->infer_ms = (int)((t2 - t1) / 1000);
  out->width = img.width;
  out->height = img.height;
  size_t used = 0;
  for (const auto &r : res) {
    out->count++;
    const char *name = r.category >= 0 && r.category < (int)(sizeof(k_coco) / sizeof(k_coco[0])) ? k_coco[r.category] : "?";
    int n = snprintf(out->labels + used, sizeof(out->labels) - used, "%s%s %.2f", used ? ", " : "", name,
                     (double)r.score);
    if (n < 0 || (size_t)n >= sizeof(out->labels) - used) break;
    used += (size_t)n;
  }
  free_img(&img);
  return true;
}

int detect_local_boxes(const uint8_t *jpeg, size_t len, loci_detection_t *out, int max) {
  if (!detect_local_init()) return -1;
  dl::image::img_t img = {};
  if (!decode_rgb565(jpeg, len, &img)) return -1;
  std::list<dl::detect::result_t> &res = s_detect->run(img);
  const float w = (float)img.width, h = (float)img.height;
  int n = 0;
  for (const auto &r : res) {
    if (n >= max) break;
    if (r.category < 0 || r.category >= (int)(sizeof(k_coco) / sizeof(k_coco[0])) || r.box.size() < 4) continue;
    loci_detection_t *d = &out[n++];
    snprintf(d->label, sizeof(d->label), "%s", k_coco[r.category]);
    d->score = r.score;
    d->cx = ((r.box[0] + r.box[2]) / 2.0f) / w;
    d->cy = ((r.box[1] + r.box[3]) / 2.0f) / h;
    d->w = (r.box[2] - r.box[0]) / w;
    d->h = (r.box[3] - r.box[1]) / h;
  }
  free_img(&img);
  return n;
}
