#include "host_link.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "esp_check.h"
#include "esp_crt_bundle.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs.h"

#include "audio.h"
#include "board_p4_nano.h"
#include "wifi_link.h"

static const char *TAG = "host";

static char s_base[128];
static char s_default_base[128];  // from secrets.h; `host default` goes back to it
static char s_device[32];
static char s_token[80];  // X-Device-Token; empty on a LAN-only setup
static uint32_t s_clips;
#define RESP_CAP 8192  // host replies are small, flat JSON (8 kB covers a long detections list)
static char *s_resp;  // RESP_CAP bytes, allocated in PSRAM (internal RAM is scarce, see README)
static int64_t s_clock_offset_ms;  // host Unix ms - esp_timer ms, from the last frame reply
static bool s_clock_synced;
static int s_failures;  // consecutive failed frame posts
static int s_frame_timeout_ms = 8000;

static int64_t json_int64(const char *body, const char *key) {
  char needle[48];
  snprintf(needle, sizeof(needle), "\"%s\":", key);
  const char *p = strstr(body, needle);
  if (!p) return 0;
  return atoll(p + strlen(needle));
}

int64_t host_now_ms(void) {
  return s_clock_synced ? esp_timer_get_time() / 1000 + s_clock_offset_ms : 0;
}
const char *host_last_reply(void) { return s_resp ? s_resp : ""; }
const char *host_device_id(void) { return s_device; }
const char *host_link_base(void) { return s_base; }
bool host_link_has_token(void) { return s_token[0] != 0; }
bool host_reachable(void) { return s_failures < 2; }
void host_set_frame_timeout_ms(int ms) { s_frame_timeout_ms = ms; }

// --- tiny JSON peeking (the host keeps replies flat so this is enough) ---------------------
static int json_int(const char *body, const char *key, int fallback) {
  char needle[48];
  snprintf(needle, sizeof(needle), "\"%s\":", key);
  const char *p = strstr(body, needle);
  if (!p) return fallback;
  p += strlen(needle);
  while (*p == ' ') p++;
  if (!strncmp(p, "true", 4)) return 1;
  if (!strncmp(p, "false", 5)) return 0;
  if (!strncmp(p, "null", 4)) return fallback;
  return atoi(p);
}

static void json_string(const char *body, const char *key, char *out, size_t cap) {
  out[0] = 0;
  char needle[48];
  snprintf(needle, sizeof(needle), "\"%s\":\"", key);
  const char *p = strstr(body, needle);
  if (!p) return;
  p += strlen(needle);
  const char *e = strchr(p, '"');
  size_t n = e ? (size_t)(e - p) : strlen(p);
  if (n >= cap) n = cap - 1;
  memcpy(out, p, n);
  out[n] = 0;
}

static void json_array(const char *body, const char *key, char *out, size_t cap) {
  out[0] = 0;
  char needle[48];
  snprintf(needle, sizeof(needle), "\"%s\":[", key);
  const char *p = strstr(body, needle);
  if (!p) return;
  p += strlen(needle);
  const char *e = strchr(p, ']');
  size_t n = e ? (size_t)(e - p) : strlen(p);
  if (n >= cap) n = cap - 1;
  memcpy(out, p, n);
  out[n] = 0;
}

static void json_escape(const char *in, char *out, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; in[i] && j + 2 < cap; i++) {
    if (in[i] == '"' || in[i] == '\\') out[j++] = '\\';
    out[j++] = in[i];
  }
  out[j] = 0;
}

// --- HTTP helpers --------------------------------------------------------------------------
static esp_http_client_handle_t client_new(const char *url, esp_http_client_method_t method, int timeout_ms) {
  esp_http_client_config_t cfg = {
      .url = url,
      .method = method,
      .timeout_ms = timeout_ms,
      .buffer_size = 4096,
      .buffer_size_tx = 2048,
      .keep_alive_enable = false,
      .crt_bundle_attach = esp_crt_bundle_attach,  // https:// hosts (the Cloudflare Tunnel) verify against the IDF CA bundle
  };
  esp_http_client_handle_t c = esp_http_client_init(&cfg);
  if (c && s_token[0]) esp_http_client_set_header(c, "X-Device-Token", s_token);
  return c;
}

// One persistent, keep-alive connection for the frame stream: a fresh TCP connection per frame
// pays the handshake and TCP slow start on every 140 kB upload. The handle is kept open between
// requests and only closed (not freed) after an error so the next frame reconnects.
static esp_http_client_handle_t s_frame_client;
static int64_t s_frame_last_ms;  // when the last frame exchange finished
// The frame loop task owns s_frame_client. The console task must not free it (it may be in the
// middle of a TLS handshake); it only marks it stale and the next frame recreates it.
static volatile bool s_frame_client_stale;

static esp_http_client_handle_t frame_client(const char *url) {
  if (s_frame_client && s_frame_client_stale) {  // host URL or token changed since the last frame
    esp_http_client_cleanup(s_frame_client);
    s_frame_client = NULL;
    s_frame_last_ms = 0;
  }
  s_frame_client_stale = false;
  if (s_frame_client && s_frame_last_ms && esp_timer_get_time() / 1000 - s_frame_last_ms > 4000) {
    // the host (uvicorn) drops idle keep-alive connections after 5 s; writing into that half-closed
    // socket fails and costs a frame, so after a long pause reconnect on purpose
    esp_http_client_close(s_frame_client);
  }
  if (!s_frame_client) {
    esp_http_client_config_t cfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = s_frame_timeout_ms,
        .buffer_size = 8192,
        .buffer_size_tx = 8192,
        .keep_alive_enable = true,
        .keep_alive_idle = 10,
        .keep_alive_interval = 5,
        .keep_alive_count = 3,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };
    s_frame_client = esp_http_client_init(&cfg);
  } else {
    esp_http_client_set_url(s_frame_client, url);
    esp_http_client_set_method(s_frame_client, HTTP_METHOD_POST);
    esp_http_client_set_timeout_ms(s_frame_client, s_frame_timeout_ms);
  }
  if (s_frame_client && s_token[0]) esp_http_client_set_header(s_frame_client, "X-Device-Token", s_token);
  return s_frame_client;
}

// Ask the frame loop to drop its keep-alive connection (the host URL, scheme or token changed).
// Called from the console task, so nothing is freed here; see s_frame_client_stale.
static void frame_client_reset(void) {
  s_frame_client_stale = true;
  s_failures = 0;
}

// Send `len` bytes of body, then read the (small) response into s_resp. Returns the HTTP status.
static int exchange(esp_http_client_handle_t c, const uint8_t *body, size_t len) {
  if (!s_resp) return -1;
  esp_err_t err = esp_http_client_open(c, (int)len);
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "open failed: %s", esp_err_to_name(err));
    return -1;
  }
  size_t sent = 0;
  while (sent < len) {
    int n = esp_http_client_write(c, (const char *)body + sent, (int)(len - sent));
    if (n <= 0) {
      ESP_LOGW(TAG, "write failed after %u bytes", (unsigned)sent);
      return -1;
    }
    sent += (size_t)n;
  }
  if (esp_http_client_fetch_headers(c) < 0) return -1;
  int status = esp_http_client_get_status_code(c);
  int got = esp_http_client_read_response(c, s_resp, RESP_CAP - 1);
  if (got < 0) got = 0;
  s_resp[got] = 0;
  if (got >= RESP_CAP - 1) esp_http_client_flush_response(c, NULL);  // keep the stream in sync
  return status;
}

// --- settings in NVS (namespace "loci": "host", "token") ------------------------------------
#define NVS_NS "loci"

static bool nvs_get_string(const char *key, char *out, size_t cap) {
  nvs_handle_t h;
  if (nvs_open(NVS_NS, NVS_READONLY, &h) != ESP_OK) return false;
  size_t len = cap;
  esp_err_t err = nvs_get_str(h, key, out, &len);
  nvs_close(h);
  if (err != ESP_OK) out[0] = 0;
  return err == ESP_OK && out[0];
}

static void nvs_put_string(const char *key, const char *value) {
  nvs_handle_t h;
  if (nvs_open(NVS_NS, NVS_READWRITE, &h) != ESP_OK) return;
  if (value && value[0]) {
    nvs_set_str(h, key, value);
  } else {
    nvs_erase_key(h, key);
  }
  nvs_commit(h);
  nvs_close(h);
}

static void trim_url(char *url) {
  size_t n = strlen(url);
  while (n && (url[n - 1] == '/' || url[n - 1] == ' ')) url[--n] = 0;
}

bool host_link_set_host(const char *url) {
  if (url && url[0]) {
    if (strncmp(url, "http://", 7) != 0 && strncmp(url, "https://", 8) != 0) {
      ESP_LOGW(TAG, "host must start with http:// or https://");
      return false;
    }
    snprintf(s_base, sizeof(s_base), "%s", url);
    trim_url(s_base);
    nvs_put_string("host", s_base);
  } else {
    snprintf(s_base, sizeof(s_base), "%s", s_default_base);
    nvs_put_string("host", NULL);
  }
  frame_client_reset();
  s_clock_synced = false;
  return true;
}

void host_link_set_token(const char *token) {
  snprintf(s_token, sizeof(s_token), "%s", token ? token : "");
  nvs_put_string("token", s_token);
  frame_client_reset();  // the keep-alive handle carries the old header
}

void host_link_init(const char *base_url, const char *device_id, const char *token) {
  snprintf(s_default_base, sizeof(s_default_base), "%s", base_url);
  trim_url(s_default_base);
  snprintf(s_device, sizeof(s_device), "%s", device_id);
  snprintf(s_token, sizeof(s_token), "%s", token ? token : "");
  if (!nvs_get_string("host", s_base, sizeof(s_base))) snprintf(s_base, sizeof(s_base), "%s", s_default_base);
  char stored[sizeof(s_token)];
  if (nvs_get_string("token", stored, sizeof(stored))) snprintf(s_token, sizeof(s_token), "%s", stored);
  ESP_LOGI(TAG, "host %s%s, device token %s", s_base, strcmp(s_base, s_default_base) ? " (from NVS)" : "",
           s_token[0] ? "set" : "none");
  if (!s_resp) s_resp = heap_caps_calloc(1, RESP_CAP, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!s_resp) s_resp = calloc(1, RESP_CAP);
  if (!s_resp) ESP_LOGE(TAG, "no memory for the reply buffer; host requests will fail");
}

esp_err_t host_post_frame(const uint8_t *jpeg, size_t len, uint32_t seq, host_frame_result_t *out) {
  return host_post_frame_ex(jpeg, len, seq, 0, false, out);
}

esp_err_t host_post_frame_ex(const uint8_t *jpeg, size_t len, uint32_t seq, int64_t captured_ms, bool backlog,
                             host_frame_result_t *out) {
  char url[224], hdr[32];
  snprintf(url, sizeof(url), "%s/api/esp32/frame?device=%s&seq=%lu", s_base, s_device, (unsigned long)seq);
  esp_http_client_handle_t c = frame_client(url);
  if (!c) return ESP_ERR_NO_MEM;
  esp_http_client_set_header(c, "Content-Type", "image/jpeg");
  esp_http_client_set_header(c, "X-Device-Id", s_device);
  snprintf(hdr, sizeof(hdr), "%lld", esp_timer_get_time() / 1000);
  esp_http_client_set_header(c, "X-Device-Ms", hdr);
  snprintf(hdr, sizeof(hdr), "%d", wifi_link_rssi());
  esp_http_client_set_header(c, "X-Rssi", hdr);
  // headers persist on the shared handle: set or remove the backlog ones every time
  if (backlog) {
    esp_http_client_set_header(c, "X-Backlog", "1");
    if (captured_ms > 0) {
      snprintf(hdr, sizeof(hdr), "%lld", (long long)captured_ms);
      esp_http_client_set_header(c, "X-Captured-Ms", hdr);
    } else {
      esp_http_client_delete_header(c, "X-Captured-Ms");
    }
  } else {
    esp_http_client_delete_header(c, "X-Backlog");
    esp_http_client_delete_header(c, "X-Captured-Ms");
  }

  int status = exchange(c, jpeg, len);
  s_frame_last_ms = esp_timer_get_time() / 1000;
  if (status != 200) {
    esp_http_client_close(c);  // drop the connection; the next frame reconnects
    s_failures++;
    ESP_LOGW(TAG, "frame #%lu -> HTTP %d %s", (unsigned long)seq, status, status > 0 ? s_resp : "");
    return ESP_FAIL;
  }
  s_failures = 0;
  memset(out, 0, sizeof(*out));
  out->dropped = json_int(s_resp, "dropped", 0) == 1;
  out->audio_pending = json_int(s_resp, "audio_pending", 0);
  out->next_ms = json_int(s_resp, "next_ms", 0);
  json_array(s_resp, "labels", out->labels, sizeof(out->labels));
  json_string(s_resp, "memory_rev", out->memory_rev, sizeof(out->memory_rev));
  int64_t now = json_int64(s_resp, "now_ms");
  if (now > 1000000000000LL) {
    s_clock_offset_ms = now - esp_timer_get_time() / 1000;
    s_clock_synced = true;
  }
  return ESP_OK;
}

esp_err_t host_fetch_file(const char *path_and_query, const char *dest_path, size_t *bytes_out) {
  char url[256], tmp[128];
  snprintf(url, sizeof(url), "%s%s", s_base, path_and_query);
  snprintf(tmp, sizeof(tmp), "%s.part", dest_path);
  esp_http_client_handle_t c = client_new(url, HTTP_METHOD_GET, 30000);
  if (!c) return ESP_ERR_NO_MEM;
  esp_err_t err = esp_http_client_open(c, 0);
  if (err != ESP_OK) {
    esp_http_client_cleanup(c);
    return err;
  }
  esp_http_client_fetch_headers(c);
  int status = esp_http_client_get_status_code(c);
  if (status != 200) {
    esp_http_client_close(c);
    esp_http_client_cleanup(c);
    return status == 204 ? ESP_ERR_NOT_FOUND : ESP_FAIL;
  }
  FILE *f = fopen(tmp, "wb");
  if (!f) {
    esp_http_client_close(c);
    esp_http_client_cleanup(c);
    return ESP_FAIL;
  }
  static uint8_t *buf;
  if (!buf) buf = heap_caps_malloc(4096, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  size_t total = 0;
  bool ok = true;
  for (;;) {
    int n = esp_http_client_read(c, (char *)buf, 4096);
    if (n < 0) {
      ok = false;
      break;
    }
    if (n == 0) break;
    if (fwrite(buf, 1, (size_t)n, f) != (size_t)n) {
      ok = false;
      break;
    }
    total += (size_t)n;
  }
  fclose(f);
  esp_http_client_close(c);
  esp_http_client_cleanup(c);
  if (!ok || total == 0) {
    unlink(tmp);
    return ESP_FAIL;
  }
  unlink(dest_path);
  if (rename(tmp, dest_path) != 0) {
    unlink(tmp);
    return ESP_FAIL;
  }
  if (bytes_out) *bytes_out = total;
  return ESP_OK;
}

bool host_fetch_and_play_audio(int wait_s) {
  if (!audio_ready()) return false;
  char url[224];
  snprintf(url, sizeof(url), "%s/api/esp32/audio/next?device=%s&wait=%d", s_base, s_device, wait_s);
  esp_http_client_handle_t c = client_new(url, HTTP_METHOD_GET, (wait_s + 15) * 1000);
  if (!c) return false;
  bool played = false;
  if (esp_http_client_open(c, 0) == ESP_OK) {
    int64_t content_len = esp_http_client_fetch_headers(c);
    int status = esp_http_client_get_status_code(c);
    if (status == 200) {
      static uint8_t *buf;
      if (!buf) buf = heap_caps_malloc(4096, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
      size_t total = 0, carry = 0;
      int64_t t0 = esp_timer_get_time();
      for (;;) {
        int n = esp_http_client_read(c, (char *)buf + carry, (int)(4096 - carry));
        if (n <= 0) break;
        size_t have = carry + (size_t)n;
        size_t even = have & ~(size_t)1;
        audio_play_pcm16((const int16_t *)buf, even / 2);
        carry = have - even;
        if (carry) buf[0] = buf[even];
        total += (size_t)n;
        if (content_len > 0 && (int64_t)total >= content_len) break;
      }
      played = total > 0;
      if (played) s_clips++;
      ESP_LOGI(TAG, "played %u bytes (%.1f s) in %lld ms", (unsigned)total,
               total / (2.0f * LOCI_AUDIO_RATE), (esp_timer_get_time() - t0) / 1000);
    } else if (status != 204) {
      ESP_LOGW(TAG, "audio/next -> HTTP %d", status);
    }
  }
  esp_http_client_close(c);
  esp_http_client_cleanup(c);
  return played;
}

esp_err_t host_post_voice(const uint8_t *pcm, size_t len) {
  char url[224];
  snprintf(url, sizeof(url), "%s/api/esp32/voice?device=%s&rate=%d", s_base, s_device, LOCI_AUDIO_RATE);
  esp_http_client_handle_t c = client_new(url, HTTP_METHOD_POST, 45000);
  if (!c) return ESP_ERR_NO_MEM;
  esp_http_client_set_header(c, "Content-Type", "audio/L16; rate=16000");
  esp_http_client_set_header(c, "X-Device-Id", s_device);
  int status = exchange(c, pcm, len);
  esp_http_client_close(c);
  esp_http_client_cleanup(c);
  if (status != 200) {
    ESP_LOGW(TAG, "voice -> HTTP %d", status);
    return ESP_FAIL;
  }
  char heard[128], answer[256];
  json_string(s_resp, "transcription", heard, sizeof(heard));
  json_string(s_resp, "answer", answer, sizeof(answer));
  ESP_LOGI(TAG, "heard: \"%s\"", heard);
  ESP_LOGI(TAG, "answer: %s", answer);
  if (json_int(s_resp, "audio_queued", 0) == 1) host_fetch_and_play_audio(5);
  return ESP_OK;
}

static esp_err_t post_text(const char *path, const char *text, bool play) {
  char url[224], escaped[256], payload[320];
  snprintf(url, sizeof(url), "%s%s", s_base, path);
  json_escape(text, escaped, sizeof(escaped));
  snprintf(payload, sizeof(payload), "{\"text\":\"%s\",\"device\":\"%s\"}", escaped, s_device);
  esp_http_client_handle_t c = client_new(url, HTTP_METHOD_POST, 30000);
  if (!c) return ESP_ERR_NO_MEM;
  esp_http_client_set_header(c, "Content-Type", "application/json");
  int status = exchange(c, (const uint8_t *)payload, strlen(payload));
  esp_http_client_close(c);
  esp_http_client_cleanup(c);
  if (status != 200) {
    ESP_LOGW(TAG, "%s -> HTTP %d %s", path, status, status > 0 ? s_resp : "");
    return ESP_FAIL;
  }
  char answer[256];
  json_string(s_resp, "answer", answer, sizeof(answer));
  if (answer[0]) ESP_LOGI(TAG, "answer: %s", answer);
  if (play && (json_int(s_resp, "audio_queued", 0) == 1 || strstr(s_resp, "\"status\":\"queued\"")))
    host_fetch_and_play_audio(5);
  return ESP_OK;
}

esp_err_t host_ask(const char *text) { return post_text("/api/esp32/ask", text, true); }
esp_err_t host_say(const char *text) { return post_text("/api/esp32/say", text, true); }

uint32_t host_audio_clips_played(void) { return s_clips; }
