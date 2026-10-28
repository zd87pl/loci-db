#include "cloud_link.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#include <unistd.h>

#include "cJSON.h"
#include "esp_crt_bundle.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_netif_sntp.h"
#include "esp_timer.h"
#include "nvs.h"
#include "sdkconfig.h"

#include "loci_config.h"
#include "loci_hw.h"
#include "wifi_link.h"

static const char *TAG = "cloud";

#define NVS_NS "loci"
#define UNIX_MS_MIN 1700000000000LL  // Nov 2023: an earlier system clock was never set
#define RESP_CAP 4096                // JSON replies (claim, token, writes) are small
#define BACKOFF_US (60LL * 1000000)  // after a failed request the cloud is left alone for a minute
#define REFRESH_MARGIN_S 300         // renew the device token 5 min before it expires
#define IDLE_DROP_US (20LL * 1000000)
// TLS needs one large internal block on the P4 (no MBEDTLS_EXTERNAL_MEM_ALLOC there); below this the
// board answers from the card rather than risk a failed allocation mid-handshake.
#ifndef LOCI_CLOUD_MIN_INTERNAL_BLOCK
#define LOCI_CLOUD_MIN_INTERNAL_BLOCK (40 * 1024)
#endif

// Everything sizeable lives in one PSRAM block: internal RAM is the scarce part on both boards
// (the P4's retention RAM, the S3's ~17 kB heap minimum), and the S3 cannot place .bss in PSRAM.
typedef struct {
  char url[160];  // project URL, https://<worker>/p/<ref>/v1
  char space[64];
  char dev[40];
  char sec[96];  // device secret: never logged
  char room[48];
  char token[1024];  // device JWT (the host link's 80-byte token buffer would cut it)
  char auth[1040];   // "Bearer <token>"
  char resp[RESP_CAP];
  char hdr_answer[512];
  char hdr_heard[256];
  char ua[48];
  uint8_t io[4096];  // audio and download chunks
} cloud_state_t;

static cloud_state_t *S;
static int64_t s_token_until_us;  // esp_timer time after which the token is renewed
static int64_t s_skip_until_us;   // back-off after a failure
static int s_last_status;
static bool s_needs_setup;  // revoked or wrong secret: claim again
static bool s_clock_started;
#if !CONFIG_MBEDTLS_EXTERNAL_MEM_ALLOC
static bool s_low_ram_logged;
#endif
static esp_http_client_handle_t s_client;  // one keep-alive TLS client, main loop only
static int64_t s_client_used_us;

// --- NVS (namespace "loci") --------------------------------------------------------------------
static bool nvs_get_string(const char *key, char *out, size_t cap) {
  nvs_handle_t h;
  out[0] = 0;
  if (nvs_open(NVS_NS, NVS_READONLY, &h) != ESP_OK) return false;
  size_t len = cap;
  esp_err_t err = nvs_get_str(h, key, out, &len);
  nvs_close(h);
  if (err != ESP_OK) out[0] = 0;
  return out[0] != 0;
}

static bool nvs_put_string(const char *key, const char *value) {
  nvs_handle_t h;
  if (nvs_open(NVS_NS, NVS_READWRITE, &h) != ESP_OK) return false;
  esp_err_t err = value && value[0] ? nvs_set_str(h, key, value) : nvs_erase_key(h, key);
  if (err == ESP_ERR_NVS_NOT_FOUND) err = ESP_OK;
  if (err == ESP_OK) err = nvs_commit(h);
  nvs_close(h);
  return err == ESP_OK;
}

// --- small helpers -----------------------------------------------------------------------------
static void json_escape(const char *in, char *out, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; in[i] && j + 2 < cap; i++) {
    unsigned char c = (unsigned char)in[i];
    if (c < 0x20) continue;
    if (c == '"' || c == '\\') out[j++] = '\\';
    out[j++] = (char)c;
  }
  out[j] = 0;
}

// The cloud percent-encodes the answer text it puts in headers.
static void url_decode(const char *in, char *out, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; in[i] && j + 1 < cap; i++) {
    if (in[i] == '%' && in[i + 1] && in[i + 2]) {
      char hex[3] = {in[i + 1], in[i + 2], 0};
      out[j++] = (char)strtol(hex, NULL, 16);
      i += 2;
    } else {
      out[j++] = in[i];
    }
  }
  out[j] = 0;
}

static void trim(char *s) {
  size_t n = strlen(s);
  while (n && (s[n - 1] == '/' || s[n - 1] == ' ')) s[--n] = 0;
}

int64_t cloud_link_now_ms(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  int64_t ms = (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
  return ms > UNIX_MS_MIN ? ms : 0;
}

void cloud_link_start_clock(void) {
  if (s_clock_started) return;
  esp_sntp_config_t c = ESP_NETIF_SNTP_DEFAULT_CONFIG("pool.ntp.org");
  if (esp_netif_sntp_init(&c) == ESP_OK) {
    s_clock_started = true;
    ESP_LOGI(TAG, "clock: SNTP started (pool.ntp.org)");
  }
}

bool cloud_link_configured(void) { return S && S->url[0] && S->space[0] && S->dev[0] && S->sec[0]; }
const char *cloud_link_device_id(void) { return S ? S->dev : ""; }
const char *cloud_link_room(void) { return S ? S->room : ""; }

bool cloud_link_usable(void) {
  if (!cloud_link_configured() || s_needs_setup || !wifi_link_connected()) return false;
  if (esp_timer_get_time() < s_skip_until_us) return false;
#if !CONFIG_MBEDTLS_EXTERNAL_MEM_ALLOC
  if (!s_client && heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) < LOCI_CLOUD_MIN_INTERNAL_BLOCK) {
    if (!s_low_ram_logged) {
      ESP_LOGW(TAG, "largest internal block %u B < %u B: not opening TLS, answering from the card",
               (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
               (unsigned)LOCI_CLOUD_MIN_INTERNAL_BLOCK);
      s_low_ram_logged = true;
    }
    return false;
  }
  s_low_ram_logged = false;
#endif
  return true;
}

void cloud_link_init(const char *fw_version) {
  if (!S) S = heap_caps_calloc(1, sizeof(*S), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!S) S = calloc(1, sizeof(*S));
  if (!S) {
    ESP_LOGE(TAG, "no memory for the cloud link");
    return;
  }
  snprintf(S->ua, sizeof(S->ua), "loci-board/%s", fw_version ? fw_version : "?");
  nvs_get_string("c_url", S->url, sizeof(S->url));
  nvs_get_string("c_space", S->space, sizeof(S->space));
  nvs_get_string("c_dev", S->dev, sizeof(S->dev));
  nvs_get_string("c_sec", S->sec, sizeof(S->sec));
  nvs_get_string("c_room", S->room, sizeof(S->room));
  if (cloud_link_configured()) {
    ESP_LOGI(TAG, "%s space %s as %s%s%s", S->url, S->space, S->dev, S->room[0] ? ", room " : "", S->room);
  } else {
    ESP_LOGI(TAG, "not claimed (console: cloud claim <project_url> <space> <CODE>)");
  }
}

// --- HTTP --------------------------------------------------------------------------------------
static esp_err_t on_event(esp_http_client_event_t *e) {
  if (e->event_id != HTTP_EVENT_ON_HEADER || !S || !e->header_key || !e->header_value) return ESP_OK;
  if (!strcasecmp(e->header_key, "x-answer")) {
    snprintf(S->hdr_answer, sizeof(S->hdr_answer), "%s", e->header_value);
  } else if (!strcasecmp(e->header_key, "x-transcription")) {
    snprintf(S->hdr_heard, sizeof(S->hdr_heard), "%s", e->header_value);
  }
  return ESP_OK;
}

static void client_drop(void) {
  if (s_client) esp_http_client_cleanup(s_client);
  s_client = NULL;
}

static esp_http_client_handle_t client(const char *url, esp_http_client_method_t method, int timeout_ms) {
  int64_t now = esp_timer_get_time();
  if (s_client && now - s_client_used_us > IDLE_DROP_US) client_drop();  // the edge may have closed it
  if (!s_client) {
    esp_http_client_config_t cfg = {
        .url = url,
        .method = method,
        .timeout_ms = timeout_ms,
        .buffer_size = 4096,
        .buffer_size_tx = 2048,
        .keep_alive_enable = true,
        .crt_bundle_attach = esp_crt_bundle_attach,
        .event_handler = on_event,
        .user_agent = S->ua,  // Cloudflare blocks clients without one of their own
    };
    s_client = esp_http_client_init(&cfg);
  } else {
    esp_http_client_set_url(s_client, url);
    esp_http_client_set_method(s_client, method);
    esp_http_client_set_timeout_ms(s_client, timeout_ms);
  }
  S->hdr_answer[0] = S->hdr_heard[0] = 0;
  s_client_used_us = now;
  return s_client;
}

// A failed request: drop the connection; no answer or a server error pauses the cloud for a minute.
static void failed(int status) {
  s_last_status = status;
  client_drop();
  if (status <= 0 || status >= 500) s_skip_until_us = esp_timer_get_time() + BACKOFF_US;
}

// Send the request; returns the HTTP status (<= 0: no answer). The body is read by the caller.
static int send_request(esp_http_client_handle_t c, const void *body, size_t len) {
  if (esp_http_client_open(c, (int)len) != ESP_OK) return -1;
  size_t sent = 0;
  while (sent < len) {
    int n = esp_http_client_write(c, (const char *)body + sent, (int)(len - sent));
    if (n <= 0) return -1;
    sent += (size_t)n;
  }
  if (esp_http_client_fetch_headers(c) < 0) return -1;
  s_last_status = esp_http_client_get_status_code(c);
  return s_last_status;
}

static void read_json(esp_http_client_handle_t c) {
  int got = esp_http_client_read_response(c, S->resp, RESP_CAP - 1);
  S->resp[got > 0 ? got : 0] = 0;
  if (got >= RESP_CAP - 1) esp_http_client_flush_response(c, NULL);
}

static bool fetch_token(void) {
  char url[200], body[200];
  snprintf(url, sizeof(url), "%s/devices/token", S->url);
  snprintf(body, sizeof(body), "{\"device_id\":\"%s\",\"device_secret\":\"%s\"}", S->dev, S->sec);
  esp_http_client_handle_t c = client(url, HTTP_METHOD_POST, 8000);
  if (!c) return false;
  esp_http_client_delete_header(c, "Authorization");
  esp_http_client_set_header(c, "Content-Type", "application/json");
  int status = send_request(c, body, strlen(body));
  memset(body, 0, sizeof(body));  // held the secret
  if (status > 0) read_json(c);
  if (status == 401 || status == 404) {
    s_needs_setup = true;  // revoked, or claimed in another project: a new claim code is needed
    ESP_LOGE(TAG, "the cloud no longer accepts this board (HTTP %d): claim it again", status);
    failed(status);
    return false;
  }
  if (status != 200) {
    ESP_LOGW(TAG, "token -> HTTP %d", status);
    failed(status);
    return false;
  }
  cJSON *j = cJSON_Parse(S->resp);
  const cJSON *tok = cJSON_GetObjectItem(j, "token");
  const cJSON *ttl = cJSON_GetObjectItem(j, "expires_in");
  bool ok = cJSON_IsString(tok) && strlen(tok->valuestring) < sizeof(S->token);
  if (ok) {
    snprintf(S->token, sizeof(S->token), "%s", tok->valuestring);
    int life = cJSON_IsNumber(ttl) ? ttl->valueint : 3600;  // a Worker before expires_in: the default TTL
    int renew = life - REFRESH_MARGIN_S > 60 ? life - REFRESH_MARGIN_S : 60;
    s_token_until_us = esp_timer_get_time() + (int64_t)renew * 1000000;
    s_needs_setup = false;
  } else {
    ESP_LOGW(TAG, "token reply without a usable token");
  }
  cJSON_Delete(j);
  memset(S->resp, 0, RESP_CAP);  // held the token
  return ok;
}

static bool ensure_token(void) {
  if (S->token[0] && esp_timer_get_time() < s_token_until_us) return true;
  return fetch_token();
}

// An authorised request (Bearer device token); a 401 renews the token once. Returns the HTTP
// status with the response body still to read from *out.
static int authed(const char *url, esp_http_client_method_t method, const char *ctype, const void *body, size_t len,
                  int timeout_ms, esp_http_client_handle_t *out) {
  *out = NULL;
  for (int attempt = 0; attempt < 2; attempt++) {
    if (!ensure_token()) return -2;
    esp_http_client_handle_t c = client(url, method, timeout_ms);
    if (!c) return -1;
    snprintf(S->auth, sizeof(S->auth), "Bearer %s", S->token);
    esp_http_client_set_header(c, "Authorization", S->auth);
    if (ctype) {
      esp_http_client_set_header(c, "Content-Type", ctype);
    } else {
      esp_http_client_delete_header(c, "Content-Type");
    }
    int status = send_request(c, body, len);
    if (status == 401 && attempt == 0) {
      esp_http_client_flush_response(c, NULL);
      S->token[0] = 0;  // superseded or expired: renew and retry once
      continue;
    }
    *out = c;
    return status;
  }
  return 401;
}

// --- claim ---------------------------------------------------------------------------------------
esp_err_t cloud_link_claim(const char *project_url, const char *space, const char *code) {
  if (!S) return ESP_ERR_NO_MEM;
  if (strncmp(project_url, "https://", 8) != 0 && strncmp(project_url, "http://", 7) != 0) {
    ESP_LOGW(TAG, "the project URL must start with https:// (e.g. https://<worker>/p/<ref>/v1)");
    return ESP_ERR_INVALID_ARG;
  }
  if (!space[0] || strlen(space) >= sizeof(S->space) || !code[0] || strlen(code) > 32) return ESP_ERR_INVALID_ARG;
  char base[sizeof(S->url)], url[220], body[96];
  snprintf(base, sizeof(base), "%s", project_url);
  trim(base);
  snprintf(url, sizeof(url), "%s/devices/claim", base);
  snprintf(body, sizeof(body), "{\"claim_code\":\"%s\"}", code);
  esp_http_client_handle_t c = client(url, HTTP_METHOD_POST, 10000);
  if (!c) return ESP_ERR_NO_MEM;
  esp_http_client_delete_header(c, "Authorization");
  esp_http_client_set_header(c, "Content-Type", "application/json");
  int status = send_request(c, body, strlen(body));
  if (status > 0) read_json(c);
  if (status != 200) {
    ESP_LOGW(TAG, "claim -> HTTP %d %s", status, status > 0 ? S->resp : "(no answer)");
    failed(status);
    return ESP_FAIL;
  }
  cJSON *j = cJSON_Parse(S->resp);
  const cJSON *dev = cJSON_GetObjectItem(j, "device_id");
  const cJSON *sec = cJSON_GetObjectItem(j, "device_secret");
  bool ok = cJSON_IsString(dev) && cJSON_IsString(sec) && strlen(dev->valuestring) < sizeof(S->dev) &&
            strlen(sec->valuestring) < sizeof(S->sec);
  if (ok) {
    // Stored before anything is reported: the code is spent now, so losing the secret means a new code.
    ok = nvs_put_string("c_url", base) && nvs_put_string("c_space", space) && nvs_put_string("c_dev", dev->valuestring) &&
         nvs_put_string("c_sec", sec->valuestring);
    if (ok) {
      snprintf(S->url, sizeof(S->url), "%s", base);
      snprintf(S->space, sizeof(S->space), "%s", space);
      snprintf(S->dev, sizeof(S->dev), "%s", dev->valuestring);
      snprintf(S->sec, sizeof(S->sec), "%s", sec->valuestring);
    }
  }
  cJSON_Delete(j);
  memset(S->resp, 0, RESP_CAP);  // held the secret
  if (!ok) {
    ESP_LOGE(TAG, "claim reply could not be stored: create a new claim code (devices.py create)");
    return ESP_FAIL;
  }
  S->token[0] = 0;
  s_needs_setup = false;
  s_skip_until_us = 0;
  ESP_LOGI(TAG, "claimed: device %s, space %s (secret stored in NVS)", S->dev, S->space);
  if (fetch_token()) ESP_LOGI(TAG, "device token ok");
  return ESP_OK;
}

void cloud_link_forget(void) {
  if (!S) return;
  const char *keys[] = {"c_url", "c_space", "c_dev", "c_sec"};
  for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) nvs_put_string(keys[i], NULL);
  memset(S->sec, 0, sizeof(S->sec));
  S->url[0] = S->space[0] = S->dev[0] = S->token[0] = 0;
  client_drop();
  ESP_LOGI(TAG, "cloud credentials erased");
}

void cloud_link_set_room(const char *room) {
  if (!S) return;
  snprintf(S->room, sizeof(S->room), "%s", room ? room : "");
  trim(S->room);
  nvs_put_string("c_room", S->room);
  ESP_LOGI(TAG, "room %s", S->room[0] ? S->room : "(none)");
}

void cloud_link_print_status(void) {
  if (!S) return;
  int64_t now = esp_timer_get_time();
  int token_s = S->token[0] && now < s_token_until_us ? (int)((s_token_until_us - now) / 1000000) : 0;
  int skip_s = now < s_skip_until_us ? (int)((s_skip_until_us - now) / 1000000) : 0;
  ESP_LOGI(TAG, "%s | space %s | device %s | room %s | token %s%d s | last HTTP %d%s | clock %s",
           S->url[0] ? S->url : "(not claimed)", S->space[0] ? S->space : "-", S->dev[0] ? S->dev : "-",
           S->room[0] ? S->room : "-", token_s ? "renews in " : "none ", token_s, s_last_status,
           s_needs_setup ? " | NEEDS A NEW CLAIM" : skip_s ? " | paused" : "", cloud_link_now_ms() ? "set" : "NOT SET");
  ESP_LOGI(TAG, "internal free %u B, largest block %u B%s",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
#if CONFIG_IDF_TARGET_ESP32P4
           heap_caps_get_free_size(MALLOC_CAP_RETENTION) < 8192 ? " | retention RAM LOW" : ""
#else
           ""
#endif
  );
}

// --- answers -------------------------------------------------------------------------------------
// Play a streamed 16 kHz PCM16 body (or just print the answer on a board without a speaker).
static void play_reply(esp_http_client_handle_t c) {
  char text[384];
  if (S->hdr_heard[0]) {
    url_decode(S->hdr_heard, text, sizeof(text));
    ESP_LOGI(TAG, "heard: \"%s\"", text);
  }
  url_decode(S->hdr_answer, text, sizeof(text));
  if (text[0]) ESP_LOGI(TAG, "answer: %s", text);
  if (!audio_speaker_ready()) {
    esp_http_client_flush_response(c, NULL);
    return;
  }
  int64_t content_len = esp_http_client_get_content_length(c);
  size_t total = 0, carry = 0;
  int64_t t0 = esp_timer_get_time();
  for (;;) {
    int n = esp_http_client_read(c, (char *)S->io + carry, (int)(sizeof(S->io) - carry));
    if (n <= 0) break;
    size_t have = carry + (size_t)n;
    size_t even = have & ~(size_t)1;
    audio_play_pcm16((const int16_t *)S->io, even / 2);
    carry = have - even;
    if (carry) S->io[0] = S->io[even];
    total += (size_t)n;
    if (content_len > 0 && (int64_t)total >= content_len) break;
  }
  ESP_LOGI(TAG, "played %.1f s from the cloud in %lld ms", total / (2.0f * LOCI_AUDIO_RATE),
           (esp_timer_get_time() - t0) / 1000);
}

esp_err_t cloud_link_ask(const char *text) {
  if (!cloud_link_usable()) return ESP_ERR_INVALID_STATE;
  char url[256], escaped[512], body[560];
  snprintf(url, sizeof(url), "%s/spaces/%s/board/ask", S->url, S->space);
  json_escape(text, escaped, sizeof(escaped));
  snprintf(body, sizeof(body), "{\"text\":\"%s\"}", escaped);
  esp_http_client_handle_t c = NULL;
  int status = authed(url, HTTP_METHOD_POST, "application/json", body, strlen(body), 10000, &c);
  if (status != 200) {
    if (c && status > 0) read_json(c);
    ESP_LOGW(TAG, "ask -> HTTP %d %s", status, c && status > 0 ? S->resp : "");
    failed(status);
    return ESP_FAIL;
  }
  play_reply(c);
  return ESP_OK;
}

esp_err_t cloud_link_voice(const uint8_t *pcm, size_t len) {
  if (!cloud_link_usable()) return ESP_ERR_INVALID_STATE;
  char url[256];
  snprintf(url, sizeof(url), "%s/spaces/%s/board/voice", S->url, S->space);
  esp_http_client_handle_t c = NULL;
  // Upload, Whisper, the answer and its speech: 12 s, then the board answers from the card.
  int status = authed(url, HTTP_METHOD_POST, "audio/L16; rate=16000", pcm, len, 12000, &c);
  if (status == 422 && c) {
    read_json(c);
    if (strstr(S->resp, "not_heard")) {
      ESP_LOGI(TAG, "the cloud heard nothing");
      return ESP_ERR_NOT_FOUND;
    }
  }
  if (status != 200) {
    if (c && status > 0 && status != 422) read_json(c);
    ESP_LOGW(TAG, "voice -> HTTP %d %s", status, c && status > 0 ? S->resp : "");
    failed(status);
    return ESP_FAIL;
  }
  play_reply(c);
  return ESP_OK;
}

// --- cache files -----------------------------------------------------------------------------
esp_err_t cloud_link_fetch_file(const char *host_path, const char *dest_path, size_t *bytes_out) {
  static const char prefix[] = "/api/esp32/";
  if (strncmp(host_path, prefix, sizeof(prefix) - 1) != 0) return ESP_ERR_INVALID_ARG;
  if (!cloud_link_usable()) return ESP_ERR_INVALID_STATE;
  char url[600], tmp[128];
  snprintf(url, sizeof(url), "%s/spaces/%s/board/%s", S->url, S->space, host_path + sizeof(prefix) - 1);
  snprintf(tmp, sizeof(tmp), "%s.part", dest_path);
  esp_http_client_handle_t c = NULL;
  int status = authed(url, HTTP_METHOD_GET, NULL, NULL, 0, 15000, &c);
  if (status == 204) {
    esp_http_client_flush_response(c, NULL);
    return ESP_ERR_NOT_FOUND;
  }
  if (status != 200) {
    if (c && status > 0) read_json(c);
    ESP_LOGW(TAG, "%s -> HTTP %d", host_path, status);
    failed(status);
    return ESP_FAIL;
  }
  FILE *f = fopen(tmp, "wb");
  if (!f) {
    esp_http_client_flush_response(c, NULL);
    return ESP_FAIL;
  }
  int64_t content_len = esp_http_client_get_content_length(c);
  size_t total = 0;
  bool ok = true;
  for (;;) {
    int n = esp_http_client_read(c, (char *)S->io, sizeof(S->io));
    if (n < 0) {
      ok = false;
      break;
    }
    if (n == 0) break;
    if (fwrite(S->io, 1, (size_t)n, f) != (size_t)n) {
      ok = false;
      break;
    }
    total += (size_t)n;
    if (content_len > 0 && (int64_t)total >= content_len) break;
  }
  fclose(f);
  if (!ok || total == 0) {
    unlink(tmp);
    failed(-1);
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

// --- sightings ---------------------------------------------------------------------------------
esp_err_t cloud_link_post_observations(const char *json, size_t len, int *written) {
  if (written) *written = 0;
  if (!cloud_link_usable()) return ESP_ERR_INVALID_STATE;
  char url[256];
  snprintf(url, sizeof(url), "%s/spaces/%s/observations", S->url, S->space);
  esp_http_client_handle_t c = NULL;
  int status = authed(url, HTTP_METHOD_POST, "application/json", json, len, 10000, &c);
  if (c && status > 0) read_json(c);
  if (status != 200 && status != 201) {
    ESP_LOGW(TAG, "observations -> HTTP %d %s", status, c && status > 0 ? S->resp : "");
    failed(status);
    // A refused batch (4xx other than 429) cannot get better by retrying it.
    return status >= 400 && status < 500 && status != 429 ? ESP_ERR_INVALID_RESPONSE : ESP_FAIL;
  }
  cJSON *j = cJSON_Parse(S->resp);
  const cJSON *a = cJSON_GetObjectItem(j, "accepted");
  const cJSON *u = cJSON_GetObjectItem(j, "updated");
  if (written) *written = (cJSON_IsNumber(a) ? a->valueint : 0) + (cJSON_IsNumber(u) ? u->valueint : 0);
  cJSON_Delete(j);
  return ESP_OK;
}
