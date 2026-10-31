#include "wifi_link.h"

#include <string.h>

#include "esp_check.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

static const char *TAG = "wifi";
static EventGroupHandle_t s_events;
static char s_ip[16] = "0.0.0.0";
static int s_retries;
static bool s_paused;  // `wifi off`: stay disconnected until `wifi on`
#define CONNECTED_BIT BIT0

static void on_wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
    xEventGroupClearBits(s_events, CONNECTED_BIT);
    if (s_paused) {
      ESP_LOGW(TAG, "disconnected (wifi off)");
      return;
    }
    s_retries++;
    ESP_LOGW(TAG, "disconnected (reason %d), reconnecting (%d)",
             ((wifi_event_sta_disconnected_t *)data)->reason, s_retries);
    vTaskDelay(pdMS_TO_TICKS(s_retries > 5 ? 3000 : 500));
    esp_wifi_connect();
  } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
    snprintf(s_ip, sizeof(s_ip), IPSTR, IP2STR(&ev->ip_info.ip));
    s_retries = 0;
    ESP_LOGI(TAG, "got ip %s", s_ip);
    xEventGroupSetBits(s_events, CONNECTED_BIT);
  }
}

esp_err_t wifi_link_start(const char *ssid, const char *password) {
  s_events = xEventGroupCreate();
  ESP_RETURN_ON_ERROR(esp_netif_init(), TAG, "netif");
  ESP_RETURN_ON_ERROR(esp_event_loop_create_default(), TAG, "event loop");
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_RETURN_ON_ERROR(esp_wifi_init(&cfg), TAG, "esp_wifi_init (on the P4: is the C6 running the hosted slave?)");
  ESP_RETURN_ON_ERROR(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &on_wifi_event, NULL), TAG, "h1");
  ESP_RETURN_ON_ERROR(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &on_wifi_event, NULL), TAG, "h2");

  wifi_config_t wcfg = {0};
  strncpy((char *)wcfg.sta.ssid, ssid, sizeof(wcfg.sta.ssid) - 1);
  strncpy((char *)wcfg.sta.password, password, sizeof(wcfg.sta.password) - 1);
  wcfg.sta.threshold.authmode = strlen(password) ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
  wcfg.sta.pmf_cfg.capable = true;
  ESP_RETURN_ON_ERROR(esp_wifi_set_mode(WIFI_MODE_STA), TAG, "mode");
  ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_STA, &wcfg), TAG, "config");
  ESP_RETURN_ON_ERROR(esp_wifi_start(), TAG, "start");
  esp_wifi_set_ps(WIFI_PS_NONE);  // latency matters more than power on a desk demo
  ESP_LOGI(TAG, "connecting to %s", ssid);
  return ESP_OK;
}

bool wifi_link_connected(void) {
  return s_events && (xEventGroupGetBits(s_events) & CONNECTED_BIT);
}

bool wifi_link_wait(uint32_t timeout_ms) {
  if (!s_events) return false;
  EventBits_t bits = xEventGroupWaitBits(s_events, CONNECTED_BIT, pdFALSE, pdTRUE, pdMS_TO_TICKS(timeout_ms));
  return (bits & CONNECTED_BIT) != 0;
}

int wifi_link_rssi(void) {
  wifi_ap_record_t ap;
  if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) return ap.rssi;
  return 0;
}

const char *wifi_link_ip(void) { return s_ip; }

void wifi_link_pause(bool off) {
  if (!s_events || off == s_paused) return;
  s_paused = off;
  if (off) {
    esp_wifi_disconnect();
  } else {
    s_retries = 0;
    esp_wifi_connect();
  }
}
