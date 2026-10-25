#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

// Station mode on the ESP32-C6 companion (esp_wifi_remote over ESP-Hosted/SDIO).
esp_err_t wifi_link_start(const char *ssid, const char *password);
bool wifi_link_connected(void);
// Block up to timeout_ms for an IP address; returns true when connected.
bool wifi_link_wait(uint32_t timeout_ms);
int wifi_link_rssi(void);
const char *wifi_link_ip(void);
