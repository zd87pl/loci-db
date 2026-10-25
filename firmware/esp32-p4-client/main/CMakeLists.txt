idf_component_register(
    SRCS "app_main.c" "camera.c" "audio.c" "wifi_link.c" "host_link.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_wifi esp_netif esp_event nvs_flash esp_http_client esp_timer
                  esp_driver_i2s esp_driver_i2c esp_driver_gpio esp_driver_uart vfs
)
