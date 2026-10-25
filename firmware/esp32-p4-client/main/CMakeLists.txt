idf_component_register(
    SRCS "app_main.c" "camera.c" "audio.c" "wifi_link.c" "host_link.c" "storage.c" "local_memory.c" "voice_local.c"
    INCLUDE_DIRS "."
    LDFRAGMENTS "linker.lf"
    PRIV_REQUIRES esp_wifi esp_netif esp_event nvs_flash esp_http_client esp_timer
                  esp_driver_i2s esp_driver_i2c esp_driver_gpio esp_driver_uart vfs
                  fatfs sdmmc esp_driver_sdmmc esp_driver_sd_intf esp-sr
)
