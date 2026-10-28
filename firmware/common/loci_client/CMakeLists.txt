# Board-independent part of the LOCI thin client (host link, cloud link, on-board sightings), shared
# by firmware/esp32-p4-client and firmware/esp32-s3-client (each adds this directory to
# EXTRA_COMPONENT_DIRS). The board project
# implements loci_hw.h (camera, audio, microSD slot) and calls loci_app_run() from app_main().
idf_component_register(
    SRCS "loci_app.c" "host_link.c" "cloud_link.c" "sightings.c" "local_memory.c" "storage.c" "voice_local.c"
         "wifi_link.c"
    INCLUDE_DIRS "include"
    REQUIRES sdmmc
    PRIV_REQUIRES esp_wifi esp_netif esp_event nvs_flash esp_http_client esp_timer esp_driver_gpio
                  esp_driver_uart esp_driver_usb_serial_jtag vfs fatfs esp-sr mbedtls espressif__cjson
)
