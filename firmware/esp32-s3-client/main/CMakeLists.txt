idf_component_register(
    SRCS "app_main.c" "camera.c" "audio.c" "storage_board.c"
    INCLUDE_DIRS "."
    # loci_client calls back into this component (loci_hw.h): keep every object, not only referenced ones
    WHOLE_ARCHIVE
    PRIV_REQUIRES loci_client esp_driver_i2s esp_driver_i2c esp_driver_gpio esp_driver_spi esp_driver_sdspi
                  sdmmc spi_flash esp_timer loci_detect
)
