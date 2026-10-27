// microSD slot of the W11 expansion board: SPI mode on SPI2 (SCK 7, MISO 8, MOSI 9, CS 21, the
// XIAO ESP32S3 Sense wiring). storage.c in loci_client does FatFs, VFS and the mount.

#include <stdlib.h>

#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdmmc_cmd.h"

#include "board_w11.h"
#include "loci_hw.h"

static const char *TAG = "storage";
#define SD_SPI_HOST SPI2_HOST
static sdspi_dev_handle_t s_dev = -1;
static bool s_bus_up;

esp_err_t storage_board_open_card(sdmmc_card_t **card_out) {
  if (!s_bus_up) {
    spi_bus_config_t bus = {
        .mosi_io_num = BOARD_SD_MOSI,
        .miso_io_num = BOARD_SD_MISO,
        .sclk_io_num = BOARD_SD_SCK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 16 * 1024,
    };
    ESP_RETURN_ON_ERROR(spi_bus_initialize(SD_SPI_HOST, &bus, SDSPI_DEFAULT_DMA), TAG, "SPI bus");
    s_bus_up = true;
  }
  esp_err_t err = sdspi_host_init();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) return err;

  sdspi_device_config_t dev = SDSPI_DEVICE_CONFIG_DEFAULT();
  dev.host_id = SD_SPI_HOST;
  dev.gpio_cs = BOARD_SD_CS;
  ESP_RETURN_ON_ERROR(sdspi_host_init_device(&dev, &s_dev), TAG, "SD SPI device");

  sdmmc_host_t host = SDSPI_HOST_DEFAULT();
  host.slot = s_dev;
  host.max_freq_khz = BOARD_SD_FREQ_KHZ;
  sdmmc_card_t *card = calloc(1, sizeof(sdmmc_card_t));
  if (!card) {
    err = ESP_ERR_NO_MEM;
    goto fail;
  }
  // A soft reset of the S3 does not power-cycle the card; one that was mid-write can answer the
  // first init sequences with garbage (CRC / "send_if_cond" errors). Give it a few tries.
  for (int attempt = 0; attempt < 4; attempt++) {
    err = sdmmc_card_init(&host, card);  // the card keeps its own copy of `host`
    if (err == ESP_OK) break;
    ESP_LOGW(TAG, "card init attempt %d: %s", attempt + 1, esp_err_to_name(err));
    vTaskDelay(pdMS_TO_TICKS(250));
  }
  if (err != ESP_OK) {
    free(card);
    goto fail;
  }
  *card_out = card;
  return ESP_OK;

fail:
  sdspi_host_remove_device(s_dev);
  s_dev = -1;
  return err;
}

void storage_board_close_card(sdmmc_card_t *card) {
  free(card);
  if (s_dev >= 0) sdspi_host_remove_device(s_dev);
  s_dev = -1;
}
