// microSD slot of the ESP32-P4-NANO: SDMMC slot 0, 4-bit, card power from on-chip LDO channel 4.
//
// ESP-Hosted drives the ESP32-C6 through SDMMC slot 1 and has created the one SDMMC controller
// long before the app runs; a second sdmmc_host_init() returns ESP_ERR_NOT_FOUND on IDF 6.1 and
// the all-in-one FAT mount helper gives up on that. So the card on slot 0 is brought up by hand on
// the shared controller (slot init -> card init); storage.c does FatFs, VFS and the mount.

#include <stdlib.h>

#include "driver/sdmmc_host.h"
#include "esp_check.h"
#include "esp_log.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "sdmmc_cmd.h"

#include "board_p4_nano.h"
#include "loci_hw.h"

static const char *TAG = "storage";
static sdmmc_host_t s_host;
static sd_pwr_ctrl_handle_t s_pwr;

esp_err_t storage_board_open_card(sdmmc_card_t **card_out) {
  s_host = (sdmmc_host_t)SDMMC_HOST_DEFAULT();
  s_host.slot = SDMMC_HOST_SLOT_0;  // CLK 43, CMD 44, D0-D3 39-42: the slot-0 IOMUX pins on the P4
  s_host.max_freq_khz = SDMMC_FREQ_DEFAULT;  // 20 MHz, the same clock as the C6's SDIO slot (sdkconfig.defaults)

  esp_err_t err = sdmmc_host_init();
  if (err == ESP_OK) {
    ESP_LOGI(TAG, "SDMMC controller created by the app");
  } else {
    ESP_LOGI(TAG, "SDMMC controller already up (%s); sharing it with ESP-Hosted", esp_err_to_name(err));
  }

  sd_pwr_ctrl_ldo_config_t ldo_cfg = {.ldo_chan_id = BOARD_SD_LDO_CHANNEL};
  ESP_RETURN_ON_ERROR(sd_pwr_ctrl_new_on_chip_ldo(&ldo_cfg, &s_pwr), TAG, "SD LDO power control");
  s_host.pwr_ctrl_handle = s_pwr;

  sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
  slot.cd = SDMMC_SLOT_NO_CD;
  slot.wp = SDMMC_SLOT_NO_WP;
  slot.width = 4;
  slot.flags = 0;
  err = sdmmc_host_init_slot(SDMMC_HOST_SLOT_0, &slot);
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "slot 0 init failed: %s", esp_err_to_name(err));
    goto fail_pwr;
  }

  sdmmc_card_t *card = calloc(1, sizeof(sdmmc_card_t));
  if (!card) {
    err = ESP_ERR_NO_MEM;
    goto fail_slot;
  }
  err = sdmmc_card_init(&s_host, card);
  if (err != ESP_OK) {
    free(card);
    goto fail_slot;
  }
  *card_out = card;
  return ESP_OK;

fail_slot:
  sdmmc_host_deinit_slot(SDMMC_HOST_SLOT_0);
fail_pwr:
  sd_pwr_ctrl_del_on_chip_ldo(s_pwr);
  s_pwr = NULL;
  s_host.pwr_ctrl_handle = NULL;
  return err;
}

void storage_board_close_card(sdmmc_card_t *card) {
  free(card);
  sdmmc_host_deinit_slot(SDMMC_HOST_SLOT_0);
  sd_pwr_ctrl_del_on_chip_ldo(s_pwr);
  s_pwr = NULL;
  s_host.pwr_ctrl_handle = NULL;
}
