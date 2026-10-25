#include "storage.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "diskio_impl.h"
#include "diskio_sdmmc.h"
#include "driver/sdmmc_host.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "ff.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "sdmmc_cmd.h"

#include "board_p4_nano.h"

static const char *TAG = "storage";
static sdmmc_card_t *s_card;  // non-NULL once the card is initialised and registered with FatFs/VFS
static sdmmc_host_t s_host;
static sd_pwr_ctrl_handle_t s_pwr;
static FATFS *s_fs;
static BYTE s_pdrv = FF_DRV_NOT_USED;
static char s_drive[3] = "0:";
static storage_info_t s_info;

// ESP-Hosted drives the ESP32-C6 through SDMMC slot 1 and has created the one SDMMC controller
// long before the app runs; a second sdmmc_host_init() returns ESP_ERR_NOT_FOUND on IDF 6.1 and
// the all-in-one FAT mount helper gives up on that. So the card on slot 0 is brought up by hand on
// the shared controller: slot init -> card init -> FatFs drive -> VFS -> f_mount, one log line each.
// The card stays registered after a "no filesystem" result so storage_init(true) can format later.

static esp_err_t bring_up_card(void) {
  s_host = (sdmmc_host_t)SDMMC_HOST_DEFAULT();
  s_host.slot = SDMMC_HOST_SLOT_0;  // CLK 43, CMD 44, D0-D3 39-42: the slot-0 IOMUX pins on the P4
  s_host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;

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
    ESP_LOGW(TAG, "no card or card init failed: %s", esp_err_to_name(err));
    goto fail_card;
  }
  snprintf(s_info.name, sizeof(s_info.name), "%.7s", card->cid.name);
  s_info.capacity_bytes = (uint64_t)card->csd.capacity * card->csd.sector_size;
  s_info.speed_khz = card->real_freq_khz;
  s_info.bus_width = card->log_bus_width ? (1 << card->log_bus_width) : 1;
  ESP_LOGI(TAG, "card \"%s\": %.1f GB, %d-bit @ %d kHz", s_info.name, s_info.capacity_bytes / 1e9,
           s_info.bus_width, s_info.speed_khz);

  err = ff_diskio_get_drive(&s_pdrv);
  if (err != ESP_OK || s_pdrv == FF_DRV_NOT_USED) {
    ESP_LOGW(TAG, "no free FatFs drive slot (%d volumes configured)", FF_VOLUMES);
    err = ESP_ERR_NO_MEM;
    goto fail_card;
  }
  ff_diskio_register_sdmmc(s_pdrv, card);
  s_drive[0] = (char)('0' + s_pdrv);

  esp_vfs_fat_conf_t conf = {.base_path = STORAGE_MOUNT, .fat_drive = s_drive, .max_files = 8};
  err = esp_vfs_fat_register(&conf, &s_fs);
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGW(TAG, "VFS registration failed: %s", esp_err_to_name(err));
    goto fail_diskio;
  }
  s_card = card;
  return ESP_OK;

fail_diskio:
  ff_diskio_unregister(s_pdrv);
  s_pdrv = FF_DRV_NOT_USED;
fail_card:
  free(card);
fail_slot:
  sdmmc_host_deinit_slot(SDMMC_HOST_SLOT_0);
fail_pwr:
  sd_pwr_ctrl_del_on_chip_ldo(s_pwr);
  s_pwr = NULL;
  s_host.pwr_ctrl_handle = NULL;
  return err;
}

esp_err_t storage_init(bool allow_format) {
  if (s_info.mounted) return ESP_OK;
  if (!s_card) {
    memset(&s_info, 0, sizeof(s_info));
    esp_err_t err = bring_up_card();
    if (err != ESP_OK) return err;
  }

  FRESULT res = f_mount(s_fs, s_drive, 1);
  if (res == FR_NO_FILESYSTEM || res == FR_INT_ERR) {
    s_info.format_needed = true;
    if (!allow_format) {
      ESP_LOGW(TAG, "card has no FAT filesystem (exFAT or blank, f_mount=%d): format it as FAT32 to use it", res);
      return ESP_FAIL;
    }
    ESP_LOGW(TAG, "formatting the card as FAT32 (f_mount=%d)...", res);
    const size_t workbuf_size = 64 * 1024;
    void *workbuf = ff_memalloc(workbuf_size);
    if (!workbuf) return ESP_ERR_NO_MEM;
    MKFS_PARM opt = {.fmt = FM_FAT32, .au_size = 64 * 1024};
    res = f_mkfs(s_drive, &opt, workbuf, workbuf_size);
    free(workbuf);
    if (res != FR_OK) {
      ESP_LOGE(TAG, "f_mkfs failed (%d)", res);
      return ESP_FAIL;
    }
    res = f_mount(s_fs, s_drive, 1);
    s_info.format_needed = res != FR_OK;
  }
  if (res != FR_OK) {
    ESP_LOGW(TAG, "f_mount failed (%d)", res);
    return ESP_FAIL;
  }

  s_info.mounted = true;
  char dir[64];
  storage_dir("", dir, sizeof(dir));
  storage_info();
  ESP_LOGI(TAG, "microSD mounted at %s: %.1f GB free of %.1f GB", STORAGE_MOUNT, s_info.free_bytes / 1e9,
           s_info.capacity_bytes / 1e9);
  return ESP_OK;
}

bool storage_ready(void) { return s_card != NULL && s_info.mounted; }

storage_info_t storage_info(void) {
  if (storage_ready()) {
    uint64_t total = 0, free_b = 0;
    if (esp_vfs_fat_info(STORAGE_MOUNT, &total, &free_b) == ESP_OK) s_info.free_bytes = free_b;
  }
  return s_info;
}

const char *storage_dir(const char *sub, char *out, size_t cap) {
  if (sub && sub[0]) {
    snprintf(out, cap, "%s/%s", STORAGE_DIR, sub);
  } else {
    snprintf(out, cap, "%s", STORAGE_DIR);
  }
  if (!storage_ready()) return out;
  struct stat st;
  if (stat(STORAGE_DIR, &st) != 0) mkdir(STORAGE_DIR, 0775);
  if (sub && sub[0] && stat(out, &st) != 0) mkdir(out, 0775);
  return out;
}

int storage_write_file(const char *path, const void *data, size_t len) {
  if (!storage_ready()) return -1;
  FILE *f = fopen(path, "wb");
  if (!f) {
    ESP_LOGW(TAG, "fopen %s failed: %d", path, errno);
    return -1;
  }
  size_t n = fwrite(data, 1, len, f);
  fclose(f);
  return (int)n;
}
