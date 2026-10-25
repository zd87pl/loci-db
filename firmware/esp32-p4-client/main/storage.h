#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

// microSD on the ESP32-P4-NANO (SDMMC slot 0, 4-bit, powered by on-chip LDO channel 4), mounted at
// STORAGE_MOUNT with FAT. A card formatted as exFAT (the factory default above 32 GB) cannot be
// mounted by IDF's FatFs; storage_init reports that so the caller can decide to format.
#define STORAGE_MOUNT "/sdcard"
#define STORAGE_DIR STORAGE_MOUNT "/loci"

typedef struct {
  bool mounted;
  bool format_needed;  // mount failed with "no filesystem": exFAT or blank card
  char name[8];
  uint64_t capacity_bytes;
  uint64_t free_bytes;
  int speed_khz;
  int bus_width;
} storage_info_t;

// Mount the card. When allow_format is true a card without a FAT filesystem is formatted (FAT32).
esp_err_t storage_init(bool allow_format);
bool storage_ready(void);
storage_info_t storage_info(void);  // refreshes free space when mounted
// Make sure STORAGE_DIR/<sub> exists; returns the full path in `out`.
const char *storage_dir(const char *sub, char *out, size_t cap);
// Write a whole buffer to a file (creates/truncates). Returns bytes written or -1.
int storage_write_file(const char *path, const void *data, size_t len);
