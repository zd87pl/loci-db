// Waveshare ESP32-P4-NANO pin map (verified against the vendor BSP header and the board's
// original firmware log). Everything can be overridden with -D flags if a different P4 board
// is wired the same way.
#pragma once

#define BOARD_NAME "Waveshare ESP32-P4-NANO"

// Shared I2C bus: ES8311 codec (0x18), camera SCCB, touch controller when a display is attached
#ifndef BOARD_I2C_PORT
#define BOARD_I2C_PORT 0
#endif
#ifndef BOARD_I2C_SDA
#define BOARD_I2C_SDA 7
#endif
#ifndef BOARD_I2C_SCL
#define BOARD_I2C_SCL 8
#endif

// ES8311 on I2S0; NS4150B power amplifier enable on GPIO53 (active high)
#ifndef BOARD_I2S_PORT
#define BOARD_I2S_PORT 0
#endif
#ifndef BOARD_I2S_MCLK
#define BOARD_I2S_MCLK 13
#endif
#ifndef BOARD_I2S_BCLK
#define BOARD_I2S_BCLK 12
#endif
#ifndef BOARD_I2S_WS
#define BOARD_I2S_WS 10
#endif
#ifndef BOARD_I2S_DOUT
#define BOARD_I2S_DOUT 9  // P4 -> codec DAC (DSDIN)
#endif
#ifndef BOARD_I2S_DIN
#define BOARD_I2S_DIN 11  // codec ADC (ASDOUT) -> P4
#endif
#ifndef BOARD_PA_PIN
#define BOARD_PA_PIN 53
#endif

// MIPI-CSI camera: 2 lanes on the dedicated connector; control over the shared I2C bus
#ifndef BOARD_CAM_SCCB_FREQ
#define BOARD_CAM_SCCB_FREQ 100000
#endif
#ifndef BOARD_CAM_RESET
#define BOARD_CAM_RESET -1
#endif
#ifndef BOARD_CAM_PWDN
#define BOARD_CAM_PWDN -1
#endif

// microSD: SDMMC slot 0 (IOMUX: CLK 43, CMD 44, D0 39, D1 40, D2 41, D3 42), card power from LDO channel 4
#ifndef BOARD_SD_LDO_CHANNEL
#define BOARD_SD_LDO_CHANNEL 4
#endif

// BOOT button (strapping GPIO35) doubles as push-to-talk; -1 disables it
#ifndef BOARD_BTN_PTT
#define BOARD_BTN_PTT 35
#endif

#include "loci_config.h"  // LOCI_AUDIO_RATE, LOCI_FPS, microSD usage and other board-independent settings

#ifndef LOCI_JPEG_QUALITY
#define LOCI_JPEG_QUALITY 80
#endif
#ifndef LOCI_SPK_VOLUME
#define LOCI_SPK_VOLUME 75  // 0..100
#endif
#ifndef LOCI_MIC_GAIN_DB
#define LOCI_MIC_GAIN_DB 36.0f
#endif
