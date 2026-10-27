// Meshnology W11 "ESP32S3 Mini Module" + its expansion board (DVP camera, PDM microphone,
// microSD over SPI), verified against the vendor wiki and the board's factory test firmware.
// The pin map is the Seeed XIAO ESP32S3 Sense one (the vendor's Arduino board setting is
// XIAO_ESP32S3), so a XIAO ESP32S3 Sense runs this build unchanged. Every pin can be overridden
// with -D flags (idf_build_set_property(COMPILE_DEFINITIONS ...) in the project CMakeLists.txt).
#pragma once

#ifndef BOARD_NAME
#define BOARD_NAME "Meshnology W11 (ESP32-S3)"
#endif

// DVP camera on the expansion board's 24-pin FPC (OV3660 on the desk; the vendor also ships OV5640).
// GPIO48 doubles as the on-board WS2812 RGB LED: the LED cannot be used while the camera runs.
#define BOARD_CAM_PWDN -1
#define BOARD_CAM_RESET -1
#define BOARD_CAM_XCLK 10
#define BOARD_CAM_SIOD 40
#define BOARD_CAM_SIOC 39
#define BOARD_CAM_D7 48
#define BOARD_CAM_D6 11
#define BOARD_CAM_D5 12
#define BOARD_CAM_D4 14
#define BOARD_CAM_D3 16
#define BOARD_CAM_D2 18
#define BOARD_CAM_D1 17
#define BOARD_CAM_D0 15
#define BOARD_CAM_VSYNC 38
#define BOARD_CAM_HREF 47
#define BOARD_CAM_PCLK 13
// 10 MHz, not the usual 20: at 20 MHz (and the faster pixel clock of XGA) the camera interfered with
// the Wi-Fi (pings 0.2-8 s, up to 87 % loss at -49 dBm) and with the microSD's SPI lines on the
// shared expansion board (card init CRC errors). 10 MHz still gives the sensor frames to spare at 2 fps.
#ifndef BOARD_CAM_XCLK_HZ
#define BOARD_CAM_XCLK_HZ 10000000
#endif

// PDM microphone on the expansion board (I2S0 in PDM RX mode)
#ifndef BOARD_MIC_CLK
#define BOARD_MIC_CLK 42
#endif
#ifndef BOARD_MIC_DATA
#define BOARD_MIC_DATA 41
#endif

// microSD on SPI2; CS is shared with the active-low USER LED
#ifndef BOARD_SD_SCK
#define BOARD_SD_SCK 7
#endif
#ifndef BOARD_SD_MISO
#define BOARD_SD_MISO 8
#endif
#ifndef BOARD_SD_MOSI
#define BOARD_SD_MOSI 9
#endif
#ifndef BOARD_SD_CS
#define BOARD_SD_CS 21
#endif
#ifndef BOARD_SD_FREQ_KHZ
#define BOARD_SD_FREQ_KHZ 10000  // 20 MHz worked until the camera clock disturbed it; 10 MHz has margin
#endif

// QMI8658C 6-axis IMU on its own I2C bus (the camera's SCCB uses another port)
#ifndef BOARD_IMU_SDA
#define BOARD_IMU_SDA 5
#endif
#ifndef BOARD_IMU_SCL
#define BOARD_IMU_SCL 4
#endif

// BOOT button (strapping GPIO0, free after reset) doubles as push-to-talk; -1 disables it
#ifndef BOARD_BTN_PTT
#define BOARD_BTN_PTT 0
#endif

#include "loci_config.h"  // LOCI_AUDIO_RATE, LOCI_FPS, microSD usage and other board-independent settings

// esp32-camera JPEG quality: 0..63, lower is better (10-12 is good for detection)
#ifndef LOCI_JPEG_QUALITY
#define LOCI_JPEG_QUALITY 12
#endif
// FRAMESIZE_* of esp32-camera. XGA (1024x768, 60-100 kB) is all the host's YOLO uses at
// imgsz 960; UXGA (1600x1200) reached 370 kB in a bright room and overflowed its 384 kB buffer.
#ifndef LOCI_FRAME_SIZE
#define LOCI_FRAME_SIZE FRAMESIZE_XGA
#endif
// Largest size `res` may switch to at runtime (the frame buffers are allocated for it)
#ifndef LOCI_FRAME_SIZE_MAX
#define LOCI_FRAME_SIZE_MAX FRAMESIZE_UXGA
#endif
// PDM microphones are quiet: linear gain after the DC blocker
#ifndef LOCI_MIC_GAIN
#define LOCI_MIC_GAIN 8
#endif
