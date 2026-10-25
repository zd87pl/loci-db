// Pin maps for the boards the LOCI thin client has been built for.
//
// Select a board with one of the -DLOCI_BOARD_* flags in platformio.ini. Every pin can be
// overridden individually with -DLOCI_<NAME>=<gpio>, so a different amplifier wiring or a
// board that is not listed here needs no code change: define LOCI_BOARD_CUSTOM and pass all
// camera pins on the command line.
//
// Conventions
//   LOCI_CAM_*      DVP camera pins as expected by esp32-camera (camera_config_t)
//   LOCI_I2S_*      speaker amplifier (MAX98357A / PCM5102 / any I2S DAC): BCLK, LRC (WS), DOUT
//   LOCI_MIC_*      PDM microphone (CLK, DATA); LOCI_HAS_MIC=0 when the board has none
//   LOCI_BTN_PTT    push-to-talk button to GND (internal pull-up); -1 = no button
//   LOCI_LED        status LED (-1 = none); LOCI_LED_INVERTED=1 when active-low
#pragma once

#define LOCI_PIN_NONE -1

// ------------------------------------------------------------------------------------
// Seeed Studio XIAO ESP32S3 Sense (OV2640 camera board + PDM mic + SD slot)
// Header: D0=GPIO1 D1=GPIO2 D2=GPIO3 D3=GPIO4 D4=GPIO5(SDA) D5=GPIO6(SCL) D6=GPIO43(TX)
//         D7=GPIO44(RX) D8=GPIO7(SCK) D9=GPIO8(MISO) D10=GPIO9(MOSI)
// ------------------------------------------------------------------------------------
#if defined(LOCI_BOARD_XIAO_S3_SENSE)
#define LOCI_BOARD_NAME "Seeed XIAO ESP32S3 Sense"
#define LOCI_CAM_PWDN LOCI_PIN_NONE
#define LOCI_CAM_RESET LOCI_PIN_NONE
#define LOCI_CAM_XCLK 10
#define LOCI_CAM_SIOD 40
#define LOCI_CAM_SIOC 39
#define LOCI_CAM_Y9 48
#define LOCI_CAM_Y8 11
#define LOCI_CAM_Y7 12
#define LOCI_CAM_Y6 14
#define LOCI_CAM_Y5 16
#define LOCI_CAM_Y4 18
#define LOCI_CAM_Y3 17
#define LOCI_CAM_Y2 15
#define LOCI_CAM_VSYNC 38
#define LOCI_CAM_HREF 47
#define LOCI_CAM_PCLK 13
#ifndef LOCI_HAS_MIC
#define LOCI_HAS_MIC 1
#endif
#ifndef LOCI_MIC_CLK
#define LOCI_MIC_CLK 42
#endif
#ifndef LOCI_MIC_DATA
#define LOCI_MIC_DATA 41
#endif
#ifndef LOCI_I2S_BCLK
#define LOCI_I2S_BCLK 1  // D0
#endif
#ifndef LOCI_I2S_LRC
#define LOCI_I2S_LRC 2  // D1
#endif
#ifndef LOCI_I2S_DOUT
#define LOCI_I2S_DOUT 3  // D2
#endif
#ifndef LOCI_BTN_PTT
#define LOCI_BTN_PTT 4  // D3 to GND
#endif
#ifndef LOCI_LED
#define LOCI_LED 21  // on-board user LED, active-low
#endif
#ifndef LOCI_LED_INVERTED
#define LOCI_LED_INVERTED 1
#endif

// ------------------------------------------------------------------------------------
// AI-Thinker ESP32-CAM (classic ESP32, OV2640, flash LED on GPIO4, SD on 2/4/12/13/14/15)
// The camera occupies I2S0 on the classic ESP32, so the speaker must use I2S1.
// GPIO12 is a flash-voltage strapping pin: never give it a pull-up at boot.
// ------------------------------------------------------------------------------------
#elif defined(LOCI_BOARD_AI_THINKER)
#define LOCI_BOARD_NAME "AI-Thinker ESP32-CAM"
#define LOCI_CAM_PWDN 32
#define LOCI_CAM_RESET LOCI_PIN_NONE
#define LOCI_CAM_XCLK 0
#define LOCI_CAM_SIOD 26
#define LOCI_CAM_SIOC 27
#define LOCI_CAM_Y9 35
#define LOCI_CAM_Y8 34
#define LOCI_CAM_Y7 39
#define LOCI_CAM_Y6 36
#define LOCI_CAM_Y5 21
#define LOCI_CAM_Y4 19
#define LOCI_CAM_Y3 18
#define LOCI_CAM_Y2 5
#define LOCI_CAM_VSYNC 25
#define LOCI_CAM_HREF 23
#define LOCI_CAM_PCLK 22
#ifndef LOCI_HAS_MIC
#define LOCI_HAS_MIC 0
#endif
#ifndef LOCI_I2S_BCLK
#define LOCI_I2S_BCLK 14
#endif
#ifndef LOCI_I2S_LRC
#define LOCI_I2S_LRC 15
#endif
#ifndef LOCI_I2S_DOUT
#define LOCI_I2S_DOUT 2
#endif
#ifndef LOCI_BTN_PTT
#define LOCI_BTN_PTT 13
#endif
#ifndef LOCI_LED
#define LOCI_LED 33  // red LED next to the module, active-low
#endif
#ifndef LOCI_LED_INVERTED
#define LOCI_LED_INVERTED 1
#endif

// ------------------------------------------------------------------------------------
// Freenove ESP32-S3 WROOM CAM (N8R8): OV2640/OV3660, WS2812 on GPIO48, SD on 38/39/40
// ------------------------------------------------------------------------------------
#elif defined(LOCI_BOARD_FREENOVE_S3)
#define LOCI_BOARD_NAME "Freenove ESP32-S3 WROOM CAM"
#define LOCI_CAM_PWDN LOCI_PIN_NONE
#define LOCI_CAM_RESET LOCI_PIN_NONE
#define LOCI_CAM_XCLK 15
#define LOCI_CAM_SIOD 4
#define LOCI_CAM_SIOC 5
#define LOCI_CAM_Y9 16
#define LOCI_CAM_Y8 17
#define LOCI_CAM_Y7 18
#define LOCI_CAM_Y6 12
#define LOCI_CAM_Y5 10
#define LOCI_CAM_Y4 8
#define LOCI_CAM_Y3 9
#define LOCI_CAM_Y2 11
#define LOCI_CAM_VSYNC 6
#define LOCI_CAM_HREF 7
#define LOCI_CAM_PCLK 13
#ifndef LOCI_HAS_MIC
#define LOCI_HAS_MIC 0
#endif
#ifndef LOCI_I2S_BCLK
#define LOCI_I2S_BCLK 41
#endif
#ifndef LOCI_I2S_LRC
#define LOCI_I2S_LRC 42
#endif
#ifndef LOCI_I2S_DOUT
#define LOCI_I2S_DOUT 1
#endif
#ifndef LOCI_BTN_PTT
#define LOCI_BTN_PTT 21
#endif
#ifndef LOCI_LED
#define LOCI_LED 2
#endif
#ifndef LOCI_LED_INVERTED
#define LOCI_LED_INVERTED 0
#endif

// ------------------------------------------------------------------------------------
// Anything else: pass every pin with -D flags (see README "Other boards").
// ------------------------------------------------------------------------------------
#elif defined(LOCI_BOARD_CUSTOM)
#define LOCI_BOARD_NAME "custom board"
#if !defined(LOCI_CAM_XCLK) || !defined(LOCI_CAM_SIOD) || !defined(LOCI_CAM_SIOC) || \
    !defined(LOCI_CAM_Y9) || !defined(LOCI_CAM_Y2) || !defined(LOCI_CAM_VSYNC) ||    \
    !defined(LOCI_CAM_HREF) || !defined(LOCI_CAM_PCLK)
#error "LOCI_BOARD_CUSTOM needs every LOCI_CAM_* pin as a -D flag (see README)"
#endif
#ifndef LOCI_CAM_PWDN
#define LOCI_CAM_PWDN LOCI_PIN_NONE
#endif
#ifndef LOCI_CAM_RESET
#define LOCI_CAM_RESET LOCI_PIN_NONE
#endif
#ifndef LOCI_HAS_MIC
#define LOCI_HAS_MIC 0
#endif
#ifndef LOCI_BTN_PTT
#define LOCI_BTN_PTT LOCI_PIN_NONE
#endif
#ifndef LOCI_LED
#define LOCI_LED LOCI_PIN_NONE
#endif
#ifndef LOCI_LED_INVERTED
#define LOCI_LED_INVERTED 0
#endif

#else
#error "Select a board: -DLOCI_BOARD_XIAO_S3_SENSE | -DLOCI_BOARD_AI_THINKER | -DLOCI_BOARD_FREENOVE_S3 | -DLOCI_BOARD_CUSTOM"
#endif

// Common defaults (overridable) --------------------------------------------------------
#ifndef LOCI_FPS
#define LOCI_FPS 2
#endif
#ifndef LOCI_FRAME_SIZE
#define LOCI_FRAME_SIZE FRAMESIZE_VGA
#endif
#ifndef LOCI_JPEG_QUALITY
#define LOCI_JPEG_QUALITY 12
#endif
#ifndef LOCI_CAM_VFLIP
#define LOCI_CAM_VFLIP 0
#endif
#ifndef LOCI_CAM_HMIRROR
#define LOCI_CAM_HMIRROR 0
#endif
#ifndef LOCI_AUDIO_RATE
#define LOCI_AUDIO_RATE 16000
#endif
#ifndef LOCI_SPK_GAIN
#define LOCI_SPK_GAIN 1.0f
#endif
#ifndef LOCI_MIC_GAIN
#define LOCI_MIC_GAIN 6
#endif
#ifndef LOCI_PTT_MAX_S
#define LOCI_PTT_MAX_S 8
#endif
#ifndef LOCI_I2S_BCLK
#define LOCI_I2S_BCLK LOCI_PIN_NONE
#endif
#ifndef LOCI_I2S_LRC
#define LOCI_I2S_LRC LOCI_PIN_NONE
#endif
#ifndef LOCI_I2S_DOUT
#define LOCI_I2S_DOUT LOCI_PIN_NONE
#endif
#ifndef LOCI_MIC_CLK
#define LOCI_MIC_CLK LOCI_PIN_NONE
#endif
#ifndef LOCI_MIC_DATA
#define LOCI_MIC_DATA LOCI_PIN_NONE
#endif

// The classic ESP32's camera driver owns I2S0; the S3 camera uses LCD_CAM instead.
// Speaker always on I2S1, PDM microphone (S3 only) on I2S0.
#define LOCI_SPK_I2S_PORT I2S_NUM_1
#define LOCI_MIC_I2S_PORT I2S_NUM_0
