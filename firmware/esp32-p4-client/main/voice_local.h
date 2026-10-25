#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Offline speech commands (ESP-SR MultiNet7 English on the P4), used when the host cannot be
// reached: the user holds the button and says "where is my cup" / "what do you see"; the clip is run
// through the AFE (noise suppression) and MultiNet; the matched command names an object that the
// cached memory can answer from the card. No wake word: the button is the trigger.
//
// Models live in the "model" flash partition (CONFIG_MODEL_IN_FLASH); `idf.py flash` writes them.

#define VOICE_CMD_NONE (-1)
#define VOICE_CMD_WHAT_DO_YOU_SEE 0
#define VOICE_CMD_FIRST_OBJECT 1

// Load models and create the engines. False (and log) when the model partition is missing.
bool voice_local_init(void);
bool voice_local_ready(void);

// Rebuild the command list: "what do you see" plus "where is my <label>" / "where are my <label>"
// for the given labels (lower case). A fixed set of common household objects is always included.
void voice_local_set_labels(const char labels[][32], int count);

// Run a recorded 16 kHz mono PCM16 clip through AFE + MultiNet. Returns VOICE_CMD_NONE or a command id;
// for object commands `label_out` receives the object label. Takes about real-time/4 on the P4.
int voice_local_recognize(const int16_t *pcm, size_t samples, char *label_out, size_t label_cap);
