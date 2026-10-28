// Board-independent settings of the LOCI thin client. A board project can override any of them for
// the whole build from its top-level CMakeLists.txt, e.g.
//   idf_build_set_property(COMPILE_DEFINITIONS "LOCI_FPS=1" APPEND)
// Pins and codec/camera tuning live in the board header of each project (board_*.h).
#pragma once

// Audio format the host speaks: 16 kHz mono PCM16
#ifndef LOCI_AUDIO_RATE
#define LOCI_AUDIO_RATE 16000
#endif
#ifndef LOCI_FPS
#define LOCI_FPS 2
#endif
#ifndef LOCI_PTT_MAX_S
#define LOCI_PTT_MAX_S 8
#endif
#ifndef LOCI_SD_FORMAT_IF_NEEDED
#define LOCI_SD_FORMAT_IF_NEEDED 0  // 1 = format a non-FAT card as FAT32 at boot (destroys its content)
#endif

// microSD usage (see local_memory.h)
#ifndef LOCI_SD_KEEP_EVERY
#define LOCI_SD_KEEP_EVERY 10  // save every Nth live frame + host reply as a dataset (0 = off)
#endif
#ifndef LOCI_SD_MIN_FREE_MB
#define LOCI_SD_MIN_FREE_MB 512  // stop writing when the card has less than this free
#endif
#ifndef LOCI_BACKLOG_MAX_FILES
#define LOCI_BACKLOG_MAX_FILES 3000  // ~25 minutes at 2 fps
#endif
#ifndef LOCI_OFFLINE_FPS
#define LOCI_OFFLINE_FPS 1  // capture rate to the card while the host is unreachable
#endif
#ifndef OFFLINE_PROMPT_URL
// Neutral wording: in cloud mode there is no host to name
#define OFFLINE_PROMPT_URL "I%27m%20offline%20right%20now.%20Hold%20the%20button%20again%20when%20we%20are%20back%20online."
#endif
