cmake_minimum_required(VERSION 3.16)
# host and cloud link, microSD memory, offline commands, the main loop and the on-board detector
# are shared with the P4 client
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../common/loci_client" "${CMAKE_CURRENT_LIST_DIR}/../common/loci_detect")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(loci_s3_client)

# ESP-DL compiles only the ops and Conv kernels named in registered compile_req files, and all of
# them when none is registered. esp-sr registers the list for its speech models; coco_detect
# registers none, so the detector's ops (Swish, int8 Conv kernels) would be stripped and its model
# would fail to load ("Do not support Swish"). Dropping the lists here, before ESP-DL's deferred
# finalize step runs, builds the full operator set both libraries work with.
idf_build_set_property(ESPDL_COMPILE_REQS "")
