cmake_minimum_required(VERSION 3.16)
# host and cloud link, microSD memory, offline commands, the main loop and the on-board detector
# are shared with the S3 client
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../common/loci_client" "${CMAKE_CURRENT_LIST_DIR}/../common/loci_detect")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(loci_p4_client)

# As in the S3 client: esp-sr registers ESP-DL compile lists for its speech models only, which would
# strip the detector's ops (Swish, int8 Conv kernels); without lists ESP-DL builds the full set.
idf_build_set_property(ESPDL_COMPILE_REQS "")
