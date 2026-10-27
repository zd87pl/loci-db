cmake_minimum_required(VERSION 3.16)
# host link, microSD memory, offline commands and the main loop are shared with the S3 client
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../common/loci_client")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(loci_p4_client)
