# On-board COCO detector (ESP-DL YOLO11n, 320x320 int8), shared by the board projects that run one
# (firmware/esp32-s3-client, firmware/esp32-p4-client). A board opts in by adding this directory
# to EXTRA_COMPONENT_DIRS, a "coco_det" flash partition and the COCO_DETECT settings in its
# sdkconfig.defaults; loci_app's cloud mode then sends what it sees as sightings (see sightings.h).
idf_component_register(
    SRCS "detect_local.cpp"
    INCLUDE_DIRS "include"
    REQUIRES loci_client
    PRIV_REQUIRES coco_detect esp_new_jpeg esp_timer
)
