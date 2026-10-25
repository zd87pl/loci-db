# Verified hardware facts for the handheld demo (2026-10-24)


## Camera
- Model: **OAK-D-LITE-AF** (autofocus), board DM9095 rev R3M2E5, platform RVC2 (Myriad X), device id 19443010D1523C7E00.
- **Has an IMU: BMI270** (this revision; older Lites have none). Treat IMU as optional.
- Sensors: CAM_A IMX214 4208x3120 color; CAM_B/CAM_C OV7251 640x480 mono. Stereo baseline 7.5 cm. CAM_A FOV 69 deg.
- CAM_A intrinsics (from device calibration): 640x400 fx=459.2 fy=458.7 cx=322.9 cy=208.7; 1280x800 fx=918.3 fy=917.5 cx=645.8 cy=417.4; 1920x1080 fx=1377.5 fy=1376.2 cx=968.7 cy=566.1.
  Distortion CAM_A (rational model, 8 coeffs k1 k2 p1 p2 k3 k4 k5 k6): [-0.7791, -9.3857, 0.0006, 0.0004, 31.35, -0.9325, -8.7559, 30.4295]. Always read these from `device.readCalibration()` at runtime; the numbers above are for tests/sim only.
- Extrinsics CAM_B->CAM_A translation (cm): [-3.76, 0.02, -0.14].

## Software versions that work
- DepthAI **3.11.0** (latest, 2026-10-10) on the Mac (Python 3.11 venv in the scratchpad `oakenv`); DepthAI 3.5.0 is on the Pi venv `~/loci-handheld/.venv` (upgrade to 3.11.0 for parity).
- v3 API only. `dai.node.StereoDepth.PresetMode` members: ACCURACY, DEFAULT, DENSITY, FACE, FAST_ACCURACY, FAST_DENSITY, HIGH_DETAIL, ROBOTICS (no HIGH_DENSITY).
- Model: `dai.NNModelDescription("yolov6-nano")` -> zoo archive `YOLOv6_Nano-R2_COCO_512x384.rvc2.tar.xz` (8.1 MB, 80 COCO classes, YOLO parser, input 512x384). Cache dir controlled by env `DEPTHAI_ZOO_CACHE_PATH` or `dai.getModelFromZoo(desc, useCached=True, cacheDirectory=...)`. First fetch needs internet (~2 s). Slugs `yolov8n` / `luxonis/yolov8-nano:coco-512x288` do NOT exist; `luxonis/yolov6-nano:r2-coco-512x288` does.

## The pipeline that WORKS (verified live from the Mac, 10 fps, 12 s, 122 frames)
```python
import depthai as dai
PM = dai.node.StereoDepth.PresetMode
with dai.Pipeline() as pipeline:                      # connects to the first device
    dev = pipeline.getDefaultDevice()                  # dev.readCalibration(), dev.getConnectedIMU(), dev.getUsbSpeed()
    cam   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=10)
    left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=10)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=10)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput((640, 400)), right=right.requestOutput((640, 400)),
        presetMode=PM.FAST_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setPostProcessingHardwareResources(2, 2)    # REQUIRED on OAK-D Lite: frees shaves so the 8-shave blob fits
    nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(cam, stereo, dai.NNModelDescription("yolov6-nano"))
    nn.setConfidenceThreshold(0.5); nn.setBoundingBoxScaleFactor(0.5)
    nn.setDepthLowerThreshold(100); nn.setDepthUpperThreshold(8000)   # mm
    labels = nn.getClasses()                            # list[str], 80 COCO names
    q = nn.out.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()
    while pipeline.isRunning():
        msg = q.tryGet()                                # dai.SpatialImgDetections or None
        if msg is None: continue
        for d in msg.detections:                        # d.label (int), d.confidence, d.xmin/ymin/xmax/ymax normalized [0,1]
            sc = d.spatialCoordinates                   # mm, camera frame; X right, Z forward; Y sign: see below
```
- FAILS with `presetMode=HIGH_DETAIL` or `DEFAULT` (even with post-processing shaves 3): "Blob compiled for 8 shaves, but only 7 are available" (NN gets shaves 0-6, stereo 9-12, ImageManip 15, SpatialLocationCalculator 14).
- Pipeline start ~0.6 s, first detections ~0.9 s after start. Steady 10.1 fps over USB 2 (HIGH speed) — USB 2 is enough.
- Fallback that also fits: plain `dai.node.DetectionNetwork.build(cam, dai.NNModelDescription("yolov6-nano"))` + `stereo.depth.createOutputQueue()` and host-side median depth in the box + back-projection with the 640x400 intrinsics (code in the spike script). Prefer the spatial network.

## Measured behaviour / gotchas
- A detection whose box has no valid depth returns `spatialCoordinates` x=y=z=0 exactly -> treat z <= 0 as "no depth" (do not store a 0 m position).
- Depth jitter: a refrigerator at ~6 m read 5.5-7.4 m frame to frame; a person at 3.3 m was stable to ±0.05 m. Clamp usable range to ~0.3-5 m and let the consensus buffer average.
- **Y-axis sign is unverified**: boxes at the bottom of the image (cy ~0.9) came back with Y = +1.4 m while the camera was likely tilted; the implementation must make the camera->OpenCV axis mapping a single constant and the hardware checklist must include "hold a cup above the lens centre and confirm the sign of Y".
- Opening the device right after a failed pipeline raises X_LINK_INSUFFICIENT_PERMISSIONS / X_LINK_ERROR for ~10 s while the watchdog resets it; poll `dai.Device.getAllAvailableDevices()` until state is X_LINK_UNBOOTED before reconnecting. Build reconnect logic this way.
- macOS: after firmware boot the device re-enumerates as 03e7:f63b "Luxonis Device"; macOS 26 keeps it unregistered unless Privacy & Security > "Allow accessories to connect" is Always (now set on the dev Mac).
- A stale "Device ... has crashed. Crash dump logs ..." error line can be printed on open after earlier failures; harmless.

## Pi state
- Pi 5 16 GB, Raspberry Pi OS trixie, Python 3.13.5, ssh ziggy@loci-pi.local (key auth works), no passwordless sudo.
- `~/loci-handheld/.venv` has depthai 3.5.0, numpy 2.5.4, opencv-python-headless 5.0.0 (aruco AprilTag 36h11 dict works), fastapi 0.162, uvicorn 0.54, pydantic 2.14, httpx, pytest 9.1, model2vec 0.10.0 (tokenizers 0.23.3, safetensors 0.8.0).
- Models staged: `~/loci-handheld/models/zoo/<hash>/YOLOv6_Nano-R2_COCO_512x384.rvc2.tar.xz` (cacheDirectory=~/loci-handheld/models/zoo) and `~/loci-handheld/models/embeddings/potion-base-8M` (model2vec StaticModel, dim 256, local load 0.13 s, 21 labels in 1.3 ms; cup/mug 0.47, keys/key chain 0.51, couch/sofa 0.68, cup/dog 0.14).
- Camera is NOT yet visible on the Pi: the user's USB-C to USB-A cable produced no USB attach event at all; the camera works on the Mac over a USB-C to USB-C (USB 2) cable. Waiting on a data-capable cable/adapter. The Pi's PSU negotiates 3 A (USB capped at 600 mA, `usb_max_current_enable=0`); config.txt has `dtoverlay=dwc2,dr_mode=host`.
- Spike script with the working pipeline and host-depth fallback: /private/tmp/claude-501/-Users-0x7d0-git-loci-db/8139ebf0-1ba6-436c-8885-212cc37c9445/scratchpad/oak_step1_live.py

## Axis convention conclusion (added after cross-checking research_depthai.md)
- Two independent indications agree that DepthAI v3 `spatialCoordinates` on RVC2 are **+X right, +Y DOWN, +Z forward (OpenCV/RDF)**: (1) the v3 host code (`SpatialUtils.cpp`, see research_depthai.md section on conventions) computes y from pixel row minus cy, i.e. image-down positive; (2) in the live test every box near the bottom of the frame (cy 0.68-0.93) came back with Y = +1.4 to +2.4 m at Z = 3-7 m, which is below the optical axis for an upright camera. The old v2 docs say "Y up" — ignore for v3.
- Still keep the axis mapping a single constant (`OAK_Y_AXIS = "down"` default) and keep the cup-above-the-lens check in the hardware checklist; the implementation must not hard-code the sign in more than one place.
- Consequence: a camera-frame point from DepthAI can be fed to OpenCV solvePnP-based tag poses with NO axis flip.
