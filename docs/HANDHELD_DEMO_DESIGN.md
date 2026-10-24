<!-- Synthesized by the design workflow on 2026-10-24; scratch-path references below point at the session scratchpad and are informational. Companion plan: HANDHELD_DEMO_PLAN.md -->

# LOCI handheld demo (Pi 5 + OAK-D Lite) — FINAL DESIGN

Synthesised 2026-10-24 from Design 2 (demo-impact, judges' winner) with every judge-endorsed graft from Designs 0 and 1 and every judge-found defect fixed against the live source tree. Repository: `/Users/0x7d0/git/loci-db`, branch `chore/m0-quick-wins`.

## 0. Preconditions the implementer must know

1. **The working tree is mid-merge with `origin/main` (MERGE_HEAD `13330e2`).** `loci/local_client.py`, `loci/temporal/decay.py`, `loci/client.py`, `loci/async_client.py`, `loci/cloud_transport.py` and 11 other files still carry conflict markers, so `import loci` is a `SyntaxError` and none of the four demo tests can run today. The user must finish that merge first. This design targets the **post-merge `LocalLociClient`** (the `origin/main` version, read with `git show 13330e2:loci/local_client.py`):
   - one raw collection `loci_data` (`client._data_collection`) plus one `loci_summary`; epochs are purely logical; `_known_collections`, `_list_active_epochs`, `_ensure_collection(name)` and per-epoch `collection_name()` are gone (the latter is marked LEGACY);
   - `insert()` / `insert_batch()` validate vector dimension (`_validate_vector`) and assign `str(uuid4())` ids; `insert_batch` sorts by `(scene_id, timestamp_ms)` and links causally;
   - `MemoryStore.delete_points(collection, ids)` exists; `set_payload` never re-encodes Hilbert ids;
   - decay default is a one-hour half-life, exponent clamped, ties broken by raw similarity (the "oldest-first after 2 h" hazard is fixed upstream);
   - `client.store` is a public property; `query(vector, spatial_bounds, time_window_ms, limit, *, _extra_payload_filter, min_confidence)` is unchanged in shape; a list value in `_extra_payload_filter` becomes a `MatchAny`.
   - The staged `demo_spatial/app/spatial_memory.py` already merges by **delete + re-insert through `client.insert`** (fresh id), and the staged `tests/test_spatial_deduplication.py` requires exactly that (`test_merge_recomputes_spatial_index_across_epochs`: 15 cross-epoch merges collapse to ONE record; a raw `mem._client.query(vector=_label_embedding("keys"), ...)` must return a hit with `scene_id == "keys"`).
2. `MemoryStore._matches` filters **top-level payload keys only**: `metadata` is round-tripped but not filterable, so the filterable key stays `scene_id`.
3. `MemoryStore.search` truncates with `np.argpartition` **before** decay re-ranking; with pure label vectors every point of a label ties, so exact-label recency needs a timestamp window from a side index (section 7).
4. Implementation may only **add new files** or edit `demo_spatial/app/{scene_ingestion,spatial_memory,main,voice_pipeline}.py` and `tests/`. `loci/`, `demo_spatial/requirements.txt`, `index.html`, `scanner.html` and every other modified file are off limits; never commit.
5. Test runner on the Mac: `/opt/homebrew/bin/pytest` (Python 3.14; has fastapi + httpx, so `TestClient` works; **no** cv2, pyapriltags, tokenizers, safetensors, depthai). CI runs `pytest tests/ -v` on 3.11/3.12 with `pip install -e ".[dev]"` and `ruff check` with `S` rules active for `demo_spatial/`. Every hardware-adjacent test therefore uses `pytest.importorskip(...)`, and the simulator uses `numpy.random.default_rng` (ruff S311 forbids `random`). For the skipped tests create a scratch venv: `python3.11 -m venv /private/tmp/claude-501/-Users-0x7d0-git-loci-db/8139ebf0-1ba6-436c-8885-212cc37c9445/scratchpad/devenv && pip install -e . -r deploy/handheld/requirements-dev.txt`.

## 1. Decisions (every open question resolved)

| # | Question | Decision and assumption |
|---|---|---|
| D1 | DepthAI v2 or v3 | **v3, `depthai==3.11.0`** (cp313 aarch64 wheel installs on the Pi in 17 s; this exact OAK-D-LITE-AF streamed live under v3 from the Mac). v2 is documented only as a fallback behind the same `DetectionSource` protocol and the `LOCI_OAK_AXES=ruf` flag. |
| D2 | Stereo preset / resources | Exactly the live-verified pipeline: `FAST_DENSITY`, `setDepthAlign(CAM_A)`, `setPostProcessingHardwareResources(2, 2)`, **no** LR-check and **no** extended disparity by default (`DEFAULT`/`HIGH_DETAIL` fail: "blob compiled for 8 shaves, only 7 available"). LRC/extended are opt-in env flags tried in step 1. |
| D3 | Detection model | zoo `yolov6-nano` → `YOLOv6_Nano-R2_COCO_512x384.rvc2.tar.xz` (8.1 MB, COCO-80), already cached on the Pi under `~/loci-handheld/models/zoo`; loaded offline with `DEPTHAI_ZOO_INTERNET_CHECK=0` or `dai.NNArchive(path)`. Vocabulary is closed; "mug" → `cup` via alias; "keys" is undetectable and the assistant says so. |
| D4 | Y sign of spatialCoordinates | Assume **RDF (+x right, +y down, +z forward), origin RGB camera**; one constant `LOCI_OAK_AXES=rdf` (`ruf` flips y) applied in exactly one function; `oak_console.py --check-y-sign` verifies on first camera contact. |
| D5 | Tag stream | Second RGB output `1280x800 NV12 @ 5 fps`, **raw (no on-device undistortion)**, distortion coefficients from `frame.getTransformation().getDistortionCoefficients()` passed to solvePnP (saves a device-side warp on a shave-starved RVC2; 61 Mbit/s). Fallback ladder via `LOCI_TAG_STREAM`: `1280x800@5` → `640x400@10` (print 24 cm tags) → `off` (camera-frame only). `LOCI_TAG_UNDISTORT=1` switches to `enableUndistortion=True` + `dist=None`. |
| D6 | Tag detector | `pyapriltags==3.4.3.1` (wheel, 9 ms/frame at 1280x800 qd=2) with corner remap `[1,0,3,2]`; `cv2.aruco` fallback behind the same `TagDetector` protocol; never bare `SOLVEPNP_IPPE_SQUARE` (robust solver from the validated `apriltag_pose.py`). |
| D7 | Tags per room | One anchor tag per room is the minimum; co-visible extra tags self-register (`RoomAnchorMap`) and trigger the joint multi-tag solve. Anchor = origin, floor at y=0. |
| D8 | Pose hold policy | `PoseHold(hold_s=10)`: anchored while age ≤ 10 s (confidence 1→0 linear); held pose still used until `stale_s=60` with `anchored=False`, `anchor_confidence=0`; after that observations are stored in the **camera frame** under the `unanchored/<label>` pseudo-room. Anchored and unanchored observations never merge. No sticky-room timeout of 900 s. |
| D9 | scene_id layout | `scene_id == label` for legacy image-frame observations (staged test asserts `h.scene_id == "keys"`); `"<room>/<label>"` for room-frame observations; `"unanchored/<label>"` for camera-frame observations. `where_is(label)` filters `{"scene_id": [label, *room/label, unanchored/label]}` (one MatchAny). The frame is also written to `metadata.frame`. |
| D10 | Merge rule | Keyed lookup in a per-scene side index (no vector query); match = same `scene_id`, same `anchored`, within `dedup_window_ms`, metric distance ≤ `dedup_radius_m` (room/camera frames) or proxy-IoU > `dedup_iou_threshold` (image frame). Merge = `store.delete_points(client._data_collection, [old])` + `client.insert(merged)` returning the **new** id; **no same-epoch rule**. All private-client touches live in `loci_bridge.py`. |
| D11 | Decay | Keep the client default (one-hour half-life, clamped). Recency is enforced by the side-index time window and an explicit `timestamp_ms` sort; no `decay_lambda` override. |
| D12 | Embedder | potion-base-8M (256-d) through the hand-rolled `StaticLabelEmbedder` (tokenizers + safetensors only); `HashLabelEmbedder` reproduces today's 128-d maths byte-for-byte. **`LOCI_LABEL_EMBEDDER` defaults to `hash`** (deterministic tests, `_label_embedding` stays a valid 128-d query vector); the Pi env sets `static`; `auto` exists. `spatial_memory._label_embedding` and `EMBED_DIM` remain exported. Vectors are pure label embeddings (position blend dropped). |
| D13 | Label resolution | Only in the voice/API layer (`LabelResolver`), never inside `where_is`. Vocabulary = detector classes ∪ `memory.tracked_labels`; order exact > alias > embedding ≥ 0.45 (static embedder only) > deny-list > none + suggestions. Unknown nouns get "I can't recognise 'keys' with this camera…", never a wrong answer. Whole-utterance embedding is not used; only the parsed noun. |
| D14 | Persistence | JSON snapshot of vector-less `ObservationRecord`s + fsync'd JSONL journal with `upsert` **and `delete`** ops, written by a `PersistenceWriter` thread (never on the camera thread); restore through public `LocalLociClient.insert_batch`, followed **synchronously** by a fresh snapshot and journal truncation before any new write (so journal ids always match live ids). `.bak` fallback, corrupt-file quarantine, `LOCI_MAX_RECORDS` prune. No Qdrant (server process, no embedded mode, no benefit). |
| D15 | Restore with a different embedder | Non-event: vectors are never persisted; they are recomputed from `metadata.label` with the current embedder on load. The embedder name is recorded for diagnostics only. |
| D16 | Room → LOCI box | Per-room `extents_m` (default x∈[−3,3], y∈[0,2.5], z∈[0,6] m); `u = clip((p−lo)/(hi−lo), 0, 1)`; `metadata.clipped=true` when clipping happened; raw metres always kept in `metadata.pos_m`. Camera box x∈[−3,3], y∈[−1.5,1.5], z∈[0,6]. Hilbert r4 cell ≈ 0.375 × 0.16 × 0.375 m — accepted overfetch. |
| D17 | Thread model | One `handheld-runtime` thread pulls `source.events()`, detects tags, localises, feeds `SceneIngestion.process_detections` synchronously; one `persistence-writer` thread; the asyncio loop runs a 5 Hz `_live_push_loop` that polls runtime state and broadcasts — no cross-thread asyncio calls. `SpatialMemory` methods take an `RLock`. |
| D18 | Reconnect | Supervisor inside `OakDetectionSource.events()`: on any exception close device, state `waiting_for_camera`, poll `dai.Device.getAllAvailableDevices()` for `X_LINK_UNBOOTED`, backoff 1→5 s, forever; 5 s stall watchdog; `reconnects`/`usb_speed` in `SourceStatus`. systemd `Restart=on-failure` is only the outer loop. |
| D19 | Timestamps | `HostClock` maps `msg.getTimestamp()` (steady) to Unix ms with a min-latency offset; falls back to arrival time if |converted − arrival| > 1 s. |
| D20 | Consensus in metric space | Buffer keyed by `(label, frame, room)`; metric distance ≤ `consensus_radius_m` (0.3) when both entries have metres, else IoU proxy; matched entries are **consumed** on fire; `consensus_refire_ms` hold-off (ctor default 0 = legacy; handheld env 2000). |
| D21 | Configuration | Env vars + one `rooms.json`; no YAML, no pyyaml. |
| D22 | API surface | Existing endpoints and WS shapes unchanged (`/api/voice/text-query` still returns TTS audio). New text-first `POST /api/query` (TTS only when `tts: true`), `/phone`, `/api/rooms`, `/api/anchor`, `/api/objects?room=`, `/api/objects/changes`, `/api/handheld/status`, `/api/handheld/snapshot`, `/api/memory/clear`; `/api/voice/tts` sniffs `RIFF` → `audio/wav`. |
| D23 | Perspective wording | Room frame is right-handed, y up, z into the room, **+x to the viewer's right when facing the tag**, which is the walker's **left** when entering with their back to the tag. Answers use the walk-in perspective and say so ("as you walk in from the door tag"); when the camera is currently anchored in the same room the answer is egocentric ("about 3 m ahead of you and a little to your left"). Encoded once in `phrasing.py` with a unit test. |
| D24 | `gone` in changes | Reported only when the camera was anchored in that room for ≥ 5 s inside the window after the object's last sighting (presence log fed by the runtime at 1 Hz, in-memory only). Otherwise unknown objects are simply not mentioned. |
| D25 | Landmarks | Optional per-room `landmarks` in `rooms.json` (nearest within 1.2 m wins, "on the kitchen table"); the demo works without them. |
| D26 | Pi layout | `~/loci-handheld/{src,.venv,models,data,env}`; the existing `.venv` (depthai 3.5.0) is upgraded in place to 3.11.0; `~/loci-db` is never touched by rsync; the Piper model is **copied** from `~/loci-db/deploy/oak-rpi5/models/` into `~/loci-handheld/models/piper/`. Only one mandatory sudo step (`usb_max_current_enable=1`). |
| D27 | Voice | Typed queries are the committed path; voice reuses `voice_pipeline.py` (`WHISPER_LOCAL=1`, Piper) with the piper ≥1.3 `synthesize_wav` API handled; phone mic needs `LOCI_TLS=1`; the page also offers browser `speechSynthesis`. |
| D28 | Demo story object | A **cup** (COCO class); the user may say "mug" (alias). "keys" demonstrates the honest refusal path. |
| D29 | IMU | Not used (this unit has a BMI270 but the design does not depend on it). |
| D30 | Hotspot vs LAN | LAN + `http://loci-pi.local:8001/phone` is the default; the NetworkManager hotspot is a documented manual alternative (phone opens `http://10.42.0.1:8001/phone`). Android may not resolve `.local`; the IP and a QR (`/api/scanner-qr`) are printed at startup. |

## 2. Architecture

```
OAK-D Lite ──USB──> Pi 5: handheld-runtime thread
   on camera: YOLOv6-nano (COCO-80) + stereo depth -> SpatialImgDetections (mm, RDF)
   second RGB output 1280x800 NV12 @5 fps -> luma -> pyapriltags -> TagFrame
   OakDetectionSource.events() yields DetectionFrame | TagFrame   (reconnect supervisor inside)
        |                      |
        |                      v  RoomLocalizer.update(TagFrame) -> PoseHold -> CameraPose
        v
   DetectionFrame -> localizer.current(ts) -> Detection(pos_room_m, room, frame, anchored)
        -> SceneIngestion.process_detections() (metric consensus, consume, hold-off)
        -> SpatialMemory.observe()  [RLock]  -> LocalLociClient.insert / keyed merge
        -> PersistenceWriter queue  (journal 1 Hz fsync, snapshot every 20 s)
FastAPI (uvicorn, 1 worker, no --reload): /api/query, /phone, WS 5 Hz push loop polling runtime state
Phone (LAN or hotspot) -> typed question -> LabelResolver -> where_is / changes -> phrasing -> aria-live answer
```

Simulated twin: `SimDetectionSource` emits the same `DetectionFrame`/`TagFrame` objects from a scripted two-room scenario (`mug_two_rooms`), so the whole chain (tags → PnP → room metres → LOCI → answer) runs on the Mac/CI without hardware; `oracle_pose=True` bypasses PnP where cv2 is absent.

## 3. Module and file plan

New files (all under the repo root):

| File | Owner task | Purpose |
|---|---|---|
| `demo_spatial/app/detection_source.py` | T2 | Hardware-neutral contract: `SpatialDetection`, `TagDetection`, `TagFrame`, `CameraPose`, `DetectionFrame`, `SourceStatus`, `DetectionSource` protocol, `AXIS_FIX`/`depthai_xyz_to_cam`, `HostClock`, `COCO80`, `Scenario`/`SimDetectionSource`, `JsonlRecorder`/`JsonlReplaySource`, `make_detection_source()` |
| `demo_spatial/app/oak_source.py` | T2 | Only module importing `depthai` (lazily): `OakConfig`, `build_pipeline`, `spatial_to_detections`, `OakDetectionSource` with supervisor + watchdog, `wait_for_unbooted` |
| `demo_spatial/app/geometry.py` | T3 | numpy SE(3) helpers and the validated robust PnP (`solve_tag_pose`, `solve_room_pose_multi`, cv2 lazily) |
| `demo_spatial/app/room_anchor.py` | T3 | `TagDetector` protocol + `PyAprilTagsDetector`/`ArucoTagDetector`, `TagFilter`, `RoomAnchorMap`, `PoseHold`, `RoomSpec`/`RoomMap`, `room_to_unit`/`unit_to_room`, `RoomLocalizer`, `tags.json` persistence |
| `demo_spatial/app/label_embedding.py` | T6 | `LabelEmbedder` protocol, `HashLabelEmbedder`, `StaticLabelEmbedder`, `make_embedder`, `static_model_available`, `normalize_label`, `hash_label_embedding` |
| `demo_spatial/app/label_resolver.py` | T6 | `Resolution`, `LabelResolver`, `DEFAULT_ALIASES`, `DEFAULT_DENYLIST` |
| `demo_spatial/app/loci_bridge.py` | T4 | The only place that touches private `LocalLociClient` members (`_data_collection`) |
| `demo_spatial/app/persistence.py` | T4 | `ObservationRecord`, `MemoryPersistence` (snapshot + journal), `PersistenceWriter` thread, `RestoreReport` |
| `demo_spatial/app/phrasing.py` | T10 | Position wording (walk-in perspective, egocentric, heights, landmarks), answer templates, suffixes |
| `demo_spatial/app/handheld.py` | T7 | `HandheldConfig.from_env()`, `HandheldRuntime` thread, `build_handheld()` wiring |
| `demo_spatial/static/phone.html` | T7 | Mobile-first offline page |
| `deploy/handheld/config/rooms.example.json` | T3 | Example room configuration |
| `deploy/handheld/scripts/oak_console.py` | T1 | Build-order step 1 console (`--sim`, `--check-y-sign`, `--record`) |
| `deploy/handheld/{README.md, requirements-pi.txt, requirements-dev.txt, env.example, install_pi.sh, run_pi.sh, deploy_from_mac.sh, smoke_test.sh, loci-handheld.service, loci-handheld.user.service, scripts/fetch_models.py, scripts/make_tag_sheet.py, models/.gitignore, models/README.md}` | T8 | Pi deployment |
| `deploy/handheld/HARDWARE_VERIFICATION.md` | T9 | Step 1–5 hardware plan with exact commands |

Edited files (demo_spatial only): `scene_ingestion.py` (T5), `spatial_memory.py` (T4), `voice_pipeline.py` (T10), `main.py` (T7). `vlm_client.py` is untouched (stays optional and off without keys).

Tests (new): `tests/test_detection_source.py`, `tests/test_oak_source_offline.py` (T2); `tests/test_oak_console.py` (T1); `tests/test_geometry.py`, `tests/test_tag_pose.py`, `tests/test_room_anchor.py` (T3); `tests/test_spatial_memory_rooms.py`, `tests/test_persistence.py` (T4); `tests/test_scene_ingestion_detections.py` (T5); `tests/test_label_embedding.py`, `tests/test_label_resolver.py` (T6); `tests/test_handheld_api.py`, `tests/test_handheld_runtime.py` (T7); `tests/test_make_tag_sheet.py` (T8); `tests/test_phrasing.py`, `tests/test_voice_room_answers.py` (T10). The four existing demo tests must keep passing unchanged.

## 4. Contracts

Everything below is normative. Tasks code against these names before the other side exists. All modules live in `demo_spatial/app/` and are imported by tests as `app.<module>` after `sys.path.insert(0, "<repo>/demo_spatial")`. Python 3.11+ syntax, full type hints, `from __future__ import annotations`, ruff-clean under the repo `pyproject.toml` (line length 100, rules E,F,I,W,UP,B,SIM,S).

### 4.1 Coordinate conventions (single source of truth)

| Frame | Axes | Units | Origin | Where it is produced |
|---|---|---|---|---|
| **DepthAI v3 spatialCoordinates** | +x right, +y DOWN, +z forward (RDF) | mm | RGB camera CAM_A optical centre (depth aligned to CAM_A) | OAK firmware; `d.spatialCoordinates.{x,y,z}`; `x=y=z=0` exactly means "no depth" → drop |
| **Camera frame (OpenCV)** | +x right, +y down, +z forward | metres | same | `detection_source.depthai_xyz_to_cam(xyz_mm, axes)`; `axes="rdf"` → identity, `"ruf"` → `diag(1,-1,1)` (DepthAI v2 or a firmware surprise). This is the ONLY place the sign lives. `SpatialDetection.xyz_cam_m` is always in this frame. |
| **Image frame (legacy)** | cx right, cy down | normalised [0,1] | top-left of the frame | browser/YOLO mode; `Detection.cx, cy`; LOCI `x=cx, y=cy, z=min(depth_m/5, 1)` |
| **Tag frame** | +x right, +y up, +z out of the printed face toward the viewer | metres | tag centre | corners TL,TR,BR,BL = `SOLVEPNP_IPPE_SQUARE` object order `(-s,+s),(+s,+s),(+s,-s),(-s,-s)`; cv2.aruco returns this order; pyapriltags corners are remapped `corners[[1,0,3,2]]`. A frontal tag has `R_cam_tag = diag(1,-1,-1)`. |
| **Room frame** | right-handed, +y up, +z from the anchor wall into the room, +x to the viewer's RIGHT when FACING the anchor tag (= the walker's LEFT when entering with their back to the tag) | metres | on the floor directly below the anchor tag centre | `T_room_tag_anchor = make_T(I, (0, anchor_height_m, 0))`; `T_room_cam = T_room_tag @ invert_T(T_cam_tag)`; `p_room = apply_T(T_room_cam, p_cam)` |
| **Camera box** (unanchored observations) | camera frame clipped | metres → [0,1] | — | `CAMERA_BOX = {"x": (-3.0, 3.0), "y": (-1.5, 1.5), "z": (0.0, 6.0)}` |
| **LOCI box** | x,y,z ∈ [0,1] | — | — | `room_to_unit(p_m, extents) -> (u: tuple[float,float,float], clipped: bool)`; `u_i = clip((p_i - lo_i)/(hi_i - lo_i), 0, 1)`; `unit_to_room(u, extents)` inverts. Default room extents `{"x": (-3.0, 3.0), "y": (0.0, 2.5), "z": (0.0, 6.0)}`. |

Transform naming: `T_a_b` is a 4×4 homogeneous matrix such that `p_a = T_a_b @ p_b`. Timestamps are Unix **milliseconds** (`int`) everywhere; `now_s` arguments are `time.monotonic()` seconds.

### 4.2 `detection_source.py` (T2) — no `loci`, no `cv2`, no `depthai` at import time

```python
COCO80: list[str]                       # the 80 class names in the model's order ("person" ... "toothbrush")
AXIS_FIX: dict[str, np.ndarray]         # {"rdf": eye(3), "ruf": diag(1,-1,1)}
def depthai_xyz_to_cam(xyz_mm: Sequence[float], axes: str = "rdf") -> tuple[float, float, float] | None
    # returns None when z_mm <= 0 (no depth); else metres in the OpenCV camera frame

@dataclass(frozen=True)
class SpatialDetection:
    label: str                 # COCO name, lower case
    label_id: int
    confidence: float          # [0,1]
    xmin: float; ymin: float; xmax: float; ymax: float      # normalised [0,1] box
    xyz_cam_m: tuple[float, float, float] | None            # camera frame, metres; None = no depth
    @property cx, cy, width, height -> float
    @property depth_m -> float | None                       # xyz_cam_m[2]
    def to_dict(self) -> dict

@dataclass
class TagDetection:
    tag_id: int
    corners_px: np.ndarray     # shape (4,2) float64, TL,TR,BR,BL in pixels of the TagFrame image
    decision_margin: float     # pyapriltags margin; aruco fallback reports 100.0
    hamming: int               # 0 required downstream

@dataclass
class TagFrame:
    timestamp_ms: int
    width: int; height: int
    K: np.ndarray              # (3,3) intrinsics for THIS image size
    dist: np.ndarray | None    # distortion coefficients (len 4/5/8/12/14) or None if undistorted
    gray: np.ndarray | None    # uint8 (H,W) luma; None when `tags` are already detected (sim)
    tags: list[TagDetection] | None   # pre-detected corners (sim / replay); None means "detect on gray"

@dataclass
class CameraPose:
    room: str | None           # room name or None
    frame: str                 # "room" (T usable) | "camera" (no usable pose)
    T_room_cam: np.ndarray | None     # (4,4) or None when frame == "camera"
    anchored: bool             # True while a tag was accepted <= hold_s ago
    anchor_confidence: float   # 1 - age/hold_s clipped to [0,1]; 0 when not anchored
    tag_ids: tuple[int, ...]   # tags used for the last accepted pose
    age_s: float               # seconds since the last accepted tag pose (inf if never)
    def to_dict(self) -> dict  # T flattened to "camera_xyz_m": [x,y,z] and "heading_deg": float (yaw of camera +z in the room x-z plane, 0 = looking toward -z i.e. at the anchor wall, +90 = looking toward +x)

@dataclass
class DetectionFrame:
    timestamp_ms: int          # host Unix ms (HostClock-converted when a device timestamp exists)
    seq: int
    detections: list[SpatialDetection]
    source: str                # "oak" | "sim" | "replay"
    device_ts_ms: int | None = None
    oracle_pose: CameraPose | None = None   # sim only: bypasses the localizer when set

SourceEvent = DetectionFrame | TagFrame

@dataclass
class SourceStatus:
    state: str                 # "starting" | "running" | "waiting_for_camera" | "stopped" | "error" | "done"
    fps: float
    frames: int
    reconnects: int
    last_frame_age_s: float | None
    usb_speed: str | None
    device: str | None         # device name / mxid
    error: str | None
    def to_dict(self) -> dict

class DetectionSource(Protocol):
    name: str
    classes: list[str]
    def start(self) -> None: ...
    def stop(self) -> None: ...                       # idempotent; events() returns soon after
    def events(self) -> Iterator[SourceEvent]: ...    # blocking generator; ends when stopped (or scenario done)
    def status(self) -> SourceStatus: ...

class HostClock:
    def __init__(self, max_skew_ms: int = 1000, warmup: int = 20) -> None
    def to_unix_ms(self, device_ms: int, arrival_ms: int) -> int   # offset = min(arrival - device) over warmup; falls back to arrival_ms when |converted - arrival| > max_skew_ms (counted in .fallbacks)
    fallbacks: int

@dataclass
class SimObject: label: str; pos_room_m: tuple[float, float, float]; size_m: float = 0.15
@dataclass
class Segment:
    duration_s: float; room: str | None; camera_xyz_m: tuple[float, float, float] | None
    yaw_deg: float = 0.0        # see CameraPose.to_dict heading
    objects: list[SimObject] = field(default_factory=list)
    tag_ids: tuple[int, ...] = ()   # tags visible in this segment (projected with the room's tag poses)
@dataclass
class Scenario:
    name: str; segments: list[Segment]
    tag_size_m: float = 0.16
    rooms: dict[str, dict] = field(default_factory=dict)   # room -> {"anchor_tag": int, "anchor_height_m": float}
    K_tag: np.ndarray = field(default_factory=lambda: np.array([[918.3,0,645.8],[0,917.5,417.4],[0,0,1]]))  # 1280x800
    tag_frame_size: tuple[int, int] = (1280, 800)
    K_det: np.ndarray = ...     # 640x400: fx 459.2 fy 458.7 cx 322.9 cy 208.7 (for projecting boxes)
    @classmethod def mug_two_rooms(cls) -> Scenario
    def to_json(self) -> str; @classmethod def from_json(cls, s: str) -> Scenario

DEMO_SCENARIO = Scenario.mug_two_rooms()   # 0-6 s kitchen (tag 0; cup (1.0,0.8,1.0), bottle (-0.6,0.9,1.2); camera (0,1.4,2.5) yaw 0),
                                            # 6-9 s hallway (room None, no tags, no detections),
                                            # 9-13 s living_room (tag 1; laptop (-0.8,0.7,1.0), couch (1.0,0.4,1.5)),
                                            # 13-18 s kitchen (tag 0; cup moved to (-1.2,0.8,1.5), bottle unchanged)
                                            # kitchen = {"anchor_tag": 0, "anchor_height_m": 1.4}; living_room = {"anchor_tag": 1, ...}

class SimDetectionSource:                   # implements DetectionSource; name="sim"; classes=COCO80
    def __init__(self, scenario: Scenario | str = "mug_two_rooms", fps: float = 10.0, realtime: bool = False,
                 oracle_pose: bool = False, emit_tags: bool = True, loop: bool = False, seed: int = 0,
                 noise_px: float = 0.3, noise_xy_m: float = 0.02, noise_z_m: float = 0.04,
                 tag_fps: float = 5.0, start_ms: int | None = None) -> None
    # Per detection frame: p_cam = inv(T_room_cam) @ p_room + N(0, noise); box projected with K_det (640x400) from size_m;
    # dropped when p_cam.z <= 0.3. Tag frames at tag_fps: corners projected with K_tag (dist=None), noise_px added,
    # emitted as TagFrame(gray=None, tags=[...]). With oracle_pose=True no TagFrames are emitted and each DetectionFrame
    # carries oracle_pose=CameraPose(room, "room", T_room_cam, anchored=True, 1.0, (tag,), 0.0) (or frame "camera" in the hallway).
    # Uses numpy.random.default_rng(seed) only. timestamp_ms starts at start_ms (default: now) and advances 1000/fps per frame.
    frames_total: int                        # len(scenario) * fps

class JsonlRecorder:
    def __init__(self, path: Path) -> None
    def write(self, ev: SourceEvent) -> None      # TagFrame written without `gray`
    def close(self) -> None
class JsonlReplaySource:                          # implements DetectionSource; name="replay"
    def __init__(self, path: Path, realtime: bool = False, loop: bool = False) -> None

def make_detection_source(kind: str | None = None, *, env: Mapping[str, str] | None = None) -> DetectionSource | None
    # kind or LOCI_DETECTION_SOURCE: "none" -> None, "sim" -> SimDetectionSource(from LOCI_SIM_*), "replay" -> JsonlReplaySource(LOCI_REPLAY_PATH),
    # "oak" -> oak_source.OakDetectionSource(OakConfig.from_env()) (imported lazily)
```

### 4.3 `oak_source.py` (T2) — `import depthai as dai` only inside functions

```python
@dataclass
class OakConfig:
    fps: float = 10.0                          # LOCI_OAK_FPS
    model: str = "yolov6-nano"                 # LOCI_OAK_MODEL: zoo name or path to a .rvc2.tar.xz NNArchive
    usb2: bool = False                         # LOCI_OAK_USB2 -> dai.Device(dai.UsbSpeed.HIGH)
    axes: str = "rdf"                          # LOCI_OAK_AXES: "rdf" | "ruf"
    stereo_preset: str = "FAST_DENSITY"        # LOCI_OAK_STEREO_PRESET (member of dai.node.StereoDepth.PresetMode)
    pp_shaves: int = 2                         # LOCI_OAK_PP_SHAVES -> setPostProcessingHardwareResources(n, n)
    lr_check: bool = False                     # LOCI_OAK_STEREO_LRC
    extended_disparity: bool = False           # LOCI_OAK_STEREO_EXTENDED
    confidence: float = 0.5                    # LOCI_OAK_CONF
    bbox_scale: float = 0.5                    # LOCI_OAK_BBOX_SCALE
    depth_min_mm: int = 200                    # LOCI_OAK_DEPTH_MIN_MM
    depth_max_mm: int = 6000                   # LOCI_OAK_DEPTH_MAX_MM
    spatial_algo: str = "median"               # LOCI_OAK_SPATIAL_ALGO: median|average|min|max|mode|default ("default" = do not call the setter)
    tag_stream: tuple[int, int, float] | None = (1280, 800, 5.0)   # LOCI_TAG_STREAM "1280x800@5" | "640x400@10" | "off"
    tag_undistort: bool = False                # LOCI_TAG_UNDISTORT -> requestOutput(..., enableUndistortion=True) and dist=None
    stall_timeout_s: float = 5.0               # LOCI_OAK_STALL_S
    reconnect_backoff_s: tuple[float, float] = (1.0, 5.0)
    @classmethod def from_env(cls, env: Mapping[str, str] | None = None) -> OakConfig
    @staticmethod def parse_tag_stream(s: str) -> tuple[int, int, float] | None

@dataclass
class PipelineHandles:
    pipeline: Any; device: Any
    q_det: Any                  # sdn.out.createOutputQueue(maxSize=4, blocking=False)
    q_tag: Any | None           # tag output queue (maxSize=2, blocking=False) or None
    labels: list[str]           # sdn.getClasses()
    tag_size: tuple[int, int] | None

def build_pipeline(dai: Any, device: Any, cfg: OakConfig) -> PipelineHandles
    # pipeline = dai.Pipeline(device)
    # cam = Camera.build(CAM_A, sensorFps=cfg.fps); left/right = Camera.build(CAM_B/CAM_C, sensorFps=cfg.fps).requestOutput((640,400))
    # stereo = StereoDepth.build(left, right, presetMode=PresetMode[cfg.stereo_preset]); stereo.setDepthAlign(CAM_A);
    # stereo.setPostProcessingHardwareResources(cfg.pp_shaves, cfg.pp_shaves); if cfg.lr_check: setLeftRightCheck(True); if cfg.extended_disparity: setExtendedDisparity(True)
    # model = dai.NNArchive(cfg.model) if cfg.model ends with ".tar.xz" and exists else dai.NNModelDescription(cfg.model, platform="RVC2")
    # sdn = SpatialDetectionNetwork.build(cam, stereo, model, fps=cfg.fps); sdn.input.setBlocking(False); setConfidenceThreshold; setBoundingBoxScaleFactor;
    # setDepthLowerThreshold/UpperThreshold; if cfg.spatial_algo != "default": setSpatialCalculationAlgorithm(SpatialLocationCalculatorAlgorithm[ALGO.upper()])
    # tag: out = cam.requestOutput((w,h), type=dai.ImgFrame.Type.NV12, fps=tag_fps[, enableUndistortion=True when cfg.tag_undistort])
def spatial_to_detections(msg: Any, labels: list[str], cfg: OakConfig) -> list[SpatialDetection]   # pure; z<=0 -> xyz None; unknown label index -> f"class_{i}"
def tag_frame_from_img(frame: Any, cfg: OakConfig, timestamp_ms: int) -> TagFrame
    # gray = frame.getFrame()[:H] when frame.getFrame().shape == (H*3//2, W) else cv2.cvtColor(frame.getCvFrame(), COLOR_BGR2GRAY)
    # tf = frame.getTransformation(); K = np.array(tf.getIntrinsicMatrix()); dist = None if cfg.tag_undistort else np.array(tf.getDistortionCoefficients())
def wait_for_unbooted(dai: Any, timeout_s: float = 45.0, poll_s: float = 1.0) -> bool   # polls dai.Device.getAllAvailableDevices() for state X_LINK_UNBOOTED

class OakDetectionSource:                      # implements DetectionSource; name="oak"; classes filled after first open
    def __init__(self, cfg: OakConfig, recorder: JsonlRecorder | None = None, clock: HostClock | None = None,
                 opener: Callable[[], PipelineHandles] | None = None) -> None   # opener injectable for tests
    def events(self) -> Iterator[SourceEvent]
    # supervisor: while not stopped: try open -> build -> loop { q_det.tryGet() -> DetectionFrame; q_tag.tryGet() -> TagFrame; sleep 2 ms;
    #   if no detection message for cfg.stall_timeout_s while pipeline.isRunning(): raise StallError }
    #   except Exception: status.state="waiting_for_camera", reconnects += 1, close pipeline/device in finally, wait_for_unbooted, sleep backoff (1 -> 5 s doubling), continue
```

Console entry point (T1): `deploy/handheld/scripts/oak_console.py [--seconds N] [--sim] [--scenario NAME] [--usb2] [--model NAME|PATH] [--no-tags] [--record out.jsonl] [--check-y-sign] [--rate 2.0]`. Adds the repo root to `sys.path`, imports only `demo_spatial.app.detection_source` / `oak_source` (never `loci`), prints one line per detection at ≤ `--rate` Hz: `HH:MM:SS.mmm  label  conf  x=+0.12m y=-0.05m z=1.84m  box=(xmin,ymin,xmax,ymax)`, a 1 s summary `fps=10.1 det/s=.. labels={cup:3,...} state=running usb=HIGH reconnects=0`, and with `--check-y-sign` a final verdict: mean y of boxes with `cy < 0.4` vs `cy > 0.6`; prints `RDF confirmed (above-centre objects have y<0)` or `Y appears UP: set LOCI_OAK_AXES=ruf`. Exit code 0 when ≥ 1 detection frame was received (or sim ran to completion), 2 otherwise.

### 4.4 `geometry.py` (T3) — numpy only; cv2 imported lazily inside the two PnP functions

```python
Array = np.ndarray
def tag_object_points(tag_size_m: float) -> Array                 # (4,3) TL,TR,BR,BL
def make_T(R: Array, t: Any) -> Array; def invert_T(T: Array) -> Array; def apply_T(T: Array, p: Any) -> Array
def rot_x(a) / rot_y(a) / rot_z(a) -> Array; def rodrigues(rvec) -> Array; def log_rot(R) -> Array
def rotation_angle_deg(R1: Array, R2: Array) -> float; def so3_mean(Rs: list[Array]) -> Array; def mean_T(Ts: list[Array]) -> Array
def yaw_deg_of(T_room_cam: Array) -> float                        # heading as defined in CameraPose.to_dict
def camera_pose_T(xyz_room_m: Sequence[float], yaw_deg: float) -> Array   # inverse of the above for the simulator/tests
FRONTAL_R_CAM_TAG = np.diag([1.0, -1.0, -1.0])
@dataclass
class TagPose: tag_id: int; T_cam_tag: Array; reproj_err_px: float; ambiguity_ratio: float; side_px: float; view_angle_deg: float
def solve_tag_pose(tag_id: int, corners_px: Any, K: Array, dist: Array | None, tag_size_m: float, min_distinct_deg: float = 3.0) -> TagPose
    # IPPE_SQUARE candidates (solvePnPGeneric) + SOLVEPNP_ITERATIVE, each solvePnPRefineLM-refined, lowest reprojection RMS wins; raises ValueError when no solution
def solve_room_pose_multi(detections: list[tuple[int, Any]], tag_map: dict[int, Array], K: Array, dist: Array | None, tag_size_m: float) -> tuple[Array, float] | None
def normalize_upright(T_cam_tag: Array) -> tuple[Array, bool]
def project_points(points_3d: Array, T_cam_x: Array, K: Array) -> Array     # pinhole, no distortion; used by the simulator and tests
```
`geometry.py` is a faithful port of `/private/tmp/claude-501/-Users-0x7d0-git-loci-db/8139ebf0-1ba6-436c-8885-212cc37c9445/scratchpad/design/apriltag_pose.py` (81 self-checks passed on Mac and Pi); do not change the maths.

### 4.5 `room_anchor.py` (T3)

```python
class TagDetector(Protocol):
    name: str
    def detect(self, gray: np.ndarray) -> list[TagDetection]: ...   # corners TL,TR,BR,BL
class PyAprilTagsDetector:  # pyapriltags, families="tag36h11", nthreads=1, quad_decimate=2.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25; keeps hamming==0 and decision_margin>=min_margin; corners[[1,0,3,2]]
    def __init__(self, family: str = "tag36h11", quad_decimate: float = 2.0, nthreads: int = 1, min_margin: float = 30.0) -> None
class ArucoTagDetector:     # cv2.aruco DICT_APRILTAG_36h11, CORNER_REFINE_SUBPIX; decision_margin=100.0, hamming=0
    def __init__(self, family: str = "tag36h11") -> None
def make_tag_detector(kind: str = "auto") -> TagDetector | None     # LOCI_TAG_DETECTOR auto|pyapriltags|aruco|none; auto = pyapriltags if importable else aruco if cv2 else None

@dataclass
class TagFilter: min_side_px=30.0; max_reproj_err_px=1.0; min_ambiguity_ratio=1.5; max_view_angle_deg=60.0; max_range_m=6.0
    def accept(self, p: TagPose) -> bool
@dataclass
class RoomAnchorMap: room: str; anchor_height_m=1.4; register_samples=10; tags: dict[int, Array]; anchor_id: int | None
    def observe(self, poses: list[TagPose]) -> Array | None     # T_room_cam or None; self-registers co-visible tags
    def to_dict(self) -> dict; @classmethod def from_dict(cls, d: dict) -> RoomAnchorMap
@dataclass
class PoseHold: hold_s=10.0; alpha_t=0.35; alpha_r=0.35; max_jump_m=0.75; max_jump_deg=25.0; jump_confirm=3
    def update(self, T_room_cam: Array, now_s: float, room: str) -> None
    def current(self, now_s: float) -> tuple[Array | None, float, bool]   # (T, anchor_confidence, anchored)

@dataclass
class RoomSpec:
    name: str; anchor_tag: int; anchor_height_m: float = 1.4; anchor_name: str = "door tag"
    extents_m: dict[str, tuple[float, float]] = DEFAULT_ROOM_EXTENTS
    landmarks: dict[str, tuple[float, float, float]] = {}
    extra_tags: dict[int, Array] = {}            # manually surveyed T_room_tag for tags never co-visible with the anchor
class RoomMap:
    rooms: dict[str, RoomSpec]; tag_size_m: float; tag_family: str
    @classmethod def load(cls, path: str | os.PathLike[str]) -> RoomMap     # rooms.json (schema below)
    @classmethod def default(cls) -> RoomMap                                 # kitchen(tag 0) + living_room(tag 1)
    def room_for_tag(self, tag_id: int) -> RoomSpec | None
    def extents(self, room: str | None) -> dict[str, tuple[float, float]]    # unknown/None room -> CAMERA_BOX
    def get(self, room: str) -> RoomSpec | None
    def names(self) -> list[str]
def slug(room: str) -> str                      # lower, spaces/"/" -> "_"
def room_to_unit(p_m: Sequence[float], extents: dict[str, tuple[float, float]]) -> tuple[tuple[float, float, float], bool]
def unit_to_room(u: Sequence[float], extents: dict[str, tuple[float, float]]) -> tuple[float, float, float]
CAMERA_BOX: dict[str, tuple[float, float]]; DEFAULT_ROOM_EXTENTS: dict[str, tuple[float, float]]

class RoomLocalizer:
    def __init__(self, rooms: RoomMap, detector: TagDetector | None, *, tag_filter: TagFilter = TagFilter(),
                 hold: PoseHold | None = None, stale_s: float = 60.0, tags_path: Path | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None
    def update(self, tf: TagFrame) -> CameraPose       # detect (if tf.gray and detector) or use tf.tags; solve; filter; map.observe (multi-tag when >=2 known); hold.update; room switch resets hold
    def set_oracle(self, pose: CameraPose, now_s: float | None = None) -> None   # sim/test injection
    def current(self, now_s: float | None = None) -> CameraPose
        # hold.current(now) -> (T, conf, anchored): T and age <= stale_s -> CameraPose(room, "room", T, anchored, conf, tag_ids, age);
        # else CameraPose(last_room_or_None, "camera", None, False, 0.0, (), age)
    def to_room(self, pose: CameraPose, xyz_cam_m: Sequence[float]) -> tuple[float, float, float] | None   # None when pose.frame != "room"
    def status(self) -> dict          # {"room", "frame", "anchored", "anchor_confidence", "age_s", "tags_visible": [...], "tags_registered": {room: [ids]}, "detector": name, "last_tag_ms", "tag_frames", "poses_accepted", "poses_rejected"}
    def save(self) -> None; def load(self) -> None   # tags.json = {"format": 1, "rooms": {room: RoomAnchorMap.to_dict()}}
```

`rooms.json` schema (`LOCI_ROOMS_FILE`, example at `deploy/handheld/config/rooms.example.json`):
```json
{"format": 1, "tag_size_m": 0.160, "tag_family": "tag36h11",
 "rooms": [
  {"name": "kitchen", "anchor_tag": 0, "anchor_height_m": 1.4, "anchor_name": "door tag",
   "extents_m": {"x": [-3.0, 3.0], "y": [0.0, 2.5], "z": [0.0, 6.0]},
   "landmarks": {"table": [1.0, 0.75, 1.2]},
   "extra_tags": {}},
  {"name": "living_room", "anchor_tag": 1, "anchor_height_m": 1.4, "anchor_name": "door tag"}
 ]}
```

### 4.6 `label_embedding.py` and `label_resolver.py` (T6)

```python
@runtime_checkable
class LabelEmbedder(Protocol):
    name: str; dim: int
    def embed(self, text: str) -> list[float]: ...
    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...
def normalize_label(text: str) -> str                       # strip, lower, collapse whitespace
def hash_label_embedding(label: str, dim: int = 128) -> list[float]   # EXACT port of today's spatial_memory._label_embedding (sin/cos first 8 dims + LCG), L2-normalised
class HashLabelEmbedder:  name = "hash:128"; dim = 128
class StaticLabelEmbedder: # model2vec format: tokenizers + safetensors + numpy; encode(add_special_tokens=False), drop [UNK], mean-pool, L2-normalise; per-label cache
    def __init__(self, model_dir: str | os.PathLike[str]) -> None   # raises ImportError / OSError / KeyError
    name = f"static:{Path(model_dir).name}"; dim = 256 for potion-base-8M
DEFAULT_MODEL_DIR = <repo>/deploy/handheld/models/potion-base-8M
def static_model_available(model_dir: str | None = None) -> bool
def make_embedder(kind: str | None = None, model_dir: str | None = None, *, env: Mapping[str, str] | None = None) -> LabelEmbedder
    # kind or LOCI_LABEL_EMBEDDER (default "hash"): "hash" | "static" (raises if unavailable) | "auto" (static if available else hash)
    # model_dir or LOCI_EMBED_MODEL_DIR or DEFAULT_MODEL_DIR
def cosine(a: Sequence[float], b: Sequence[float]) -> float

@dataclass
class Resolution:
    label: str | None; how: str          # "exact" | "alias" | "embedding" | "none"
    score: float; suggestions: list[str]
    def to_dict(self) -> dict
DEFAULT_ALIASES: dict[str, str] = {"mug": "cup", "phone": "cell phone", "mobile": "cell phone", "television": "tv", "telly": "tv",
    "sofa": "couch", "settee": "couch", "table": "dining table", "water bottle": "bottle", "purse": "handbag", "bag": "handbag",
    "remote control": "remote", "plant": "potted plant", "computer": "laptop", "glass": "wine glass", "bike": "bicycle"}
DEFAULT_DENYLIST: set[frozenset[str]] = {{"keys","keyboard"}, {"dog","hot dog"}, {"bear","teddy bear"}}
class LabelResolver:
    def __init__(self, vocabulary: Iterable[str], embedder: LabelEmbedder, *, aliases: Mapping[str, str] = DEFAULT_ALIASES,
                 denylist: Iterable[Iterable[str]] = DEFAULT_DENYLIST, threshold: float = 0.45, max_suggestions: int = 5) -> None
    def set_vocabulary(self, labels: Iterable[str]) -> None     # replaces; embeddings cached
    def resolve(self, text: str) -> Resolution
        # norm = normalize_label(text) with a leading "my "/"the " and a trailing "s" (plural) tried as variants;
        # 1 exact in vocabulary -> ("exact", 1.0); 2 alias -> ("alias", 1.0) if alias target in vocabulary;
        # 3 if embedder.name startswith "static": argmax cosine over vocabulary, skip denylisted pairs, accept >= threshold -> ("embedding", score);
        # 4 else ("none", best_score, suggestions = top-k by cosine (static) or [] (hash))
```

### 4.7 `scene_ingestion.py` (T5) — additions only; `process_frame` unchanged

```python
@dataclass
class Detection:            # existing fields kept in order: label, cx, cy, width, height, confidence, source="yolo", depth_m=None
    xyz_cam_m: tuple[float, float, float] | None = None   # camera frame, metres
    pos_room_m: tuple[float, float, float] | None = None  # room frame, metres (set by the runtime before process_detections)
    room: str | None = None
    frame: str = "image"                                   # "image" | "camera" | "room"
    anchored: bool = False
    anchor_confidence: float = 0.0
    def to_dict(self) -> dict   # adds xyz_cam_m, pos_room_m, room, frame, anchored, anchor_confidence, distance_m when present

class SceneIngestion:
    def __init__(self, memory, vlm_client=None, use_vlm_fallback=True, classes=None, consensus_window_ms=1000,
                 consensus_min_count=2, consensus_iou_threshold=0.4, *, consensus_radius_m: float = 0.3,
                 consensus_refire_ms: int = 0) -> None
    def process_detections(self, detections: list[Detection], timestamp_ms: int) -> list[Detection]
        # synchronous; for each det: _update_consensus(label=..., cx, cy, confidence, timestamp_ms, depth_m=det.depth_m,
        #   xyz_cam_m=det.xyz_cam_m, pos_room_m=det.pos_room_m, room=det.room, frame=det.frame, anchored=det.anchored,
        #   anchor_confidence=det.anchor_confidence, source=det.source); sets last_detections, frame_count += 1
    def _update_consensus(self, label, cx, cy, confidence, timestamp_ms, depth_m, *, xyz_cam_m=None, pos_room_m=None,
                          room=None, frame="image", anchored=False, anchor_confidence=0.0, source="yolo") -> None
        # buffer key = (label, frame, room or ""); entries are _BufEntry dataclasses; eviction by window as today;
        # match: both have pos_room_m -> euclid <= consensus_radius_m; elif both have xyz_cam_m -> euclid <= consensus_radius_m; else _iou > consensus_iou_threshold
        # fire when len(matches)+1 >= consensus_min_count AND (timestamp_ms - last_fire[key]) >= consensus_refire_ms:
        #   observe(label=, cx=avg, cy=avg, confidence=max, timestamp_ms=, depth_m=latest, xyz_cam_m=mean, pos_room_m=mean, room=, anchored=all(anchored),
        #           anchor_confidence=mean, source=) ; CONSUME the matched entries (delete from buffer); last_fire[key] = timestamp_ms
        # else: append entry (also during hold-off, so averaging continues)
    @property consensus_stats -> dict   # {"buffer_entries": int, "fires": int, "suppressed_by_holdoff": int}
```
`observe` is always called with keyword arguments; the legacy call shape (`label, cx, cy, confidence, timestamp_ms, depth_m`) must remain exactly what the mock in `tests/test_temporal_consensus.py` sees plus the new keywords.

### 4.8 `spatial_memory.py` (T4), `loci_bridge.py` (T4), `persistence.py` (T4)

```python
EMBED_DIM = 128                                   # kept for compatibility
def _label_embedding(label: str) -> list[float]  # == label_embedding.hash_label_embedding(label, 128); staged test imports it

@dataclass
class ObjectObservation:   # existing: label, cx, cy, confidence, timestamp_ms, state_id, depth_m=None
    room: str | None = None; frame: str = "image"
    pos_m: tuple[float, float, float] | None = None       # room metres
    xyz_cam_m: tuple[float, float, float] | None = None
    anchored: bool = False; anchor_confidence: float = 0.0
    source: str = "camera"; first_seen_ms: int | None = None; merge_count: int = 0
    restored: bool = False                                 # timestamp_ms < memory.started_at_ms
    scene_id: str = ""
    age_seconds (property, unchanged); position_description (property: legacy wording for frame "image", else phrasing.describe_position(frame, pos_m, xyz_cam_m))
    def to_dict(self) -> dict   # existing keys + room, frame, x_m, y_m, z_m (None when no pos_m), xyz_cam_m, anchored, anchor_confidence, source, first_seen_ms, restored, scene_id

@dataclass
class Change: kind: str   # "new" | "moved" | "gone" | "seen"
    label: str; room: str | None; last_seen_ms: int; first_seen_ms: int | None
    pos_m: tuple[float, float, float] | None; pos_before_m: tuple[float, float, float] | None; delta_m: float | None
    def to_dict(self) -> dict

def scene_key(label: str, frame: str, room: str | None) -> str
    # "image" -> label ; "camera" -> f"unanchored/{label}" ; "room" -> f"{slug(room)}/{label}"   (label already normalised)
def split_scene_key(scene_id: str) -> tuple[str | None, str]    # (room_or_None_or_"unanchored", label)

class SpatialMemory:
    def __init__(self, epoch_size_ms: int = 5000, label_confidence_overrides: dict[str, float] | None = None,
                 dedup_iou_threshold: float = 0.5, dedup_window_ms: int = 5000, *,
                 embedder: LabelEmbedder | None = None,          # default make_embedder() (-> hash unless env says otherwise)
                 rooms: RoomMap | None = None,                   # default RoomMap.default()
                 dedup_radius_m: float = 0.35, moved_threshold_m: float = 0.5,
                 persistence: MemoryPersistence | None = None) -> None
        # self._client = LocalLociClient(epoch_size_ms=..., vector_size=embedder.dim, distance="cosine")  (decay default kept)
        # self._lock = threading.RLock(); self.started_at_ms = now; self.version = 0
    # --- write ---
    def observe(self, label: str, cx: float, cy: float, confidence: float = 1.0, timestamp_ms: int | None = None,
                depth_m: float | None = None, min_confidence: float = 0.55, *,
                xyz_cam_m: tuple[float, float, float] | None = None, pos_room_m: tuple[float, float, float] | None = None,
                room: str | None = None, anchored: bool = False, anchor_confidence: float = 0.0,
                source: str = "camera") -> str | None
        # frame = "room" if pos_room_m is not None and room else "camera" if xyz_cam_m is not None else "image"
        # x,y,z: room -> room_to_unit(pos_room_m, rooms.extents(room)); camera -> room_to_unit(xyz_cam_m, CAMERA_BOX); image -> (cx, cy, min(depth/5,1) or 0)
        # vector = embedder.embed(label); scene_id = scene_key(label, frame, room)
        # metadata = {"label", "room", "frame", "pos_m", "xyz_cam_m", "cx", "cy", "depth_m", "anchored", "anchor_confidence", "source",
        #             "first_seen_ms", "merge_count", "clipped", "embedder": embedder.name}
        # merge via _try_merge (keyed, see 7.3) else client.insert; update _scene_index, _latest[label], _latest_by_scene[scene_id], _records; persistence.append(...); version += 1
    def note_presence(self, room: str, timestamp_ms: int) -> None      # runtime calls at <=1 Hz while anchored in `room`
    # --- read ---
    def where_is(self, label: str, limit: int = 5, min_confidence: float | None = None, *, room: str | None = None) -> list[ObjectObservation]
        # scene_ids = [label, *(f"{slug(r)}/{label}" for r in rooms_seen), f"unanchored/{label}"] or [f"{slug(room)}/{label}"] when room given
        # window = (ts of the limit-th newest index slot across scene_ids [with confidence >= min_confidence], newest ts) ; no slots -> []
        # client.query(vector=embed(label), time_window_ms=window, limit=limit*4, _extra_payload_filter={"scene_id": scene_ids}, min_confidence=...)
        # -> ObjectObservation list sorted by timestamp_ms desc, truncated to limit. NEVER resolves labels; exact filter only.
    def history(self, label: str, limit: int = 20, *, room: str | None = None) -> list[ObjectObservation]
    def objects_in_region(self, x_min, x_max, y_min, y_max, limit: int = 20, *, room: str | None = None) -> list[ObjectObservation]
        # spatial_bounds as today; internal limit 200; room -> _extra_payload_filter {"scene_id": [f"{slug(room)}/{l}" for l in labels_seen_in_room]}; dedupe by label newest-first
    def current_objects(self, room: str | None = None) -> list[ObjectObservation]      # from _latest (per label) filtered by room
    def recent_changes(self, window_seconds: float = 30.0, room: str | None = None) -> list[ObjectObservation]
    def changes(self, room: str | None, window_seconds: float = 600.0, *, now_ms: int | None = None) -> list[Change]   # see 7.5
    def resolve_label(self, text: str) -> Resolution        # LabelResolver over detector vocabulary (set_vocabulary) ∪ tracked_labels
    def set_vocabulary(self, labels: Iterable[str]) -> None
    @property rooms_seen -> list[str]; @property tracked_labels -> list[str]; @property observation_count -> int; @property rooms -> RoomMap
    def stats(self) -> dict      # existing keys + "rooms_seen", "frames": {"image": n, "camera": n, "room": n}, "persistence": persistence.status() or None, "embedder": name, "version"
    # --- lifecycle ---
    def restore(self) -> RestoreReport    # loads via persistence, insert_batch, rebuilds indices, snapshot synchronously; no-op report when persistence is None
    def flush(self) -> None               # synchronous snapshot
    def clear(self) -> None               # new client, empty indices, persistence.clear()
    def close(self) -> None               # flush + stop writer
    def export_records(self) -> list[ObservationRecord]
```

`loci_bridge.py` (the only module allowed to touch private client members):
```python
def data_collection(client: LocalLociClient) -> str              # client._data_collection
def delete_point(client: LocalLociClient, state_id: str) -> bool # client.store.delete_points(data_collection(client), [state_id]) > 0
def point_count(client: LocalLociClient) -> int                  # client.store.collection_count(data_collection(client))
def make_client(epoch_size_ms: int, vector_size: int) -> LocalLociClient
```

`persistence.py`:
```python
class ObservationRecord(TypedDict):
    id: str; scene_id: str; label: str; room: str | None; frame: str
    x: float; y: float; z: float; timestamp_ms: int; confidence: float
    prev_state_id: str | None; metadata: dict[str, Any]
@dataclass
class RestoreReport: records: int; rooms: list[str]; newest_ms: int | None; bad_lines: int; source: str   # source: "snapshot" | "snapshot+journal" | "bak" | "empty" | "corrupt"
    def to_dict(self) -> dict
class MemoryPersistence:
    def __init__(self, data_dir: str | os.PathLike[str], *, snapshot_interval_s: float = 20.0, journal_fsync_s: float = 1.0,
                 max_records: int = 100_000, enabled: bool = True) -> None
    snapshot_path = data_dir/"memory.snapshot.json"; bak_path = data_dir/"memory.snapshot.bak.json"; journal_path = data_dir/"memory.journal.jsonl"
    def load(self) -> tuple[list[ObservationRecord], dict, RestoreReport]     # (records after journal overlay, header, report); corrupt snapshot -> renamed memory.snapshot.corrupt-<ts>.json then .bak tried
    def begin_session(self, header: dict, records: Iterable[ObservationRecord]) -> None   # SYNCHRONOUS snapshot + journal truncate (called by SpatialMemory.restore before any write) and starts the writer thread
    def append_upsert(self, rec: ObservationRecord) -> None   # enqueue {"op":"upsert","rec":rec}; never blocks the caller
    def append_delete(self, state_id: str) -> None            # enqueue {"op":"delete","id":id}
    def mark_dirty(self) -> None
    def snapshot_now(self, header: dict, records: Iterable[ObservationRecord]) -> None   # atomic: tmp -> fsync -> os.replace; previous -> .bak; journal truncated after
    def request_snapshot(self) -> None                        # writer thread snapshots on its next tick (uses the provider registered in begin_session)
    def status(self) -> dict  # {"enabled", "data_dir", "last_snapshot_ms", "journal_lines", "journal_bytes", "queue", "errors", "last_restore": report}
    def clear(self) -> None; def close(self) -> None           # flush queue, final snapshot, join thread
```
Snapshot file: `{"format": 2, "saved_at_ms": int, "embedder": {"name": str, "dim": int}, "epoch_size_ms": int, "observation_count": int, "started_at_ms": int, "records": [ObservationRecord, ...]}` — records newest-last; vectors are never stored. Journal lines: `{"op":"upsert","rec":{...}}` or `{"op":"delete","id":"..."}`; unparsable lines (torn tail) are skipped and counted. Restore order: snapshot (or `.bak`) → journal overlay by id (last op wins; `delete` removes) → keep newest `max_records` → sort by `(scene_id, timestamp_ms)` → `WorldState(x,y,z,timestamp_ms, vector=embed(metadata.label), scene_id, confidence, metadata)` → one `client.insert_batch` → new ids → `begin_session(header, records_with_new_ids)`.

### 4.9 `phrasing.py` (T10)

```python
def format_age(seconds: float) -> str; def format_clock(timestamp_ms: int) -> str
def round_half_m(x: float) -> float
def height_phrase(y_m: float) -> str          # <0.3 "on the floor" | <0.6 "low down" | <=1.1 "at table height" | <=1.5 "at chest height" | else "up high"
def lateral_phrase(x_m: float, perspective: str) -> str   # perspective "walk_in": x>0 -> "to your left", x<0 -> "to your right"; |x|<0.5 -> "straight ahead"; perspective "facing_tag": mirrored
def describe_position(frame: str, pos_m=None, xyz_cam_m=None, cx=0.5, cy=0.5, *, anchor_name="door tag", room=None) -> str
    # room: "about {z} m into the {room} and {|x|} m {lateral} as you walk in from the {anchor_name}, {height}"
    # camera: "about {z} m in front of the camera{, slightly left/right}, {height relative to camera}"   (y<-0.3 "above eye line", y>0.3 "below eye line")
    # image: legacy "on the left, toward the back" wording (identical to today's text)
def egocentric_phrase(pos_room_m, T_room_cam) -> str      # p_cam = inv(T) @ p: "about {z} m ahead of you{ and {|x|} m to your left/right}, {height}"
def landmark_phrase(pos_room_m, landmarks: Mapping[str, Sequence[float]], max_dist_m: float = 1.2) -> str | None   # "on the kitchen table" style: f"near the {name}"
def where_is_answer(spoken_name: str, obs, *, room_spec=None, current_pose: CameraPose | None = None, now_ms: int | None = None,
                    resolved_label: str | None = None, started_at_ms: int | None = None) -> str
    # "Your mug (I track it as a cup) was last seen in the kitchen, {landmark or describe_position/egocentric}. That was 3 minutes ago, at 2:15 PM."
    # + ", roughly — I wasn't sure of my position" when not anchored (frame camera or anchored False); + " That was before I restarted." when restored
def changes_answer(changes: list[Change], room: str | None, window_s: float) -> str
    # "In the kitchen in the last 10 minutes: the cup moved about 2.5 m; a bottle is new; I no longer see the laptop." / "Nothing changed in the kitchen in the last 10 minutes."
def unknown_label_answer(name: str, resolution: Resolution, examples: Sequence[str]) -> str
    # "I can't recognise 'keys' with this camera. Did you mean keyboard? I can look for cup, bottle, cell phone, remote, book."
def not_seen_answer(name: str, room: str | None) -> str
```

### 4.10 `voice_pipeline.py` (T10) — additions

```python
@dataclass
class QueryIntent: kind: str; object_name: str; raw_text: str; room: str = ""     # room extracted from "in the kitchen" / "in my living room"; stripped from object_name
_CHANGES_PATTERNS += r"what(?:'s| has| have| is)\s+(?:changed|moved|new|different)(?:\s+(?:in|around)\s+(?:the\s+|my\s+)?(?P<room>[\w ]+?))?(?:\?|$)"
_LIST_PATTERNS += r"what(?:'s| is)\s+in\s+(?:the\s+|my\s+)?(?P<room>[\w ]+?)(?:\?|$)"
def parse_intent(text: str, known_rooms: Iterable[str] = ()) -> QueryIntent   # room names matched case-insensitively with "_"/" " equivalence
def build_response_text(intent: QueryIntent, memory: SpatialMemory, vlm_answer: str = "", *, current_pose: CameraPose | None = None) -> str
    # where_is/history: res = memory.resolve_label(intent.object_name); none -> unknown_label_answer; else memory.where_is(res.label, room=intent.room or None) -> phrasing.where_is_answer
    # changes: memory.changes(intent.room or None, 600) -> phrasing.changes_answer ; list_objects: current_objects(room)
class VoicePipeline:
    def __init__(self, memory, vlm_client=None, *, pose_provider: Callable[[], CameraPose | None] | None = None) -> None
    class QueryResult: transcription; intent; answer_text; answer_audio; latency_ms; resolution: Resolution | None = None; results: list[ObjectObservation] = []; changes: list[Change] = []
    async def handle_text_query(self, text: str, *, synthesize_response: bool = True) -> QueryResult   # default unchanged (TTS on)
    @staticmethod def audio_mime(audio: bytes) -> str     # b"RIFF" -> "audio/wav", b"ID3"/0xFFFB -> "audio/mpeg", else "application/octet-stream"
TTS._run_piper_sync: uses voice.synthesize_wav(text, wf) when hasattr(voice, "synthesize_wav") else voice.synthesize(text, wf)
```

### 4.11 `handheld.py` and `main.py` (T7)

```python
@dataclass
class HandheldConfig:
    detection_source: str = "none"; data_dir: Path; persist: bool = True; snapshot_interval_s: float = 20.0
    rooms_file: Path | None; tag_size_m: float = 0.16; tag_detector: str = "auto"; pose_hold_s: float = 10.0; pose_stale_s: float = 60.0
    consensus_refire_ms: int = 2000; consensus_radius_m: float = 0.3; dedup_radius_m: float = 0.35; moved_threshold_m: float = 0.5
    ws_push_hz: float = 5.0; sim_realtime: bool; sim_oracle_pose: bool; label_embedder: str
    @classmethod def from_env(cls, env: Mapping[str, str] | None = None) -> HandheldConfig

class HandheldRuntime:
    def __init__(self, source: DetectionSource, localizer: RoomLocalizer | None, ingestion: SceneIngestion, memory: SpatialMemory,
                 cfg: HandheldConfig, recorder: JsonlRecorder | None = None) -> None
    def start(self) -> None                   # daemon thread "handheld-runtime"
    def stop(self, timeout_s: float = 5.0) -> None
    def run_until_done(self, max_events: int | None = None) -> int   # synchronous; tests; returns detection frames processed
    def step(self, ev: SourceEvent) -> None   # TagFrame -> localizer.update ; DetectionFrame -> pose = ev.oracle_pose or localizer.current(); for each SpatialDetection build Detection(... frame=pose.frame, room=pose.room, pos_room_m=localizer.to_room(...), anchored, anchor_confidence); ingestion.process_detections; memory.note_presence(room, ts) when anchored (<=1 Hz)
    latest_frame: DetectionFrame | None; latest_pose: CameraPose | None; seq: int; frames: int; errors: int; done: bool
    def status(self) -> dict   # {"source": SourceStatus.to_dict(), "localizer": localizer.status() or None, "frames", "errors", "done", "thread_alive", "ingestion": ingestion.consensus_stats}

def build_handheld(memory: SpatialMemory, ingestion: SceneIngestion, cfg: HandheldConfig) -> tuple[HandheldRuntime | None, RoomLocalizer | None]
```

`main.py` keeps module-level singletons and every existing endpoint/WS message unchanged. Construction order at import: `cfg = HandheldConfig.from_env()`, `rooms = RoomMap.load(cfg.rooms_file) if cfg.rooms_file else RoomMap.default()`, `persistence = MemoryPersistence(cfg.data_dir, ...) if cfg.persist else None`, `memory = SpatialMemory(epoch_size_ms=5000, embedder=make_embedder(), rooms=rooms, dedup_radius_m=..., moved_threshold_m=..., persistence=persistence)`, `ingestion = SceneIngestion(memory, vlm_client, use_vlm_fallback=True, consensus_refire_ms=cfg.consensus_refire_ms, consensus_radius_m=cfg.consensus_radius_m)`, `runtime, localizer = build_handheld(memory, ingestion, cfg)`, `voice = VoicePipeline(memory, vlm_client, pose_provider=lambda: localizer.current() if localizer else None)`. Lifespan: `restore_report = memory.restore()`; `memory.set_vocabulary(runtime.source.classes if runtime else ingestion.classes)`; `runtime.start()`; start `_live_push_loop()` task; `yield`; cancel task; `runtime.stop()`; `await ingestion.stop_capture()`; `memory.close()`. `async def _broadcast(msg: dict) -> None` factors the duplicated WS fan-out and is used by the existing ingest endpoints too.

REST additions (existing responses unchanged except where noted as additive):

| Endpoint | Request | Response |
|---|---|---|
| `GET /phone` | — | `phone.html` |
| `POST /api/query` | `{"text": str, "tts": bool=false, "room": str|null}` | `{"answer": str, "intent": {"kind","object_name","room"}, "resolution": Resolution.to_dict()|null, "results": [ObjectObservation.to_dict()], "changes": [Change.to_dict()], "latency_ms": float, "audio_b64": str|null, "audio_mime": str|null}` |
| `POST /api/voice/text-query` | unchanged | unchanged keys + `"intent.room"`, `"resolution"`, `"audio_mime"` |
| `POST /api/voice/query` | unchanged | unchanged + `"resolution"`, `"audio_mime"` |
| `POST /api/voice/tts` | unchanged | `Response(media_type=VoicePipeline.audio_mime(audio))` |
| `GET /api/rooms` | — | `{"rooms": [{"name","anchor_tag","anchor_height_m","anchor_name","extents_m","landmarks","tags_registered":[ids],"last_anchored_ms":int|null}], "current": CameraPose.to_dict()|null, "tag_size_m": float}` |
| `GET /api/anchor` | — | `CameraPose.to_dict()` (or `{"room": null, "frame": "camera", "anchored": false, ...}` when no localizer) |
| `GET /api/objects?room=` | optional room | existing shape, filtered |
| `GET /api/objects/changes?room=&window_seconds=600` | — | `{"room": str|null, "window_seconds": float, "changes": [Change.to_dict()], "count": int}` |
| `GET /api/objects/changes/recent` | unchanged | unchanged |
| `GET /api/handheld/status` | — | `{"config": {...non-secret cfg...}, "runtime": runtime.status()|null, "memory": memory.stats(), "restore": RestoreReport.to_dict()|null, "started_at_ms": int}` |
| `POST /api/handheld/snapshot` | — | `{"status": "ok", "persistence": persistence.status()}` (calls `memory.flush()`) |
| `POST /api/memory/clear` | `{"confirm": true}` | `{"status": "cleared"}` (400 without confirm) |

WebSocket `/ws` messages (JSON): existing `{"type":"snapshot","objects":[...],"observation_count":N,"stats":{...}}` every 5 s (+ `"rooms_seen": [...]`, `"anchor": CameraPose.to_dict()|null`); existing `{"type":"detections","detections":[Detection.to_dict()...],"frame_count":N,"observation_count":N}` for browser frames; new pushes from `_live_push_loop` at ≤ `LOCI_WS_PUSH_HZ`: `{"type":"detections","source":"oak"|"sim","timestamp_ms":int,"seq":int,"detections":[{...Detection.to_dict(), "distance_m": float|null}],"frame_count":N,"observation_count":N,"anchor": CameraPose.to_dict()}` when `runtime.seq` changed, and `{"type":"anchor", **CameraPose.to_dict(), "source_state": str}` once per second.

`CameraPose.to_dict()` = `{"room": str|null, "frame": "room"|"camera", "anchored": bool, "anchor_confidence": float, "camera_xyz_m": [x,y,z]|null, "heading_deg": float|null, "tag_ids": [..], "age_s": float|null}`.

### 4.12 Environment variables (all read once at import/startup; defaults in parentheses)

Source: `LOCI_DETECTION_SOURCE` (none|sim|oak|replay; `none`), `LOCI_SIM_SCENARIO` (`mug_two_rooms`), `LOCI_SIM_REALTIME` (`1`), `LOCI_SIM_ORACLE_POSE` (`0`), `LOCI_SIM_LOOP` (`0`), `LOCI_REPLAY_PATH`, `LOCI_RECORD_PATH` (unset = no recording).
OAK: `LOCI_OAK_FPS` (10), `LOCI_OAK_MODEL` (`yolov6-nano`), `LOCI_OAK_USB2` (0), `LOCI_OAK_AXES` (`rdf`), `LOCI_OAK_STEREO_PRESET` (`FAST_DENSITY`), `LOCI_OAK_PP_SHAVES` (2), `LOCI_OAK_STEREO_LRC` (0), `LOCI_OAK_STEREO_EXTENDED` (0), `LOCI_OAK_CONF` (0.5), `LOCI_OAK_BBOX_SCALE` (0.5), `LOCI_OAK_DEPTH_MIN_MM` (200), `LOCI_OAK_DEPTH_MAX_MM` (6000), `LOCI_OAK_SPATIAL_ALGO` (`median`), `LOCI_OAK_STALL_S` (5), `LOCI_TAG_STREAM` (`1280x800@5`), `LOCI_TAG_UNDISTORT` (0), `DEPTHAI_ZOO_CACHE_PATH` (Pi: `/home/ziggy/loci-handheld/models/zoo`), `DEPTHAI_ZOO_INTERNET_CHECK` (Pi: 0).
Anchoring: `LOCI_TAG_DETECTOR` (`auto`), `LOCI_TAG_SIZE_M` (overrides rooms.json), `LOCI_ROOMS_FILE` (`deploy/handheld/config/rooms.example.json` when present, else built-in default), `LOCI_POSE_HOLD_S` (10), `LOCI_POSE_STALE_S` (60).
Embedding: `LOCI_LABEL_EMBEDDER` (`hash`; Pi env `static`), `LOCI_EMBED_MODEL_DIR` (repo `deploy/handheld/models/potion-base-8M`; Pi `/home/ziggy/loci-handheld/models/embeddings/potion-base-8M`).
Memory: `LOCI_DATA_DIR` (`~/.local/share/loci-handheld`; Pi `/home/ziggy/loci-handheld/data`), `LOCI_PERSIST` (1), `LOCI_SNAPSHOT_INTERVAL_S` (20), `LOCI_JOURNAL_FSYNC_S` (1), `LOCI_MAX_RECORDS` (100000), `LOCI_CONSENSUS_REFIRE_MS` (2000 in env.example; code default 0), `LOCI_CONSENSUS_RADIUS_M` (0.3), `LOCI_DEDUP_RADIUS_M` (0.35), `LOCI_MOVED_THRESHOLD_M` (0.5).
Server: `PORT` (8001), `LOCI_WS_PUSH_HZ` (5), `LOCI_TLS` (0), `CORS_ORIGINS` (existing). Voice (existing): `TTS_ENGINE` (Pi `piper`), `PIPER_MODEL_PATH` (Pi `/home/ziggy/loci-handheld/models/piper/en_US-lessac-medium.onnx`), `WHISPER_LOCAL` (Pi 1), `WHISPER_LOCAL_MODEL` (Pi `tiny.en`), `OPENAI_API_KEY` (unset).

### 4.13 File paths

Repo: `demo_spatial/app/*.py` (modules above), `demo_spatial/static/phone.html`, `deploy/handheld/**` (see §3), `deploy/handheld/models/` (weights git-ignored; `potion-base-8M/{model.safetensors,tokenizer.json,config.json}` and `zoo/`), `tests/test_*.py`.
Pi (`ziggy@loci-pi.local`, absolute paths, never `/tmp`): `/home/ziggy/loci-handheld/src` (rsync target, excludes `.git venv __pycache__ demo_spatial/.certs`), `/home/ziggy/loci-handheld/.venv` (existing; upgraded), `/home/ziggy/loci-handheld/models/{zoo,embeddings/potion-base-8M,piper}`, `/home/ziggy/loci-handheld/data/{memory.snapshot.json,memory.snapshot.bak.json,memory.journal.jsonl,tags.json}`, `/home/ziggy/loci-handheld/env` (EnvironmentFile), `/home/ziggy/loci-handheld/rooms.json`, `/home/ziggy/.config/systemd/user/loci-handheld.service` (user unit) or `/etc/systemd/system/loci-handheld.service` (system unit, sudo). `~/loci-db` on the Pi is never written.

### 4.14 Simulator JSONL (record/replay) line format

`{"t": "det", "timestamp_ms": int, "seq": int, "source": str, "device_ts_ms": int|null, "detections": [{"label","label_id","confidence","xmin","ymin","xmax","ymax","xyz_cam_m": [x,y,z]|null}]}` and `{"t": "tag", "timestamp_ms": int, "width": int, "height": int, "K": [[...]], "dist": [...]|null, "tags": [{"tag_id", "corners_px": [[x,y]x4], "decision_margin", "hamming"}]}`.

## 5. Detection source details

**Pipeline (v3, names introspected on 3.11.0 and live-verified from the Mac):** see `build_pipeline` in §4.3. Device open: `dai.Device()` (or `dai.Device(dai.UsbSpeed.HIGH)` with `LOCI_OAK_USB2=1`), then `dai.Pipeline(device)`; `pipeline.start()`; close pipeline then device in `finally`. Model: `dai.NNArchive(path)` when `LOCI_OAK_MODEL` names an existing `.tar.xz`, else `dai.NNModelDescription(name, platform="RVC2")` resolved from `DEPTHAI_ZOO_CACHE_PATH` (the Pi already holds the archive; `DEPTHAI_ZOO_INTERNET_CHECK=0` keeps demo time offline). Queues: detections `maxSize=4, blocking=False`, tag stream `maxSize=2, blocking=False`, polled with `tryGet()` + 2 ms sleep (the pattern that measured 10.1 fps).

**Per-frame data:** each `SpatialImgDetections` → `DetectionFrame(timestamp_ms=HostClock.to_unix_ms(device_ms, arrival_ms), detections=spatial_to_detections(msg, labels, cfg))`; `spatial_to_detections` maps mm → metres through `depthai_xyz_to_cam(…, cfg.axes)` and returns `xyz_cam_m=None` when `z <= 0` (verified: no-depth boxes are exact zeros; never store 0 m). Each tag-stream `ImgFrame` → `tag_frame_from_img` (luma plane trick with the `getCvFrame` fallback; `K`/`dist` from the frame's `ImgTransformation`).

**Resilience:** supervisor + stall watchdog as in §4.3. States and counters surface in `SourceStatus` → `/api/handheld/status`. `stop()` sets an event; `events()` returns within one poll interval; the runtime joins the thread (5 s). The device is reopened after unplug/replug indefinitely; systemd restarts only cover process death.

**Simulator:** `SimDetectionSource` (§4.2). Geometry: the camera pose per segment is `camera_pose_T(camera_xyz_m, yaw_deg)` (yaw 0 = looking toward −z, i.e. at the anchor wall); objects are transformed into the camera frame, boxes projected with the 640x400 K from a `size_m` cube; tag corners projected with the 1280x800 K (`dist=None`). Everything deterministic under `default_rng(seed)`. `realtime=False` runs as fast as possible (tests); the server uses `LOCI_SIM_REALTIME=1`.

**Replay:** `JsonlRecorder` writes every event the runtime consumed when `LOCI_RECORD_PATH` is set (TagFrames without `gray`); `JsonlReplaySource` plays a recording back, so 60 s of real Pi detections become hardware-free regression data.

## 6. Geometry and anchoring details

Chain per tag frame (5 fps): `TagDetector.detect(gray)` (or `tf.tags`) → for each tag `solve_tag_pose(id, corners, tf.K, tf.dist, tag_size_m)` → `TagFilter.accept` (side ≥ 30 px, reproj ≤ 1 px, ambiguity ≥ 1.5, view ≤ 60°, range ≤ 6 m) → group accepted poses by room via `RoomMap.room_for_tag` (plus tags already registered in a room's `RoomAnchorMap`) → if ≥ 2 known tags of the same room are visible use `solve_room_pose_multi` over all their corners, else `RoomAnchorMap.observe` (anchor = `make_T(I, (0, anchor_height_m, 0))`; co-visible tags register after 10 frames and are persisted to `tags.json`) → `PoseHold.update(T_room_cam, now_s, room)` (room change resets the hold) → `RoomLocalizer.current()` yields the `CameraPose` used for every detection frame until the next tag frame.

Unknown tag ids are logged once and ignored. `RoomSpec.extra_tags` allows manually surveyed poses for tags never co-visible with the anchor. `normalize_upright` is applied when `LOCI_TAG_UPRIGHT_FIX=1` (default 0; only needed if tags were printed with the AprilRobotics canonical orientation).

Localisation of a detection: `pos_room_m = apply_T(T_room_cam, xyz_cam_m)` when `pose.frame == "room"`; the `Detection` carries `room`, `anchored` (age ≤ hold_s) and `anchor_confidence`; after `stale_s` the frame falls to `"camera"` and `scene_key` becomes `unanchored/<label>`.

Accuracy expectations (research Monte-Carlo, 16 cm tag, 1280 px): pose-grade to ~3 m oblique / ~2 m frontal from one tag; two tags cut camera error to 2.5–5 cm median. Tags are mounted at 1.3–1.5 m beside each door facing into the room, ideally a second tag on an adjacent wall.

## 7. LOCI data model, queries and merge

### 7.1 Per observation
`WorldState(x, y, z, timestamp_ms, vector=embedder.embed(label), scene_id=scene_key(label, frame, room), confidence, metadata={...})` with the metadata keys of §4.8. Client: `LocalLociClient(epoch_size_ms, vector_size=embedder.dim, distance="cosine")` (default decay). `scale_level` stays `"patch"`.

### 7.2 Side index
`_scene_index: dict[str, list[_Slot]]` sorted by `timestamp_ms` (bisect insort), `_Slot(state_id, timestamp_ms, confidence, x, y, z, pos_m, xyz_cam_m, cx, cy, anchored, first_seen_ms)`. `_records: dict[state_id, ObservationRecord]` mirrors every live point (source of truth for snapshots). `_latest[label]`, `_latest_by_scene[scene_id]`. All rebuilt from records on restore. Memory cost ≈ 1 KB per point; `LOCI_MAX_RECORDS` bounds it.

### 7.3 Merge (`_try_merge`)
```
slots = newest-first slots of scene_id with ts >= new_ts - dedup_window_ms (max 20)
for s in slots:
    if s.anchored != anchored: continue
    if frame == "image": match = _iou_proxy(cx, cy, s.cx, s.cy) > dedup_iou_threshold
    else: match = dist(pos_or_cam_new, pos_or_cam_slot) <= dedup_radius_m
    if not match: continue
    w0, w1 = s.confidence, confidence; merged metres/cx/cy = (w0*old + w1*new)/(w0+w1); conf = max
    metadata = {..., first_seen_ms: s.first_seen_ms, merge_count: s.merge_count + 1}
    loci_bridge.delete_point(client, s.state_id); persistence.append_delete(s.state_id); remove slot/record
    new_id = client.insert(WorldState(...merged..., timestamp_ms=new_ts))   # Hilbert ids + causal link recomputed
    add slot/record/journal upsert; return new_id
return None
```
This is exactly what the staged `tests/test_spatial_deduplication.py` expects (fresh id on merge; 15 cross-epoch merges collapse to one record; region query at the old position empty; time-window query at the new timestamp hits with `scene_id == "keys"`).

### 7.4 Queries
`where_is` as in §4.8: exact `scene_id` MatchAny + side-index time window, Python sort by `timestamp_ms` desc. Because the window contains ≤ `limit` slots per scene set and the client over-fetches ×3, the argpartition tie truncation can never drop the newest. `objects_in_region` keeps `spatial_bounds`, fetches 200 and dedupes by label newest-first. `history` = `where_is(limit=20)`.

### 7.5 `changes(room, window_seconds)`
```
since = now_ms - window; for each scene_id with room prefix (or all scenes when room is None):
  slots = sorted by ts; latest = slots[-1]; first_seen = min(s.first_seen_ms or s.timestamp_ms)
  if latest.timestamp_ms < since:
      if room presence (note_presence intervals) covers >= 5 s inside [max(since, latest.ts), now]: kind = "gone"   else skip
  elif first_seen >= since: kind = "new"
  else:
      older = [s for s in slots[:-1] if s.timestamp_ms < latest.timestamp_ms]
      if older and (latest.first_seen_ms or latest.ts) >= since and dist(older[-1].pos_m, latest.pos_m) > moved_threshold_m: kind = "moved" (pos_before = older[-1].pos_m, delta)
      else: kind = "seen"
return sorted by kind priority (moved, new, gone, seen) then last_seen desc
```
Presence log: `_presence: dict[room, list[(start_ms, end_ms)]]`, intervals extended when the gap < 5 s; in-memory only.

## 8. Embeddings and label resolution
As in §4.6. `HashLabelEmbedder` must be byte-identical to today's `_label_embedding` (test compares against the inlined legacy function). `StaticLabelEmbedder` is the research's verified loader (bit-identical to model2vec on the Pi, 0.04 s load, 87 MB RSS). Weights are fetched by `deploy/handheld/scripts/fetch_models.py` from HF revision `bf8b056651a2c21b8d2565580b8569da283cab23` with sha256 checks (`model.safetensors f65d0f32…3de2`, `tokenizer.json e67e803f…bb50`, `config.json 2a6ac0e9…7553`) and are git-ignored. On the Pi the directory already exists at `/home/ziggy/loci-handheld/models/embeddings/potion-base-8M`.

`LabelResolver` vocabulary = source classes (COCO80 for oak/sim, `SceneIngestion.classes` for browser mode) ∪ `memory.tracked_labels`, refreshed on every `resolve` call (cheap). Suggestions come from embedding neighbours when static, else from the alias table keys.

## 9. Persistence
As in §4.8/§D14. Write path: `observe()` appends records to a `queue.SimpleQueue`; the `persistence-writer` thread drains it, appends JSON lines, `flush()`+`fsync()` at most once per `journal_fsync_s`, and takes a snapshot when `dirty and now - last_snapshot >= snapshot_interval_s` or on `request_snapshot()`. The snapshot provider (registered in `begin_session`) is `memory.export_records`, which copies `list(_records.values())` under the lock (≈ 10 ms at 1e5 records) and serialises outside it. Shutdown: `memory.close()` → final snapshot. Restore report is shown on the phone page banner ("Restored 143 sightings in 2 rooms, newest 4 min ago") and `ObjectObservation.restored` makes answers say "That was before I restarted." Growth: ≈ 500 B/record; with the 2 s hold-off a static object produces ≤ 0.5 records/s while in view and merges replace rather than add; `LOCI_MAX_RECORDS` prunes the oldest at snapshot/restore. Test isolation: `LOCI_PERSIST=0` or `LOCI_DATA_DIR=<tmp_path>`.

## 10. Queries, answers, phone UI
Endpoints and WS as in §4.11. Latency budget for a typed `/api/query`: parse µs + resolver 0.2 ms + `where_is` ≤ 7 ms + phrasing → well under 300 ms; TTS only when `tts: true`.

`phone.html` (vanilla JS, no CDN, works over plain HTTP on LAN/hotspot, dark/light, 48 px targets, 16 px gutters): (1) status strip — room name large, anchor dot (green anchored / amber held / grey camera-only), source state, fps, sighting count, restore banner; (2) ask box + Send, quick chips ("Where is my mug?", "What changed in the kitchen?", "What do you see?"), answer rendered into `role="status" aria-live="assertive"` so VoiceOver/TalkBack read it, a "Speak" button (browser `speechSynthesis`; if the response carries `audio_b64` it plays that instead with `audio_mime`), a mic button only when `window.isSecureContext && MediaRecorder` and `/health.pipeline.stt_available`; (3) "seeing now" strip from WS `detections` (label + `distance_m`); (4) per-room SVG top-down map (x across, z up, tag mark at bottom centre, landmarks, object dots solid when anchored, hollow otherwise, camera arrow from `anchor`); (5) recent changes list from `/api/objects/changes`. WebSocket auto-reconnects with backoff. It consumes only `/api/query`, `/api/rooms`, `/api/anchor`, `/api/objects`, `/api/objects/changes`, `/api/handheld/status`, `/api/voice/query`, `/ws`.

## 11. Pi deployment (`deploy/handheld/`)
- `requirements-pi.txt`: `depthai==3.11.0`, `opencv-python-headless==4.13.0.92`, `numpy>=2.2,<3`, `pyapriltags==3.4.3.1`, `tokenizers==0.23.3`, `safetensors==0.8.0`, `fastapi>=0.115,<1`, `uvicorn[standard]>=0.30,<1`, `pydantic>=2.7,<3`, `python-multipart`, `segno`, `httpx`, `qdrant-client>=1.7`, `hilbertcurve>=2.0.5`, `pytest`; optional block (commented): `piper-tts>=1.3`, `faster-whisper`, `model2vec==0.10.0`. Then `pip install -e /home/ziggy/loci-handheld/src --no-deps`. `requirements-dev.txt` (Mac scratch venv): the same minus depthai pin optionality (`depthai==3.11.0` also installs on macOS arm64).
- `install_pi.sh` (run as ziggy, no sudo): prints the one mandatory sudo block (`usb_max_current_enable=1` in `/boot/firmware/config.txt` + reboot) and the optional ones (`loginctl enable-linger ziggy`, system unit, hotspot); verifies `/etc/udev/rules.d/80-movidius.rules`; creates or upgrades `~/loci-handheld/.venv` (depthai 3.5.0 → 3.11.0); installs pins + editable repo; copies the Piper model from `~/loci-db/deploy/oak-rpi5/models/en_US-lessac-medium.onnx` (+ `.json`) into `~/loci-handheld/models/piper/` if present; runs `fetch_models.py` only for files that are missing (zoo archive and potion files already exist on the Pi); writes `~/loci-handheld/env` from `env.example` and `~/loci-handheld/rooms.json` from the example if absent; creates `data/`; runs `python -c "import depthai, cv2, pyapriltags, tokenizers, safetensors, loci, demo_spatial.app.main"` as a self-test; installs the user unit into `~/.config/systemd/user/` when `systemctl --user` works.
- `run_pi.sh`: `set -a; source /home/ziggy/loci-handheld/env; set +a; cd /home/ziggy/loci-handheld/src; exec .venv/bin/uvicorn demo_spatial.app.main:app --host 0.0.0.0 --port ${PORT:-8001} --workers 1` (+ `--ssl-keyfile/--ssl-certfile` generated under `~/loci-handheld/certs` when `LOCI_TLS=1`). **Never `--reload`.**
- `loci-handheld.service` (system): `User=ziggy`, `WorkingDirectory=/home/ziggy/loci-handheld/src`, `EnvironmentFile=/home/ziggy/loci-handheld/env`, `ExecStart=/home/ziggy/loci-handheld/src/deploy/handheld/run_pi.sh`, `Restart=on-failure`, `RestartSec=5`, `TimeoutStopSec=20`, `KillMode=mixed`, `After=network-online.target`, `SupplementaryGroups=plugdev video`. `loci-handheld.user.service`: same without `User=`, `WantedBy=default.target`. All paths absolute (systemd does not expand `~`).
- `deploy_from_mac.sh [--no-restart] [--logs]`: `rsync -az --delete --exclude .git --exclude venv --exclude __pycache__ --exclude demo_spatial/.certs --exclude deploy/handheld/models/ ./ ziggy@loci-pi.local:/home/ziggy/loci-handheld/src/`, then `ssh ziggy@loci-pi.local 'systemctl --user restart loci-handheld 2>/dev/null || (pkill -f "uvicorn demo_spatial.app.main" ; nohup /home/ziggy/loci-handheld/src/deploy/handheld/run_pi.sh >/home/ziggy/loci-handheld/data/server.log 2>&1 &)'`, optional `journalctl --user -u loci-handheld -f`.
- `smoke_test.sh`: the step-2 checklist of §14 as a script (exit non-zero on the first failure).
- `env.example`: the Pi values of §4.12. `README.md`: bring-up, sudo steps, cable/port, models, tags printing/mounting, hotspot, service, troubleshooting (X_LINK errors, shave budget, 600 mA cap), demo story script.
- `scripts/make_tag_sheet.py --ids 0 1 2 3 --size-mm 160 --out deploy/handheld/tags/`: the verified recipe (cv2.aruco `generateImageMarker`, 8 cells × 236 px = 159.85 mm at 300 dpi, 2 cm quiet zone, PNG with dpi + PDF, id caption + "this way up").

## 12. Testing
Gate on the Mac: `/opt/homebrew/bin/pytest tests/ -q` must pass (after the user resolves the merge) with everything hardware-adjacent skipping; `ruff check demo_spatial/ tests/` clean. Scratch dev venv (3.11, `requirements-dev.txt`) runs the cv2/pyapriltags/tokenizers/depthai-attribute tests too. CI (3.11/3.12, no cv2) runs the numpy-only, sim-oracle and API tests. Test files per task are listed in §13; all import via `sys.path.insert(0, "<repo>/demo_spatial")` like the existing demo tests and must not change the four existing demo tests.

## 13. Implementation tasks (parallel agents, same working tree, disjoint files)

Rules for every task: add new files or edit only the named demo_spatial files; never touch `loci/`, `demo_spatial/requirements.txt`, `index.html`, `scanner.html` or any other modified file; never commit; keep `/opt/homebrew/bin/pytest tests/ -q` and `ruff check demo_spatial/ tests/` green; code against §4 for anything owned by another task (stub locally in tests with `MagicMock`/fakes when the other module does not exist yet).

| ID | Title | Files owned (exact) | Depends on | HW |
|---|---|---|---|---|
| T1 | Step-1 console script | `deploy/handheld/scripts/oak_console.py`, `tests/test_oak_console.py` | T2 | run on Pi; tests no |
| T2 | Detection source + simulator + OAK source | `demo_spatial/app/detection_source.py`, `demo_spatial/app/oak_source.py`, `tests/test_detection_source.py`, `tests/test_oak_source_offline.py` | — | no |
| T3 | Geometry + AprilTag room anchoring | `demo_spatial/app/geometry.py`, `demo_spatial/app/room_anchor.py`, `deploy/handheld/config/rooms.example.json`, `tests/test_geometry.py`, `tests/test_tag_pose.py`, `tests/test_room_anchor.py` | T2 (dataclasses) | no |
| T4 | SpatialMemory: room frame, data model, embedder hook, merge fix, persistence | `demo_spatial/app/spatial_memory.py`, `demo_spatial/app/loci_bridge.py`, `demo_spatial/app/persistence.py`, `tests/test_spatial_memory_rooms.py`, `tests/test_persistence.py` | T6 (embedder/resolver API), T3 (RoomMap/room_to_unit), T10 (phrasing.describe_position, lazy import) | no |
| T5 | SceneIngestion: metric detections entry point + metric consensus | `demo_spatial/app/scene_ingestion.py`, `tests/test_scene_ingestion_detections.py` | — (observe keyword contract) | no |
| T6 | Label embedder + resolver | `demo_spatial/app/label_embedding.py`, `demo_spatial/app/label_resolver.py`, `tests/test_label_embedding.py`, `tests/test_label_resolver.py` | — | no |
| T7 | FastAPI wiring, runtime thread, phone page | `demo_spatial/app/handheld.py`, `demo_spatial/app/main.py`, `demo_spatial/static/phone.html`, `tests/test_handheld_api.py`, `tests/test_handheld_runtime.py` | T2, T3, T4, T5, T6, T10 | no |
| T8 | Pi deployment scripts + README | `deploy/handheld/README.md`, `deploy/handheld/requirements-pi.txt`, `deploy/handheld/requirements-dev.txt`, `deploy/handheld/env.example`, `deploy/handheld/install_pi.sh`, `deploy/handheld/run_pi.sh`, `deploy/handheld/deploy_from_mac.sh`, `deploy/handheld/smoke_test.sh`, `deploy/handheld/loci-handheld.service`, `deploy/handheld/loci-handheld.user.service`, `deploy/handheld/scripts/fetch_models.py`, `deploy/handheld/scripts/make_tag_sheet.py`, `deploy/handheld/models/.gitignore`, `deploy/handheld/models/README.md`, `tests/test_make_tag_sheet.py`, `tests/test_deploy_scripts.py` | — (env contract) | no |
| T9 | Hardware verification plan (document) | `deploy/handheld/HARDWARE_VERIFICATION.md` | T1, T7, T8 (names only) | plan only |
| T10 | Phrasing + voice pipeline room answers | `demo_spatial/app/phrasing.py`, `demo_spatial/app/voice_pipeline.py`, `tests/test_phrasing.py`, `tests/test_voice_room_answers.py` | T4 (memory API), T6 (Resolution), T2 (CameraPose) | no |

### T1 — Step-1 console
Public API: CLI of §4.3 (`oak_console.py`). Implementation: `argparse`; `--sim` builds `SimDetectionSource(scenario, realtime=True)`; otherwise `OakDetectionSource(OakConfig.from_env() overridden by flags)`; iterates `events()`, prints detections at ≤ `--rate` Hz, 1 s summaries from `source.status()`, `--record` wraps a `JsonlRecorder`, `--check-y-sign` accumulates `(cy, y_m)` pairs and prints the verdict; SIGINT stops cleanly (`source.stop()`); exit codes as specified. No `loci` import anywhere on its import path. Tests (`tests/test_oak_console.py`): `--sim --seconds 2` runs via `runpy`/subprocess and prints ≥ 1 detection line and a summary; `--check-y-sign` verdict function with synthetic pairs (RDF vs flipped); `importorskip("depthai")` test that `--seconds 0` constructs the OAK source without opening a device (opener injected).

### T2 — Detection source + simulator + OAK source
Public API: §4.2 and §4.3 verbatim. Tests: `test_detection_source.py` — `depthai_xyz_to_cam` (rdf identity, ruf flips y, z≤0 → None); `HostClock` offset/min-latency/fallback counter; `SimDetectionSource` determinism (same seed ⇒ identical events), timeline (frame counts per segment, tags only in tag segments, `oracle_pose` frames carry room/T, hallway frame "camera"), projected boxes inside [0,1], `JsonlRecorder`→`JsonlReplaySource` round trip equality, `make_detection_source` dispatch from an env mapping. `test_oak_source_offline.py` — `OakConfig.from_env` parsing incl. `parse_tag_stream("640x400@10")`/`"off"`; `build_pipeline` against a **recording fake `dai` module** (asserts `PresetMode.FAST_DENSITY`, `setDepthAlign(CAM_A)`, `setPostProcessingHardwareResources(2,2)`, no LRC/extended calls by default and present when enabled, thresholds, `requestOutput((1280,800), NV12, fps=5)` without `enableUndistortion` unless `tag_undistort`); `spatial_to_detections` with fake messages (zeros dropped, axes applied, label index→name); `tag_frame_from_img` luma-plane and fallback branches; supervisor state machine with an injected opener that fails twice then yields frames (reconnects == 2, states observed, `stop()` joins); `importorskip("depthai")` attribute check that every `dai.*` name used by `build_pipeline` resolves on 3.11.0.

### T3 — Geometry + room anchoring
Public API: §4.4, §4.5, and `rooms.example.json` (§4.5 schema; kitchen tag 0 with a `table` landmark at (1.0, 0.75, 1.2); living_room tag 1). Tests: `test_geometry.py` (numpy only) — `invert_T`/`apply_T` identities, `camera_pose_T` ↔ `yaw_deg_of` round trip, room chain with known poses (camera at (0,1.4,2.5) yaw 0 ⇒ camera point (0,0,1) lands at room (0,1.4,1.5); point (0.5,0,1) lands at x=+0.5 — the viewer's right when facing the tag), `mean_T`, `rotation_angle_deg`, `project_points`. `test_tag_pose.py` (`importorskip("cv2")`) — frontal tag ⇒ `R=diag(1,-1,-1)`; exact solve for random poses and the pitch-0 degenerate wall-tag cases; pyapriltags remap when importable; `solve_room_pose_multi` with two walls beats single-tag error; `normalize_upright`; a source-level guard that `geometry.py` never calls `solvePnP` with `SOLVEPNP_IPPE_SQUARE` outside `solve_tag_pose`. `test_room_anchor.py` (numpy + fake `TagDetector`/pre-detected `TagFrame.tags`) — `TagFilter` thresholds; `RoomAnchorMap` registration after 10 co-visible frames and `to_dict/from_dict`; `PoseHold` EMA, jump gating, `anchored=False` after 10 s, `frame="camera"` after 60 s; room switch resets; `RoomLocalizer.update/current/to_room` with projected corners when cv2 exists (object localised within 5 cm) and with `set_oracle` otherwise; `room_to_unit` clipping flag and `unit_to_room` inverse; `RoomMap.load` of the example file; `tags.json` save/load round trip in `tmp_path`.

### T4 — SpatialMemory + persistence
Public API: §4.8 verbatim (plus `loci_bridge`). Must keep the four existing demo tests green **unchanged**, including the staged `test_merge_recomputes_spatial_index_across_epochs` (`_label_embedding` export, fresh id on merge, cross-epoch collapse, `scene_id == "keys"` for room-less observations). Tests: `test_spatial_memory_rooms.py` — scene keys for image/camera/room frames; metadata round-trip; `where_is` across rooms and with `room=`; newest-first with > 2 h-old timestamps and with > 20 points of one label (argpartition regression); keyed merge works with `client.query` monkeypatched to raise; metric merge within 0.35 m, no merge across anchored/unanchored or across rooms; merged record keeps `first_seen_ms`, increments `merge_count`, old id gone from a region query, new id found; `changes()` new/moved/seen/gone with a presence log; `objects_in_region(room=)`; `resolve_label` uses tracked labels (keys resolves exact after a browser-mode observe); `stats()` keys; observe from a thread while querying (no exception). `test_persistence.py` — observe → `flush()` → new `SpatialMemory` on the same dir → identical `where_is` results (positions, timestamps, metadata) and `restored=True`; journal overlay (upsert after snapshot, delete op after merge) applied on restore; torn last line tolerated and counted; corrupt snapshot quarantined and `.bak` used; `max_records` prune; `begin_session` truncates the journal and the snapshot is current before the first post-restore write; atomic write leaves no `.tmp`; writer thread stops on `close()`; embedder name mismatch simply re-embeds.

### T5 — SceneIngestion
Public API: §4.7 verbatim. Existing `tests/test_temporal_consensus.py` must pass unchanged (note: consumption means a third frame starts a new buffer; the existing assertions use ≤ 2 or exactly 3 frames and `assert_called_once`, which hold). Tests: `test_scene_ingestion_detections.py` — `process_detections` with room-frame `Detection`s calls `observe` once per `min_count` frames with averaged `pos_room_m`, `anchored=all`, mean `anchor_confidence`, `frame`/`room` forwarded; metric match within 0.3 m while image boxes drift (and no match beyond); image-frame legacy path unchanged; camera-frame path; buffer keys isolate frames/rooms; `consensus_refire_ms=2000` suppresses fires and `consensus_stats` counts them; eviction by window; `Detection.to_dict` includes the new keys only when set; legacy keyword call shape still accepted.

### T6 — Embedder + resolver
Public API: §4.6 verbatim. Tests: `test_label_embedding.py` — `hash_label_embedding("cup")` equals the legacy function (inline the old implementation in the test), dim 128, L2 norm 1, `make_embedder()` default is hash with an empty env, `"static"` raises without a model, `"auto"` falls back; `importorskip`-style skip of the semantic test unless `static_model_available()` (cup/mug > 0.40 > cup/dog; keys/key chain > 0.45 > keys/dog; phone/cell phone > 0.70; bottle/water bottle > 0.80; dim 256); model2vec parity test skipped unless `model2vec` importable. `test_label_resolver.py` — exact, alias (mug→cup), plural/“my” stripping, denylist (keys never → keyboard), threshold with a fake static embedder, suggestions, vocabulary union behaviour, hash embedder ⇒ no embedding step.

### T7 — FastAPI wiring + runtime + phone page
Public API: §4.11 verbatim. Keep every existing endpoint and WS message shape; add `_broadcast`. Tests: `test_handheld_runtime.py` — `HandheldRuntime.step` with sim events (oracle pose) fills `Detection.pos_room_m/room/frame`, calls `ingestion.process_detections`, notes presence ≤ 1 Hz, `run_until_done` returns the frame count, `status()` shape, `stop()` joins a started thread. `test_handheld_api.py` — env (`LOCI_DETECTION_SOURCE=sim`, `LOCI_SIM_REALTIME=0`, `LOCI_SIM_ORACLE_POSE=1`, `LOCI_DATA_DIR=<tmp>`, `LOCI_LABEL_EMBEDDER=hash`) set **before** importing `app.main` (use `importlib.reload` per test module and `monkeypatch.setenv`); with `TestClient(app)` lifespan, `runtime.run_until_done()` then: `/api/anchor` reports kitchen; `/api/objects?room=kitchen` lists `cup` and `bottle` with `x_m` etc.; `POST /api/query {"text":"where is my mug"}` → `resolution.label == "cup"`, answer mentions "kitchen", position within 0.3 m of (−1.2, 0.8, 1.5); `"what changed in the kitchen"` → a `moved` change for cup; `"where are my keys"` → unknown-label answer, no 500; `/api/objects/changes?room=kitchen`; `/api/rooms`; `/api/handheld/status`; `/phone` 200 with `aria-live`; WS first message is a `snapshot`; `/api/voice/text-query` still returns `audio_b64` key (null without TTS backend) and `resolution`; `/api/voice/tts` MIME sniff via a monkeypatched TTS returning `b"RIFF..."`; persistence across a simulated restart (reload `app.main` with the same `LOCI_DATA_DIR` and `LOCI_DETECTION_SOURCE=none`, then `/api/objects` still lists the cup and `/api/handheld/status.restore.records > 0`); legacy `/api/ingest/frame` path still answers (YOLO absent ⇒ empty detections).

### T8 — Deployment
Files per §11. Tests: `tests/test_make_tag_sheet.py` (`importorskip("cv2")`, `importorskip("PIL")`): generates ids 0–1 into `tmp_path`, PNG is 2360 px wide with 300 dpi metadata and a black square of 1888 px; `tests/test_deploy_scripts.py`: `bash -n` on every `.sh`, `env.example` parses as KEY=VALUE and contains every `LOCI_*` key of §4.12, both unit files contain absolute paths only (no `~`) and `Restart=on-failure`, `fetch_models.py --help` runs, `requirements-pi.txt` pins the exact versions of §11.

### T9 — Hardware verification plan
Document only: expand §14 into `deploy/handheld/HARDWARE_VERIFICATION.md` with a checkbox per item, expected output snippets, and the fallback to take when an item fails (e.g. tag stream ladder, `LOCI_OAK_AXES=ruf`, powered hub).

### T10 — Phrasing + voice
Public API: §4.9, §4.10 verbatim. Tests: `test_phrasing.py` — perspective table (x=+1 ⇒ "to your left" walk-in, "to your right" facing), straight-ahead band, height bands, egocentric phrase with a known pose (object 3 m ahead and 1 m to the camera's left ⇒ "3 m ahead of you and 1 m to your left"), landmark within/outside 1.2 m, suffixes (unanchored, restored), `changes_answer` wording for each kind and the empty case, `unknown_label_answer` with/without suggestion. `test_voice_room_answers.py` — `parse_intent` room capture for "what changed in the kitchen", "where is my mug in the living room", "what's in the kitchen"; `build_response_text` with a `MagicMock` memory: alias disclosure, room answer, camera-frame answer, unknown noun, history with rooms; `handle_text_query(synthesize_response=False)` returns no audio; `audio_mime` sniff; `_run_piper_sync` uses `synthesize_wav` when present (fake voice object).

Suggested order: T2, T3, T5, T6 in parallel → T4 and T10 → T7 → T1/T8/T9 any time (T1 after T2).

## 14. Hardware verification plan (steps 1–5, exact commands; T9 expands this)

All commands run on the Pi as `ziggy` unless prefixed `mac$`. Shell on the Pi: `ssh ziggy@loci-pi.local`. Venv: `source /home/ziggy/loci-handheld/.venv/bin/activate`. Env: `set -a; source /home/ziggy/loci-handheld/env; set +a`.

**Step 0 — preconditions (once)**
1. Camera on a **blue USB-A** port with a data-capable cable (USB-C is power-only): `lsusb | grep -i 03e7` lists `03e7:2485` (unbooted) — if nothing appears, swap the cable first.
2. `vcgencmd get_config usb_max_current_enable` → `usb_max_current_enable=1`. If `0`: `sudo sh -c 'echo usb_max_current_enable=1 >> /boot/firmware/config.txt' && sudo reboot` (the one mandatory sudo step), then re-check and `od -An -tx4 /proc/device-tree/chosen/power/max_current` → `00001388`.
3. `vcgencmd get_throttled` → `throttled=0x0`; `vcgencmd measure_temp` < 70 °C.
4. `mac$ deploy/handheld/deploy_from_mac.sh --no-restart` then on the Pi `bash /home/ziggy/loci-handheld/src/deploy/handheld/install_pi.sh` (upgrades depthai to 3.11.0, self-test prints OK).

**Step 1 — console (build order 1)**
1. `python /home/ziggy/loci-handheld/src/deploy/handheld/scripts/oak_console.py --sim --seconds 5` prints scripted detections (software sanity, no camera).
2. `python .../oak_console.py --seconds 30` → `state=running`, `fps≥8`, labels with `x= y= z=`; note `usb=` (HIGH is fine). If it fails to boot with the default preset, record the error text; try `LOCI_OAK_PP_SHAVES=3`, then `LOCI_TAG_STREAM=off`, then `--usb2`.
3. Geometry: cup on a tape at 1.00 m and 2.00 m on the optical axis → `z` within ±5 %; cup moved to the camera's right → `x > 0`; `python .../oak_console.py --seconds 20 --check-y-sign` while holding the cup above the lens centre → `RDF confirmed`; otherwise set `LOCI_OAK_AXES=ruf` in `env` and repeat. Boxes without depth must print no XYZ.
4. Optional flags: `LOCI_OAK_STEREO_LRC=1`, then `LOCI_OAK_STEREO_EXTENDED=1` — keep whichever still boots (MinZ improves from ~40 cm to ~20 cm with extended).
5. Resilience: unplug the camera for 10 s mid-run and replug → summary shows `state=waiting_for_camera` then `running`, `reconnects=1`, process alive. `top -H -p $(pgrep -f oak_console)` → runtime thread < 25 % of one core.
6. `python .../oak_console.py --seconds 60 --record /home/ziggy/loci-handheld/data/step1.jsonl`; `mac$ scp` it into the scratchpad for replay tests.

**Step 2 — adapter + persistence (build order 2)**
1. `systemctl --user restart loci-handheld` (or `run_pi.sh`); `curl -s localhost:8001/health`; `curl -s localhost:8001/api/handheld/status | python -m json.tool` → `runtime.source.state == "running"`, `fps ≥ 8`.
2. Place a cup on a table 1–2 m away for 5 s: `curl -s 'localhost:8001/api/objects'` lists `cup` with `frame: "camera"`, `xyz_cam_m`, `anchored: false`; `observation_count` grows ≤ 1 per 2 s per static object (`consensus_refire_ms=2000`).
3. `curl -s -X POST localhost:8001/api/query -H 'content-type: application/json' -d '{"text":"where is my mug"}'` → `resolution.label == "cup"`, `latency_ms < 300`, answer says "in front of the camera … roughly".
4. `systemctl --user restart loci-handheld`; after 10 s `curl -s localhost:8001/api/handheld/status` → `restore.records > 0`; the same query answers with "before I restarted". `sudo reboot` (or power-cycle) and repeat. `kill -9 $(pgrep -f 'uvicorn demo_spatial')` mid-walk → after restart at most ~1 s of sightings missing (journal).

**Step 3 — AprilTag anchoring (build order 3)**
1. `mac$ python deploy/handheld/scripts/make_tag_sheet.py --ids 0 1 2 3`; print at 100 %, measure the black square with a ruler, put the value in `rooms.json` `tag_size_m`; mount tag 0 (kitchen) and tag 1 (living room) at 1.4 m beside each door facing into the room, matte and flat; optionally tag 2 on an adjacent kitchen wall.
2. Stand 2.00 m in front of tag 0 (tape): `curl -s localhost:8001/api/anchor` → `room: "kitchen"`, `anchored: true`, `anchor_confidence ≥ 0.8`, `camera_xyz_m[2] ≈ 2.00 ± 0.06`, `camera_xyz_m[1] ≈ eye height`. If z is > 3 % off, note the ratio (EEPROM intrinsics check) and report it; `LOCI_TAG_FX_SCALE` is **not** a feature — fall back to `LOCI_TAG_UNDISTORT=1` and re-measure.
3. Mark a floor point 1.0 m to the LEFT of the tag when facing it (room x = −1.0) and 2.0 m out (z = 2.0); place the cup there at table height: `curl -s localhost:8001/api/objects/cup/location` → `x_m ≈ −1.0 ± 0.15`, `y_m ≈ 0.7–0.8`, `z_m ≈ 2.0 ± 0.15`, `anchored: true`.
4. Look away from the tag for 15 s → `/api/anchor` `anchored: false`, `anchor_confidence: 0`, still `frame: "room"`; after 60 s `frame: "camera"`; look back → re-anchors within 1 s.
5. With tags 0 and 2 both in view for ≥ 2 s, `/api/rooms` shows tag 2 under `tags_registered` and `/home/ziggy/loci-handheld/data/tags.json` gains it.
6. Walk kitchen → living room → kitchen; `/api/anchor.room` switches at each door tag; cup position before/after the loop differs by < 0.3 m.
7. If the tag stream failed to start in step 1: `LOCI_TAG_STREAM=640x400@10` and 24 cm tags; re-measure the usable range (expect ~1.5 m oblique).

**Step 4 — phone UI (build order 4)**
1. Phone on the same Wi-Fi: open `http://loci-pi.local:8001/phone` (Android: use the IP printed by `curl -s localhost:8001/api/local-ip`). Status strip shows the room and a green dot while a tag is in view; "seeing now" updates within 1 s.
2. Type "Where is my mug?" → kitchen answer in < 1 s; VoiceOver/TalkBack reads it (aria-live). Tap "Speak" → browser TTS.
3. Move the cup ≥ 1.5 m, wait 5 s, ask again → new position; "What changed in the kitchen?" → "the cup moved about … m". Carry the cup out of the kitchen for 10 s while anchored, ask again → "I no longer see the cup".
4. Restart the service with the page open → WebSocket reconnects and the restore banner appears.
5. Optional voice: `LOCI_TLS=1` in `env`, restart, open `https://…:8001/phone`, accept the certificate, hold the mic button, ask; with `TTS_ENGINE=piper` the answer plays (`audio_mime: audio/wav`).
6. Optional hotspot: `sudo nmcli device wifi hotspot ifname wlan0 ssid loci-pi password <pw>`; phone opens `http://10.42.0.1:8001/phone`.

**Step 5 — battery walkaround (build order 5)**
1. On the power bank: `vcgencmd get_config usb_max_current_enable` still `1`, `get_throttled` `0x0`; if the bank negotiates < 5 A and the camera drops, use a powered hub / OAK Y-adapter.
2. 20-minute two-room walk performing the story twice (put the cup down, walk away, ask, move it, ask, power-cycle, ask). Watch `journalctl --user -u loci-handheld -f` for `X_LINK` errors and tracebacks (expect none), `/api/handheld/status` for `reconnects`, anchored ratio (`memory.frames.room / total`), and `measure_temp`.
3. Tune in `env` and `rooms.json`: `LOCI_CONSENSUS_REFIRE_MS` (1000–3000), `consensus_min_count` (2→3 if false positives), `LOCI_CONSENSUS_RADIUS_M`, `LOCI_DEDUP_RADIUS_M` (0.25–0.5), `LOCI_POSE_HOLD_S` (5–15), per-label `label_confidence_overrides`; target: static cup ⇒ one record per 2 s window, moved cup reported moved within 3 s, position error < 0.5 m at ≤ 3 m.
4. `sudo reboot` → service auto-starts (user unit needs `loginctl enable-linger ziggy` once), memory restored, phone answers within 60 s of boot.

## 15. Risks

1. **Merge precondition**: nothing imports until the user resolves the `origin/main` merge; if the resolution diverges from `13330e2`'s `LocalLociClient` (e.g. keeps per-epoch collections), `loci_bridge.py` and the staged dedup test must be revisited first.
2. **Second RGB output (1280x800 NV12 @5) unverified on the Lite** next to the 8-shave blob; ladder `640x400@10` (24 cm tags) → `off` (camera-frame only demo still works).
3. **Y sign** of RVC2 firmware spatial coordinates under v3 is inferred, not proven: one constant + the step-1 check.
4. **Camera not enumerating on the Pi** (cable/port/600 mA cap): the sim source keeps all software work unblocked; a 3 A power bank re-triggers the cap in step 5 (powered hub as plan B).
5. **No odometry**: a held pose while walking can place objects metres off; mitigated by the 10 s hold, `stale_s`, honest `anchored`/`anchor_confidence`, more tags per room; the BMI270 is a later upgrade.
6. **Single frontal-tag planar ambiguity** (1–3 m outliers): oblique mounting, `TagFilter`, 3-frame jump confirmation, two tags per room with the joint solve.
7. **Rolling-shutter motion blur** of the IMX214 while walking may starve tag detection; slow down near tags; mono global-shutter cameras are the documented fallback (needs CAM_B→CAM_A extrinsics).
8. **COCO vocabulary**: no keys/mug/wallet/glasses; story uses a cup and the refusal path.
9. **Depth jitter** beyond ~4 m (±1 m at 6 m) and MinZ ~40 cm without extended disparity; thresholds 0.2–6 m and consensus averaging; tune in step 5.
10. **Consumption changes browser-mode cadence** (fewer, merged observations); existing tests pass, desktop page responsiveness to be eyeballed.
11. **Private client surface** (`_data_collection`) isolated in `loci_bridge.py`; expect one-line breakage if loci renames it.
12. **RLock + MemoryStore**: all access goes through `SpatialMemory`; snapshot copies under the lock in ~10 ms at 1e5 records; journal/snapshot IO is off the camera thread.
13. **Voice stack on cp313/aarch64** (ctranslate2, piper-tts wheels exist; piper ≥1.3 API renamed) is optional; typed path is guaranteed; phone mic needs HTTPS.
14. **Persistence edge**: a crash between `restore()` and `begin_session()` is impossible by construction (synchronous), but a crash during the first snapshot write leaves the previous snapshot + `.bak`; journal lines are never applied against a snapshot they were not written for because the journal is truncated inside `begin_session`.
15. **Effort vs. "this week"**: ~10 focused days; the 6-day cut drops voice, `gone`, landmarks and the map.

## 16. Effort
Phase 0 (T6 embedder/resolver + T4 merge/index) 1.5 d; step 1 (T2 + T1) 1.5 d; step 2 (T5 + T4 persistence + T7 runtime) 2 d; step 3 (T3 + hardware tuning) 2.5 d; step 4 (T10 + T7 endpoints/page) 1.5 d; step 5 + T8/T9 1 d → **10 days**.
