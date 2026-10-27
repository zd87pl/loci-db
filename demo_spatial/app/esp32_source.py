"""ESP32 thin-client frame receiver.

A camera board (an ESP32-S3 with an OV2640, say) has no GPU and no depth sensor. It
streams JPEG frames to this host over HTTP; the host runs YOLO, turns the boxes into the
demo's :class:`~app.scene_ingestion.Detection` objects and feeds :class:`SceneIngestion`
exactly like the browser-camera path, so the objects list, the queries and the spoken
answers are shared with every other source. Answers travel back the other way as raw PCM
through a per-device audio queue that the board drains into its speaker.

Layout
------
* :class:`Esp32Config`          — environment-driven settings (rate cap, model, rotation).
* :class:`HostYoloDetector`     — lazy ultralytics YOLO → ``list[Detection]`` (``source="esp32"``).
* :class:`Esp32Hub`             — per-device state, host-side rate limiting, the
  detection → ingestion hand-off, and the per-device audio queue.
* :class:`Esp32DetectionSource` — adapter exposing the hub as a handheld ``DetectionSource``
  so the same board can later drive the handheld runtime (``detection_source`` is imported
  lazily, so this module loads on branches that do not have it yet).

Observations are stored in the **image frame** (``cx``/``cy`` normalised, depth ``None``)
like the browser demo. A monocular depth estimate, when a ``depth_estimator`` is plugged
in, is attached as ``depth_m`` but carries no confidence of its own: downstream consumers
must treat it as a hint, never as a measurement.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .scene_ingestion import Detection

logger = logging.getLogger(__name__)

JPEG_MAGIC = b"\xff\xd8"
DEFAULT_DEVICE = "esp32"

Detector = Callable[[bytes], list[Detection]]
DepthEstimator = Callable[[bytes, list[Detection]], list[float | None]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Esp32Config:
    """Settings for the ESP32 path; every field has an ``LOCI_ESP32_*`` variable."""

    max_fps: float = 5.0  # LOCI_ESP32_MAX_FPS: host-side cap, faster frames are dropped
    min_confidence: float = 0.40  # LOCI_ESP32_MIN_CONF: YOLO threshold for the small camera
    rotate_deg: int = 0  # LOCI_ESP32_ROTATE: 0/90/180/270 clockwise, applied host-side
    yolo_model: str = "yolov8n.pt"  # LOCI_ESP32_YOLO_MODEL, else YOLO_MODEL, else yolov8n.pt
    audio_rate: int = 16000  # LOCI_ESP32_AUDIO_RATE: PCM rate the board plays/records
    audio_gain: float = 1.0  # LOCI_ESP32_AUDIO_GAIN: linear gain applied before queueing
    audio_queue_max: int = 8  # oldest clip is dropped when a device falls this far behind
    stale_after_s: float = 10.0  # device reported "offline" after this many silent seconds
    dataset_dir: str = (
        ""  # LOCI_ESP32_DATASET_DIR: save every Nth frame + detections here ("" = off)
    )
    dataset_every: int = 10  # LOCI_ESP32_DATASET_EVERY
    default_device: str = DEFAULT_DEVICE  # LOCI_ESP32_DEVICE: id used when the board sends none
    warmup: bool = False  # LOCI_ESP32_WARMUP=1: load YOLO in a thread at install time
    depth: str = "none"  # LOCI_ESP32_DEPTH: "none" | "mono" (needs a plugged-in estimator)
    white_balance: str = "none"  # LOCI_ESP32_WB: "none" | "gray" (gray-world before YOLO)
    imgsz: int = 640  # LOCI_ESP32_IMGSZ: YOLO inference size; 960 pays off for 1280x960 frames
    # LOCI_ESP32_MAX_UPLOAD: 413 above this; a P4 frame is 35-400 kB, 8 s of speech 256 kB
    max_upload_bytes: int = 1_500_000
    # LOCI_ESP32_BODY_DEADLINE_S: 408 when an upload body takes longer (a stalled board
    # otherwise holds its socket until TCP keepalive gives up)
    body_deadline_s: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Esp32Config:
        env = os.environ if env is None else env
        rotate = _env_int(env, "LOCI_ESP32_ROTATE", 0) % 360
        if rotate not in (0, 90, 180, 270):
            logger.warning("LOCI_ESP32_ROTATE must be 0/90/180/270, got %s; using 0", rotate)
            rotate = 0
        return cls(
            max_fps=_env_float(env, "LOCI_ESP32_MAX_FPS", 5.0),
            min_confidence=_env_float(env, "LOCI_ESP32_MIN_CONF", 0.40),
            rotate_deg=rotate,
            yolo_model=env.get("LOCI_ESP32_YOLO_MODEL") or env.get("YOLO_MODEL") or "yolov8n.pt",
            audio_rate=_env_int(env, "LOCI_ESP32_AUDIO_RATE", 16000),
            audio_gain=_env_float(env, "LOCI_ESP32_AUDIO_GAIN", 1.0),
            audio_queue_max=_env_int(env, "LOCI_ESP32_AUDIO_QUEUE_MAX", 8),
            stale_after_s=_env_float(env, "LOCI_ESP32_STALE_S", 10.0),
            dataset_dir=env.get("LOCI_ESP32_DATASET_DIR") or "",
            dataset_every=max(1, _env_int(env, "LOCI_ESP32_DATASET_EVERY", 10)),
            default_device=env.get("LOCI_ESP32_DEVICE") or DEFAULT_DEVICE,
            warmup=env.get("LOCI_ESP32_WARMUP", "0") == "1",
            depth=(env.get("LOCI_ESP32_DEPTH") or "none").lower(),
            white_balance=(env.get("LOCI_ESP32_WB") or "none").lower(),
            imgsz=max(320, min(1920, _env_int(env, "LOCI_ESP32_IMGSZ", 640))),
            max_upload_bytes=max(64_000, _env_int(env, "LOCI_ESP32_MAX_UPLOAD", 1_500_000)),
            body_deadline_s=max(1.0, _env_float(env, "LOCI_ESP32_BODY_DEADLINE_S", 30.0)),
        )

    def to_dict(self) -> dict:
        return {
            "max_fps": self.max_fps,
            "min_confidence": self.min_confidence,
            "rotate_deg": self.rotate_deg,
            "yolo_model": self.yolo_model,
            "audio_rate": self.audio_rate,
            "audio_gain": self.audio_gain,
            "audio_queue_max": self.audio_queue_max,
            "stale_after_s": self.stale_after_s,
            "dataset_dir": self.dataset_dir,
            "dataset_every": self.dataset_every,
            "default_device": self.default_device,
            "depth": self.depth,
            "white_balance": self.white_balance,
            "imgsz": self.imgsz,
            "max_upload_bytes": self.max_upload_bytes,
            "body_deadline_s": self.body_deadline_s,
        }

    @property
    def frame_interval_ms(self) -> int:
        """What the host asks the board to wait between frames."""
        return int(1000 / self.max_fps) if self.max_fps > 0 else 0


# ---------------------------------------------------------------------------
# Host-side detector
# ---------------------------------------------------------------------------


def jpeg_size(jpeg: bytes) -> tuple[int, int] | None:
    """(width, height) from the JPEG header without decoding the pixels."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(jpeg)) as img:
            return img.size
    except Exception:
        return None


def gray_world(img: Any) -> Any:
    """Gray-world white balance: scale each channel so the three means coincide.

    Cheap colour-cast removal for sensors whose auto white balance is off or has not
    converged (the OV5647 on the P4 arrives with a strong blue cast). Returns a new
    Pillow image; a grey or nearly grey frame is returned unchanged.
    """
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    if means.min() < 1.0:
        return img
    target = float(means.mean())
    gains = np.clip(target / means, 0.5, 3.0)
    if np.allclose(gains, 1.0, atol=0.02):
        return img
    from PIL import Image

    out = np.clip(arr * gains, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def corrected_jpeg(jpeg: bytes, *, rotate_deg: int = 0, quality: int = 85) -> bytes:
    """The frame as the detector sees it (rotated, gray-world balanced), re-encoded as JPEG."""
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if rotate_deg % 360:
        img = img.rotate(-(rotate_deg % 360), expand=True)
    img = gray_world(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class HostYoloDetector:
    """Lazy ultralytics YOLO wrapper producing ``Detection`` objects with ``source="esp32"``.

    The model loads on the first frame (or from :meth:`load` in a warm-up thread). A load
    failure is remembered and reported through :attr:`error`; detection then returns ``[]``
    rather than raising, so the board keeps streaming and the status endpoint says why
    nothing is detected.
    """

    def __init__(
        self,
        model_path: str,
        *,
        classes: list[str] | None = None,
        rotate_deg: int = 0,
        conf: float = 0.4,
        imgsz: int = 640,
        white_balance: str = "none",
    ) -> None:
        self._wb = white_balance
        self.model_path = model_path
        self._classes = list(classes or [])
        self._rotate = rotate_deg % 360
        self._conf = conf
        self._imgsz = imgsz
        self._model: Any = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if self._error is not None:
                return False
            try:
                from ultralytics import YOLO

                model = YOLO(self.model_path)
                if "world" in self.model_path.lower() and self._classes:
                    model.set_classes(self._classes)
                self._model = model
                logger.info("ESP32 detector ready: %s", self.model_path)
                return True
            except Exception as e:  # ImportError, missing weights, torch problems...
                self._error = f"{type(e).__name__}: {e}"
                logger.error("ESP32 detector failed to load %s: %s", self.model_path, self._error)
                return False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    def __call__(self, jpeg: bytes) -> list[Detection]:
        if not self.load():
            return []
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if self._rotate:
            # PIL rotates counter-clockwise; the config is expressed clockwise.
            img = img.rotate(-self._rotate, expand=True)
        if self._wb == "gray":
            img = gray_world(img)
        w, h = img.size
        results = self._model.predict(
            source=np.array(img),
            conf=self._conf,
            imgsz=self._imgsz,
            verbose=False,
            stream=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = str(result.names[cls_id]).lower()
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label=label,
                        cx=((x1 + x2) / 2) / w,
                        cy=((y1 + y2) / 2) / h,
                        width=(x2 - x1) / w,
                        height=(y2 - y1) / h,
                        confidence=conf,
                        source="esp32",
                    )
                )
        return detections

    def status(self) -> dict:
        return {"model": self.model_path, "loaded": self.loaded, "error": self._error}


# ---------------------------------------------------------------------------
# Per-device state and frame results
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    """Outcome of one posted frame; :meth:`to_dict` is the board's response body."""

    device_id: str
    seq: int
    timestamp_ms: int
    detections: list[Detection]
    dropped: bool = False
    reason: str | None = None
    width: int | None = None
    height: int | None = None
    audio_pending: int = 0
    next_interval_ms: int = 0
    infer_ms: float = 0.0
    now_ms: int = 0  # host Unix time; the board derives Unix timestamps for offline frames from it
    memory_rev: str = ""  # changes when remembered objects change; the board refreshes its cache
    backlog: bool = False  # frame captured earlier and uploaded late (store-and-forward)

    @property
    def labels(self) -> list[str]:
        return [d.label for d in self.detections]

    def to_dict(self) -> dict:
        # Kept small on purpose: the firmware parses it with strstr(), not a JSON library.
        return {
            "ok": True,
            "device": self.device_id,
            "seq": self.seq,
            "dropped": self.dropped,
            "reason": self.reason,
            "n": len(self.detections),
            "labels": self.labels,
            "detections": [d.to_dict() for d in self.detections],
            "audio_pending": self.audio_pending,
            "next_ms": self.next_interval_ms,
            "infer_ms": round(self.infer_ms, 1),
            "now_ms": self.now_ms,
            "memory_rev": self.memory_rev,
            "backlog": self.backlog,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Esp32Device:
    device_id: str
    first_seen_ms: int
    last_seen_ms: int
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    errors: int = 0
    last_seq: int = 0
    last_jpeg: bytes | None = None
    last_size: tuple[int, int] | None = None
    last_detections: list[Detection] = field(default_factory=list)
    last_infer_ms: float = 0.0
    backlog_frames: int = 0  # frames uploaded late from the board's card
    dataset_frames: int = 0  # frames saved to LOCI_ESP32_DATASET_DIR on the host
    fps: float = 0.0
    rssi: int | None = None
    device_ms: int | None = None  # the board's millis() of the last frame, for the status page
    audio_queue: deque[bytes] = field(default_factory=deque)
    audio_queued_total: int = 0
    audio_sent_total: int = 0
    audio_dropped_total: int = 0
    _last_arrival_mono: float | None = None
    _last_processed_mono: float | None = None
    _audio_event: asyncio.Event | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    def to_dict(self, now_ms: int, stale_after_s: float) -> dict:
        age_s = (now_ms - self.last_seen_ms) / 1000.0
        return {
            "device_id": self.device_id,
            "online": age_s <= stale_after_s,
            "last_frame_age_s": round(age_s, 1),
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "backlog_frames": self.backlog_frames,
            "dataset_frames": self.dataset_frames,
            "errors": self.errors,
            "fps": round(self.fps, 2),
            "last_seq": self.last_seq,
            "last_size": list(self.last_size) if self.last_size else None,
            "last_infer_ms": round(self.last_infer_ms, 1),
            "last_labels": [d.label for d in self.last_detections],
            "rssi": self.rssi,
            "device_ms": self.device_ms,
            "audio_queue": len(self.audio_queue),
            "audio_queued_total": self.audio_queued_total,
            "audio_sent_total": self.audio_sent_total,
            "audio_dropped_total": self.audio_dropped_total,
        }


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------


class Esp32Hub:
    """Receives frames from one or more boards and feeds the shared ingestion pipeline.

    ``ingestion`` is the demo's :class:`SceneIngestion`. On the handheld branch its
    ``process_detections`` is used; on older trees the hub replicates it with the stable
    ``_update_consensus`` call shape so the browser path's consensus and storage gates apply
    unchanged. ``detector`` is any ``bytes -> list[Detection]`` callable (tests pass fakes).
    """

    def __init__(
        self,
        ingestion: Any,
        cfg: Esp32Config | None = None,
        *,
        detector: Detector | None = None,
        depth_estimator: DepthEstimator | None = None,
        clock: Callable[[], float] = time.time,
        memory: Any = None,
    ) -> None:
        self.cfg = cfg or Esp32Config()
        self._ingestion = ingestion
        self._memory = memory  # SpatialMemory-like (current_objects) for the board's local cache
        self._detector: Detector = detector or HostYoloDetector(
            self.cfg.yolo_model,
            classes=list(getattr(ingestion, "classes", []) or []),
            rotate_deg=self.cfg.rotate_deg,
            conf=self.cfg.min_confidence,
            imgsz=self.cfg.imgsz,
            white_balance=self.cfg.white_balance,
        )
        self._depth = depth_estimator if self.cfg.depth != "none" else None
        self._clock = clock
        self._devices: dict[str, Esp32Device] = {}
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        self.frames_total = 0

    # -- devices -----------------------------------------------------------------

    @property
    def detector(self) -> Detector:
        return self._detector

    @property
    def ingestion(self) -> Any:
        return self._ingestion

    def device(self, device_id: str | None = None) -> Esp32Device:
        key = device_id or self.cfg.default_device
        dev = self._devices.get(key)
        if dev is None:
            now_ms = int(self._clock() * 1000)
            dev = Esp32Device(device_id=key, first_seen_ms=now_ms, last_seen_ms=now_ms)
            self._devices[key] = dev
        return dev

    def devices(self) -> list[Esp32Device]:
        return list(self._devices.values())

    def corrected_latest_jpeg(self, device_id: str | None = None) -> bytes | None:
        """Latest frame as the detector sees it (rotation + white balance applied)."""
        jpeg = self.latest_jpeg(device_id)
        if jpeg is None:
            return None
        if self.cfg.white_balance == "none" and self.cfg.rotate_deg == 0:
            return jpeg
        try:
            return corrected_jpeg(jpeg, rotate_deg=self.cfg.rotate_deg)
        except Exception as e:  # a truncated JPEG from the board must not break the preview
            logger.debug("corrected preview failed: %s", e)
            return jpeg

    def latest_jpeg(self, device_id: str | None = None) -> bytes | None:
        dev = self._devices.get(device_id or self.cfg.default_device)
        return dev.last_jpeg if dev else None

    # -- frames ------------------------------------------------------------------

    async def ingest_frame(
        self,
        jpeg: bytes,
        *,
        device_id: str | None = None,
        seq: int | None = None,
        device_ms: int | None = None,
        rssi: int | None = None,
        captured_ms: int | None = None,
        backlog: bool = False,
    ) -> FrameResult:
        """Detect objects in one JPEG frame and feed them to the shared ingestion.

        Raises ``ValueError`` for anything that is not a JPEG. Frames arriving faster than
        ``cfg.max_fps`` are acknowledged but not processed (``dropped=True``), which keeps a
        chatty board from starving the event loop; the latest JPEG is still kept for the
        preview endpoints.
        """
        if len(jpeg) < 4 or not jpeg.startswith(JPEG_MAGIC):
            raise ValueError("body is not a JPEG frame")

        dev = self.device(device_id)
        now_ms = int(self._clock() * 1000)
        mono = time.monotonic()
        dev.frames_received += 1
        dev.last_seen_ms = now_ms
        dev.device_ms = device_ms
        if rssi is not None:
            dev.rssi = rssi
        # The preview, the live fps and (below) the rate limiter and the last detections
        # describe the live stream. A backlog frame interleaved with it is minutes old: it
        # must not replace the preview or "what's in front of me", and stamping the rate
        # limiter with it made the next live frame "rate_limited" while the card drained.
        if not backlog:
            dev.last_jpeg = jpeg
            if dev._last_arrival_mono is not None:
                dt = mono - dev._last_arrival_mono
                if dt > 0:
                    inst = 1.0 / dt
                    dev.fps = inst if dev.fps == 0.0 else 0.8 * dev.fps + 0.2 * inst
            dev._last_arrival_mono = mono
        dev.last_seq = dev.last_seq + 1 if seq is None else seq
        self.frames_total += 1

        # Store-and-forward: a frame the board kept on its card while the host was unreachable is
        # stored at its capture time (the board's clock is derived from our now_ms), never rate
        # limited (the backlog is spaced in time already), and not treated as "live".
        ts = now_ms
        if backlog and captured_ms and 0 < captured_ms <= now_ms + 60_000:
            if now_ms - captured_ms <= 30 * 24 * 3600 * 1000:
                ts = int(captured_ms)
            dev.backlog_frames += 1

        min_interval = 1.0 / self.cfg.max_fps if self.cfg.max_fps > 0 else 0.0
        if (
            not backlog
            and dev._last_processed_mono is not None
            and min_interval > 0
            and mono - dev._last_processed_mono < 0.9 * min_interval
        ):
            dev.frames_dropped += 1
            return FrameResult(
                device_id=dev.device_id,
                seq=dev.last_seq,
                timestamp_ms=now_ms,
                detections=[],
                dropped=True,
                reason="rate_limited",
                width=dev.last_size[0] if dev.last_size else None,
                height=dev.last_size[1] if dev.last_size else None,
                audio_pending=len(dev.audio_queue),
                next_interval_ms=self.cfg.frame_interval_ms,
            )
        if not backlog:
            dev._last_processed_mono = mono

        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            detections = await loop.run_in_executor(None, self._detector, jpeg)
        except Exception as e:
            dev.errors += 1
            logger.error("ESP32 detector error on %s: %s", dev.device_id, e)
            detections = []
        detections = [d for d in detections if d.confidence >= self.cfg.min_confidence]
        if self._depth is not None and detections:
            try:
                for det, z in zip(detections, self._depth(jpeg, detections), strict=False):
                    if z is not None:
                        det.depth_m = float(z)
            except Exception as e:
                logger.warning("ESP32 depth estimator failed: %s", e)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        self._feed(detections, ts)

        dev.frames_processed += 1
        if not backlog:
            dev.last_detections = detections
        dev.last_infer_ms = infer_ms
        dev.last_size = jpeg_size(jpeg) or dev.last_size
        if self.cfg.dataset_dir and dev.frames_processed % self.cfg.dataset_every == 0:
            await loop.run_in_executor(None, self._dataset_save, dev, jpeg, ts, detections)
        result = FrameResult(
            device_id=dev.device_id,
            seq=dev.last_seq,
            timestamp_ms=ts,
            detections=detections,
            width=dev.last_size[0] if dev.last_size else None,
            height=dev.last_size[1] if dev.last_size else None,
            audio_pending=len(dev.audio_queue),
            next_interval_ms=self.cfg.frame_interval_ms,
            infer_ms=infer_ms,
            now_ms=now_ms,
            memory_rev=self.memory_rev(),
            backlog=backlog,
        )
        self._publish(result)
        return result

    def _feed(self, detections: list[Detection], timestamp_ms: int) -> None:
        """Hand host-side detections to SceneIngestion (consensus + SpatialMemory)."""
        process = getattr(self._ingestion, "process_detections", None)
        if callable(process):
            process(detections, timestamp_ms=timestamp_ms)
            return
        # Legacy SceneIngestion (before the handheld branch landed): replicate
        # process_detections with the stable legacy call shape of _update_consensus.
        for det in detections:
            self._ingestion._update_consensus(
                label=det.label,
                cx=det.cx,
                cy=det.cy,
                confidence=det.confidence,
                timestamp_ms=timestamp_ms,
                depth_m=det.depth_m,
            )
        self._ingestion.last_detections = detections
        self._ingestion.frame_count += 1

    # -- audio queue -------------------------------------------------------------

    def queue_audio(self, pcm: bytes, device_id: str | None = None) -> int:
        """Queue a PCM clip for the board's speaker; returns the queue length."""
        dev = self.device(device_id)
        if not pcm:
            return len(dev.audio_queue)
        while len(dev.audio_queue) >= max(1, self.cfg.audio_queue_max):
            dev.audio_queue.popleft()
            dev.audio_dropped_total += 1
        dev.audio_queue.append(bytes(pcm))
        dev.audio_queued_total += 1
        self._notify_audio(dev)
        return len(dev.audio_queue)

    def _notify_audio(self, dev: Esp32Device) -> None:
        ev = dev._audio_event
        if ev is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if dev._loop is not None and running is not dev._loop:
            dev._loop.call_soon_threadsafe(ev.set)
        else:
            ev.set()

    async def next_audio(self, device_id: str | None = None, wait_s: float = 0.0) -> bytes | None:
        """Pop the next PCM clip, waiting up to ``wait_s`` for one (long poll)."""
        dev = self.device(device_id)
        if dev.audio_queue:
            dev.audio_sent_total += 1
            return dev.audio_queue.popleft()
        if wait_s <= 0:
            return None
        loop = asyncio.get_running_loop()
        if dev._audio_event is None or dev._loop is not loop:
            dev._audio_event = asyncio.Event()
            dev._loop = loop
        dev._audio_event.clear()
        try:
            await asyncio.wait_for(dev._audio_event.wait(), timeout=wait_s)
        except TimeoutError:
            return None
        if dev.audio_queue:
            dev.audio_sent_total += 1
            return dev.audio_queue.popleft()
        return None

    def pending_audio(self, device_id: str | None = None) -> int:
        dev = self._devices.get(device_id or self.cfg.default_device)
        return len(dev.audio_queue) if dev else 0

    # -- dataset recording (host side) -------------------------------------------

    def _dataset_save(
        self, dev: Esp32Device, jpeg: bytes, ts: int, detections: list[Detection]
    ) -> None:
        """Write <dataset_dir>/<device>/<ts>.jpg and .json (pseudo-labels for a custom detector)."""
        import json
        from pathlib import Path

        try:
            folder = Path(self.cfg.dataset_dir).expanduser() / dev.device_id
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"{ts}-{dev.frames_processed:06d}"  # unique even when two frames share a ms
            (folder / f"{stem}.jpg").write_bytes(jpeg)
            record = {
                "timestamp_ms": ts,
                "device": dev.device_id,
                "width": dev.last_size[0] if dev.last_size else None,
                "height": dev.last_size[1] if dev.last_size else None,
                "model": getattr(self._detector, "model_path", None),
                "detections": [d.to_dict() for d in detections],
            }
            (folder / f"{stem}.json").write_text(json.dumps(record))
            dev.dataset_frames += 1
        except Exception as e:
            logger.warning("dataset save failed: %s", e)

    # -- memory snapshot for the board's local cache -----------------------------

    def memory_objects(self, limit: int = 12) -> list[dict]:
        """Compact view of what the host remembers, newest first (for the board's SD cache)."""
        mem = self._memory
        if mem is None or not hasattr(mem, "current_objects"):
            return []
        out: list[dict] = []
        try:
            objects = list(mem.current_objects())
        except Exception as e:
            logger.warning("memory snapshot failed: %s", e)
            return []
        for o in objects[:limit]:
            out.append(
                {
                    "label": getattr(o, "label", "?"),
                    "where": getattr(o, "position_description", "") or "",
                    "room": getattr(o, "room", None),
                    "last_seen_ms": int(getattr(o, "timestamp_ms", 0) or 0),
                    "age_s": int(getattr(o, "age_seconds", 0) or 0),
                }
            )
        return out

    def memory_rev(self, objects: list[dict] | None = None) -> str:
        """Short digest of the remembered objects; stable while nothing moves or appears."""
        import hashlib

        objs = self.memory_objects() if objects is None else objects
        key = "|".join(f"{o['label']}@{o['where']}@{o.get('room') or ''}" for o in objs)
        return hashlib.blake2s(key.encode(), digest_size=4).hexdigest() if objs else "empty"

    @staticmethod
    def memory_summary_text(objects: list[dict], *, max_items: int = 6) -> str:
        """One spoken sentence the board can play when it cannot reach the host."""
        if not objects:
            return "I can't reach the host right now, and I don't remember any objects yet."
        parts = []
        for o in objects[:max_items]:
            where = f" {o['where']}" if o.get("where") else ""
            room = f" in the {o['room']}" if o.get("room") else ""
            parts.append(f"{o['label']}{room}{where}")
        more = len(objects) - max_items
        tail = f", and {more} more" if more > 0 else ""
        lead = "I can't reach the host right now. The last things I remember: "
        return lead + "; ".join(parts) + tail + "."

    def memory_snapshot(self) -> dict:
        objs = self.memory_objects()
        return {
            "rev": self.memory_rev(objs),
            "count": len(objs),
            "objects": objs,
            "summary": self.memory_summary_text(objs),
            "now_ms": int(self._clock() * 1000),
        }

    # -- subscribers (DetectionSource adapter) -----------------------------------

    def subscribe(self, maxsize: int = 64) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, result: FrameResult) -> None:
        with self._sub_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(result)
            except queue.Full:
                try:
                    q.get_nowait()  # drop the oldest; a live feed must not block the host
                    q.put_nowait(result)
                except (queue.Empty, queue.Full):
                    pass

    # -- status ------------------------------------------------------------------

    def status(self) -> dict:
        now_ms = int(self._clock() * 1000)
        det_status = getattr(self._detector, "status", None)
        return {
            "config": self.cfg.to_dict(),
            "detector": det_status() if callable(det_status) else {"custom": True},
            "frames_total": self.frames_total,
            "devices": {
                dev.device_id: dev.to_dict(now_ms, self.cfg.stale_after_s)
                for dev in self._devices.values()
            },
            "ingestion_frame_count": getattr(self._ingestion, "frame_count", None),
            "memory_rev": self.memory_rev(),
        }


# ---------------------------------------------------------------------------
# DetectionSource adapter (handheld runtime)
# ---------------------------------------------------------------------------


class Esp32DetectionSource:
    """Expose an :class:`Esp32Hub` as a handheld ``DetectionSource``.

    Frames arrive from the board over HTTP, so there is nothing to open: ``start`` just
    subscribes to the hub and ``events`` turns each processed frame into a ``DetectionFrame``
    of ``SpatialDetection`` with ``xyz_cam_m=None`` (no depth). ``app.detection_source`` is
    imported lazily inside ``events``/``status`` so this module imports on trees where the
    handheld modules are not present.
    """

    name = "esp32"

    def __init__(self, hub: Esp32Hub, device_id: str | None = None) -> None:
        self._hub = hub
        self._device_id = device_id or hub.cfg.default_device
        self.classes: list[str] = list(getattr(hub.ingestion, "classes", []) or [])
        self._q: queue.Queue | None = None
        self._stopped = threading.Event()
        self._frames = 0
        self._error: str | None = None

    def start(self) -> None:
        self._stopped.clear()
        if self._q is None:
            self._q = self._hub.subscribe()

    def stop(self) -> None:
        self._stopped.set()
        if self._q is not None:
            self._hub.unsubscribe(self._q)
            with _suppress_full():
                self._q.put_nowait(None)

    def events(self) -> Iterator[Any]:
        from .detection_source import COCO80, DetectionFrame, SpatialDetection

        if self._q is None:
            self.start()
        assert self._q is not None
        label_ids = {name: i for i, name in enumerate(COCO80)}
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stopped.is_set():
                    break
                continue
            if item is None:  # stop() sentinel: everything queued before it has been yielded
                break
            result: FrameResult = item
            if result.device_id != self._device_id:
                continue
            dets = [
                SpatialDetection(
                    label=d.label,
                    label_id=label_ids.get(d.label, -1),
                    confidence=d.confidence,
                    xmin=max(0.0, d.cx - d.width / 2),
                    ymin=max(0.0, d.cy - d.height / 2),
                    xmax=min(1.0, d.cx + d.width / 2),
                    ymax=min(1.0, d.cy + d.height / 2),
                    xyz_cam_m=None,
                )
                for d in result.detections
            ]
            self._frames += 1
            yield DetectionFrame(
                timestamp_ms=result.timestamp_ms,
                seq=result.seq,
                detections=dets,
                source=self.name,
                device_ts_ms=None,
            )

    def status(self) -> Any:
        from .detection_source import SourceStatus

        dev = self._hub.device(self._device_id)
        now_ms = int(time.time() * 1000)
        age = (now_ms - dev.last_seen_ms) / 1000.0 if dev.frames_received else None
        if self._stopped.is_set():
            state = "stopped"
        elif dev.frames_received == 0 or (age is not None and age > self._hub.cfg.stale_after_s):
            state = "waiting_for_camera"
        else:
            state = "running"
        return SourceStatus(
            state=state,
            fps=dev.fps,
            frames=self._frames,
            reconnects=0,
            last_frame_age_s=age,
            usb_speed=None,
            device=self._device_id,
            error=self._error,
        )


class _suppress_full:
    """Tiny context manager: ignore ``queue.Full`` on the stop sentinel."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is queue.Full
