"""ESP32 frame receiver: rate limiting, detection → SceneIngestion hand-off, audio queue.

Fakes only: the "board" is a JPEG made with Pillow, the "YOLO" is a callable returning
fixed boxes, and SpatialMemory is a MagicMock (as in tests/test_temporal_consensus.py).
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.esp32_source import (  # noqa: E402
    Esp32Config,
    Esp32DetectionSource,
    Esp32Hub,
    FrameResult,
    HostYoloDetector,
    jpeg_size,
)
from app.scene_ingestion import Detection, SceneIngestion  # noqa: E402


def _jpeg(w: int = 64, h: int = 48) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _ingestion(**kwargs) -> SceneIngestion:
    memory = MagicMock()
    memory.observe = MagicMock(return_value="state-1")
    return SceneIngestion(
        memory=memory,
        vlm_client=None,
        use_vlm_fallback=False,
        consensus_min_count=kwargs.pop("consensus_min_count", 2),
        **kwargs,
    )


def _cup(conf: float = 0.9) -> Detection:
    return Detection("cup", cx=0.5, cy=0.5, width=0.2, height=0.2, confidence=conf, source="esp32")


class FakeClock:
    def __init__(self, t: float = 1_700_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_from_env_defaults_and_overrides() -> None:
    cfg = Esp32Config.from_env({})
    assert cfg.max_fps == 5.0 and cfg.yolo_model == "yolov8n.pt" and cfg.audio_rate == 16000
    assert cfg.frame_interval_ms == 200
    cfg = Esp32Config.from_env(
        {
            "LOCI_ESP32_MAX_FPS": "2",
            "LOCI_ESP32_ROTATE": "45",  # not a right angle → 0 with a warning
            "YOLO_MODEL": "yolov8s.pt",
            "LOCI_ESP32_MIN_CONF": "0.3",
            "LOCI_ESP32_DEVICE": "pendant-1",
        }
    )
    assert cfg.max_fps == 2 and cfg.rotate_deg == 0 and cfg.yolo_model == "yolov8s.pt"
    assert cfg.min_confidence == 0.3 and cfg.default_device == "pendant-1"
    assert Esp32Config.from_env({"LOCI_ESP32_ROTATE": "270"}).rotate_deg == 270
    assert (
        Esp32Config.from_env({"LOCI_ESP32_YOLO_MODEL": "x.pt", "YOLO_MODEL": "y.pt"}).yolo_model
        == "x.pt"
    )
    assert "yolo_model" in cfg.to_dict()


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def test_rejects_non_jpeg() -> None:
    hub = Esp32Hub(_ingestion(), Esp32Config(), detector=lambda _b: [])
    with pytest.raises(ValueError):
        _run(hub.ingest_frame(b"PNG\x00\x00\x00\x00"))


def test_detections_reach_consensus_and_memory() -> None:
    ing = _ingestion(consensus_min_count=2)
    clock = FakeClock()
    calls: list[bytes] = []

    def detector(jpeg: bytes) -> list[Detection]:
        calls.append(jpeg)
        return [_cup(), Detection("ghost", 0.1, 0.1, 0.1, 0.1, confidence=0.2, source="esp32")]

    hub = Esp32Hub(ing, Esp32Config(max_fps=0), detector=detector, clock=clock)
    frame = _jpeg()

    async def go():
        r1 = await hub.ingest_frame(frame, device_id="b1", seq=7, device_ms=1234, rssi=-61)
        clock.t += 0.3
        r2 = await hub.ingest_frame(frame, device_id="b1", seq=8)
        return r1, r2

    r1, r2 = _run(go())
    assert isinstance(r1, FrameResult) and not r1.dropped
    assert r1.labels == ["cup"]  # the 0.2-confidence ghost is filtered by min_confidence
    assert r1.seq == 7 and r2.seq == 8 and r1.width == 64 and r1.height == 48
    assert len(calls) == 2
    # one frame buffers, the second one fires the consensus → observe()
    ing._memory.observe.assert_called_once()
    kwargs = ing._memory.observe.call_args.kwargs
    assert kwargs["label"] == "cup" and kwargs["timestamp_ms"] == int(clock.t * 1000)
    assert ing.frame_count == 2 and [d.label for d in ing.last_detections] == ["cup"]
    dev = hub.device("b1")
    assert dev.frames_received == 2 and dev.frames_processed == 2 and dev.rssi == -61
    assert dev.device_ms is None  # second frame carried no X-Device-Ms
    body = r2.to_dict()
    assert body["ok"] is True and body["n"] == 1 and body["labels"] == ["cup"]
    assert body["audio_pending"] == 0 and body["next_ms"] == 0


def test_uses_process_detections_when_ingestion_has_it() -> None:
    ing = MagicMock()
    ing.classes = ["cup"]
    ing.frame_count = 0
    ing.process_detections = MagicMock(return_value=[])
    hub = Esp32Hub(ing, Esp32Config(max_fps=0), detector=lambda _b: [_cup()])
    _run(hub.ingest_frame(_jpeg()))
    ing.process_detections.assert_called_once()
    (dets,) = ing.process_detections.call_args.args
    assert dets[0].label == "cup" and dets[0].source == "esp32"
    assert "timestamp_ms" in ing.process_detections.call_args.kwargs
    ing._update_consensus.assert_not_called()


def test_rate_limit_drops_but_keeps_preview() -> None:
    ing = _ingestion()
    hub = Esp32Hub(ing, Esp32Config(max_fps=5.0), detector=lambda _b: [_cup()])
    a, b = _jpeg(32, 32), _jpeg(16, 16)

    async def go():
        r1 = await hub.ingest_frame(a)
        r2 = await hub.ingest_frame(b)  # arrives "immediately" → dropped
        return r1, r2

    r1, r2 = _run(go())
    assert not r1.dropped and r2.dropped and r2.reason == "rate_limited"
    assert r2.next_interval_ms == 200 and r2.detections == []
    dev = hub.device()
    assert dev.frames_dropped == 1 and dev.frames_processed == 1
    assert hub.latest_jpeg() == b  # preview still shows the newest frame
    assert ing.frame_count == 1


def test_backlog_frames_leave_the_live_stream_alone() -> None:
    """Card frames drained between live frames must not rate-limit them or replace the preview."""
    hub = Esp32Hub(_ingestion(), Esp32Config(max_fps=5.0), detector=lambda _b: [_cup()])
    live1, old, live2 = _jpeg(32, 32), _jpeg(24, 24), _jpeg(16, 16)

    async def go():
        r1 = await hub.ingest_frame(live1)
        await asyncio.sleep(0.25)  # one live interval later ...
        rb = await hub.ingest_frame(old, backlog=True, captured_ms=1)  # ... a card frame goes up
        r2 = await hub.ingest_frame(live2)  # and the next live frame right after it
        return r1, rb, r2

    r1, rb, r2 = _run(go())
    assert not r1.dropped and not rb.dropped and not r2.dropped
    dev = hub.device()
    assert dev.frames_processed == 3 and dev.frames_dropped == 0 and dev.backlog_frames == 1
    assert hub.latest_jpeg() == live2
    rb_only = Esp32Hub(_ingestion(), Esp32Config(max_fps=5.0), detector=lambda _b: [_cup()])
    _run(rb_only.ingest_frame(old, backlog=True))
    assert rb_only.device().last_detections == [] and rb_only.device().fps == 0.0


def test_detector_exception_is_counted_not_raised() -> None:
    def boom(_b: bytes) -> list[Detection]:
        raise RuntimeError("cuda fell over")

    hub = Esp32Hub(_ingestion(), Esp32Config(max_fps=0), detector=boom)
    r = _run(hub.ingest_frame(_jpeg()))
    assert r.detections == [] and hub.device().errors == 1


def test_depth_estimator_is_optional_and_attached() -> None:
    def depth(_jpeg: bytes, dets: list[Detection]) -> list[float | None]:
        return [1.5 for _ in dets]

    hub = Esp32Hub(
        _ingestion(),
        Esp32Config(max_fps=0, depth="mono"),
        detector=lambda _b: [_cup()],
        depth_estimator=depth,
    )
    r = _run(hub.ingest_frame(_jpeg()))
    assert r.detections[0].depth_m == 1.5
    # depth="none" ignores a supplied estimator
    hub2 = Esp32Hub(
        _ingestion(), Esp32Config(max_fps=0), detector=lambda _b: [_cup()], depth_estimator=depth
    )
    assert _run(hub2.ingest_frame(_jpeg())).detections[0].depth_m is None


def test_status_reports_devices_and_detector() -> None:
    clock = FakeClock()
    hub = Esp32Hub(
        _ingestion(), Esp32Config(max_fps=0, stale_after_s=5), detector=lambda _b: [], clock=clock
    )
    _run(hub.ingest_frame(_jpeg(), device_id="cam-a"))
    st = hub.status()
    assert st["devices"]["cam-a"]["online"] is True and st["detector"] == {"custom": True}
    clock.t += 30
    assert hub.status()["devices"]["cam-a"]["online"] is False
    assert st["config"]["max_fps"] == 0


# ---------------------------------------------------------------------------
# audio queue
# ---------------------------------------------------------------------------


def test_audio_queue_fifo_bounded_and_longpoll() -> None:
    hub = Esp32Hub(_ingestion(), Esp32Config(audio_queue_max=2), detector=lambda _b: [])
    assert hub.queue_audio(b"", "d") == 0
    assert hub.queue_audio(b"a" * 10, "d") == 1
    assert hub.queue_audio(b"b" * 10, "d") == 2
    assert hub.queue_audio(b"c" * 10, "d") == 2  # oldest dropped
    assert hub.device("d").audio_dropped_total == 1
    assert hub.pending_audio("d") == 2

    async def drain():
        first = await hub.next_audio("d")
        second = await hub.next_audio("d")
        third = await hub.next_audio("d", wait_s=0.05)
        return first, second, third

    first, second, third = _run(drain())
    assert first == b"b" * 10 and second == b"c" * 10 and third is None
    assert hub.device("d").audio_sent_total == 2

    async def longpoll():
        async def producer():
            await asyncio.sleep(0.05)
            hub.queue_audio(b"late", "d")

        task = asyncio.create_task(producer())
        got = await hub.next_audio("d", wait_s=2.0)
        await task
        return got

    assert _run(longpoll()) == b"late"


def test_audio_queue_notify_from_other_thread() -> None:
    hub = Esp32Hub(_ingestion(), Esp32Config(), detector=lambda _b: [])

    async def go():
        def worker():
            hub.queue_audio(b"thread", "t")

        timer = threading.Timer(0.05, worker)
        timer.start()
        got = await hub.next_audio("t", wait_s=2.0)
        timer.join()
        return got

    assert _run(go()) == b"thread"


# ---------------------------------------------------------------------------
# DetectionSource adapter (handheld runtime)
# ---------------------------------------------------------------------------


def test_detection_source_adapter_yields_frames() -> None:
    pytest.importorskip("app.detection_source")
    ing = _ingestion()
    ing._classes = ["cup", "person"]
    hub = Esp32Hub(ing, Esp32Config(max_fps=0), detector=lambda _b: [_cup()])
    src = Esp32DetectionSource(hub)
    src.start()
    assert src.name == "esp32" and src.classes == ["cup", "person"]
    _run(hub.ingest_frame(_jpeg(), seq=3))
    src.stop()
    frames = list(src.events())
    assert len(frames) == 1
    fr = frames[0]
    assert fr.source == "esp32" and fr.seq == 3
    det = fr.detections[0]
    assert det.label == "cup" and det.xyz_cam_m is None and det.depth_m is None
    assert det.cx == pytest.approx(0.5) and det.width == pytest.approx(0.2)
    assert det.label_id == 41  # COCO index of "cup"
    assert src.status().state == "stopped"


def test_detection_source_status_without_frames() -> None:
    pytest.importorskip("app.detection_source")
    hub = Esp32Hub(_ingestion(), Esp32Config(), detector=lambda _b: [])
    src = Esp32DetectionSource(hub, device_id="x")
    src.start()
    st = src.status()
    assert st.state == "waiting_for_camera" and st.device == "x" and st.frames == 0
    src.stop()


# ---------------------------------------------------------------------------
# host detector (no model on CI)
# ---------------------------------------------------------------------------


def test_host_yolo_detector_remembers_load_failure(tmp_path) -> None:
    det = HostYoloDetector(str(tmp_path / "missing-weights.pt"), conf=0.4)
    assert det(_jpeg()) == []
    assert det.loaded is False and det.error  # ultralytics missing or weights missing: both fine
    assert det.status()["model"].endswith("missing-weights.pt")
    assert det(_jpeg()) == []  # second call does not retry the load


def test_jpeg_size() -> None:
    assert jpeg_size(_jpeg(40, 30)) == (40, 30)
    assert jpeg_size(b"\xff\xd8garbage") is None
