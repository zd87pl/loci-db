"""Host-side image handling for the ESP32 path: white balance, inference size, corrected preview."""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.esp32_api import install_esp32_routes  # noqa: E402
from app.esp32_source import (  # noqa: E402
    Esp32Config,
    Esp32Hub,
    HostYoloDetector,
    corrected_jpeg,
    gray_world,
)
from app.scene_ingestion import SceneIngestion  # noqa: E402


def _blue_jpeg(w: int = 64, h: int = 48) -> bytes:
    """A frame with the P4's blue cast: mean RGB roughly (107, 157, 191)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = 107
    arr[..., 1] = 157
    arr[..., 2] = 191
    arr[: h // 2, : w // 2] = (60, 90, 120)  # a darker patch so the image is not flat
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _means(jpeg: bytes) -> list[float]:
    arr = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.float32)
    return [float(m) for m in arr.reshape(-1, 3).mean(axis=0)]


def test_config_reads_white_balance_and_imgsz() -> None:
    cfg = Esp32Config.from_env({})
    assert cfg.white_balance == "none" and cfg.imgsz == 640
    cfg = Esp32Config.from_env({"LOCI_ESP32_WB": "Gray", "LOCI_ESP32_IMGSZ": "960"})
    assert cfg.white_balance == "gray" and cfg.imgsz == 960
    assert Esp32Config.from_env({"LOCI_ESP32_IMGSZ": "64"}).imgsz == 320  # clamped
    assert cfg.to_dict()["white_balance"] == "gray" and cfg.to_dict()["imgsz"] == 960


def test_gray_world_equalises_channel_means() -> None:
    img = Image.open(io.BytesIO(_blue_jpeg())).convert("RGB")
    before = np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    assert before[2] - before[0] > 60  # strong blue cast going in
    out = gray_world(img)
    after = np.asarray(out, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    assert after.max() - after.min() < 3.0
    # an already neutral image is returned as-is
    grey = Image.new("RGB", (16, 16), (120, 121, 119))
    assert gray_world(grey) is grey


def test_corrected_jpeg_rotates_and_balances() -> None:
    raw = _blue_jpeg(64, 48)
    out = corrected_jpeg(raw, rotate_deg=90)
    assert out.startswith(b"\xff\xd8")
    im = Image.open(io.BytesIO(out))
    assert im.size == (48, 64)
    m = _means(out)
    assert max(m) - min(m) < 6.0


def test_detector_option_is_stored_without_a_model(tmp_path) -> None:
    det = HostYoloDetector(str(tmp_path / "nope.pt"), white_balance="gray", imgsz=960)
    assert det._wb == "gray" and det._imgsz == 960
    assert det(_blue_jpeg()) == []  # no weights: silently empty, error remembered


def test_frame_endpoint_serves_corrected_preview() -> None:
    memory = MagicMock()
    memory.observe = MagicMock(return_value="s1")
    memory.observation_count = 0
    memory.tracked_labels = []
    ingestion = SceneIngestion(memory=memory, vlm_client=None, use_vlm_fallback=False)
    cfg = Esp32Config(max_fps=0, white_balance="gray")
    hub = Esp32Hub(ingestion, cfg, detector=lambda _b: [])
    app = FastAPI()
    install_esp32_routes(
        app, ingestion=ingestion, memory=memory, voice=MagicMock(), cfg=cfg, hub=hub
    )
    client = TestClient(app)
    raw = _blue_jpeg()
    assert (
        client.post(
            "/api/esp32/frame", content=raw, headers={"Content-Type": "image/jpeg"}
        ).status_code
        == 200
    )
    assert client.get("/api/esp32/frame.jpg").content == raw
    corrected = client.get("/api/esp32/frame.jpg?corrected=1").content
    assert corrected != raw and corrected.startswith(b"\xff\xd8")
    m = _means(corrected)
    assert max(m) - min(m) < 6.0
