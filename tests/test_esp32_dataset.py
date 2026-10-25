"""Host-side dataset recording and the YOLO export tool (no ultralytics, no hardware)."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from unittest.mock import MagicMock

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial", "tools"))

import esp32_dataset as ds  # noqa: E402
from app.esp32_source import Esp32Config, Esp32Hub  # noqa: E402
from app.scene_ingestion import Detection  # noqa: E402


def _jpeg(w: int = 40, h: int = 30) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def _hub(tmp_path, every: int = 2) -> Esp32Hub:
    ing = MagicMock()
    ing.classes = ["cup"]
    ing.frame_count = 0
    ing.process_detections = MagicMock(return_value=[])
    cfg = Esp32Config(max_fps=0, dataset_dir=str(tmp_path / "ds"), dataset_every=every)
    dets = [
        Detection("cup", 0.5, 0.5, 0.2, 0.3, 0.9, source="esp32"),
        Detection("bottle", 0.2, 0.7, 0.1, 0.4, 0.45, source="esp32"),
    ]
    return Esp32Hub(ing, cfg, detector=lambda _b: dets, clock=lambda: 1_700_000_000.0)


def test_config_reads_dataset_settings() -> None:
    cfg = Esp32Config.from_env({"LOCI_ESP32_DATASET_DIR": "~/x", "LOCI_ESP32_DATASET_EVERY": "3"})
    assert cfg.dataset_dir == "~/x" and cfg.dataset_every == 3
    assert (
        Esp32Config.from_env({}).dataset_dir == "" and Esp32Config.from_env({}).dataset_every == 10
    )
    assert cfg.to_dict()["dataset_every"] == 3


def test_hub_records_every_nth_frame(tmp_path) -> None:
    hub = _hub(tmp_path, every=2)

    async def go():
        for _ in range(4):
            await hub.ingest_frame(_jpeg(), device_id="p4")

    asyncio.run(go())
    folder = tmp_path / "ds" / "p4"
    jpgs = sorted(folder.glob("*.jpg"))
    assert len(jpgs) == 2 and hub.device("p4").dataset_frames == 2
    meta = json.loads(jpgs[0].with_suffix(".json").read_text())
    assert meta["device"] == "p4" and meta["width"] == 40 and meta["height"] == 30
    assert [d["label"] for d in meta["detections"]] == ["cup", "bottle"]
    assert hub.status()["devices"]["p4"]["dataset_frames"] == 2


def test_export_builds_yolo_dataset(tmp_path) -> None:
    hub = _hub(tmp_path, every=1)

    async def go():
        for _ in range(6):
            await hub.ingest_frame(_jpeg(), device_id="p4")

    asyncio.run(go())
    root = tmp_path / "ds"
    st = ds.stats(root, min_conf=0.0)
    assert st["frames"] == 6 and st["boxes_per_class"] == {"cup": 6, "bottle": 6}
    # confidence floor drops the 0.45 bottle pseudo-labels
    report = ds.export(
        root, tmp_path / "yolo", None, None, min_conf=0.5, val_fraction=0.34, keep_empty=False
    )
    assert report["classes"] == ["cup"] and report["frames_exported"] == 6
    assert report["train"] + report["val"] == 6 and report["val"] == 2
    label_files = sorted((tmp_path / "yolo" / "labels" / "train").glob("*.txt"))
    assert label_files and label_files[0].read_text().startswith(
        "0 0.500000 0.500000 0.200000 0.300000"
    )
    yaml_text = (tmp_path / "yolo" / "data.yaml").read_text()
    assert "names:\n  0: cup\n" in yaml_text and "train: images/train" in yaml_text
    assert len(list((tmp_path / "yolo" / "images" / "val").glob("*.jpg"))) == 2


def test_hand_labels_override_pseudo_labels(tmp_path) -> None:
    hub = _hub(tmp_path, every=1)

    async def go():
        await hub.ingest_frame(_jpeg(), device_id="p4")
        await hub.ingest_frame(_jpeg(), device_id="p4")

    asyncio.run(go())
    root = tmp_path / "ds"
    stems = sorted(p.stem for p in (root / "p4").glob("*.jpg"))
    labels = tmp_path / "hand"
    labels.mkdir()
    (labels / "classes.txt").write_text("keys\ncup\n")
    (labels / f"{stems[0]}.txt").write_text("0 0.3 0.4 0.1 0.1\n1 0.6 0.6 0.2 0.2\n")
    report = ds.export(root, tmp_path / "yolo2", ["keys", "cup"], labels, 0.5, 0.5, keep_empty=True)
    assert report["boxes"] == {"keys": 1, "cup": 2}  # one hand-labelled frame, one pseudo-labelled
    assert report["classes"] == ["keys", "cup"]
    all_lines = []
    for split in ("train", "val"):
        for f in (tmp_path / "yolo2" / "labels" / split).glob("*.txt"):
            all_lines += f.read_text().splitlines()
    assert any(line.startswith("0 0.300000 0.400000") for line in all_lines)


def test_yolo_lines_clip_and_skip() -> None:
    names = ["cup"]
    lines = ds.yolo_lines(
        [
            ("cup", 1.2, -0.1, 0.5, 0.5, 0.9),
            ("dog", 0.5, 0.5, 0.1, 0.1, 0.9),
            ("cup", 0.5, 0.5, 0.0, 0.2, 0.9),
        ],
        names,
    )
    assert lines == ["0 1.000000 0.000000 0.500000 0.500000"]
    assert ds.parse_yolo_txt("1 0.5 0.5 0.1 0.1\nbad line\n9 0 0 0 0\n", ["a", "b"]) == [
        ("b", 0.5, 0.5, 0.1, 0.1, 1.0)
    ]
