"""Turn the ESP32 host's recorded frames into a YOLO dataset and fine-tune a detector on them.

The host saves every Nth processed frame as ``<LOCI_ESP32_DATASET_DIR>/<device>/<ts>.jpg`` with a
``<ts>.json`` side-car holding the detections it made (COCO labels from yolov8n: *pseudo-labels*).
The board keeps the same layout on its card under ``/sdcard/loci/frames``. This tool:

    stats   count frames and labels per class
    export  write YOLO-format images/labels + data.yaml (train/val split), optionally only some
            classes, optionally overriding pseudo-labels with hand-made YOLO .txt labels
    train   fine-tune a YOLO model on the exported dataset (ultralytics)

Pseudo-labels calibrate a model to this camera (lens, colour, viewpoint); classes COCO does not
know (keys, wallet, glasses) need hand labels: draw them in any YOLO-format tool and pass the folder
with ``--labels``; a file named ``<ts>.txt`` there replaces the pseudo-labels of that frame.

Examples::

    python demo_spatial/tools/esp32_dataset.py stats ~/.local/share/loci-esp32/dataset
    python demo_spatial/tools/esp32_dataset.py export ~/.local/share/loci-esp32/dataset \
        --out /tmp/loci-yolo --classes cup,bottle,laptop,keys --labels ~/labels --min-conf 0.5
    python demo_spatial/tools/esp32_dataset.py train /tmp/loci-yolo/data.yaml --epochs 40
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Frame:
    jpg: Path
    meta: dict
    labels: list[tuple[str, float, float, float, float, float]] = field(default_factory=list)
    # (label, cx, cy, w, h, confidence) with normalised coordinates

    @property
    def stem(self) -> str:
        return self.jpg.stem


def load_frames(root: Path, min_conf: float = 0.0) -> list[Frame]:
    """Every ``<ts>.jpg`` + ``<ts>.json`` pair below ``root`` (host layout or the card's)."""
    frames: list[Frame] = []
    for jpg in sorted(root.rglob("*.jpg")):
        side = jpg.with_suffix(".json")
        meta: dict = {}
        if side.exists():
            try:
                meta = json.loads(side.read_text())
            except json.JSONDecodeError:
                meta = {}
        dets = meta.get("detections") or []
        labels = []
        for d in dets:
            try:
                conf = float(d.get("confidence", 1.0))
                if conf < min_conf:
                    continue
                labels.append(
                    (
                        str(d["label"]).strip().lower(),
                        float(d["cx"]),
                        float(d["cy"]),
                        float(d["width"]),
                        float(d["height"]),
                        conf,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        frames.append(Frame(jpg=jpg, meta=meta, labels=labels))
    return frames


def parse_yolo_txt(
    text: str, names: list[str]
) -> list[tuple[str, float, float, float, float, float]]:
    """Hand-made YOLO label file (``class cx cy w h`` per line) → our label tuples."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if 0 <= idx < len(names):
            out.append((names[idx], cx, cy, w, h, 1.0))
    return out


def yolo_lines(labels, names: list[str]) -> list[str]:
    """Label tuples → YOLO lines for the classes in ``names`` (others are skipped)."""
    index = {n: i for i, n in enumerate(names)}
    lines = []
    for label, cx, cy, w, h, _conf in labels:
        if label not in index:
            continue
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{index[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def split_frames(
    frames: list[Frame], val_fraction: float, seed: int = 0
) -> tuple[list[Frame], list[Frame]]:
    """Deterministic train/val split; frames are shuffled, not interleaved by time."""
    rng = random.Random(seed)  # noqa: S311 - dataset shuffling, not cryptography
    order = list(frames)
    rng.shuffle(order)
    n_val = int(round(len(order) * val_fraction))
    if len(order) > 1:
        n_val = max(1, min(n_val, len(order) - 1))
    return order[n_val:], order[:n_val]


def export(
    root: Path,
    out: Path,
    classes: list[str] | None,
    labels_dir: Path | None,
    min_conf: float,
    val_fraction: float,
    keep_empty: bool,
    seed: int = 0,
) -> dict:
    frames = load_frames(root, min_conf=min_conf)
    names: list[str] = list(classes) if classes else []
    if labels_dir is not None:
        classes_file = labels_dir / "classes.txt"
        hand_names = classes_file.read_text().split() if classes_file.exists() else names
        for fr in frames:
            txt = labels_dir / f"{fr.stem}.txt"
            if txt.exists():
                fr.labels = parse_yolo_txt(txt.read_text(), hand_names)
                fr.meta["hand_labelled"] = True
    if not names:
        counts = Counter(lbl for fr in frames for (lbl, *_rest) in fr.labels)
        names = sorted(counts)
    kept = [fr for fr in frames if keep_empty or yolo_lines(fr.labels, names)]
    train, val = split_frames(kept, val_fraction, seed)
    for split_name, items in (("train", train), ("val", val)):
        (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for fr in items:
            shutil.copy2(fr.jpg, out / "images" / split_name / fr.jpg.name)
            lines = yolo_lines(fr.labels, names)
            (out / "labels" / split_name / f"{fr.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    boxes = Counter(lbl for fr in kept for (lbl, *_rest) in fr.labels if lbl in names)
    return {
        "frames_found": len(frames),
        "frames_exported": len(kept),
        "train": len(train),
        "val": len(val),
        "classes": names,
        "boxes": dict(boxes),
        "data_yaml": str(data_yaml),
    }


def stats(root: Path, min_conf: float) -> dict:
    frames = load_frames(root, min_conf=min_conf)
    per_class = Counter(lbl for fr in frames for (lbl, *_rest) in fr.labels)
    devices = Counter(fr.meta.get("device", fr.jpg.parent.name) for fr in frames)
    return {
        "frames": len(frames),
        "with_detections": sum(1 for fr in frames if fr.labels),
        "devices": dict(devices),
        "boxes_per_class": dict(per_class.most_common()),
    }


def train(data_yaml: Path, model: str, epochs: int, imgsz: int, batch: int, project: Path) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed in this environment")
    results = YOLO(model).train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name="loci-esp32",
    )
    save_dir = getattr(results, "save_dir", None) or project / "loci-esp32"
    return str(Path(save_dir) / "weights" / "best.pt")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="count frames and labels")
    s.add_argument("root", type=Path)
    s.add_argument("--min-conf", type=float, default=0.0)

    e = sub.add_parser("export", help="write a YOLO dataset")
    e.add_argument("root", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--classes", help="comma-separated class names (default: all seen)")
    e.add_argument(
        "--labels", type=Path, help="folder with hand-made <ts>.txt YOLO labels (+ classes.txt)"
    )
    e.add_argument("--min-conf", type=float, default=0.5, help="pseudo-label confidence floor")
    e.add_argument("--val", type=float, default=0.2, help="validation fraction")
    e.add_argument("--keep-empty", action="store_true", help="also export frames without boxes")
    e.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train", help="fine-tune a YOLO model")
    t.add_argument("data_yaml", type=Path)
    t.add_argument("--model", default="yolov8n.pt")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--imgsz", type=int, default=960)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--project", type=Path, default=Path("runs"))

    args = ap.parse_args(argv)
    if args.cmd == "stats":
        print(json.dumps(stats(args.root, args.min_conf), indent=2))
    elif args.cmd == "export":
        classes = (
            [c.strip().lower() for c in args.classes.split(",") if c.strip()]
            if args.classes
            else None
        )
        report = export(
            args.root,
            args.out,
            classes,
            args.labels,
            args.min_conf,
            args.val,
            args.keep_empty,
            args.seed,
        )
        print(json.dumps(report, indent=2))
        print(f"\nnext: python {Path(__file__).name} train {report['data_yaml']} --epochs 30")
    elif args.cmd == "train":
        best = train(args.data_yaml, args.model, args.epochs, args.imgsz, args.batch, args.project)
        print(f"best weights: {best}\nuse them with LOCI_ESP32_YOLO_MODEL={best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
