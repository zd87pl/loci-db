#!/usr/bin/env python3
"""Printable AprilTag 36h11 sheets for room anchoring: PNG at 300 dpi (+ PDF) per tag id.

Recipe (verified on the Mac and the Pi): ``cv2.aruco.generateImageMarker`` draws the 8x8-cell
marker (6x6 data bits + the 1-cell black border); with ``--size-mm 160`` at 300 dpi one cell is
236 px, so the black square is 1888 px = 159.85 mm, framed by a one-cell (~2 cm) white quiet
zone and a caption band with the id, the exact printed size and a "this way up" arrow.

    python deploy/handheld/scripts/make_tag_sheet.py --ids 0 1 2 3 --size-mm 160 \
        --out deploy/handheld/tags

Print at 100 % (no "fit to page"), measure the black square with a ruler and put that value
(metres) into rooms.json ``tag_size_m``. cv2 and Pillow are imported lazily so ``--help`` and the
layout maths work without them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MM_PER_INCH = 25.4
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "deploy" / "handheld" / "tags"

# family -> (cv2.aruco dictionary name, cells per side including the black border, id count)
FAMILIES: dict[str, tuple[str, int, int]] = {
    "tag36h11": ("DICT_APRILTAG_36h11", 8, 587),
    "tag25h9": ("DICT_APRILTAG_25h9", 7, 35),
    "tag16h5": ("DICT_APRILTAG_16h5", 6, 30),
}


@dataclass(frozen=True)
class SheetPlan:
    """Pixel layout of one sheet."""

    size_mm: float
    dpi: int
    cells: int
    quiet_cells: int
    caption_px: int
    cell_px: int
    tag_px: int
    canvas_w: int
    canvas_h: int
    actual_mm: float

    @property
    def margin_px(self) -> int:
        return self.cell_px * self.quiet_cells

    @property
    def tag_size_m(self) -> float:
        return self.actual_mm / 1000.0

    @property
    def sheet_mm(self) -> tuple[float, float]:
        return (
            self.canvas_w / self.dpi * MM_PER_INCH,
            self.canvas_h / self.dpi * MM_PER_INCH,
        )


def plan_sheet(
    size_mm: float = 160.0,
    dpi: int = 300,
    *,
    cells: int = 8,
    quiet_cells: int = 1,
    caption_px: int = 360,
) -> SheetPlan:
    """Round the cell to whole pixels so the printed square is exact at the given dpi."""
    if size_mm <= 0 or dpi <= 0 or cells <= 0:
        raise ValueError("size_mm, dpi and cells must be positive")
    cell_px = round(size_mm / MM_PER_INCH * dpi / cells)
    if cell_px < 8:
        raise ValueError(f"{size_mm} mm at {dpi} dpi gives {cell_px} px per cell; use a larger tag")
    tag_px = cell_px * cells
    margin = cell_px * quiet_cells
    return SheetPlan(
        size_mm=size_mm,
        dpi=dpi,
        cells=cells,
        quiet_cells=quiet_cells,
        caption_px=caption_px,
        cell_px=cell_px,
        tag_px=tag_px,
        canvas_w=tag_px + 2 * margin,
        canvas_h=tag_px + 2 * margin + caption_px,
        actual_mm=tag_px / dpi * MM_PER_INCH,
    )


def caption_lines(tag_id: int, plan: SheetPlan, family: str = "tag36h11") -> list[str]:
    return [
        f"AprilTag {family}   id {tag_id}",
        f"black square {plan.actual_mm:.1f} mm  ->  rooms.json tag_size_m: {plan.tag_size_m:.4f}",
        "THIS WAY UP  -  print at 100% (no fit-to-page), then measure the black square",
    ]


def output_stem(tag_id: int, plan: SheetPlan, family: str = "tag36h11") -> str:
    return f"{family}_{tag_id:03d}_{round(plan.size_mm)}mm"


def render_marker(tag_id: int, tag_px: int, family: str = "tag36h11") -> Any:
    """uint8 (tag_px, tag_px) marker image, 0 = black, 255 = white (cv2 imported lazily)."""
    import cv2  # lazy: optional dependency

    dict_name, cells, count = FAMILIES[family]
    if not 0 <= tag_id < count:
        raise ValueError(f"{family} has ids 0..{count - 1}, got {tag_id}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_px, borderBits=1)
    if marker.shape != (tag_px, tag_px):
        raise RuntimeError(f"unexpected marker shape {marker.shape}")
    return marker


def _font(size: int) -> Any:
    from PIL import ImageFont  # lazy

    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def compose_sheet(marker: Any, plan: SheetPlan, tag_id: int, family: str = "tag36h11") -> Any:
    """White canvas with the marker inside its quiet zone and the caption band below."""
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = np.full((plan.canvas_h, plan.canvas_w), 255, dtype=np.uint8)
    m = plan.margin_px
    canvas[m : m + plan.tag_px, m : m + plan.tag_px] = marker
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    top = 2 * m + plan.tag_px  # caption band starts here
    band = plan.caption_px
    big, small = max(24, band // 5), max(16, band // 9)
    lines = caption_lines(tag_id, plan, family)
    # up arrow at the left of the band, pointing at the tag
    ax, ay = m + big // 2, top + band // 2
    half = big // 2
    draw.polygon(
        [(ax, ay - half - half // 2), (ax - half, ay - half // 4), (ax + half, ay - half // 4)],
        fill=0,
    )
    draw.rectangle([ax - half // 4, ay - half // 4, ax + half // 4, ay + half + half // 2], fill=0)
    tx = ax + big + small
    draw.text((tx, top + band * 0.08), lines[0], fill=0, font=_font(big))
    draw.text((tx, top + band * 0.45), lines[1], fill=0, font=_font(small))
    draw.text((tx, top + band * 0.70), lines[2], fill=0, font=_font(small))
    return img


def write_sheet(
    img: Any, out_dir: Path, tag_id: int, plan: SheetPlan, family: str = "tag36h11", *, pdf: bool
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(tag_id, plan, family)
    png = out_dir / f"{stem}.png"
    img.save(png, dpi=(plan.dpi, plan.dpi))
    pdf_path: Path | None = None
    if pdf:
        pdf_path = out_dir / f"{stem}.pdf"
        img.save(pdf_path, resolution=plan.dpi)
    return png, pdf_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_tag_sheet.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3], help="tag ids (default 0-3)")
    p.add_argument("--size-mm", type=float, default=160.0, help="black square edge (default 160)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--family", choices=sorted(FAMILIES), default="tag36h11")
    p.add_argument("--quiet-cells", type=int, default=1, help="white border in cells (default 1)")
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help=f"output dir (default {DEFAULT_OUT})"
    )
    p.add_argument("--no-pdf", action="store_true", help="PNG only")
    p.add_argument(
        "--pdf-all", action="store_true", help="also write one multi-page tags_all.pdf (needs PDF)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = FAMILIES[args.family][1]
    plan = plan_sheet(args.size_mm, args.dpi, cells=cells, quiet_cells=args.quiet_cells)
    try:
        images = []
        for tag_id in args.ids:
            marker = render_marker(tag_id, plan.tag_px, args.family)
            img = compose_sheet(marker, plan, tag_id, args.family)
            png, pdf = write_sheet(img, args.out, tag_id, plan, args.family, pdf=not args.no_pdf)
            images.append(img)
            w_mm, h_mm = plan.sheet_mm
            print(
                f"id {tag_id:3d}: {plan.tag_px} px black square = {plan.actual_mm:.2f} mm "
                f"(cell {plan.cell_px} px), sheet {w_mm:.0f} x {h_mm:.0f} mm -> {png}"
                + (f", {pdf.name}" if pdf else "")
            )
        if args.pdf_all and not args.no_pdf and images:
            all_pdf = args.out / "tags_all.pdf"
            images[0].save(all_pdf, save_all=True, append_images=images[1:], resolution=plan.dpi)
            print(f"multi-page: {all_pdf}")
    except ImportError as e:
        print(
            f"error: {e}. Needs opencv-python-headless and Pillow "
            f"(pip install -r deploy/handheld/requirements-dev.txt)",
            file=sys.stderr,
        )
        return 2
    print(
        f'rooms.json: "tag_size_m": {plan.tag_size_m:.4f}   '
        "(measure the printed square and use the measured value)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
