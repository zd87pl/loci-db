"""Tests for deploy/handheld/scripts/make_tag_sheet.py (T8).

The layout maths and the CLI ``--help`` run everywhere; rendering needs cv2 + Pillow and is
skipped where they are missing (the Mac system Python has neither; the scratch venv has both).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "deploy" / "handheld" / "scripts" / "make_tag_sheet.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_tag_sheet", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve postponed annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- layout maths (no cv2)


def test_plan_sheet_160mm_at_300dpi_matches_the_verified_recipe() -> None:
    mts = _load()
    plan = mts.plan_sheet(160.0, 300)
    assert plan.cell_px == 236
    assert plan.tag_px == 1888
    assert plan.canvas_w == 2360  # 1888 + two 236 px quiet-zone cells
    assert plan.canvas_h == 2360 + plan.caption_px
    assert plan.actual_mm == pytest.approx(159.85, abs=0.01)
    assert plan.tag_size_m == pytest.approx(0.15985, abs=1e-5)
    w_mm, h_mm = plan.sheet_mm
    assert w_mm == pytest.approx(199.8, abs=0.1)
    assert h_mm > w_mm  # caption band below the tag


@pytest.mark.parametrize(
    ("size_mm", "dpi", "cell_px"),
    [(160.0, 300, 236), (240.0, 300, 354), (160.0, 600, 472), (100.0, 300, 148)],
)
def test_plan_sheet_rounds_cells_to_whole_pixels(size_mm: float, dpi: int, cell_px: int) -> None:
    mts = _load()
    plan = mts.plan_sheet(size_mm, dpi)
    assert plan.cell_px == cell_px
    assert plan.tag_px == 8 * cell_px
    # the printed size is reported from the rounded pixel count, not the requested millimetres
    assert plan.actual_mm == pytest.approx(plan.tag_px / dpi * 25.4)


def test_plan_sheet_rejects_unprintable_sizes() -> None:
    mts = _load()
    with pytest.raises(ValueError):
        mts.plan_sheet(0.0, 300)
    with pytest.raises(ValueError):
        mts.plan_sheet(5.0, 72)  # < 8 px per cell


def test_caption_mentions_id_measured_size_and_orientation() -> None:
    mts = _load()
    plan = mts.plan_sheet(160.0, 300)
    lines = mts.caption_lines(7, plan)
    text = " ".join(lines)
    assert "id 7" in text
    assert "159.9 mm" in text
    assert "tag_size_m: 0.1599" in text
    assert "THIS WAY UP" in text
    assert mts.output_stem(7, plan) == "tag36h11_007_160mm"


def test_help_runs_without_cv2_or_pillow() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    assert "--size-mm" in proc.stdout
    assert "--ids" in proc.stdout


# ---------------------------------------------------------------- rendering (cv2 + Pillow)


def _largest_black_component(gray) -> tuple[int, int, int, int]:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    inv = (gray < 128).astype(np.uint8)
    _n, _labels, stats, _c = cv2.connectedComponentsWithStats(inv, connectivity=4)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    return (
        int(stats[k, cv2.CC_STAT_LEFT]),
        int(stats[k, cv2.CC_STAT_TOP]),
        int(stats[k, cv2.CC_STAT_WIDTH]),
        int(stats[k, cv2.CC_STAT_HEIGHT]),
    )


def test_generates_png_2360_wide_at_300dpi_with_1888px_black_square(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    from PIL import Image

    mts = _load()
    rc = mts.main(["--ids", "0", "1", "--out", str(tmp_path)])
    assert rc == 0
    for tag_id in (0, 1):
        png = tmp_path / f"tag36h11_{tag_id:03d}_160mm.png"
        pdf = tmp_path / f"tag36h11_{tag_id:03d}_160mm.pdf"
        assert png.is_file() and pdf.is_file()
        with Image.open(png) as im:
            assert im.size[0] == 2360
            assert im.size[1] > 2360
            dpi = im.info.get("dpi")
            assert dpi is not None
            assert dpi[0] == pytest.approx(300, abs=0.5)
            assert dpi[1] == pytest.approx(300, abs=0.5)
            gray = np.array(im.convert("L"))
        x, y, w, h = _largest_black_component(gray)
        assert (w, h) == (1888, 1888)
        assert (x, y) == (236, 236)  # one quiet-zone cell from the top-left corner
        # quiet zone is white all around the square
        assert gray[: 236 - 1, :].min() == 255
        assert gray[:, : 236 - 1].min() == 255
        assert gray[:, 236 + 1888 + 1 :].min() == 255
    assert (tmp_path / "tags_all.pdf").exists() is False  # only with --pdf-all


def test_marker_round_trips_through_the_aruco_detector(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")

    mts = _load()
    plan = mts.plan_sheet(160.0, 300)
    for tag_id in (0, 1, 42):
        marker = mts.render_marker(tag_id, plan.tag_px)
        assert marker.shape == (1888, 1888)
        assert marker.dtype == np.uint8
        sheet = np.array(mts.compose_sheet(marker, plan, tag_id).convert("L"))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _rejected = detector.detectMarkers(sheet)
        assert ids is not None and ids.ravel().tolist() == [tag_id]
        tl, tr, br, bl = corners[0][0]
        # corner 0 is the top-left of the printed square when the caption is at the bottom
        assert tl[0] == pytest.approx(236, abs=2) and tl[1] == pytest.approx(236, abs=2)
        assert tr[0] == pytest.approx(236 + 1888 - 1, abs=2) and tr[1] == pytest.approx(236, abs=2)
        assert br[1] == pytest.approx(236 + 1888 - 1, abs=2)
        assert bl[0] == pytest.approx(236, abs=2)


def test_pdf_all_and_other_sizes(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from PIL import Image

    mts = _load()
    rc = mts.main(["--ids", "3", "--size-mm", "240", "--out", str(tmp_path), "--pdf-all"])
    assert rc == 0
    png = tmp_path / "tag36h11_003_240mm.png"
    assert png.is_file()
    with Image.open(png) as im:
        assert im.size[0] == 354 * 10
    assert (tmp_path / "tags_all.pdf").is_file()


def test_render_marker_rejects_unknown_id() -> None:
    pytest.importorskip("cv2")
    mts = _load()
    with pytest.raises(ValueError):
        mts.render_marker(587, 1888)
