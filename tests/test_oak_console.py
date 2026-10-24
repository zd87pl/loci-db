"""Tests for deploy/handheld/scripts/oak_console.py (T1): simulator and fakes only, no camera."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import types
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from demo_spatial.app.detection_source import (
    DetectionFrame,
    JsonlReplaySource,
    SourceStatus,
    SpatialDetection,
)
from demo_spatial.app.oak_source import OakDetectionSource, PipelineHandles

REPO = Path(__file__).resolve().parents[1]
CONSOLE = REPO / "deploy" / "handheld" / "scripts" / "oak_console.py"

DET_LINE = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+(?P<label>\S+)\s+\d\.\d{2}\s+"
    r"(?P<xyz>x=[+-]\d+\.\d{2}m y=[+-]\d+\.\d{2}m z=\d+\.\d{2}m|\(no depth\))\s+"
    r"box=\(\d\.\d{2},\d\.\d{2},\d\.\d{2},\d\.\d{2}\)$"
)
SUMMARY_LINE = re.compile(
    r"^fps=\d+\.\d det/s=\d+\.\d tags/s=\d+\.\d labels=\{[^}]*\} state=\S+ usb=\S+ reconnects=\d+"
)
DONE_LINE = re.compile(
    r"^done: frames=(?P<frames>\d+) detections=(?P<dets>\d+) tag_frames=(?P<tags>\d+)"
)


@pytest.fixture(scope="module")
def console() -> Any:
    spec = importlib.util.spec_from_file_location("oak_console_under_test", CONSOLE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses with postponed annotations resolve names via sys.modules[cls.__module__]
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def clean_oak_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(("LOCI_OAK_", "LOCI_TAG_")):
            monkeypatch.delenv(key, raising=False)


def _run_main(console: Any, argv: list[str], **kwargs: Any) -> tuple[int, str]:
    buf = io.StringIO()
    rc = console.main(argv, out=buf, **kwargs)
    return rc, buf.getvalue()


def _lines(text: str, pattern: re.Pattern[str]) -> list[re.Match[str]]:
    return [m for m in (pattern.match(line) for line in text.splitlines()) if m]


def _subprocess(*args: str, cwd: Path, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.run(
        [sys.executable, str(CONSOLE), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
        env=env,
    )


# ---------------------------------------------------------------------------
# Fakes for the OAK path (no depthai): the injected opener returns un-started handles
# ---------------------------------------------------------------------------


class FakeQueue:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def tryGet(self) -> Any:  # noqa: N802 - depthai spelling
        return self._items.pop(0) if self._items else None


class FakePipeline:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def isRunning(self) -> bool:  # noqa: N802
        return True


class FakeDevice:
    def __init__(self) -> None:
        self.closed = 0

    def getUsbSpeed(self) -> Any:  # noqa: N802
        return types.SimpleNamespace(name="HIGH")

    def getDeviceName(self) -> str:  # noqa: N802
        return "OAK-D-LITE"

    def close(self) -> None:
        self.closed += 1


def _det(label: int, conf: float, box: tuple[float, ...], xyz: tuple[float, float, float]) -> Any:
    xmin, ymin, xmax, ymax = box
    return types.SimpleNamespace(
        label=label,
        confidence=conf,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        spatialCoordinates=types.SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
    )


def _msg(ts_s: float, *dets: Any) -> Any:
    return types.SimpleNamespace(
        detections=list(dets), getTimestamp=lambda: timedelta(seconds=ts_s)
    )


def _fake_handles(messages: list[Any]) -> PipelineHandles:
    return PipelineHandles(
        pipeline=FakePipeline(),
        device=FakeDevice(),
        q_det=FakeQueue(messages),
        q_tag=None,
        labels=["person", "bicycle", "car", "cup"],
        tag_size=None,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_format_detection_line_with_and_without_depth(console: Any) -> None:
    ts = 1_700_000_000_123
    with_depth = SpatialDetection("cup", 41, 0.91, 0.10, 0.20, 0.30, 0.40, (0.12, -0.05, 1.84))
    line = console.format_detection_line(ts, with_depth)
    assert DET_LINE.match(line), line
    assert "x=+0.12m y=-0.05m z=1.84m" in line
    assert "box=(0.10,0.20,0.30,0.40)" in line
    tail = ".123  cup          0.91  x=+0.12m y=-0.05m z=1.84m  box=(0.10,0.20,0.30,0.40)"
    assert line.endswith(tail)

    no_depth = SpatialDetection("person", 0, 0.60, 0.5, 0.5, 0.9, 0.9, None)
    line = console.format_detection_line(ts, no_depth)
    assert DET_LINE.match(line), line
    assert "(no depth)" in line and " x=" not in line and " z=" not in line


def test_format_summary_fields_and_label_order(console: Any) -> None:
    status = SourceStatus("running", 10.07, 123, 0, 0.05, "HIGH", "OAK-D-LITE", None)
    labels = Counter({"cup": 3, "cell phone": 1, "bottle": 3})
    line = console.format_summary(status, 4.04, labels, 5.0)
    assert SUMMARY_LINE.match(line), line
    assert line == (
        "fps=10.1 det/s=4.0 tags/s=5.0 labels={bottle:3,cup:3,cell_phone:1} "
        "state=running usb=HIGH reconnects=0"
    )
    waiting = SourceStatus("waiting_for_camera", 0.0, 0, 2, None, None, None, "X_LINK_ERROR")
    line = console.format_summary(waiting, 0.0, Counter())
    assert SUMMARY_LINE.match(line), line
    assert "labels={} state=waiting_for_camera usb=- reconnects=2" in line
    assert line.endswith(' error="X_LINK_ERROR"')


def test_format_clock_uses_milliseconds(console: Any) -> None:
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}\.042", console.format_clock(1_700_000_000_042))


# ---------------------------------------------------------------------------
# Y-sign verdict
# ---------------------------------------------------------------------------


def test_y_sign_verdict_rdf_vs_flipped(console: Any) -> None:
    rdf_pairs = [(0.20, -0.30), (0.25, -0.20), (0.80, 0.40), (0.90, 0.50), (0.50, 0.01)]
    v = console.y_sign_verdict(rdf_pairs)
    assert v.verdict == "rdf" and v.message == console.RDF_CONFIRMED
    assert v.message == "RDF confirmed (above-centre objects have y<0)"
    assert (v.n_above, v.n_below) == (2, 2)  # the cy=0.5 sample is in neither band
    assert v.mean_above == pytest.approx(-0.25) and v.mean_below == pytest.approx(0.45)
    assert "n_above=2" in v.stats_line() and "mean_y_below=+0.45" in v.stats_line()

    flipped = [(cy, -y) for cy, y in rdf_pairs]
    v = console.y_sign_verdict(flipped)
    assert v.verdict == "up" and v.message == "Y appears UP: set LOCI_OAK_AXES=ruf"

    # already running with the flip applied and still wrong: advise removing it, not adding it
    v = console.y_sign_verdict(flipped, axes="ruf")
    assert v.verdict == "up" and "LOCI_OAK_AXES=rdf" in v.message and "=ruf" not in v.message[-20:]


def test_y_sign_verdict_single_band_and_tilt(console: Any) -> None:
    # the step-1 check: a cup held above the lens centre, nothing below
    assert console.y_sign_verdict([(0.10, -0.40), (0.30, -0.20)]).verdict == "rdf"
    assert console.y_sign_verdict([(0.10, 0.40)]).verdict == "up"
    # only objects below the centre (the simulator's desk scene)
    assert console.y_sign_verdict([(0.9, 0.5), (0.8, 0.3)]).verdict == "rdf"
    assert console.y_sign_verdict([(0.9, -0.5)]).verdict == "up"
    # tilted camera: every y positive, but above-centre still smaller than below-centre
    v = console.y_sign_verdict([(0.2, 0.10), (0.8, 0.90)])
    assert v.verdict == "rdf"
    # nothing usable
    for pairs in ([], [(0.5, 0.1), (0.45, -0.1)]):
        v = console.y_sign_verdict(pairs)
        assert v.verdict == "inconclusive" and v.message.startswith("Y sign inconclusive")
        assert (v.n_above, v.n_below, v.mean_above, v.mean_below) == (0, 0, None, None)


# ---------------------------------------------------------------------------
# Simulator through the CLI
# ---------------------------------------------------------------------------


def test_sim_two_seconds_subprocess_prints_detections_and_summary(tmp_path: Path) -> None:
    t0 = time.monotonic()
    proc = _subprocess("--sim", "--seconds", "2", cwd=tmp_path, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - t0 < 20.0
    out = proc.stdout
    assert out.splitlines()[0].startswith("source=sim scenario=mug_two_rooms")
    dets = _lines(out, DET_LINE)
    assert len(dets) >= 1, out
    assert {m.group("label") for m in dets} <= {"cup", "bottle"}
    assert all(m.group("xyz").startswith("x=") for m in dets), "sim boxes always carry depth"
    summaries = _lines(out, SUMMARY_LINE)
    assert len(summaries) >= 1, out
    assert any("state=running" in s.group(0) or "state=stopped" in s.group(0) for s in summaries)
    assert "cup:" in summaries[-1].group(0)
    done = _lines(out, DONE_LINE)
    assert len(done) == 1 and 10 <= int(done[0].group("frames")) <= 30
    assert "state=stopped" in out.splitlines()[-1]


def test_sim_fast_runs_to_completion(console: Any) -> None:
    rc, out = _run_main(console, ["--sim", "--no-realtime"])
    assert rc == 0, out
    assert "state=done" in out
    done = _lines(out, DONE_LINE)[0]
    assert int(done.group("frames")) == 180  # 18 s scenario at 10 fps
    assert int(done.group("tags")) > 0
    assert "cup" in out and "laptop" in out  # final labels= covers both rooms


def test_rate_limits_printed_frames(console: Any) -> None:
    rc, out = _run_main(console, ["--sim", "--no-realtime", "--rate", "1"])
    assert rc == 0
    # the whole scenario runs in milliseconds: at 1 Hz only the first frame is printed
    dets = _lines(out, DET_LINE)
    assert [m.group("label") for m in dets] == ["cup", "bottle"]

    rc, out = _run_main(console, ["--sim", "--no-realtime", "--rate", "0"])
    assert rc == 0
    total = int(_lines(out, DONE_LINE)[0].group("dets"))
    assert total > 100 and len(_lines(out, DET_LINE)) == total  # rate 0 = every detection


def test_check_y_sign_with_sim(console: Any) -> None:
    rc, out = _run_main(console, ["--sim", "--no-realtime", "--check-y-sign"])
    assert rc == 0
    # the scripted objects sit below the camera: positive camera y, boxes in the lower band
    stats = r"^y-sign check: n_above=0 mean_y_above=n/a n_below=\d+ mean_y_below=\+"
    assert re.search(stats, out, re.M)
    verdict = out.rstrip().splitlines()[-1]
    assert verdict == "verdict: RDF confirmed (above-centre objects have y<0)"


def test_seed_and_scenario_flags_and_dry_run(console: Any) -> None:
    rc, out = _run_main(console, ["--sim", "--seconds", "0", "--seed", "7", "--fps", "5"])
    assert rc == 0
    assert out.splitlines()[0] == (
        "source=sim scenario=mug_two_rooms fps=5 frames_total=90 realtime=True seed=7"
    )
    assert "dry run" in out
    rc, _ = _run_main(console, ["--sim", "--seconds", "-1"])
    assert rc == 2
    with pytest.raises(SystemExit):  # argparse rejects unknown axes
        console.main(["--axes", "xyz", "--seconds", "0"])


def test_unknown_scenario_is_a_clean_error(console: Any, capsys: Any) -> None:
    rc, _ = _run_main(console, ["--sim", "--scenario", "no_such_scenario"])
    assert rc == 2
    assert "unknown scenario" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Record / replay
# ---------------------------------------------------------------------------


def test_record_then_replay(console: Any, tmp_path: Path) -> None:
    rec = tmp_path / "sub" / "step1.jsonl"
    rc, out = _run_main(console, ["--sim", "--no-realtime", "--record", str(rec)])
    assert rc == 0 and rec.is_file()
    assert f"recording -> {rec}" in out
    frames = int(_lines(out, DONE_LINE)[0].group("frames"))
    tags = int(_lines(out, DONE_LINE)[0].group("tags"))
    n_lines = len(rec.read_text().splitlines())
    assert n_lines == frames + tags
    assert f"recorded {n_lines} lines -> {rec}" in out

    replay = JsonlReplaySource(rec)
    replay.start()
    det_frames = [ev for ev in replay.events() if isinstance(ev, DetectionFrame)]
    assert len(det_frames) == frames
    assert det_frames[0].detections[0].xyz_cam_m is not None

    rc, out = _run_main(console, ["--replay", str(rec), "--no-realtime"])
    assert rc == 0, out
    assert out.splitlines()[0].startswith("source=replay")
    assert int(_lines(out, DONE_LINE)[0].group("frames")) == frames
    assert "state=done" in out


def test_replay_missing_file_exit_2(console: Any, tmp_path: Path, capsys) -> None:
    rc, out = _run_main(console, ["--replay", str(tmp_path / "nope.jsonl")])
    assert rc == 2 and out == ""
    assert "replay file not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# OAK path (env-driven config, flag overrides, injected opener, dry run)
# ---------------------------------------------------------------------------


def test_build_source_oak_flags_override_env(
    console: Any, clean_oak_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCI_OAK_PP_SHAVES", "3")
    monkeypatch.setenv("LOCI_TAG_STREAM", "640x400@10")
    monkeypatch.setenv("LOCI_OAK_STEREO_LRC", "1")
    args = console.build_parser().parse_args([])
    src = console.build_source(args, opener=lambda: None)
    assert isinstance(src, OakDetectionSource)
    cfg = src.cfg
    assert (cfg.pp_shaves, cfg.tag_stream, cfg.lr_check) == (3, (640, 400, 10.0), True)
    assert (cfg.usb2, cfg.model, cfg.axes, cfg.fps) == (False, "yolov6-nano", "rdf", 10.0)

    args = console.build_parser().parse_args(
        ["--usb2", "--model", "/models/x.rvc2.tar.xz", "--no-tags", "--axes", "ruf", "--fps", "15"]
    )
    cfg = console.build_source(args, opener=lambda: None).cfg
    assert cfg.usb2 is True and cfg.model == "/models/x.rvc2.tar.xz"
    assert cfg.tag_stream is None and cfg.axes == "ruf" and cfg.fps == 15.0
    assert cfg.pp_shaves == 3  # untouched env values survive the overrides


def test_oak_dry_run_never_calls_opener(console: Any, clean_oak_env: None) -> None:
    calls: list[int] = []

    def opener() -> PipelineHandles:
        calls.append(1)
        raise AssertionError("the device must not be opened with --seconds 0")

    rc, out = _run_main(console, ["--seconds", "0", "--usb2", "--no-tags"], opener=opener)
    assert rc == 0 and not calls
    first, second = out.splitlines()[:2]
    assert first.startswith("source=oak config=")
    cfg = json.loads(first.split("config=", 1)[1])
    assert cfg["usb2"] is True and cfg["tag_stream"] is None
    assert cfg["stereo_preset"] == "FAST_DENSITY" and cfg["pp_shaves"] == 2
    assert second.startswith("dry run:")


def test_oak_dry_run_with_real_depthai_does_not_open_device(
    console: Any, clean_oak_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    dai = pytest.importorskip("depthai")

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("dai.Device()/dai.Pipeline() must not be called")

    monkeypatch.setattr(dai, "Device", boom)
    monkeypatch.setattr(dai, "Pipeline", boom)
    rc, out = _run_main(console, ["--seconds", "0", "--model", "yolov6-nano"])
    assert rc == 0 and out.startswith("source=oak config=")
    # the same with the injected opener: still a dry run
    rc, _ = _run_main(console, ["--seconds", "0"], opener=boom)
    assert rc == 0


@pytest.mark.parametrize(
    ("y_mm", "expected"),
    [(-300.0, "RDF confirmed (above-centre objects have y<0)"), (300.0, "Y appears UP")],
)
def test_oak_path_streams_from_injected_opener(
    console: Any, clean_oak_env: None, y_mm: float, expected: str
) -> None:
    # a cup above the image centre (cy 0.2) with depth, a person without depth (exact zeros)
    msgs = [
        _msg(
            10.0 + 0.1 * i,
            _det(3, 0.90, (0.1, 0.1, 0.3, 0.3), (120.0, y_mm, 1500.0)),
            _det(0, 0.70, (0.4, 0.5, 0.8, 0.95), (0.0, 0.0, 0.0)),
        )
        for i in range(5)
    ]
    handles = _fake_handles(msgs)
    rc, out = _run_main(
        console,
        ["--seconds", "0.5", "--rate", "0", "--check-y-sign"],
        opener=lambda: handles,
    )
    assert rc == 0, out
    assert out.splitlines()[0].startswith("source=oak config=")
    dets = _lines(out, DET_LINE)
    cups = [m for m in dets if m.group("label") == "cup"]
    persons = [m for m in dets if m.group("label") == "person"]
    assert len(cups) == 5 and len(persons) == 5
    assert all(m.group("xyz") == f"x=+0.12m y={y_mm / 1000:+.2f}m z=1.50m" for m in cups)
    assert all(m.group("xyz") == "(no depth)" for m in persons), "no XYZ for zero coordinates"
    summary = _lines(out, SUMMARY_LINE)[-1].group(0)
    assert "usb=HIGH" in summary and "reconnects=0" in summary and "state=stopped" in summary
    done = _lines(out, DONE_LINE)[0]
    assert (int(done.group("frames")), int(done.group("dets"))) == (5, 10)
    assert "y-sign check: n_above=5 " in out and "n_below=0" in out
    assert out.rstrip().splitlines()[-1].startswith(f"verdict: {expected}")
    assert handles.pipeline.started == 1 and handles.pipeline.stopped == 1
    assert handles.device.closed == 1


def test_exit_code_2_without_any_frame(console: Any, clean_oak_env: None) -> None:
    handles = _fake_handles([])
    rc, out = _run_main(console, ["--seconds", "0.3"], opener=lambda: handles)
    assert rc == 2
    assert "done: frames=0 detections=0" in out and "state=stopped" in out
    assert not _lines(out, DET_LINE)


def test_oak_records_tag_frames_too(console: Any, clean_oak_env: None, tmp_path: Path) -> None:
    class _Transformation:
        def getIntrinsicMatrix(self) -> list[list[float]]:  # noqa: N802
            return [[918.3, 0.0, 645.8], [0.0, 917.5, 417.4], [0.0, 0.0, 1.0]]

        def getDistortionCoefficients(self) -> list[float]:  # noqa: N802
            return [-0.77, -9.38, 0.0006, 0.0004, 31.35, -0.93, -8.75, 30.42]

        def getSize(self) -> tuple[int, int]:  # noqa: N802
            return (16, 8)

    class _Img:
        def __init__(self) -> None:
            self._raw = np.zeros((12, 16), dtype=np.uint8)  # NV12 of a 16x8 frame

        def getFrame(self) -> Any:  # noqa: N802
            return self._raw

        def getTransformation(self) -> Any:  # noqa: N802
            return _Transformation()

        def getWidth(self) -> int:  # noqa: N802
            return 16

        def getHeight(self) -> int:  # noqa: N802
            return 8

        def getTimestamp(self) -> timedelta:  # noqa: N802
            return timedelta(seconds=10.05)

    handles = _fake_handles([_msg(10.0, _det(3, 0.9, (0.1, 0.1, 0.3, 0.3), (0.0, 0.0, 1500.0)))])
    handles.q_tag = FakeQueue([_Img()])
    handles.tag_size = (16, 8)
    rec = tmp_path / "oak.jsonl"
    rc, out = _run_main(console, ["--seconds", "0.3", "--record", str(rec)], opener=lambda: handles)
    assert rc == 0, out
    kinds = [json.loads(line)["t"] for line in rec.read_text().splitlines()]
    assert sorted(kinds) == ["det", "tag"]
    assert "tag_frames=1" in out
    tag = next(json.loads(line) for line in rec.read_text().splitlines() if '"t":"tag"' in line)
    assert tag["width"] == 16 and tag["height"] == 8 and "gray" not in tag


# ---------------------------------------------------------------------------
# Process-level behaviour: import path, help, SIGINT
# ---------------------------------------------------------------------------


def test_help_runs_from_any_cwd(tmp_path: Path) -> None:
    proc = _subprocess("--help", cwd=tmp_path, timeout=30)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--sim", "--seconds", "--rate", "--record", "--check-y-sign", "--usb2"):
        assert flag in proc.stdout


def test_import_path_never_touches_loci_cv2_or_depthai(tmp_path: Path) -> None:
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('oak_console', {str(CONSOLE)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "rc = mod.main(['--sim', '--seconds', '0'])\n"
        "rc2 = mod.main(['--seconds', '0'])\n"
        "bad = sorted(m for m in sys.modules if m == 'loci' or m.startswith('loci.')"
        " or m.split('.')[0] in ('cv2', 'depthai'))\n"
        "print('RC', rc, rc2)\n"
        "print('BAD', bad)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tmp_path),
        env={**os.environ, "LOCI_DETECTION_SOURCE": "oak"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC 0 0" in proc.stdout and "BAD []" in proc.stdout, proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigint_stops_cleanly_with_exit_zero(tmp_path: Path) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(CONSOLE), "--sim", "--seconds", "30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        time.sleep(1.5)
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, err
    assert "stopping" in err
    assert len(_lines(out, DET_LINE)) >= 1
    assert "state=stopped" in out.splitlines()[-1]
    assert int(_lines(out, DONE_LINE)[0].group("frames")) < 60, "stopped long before --seconds"
