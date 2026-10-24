#!/usr/bin/env python3
"""Build-order step 1 console: live OAK-D detections with XYZ printed to the terminal.

The first thing to run on new hardware (design section 14, step 1).  It talks to
the camera through ``demo_spatial.app.oak_source`` (DepthAI v3, the live-verified
pipeline) or to the scripted simulator, prints one line per detection, a one-second
health summary, and with ``--check-y-sign`` a verdict on the sign of the DepthAI
y axis.  Nothing on its import path touches ``loci``; ``depthai`` is imported only
when a real camera is opened.

Examples::

    # software sanity, no camera: the scripted two-room scenario in real time
    python deploy/handheld/scripts/oak_console.py --sim --seconds 5

    # real camera for 30 s, env-driven OakConfig (LOCI_OAK_*, LOCI_TAG_STREAM)
    python deploy/handheld/scripts/oak_console.py --seconds 30

    # hold a cup above the lens centre; prints "RDF confirmed" or the LOCI_OAK_AXES advice
    python deploy/handheld/scripts/oak_console.py --seconds 20 --check-y-sign

    # record 60 s of real detections for hardware-free replay tests
    python deploy/handheld/scripts/oak_console.py --seconds 60 \
        --record /home/ziggy/loci-handheld/data/step1.jsonl
    python deploy/handheld/scripts/oak_console.py --replay step1.jsonl --no-realtime

    # construct the OAK source from the environment without opening the device
    python deploy/handheld/scripts/oak_console.py --seconds 0

Output format::

    HH:MM:SS.mmm  label  conf  x=+0.12m y=-0.05m z=1.84m  box=(xmin,ymin,xmax,ymax)
    fps=10.1 det/s=4.0 tags/s=5.0 labels={cup:3,bottle:1} state=running usb=HIGH reconnects=0

Exit code 0 when at least one detection frame was received (or the sim/replay ran
to completion), 2 otherwise.  ``--seconds 0`` is a dry run and exits 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_spatial.app.detection_source import (  # noqa: E402
    SCENARIOS,
    DetectionFrame,
    DetectionSource,
    JsonlRecorder,
    JsonlReplaySource,
    SimDetectionSource,
    SourceStatus,
    SpatialDetection,
    TagFrame,
)
from demo_spatial.app.oak_source import OakConfig, OakDetectionSource  # noqa: E402

log = logging.getLogger("oak_console")

RDF_CONFIRMED = "RDF confirmed (above-centre objects have y<0)"
Y_APPEARS_UP = "Y appears UP: set LOCI_OAK_AXES=ruf"
Y_APPEARS_UP_WITH_RUF = (
    "Y appears UP with LOCI_OAK_AXES=ruf: remove the override (set LOCI_OAK_AXES=rdf)"
)
Y_INCONCLUSIVE = (
    "Y sign inconclusive: no boxes with depth away from the image centre "
    "(need cy<0.4 or cy>0.6; hold a cup above the lens centre)"
)


# ---------------------------------------------------------------------------
# Formatting (pure functions, unit-tested)
# ---------------------------------------------------------------------------


def format_clock(timestamp_ms: int) -> str:
    """Local wall-clock ``HH:MM:SS.mmm`` of a Unix millisecond timestamp."""
    ts = int(timestamp_ms)
    return datetime.fromtimestamp(ts / 1000.0).strftime("%H:%M:%S") + f".{ts % 1000:03d}"


def format_detection_line(timestamp_ms: int, det: SpatialDetection) -> str:
    """One console line per detection; boxes without depth print no XYZ."""
    box = f"box=({det.xmin:.2f},{det.ymin:.2f},{det.xmax:.2f},{det.ymax:.2f})"
    if det.xyz_cam_m is None:
        xyz = "(no depth)"
    else:
        x, y, z = det.xyz_cam_m
        xyz = f"x={x:+.2f}m y={y:+.2f}m z={z:.2f}m"
    return f"{format_clock(timestamp_ms)}  {det.label:<12s} {det.confidence:.2f}  {xyz}  {box}"


def format_labels(labels: Counter[str]) -> str:
    """``{cup:3,bottle:1}`` sorted by count desc then name; spaces become underscores."""
    items = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))
    return "{" + ",".join(f"{k.replace(' ', '_')}:{v}" for k, v in items) + "}"


def format_summary(
    status: SourceStatus,
    det_per_s: float,
    labels: Counter[str],
    tags_per_s: float = 0.0,
) -> str:
    """The one-second health line of design section 4.3."""
    usb = status.usb_speed if status.usb_speed else "-"
    line = (
        f"fps={status.fps:.1f} det/s={det_per_s:.1f} tags/s={tags_per_s:.1f} "
        f"labels={format_labels(labels)} state={status.state} usb={usb} "
        f"reconnects={status.reconnects}"
    )
    if status.error:
        line += f" error={json.dumps(status.error)}"
    return line


# ---------------------------------------------------------------------------
# Y-sign verdict (design D4 / section 14 step 1.3)
# ---------------------------------------------------------------------------


@dataclass
class YSignVerdict:
    verdict: str  # "rdf" | "up" | "inconclusive"
    message: str
    n_above: int
    n_below: int
    mean_above: float | None
    mean_below: float | None

    def stats_line(self) -> str:
        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:+.2f}"

        return (
            f"y-sign check: n_above={self.n_above} mean_y_above={fmt(self.mean_above)} "
            f"n_below={self.n_below} mean_y_below={fmt(self.mean_below)}"
        )


def y_sign_verdict(
    pairs: Sequence[tuple[float, float]],
    axes: str = "rdf",
    *,
    above_cy: float = 0.4,
    below_cy: float = 0.6,
) -> YSignVerdict:
    """Decide whether camera-frame y points down from ``(cy, y_m)`` pairs.

    Boxes above the image centre (``cy < above_cy``) must have a smaller y than boxes
    below it (``cy > below_cy``) when the frame is RDF.  With only one band present
    the sign of that band decides (above-centre: y < 0; below-centre: y > 0), which
    is what the "cup above the lens centre" check produces.
    """
    above = [float(y) for cy, y in pairs if cy < above_cy]
    below = [float(y) for cy, y in pairs if cy > below_cy]
    mean_above = sum(above) / len(above) if above else None
    mean_below = sum(below) / len(below) if below else None
    if mean_above is not None and mean_below is not None:
        rdf = mean_above < mean_below
    elif mean_above is not None:
        rdf = mean_above < 0.0
    elif mean_below is not None:
        rdf = mean_below > 0.0
    else:
        return YSignVerdict("inconclusive", Y_INCONCLUSIVE, 0, 0, None, None)
    if rdf:
        message = RDF_CONFIRMED
        if axes.lower() != "rdf":
            message += f" [with LOCI_OAK_AXES={axes.lower()} applied; keep it]"
        verdict = "rdf"
    else:
        message = Y_APPEARS_UP if axes.lower() == "rdf" else Y_APPEARS_UP_WITH_RUF
        verdict = "up"
    return YSignVerdict(verdict, message, len(above), len(below), mean_above, mean_below)


# ---------------------------------------------------------------------------
# Counters shared between the event loop and the summary thread
# ---------------------------------------------------------------------------


@dataclass
class ConsoleStats:
    frames: int = 0
    detections: int = 0
    tag_frames: int = 0
    labels_total: Counter[str] = field(default_factory=Counter)
    y_pairs: list[tuple[float, float]] = field(default_factory=list)
    _interval_dets: int = 0
    _interval_tags: int = 0
    _interval_labels: Counter[str] = field(default_factory=Counter)
    _interval_t0: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def note_frame(self, frame: DetectionFrame, collect_y: bool = False) -> None:
        with self._lock:
            self.frames += 1
            n = len(frame.detections)
            self.detections += n
            self._interval_dets += n
            for det in frame.detections:
                self.labels_total[det.label] += 1
                self._interval_labels[det.label] += 1
                if collect_y and det.xyz_cam_m is not None:
                    self.y_pairs.append((det.cy, det.xyz_cam_m[1]))

    def note_tag_frame(self) -> None:
        with self._lock:
            self.tag_frames += 1
            self._interval_tags += 1

    def interval_snapshot(self, now: float | None = None) -> tuple[float, float, Counter[str]]:
        """``(det/s, tags/s, labels)`` over the interval since the previous snapshot."""
        now = time.monotonic() if now is None else now
        with self._lock:
            dt = max(1e-3, now - self._interval_t0)
            labels = Counter(self._interval_labels)
            out = (self._interval_dets / dt, self._interval_tags / dt, labels)
            self._interval_dets = 0
            self._interval_tags = 0
            self._interval_labels = Counter()
            self._interval_t0 = now
            return out


class SummaryPrinter(threading.Thread):
    """Prints ``format_summary`` every ``interval`` seconds, even while no events arrive."""

    def __init__(
        self, source: DetectionSource, stats: ConsoleStats, out: TextIO, interval: float = 1.0
    ) -> None:
        super().__init__(name="oak-console-summary", daemon=True)
        self._source = source
        self._stats = stats
        self._out = out
        self._interval = float(interval)
        # not named ``_stop``: threading.Thread owns a private ``_stop()`` used by join()
        self._halt = threading.Event()
        self.lines = 0

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.wait(self._interval):
            try:
                write_line(self._out, summary_line(self._source, self._stats))
                self.lines += 1
            except Exception as exc:  # never kill the console over a status glitch
                log.debug("summary failed: %s", exc)


def summary_line(source: DetectionSource, stats: ConsoleStats) -> str:
    det_per_s, tags_per_s, labels = stats.interval_snapshot()
    return format_summary(source.status(), det_per_s, labels, tags_per_s)


def write_line(out: TextIO, text: str) -> None:
    out.write(text + "\n")
    out.flush()


# ---------------------------------------------------------------------------
# Source construction
# ---------------------------------------------------------------------------


def build_source(
    args: argparse.Namespace, *, opener: Callable[[], Any] | None = None
) -> DetectionSource:
    """``--sim`` -> SimDetectionSource, ``--replay`` -> JsonlReplaySource, else the OAK source."""
    realtime = not args.no_realtime
    if args.sim:
        return SimDetectionSource(
            args.scenario,
            fps=args.fps if args.fps else 10.0,
            realtime=realtime,
            seed=args.seed,
        )
    if args.replay:
        path = Path(args.replay)
        if not path.is_file():
            raise OSError(f"replay file not found: {path}")
        return JsonlReplaySource(path, realtime=realtime)
    cfg = OakConfig.from_env()
    if args.usb2:
        cfg.usb2 = True
    if args.model:
        cfg.model = args.model
    if args.no_tags:
        cfg.tag_stream = None
    if args.axes:
        cfg.axes = args.axes
    if args.fps:
        cfg.fps = float(args.fps)
    return OakDetectionSource(cfg, opener=opener)


def banner(source: DetectionSource, args: argparse.Namespace) -> str:
    if isinstance(source, OakDetectionSource):
        return f"source=oak config={json.dumps(source.cfg.to_dict(), sort_keys=True)}"
    if isinstance(source, SimDetectionSource):
        return (
            f"source=sim scenario={source.scenario.name} fps={source.fps:g} "
            f"frames_total={source.frames_total} realtime={source.realtime} seed={source.seed}"
        )
    if isinstance(source, JsonlReplaySource):
        return f"source=replay path={source.path} realtime={source.realtime}"
    return f"source={getattr(source, 'name', type(source).__name__)}"


# ---------------------------------------------------------------------------
# The event loop
# ---------------------------------------------------------------------------


def _install_signal_handlers(stop_all: Callable[[], None]) -> dict[int, Any]:
    """First SIGINT/SIGTERM stops the source cleanly; a second SIGINT aborts."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}
    hits = {"n": 0}

    def handler(signum: int, _frame: Any) -> None:
        hits["n"] += 1
        if hits["n"] == 1 or signum != signal.SIGINT:
            sys.stderr.write("\nstopping (Ctrl-C again to abort)\n")
            sys.stderr.flush()
            stop_all()
            return
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.signal(signum, handler)
        except (ValueError, OSError) as exc:  # pragma: no cover - exotic hosts
            log.debug("cannot install handler for %s: %s", signum, exc)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, old in previous.items():
        try:
            signal.signal(signum, old if old is not None else signal.SIG_DFL)
        except (ValueError, OSError, TypeError) as exc:  # pragma: no cover
            log.debug("cannot restore handler for %s: %s", signum, exc)


def make_recording_tagger(out: TextIO) -> Callable[[TagFrame], None] | None:
    """Detect AprilTags on recorded tag frames so the JSONL replays with ``tags``.

    The recorder never serialises the luma plane, so without this a recording of the
    real camera could not anchor on replay (design §4.14).  Returns ``None`` (and says
    so) when neither pyapriltags nor cv2 is importable.
    """
    try:
        from demo_spatial.app.room_anchor import make_tag_detector

        detector = make_tag_detector("auto")
    except Exception as exc:  # the console must keep working without the anchoring stack
        log.debug("tag detector unavailable: %s", exc)
        detector = None
    if detector is None:
        write_line(
            out, "note: no AprilTag detector (pyapriltags/cv2): recorded tag frames carry no tags"
        )
        return None
    write_line(out, f"tagging recorded tag frames with {detector.name}")
    failures = {"n": 0}

    def tag(tf: TagFrame) -> None:
        if tf.tags is not None or tf.gray is None:
            return
        try:
            tf.tags = list(detector.detect(tf.gray))
        except Exception as exc:
            failures["n"] += 1
            if failures["n"] <= 3:
                log.warning("tag detection failed on a recorded frame: %s", exc)
            tf.tags = None

    return tag


def run_console(
    source: DetectionSource,
    args: argparse.Namespace,
    *,
    out: TextIO,
    recorder: JsonlRecorder | None = None,
    tagger: Callable[[TagFrame], None] | None = None,
) -> int:
    """Stream ``source.events()`` to ``out`` until done, ``--seconds`` or SIGINT.

    Returns the process exit code (0 when a detection frame arrived or the source
    ran to completion, 2 otherwise).  ``tagger`` (see :func:`make_recording_tagger`)
    fills ``TagFrame.tags`` before a frame is recorded.
    """
    stats = ConsoleStats()
    stop = threading.Event()

    def stop_all() -> None:
        if not stop.is_set():
            stop.set()
            source.stop()

    timer: threading.Timer | None = None
    if args.seconds is not None and args.seconds > 0:
        timer = threading.Timer(float(args.seconds), stop_all)
        timer.daemon = True
    printer = SummaryPrinter(source, stats, out, interval=1.0)
    min_gap = 1.0 / args.rate if args.rate and args.rate > 0 else 0.0
    last_print = -math.inf
    t0 = time.monotonic()
    previous = _install_signal_handlers(stop_all)
    source.start()
    gen = source.events()
    try:
        if timer is not None:
            timer.start()
        printer.start()
        for ev in gen:
            if tagger is not None and isinstance(ev, TagFrame):
                tagger(ev)
            if recorder is not None:
                recorder.write(ev)
            if isinstance(ev, TagFrame):
                stats.note_tag_frame()
            elif isinstance(ev, DetectionFrame):
                stats.note_frame(ev, collect_y=args.check_y_sign)
                now = time.monotonic()
                if ev.detections and now - last_print >= min_gap:
                    last_print = now
                    for det in ev.detections:
                        write_line(out, format_detection_line(ev.timestamp_ms, det))
            if stop.is_set():
                break
    except KeyboardInterrupt:
        write_line(sys.stderr, "interrupted")
    finally:
        stop_all()
        if timer is not None:
            timer.cancel()
        printer.stop()
        gen.close()
        printer.join(timeout=2.0)
        if recorder is not None:
            recorder.close()
        _restore_signal_handlers(previous)

    status = source.status()
    elapsed = time.monotonic() - t0
    # The final summary covers the whole run (not the last 1 Hz interval, which the
    # printer thread may have just emptied), so its labels/rates are deterministic.
    run_s = max(1e-3, elapsed)
    write_line(
        out,
        format_summary(
            status,
            stats.detections / run_s,
            stats.labels_total,
            stats.tag_frames / run_s,
        ),
    )
    write_line(
        out,
        f"done: frames={stats.frames} detections={stats.detections} tag_frames={stats.tag_frames} "
        f"labels={format_labels(stats.labels_total)} elapsed={elapsed:.1f}s state={status.state}",
    )
    if recorder is not None:
        write_line(out, f"recorded {recorder.lines} lines -> {recorder.path}")
    if args.check_y_sign:
        axes = source.cfg.axes if isinstance(source, OakDetectionSource) else "rdf"
        verdict = y_sign_verdict(stats.y_pairs, axes)
        write_line(out, verdict.stats_line())
        write_line(out, f"verdict: {verdict.message}")
    return 0 if stats.frames >= 1 or status.state == "done" else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oak_console.py",
        description="Step-1 console: live OAK-D detections with XYZ (or the simulator).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::", 1)[-1] if __doc__ else None,
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=None,
        metavar="N",
        help="stop after N seconds (default: until done / Ctrl-C); 0 = construct only, dry run",
    )
    p.add_argument("--sim", action="store_true", help="scripted simulator instead of the camera")
    p.add_argument(
        "--scenario",
        default="mug_two_rooms",
        help=f"sim scenario name or Scenario JSON path (known: {', '.join(sorted(SCENARIOS))})",
    )
    p.add_argument("--replay", metavar="FILE", help="play a --record JSONL file instead")
    p.add_argument(
        "--no-realtime", action="store_true", help="sim/replay as fast as possible (tests)"
    )
    p.add_argument("--seed", type=int, default=0, help="simulator noise seed")
    p.add_argument("--usb2", action="store_true", help="open the device at USB 2 (HIGH) speed")
    p.add_argument("--model", metavar="NAME|PATH", help="zoo model name or .rvc2.tar.xz archive")
    p.add_argument("--no-tags", action="store_true", help="disable the AprilTag RGB stream")
    p.add_argument("--axes", choices=["rdf", "ruf"], help="override LOCI_OAK_AXES")
    p.add_argument("--fps", type=float, help="camera / simulator frame rate")
    p.add_argument("--record", metavar="FILE", help="append every event to this JSONL file")
    p.add_argument(
        "--no-tag-detect",
        action="store_true",
        help="with --record: do not run the AprilTag detector on recorded tag frames",
    )
    p.add_argument(
        "--check-y-sign",
        action="store_true",
        help="collect (cy, y) pairs and print the RDF/UP verdict at the end",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=2.0,
        metavar="HZ",
        help="print detection frames at most this often (0 = every frame)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging on stderr")
    return p


def main(
    argv: Sequence[str] | None = None,
    *,
    opener: Callable[[], Any] | None = None,
    out: TextIO | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    out = sys.stdout if out is None else out
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.seconds is not None and args.seconds < 0:
        write_line(sys.stderr, "error: --seconds must be >= 0")
        return 2
    try:
        source = build_source(args, opener=opener)
    except (ValueError, OSError) as exc:
        write_line(sys.stderr, f"error: {exc}")
        return 2
    write_line(out, banner(source, args))
    if args.seconds is not None and args.seconds == 0:
        write_line(out, "dry run: source constructed, device not opened (omit --seconds to stream)")
        return 0
    recorder = JsonlRecorder(Path(args.record)) if args.record else None
    tagger = None
    if recorder is not None:
        write_line(out, f"recording -> {recorder.path}")
        if not args.no_tag_detect:
            tagger = make_recording_tagger(out)
    return run_console(source, args, out=out, recorder=recorder, tagger=tagger)


if __name__ == "__main__":
    sys.exit(main())
