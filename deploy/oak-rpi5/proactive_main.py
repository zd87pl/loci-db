#!/usr/bin/env python3
"""Proactive assistant main loop — RPi5 + OAK-D Lite.

Single entry point that integrates:
  - OAK-D Lite spatial detection pipeline
  - Hazard detection engine (ProactiveEngine)
  - Proactive voice output with mode switching (ProactiveVoice)
  - Spatial memory for "where are my keys?" reactive queries
  - Continuous microphone STT for voice commands
  - Power/performance telemetry logging

Usage:
    python proactive_main.py [--sim] [--mode guide|reactive|quiet]
                             [--blob /path/to/model.blob]
                             [--power-log power_log.csv]
                             [--fps 30]

Environment variables (optional):
    OPENAI_API_KEY   — enables Whisper cloud STT + OpenAI TTS
    TTS_ENGINE       — "openai" (default) | "piper" | "edge"
    WHISPER_LOCAL    — "1" to force local faster-whisper
    PIPER_MODEL_PATH — path to .onnx Piper model (TTS_ENGINE=piper)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or this script's directory
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_APP_DIR = Path(__file__).parent / "app"

for _p in (_REPO_ROOT, _APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Imports — local modules (sys.path-based; oak-rpi5 dir has a hyphen)
# ---------------------------------------------------------------------------

from oak_pipeline import (  # type: ignore[import]  # noqa: E402
    OakFrame,
    OakPipeline,
    OakSimPipeline,
    OakUnavailable,
)
from proactive_engine import ProactiveEngine  # noqa: E402
from demo_spatial.app.proactive_voice import (  # noqa: E402
    ProactiveVoice,
    VoiceMode,
)
from demo_spatial.app.spatial_memory import SpatialMemory  # noqa: E402
from demo_spatial.app.voice_pipeline import TTS, VoicePipeline, WhisperSTT  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOCI proactive assistant for RPi5 + OAK-D Lite"
    )
    p.add_argument(
        "--sim",
        action="store_true",
        help="Use synthetic detections instead of OAK-D Lite hardware",
    )
    p.add_argument(
        "--mode",
        choices=["guide", "reactive", "quiet"],
        default="reactive",
        help="Initial voice mode (default: reactive)",
    )
    p.add_argument(
        "--blob",
        default=str(Path(__file__).parent / "models" / "yolov8n_oak.blob"),
        help="Path to YOLOv8n DepthAI blob",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target camera FPS (default: 30)",
    )
    p.add_argument(
        "--power-log",
        default="",
        help="CSV file to log power/performance metrics (empty = disabled)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Power measurement — RPi5 via /sys or vcgencmd
# ---------------------------------------------------------------------------


def _read_rpi5_power_mw() -> float | None:
    """Return estimated power draw in milliwatts on RPi5, or None."""
    # RPi5 exposes current via the PMIC sysfs node
    candidates = [
        "/sys/devices/platform/rp1/rp1-power/power1_input",
        "/sys/class/power_supply/BAT0/power_now",
        "/sys/class/power_supply/ac/power_now",
    ]
    for path in candidates:
        try:
            val = int(Path(path).read_text().strip())
            # power_now is in microwatts for most nodes
            return val / 1000.0
        except (FileNotFoundError, ValueError, PermissionError):
            continue
    return None


def _read_rss_mb() -> float:
    """Return RSS memory usage of this process in megabytes."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Performance tracker
# ---------------------------------------------------------------------------


class PerfTracker:
    """Tracks FPS and alert latency (frame → speech-start)."""

    def __init__(self) -> None:
        self._frame_ts: dict[int, float] = {}  # frame_id → wall time
        self._latencies_ms: list[float] = []
        self._fps_window: list[float] = []

    def record_frame(self, frame_id: int) -> None:
        now = time.perf_counter()
        self._fps_window.append(now)
        if len(self._fps_window) > 60:
            self._fps_window.pop(0)
        self._frame_ts[frame_id] = now
        # Keep only the last 200 frames to bound memory
        if len(self._frame_ts) > 200:
            oldest = min(self._frame_ts)
            del self._frame_ts[oldest]

    def record_alert_spoken(self, frame_id: int) -> None:
        ts = self._frame_ts.pop(frame_id, None)
        if ts is not None:
            latency_ms = (time.perf_counter() - ts) * 1000
            self._latencies_ms.append(latency_ms)
            if len(self._latencies_ms) > 1000:
                self._latencies_ms.pop(0)

    @property
    def fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    @property
    def p50_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        s = sorted(self._latencies_ms)
        return s[len(s) // 2]

    @property
    def p95_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        s = sorted(self._latencies_ms)
        return s[int(len(s) * 0.95)]


# ---------------------------------------------------------------------------
# Main async application
# ---------------------------------------------------------------------------


class ProactiveApp:
    """Wires together the OAK pipeline, engine, voice, and STT loops."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._shutdown = asyncio.Event()
        self._perf = PerfTracker()

        # Core components
        self._memory = SpatialMemory(epoch_size_ms=5000)
        self._tts = TTS()
        self._stt = WhisperSTT()
        self._engine = ProactiveEngine()
        self._voice = ProactiveVoice(
            self._tts,
            memory=self._memory,
            mode=VoiceMode(args.mode),
        )
        self._voice_pipeline = VoicePipeline(self._memory)

        # Camera pipeline (constructed here; started in run())
        if args.sim:
            self._camera: OakPipeline | OakSimPipeline = OakSimPipeline(
                fps=float(args.fps)
            )
            logger.info("Using simulated OAK-D Lite pipeline")
        else:
            try:
                self._camera = OakPipeline(
                    model_blob_path=args.blob,
                    target_fps=args.fps,
                )
            except OakUnavailable as e:
                logger.error("%s", e)
                logger.info("Falling back to sim pipeline")
                self._camera = OakSimPipeline(fps=float(args.fps))

        # Power/perf CSV logger
        self._csv_path = args.power_log or ""
        self._csv_file = None
        self._csv_writer = None

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all coroutines and block until shutdown signal."""
        self._camera.start()

        if self._csv_path:
            self._csv_file = open(self._csv_path, "w", newline="")  # noqa: SIM115
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["timestamp_s", "fps", "rss_mb", "power_mw", "p50_latency_ms", "mode"]
            )
            logger.info("Power/perf log → %s", self._csv_path)

        tasks = [
            asyncio.create_task(self._camera_loop(), name="camera_loop"),
            asyncio.create_task(self._voice.run(), name="voice_output"),
            asyncio.create_task(self._stt_loop(), name="stt_loop"),
            asyncio.create_task(self._telemetry_loop(), name="telemetry"),
        ]

        logger.info(
            "Proactive assistant ready — mode: %s, TTS: %s, STT: %s",
            self._voice.mode.value,
            self._tts.engine,
            "whisper" if self._stt.is_available else "unavailable",
        )

        # Announce startup
        startup_text = {
            VoiceMode.GUIDE:    "Guide mode active. I'll warn you about obstacles.",
            VoiceMode.REACTIVE: "Ready. Ask me where something is or say guide me.",
            VoiceMode.QUIET:    "Quiet mode. Say guide me when you want alerts.",
        }[self._voice.mode]
        self._voice.enqueue_text(startup_text)

        try:
            await self._shutdown.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._camera.stop()
            if self._csv_file:
                self._csv_file.close()

    # ------------------------------------------------------------------
    # Camera processing loop
    # ------------------------------------------------------------------

    async def _camera_loop(self) -> None:
        """Poll the OAK pipeline in an executor thread; process each frame."""
        loop = asyncio.get_running_loop()
        while not self._shutdown.is_set():
            frame: OakFrame | None = await loop.run_in_executor(
                None, self._camera.next_frame, 50
            )
            if frame is None:
                continue

            self._perf.record_frame(frame.frame_id)

            alerts = self._engine.process(frame.detections, frame.depth_map)
            for alert in alerts:
                self._voice.enqueue_alert(alert)
                # Measure latency: record when we enqueue (speech starts soon)
                self._perf.record_alert_spoken(frame.frame_id)

    # ------------------------------------------------------------------
    # STT / microphone loop
    # ------------------------------------------------------------------

    async def _stt_loop(self) -> None:
        """Continuously listen to the microphone and handle voice commands.

        Falls back gracefully if no microphone or STT is unavailable.
        """
        if not self._stt.is_available:
            logger.info("STT unavailable — voice command input disabled")
            return

        try:
            import pyaudio  # noqa: F401 (optional dep)
        except ImportError:
            logger.info("pyaudio not installed — voice command input disabled")
            return

        logger.info("STT microphone loop started")
        loop = asyncio.get_running_loop()
        while not self._shutdown.is_set():
            try:
                audio = await loop.run_in_executor(None, _capture_audio_chunk_blocking)
                if not audio:
                    await asyncio.sleep(0.1)
                    continue

                self._voice.set_user_speaking(False)
                transcript = await self._stt.transcribe(audio)
                if not transcript.strip():
                    continue

                logger.debug("STT transcript: %r", transcript)

                # Try mode command first
                mode_response = await self._voice.handle_mode_command(transcript)
                if mode_response:
                    self._voice.enqueue_text(mode_response)
                    continue

                # Fall back to spatial-memory query ("where are my keys?")
                result = await self._voice_pipeline.handle_text_query(transcript)
                if result.answer_text:
                    self._voice.enqueue_text(result.answer_text)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("STT loop error")
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Telemetry loop
    # ------------------------------------------------------------------

    async def _telemetry_loop(self) -> None:
        """Log FPS, RSS, power, and latency every 10 seconds."""
        while not self._shutdown.is_set():
            await asyncio.sleep(10.0)
            fps = self._perf.fps
            rss_mb = _read_rss_mb()
            power_mw = _read_rpi5_power_mw()
            p50 = self._perf.p50_latency_ms
            mode = self._voice.mode.value

            logger.info(
                "perf fps=%.1f rss=%.0fMB power=%s p50_latency=%.0fms mode=%s",
                fps,
                rss_mb,
                f"{power_mw:.0f}mW" if power_mw else "n/a",
                p50,
                mode,
            )

            # Performance target warnings
            if fps > 0 and fps < 10.0:
                logger.warning("FPS %.1f below 10 FPS target — check VPU load", fps)
            if rss_mb > 1800:
                logger.warning("RSS %.0f MB approaching 2 GB limit", rss_mb)
            if p50 > 0 and p50 > 200:
                logger.warning("Alert latency p50=%.0fms exceeds 200ms target", p50)

            if self._csv_writer:
                self._csv_writer.writerow([
                    f"{time.time():.3f}",
                    f"{fps:.1f}",
                    f"{rss_mb:.1f}",
                    f"{power_mw:.0f}" if power_mw else "",
                    f"{p50:.0f}",
                    mode,
                ])
                if self._csv_file:
                    self._csv_file.flush()

    # ------------------------------------------------------------------
    # Shutdown hook
    # ------------------------------------------------------------------

    def request_shutdown(self) -> None:
        self._shutdown.set()


# ---------------------------------------------------------------------------
# Audio capture helper (blocking — runs in executor)
# ---------------------------------------------------------------------------


def _capture_audio_chunk_blocking(
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    chunk: int = 1024,
) -> bytes:
    """Record *duration_s* seconds of mono 16-bit audio from the default mic.

    Returns raw PCM bytes, or b"" on failure.
    """
    try:
        import io
        import wave

        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk,
        )

        frames = []
        n_chunks = int(sample_rate / chunk * duration_s)
        for _ in range(n_chunks):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        pa.terminate()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    except Exception as exc:
        logger.debug("Audio capture failed: %s", exc)
        return b""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    app = ProactiveApp(args)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler(*_) -> None:
        logger.info("Shutdown signal received")
        loop.call_soon_threadsafe(app.request_shutdown)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        loop.run_until_complete(app.run())
    finally:
        loop.close()
    logger.info("Proactive assistant stopped")


if __name__ == "__main__":
    main()
