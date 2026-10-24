#!/usr/bin/env python3
"""Headset voice console for the handheld demo (task T11).

Exercise the press -> record -> STT -> answer -> TTS -> play loop without the
camera, or check the audio plumbing on the Mac and on the Pi.

Examples::

    # what PortAudio / PipeWire / evdev can see
    python deploy/handheld/scripts/voice_console.py --list-devices

    # text in, answer from the running server (POST /api/query), spoken answer to a file
    python deploy/handheld/scripts/voice_console.py --loopback-text "where is my mug" \
        --tts-out /tmp/answer.wav

    # a recorded question through local Whisper (WHISPER_LOCAL=1), no server needed
    python deploy/handheld/scripts/voice_console.py --loopback-wav question.wav --in-process

    # the real thing on this machine: Enter to talk, Enter to stop, one turn
    python deploy/handheld/scripts/voice_console.py --ptt keyboard --turns 1

Exit code 0 when an answer was produced (or a listing was printed), 2 otherwise,
130 on Ctrl-C without an answer.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_spatial.app import headset_voice as hv  # noqa: E402

log = logging.getLogger("voice_console")


# ---------------------------------------------------------------------------
# Answer functions
# ---------------------------------------------------------------------------


def answer_fn_remote(host: str, port: int, timeout_s: float = 10.0) -> Callable[[str], str]:
    """Ask a running server through ``POST /api/query`` (text-first, no TTS)."""
    url = f"http://{host}:{port}/api/query"

    def _ask(text: str) -> str:
        body = json.dumps({"text": text, "tts": False}).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310 - http to the user's own server
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer:
            raise RuntimeError(f"{url} returned no answer: {payload!r}")
        return answer

    return _ask


def answer_fn_canned() -> Callable[[str], str]:
    """Echo the parsed intent; used when neither a server nor the in-process path is up."""
    try:
        from demo_spatial.app.voice_pipeline import parse_intent
    except Exception:  # pragma: no cover - voice_pipeline always exists in this repo
        parse_intent = None  # type: ignore[assignment]

    def _ask(text: str) -> str:
        if parse_intent is None:
            return f"Loopback console: I heard '{text}'."
        intent = parse_intent(text)
        what = f" about your {intent.object_name}" if intent.object_name else ""
        return (
            f"Loopback console: I heard a {intent.kind.replace('_', ' ')} question{what}. "
            "Start the server or pass --in-process for a real answer."
        )

    return _ask


def answer_fn_in_process() -> Callable[[str], str] | None:
    """A SimDetectionSource-fed SpatialMemory answered by VoicePipeline, when available.

    Depends on the T2/T4/T7 modules; any missing piece returns ``None`` so the caller
    can fall back to the canned answer.
    """
    env = {
        "LOCI_DETECTION_SOURCE": "sim",
        "LOCI_SIM_REALTIME": "0",
        "LOCI_SIM_ORACLE_POSE": "1",
        "LOCI_PERSIST": "0",
        "LOCI_LABEL_EMBEDDER": os.environ.get("LOCI_LABEL_EMBEDDER", "hash"),
    }
    for key, value in env.items():
        os.environ.setdefault(key, value)
    try:
        from demo_spatial.app.scene_ingestion import SceneIngestion
        from demo_spatial.app.spatial_memory import SpatialMemory
        from demo_spatial.app.voice_pipeline import VoicePipeline

        # T7's module may not exist yet in this tree; importlib keeps ruff's import
        # grouping stable either way.
        handheld = importlib.import_module("demo_spatial.app.handheld")
    except Exception as e:
        log.warning("in-process memory unavailable (%s); using the canned answer", e)
        return None
    try:
        cfg = handheld.HandheldConfig.from_env()
        memory = SpatialMemory(epoch_size_ms=5000)
        ingestion = SceneIngestion(memory, None, use_vlm_fallback=False)
        runtime, localizer = handheld.build_handheld(memory, ingestion, cfg)
        if runtime is None:
            log.warning("build_handheld returned no runtime; using the canned answer")
            return None
        if hasattr(memory, "set_vocabulary"):
            memory.set_vocabulary(runtime.source.classes)
        frames = runtime.run_until_done()
        log.info("in-process simulator ran %d detection frames", frames)
        try:
            voice = VoicePipeline(
                memory, None, pose_provider=lambda: localizer.current() if localizer else None
            )
        except TypeError:
            voice = VoicePipeline(memory, None)
    except Exception as e:
        log.warning("in-process memory failed to start (%s); using the canned answer", e)
        return None

    def _ask(text: str) -> str:
        try:
            result = hv.run_maybe_async(voice.handle_text_query(text, synthesize_response=False))
        except TypeError:
            result = hv.run_maybe_async(voice.handle_text_query(text))
        return str(result.answer_text)

    return _ask


# ---------------------------------------------------------------------------
# STT / TTS / audio helpers (module-level so tests can monkeypatch them)
# ---------------------------------------------------------------------------


class FixedSTT:
    """STT that returns a given text (loopback without a microphone or Whisper)."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        return self.text


class NullTTS:
    async def synthesize(self, text: str) -> bytes:
        return b""


class NullAudio:
    name = "null"

    def record(self, max_seconds: float, stop: threading.Event | None = None) -> bytes:
        return hv.silence_wav(0.0)

    def play(self, wav_bytes: bytes) -> None:
        return None

    def list_devices(self) -> list[dict]:
        return []


class CaptureAudio:
    """Wrap an AudioIO: copy everything played to ``path`` (and still play when asked)."""

    def __init__(self, inner: Any, path: Path | None, play_through: bool) -> None:
        self._inner = inner
        self._path = path
        self._play_through = play_through
        self.name = f"capture({getattr(inner, 'name', type(inner).__name__)})"
        self.written: list[Path] = []

    def record(self, max_seconds: float, stop: threading.Event | None = None) -> bytes:
        return self._inner.record(max_seconds, stop)

    def play(self, wav_bytes: bytes) -> None:
        if self._path is not None and wav_bytes and not self._is_earcon(wav_bytes):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_bytes(wav_bytes)
            self.written.append(self._path)
        if self._play_through:
            self._inner.play(wav_bytes)

    def list_devices(self) -> list[dict]:
        return self._inner.list_devices()

    @staticmethod
    def _is_earcon(wav_bytes: bytes) -> bool:
        return wav_bytes[:4] == b"RIFF" and hv.wav_duration_s(wav_bytes) <= hv.EARCON_S + 1e-6


class NullPTT:
    name = "none"

    def wait_for_press(self, stop: threading.Event) -> bool:
        return False

    def wait_for_release(self, timeout_s: float) -> None:
        return None


def build_stt(args: argparse.Namespace) -> Any:
    if args.loopback_text:
        return FixedSTT(args.loopback_text)
    from demo_spatial.app.voice_pipeline import WhisperSTT

    stt = WhisperSTT()
    if not stt.is_available:
        log.warning("no STT backend: set WHISPER_LOCAL=1 (faster-whisper) or OPENAI_API_KEY")
    return stt


def build_tts(args: argparse.Namespace) -> Any:
    if args.no_tts:
        return NullTTS()
    from demo_spatial.app.voice_pipeline import TTS

    return TTS()


def build_audio(args: argparse.Namespace) -> Any:
    if args.loopback_text or args.loopback_wav:
        inner: Any = NullAudio()
        if args.play:
            inner = hv.make_audio_from_env(os.environ)
        return CaptureAudio(inner, Path(args.tts_out) if args.tts_out else None, bool(args.play))
    inner = hv.make_audio_from_env(os.environ)
    if args.tts_out:
        return CaptureAudio(inner, Path(args.tts_out), True)
    return inner


def choose_answer_fn(args: argparse.Namespace) -> Callable[[str], str]:
    if args.canned:
        return answer_fn_canned()
    if args.in_process:
        return answer_fn_in_process() or answer_fn_canned()
    remote = answer_fn_remote(args.host, args.port, args.timeout)
    canned = answer_fn_canned()

    def _ask(text: str) -> str:
        try:
            return remote(text)
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
            log.warning(
                "server http://%s:%d unreachable (%s); answering from the canned path",
                args.host,
                args.port,
                e,
            )
            return canned(text)

    return _ask


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def print_event(ev: dict) -> None:
    lat = ev.get("latency_ms")
    lat_s = f"{lat:8.1f} ms" if isinstance(lat, (int, float)) else " " * 11
    text = ev.get("text") or ""
    print(f"[{lat_s}] {ev['type']:<15} {text}", flush=True)


def cmd_list_devices(args: argparse.Namespace) -> int:
    print(f"platform: {sys.platform}; sounddevice: {hv.sounddevice_available()}")
    backend = args.audio or os.environ.get("LOCI_AUDIO_BACKEND") or "auto"
    try:
        audio = hv.make_audio_from_env(os.environ, backend=backend)
        devices = audio.list_devices()
    except Exception as e:
        print(f"audio backend {backend!r} failed: {e}")
        devices = []
        audio = None
    name = getattr(audio, "name", backend)
    print(f"\naudio devices ({name}):")
    if not devices:
        print("  (none listed)")
    for d in devices:
        kinds = []
        if int(d.get("max_input_channels", 0) or 0) > 0:
            kinds.append("in")
        if int(d.get("max_output_channels", 0) or 0) > 0:
            kinds.append("out")
        rate = d.get("default_samplerate")
        rate_s = f" {int(rate)} Hz" if rate else ""
        print(f"  [{d.get('index', '-'):>3}] {d.get('name', '')}  ({'/'.join(kinds)}{rate_s})")
    inputs = hv.list_input_devices()
    print("\ninput (button) devices from /proc/bus/input/devices:")
    if not inputs:
        print("  (none; not Linux or no readable /proc/bus/input/devices)")
    for d in inputs:
        print(f"  {d['event'] or '-':<22} {d['name']}")
    print(
        "\nhint: LOCI_AUDIO_IN / LOCI_AUDIO_OUT take a case-insensitive name substring; "
        "LOCI_PTT_DEVICE takes a button device name substring or /dev/input/eventN."
    )
    return 0


def cmd_loopback(args: argparse.Namespace) -> int:
    answer_fn = choose_answer_fn(args)
    stt = build_stt(args)
    tts = build_tts(args)
    audio = build_audio(args)
    if args.loopback_wav:
        wav = Path(args.loopback_wav).read_bytes()
        if wav[:4] != b"RIFF":
            log.warning(
                "%s does not look like a WAV file; Whisper may still decode it", args.loopback_wav
            )
    else:
        wav = hv.silence_wav(0.1)
    loop = hv.HeadsetVoiceLoop(
        answer_fn, stt, tts, audio, NullPTT(), earcon=False, on_event=print_event
    )
    result = loop.handle_once(wav)
    summary = {k: v for k, v in result.items() if k != "audio"}
    if args.tts_out:
        written = getattr(audio, "written", [])
        summary["tts_out"] = str(written[-1]) if written else None
        if not written:
            log.warning("no TTS audio produced, %s not written", args.tts_out)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result.get("answer") else 2


def cmd_live(args: argparse.Namespace) -> int:
    os.environ["LOCI_VOICE"] = "headset"
    answer_fn = choose_answer_fn(args)
    stt = build_stt(args)
    tts = build_tts(args)
    audio = build_audio(args)
    stop = threading.Event()
    done = {"n": 0}

    def _on_event(ev: dict) -> None:
        print_event(ev)
        if ev["type"] == "done":
            done["n"] += 1
            if args.turns and done["n"] >= args.turns:
                stop.set()

    loop = hv.make_voice_loop_from_env(
        answer_fn, stt, tts, env=os.environ, audio=audio, on_event=_on_event
    )
    if loop is None:  # pragma: no cover - LOCI_VOICE is forced above
        print("voice loop disabled", file=sys.stderr)
        return 2

    def _sigterm(_signum: int, _frame: Any) -> None:
        stop.set()

    with_signals = threading.current_thread() is threading.main_thread()
    if with_signals:
        signal.signal(signal.SIGTERM, _sigterm)
    print(
        f"voice loop: {json.dumps({k: v for k, v in loop.status().items() if k != 'last_events'})}"
    )
    try:
        loop.run(stop)
    except KeyboardInterrupt:
        stop.set()
        print("\ninterrupted", file=sys.stderr)
        return 0 if done["n"] else 130
    return 0 if done["n"] else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def apply_env_overrides(args: argparse.Namespace) -> None:
    pairs = {
        "LOCI_PTT": args.ptt,
        "LOCI_AUDIO_BACKEND": args.audio,
        "LOCI_AUDIO_IN": args.audio_in,
        "LOCI_AUDIO_OUT": args.audio_out,
        "LOCI_PTT_DEVICE": args.ptt_device,
        "LOCI_VOICE_MAX_S": None if args.max_seconds is None else str(args.max_seconds),
        "LOCI_VAD_THRESHOLD": None if args.vad_threshold is None else str(args.vad_threshold),
        "LOCI_VAD_SILENCE_S": None if args.vad_silence is None else str(args.vad_silence),
    }
    for key, value in pairs.items():
        if value is not None and value != "":
            os.environ[key] = value
    if args.toggle:
        os.environ["LOCI_PTT_TOGGLE"] = "1"
    if args.no_earcon:
        os.environ["LOCI_EARCON"] = "0"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice_console.py",
        description="Headset voice loop console (list devices, loopback tests, live turns).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::", 1)[-1] if __doc__ else None,
    )
    p.add_argument("--list-devices", action="store_true", help="print audio and button devices")
    p.add_argument("--ptt", choices=["keyboard", "evdev", "vad"], help="push-to-talk source")
    p.add_argument("--audio", choices=["auto", "sounddevice", "cli"], help="audio backend")
    p.add_argument("--audio-in", help="input device name substring (LOCI_AUDIO_IN)")
    p.add_argument("--audio-out", help="output device name substring (LOCI_AUDIO_OUT)")
    p.add_argument("--ptt-device", help="button device name substring or /dev/input/eventN")
    p.add_argument("--toggle", action="store_true", help="click starts, next click stops")
    p.add_argument("--vad-threshold", type=float, help="VAD RMS threshold (int16 scale)")
    p.add_argument("--vad-silence", type=float, help="VAD trailing silence in seconds")
    p.add_argument("--max-seconds", type=float, help="longest recording (LOCI_VOICE_MAX_S)")
    p.add_argument("--no-earcon", action="store_true", help="skip the confirmation tone")
    p.add_argument("--loopback-wav", metavar="FILE", help="run one turn on a recorded question")
    p.add_argument("--loopback-text", metavar="TEXT", help="run one turn on typed text (skip STT)")
    p.add_argument("--in-process", action="store_true", help="answer from a sim-fed memory")
    p.add_argument("--canned", action="store_true", help="answer with a canned echo (no server)")
    p.add_argument("--host", default=os.environ.get("LOCI_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))
    p.add_argument("--timeout", type=float, default=10.0, help="server request timeout (s)")
    p.add_argument("--tts-out", metavar="FILE", help="write the spoken answer to this file")
    p.add_argument("--no-tts", action="store_true", help="skip speech synthesis")
    p.add_argument("--play", action="store_true", help="also play audio in loopback modes")
    p.add_argument("--turns", type=int, default=0, help="stop live mode after N answers (0 = run)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    apply_env_overrides(args)
    t0 = time.perf_counter()
    if args.list_devices:
        return cmd_list_devices(args)
    if args.loopback_wav and args.loopback_text:
        print("use either --loopback-wav or --loopback-text", file=sys.stderr)
        return 2
    loopback = bool(args.loopback_wav or args.loopback_text)
    code = cmd_loopback(args) if loopback else cmd_live(args)
    log.debug("finished in %.1f s with exit code %d", time.perf_counter() - t0, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
