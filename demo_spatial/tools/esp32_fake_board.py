"""Pretend to be the ESP32 thin client from a laptop: post frames, fetch and play answers.

Lets you exercise the whole host path (``/api/esp32/*``) without a board, or next to a
board to compare. Frames come from one image, a folder of images, or the laptop webcam
(``--camera 0``, needs OpenCV); answers are fetched as PCM and played with ``afplay`` on
macOS, ``aplay`` on Linux, or written to ``--out-dir``.

Examples::

    python demo_spatial/tools/esp32_fake_board.py --host http://localhost:8002 \
        --image path/to/kitchen.jpg --fps 2 --ask "where is my cup"

    python demo_spatial/tools/esp32_fake_board.py --host http://localhost:8002 --camera 0

Only the standard library plus httpx (already a FastAPI dependency) is required.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import httpx

DEFAULT_HOST = "http://localhost:8002"


def _wav_from_pcm(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def play_pcm(pcm: bytes, rate: int, out_dir: Path | None) -> None:
    """Play PCM through the OS player, or save it when no player is available."""
    wav_bytes = _wav_from_pcm(pcm, rate)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"answer-{int(time.time() * 1000)}.wav"
        path.write_bytes(wav_bytes)
        print(f"  saved {path}")
        return
    player = None
    if platform.system() == "Darwin" and shutil.which("afplay"):
        player = ["afplay"]
    elif shutil.which("aplay"):
        player = ["aplay", "-q"]
    if player is None:
        print("  no audio player found; use --out-dir to save answers")
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        subprocess.run([*player, tmp_path], check=False)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def frame_source(args: argparse.Namespace):
    """Yield JPEG bytes forever from the chosen source."""
    if args.camera is not None:
        try:
            import cv2
        except ImportError:
            sys.exit("--camera needs opencv-python (pip install opencv-python)")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            sys.exit(f"could not open camera {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if ok:
                yield enc.tobytes()
    paths: list[Path] = []
    for item in args.image:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg"}))
        else:
            paths.append(p)
    if not paths:
        sys.exit("no frames: pass --image <file-or-dir> or --camera <index>")
    blobs = [p.read_bytes() for p in paths]
    i = 0
    while True:
        yield blobs[i % len(blobs)]
        i += 1


def fetch_and_play(client: httpx.Client, args: argparse.Namespace, wait: float) -> bool:
    r = client.get(
        "/api/esp32/audio/next", params={"device": args.device, "wait": wait}, timeout=wait + 10
    )
    if r.status_code != 200:
        return False
    rate = int(r.headers.get("x-audio-rate", "16000"))
    secs = len(r.content) / 2 / rate
    print(f"  ♪ answer audio {secs:.1f}s @ {rate} Hz")
    if not args.mute:
        play_pcm(r.content, rate, Path(args.out_dir) if args.out_dir else None)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default=DEFAULT_HOST, help="demo server, e.g. http://localhost:8002")
    ap.add_argument("--device", default="fake-board", help="device id to report")
    ap.add_argument("--image", action="append", default=[], help="JPEG file or folder (repeatable)")
    ap.add_argument("--camera", type=int, default=None, help="webcam index (needs OpenCV)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second to post")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = forever)")
    ap.add_argument("--ask", action="append", default=[], help="typed question after warm-up")
    ap.add_argument("--ask-after", type=int, default=6, help="frames to post before asking")
    ap.add_argument("--speak-pcm", help="raw 16 kHz PCM16 file to send as a push-to-talk clip")
    ap.add_argument("--mute", action="store_true", help="do not play answers")
    ap.add_argument("--out-dir", help="save answer WAVs here instead of playing them")
    args = ap.parse_args(argv)

    frames = frame_source(args)
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    sent = 0
    asked = False
    with httpx.Client(base_url=args.host, timeout=30.0) as client:
        print(f"posting frames to {args.host} as '{args.device}' at {args.fps} fps (ctrl-c stops)")
        try:
            while args.frames <= 0 or sent < args.frames:
                t0 = time.monotonic()
                jpeg = next(frames)
                sent += 1
                r = client.post(
                    "/api/esp32/frame",
                    content=jpeg,
                    headers={
                        "Content-Type": "image/jpeg",
                        "X-Device-Id": args.device,
                        "X-Seq": str(sent),
                        "X-Device-Ms": str(int(time.monotonic() * 1000)),
                    },
                )
                if r.status_code != 200:
                    print(f"frame {sent}: HTTP {r.status_code} {r.text[:120]}")
                else:
                    body = r.json()
                    tag = "dropped" if body.get("dropped") else f"{body.get('infer_ms', 0):.0f} ms"
                    print(f"frame {sent}: {', '.join(body.get('labels', [])) or '-'} ({tag})")
                    if body.get("audio_pending"):
                        fetch_and_play(client, args, wait=0)
                    if body.get("next_ms") and args.fps <= 0:
                        period = body["next_ms"] / 1000.0
                if not asked and sent >= args.ask_after and (args.ask or args.speak_pcm):
                    asked = True
                    for q in args.ask:
                        print(f"asking: {q}")
                        resp = client.post(
                            "/api/esp32/ask", json={"text": q, "device": args.device}
                        ).json()
                        print(f"  → {resp.get('answer')}  [{resp.get('intent', {}).get('kind')}]")
                        if resp.get("audio_queued"):
                            fetch_and_play(client, args, wait=2)
                    if args.speak_pcm:
                        pcm = Path(args.speak_pcm).read_bytes()
                        print(f"sending push-to-talk clip ({len(pcm) / 32000:.1f}s)")
                        resp = client.post(
                            "/api/esp32/voice",
                            params={"device": args.device, "rate": 16000},
                            content=pcm,
                            headers={"Content-Type": "audio/L16; rate=16000"},
                        ).json()
                        print(f"  heard: {resp.get('transcription')!r} → {resp.get('answer')}")
                        if resp.get("audio_queued"):
                            fetch_and_play(client, args, wait=2)
                if period:
                    time.sleep(max(0.0, period - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
