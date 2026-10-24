#!/usr/bin/env python3
"""Fetch and verify the models the LOCI handheld demo needs (stdlib only at import time).

Targets (see deploy/handheld/models/README.md):

* ``zoo``    YOLOv6-nano RVC2 archive from the Luxonis HubAI zoo. Needs ``depthai`` importable
             (``dai.getModelFromZoo``); without it the manual recipe is printed.
* ``embed``  potion-base-8M (model.safetensors, tokenizer.json, config.json) from the Hugging Face
             Hub at a pinned revision, sha256-verified.
* ``piper``  en_US-lessac-medium Piper voice (.onnx + .onnx.json), pinned revision, sha256-verified.
             ``--piper-copy-from DIR`` copies an existing pair instead of downloading.
* ``--whisper tiny.en``  warms the faster-whisper cache (needs ``faster_whisper`` importable).

Layouts: ``repo`` (<models>/potion-base-8M) on a dev machine, ``pi`` (<models>/embeddings/
potion-base-8M) on the Pi; ``auto`` picks ``pi`` for /home/ziggy/loci-handheld/models or when
<models>/embeddings exists.

Exit code 0 when every requested target is present afterwards, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO_MODELS = REPO_ROOT / "deploy" / "handheld" / "models"
DEFAULT_PI_MODELS = Path("/home/ziggy/loci-handheld/models")

HF_HOST = "https://huggingface.co"

POTION_REPO = "minishlab/potion-base-8M"
POTION_REV = "bf8b056651a2c21b8d2565580b8569da283cab23"
POTION_FILES: Mapping[str, tuple[int, str]] = {
    "model.safetensors": (
        30_236_760,
        "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2",
    ),
    "tokenizer.json": (
        683_666,
        "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50",
    ),
    "config.json": (202, "2a6ac0e9aaa356a68a5688070db78fc3a464fefe85d2f06a1905ce3718687553"),
}

PIPER_REPO = "rhasspy/piper-voices"
PIPER_REV = "5041a7a49ae5b3a0f30e6837769eb03fdbb231bd"
PIPER_SUBDIR = "en/en_US/lessac/medium"
PIPER_FILES: Mapping[str, tuple[int, str]] = {
    "en_US-lessac-medium.onnx": (
        63_201_294,
        "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    ),
    "en_US-lessac-medium.onnx.json": (
        4_885,
        "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
    ),
}

ZOO_MODEL = "yolov6-nano"
ZOO_PLATFORM = "RVC2"
ZOO_ARCHIVE_GLOB = "*.rvc2.tar.xz"

WHISPER_HUB_DIR = Path.home() / ".cache" / "huggingface" / "hub"

Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- helpers


def hf_url(repo: str, rev: str, path: str) -> str:
    """Resolve URL for a file on the Hugging Face Hub (always https)."""
    return f"{HF_HOST}/{repo}/resolve/{rev}/{path}"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    timeout: float = 60.0,
    retries: int = 3,
    log: Log = _print,
) -> Path:
    """Stream ``url`` to ``dest`` (via a ``.part`` file), verify size + sha256, then rename."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            # S310: the scheme is pinned to https by the check at the top of this function.
            req = urllib.request.Request(  # noqa: S310
                url, headers={"User-Agent": "loci-handheld-fetch/1"}
            )
            t0 = time.monotonic()
            with (
                urllib.request.urlopen(req, timeout=timeout) as resp,  # noqa: S310
                part.open("wb") as out,
            ):
                total = resp.headers.get("Content-Length")
                done = 0
                next_report = 0.0
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if total and done / int(total) >= next_report:
                        log(f"    {dest.name}: {done / 1e6:6.1f} / {int(total) / 1e6:.1f} MB")
                        next_report += 0.25
            size = part.stat().st_size
            if expected_size is not None and size != expected_size:
                raise OSError(f"{dest.name}: size {size} != expected {expected_size}")
            got = sha256_of(part)
            if got != expected_sha256:
                raise OSError(f"{dest.name}: sha256 {got} != expected {expected_sha256}")
            os.replace(part, dest)
            log(f"    {dest.name}: ok ({size / 1e6:.1f} MB in {time.monotonic() - t0:.1f} s)")
            return dest
        except (urllib.error.URLError, OSError, ValueError) as e:
            last_error = e
            log(f"    {dest.name}: attempt {attempt}/{retries} failed: {e}")
            if part.exists():
                part.unlink()
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 6.0))
    raise OSError(f"download failed for {url}: {last_error}")


@dataclass(frozen=True)
class Dirs:
    models: Path
    layout: str
    zoo: Path
    embed: Path
    piper: Path


def resolve_layout(models_dir: Path, layout: str = "auto") -> Dirs:
    if layout == "auto":
        pi_like = models_dir == DEFAULT_PI_MODELS or (models_dir / "embeddings").is_dir()
        layout = "pi" if pi_like else "repo"
    embed = (
        models_dir / "embeddings" / "potion-base-8M"
        if layout == "pi"
        else models_dir / "potion-base-8M"
    )
    return Dirs(models_dir, layout, models_dir / "zoo", embed, models_dir / "piper")


def default_models_dir() -> Path:
    env = os.environ.get("LOCI_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    if DEFAULT_PI_MODELS.is_dir():
        return DEFAULT_PI_MODELS
    return DEFAULT_REPO_MODELS


# ----------------------------------------------------------------------------- targets


def zoo_archive(zoo_dir: Path) -> Path | None:
    if not zoo_dir.is_dir():
        return None
    hits = sorted(zoo_dir.rglob(ZOO_ARCHIVE_GLOB))
    return hits[0] if hits else None


def zoo_manual_instructions(zoo_dir: Path) -> str:
    return (
        "    depthai is not importable here. On a machine with depthai 3.x and internet:\n"
        f'      DEPTHAI_ZOO_CACHE_PATH={zoo_dir} python -c "import depthai as dai; '
        f"print(dai.getModelFromZoo(dai.NNModelDescription('{ZOO_MODEL}', "
        f"platform='{ZOO_PLATFORM}'), useCached=True, cacheDirectory='{zoo_dir}'))\"\n"
        f"    then copy {zoo_dir}/ to the Pi (rsync -a {zoo_dir}/ "
        "ziggy@loci-pi.local:/home/ziggy/loci-handheld/models/zoo/).\n"
        "    The archive is YOLOv6_Nano-R2_COCO_512x384.rvc2.tar.xz (8.1 MB); LOCI_OAK_MODEL may "
        "also point at that file directly."
    )


def fetch_zoo(zoo_dir: Path, log: Log = _print) -> Path:
    """Download the yolov6-nano RVC2 archive into ``zoo_dir`` through depthai's zoo client."""
    try:
        import depthai as dai  # lazy: optional dependency
    except ImportError as e:
        raise ImportError("depthai not importable") from e
    zoo_dir.mkdir(parents=True, exist_ok=True)
    # getModelFromZoo honours these at call time; allow the network for this one call.
    os.environ["DEPTHAI_ZOO_CACHE_PATH"] = str(zoo_dir)
    os.environ["DEPTHAI_ZOO_INTERNET_CHECK"] = "1"
    desc = dai.NNModelDescription(ZOO_MODEL, platform=ZOO_PLATFORM)
    try:
        path = dai.getModelFromZoo(desc, useCached=True, cacheDirectory=str(zoo_dir))
    except TypeError:  # older/newer keyword spelling
        path = dai.getModelFromZoo(desc, True, str(zoo_dir))
    log(f"    zoo: {path}")
    found = zoo_archive(zoo_dir)
    if found is None:
        raise OSError(f"depthai returned {path} but no {ZOO_ARCHIVE_GLOB} under {zoo_dir}")
    return found


def file_status(directory: Path, files: Mapping[str, tuple[int, str]]) -> dict[str, str]:
    """Per file: 'ok' (present, size matches), 'missing', or 'size-mismatch'."""
    out: dict[str, str] = {}
    for name, (size, _sha) in files.items():
        p = directory / name
        if not p.is_file():
            out[name] = "missing"
        elif p.stat().st_size != size:
            out[name] = "size-mismatch"
        else:
            out[name] = "ok"
    return out


def ensure_files(
    directory: Path,
    files: Mapping[str, tuple[int, str]],
    url_for: Callable[[str], str],
    *,
    only_missing: bool,
    force: bool,
    verify: bool,
    timeout: float,
    log: Log = _print,
) -> list[str]:
    """Download what is missing (or everything with --force); return a list of problems."""
    problems: list[str] = []
    status = file_status(directory, files)
    for name, (size, sha) in files.items():
        dest = directory / name
        if status[name] == "ok" and not force:
            if verify:
                got = sha256_of(dest)
                if got == sha:
                    log(f"    {name}: ok (sha256 verified)")
                else:
                    log(f"    {name}: present but sha256 {got[:12]}... != pinned {sha[:12]}...")
                    problems.append(f"{name}: sha256 mismatch (use --force to re-download)")
            else:
                log(f"    {name}: ok")
            continue
        if only_missing and status[name] == "ok":
            continue
        try:
            download(
                url_for(name),
                dest,
                expected_sha256=sha,
                expected_size=size,
                timeout=timeout,
                log=log,
            )
        except OSError as e:
            problems.append(str(e))
    return problems


def copy_piper(src_dir: Path, piper_dir: Path, log: Log = _print) -> bool:
    """Copy an existing Piper voice pair (older checkout) instead of downloading."""
    onnx = src_dir / "en_US-lessac-medium.onnx"
    if not onnx.is_file():
        return False
    piper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, piper_dir / onnx.name)
    cfg = src_dir / "en_US-lessac-medium.onnx.json"
    if cfg.is_file():
        shutil.copy2(cfg, piper_dir / cfg.name)
    log(f"    piper: copied from {src_dir}")
    return True


def whisper_cached(model: str, hub_dir: Path = WHISPER_HUB_DIR) -> bool:
    return (hub_dir / f"models--Systran--faster-whisper-{model}").is_dir()


def fetch_whisper(model: str, log: Log = _print) -> None:
    from faster_whisper import WhisperModel  # lazy: optional dependency

    t0 = time.monotonic()
    WhisperModel(model, device="cpu", compute_type="int8")
    log(f"    whisper {model}: cached under {WHISPER_HUB_DIR} ({time.monotonic() - t0:.1f} s)")


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fetch_models.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="models root (default: $LOCI_MODELS_DIR, else /home/ziggy/loci-handheld/models when "
        "it exists, else <repo>/deploy/handheld/models)",
    )
    p.add_argument("--layout", choices=("auto", "repo", "pi"), default="auto")
    p.add_argument("--only-missing", action="store_true", help="never re-download present files")
    p.add_argument("--force", action="store_true", help="re-download even when present")
    p.add_argument("--verify", action="store_true", help="sha256-check files that are present")
    p.add_argument("--list", action="store_true", help="print the status table and exit 0")
    p.add_argument("--skip-zoo", action="store_true")
    p.add_argument("--skip-embed", action="store_true")
    p.add_argument("--skip-piper", action="store_true")
    p.add_argument(
        "--piper-copy-from",
        type=Path,
        default=None,
        help="copy en_US-lessac-medium.onnx{,.json} from this directory instead of downloading",
    )
    p.add_argument(
        "--whisper",
        metavar="MODEL",
        default=None,
        help="also warm the faster-whisper cache for MODEL (e.g. tiny.en); skipped when cached",
    )
    p.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    return p


def status_table(dirs: Dirs, whisper: str | None) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    arch = zoo_archive(dirs.zoo)
    rows.append(("zoo", "ok" if arch else "missing", str(arch or dirs.zoo)))
    for name, st in file_status(dirs.embed, POTION_FILES).items():
        rows.append((f"embed/{name}", st, str(dirs.embed / name)))
    for name, st in file_status(dirs.piper, PIPER_FILES).items():
        rows.append((f"piper/{name}", st, str(dirs.piper / name)))
    if whisper:
        rows.append(
            (
                f"whisper/{whisper}",
                "ok" if whisper_cached(whisper) else "missing",
                str(WHISPER_HUB_DIR),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models_dir = (args.models_dir or default_models_dir()).expanduser().resolve()
    dirs = resolve_layout(models_dir, args.layout)
    log = _print
    log(f"models dir: {dirs.models} (layout {dirs.layout})")

    if args.list:
        for name, st, path in status_table(dirs, args.whisper):
            log(f"  {name:<40} {st:<14} {path}")
        return 0

    problems: list[str] = []

    if not args.skip_zoo:
        log("zoo (yolov6-nano RVC2 archive)")
        arch = zoo_archive(dirs.zoo)
        if arch and not args.force:
            log(f"    ok: {arch}")
        else:
            try:
                fetch_zoo(dirs.zoo, log)
            except ImportError:
                log(zoo_manual_instructions(dirs.zoo))
                problems.append("zoo archive missing and depthai not importable")
            except Exception as e:
                log(f"    zoo fetch failed: {e}")
                problems.append(f"zoo: {e}")

    if not args.skip_embed:
        log(f"embed (potion-base-8M @ {POTION_REV[:8]}) -> {dirs.embed}")
        problems += ensure_files(
            dirs.embed,
            POTION_FILES,
            lambda n: hf_url(POTION_REPO, POTION_REV, n),
            only_missing=args.only_missing,
            force=args.force,
            verify=args.verify,
            timeout=args.timeout,
            log=log,
        )

    if not args.skip_piper:
        log(f"piper (en_US-lessac-medium @ {PIPER_REV[:8]}) -> {dirs.piper}")
        if args.piper_copy_from and (
            args.force or file_status(dirs.piper, PIPER_FILES)["en_US-lessac-medium.onnx"] != "ok"
        ):
            copy_piper(args.piper_copy_from.expanduser(), dirs.piper, log)
        problems += ensure_files(
            dirs.piper,
            PIPER_FILES,
            lambda n: hf_url(PIPER_REPO, PIPER_REV, f"{PIPER_SUBDIR}/{n}"),
            only_missing=args.only_missing,
            force=args.force,
            verify=args.verify,
            timeout=args.timeout,
            log=log,
        )

    if args.whisper:
        log(f"whisper ({args.whisper})")
        if whisper_cached(args.whisper) and not args.force:
            log("    ok (cached)")
        else:
            try:
                fetch_whisper(args.whisper, log)
            except ImportError:
                log("    faster_whisper not importable; pip install faster-whisper first")
                problems.append("whisper: faster_whisper not importable")
            except Exception as e:
                log(f"    whisper download failed: {e}")
                problems.append(f"whisper: {e}")

    if problems:
        log("PROBLEMS:")
        for p in problems:
            log(f"  - {p}")
        return 1
    log("all requested models present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
