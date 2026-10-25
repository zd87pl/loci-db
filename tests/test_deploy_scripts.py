"""Static checks for deploy/handheld (T8): shell syntax, env contract, unit files, pins, docs."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "handheld"
SCRIPTS = DEPLOY / "scripts"

# Every LOCI_* key of docs/HANDHELD_DEMO_DESIGN.md section 4.12 ...
SPEC_KEYS = [
    "LOCI_DETECTION_SOURCE",
    "LOCI_SIM_SCENARIO",
    "LOCI_SIM_REALTIME",
    "LOCI_SIM_ORACLE_POSE",
    "LOCI_SIM_LOOP",
    "LOCI_SIM_START_OFFSET_S",
    "LOCI_REPLAY_PATH",
    "LOCI_RECORD_PATH",
    "LOCI_OAK_FPS",
    "LOCI_OAK_MODEL",
    "LOCI_OAK_USB2",
    "LOCI_OAK_AXES",
    "LOCI_OAK_STEREO_PRESET",
    "LOCI_OAK_PP_SHAVES",
    "LOCI_OAK_STEREO_LRC",
    "LOCI_OAK_STEREO_EXTENDED",
    "LOCI_OAK_CONF",
    "LOCI_OAK_BBOX_SCALE",
    "LOCI_OAK_DEPTH_MIN_MM",
    "LOCI_OAK_DEPTH_MAX_MM",
    "LOCI_OAK_SPATIAL_ALGO",
    "LOCI_OAK_STALL_S",
    "LOCI_TAG_STREAM",
    "LOCI_TAG_UNDISTORT",
    "LOCI_TAG_DETECTOR",
    "LOCI_TAG_SIZE_M",
    "LOCI_ROOMS_FILE",
    "LOCI_POSE_HOLD_S",
    "LOCI_POSE_STALE_S",
    "LOCI_LABEL_EMBEDDER",
    "LOCI_EMBED_MODEL_DIR",
    "LOCI_DATA_DIR",
    "LOCI_PERSIST",
    "LOCI_SNAPSHOT_INTERVAL_S",
    "LOCI_JOURNAL_FSYNC_S",
    "LOCI_MAX_RECORDS",
    "LOCI_CONSENSUS_REFIRE_MS",
    "LOCI_CONSENSUS_RADIUS_M",
    "LOCI_DEDUP_RADIUS_M",
    "LOCI_MOVED_THRESHOLD_M",
    "LOCI_WS_PUSH_HZ",
    "LOCI_TLS",
]
# ... plus the non-LOCI keys named there ...
OTHER_SPEC_KEYS = [
    "DEPTHAI_ZOO_CACHE_PATH",
    "DEPTHAI_ZOO_INTERNET_CHECK",
    "PORT",
    "CORS_ORIGINS",
    "TTS_ENGINE",
    "PIPER_MODEL_PATH",
    "WHISPER_LOCAL",
    "WHISPER_LOCAL_MODEL",
    "OPENAI_API_KEY",
]
# ... and the T11 headset voice loop keys.
VOICE_KEYS = [
    "LOCI_VOICE",
    "LOCI_PTT",
    "LOCI_AUDIO_BACKEND",
    "LOCI_VOICE_MAX_S",
    "LOCI_EARCON",
    "LOCI_PTT_DEVICE",
    "LOCI_AUDIO_IN",
    "LOCI_AUDIO_OUT",
    "LOCI_VAD_THRESHOLD",
    "LOCI_VAD_SILENCE_S",
    "LOCI_PTT_TOGGLE",
]
# keys that mean "unset" when absent; env.example must keep them commented out
UNSET_KEYS = {"LOCI_REPLAY_PATH", "LOCI_RECORD_PATH", "LOCI_TAG_SIZE_M", "OPENAI_API_KEY"}

SH_FILES = sorted(DEPLOY.glob("*.sh"))
UNIT_FILES = [DEPLOY / "loci-handheld.service", DEPLOY / "loci-handheld.user.service"]

_ASSIGN = re.compile(r"^(?P<hash>#)?(?P<key>[A-Z][A-Z0-9_]*)=(?P<value>.*)$")


def _parse_env_example() -> tuple[dict[str, str], dict[str, str]]:
    """Return (active, commented) KEY -> VALUE maps; raise on anything else than KEY=VALUE."""
    active: dict[str, str] = {}
    commented: dict[str, str] = {}
    for n, raw in enumerate((DEPLOY / "env.example").read_text().splitlines(), 1):
        line = raw.rstrip()
        if not line:
            continue
        m = _ASSIGN.match(line)
        if m is None:
            assert line.startswith("#"), f"env.example:{n}: not KEY=VALUE or comment: {raw!r}"
            continue
        key, value = m.group("key"), m.group("value")
        assert "export" not in line and " =" not in line and "= " not in line.split("#")[0], raw
        if m.group("hash"):
            commented[key] = value
        else:
            assert key not in active, f"env.example:{n}: duplicate key {key}"
            active[key] = value
    return active, commented


# ---------------------------------------------------------------- shell scripts


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("script", SH_FILES, ids=[p.name for p in SH_FILES])
def test_shell_scripts_parse(script: Path) -> None:
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_all_expected_shell_scripts_exist_and_are_executable() -> None:
    names = {p.name for p in SH_FILES}
    assert {"install_pi.sh", "run_pi.sh", "deploy_from_mac.sh", "smoke_test.sh"} <= names
    for p in SH_FILES:
        assert os.access(p, os.X_OK), f"{p.name} is not executable"
        assert p.read_text().startswith("#!/usr/bin/env bash")
        assert "set -euo pipefail" in p.read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
@pytest.mark.parametrize("script", SH_FILES, ids=[p.name for p in SH_FILES])
def test_shell_scripts_print_help_without_side_effects(script: Path) -> None:
    proc = subprocess.run(
        ["bash", str(script), "--help"], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    assert len(proc.stdout.strip()) > 20


def test_run_pi_never_uses_reload_and_runs_one_worker() -> None:
    text = (DEPLOY / "run_pi.sh").read_text()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "--reload" not in code
    assert "--workers 1" in text
    assert "demo_spatial.app.main:app" in text
    assert "LOCI_TLS" in text and "--ssl-keyfile" in text


def test_deploy_from_mac_excludes_and_flags() -> None:
    text = (DEPLOY / "deploy_from_mac.sh").read_text()
    for exc in (
        ".git",
        "venv",
        "__pycache__",
        "demo_spatial/.certs",
        ".paperclip",
        "loci-core/target",
        "deploy/handheld/models/*",
    ):
        assert f"--exclude {exc}" in text or f"--exclude '{exc}'" in text, exc
    assert "--no-restart" in text
    assert "--logs" in text
    assert "ziggy@loci-pi.local" in text
    assert "/home/ziggy/loci-handheld/src" in text
    assert "systemctl --user restart loci-handheld" in text
    assert "--delete" in text


def test_install_pi_is_sudo_free_and_covers_the_checklist() -> None:
    text = (DEPLOY / "install_pi.sh").read_text()
    # never runs sudo itself, only prints the commands
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("echo", "#", "warn", "ok")):
            continue
        assert not stripped.startswith("sudo "), line
    for needle in (
        "usb_max_current_enable",
        "80-movidius.rules",
        "libportaudio2",
        "enable-linger",
        "requirements-pi.txt",
        "--no-deps",
        "oak-rpi5/models",
        "fetch_models.py",
        "env.example",
        "rooms.example.json",
        "import depthai, cv2, pyapriltags, tokenizers, safetensors, loci, demo_spatial.app.main",
        "loci-handheld.user.service",
        ".config/systemd/user",
    ):
        assert needle in text, needle


def test_smoke_test_covers_step_two_checks() -> None:
    text = (DEPLOY / "smoke_test.sh").read_text()
    for needle in (
        "/health",
        "/api/handheld/status",
        "/api/objects",
        "/api/query",
        "resolution.label",
        "latency_ms",
        "/api/anchor",
        "/phone",
        "aria-live",
        "where are my keys",
    ):
        assert needle in text, needle


# ---------------------------------------------------------------- env.example


def test_env_example_parses_as_key_value() -> None:
    active, commented = _parse_env_example()
    assert len(active) >= 50
    for key, value in active.items():
        assert value == value.strip(), f"{key} has surrounding whitespace"
        assert "~" not in value, f"{key}: systemd does not expand ~"
        assert "$" not in value, f"{key}: systemd does not expand $VAR"


def test_env_example_contains_every_spec_key() -> None:
    active, commented = _parse_env_example()
    present = set(active) | set(commented)
    missing = [k for k in SPEC_KEYS + OTHER_SPEC_KEYS + VOICE_KEYS if k not in present]
    assert not missing, f"env.example lacks {missing}"


def test_env_example_unset_keys_are_commented_out() -> None:
    active, commented = _parse_env_example()
    for key in UNSET_KEYS:
        assert key not in active, f"{key} must stay commented (empty != unset)"
        assert key in commented


def test_env_example_pi_values() -> None:
    active, _ = _parse_env_example()
    assert active["LOCI_DETECTION_SOURCE"] == "oak"
    assert active["LOCI_OAK_STEREO_PRESET"] == "FAST_DENSITY"
    assert active["LOCI_OAK_PP_SHAVES"] == "2"
    assert active["LOCI_OAK_AXES"] == "rdf"
    assert active["LOCI_TAG_STREAM"] == "1280x800@5"
    assert active["LOCI_LABEL_EMBEDDER"] == "static"
    assert active["LOCI_EMBED_MODEL_DIR"] == (
        "/home/ziggy/loci-handheld/models/embeddings/potion-base-8M"
    )
    assert active["LOCI_DATA_DIR"] == "/home/ziggy/loci-handheld/data"
    assert active["DEPTHAI_ZOO_CACHE_PATH"] == "/home/ziggy/loci-handheld/models/zoo"
    assert active["DEPTHAI_ZOO_INTERNET_CHECK"] == "0"
    assert active["LOCI_CONSENSUS_REFIRE_MS"] == "2000"
    assert active["PORT"] == "8001"
    assert active["TTS_ENGINE"] == "piper"
    assert active["PIPER_MODEL_PATH"].startswith("/home/ziggy/loci-handheld/models/piper/")
    assert active["WHISPER_LOCAL"] == "1"
    assert active["LOCI_VOICE"] in {"none", "headset"}
    assert active["LOCI_PTT"] in {"keyboard", "evdev", "vad"}
    assert active["LOCI_AUDIO_BACKEND"] in {"auto", "sounddevice", "cli"}
    assert active["LOCI_VAD_THRESHOLD"] == "500"
    assert active["LOCI_VAD_SILENCE_S"] == "1.2"
    for key in ("LOCI_ROOMS_FILE", "LOCI_EMBED_MODEL_DIR", "LOCI_DATA_DIR", "PIPER_MODEL_PATH"):
        assert active[key].startswith("/"), key


# ---------------------------------------------------------------- systemd units


@pytest.mark.parametrize("unit", UNIT_FILES, ids=[p.name for p in UNIT_FILES])
def test_unit_files_use_absolute_paths_and_restart_on_failure(unit: Path) -> None:
    text = unit.read_text()
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "~" not in body
    assert "$HOME" not in body
    assert "Restart=on-failure" in body
    assert "EnvironmentFile=/home/ziggy/loci-handheld/env" in body
    assert "WorkingDirectory=/home/ziggy/loci-handheld/src" in body
    exec_start = re.search(r"^ExecStart=(\S+)", body, re.M)
    assert exec_start is not None
    assert exec_start.group(1) == "/home/ziggy/loci-handheld/src/deploy/handheld/run_pi.sh"
    # the ExecStart target exists in this checkout at the same relative path
    rel = Path(exec_start.group(1)).relative_to("/home/ziggy/loci-handheld/src")
    assert (REPO / rel).is_file()
    for key in ("WorkingDirectory", "EnvironmentFile", "ExecStart"):
        for m in re.finditer(rf"^{key}=(.*)$", body, re.M):
            assert m.group(1).startswith("/"), m.group(0)
    assert "[Install]" in body and "WantedBy=" in body


def test_system_unit_vs_user_unit_specifics() -> None:
    system = (DEPLOY / "loci-handheld.service").read_text()
    user = (DEPLOY / "loci-handheld.user.service").read_text()
    assert "User=ziggy" in system
    assert "WantedBy=multi-user.target" in system
    assert re.search(r"^User=", user, re.M) is None
    assert "WantedBy=default.target" in user
    assert re.search(r"^SupplementaryGroups=", user, re.M) is None  # not permitted in user units
    for unit in (system, user):
        assert "KillMode=mixed" in unit and "TimeoutStopSec=20" in unit and "RestartSec=5" in unit


# ---------------------------------------------------------------- python scripts


def _run(script: str, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_fetch_models_help_runs() -> None:
    proc = _run("fetch_models.py", "--help")
    assert proc.returncode == 0, proc.stderr
    for flag in ("--only-missing", "--layout", "--whisper", "--verify", "--list"):
        assert flag in proc.stdout


def test_fetch_models_list_is_offline_and_reports_missing(tmp_path: Path) -> None:
    proc = _run("fetch_models.py", "--list", "--models-dir", str(tmp_path), "--whisper", "tiny.en")
    assert proc.returncode == 0, proc.stderr
    assert "layout repo" in proc.stdout
    assert "embed/model.safetensors" in proc.stdout and "missing" in proc.stdout
    assert "piper/en_US-lessac-medium.onnx" in proc.stdout
    assert "whisper/tiny.en" in proc.stdout
    # pi layout is picked for a models dir that already has an embeddings/ subdir
    (tmp_path / "embeddings").mkdir()
    proc = _run("fetch_models.py", "--list", "--models-dir", str(tmp_path))
    assert "layout pi" in proc.stdout
    assert "embeddings/potion-base-8M" in proc.stdout


def _load_fetch_models():
    spec = importlib.util.spec_from_file_location("fetch_models", SCRIPTS / "fetch_models.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve postponed annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_fetch_models_pins_and_helpers(tmp_path: Path) -> None:
    fm = _load_fetch_models()
    assert fm.POTION_REV == "bf8b056651a2c21b8d2565580b8569da283cab23"
    assert fm.POTION_FILES["model.safetensors"][1].startswith("f65d0f32")
    assert fm.POTION_FILES["model.safetensors"][1].endswith("3de2")
    assert fm.POTION_FILES["tokenizer.json"][1].startswith("e67e803f")
    assert fm.POTION_FILES["tokenizer.json"][1].endswith("bb50")
    assert fm.POTION_FILES["config.json"][1].startswith("2a6ac0e9")
    assert fm.POTION_FILES["config.json"][1].endswith("7553")
    for sha in [v[1] for v in fm.POTION_FILES.values()] + [v[1] for v in fm.PIPER_FILES.values()]:
        assert re.fullmatch(r"[0-9a-f]{64}", sha)
    url = fm.hf_url(fm.POTION_REPO, fm.POTION_REV, "config.json")
    assert url == (
        "https://huggingface.co/minishlab/potion-base-8M/resolve/"
        "bf8b056651a2c21b8d2565580b8569da283cab23/config.json"
    )
    # sha256 helper and the file_status classifier
    f = tmp_path / "config.json"
    f.write_bytes(b"x" * 202)
    assert fm.sha256_of(f) == __import__("hashlib").sha256(b"x" * 202).hexdigest()
    st = fm.file_status(tmp_path, fm.POTION_FILES)
    assert st == {"config.json": "ok", "model.safetensors": "missing", "tokenizer.json": "missing"}
    f.write_bytes(b"short")
    assert fm.file_status(tmp_path, fm.POTION_FILES)["config.json"] == "size-mismatch"
    # layouts
    d = fm.resolve_layout(tmp_path, "repo")
    assert d.embed == tmp_path / "potion-base-8M" and d.zoo == tmp_path / "zoo"
    d = fm.resolve_layout(tmp_path, "pi")
    assert d.embed == tmp_path / "embeddings" / "potion-base-8M"
    assert fm.resolve_layout(Path("/home/ziggy/loci-handheld/models"), "auto").layout == "pi"


def test_fetch_models_download_refuses_non_https(tmp_path: Path) -> None:
    fm = _load_fetch_models()
    with pytest.raises(ValueError):
        fm.download("file:///etc/passwd", tmp_path / "x", expected_sha256="0" * 64)
    with pytest.raises(ValueError):
        fm.download("http://example.invalid/x", tmp_path / "x", expected_sha256="0" * 64)


def test_fetch_models_verifies_sha_and_keeps_no_partial(tmp_path: Path, monkeypatch) -> None:
    """A fake urlopen serving wrong bytes must leave no .part and raise OSError."""
    import io

    fm = _load_fetch_models()

    class _Resp(io.BytesIO):
        headers = {"Content-Length": "5"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fm.urllib.request, "urlopen", lambda req, timeout: _Resp(b"hello"))
    monkeypatch.setattr(fm.time, "sleep", lambda s: None)
    dest = tmp_path / "config.json"
    with pytest.raises(OSError):
        fm.download(
            "https://example.invalid/config.json",
            dest,
            expected_sha256="0" * 64,
            expected_size=5,
            retries=2,
            log=lambda m: None,
        )
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))
    # and the happy path renames the .part into place
    import hashlib

    fm.download(
        "https://example.invalid/config.json",
        dest,
        expected_sha256=hashlib.sha256(b"hello").hexdigest(),
        expected_size=5,
        log=lambda m: None,
    )
    assert dest.read_bytes() == b"hello"


def test_fetch_models_ensure_files_only_missing(tmp_path: Path, monkeypatch) -> None:
    fm = _load_fetch_models()
    calls: list[str] = []

    def fake_download(url, dest, **kw):
        calls.append(url)
        dest.write_bytes(b"x" * fm.PIPER_FILES[dest.name][0])
        return dest

    monkeypatch.setattr(fm, "download", fake_download)
    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(
        b"x" * fm.PIPER_FILES["en_US-lessac-medium.onnx"][0]
    )
    problems = fm.ensure_files(
        tmp_path,
        fm.PIPER_FILES,
        lambda n: fm.hf_url(fm.PIPER_REPO, fm.PIPER_REV, f"{fm.PIPER_SUBDIR}/{n}"),
        only_missing=True,
        force=False,
        verify=False,
        timeout=1,
        log=lambda m: None,
    )
    assert problems == []
    assert calls == [
        "https://huggingface.co/rhasspy/piper-voices/resolve/"
        "5041a7a49ae5b3a0f30e6837769eb03fdbb231bd/en/en_US/lessac/medium/"
        "en_US-lessac-medium.onnx.json"
    ]


def test_fetch_models_zoo_manual_instructions_without_depthai(tmp_path: Path, monkeypatch) -> None:
    fm = _load_fetch_models()
    monkeypatch.setitem(sys.modules, "depthai", None)  # force ImportError
    with pytest.raises(ImportError):
        fm.fetch_zoo(tmp_path / "zoo", log=lambda m: None)
    text = fm.zoo_manual_instructions(tmp_path / "zoo")
    assert "getModelFromZoo" in text and "yolov6-nano" in text and "RVC2" in text
    assert fm.zoo_archive(tmp_path) is None
    arch = tmp_path / "zoo" / "abc" / "YOLOv6_Nano-R2_COCO_512x384.rvc2.tar.xz"
    arch.parent.mkdir(parents=True)
    arch.write_bytes(b"")
    assert fm.zoo_archive(tmp_path / "zoo") == arch


def test_make_tag_sheet_help_runs() -> None:
    proc = _run("make_tag_sheet.py", "--help")
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------- requirements


def _requirement_lines(name: str) -> list[str]:
    return [
        line.split("#")[0].strip()
        for line in (DEPLOY / name).read_text().splitlines()
        if line.split("#")[0].strip()
    ]


def test_requirements_pi_exact_pins() -> None:
    lines = _requirement_lines("requirements-pi.txt")
    for pin in (
        "depthai==3.11.0",
        "opencv-python-headless==5.0.0.93",
        "numpy>=2.2,<3",
        "pyapriltags==3.4.3.1",
        "tokenizers==0.23.3",
        "safetensors==0.8.0",
    ):
        assert pin in lines, pin
    names = {re.split(r"[=<>\[ ]", line)[0] for line in lines}
    for pkg in (
        "fastapi",
        "uvicorn",
        "pydantic",
        "httpx",
        "qdrant-client",
        "hilbertcurve",
        "python-multipart",
        "segno",
        "sounddevice",
        "faster-whisper",
        "piper-tts",
    ):
        assert pkg in names, pkg
    assert any(line.startswith("uvicorn[standard]") for line in lines)
    # the voice block is delimited so install_pi.sh can strip it on wheel failures
    text = (DEPLOY / "requirements-pi.txt").read_text()
    assert "# --- voice" in text and "# --- end voice" in text
    assert text.index("# --- voice") < text.index("sounddevice") < text.index("# --- end voice")
    # nothing in the pins is a loose git/url requirement
    assert not any(line.startswith(("git+", "http://", "https://")) for line in lines)


def test_requirements_dev_includes_pi_pins_and_tooling() -> None:
    lines = _requirement_lines("requirements-dev.txt")
    assert "-r requirements-pi.txt" in lines
    names = {re.split(r"[=<>\[ ]", line)[0] for line in lines}
    assert {"pillow", "pytest-asyncio", "ruff"} <= names


# ---------------------------------------------------------------- docs and gitignore


def test_models_dir_is_gitignored_except_docs() -> None:
    gi = (DEPLOY / "models" / ".gitignore").read_text().split()
    assert "*" in gi and "!.gitignore" in gi and "!README.md" in gi
    assert (DEPLOY / "models" / "README.md").is_file()
    if shutil.which("git") and (REPO / ".git").exists():
        ignored = subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", "-q", "deploy/handheld/models/zoo/x.tar.xz"],
            capture_output=True,
            timeout=30,
        )
        assert ignored.returncode == 0
        kept = subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", "-q", "deploy/handheld/models/README.md"],
            capture_output=True,
            timeout=30,
        )
        assert kept.returncode == 1


def test_readme_documents_headset_pairing_and_portaudio() -> None:
    text = (DEPLOY / "README.md").read_text()
    for needle in (
        "libportaudio2",
        "bluetoothctl",
        "pair ",
        "trust ",
        "connect ",
        "headset-head-unit",
        "HFP",
        "usb_max_current_enable",
        "LOCI_VOICE",
        "LOCI_PTT_DEVICE",
        "voice_console.py",
        "make_tag_sheet.py",
        "install_pi.sh",
        "deploy_from_mac.sh",
        "smoke_test.sh",
        "X_LINK",
        "hotspot",
    ):
        assert needle in text, needle
