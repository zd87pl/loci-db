"""Tests for the LOCI Cloud operator scripts in cloud/workers/scripts (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "cloud" / "workers" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"cloud_script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_env_file_reader_finds_the_secret_key_in_bootstraps_comment_line(tmp_path) -> None:
    env = tmp_path / "cloud.env"
    env.write_text(
        "LOCI_CLOUD_URL=https://loci-cloud.example.workers.dev\n"
        "LOCI_CLOUD_PROJECT=abcdefghij12\n"
        "LOCI_CLOUD_SPACE=home\n"
        "LOCI_CLOUD_DEVICE_ID=dev_0123456789abcdef\n"
        "# secret key (server use only): LOCI_CLOUD_KEY=loci_sk_abcdefghij12_" + "0" * 48 + "\n"
        "# publishable key (apps, read-only): loci_pk_abcdefghij12_" + "1" * 48 + "\n"
    )
    got = _load("devices").read_env_file(str(env))
    assert got["LOCI_CLOUD_URL"] == "https://loci-cloud.example.workers.dev"
    assert got["LOCI_CLOUD_SPACE"] == "home"
    assert got["LOCI_CLOUD_KEY"].startswith("loci_sk_abcdefghij12_")
    assert _load("devices").read_env_file(None) == {}
