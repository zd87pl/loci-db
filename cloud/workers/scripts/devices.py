#!/usr/bin/env python3
"""Provision, list and revoke LOCI Cloud devices (ESP32 boards) for one project.

`create` prints a one-time claim code (valid 30 min) and the console line to type on the
board, which exchanges it for its own device id and secret:

    ENV=~/.loci/cloud.env
    python3 cloud/workers/scripts/devices.py --env-file $ENV create --name "P4 desk board"
    python3 cloud/workers/scripts/devices.py --env-file $ENV list
    python3 cloud/workers/scripts/devices.py --env-file $ENV revoke dev_0123456789abcdef

The env file is the one bootstrap.py prints: LOCI_CLOUD_URL, LOCI_CLOUD_PROJECT, LOCI_CLOUD_SPACE
and the secret key (LOCI_CLOUD_KEY, possibly inside a comment line). --url / --project / --key /
--space, or the same LOCI_CLOUD_* environment variables, override it. The secret key is never
printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_LINE = re.compile(r"(LOCI_CLOUD_[A-Z_]+)=(\S+)")


def read_env_file(path: str | None) -> dict[str, str]:
    """KEY=VALUE pairs, including ones written inside comments (bootstrap.py's secret key line)."""
    if not path:
        return {}
    out: dict[str, str] = {}
    for line in Path(path).expanduser().read_text().splitlines():
        for k, v in _LINE.findall(line):
            out.setdefault(k, v)
    return out


def call(base: str, method: str, path: str, key: str | None, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method)  # noqa: S310 - scheme checked
    req.add_header("content-type", "application/json")
    # Cloudflare blocks the default Python-urllib user agent (error 1010).
    req.add_header("user-agent", "loci-cloud-devices/0.1")
    if key:
        req.add_header("authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310 - operator-supplied URL
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} failed: HTTP {e.code} {e.read().decode()[:300]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--env-file", help="file printed by bootstrap.py (e.g. ~/.loci/cloud.env)")
    ap.add_argument("--url", help="Worker URL, e.g. https://loci-cloud.<acct>.workers.dev")
    ap.add_argument("--project", help="project ref")
    ap.add_argument("--key", help="secret key (loci_sk_...)")
    ap.add_argument("--space", help="space the device may use (create)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="provision a device and print its claim code")
    c.add_argument("--name", default="ESP32 board")
    sub.add_parser("list", help="list the project's devices")
    r = sub.add_parser("revoke", help="revoke a device (its tokens stop working within a minute)")
    r.add_argument("device_id")
    a = ap.parse_args()

    env = {
        **read_env_file(a.env_file),
        **{k: v for k, v in os.environ.items() if k.startswith("LOCI_CLOUD_")},
    }
    url = (a.url or env.get("LOCI_CLOUD_URL", "")).rstrip("/")
    project = a.project or env.get("LOCI_CLOUD_PROJECT", "")
    key = a.key or env.get("LOCI_CLOUD_KEY", "")
    space = a.space or env.get("LOCI_CLOUD_SPACE", "home")
    if not url.startswith(("http://", "https://")):
        sys.exit("a Worker URL is required (--url or LOCI_CLOUD_URL), http(s) only")
    if not project or not key.startswith("loci_sk_"):
        sys.exit("the project ref and its secret key (loci_sk_...) are required")
    base = f"{url}/p/{project}/v1"

    if a.cmd == "create":
        d = call(base, "POST", "/devices", key, {"name": a.name, "spaces": [space]})
        left = max(0, round((d["claim_expires_at"] - time.time() * 1000) / 60_000))
        print(f"device     {d['device_id']}  ({d['name']}, space {space})")
        print(f"claim code {d['claim_code']}  (single use, expires in {left} min)")
        print("on the board's serial console:")
        print(f"  cloud claim {base} {space} {d['claim_code']}")
    elif a.cmd == "list":
        for d in call(base, "GET", "/devices", key)["devices"]:
            state = (
                "revoked"
                if d.get("revoked")
                else ("claimed" if d.get("claimed_at") else "unclaimed")
            )
            seen = d.get("last_seen_at")
            ago = f"{round((time.time() * 1000 - seen) / 60_000)} min ago" if seen else "never"
            print(f"{d['id']}  {state:9}  last seen {ago:>12}  {d.get('name') or ''}")
    else:
        call(base, "DELETE", f"/devices/{a.device_id}", key)
        print(f"revoked {a.device_id}")


if __name__ == "__main__":
    main()
