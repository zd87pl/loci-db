#!/usr/bin/env python3
"""Create a LOCI Cloud project and print env lines for it.

Device mode (default) also creates a space and a claimed device, and prints the
LOCI_CLOUD_* lines the demo host reads:

    python3 cloud/workers/scripts/bootstrap.py --url http://127.0.0.1:8787 \
        --admin-token "$ADMIN_TOKEN" --space home --device "ESP32 pendant" > .cloud.env

SDK mode prints LOCI_BASE_URL and LOCI_API_KEY for the Python SDK's cloud mode,
LociClient(base_url=..., api_key=...); its first insert creates the "default" space:

    python3 cloud/workers/scripts/bootstrap.py --url http://127.0.0.1:8787 \
        --admin-token "$ADMIN_TOKEN" --sdk > .loci-sdk.env

Secrets are printed to stdout only; keep that file private (chmod 600) and never commit it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def call(
    url: str, method: str, path: str, token: str | None = None, body: dict | None = None
) -> dict:
    data = None if body is None else json.dumps(body).encode()
    target = url.rstrip("/") + path  # scheme checked in main()
    req = urllib.request.Request(target, data=data, method=method)  # noqa: S310
    req.add_header("content-type", "application/json")
    # Cloudflare blocks the default Python-urllib user agent (error 1010).
    req.add_header("user-agent", "loci-cloud-bootstrap/0.1")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310 - operator-supplied URL
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} failed: HTTP {e.code} {e.read().decode()[:300]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--admin-token", required=True)
    ap.add_argument("--name", default="LOCI assistant")
    ap.add_argument("--jurisdiction", choices=["us", "eu"], default="us")
    ap.add_argument("--space", default="home")
    ap.add_argument("--device", default="ESP32 assistant")
    ap.add_argument(
        "--sdk",
        action="store_true",
        help="print LOCI_BASE_URL and LOCI_API_KEY for the Python SDK (no space or device)",
    )
    a = ap.parse_args()
    if not a.url.startswith(("http://", "https://")):
        sys.exit("--url must be http(s)")

    body = {"name": a.name, "jurisdiction": a.jurisdiction}
    proj = call(a.url, "POST", "/admin/projects", a.admin_token, body)
    ref, sk = proj["ref"], proj["secret_key"]
    sdk_url = f"{a.url.rstrip('/')}/p/{ref}"
    if a.sdk:
        print(f"LOCI_BASE_URL={sdk_url}")
        print(f"LOCI_API_KEY={sk}")
        print(f"# publishable key (read-only queries): {proj['publishable_key']}")
        return
    base = f"/p/{ref}/v1"
    space = {"space": a.space, "name": a.space.title(), "frame": "room"}
    call(a.url, "POST", f"{base}/spaces", sk, space)
    dev = call(a.url, "POST", f"{base}/devices", sk, {"name": a.device, "spaces": [a.space]})
    claimed = call(a.url, "POST", f"{base}/devices/claim", None, {"claim_code": dev["claim_code"]})
    print(f"LOCI_CLOUD_URL={a.url.rstrip('/')}")
    print(f"LOCI_CLOUD_PROJECT={ref}")
    print(f"LOCI_CLOUD_SPACE={a.space}")
    print(f"LOCI_CLOUD_DEVICE_ID={claimed['device_id']}")
    print(f"LOCI_CLOUD_DEVICE_SECRET={claimed['device_secret']}")
    print(f"# secret key (server use only): LOCI_CLOUD_KEY={sk}")
    print(f"# Python SDK base_url (with the secret key): {sdk_url}")
    print(f"# publishable key (apps, read-only): {proj['publishable_key']}")


if __name__ == "__main__":
    main()
