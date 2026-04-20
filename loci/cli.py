"""``loci`` command-line interface.

Currently ships the ``loci cloud keys`` subcommands for managing API keys
against the LOCI Cloud API admin endpoints.

Usage:
    loci cloud keys create --email ... --namespace ...
    loci cloud keys list [--tenant-id ...]
    loci cloud keys revoke <KEY_ID>

Auth: pass an admin API key via ``--api-key`` or the ``LOCI_API_KEY`` env var.
Base URL: pass ``--base-url`` or set ``LOCI_BASE_URL`` (defaults to
``https://api.loci.ai`` if neither is provided).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.loci.ai"


def _http(method: str, base_url: str, path: str, api_key: str, body: dict | None = None) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    if not url.startswith(("http://", "https://")):
        sys.stderr.write(f"error: base_url must be http(s): {base_url!r}\n")
        sys.exit(2)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310 — scheme validated above  # noqa: S310
            raw = resp.read().decode() or "{}"
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace") if exc.fp else str(exc)
        sys.stderr.write(f"error: HTTP {exc.code}: {detail}\n")
        sys.exit(1)
    except urllib.error.URLError as exc:
        sys.stderr.write(f"error: request failed: {exc}\n")
        sys.exit(1)


def _resolve_auth(args: argparse.Namespace) -> tuple[str, str]:
    base_url = args.base_url or os.environ.get("LOCI_BASE_URL") or DEFAULT_BASE_URL
    api_key = args.api_key or os.environ.get("LOCI_API_KEY")
    if not api_key:
        sys.stderr.write("error: admin API key required — pass --api-key or set LOCI_API_KEY\n")
        sys.exit(2)
    return base_url, api_key


def _cmd_keys_create(args: argparse.Namespace) -> None:
    base_url, api_key = _resolve_auth(args)
    body = {
        "tenant_email": args.email,
        "tenant_name": args.name,
        "namespace": args.namespace,
        "label": args.label,
        "rate_limit_rpm": args.rate_limit_rpm,
        "is_admin": args.admin,
    }
    # Drop Nones so the server applies its own defaults cleanly.
    body = {k: v for k, v in body.items() if v is not None}
    resp = _http("POST", base_url, "/admin/keys", api_key, body=body)

    if args.json:
        print(json.dumps(resp, indent=2))
        return

    print("=== API key created ===")
    print(f"Key ID    : {resp['key_id']}")
    print(f"Tenant ID : {resp['tenant_id']}")
    print(f"Namespace : {resp['namespace']}")
    print(f"Admin     : {resp['is_admin']}")
    print(f"Prefix    : {resp['prefix']}")
    print("")
    print("RAW KEY (shown only once — store securely):")
    print(f"  {resp['raw_key']}")


def _cmd_keys_list(args: argparse.Namespace) -> None:
    base_url, api_key = _resolve_auth(args)
    path = "/admin/keys"
    params: list[str] = []
    if args.tenant_id:
        params.append(f"tenant_id={args.tenant_id}")
    if args.include_revoked:
        params.append("include_revoked=true")
    if params:
        path += "?" + "&".join(params)

    resp = _http("GET", base_url, path, api_key)

    if args.json:
        print(json.dumps(resp, indent=2))
        return

    keys = resp.get("keys", [])
    if not keys:
        print("(no keys)")
        return

    header = f"{'KEY ID':36}  {'PREFIX':14}  {'NAMESPACE':24}  {'ADMIN':5}  {'REVOKED':7}  LABEL"
    print(header)
    print("-" * len(header))
    for k in keys:
        print(
            f"{k['id']:36}  {k['prefix']:14}  {k['namespace']:24}  "
            f"{str(k['is_admin']):5}  {str(k['revoked']):7}  {k.get('label') or ''}"
        )


def _cmd_keys_revoke(args: argparse.Namespace) -> None:
    base_url, api_key = _resolve_auth(args)
    resp = _http("DELETE", base_url, f"/admin/keys/{args.key_id}", api_key)
    if args.json:
        print(json.dumps(resp, indent=2))
        return
    print(f"revoked: {resp['key_id']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loci", description="LOCI command-line interface")
    subs = parser.add_subparsers(dest="command", required=True)

    cloud = subs.add_parser("cloud", help="LOCI Cloud API management")
    cloud_subs = cloud.add_subparsers(dest="cloud_command", required=True)

    keys = cloud_subs.add_parser("keys", help="Manage API keys")
    keys_subs = keys.add_subparsers(dest="keys_command", required=True)

    # Shared auth flags
    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--api-key", help="Admin API key (or set LOCI_API_KEY)")
        p.add_argument("--base-url", help=f"Cloud API base URL (default: {DEFAULT_BASE_URL})")
        p.add_argument("--json", action="store_true", help="Emit raw JSON response")

    create = keys_subs.add_parser("create", help="Create a new API key")
    create.add_argument("--email", required=True, help="Tenant email (unique)")
    create.add_argument("--namespace", required=True, help="Qdrant collection prefix (a-z0-9_)")
    create.add_argument("--name", help="Tenant display name (defaults to email)")
    create.add_argument("--label", help="Key label (e.g. 'prod', 'dev laptop')")
    create.add_argument("--rate-limit-rpm", type=int, help="Per-minute rate limit")
    create.add_argument("--admin", action="store_true", help="Grant admin privileges")
    _add_common(create)
    create.set_defaults(func=_cmd_keys_create)

    lst = keys_subs.add_parser("list", help="List API keys")
    lst.add_argument("--tenant-id", help="Filter by tenant UUID")
    lst.add_argument("--include-revoked", action="store_true")
    _add_common(lst)
    lst.set_defaults(func=_cmd_keys_list)

    rev = keys_subs.add_parser("revoke", help="Revoke an API key by id")
    rev.add_argument("key_id", help="UUID of the key to revoke")
    _add_common(rev)
    rev.set_defaults(func=_cmd_keys_revoke)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
