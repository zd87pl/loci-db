"""Device-token authentication for the ESP32 endpoints, with a LAN / public split.

The host is meant to sit on a LAN today and behind a Cloudflare Tunnel tomorrow. A tunnel
makes every route of the demo reachable from the whole internet, so two things happen when
a request arrives from outside (through the tunnel, which adds ``CF-Connecting-IP``, or from
a non-private address):

* only the board's surface is served: ``/api/esp32/*`` and the ``/esp32`` page; everything
  else answers 404 (see :func:`public_surface_middleware`);
* the board's endpoints need a device token, sent as ``X-Device-Token: <token>`` or
  ``Authorization: Bearer <token>``.

Modes (``LOCI_ESP32_AUTH``):

* ``auto`` (default) — token required for public requests only, so a board on the LAN keeps
  working with no configuration;
* ``always`` — token required everywhere;
* ``off`` — never required and no surface restriction (LAN demos; the tunnel script refuses
  to start in this mode).

Tokens come from ``LOCI_ESP32_TOKEN`` (one shared token), ``LOCI_ESP32_TOKEN_FILE`` (a file
holding it; ``run_esp32_demo.sh`` creates one) and ``LOCI_ESP32_TOKENS``
(``device:token,device:token`` for one token per board; a per-device token also pins the
device id, so a board cannot post as another). With no token configured, public requests
are refused rather than let through.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from dataclasses import dataclass, field

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

SHARED = "*"  # device id returned for the shared token
PUBLIC_PREFIXES = ("/api/esp32/",)
PUBLIC_PATHS = ("/esp32",)


def _is_private_ip(host: str | None) -> bool:
    """True for loopback, RFC1918 / ULA and link-local addresses, and for non-IP hosts
    (the test client reports ``testclient``), which only a local caller can produce."""
    if not host:
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return True
    return ip.is_private or ip.is_loopback or ip.is_link_local


def is_public_request(request: Request) -> bool:
    """Did this request come from outside the LAN?"""
    headers = request.headers
    if headers.get("cf-connecting-ip"):
        return True  # arrived through Cloudflare (tunnel or proxy)
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if not _is_private_ip(first):
            return True
    client = request.client.host if request.client else None
    return not _is_private_ip(client)


def path_is_board_surface(path: str) -> bool:
    return path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES)


def _read_token_file(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        return None
    return first or None


@dataclass
class Esp32Auth:
    mode: str = "auto"
    shared: str | None = None
    device_tokens: dict[str, str] = field(default_factory=dict)  # device id -> token

    @classmethod
    def from_env(cls) -> Esp32Auth:
        mode = (os.environ.get("LOCI_ESP32_AUTH") or "auto").strip().lower()
        if mode not in ("auto", "always", "off"):
            mode = "auto"
        shared = (os.environ.get("LOCI_ESP32_TOKEN") or "").strip() or _read_token_file(
            os.environ.get("LOCI_ESP32_TOKEN_FILE")
        )
        per_device: dict[str, str] = {}
        for item in (os.environ.get("LOCI_ESP32_TOKENS") or "").split(","):
            if ":" in item:
                dev, tok = item.split(":", 1)
                if dev.strip() and tok.strip():
                    per_device[dev.strip()] = tok.strip()
        return cls(mode=mode, shared=shared, device_tokens=per_device)

    @property
    def configured(self) -> bool:
        return bool(self.shared or self.device_tokens)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def verify(self, token: str | None) -> str | None:
        """Device id for a valid token (``"*"`` for the shared one), else ``None``."""
        if not token:
            return None
        found: str | None = None
        # compare against every candidate so timing does not reveal which one matched
        for dev, tok in self.device_tokens.items():
            if hmac.compare_digest(tok.encode(), token.encode()):
                found = dev
        if self.shared and hmac.compare_digest(self.shared.encode(), token.encode()):
            found = found or SHARED
        return found

    @staticmethod
    def token_from(request: Request) -> str | None:
        tok = request.headers.get("x-device-token")
        if tok:
            return tok.strip()
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def required_for(self, request: Request) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "always":
            return True
        return is_public_request(request)

    async def dependency(self, request: Request) -> None:
        """FastAPI dependency: 401 when a token is needed and missing or wrong.

        On success ``request.state.esp32_device`` holds the device id pinned by a
        per-device token (``None`` for the shared token or when no token was needed).
        """
        request.state.esp32_device = None
        token = self.token_from(request)
        device = self.verify(token) if self.configured else None
        if device is None and (token or self.required_for(request)):
            if not self.configured:
                detail = "no device token configured on the host (set LOCI_ESP32_TOKEN)"
            else:
                detail = "device token missing or wrong"
            raise HTTPException(
                status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"}
            )
        if device and device != SHARED:
            request.state.esp32_device = device

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "configured": self.configured,
            "per_device": sorted(self.device_tokens),
        }


def public_surface_middleware(auth: Esp32Auth):
    """Serve only the board's routes to requests from outside the LAN."""

    async def middleware(request: Request, call_next):
        if (
            auth.enabled
            and not path_is_board_surface(request.url.path)
            and is_public_request(request)
        ):
            return JSONResponse({"detail": "not found"}, status_code=404)
        return await call_next(request)

    return middleware
