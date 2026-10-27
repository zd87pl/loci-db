"""uvicorn HTTP protocol for the ESP32 host: a board that vanishes must not keep its socket open.

uvicorn arms its keep-alive timeout only after it has sent a response. A connection that never
sends a request, or a request whose headers or body stop arriving, is never timed out, and when the
client disappears without closing the connection (an ESP32 that resets, browns out or drops off
Wi-Fi mid-upload sends no FIN) the socket stays ESTABLISHED for good. On the desk on 2026-10-26
about twenty board resets and a weak link left 241 such sockets; the host hit launchd's default
limit of 256 open files and reset every new connection, the other board's and the Cloudflare
tunnel's included, until it was restarted.

``run_esp32_demo.sh`` starts uvicorn with ``--http demo_spatial.app.esp32_http:Esp32HttpProtocol``,
which differs from uvicorn's own protocol in two ways:

* every accepted socket gets TCP keepalive, so the OS probes a peer that has gone silent and drops
  the connection once it stops answering. A board that rebooted answers the first probe with a
  reset; one that is gone is dropped after ``idle + count * interval`` seconds (35 s by default).
  The handler waiting for that body then sees a disconnect and the file descriptor is freed.
* uvicorn's keep-alive timer is armed when the connection opens, so a client has to start its first
  request within ``--timeout-keep-alive`` seconds, as it already must between requests.

Nothing changes for a board that behaves: probes go out only after ``LOCI_HTTP_KEEPIDLE`` seconds
without traffic, and the boards send a request as soon as they connect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket

try:  # uvicorn[standard] (fastapi[standard]) brings httptools; plain uvicorn only has h11
    from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol as _BaseProtocol
except ImportError:  # pragma: no cover - depends on the installed extras
    from uvicorn.protocols.http.h11_impl import H11Protocol as _BaseProtocol

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


KEEPIDLE_S = _env_int("LOCI_HTTP_KEEPIDLE", 20)  # silence before the first probe
KEEPINTVL_S = _env_int("LOCI_HTTP_KEEPINTVL", 5)  # between unanswered probes
KEEPCNT = _env_int("LOCI_HTTP_KEEPCNT", 3)  # unanswered probes before the connection is dropped


def enable_tcp_keepalive(
    sock: socket.socket,
    idle_s: int = KEEPIDLE_S,
    interval_s: int = KEEPINTVL_S,
    count: int = KEEPCNT,
) -> bool:
    """Turn on TCP keepalive with the given timing (Linux and macOS names). False if refused."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        return False
    # Linux calls the idle time TCP_KEEPIDLE, macOS TCP_KEEPALIVE
    idle_opt = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)
    for opt, value in (
        (idle_opt, idle_s),
        (getattr(socket, "TCP_KEEPINTVL", None), interval_s),
        (getattr(socket, "TCP_KEEPCNT", None), count),
    ):
        if opt is not None:
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.IPPROTO_TCP, opt, value)
    return True


class Esp32HttpProtocol(_BaseProtocol):
    """uvicorn's HTTP protocol plus TCP keepalive and a timeout for the first request."""

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        super().connection_made(transport)
        sock = transport.get_extra_info("socket")
        is_tcp = sock is not None and sock.family in (socket.AF_INET, socket.AF_INET6)
        if is_tcp and not enable_tcp_keepalive(sock):
            logger.warning("could not enable TCP keepalive for %s", self.client)
        # uvicorn arms this timer only after a response; arm it now so a connection that never
        # sends a request is closed like an idle keep-alive one. Incoming data cancels it.
        if self.timeout_keep_alive_task is None:
            self.timeout_keep_alive_task = self.loop.call_later(
                self.timeout_keep_alive, self.timeout_keep_alive_handler
            )
