"""The ESP32 host's uvicorn protocol: a board that vanishes must not keep its socket open.

uvicorn only times out a connection between requests. ``Esp32HttpProtocol`` adds TCP keepalive
(a peer that went silent is dropped by the OS) and a timeout for the first request. These tests run
a real uvicorn server on 127.0.0.1; no hardware.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time

import pytest
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.esp32_http import Esp32HttpProtocol, enable_tcp_keepalive  # noqa: E402

_IDLE_OPT = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)


def test_enable_tcp_keepalive_sets_the_timing() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        assert enable_tcp_keepalive(sock, idle_s=7, interval_s=3, count=4)
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
        if _IDLE_OPT is not None:
            assert sock.getsockopt(socket.IPPROTO_TCP, _IDLE_OPT) == 7
        if hasattr(socket, "TCP_KEEPINTVL"):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 3
        if hasattr(socket, "TCP_KEEPCNT"):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 4


async def _app(scope, receive, send):  # minimal ASGI app: 200 "ok"
    if scope["type"] != "http":
        return
    await send(
        {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]}
    )
    await send({"type": "http.response.body", "body": b"ok"})


@pytest.fixture()
def server():
    """uvicorn with Esp32HttpProtocol on a free port, keep-alive timeout 1 s."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(
        _app,
        http=Esp32HttpProtocol,
        lifespan="off",
        log_level="warning",
        timeout_keep_alive=1,
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    assert srv.started, "uvicorn did not start"
    yield sock.getsockname()[1]
    srv.should_exit = True
    thread.join(timeout=5)
    sock.close()


def _closed_within(conn: socket.socket, seconds: float) -> bool:
    """True when the server closes the connection (EOF or reset) within `seconds`."""
    conn.settimeout(seconds)
    try:
        return conn.recv(1024) == b""
    except ConnectionResetError:
        return True
    except TimeoutError:
        return False


def test_connection_without_a_request_is_closed(server: int) -> None:
    # Plain uvicorn keeps this one open forever (its keep-alive timer starts after a response).
    with socket.create_connection(("127.0.0.1", server)) as conn:
        t0 = time.monotonic()
        assert _closed_within(conn, 5.0)
        assert time.monotonic() - t0 < 3.0


def test_requests_and_keep_alive_still_work(server: int) -> None:
    with socket.create_connection(("127.0.0.1", server)) as conn:
        for _ in range(3):  # three requests on one keep-alive connection, as the boards do
            conn.sendall(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
            conn.settimeout(3.0)
            reply = b""
            while not reply.endswith(b"ok"):
                chunk = conn.recv(1024)
                assert chunk, "connection closed early"
                reply += chunk
            assert reply.startswith(b"HTTP/1.1 200")
        # and the usual keep-alive timeout still closes it once it is idle
        assert _closed_within(conn, 5.0)


def test_accepted_sockets_have_tcp_keepalive(server: int) -> None:
    # The server side of the connection is what matters; observe it through the protocol class.
    seen: list[socket.socket] = []
    original = Esp32HttpProtocol.connection_made

    def spy(self, transport):
        seen.append(transport.get_extra_info("socket"))
        original(self, transport)

    Esp32HttpProtocol.connection_made = spy  # type: ignore[method-assign]
    try:
        with socket.create_connection(("127.0.0.1", server)) as conn:
            conn.sendall(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
            conn.settimeout(3.0)
            assert conn.recv(1024).startswith(b"HTTP/1.1 200")
            assert seen, "connection_made was not called"
            assert seen[-1].getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0
    finally:
        Esp32HttpProtocol.connection_made = original  # type: ignore[method-assign]
