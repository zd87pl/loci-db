"""Device-token auth for the ESP32 API: the LAN stays open, the tunnel side needs the token,
and from outside only the board's routes exist."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.esp32_auth import Esp32Auth, is_public_request  # noqa: E402
from test_esp32_api import _build, _jpeg  # noqa: E402

PUBLIC = {"CF-Connecting-IP": "203.0.113.9"}  # what cloudflared adds to every forwarded request


def _req(headers: dict | None = None, client_host: str = "127.0.0.1"):
    return SimpleNamespace(
        headers={k.lower(): v for k, v in (headers or {}).items()},
        client=SimpleNamespace(host=client_host),
    )


# ---------------------------------------------------------------------------
# where does a request come from?
# ---------------------------------------------------------------------------


def test_lan_and_test_clients_are_private() -> None:
    assert not is_public_request(_req())
    assert not is_public_request(_req(client_host="192.168.1.5"))
    assert not is_public_request(_req(client_host="testclient"))
    assert not is_public_request(_req({"X-Forwarded-For": "10.0.0.7"}))


def test_tunnel_and_internet_clients_are_public() -> None:
    # (203.0.113.x is a documentation range that ipaddress treats as non-global, hence 8.8.8.8)
    assert is_public_request(_req(PUBLIC))
    assert is_public_request(_req(client_host="8.8.8.8"))
    assert is_public_request(_req({"X-Forwarded-For": "8.8.8.8, 10.0.0.1"}))


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def test_verify_shared_and_per_device_tokens() -> None:
    auth = Esp32Auth(shared="abc123", device_tokens={"pendant": "p-token"})
    assert auth.verify("abc123") == "*"
    assert auth.verify("p-token") == "pendant"
    assert auth.verify("nope") is None and auth.verify(None) is None and auth.verify("") is None
    assert auth.configured and auth.enabled


def test_from_env_reads_file_mode_and_per_device_list(monkeypatch, tmp_path) -> None:
    token_file = tmp_path / "device_token"
    token_file.write_text("file-token\n")
    monkeypatch.delenv("LOCI_ESP32_TOKEN", raising=False)
    monkeypatch.setenv("LOCI_ESP32_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("LOCI_ESP32_TOKENS", "pendant:p1, desk:d2,broken,:x,y:")
    monkeypatch.setenv("LOCI_ESP32_AUTH", "bogus")
    auth = Esp32Auth.from_env()
    assert auth.shared == "file-token"
    assert auth.device_tokens == {"pendant": "p1", "desk": "d2"}
    assert auth.mode == "auto"  # unknown modes fall back to auto
    monkeypatch.setenv("LOCI_ESP32_TOKEN", "env-wins")
    monkeypatch.setenv("LOCI_ESP32_AUTH", "ALWAYS")
    auth = Esp32Auth.from_env()
    assert auth.shared == "env-wins" and auth.mode == "always"


def test_from_env_without_anything_is_unconfigured(monkeypatch) -> None:
    for var in (
        "LOCI_ESP32_TOKEN",
        "LOCI_ESP32_TOKEN_FILE",
        "LOCI_ESP32_TOKENS",
        "LOCI_ESP32_AUTH",
    ):
        monkeypatch.delenv(var, raising=False)
    auth = Esp32Auth.from_env()
    assert not auth.configured and auth.mode == "auto"


# ---------------------------------------------------------------------------
# through the API
# ---------------------------------------------------------------------------


def test_lan_needs_no_token_but_a_wrong_one_is_refused() -> None:
    t = _build(auth=Esp32Auth(shared="s3cret"))
    assert t.client.get("/api/esp32/status").status_code == 200
    assert t.client.get("/api/esp32/status", headers={"X-Device-Token": "wrong"}).status_code == 401
    assert t.client.get("/api/esp32/status").json()["auth"] == {
        "mode": "auto",
        "configured": True,
        "per_device": [],
    }


def test_public_requests_need_the_token() -> None:
    t = _build(auth=Esp32Auth(shared="s3cret"))
    r = t.client.get("/api/esp32/status", headers=PUBLIC)
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer"
    assert (
        t.client.get(
            "/api/esp32/status", headers={**PUBLIC, "X-Device-Token": "s3cret"}
        ).status_code
        == 200
    )
    assert (
        t.client.get(
            "/api/esp32/status", headers={**PUBLIC, "Authorization": "Bearer s3cret"}
        ).status_code
        == 200
    )
    assert (
        t.client.get("/api/esp32/status", headers={**PUBLIC, "X-Device-Token": "nope"}).status_code
        == 401
    )
    # the board's real traffic
    r = t.client.post(
        "/api/esp32/frame",
        content=_jpeg(),
        headers={**PUBLIC, "Content-Type": "image/jpeg", "X-Device-Token": "s3cret"},
    )
    assert r.status_code == 200 and r.json()["labels"] == ["cup"]


def test_only_the_board_surface_exists_from_outside() -> None:
    t = _build(auth=Esp32Auth(shared="s3cret"))

    @t.app.get("/api/objects")
    async def objects():
        return {"objects": []}

    assert t.client.get("/api/objects").status_code == 200  # LAN: the whole demo
    assert t.client.get("/api/objects", headers=PUBLIC).status_code == 404  # tunnel: hidden
    assert t.client.get("/docs", headers=PUBLIC).status_code == 404
    assert t.client.get("/esp32", headers=PUBLIC).status_code == 200  # the page asks for the token
    assert t.client.get("/api/esp32/frame.jpg", headers=PUBLIC).status_code == 401


def test_per_device_token_pins_the_device_id() -> None:
    t = _build(auth=Esp32Auth(device_tokens={"pendant": "p-token"}))
    r = t.client.post(
        "/api/esp32/frame",
        content=_jpeg(),
        headers={
            **PUBLIC,
            "Content-Type": "image/jpeg",
            "X-Device-Id": "imposter",
            "X-Device-Token": "p-token",
        },
    )
    assert r.status_code == 200 and r.json()["device"] == "pendant"
    assert "pendant" in t.client.get("/api/esp32/status").json()["devices"]
    assert "imposter" not in t.client.get("/api/esp32/status").json()["devices"]
    # audio polling with the pinned id
    assert (
        t.client.get(
            "/api/esp32/audio/next", headers={**PUBLIC, "X-Device-Token": "p-token"}
        ).status_code
        == 204
    )


def test_always_mode_requires_the_token_on_the_lan_too() -> None:
    t = _build(auth=Esp32Auth(mode="always", shared="s3cret"))
    assert t.client.get("/api/esp32/status").status_code == 401
    assert (
        t.client.get("/api/esp32/status", headers={"X-Device-Token": "s3cret"}).status_code == 200
    )


def test_public_request_with_no_token_configured_fails_closed() -> None:
    t = _build(auth=Esp32Auth())
    assert t.client.get("/api/esp32/status").status_code == 200  # LAN still fine
    r = t.client.get("/api/esp32/status", headers=PUBLIC)
    assert r.status_code == 401 and "configured" in r.json()["detail"]


def test_off_mode_disables_token_and_surface_checks() -> None:
    t = _build(auth=Esp32Auth(mode="off"))

    @t.app.get("/api/objects")
    async def objects():
        return {"objects": []}

    assert t.client.get("/api/esp32/status", headers=PUBLIC).status_code == 200
    assert t.client.get("/api/objects", headers=PUBLIC).status_code == 200
