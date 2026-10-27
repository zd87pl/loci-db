"""Tests for demo_spatial/app/cloud_sync.py against a fake LOCI Cloud v1 server."""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.cloud_sync import (  # noqa: E402
    CloudConfig,
    CloudError,
    CloudSync,
    LociCloudClient,
    record_to_observation,
)

REF = "abcdefghij12"


class FakeCloud:
    """Records writes; issues device tokens; can be told to fail or expire tokens."""

    def __init__(self) -> None:
        self.observations: dict[str, dict] = {}
        self.batches: list[dict] = []
        self.rev = 0
        self.tokens_issued = 0
        self.valid_tokens: set[str] = {"loci_sk_secret"}
        self.fail_next = 0
        self.user_agents: set[str] = set()

    def handle(
        self, method: str, path: str, auth: str | None, body: dict | None
    ) -> tuple[int, dict]:
        prefix = f"/p/{REF}/v1/"
        if not path.startswith(prefix):
            return 404, {"detail": "not found"}
        rest = path[len(prefix) :]
        if rest == "devices/token" and method == "POST":
            assert body is not None
            if body.get("device_secret") != "loci_ds_good":
                return 401, {"detail": "invalid device credentials"}
            self.tokens_issued += 1
            tok = f"tok{self.tokens_issued}"
            self.valid_tokens.add(tok)
            return 200, {"token": tok, "expires_at": 4_000_000_000, "spaces": ["home"]}
        token = (auth or "").removeprefix("Bearer ")
        if token not in self.valid_tokens:
            return 401, {"detail": "invalid or expired device token"}
        if self.fail_next:
            self.fail_next -= 1
            return 503, {"detail": "unavailable"}
        if rest == "spaces/home/observations" and method == "POST":
            assert body is not None
            obs, dels = body.get("observations", []), body.get("deletes", [])
            assert len(obs) <= 50 and len(dels) <= 200
            self.batches.append(body)
            for o in obs:
                self.observations.setdefault(o["id"], o)
            for d in dels:
                self.observations.pop(d, None)
            self.rev += 1
            return 201, {
                "accepted": len(obs),
                "duplicates": 0,
                "deleted": len(dels),
                "rev": self.rev,
            }
        if rest.startswith("spaces/home/digest") and method == "GET":
            return 200, {"rev": self.rev, "full": True, "objects": [], "removed": []}
        if rest == "spaces/home/ask" and method == "POST":
            return 200, {
                "answer": "Your cup was last seen in the kitchen, just now.",
                "label": "cup",
            }
        return 404, {"detail": "not found"}


@pytest.fixture()
def cloud():
    fake = FakeCloud()

    class Handler(BaseHTTPRequestHandler):
        def _go(self, method: str) -> None:
            n = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(n)) if n else None
            fake.user_agents.add(self.headers.get("user-agent") or "")
            status, payload = fake.handle(
                method, self.path, self.headers.get("authorization"), body
            )
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._go("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._go("POST")

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"  # type: ignore[attr-defined]
    yield fake
    server.shutdown()


def rec(i: int, *, frame: str = "room", label: str = "cup", t: int | None = None) -> dict:
    meta: dict = {"label": label, "frame": frame, "anchored": True}
    if frame == "room":
        meta["pos_m"] = [-1.2 + i * 0.01, 0.8, 1.5]
    if frame == "camera":
        meta["xyz_cam_m"] = [0.3, -0.1, 1.9]
    return {
        "id": f"r{i}",
        "scene_id": f"kitchen/{label}" if frame == "room" else label,
        "label": label,
        "room": "kitchen" if frame == "room" else None,
        "frame": frame,
        "x": 0.3,
        "y": 0.32,
        "z": 0.25,
        "timestamp_ms": t if t is not None else 1_700_000_000_000 + i,
        "confidence": 0.9,
        "prev_state_id": None,
        "metadata": meta,
    }


class FakeMemory:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def export_records(self) -> list[dict]:
        return [dict(r) for r in self.records]


def secret_cfg(url: str) -> CloudConfig:
    return CloudConfig(base_url=url, project=REF, space="home", secret_key="loci_sk_secret")


def test_record_mapping_uses_metres_per_frame() -> None:
    room = record_to_observation(rec(0))
    assert room is not None
    assert (room["frame"], room["x"], room["z"], room["room"]) == ("room", -1.2, 1.5, "kitchen")
    assert room["id"] == "h_r0" and room["metadata"]["anchored"] is True
    cam = record_to_observation(rec(1, frame="camera"))
    assert cam is not None and (cam["frame"], cam["x"], cam["z"]) == ("camera", 0.3, 1.9)
    img = record_to_observation(rec(2, frame="image"))
    assert img is not None and (img["frame"], img["x"], img["y"]) == ("image", 0.3, 0.32)
    no_pos = rec(3)
    no_pos["metadata"].pop("pos_m")
    fallback = record_to_observation(no_pos)
    assert fallback is not None and fallback["frame"] == "image"
    assert record_to_observation({"id": "", "label": "cup"}) is None


def test_sync_pushes_new_records_in_batches_and_deletes_removed(cloud, tmp_path: Path) -> None:
    mem = FakeMemory([rec(i) for i in range(120)])
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=tmp_path / "s.json")
    first = sync.sync_once()
    assert first["sent"] == 120 and first["deleted"] == 0
    assert [len(b["observations"]) for b in cloud.batches] == [50, 50, 20]
    assert len(cloud.observations) == 120

    assert sync.sync_once() == {"sent": 0, "deleted": 0, "rev": 3}

    mem.records = mem.records[5:] + [rec(500)]  # merge: 5 records gone, one new
    second = sync.sync_once()
    assert second["sent"] == 1 and second["deleted"] == 5
    assert "h_r0" not in cloud.observations and "h_r500" in cloud.observations
    assert sync.status()["pushed"] == 116


def test_state_survives_restart_so_deletes_are_not_lost(cloud, tmp_path: Path) -> None:
    state = tmp_path / "sync.json"
    mem = FakeMemory([rec(i) for i in range(3)])
    CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state).sync_once()
    mem.records = mem.records[1:]
    restarted = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    out = restarted.sync_once()
    assert out == {"sent": 0, "deleted": 1, "rev": 2}
    assert set(cloud.observations) == {"h_r1", "h_r2"}


def test_device_credentials_fetch_and_refresh_tokens(cloud) -> None:
    cfg = CloudConfig(
        base_url=cloud.url,
        project=REF,
        space="home",
        device_id="dev_1",
        device_secret="loci_ds_good",
    )
    client = LociCloudClient(cfg)
    client.write([record_to_observation(rec(0))])
    assert cloud.tokens_issued == 1
    cloud.valid_tokens.discard("tok1")  # token superseded server-side
    client.write([record_to_observation(rec(1))])
    assert cloud.tokens_issued == 2
    assert client.ask("where is my cup")["label"] == "cup"

    bad = LociCloudClient(
        CloudConfig(
            base_url=cloud.url, project=REF, space="home", device_id="dev_1", device_secret="x"
        )
    )
    with pytest.raises(CloudError) as exc:
        bad.digest()
    assert exc.value.status == 401


def test_failures_are_recorded_and_retried_without_losing_records(cloud) -> None:
    mem = FakeMemory([rec(i) for i in range(3)])
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)))
    cloud.fail_next = 1
    with pytest.raises(CloudError):
        sync.sync_once()
    assert sync.status()["pushed"] == 0
    assert sync.sync_once()["sent"] == 3


def test_background_thread_records_errors_and_stops(cloud) -> None:
    sync = CloudSync(FakeMemory([rec(0)]), LociCloudClient(secret_cfg(cloud.url)), interval_s=0.05)
    cloud.fail_next = 1
    sync.start()
    deadline = threading.Event()
    for _ in range(100):
        if sync.status()["pushed"] == 1:
            break
        deadline.wait(0.05)
    sync.stop()
    assert sync.status()["pushed"] == 1
    assert sync.status()["last_error"] is None


def test_config_from_env() -> None:
    assert CloudConfig.from_env({}) is None
    env = {"LOCI_CLOUD_URL": "http://x", "LOCI_CLOUD_PROJECT": REF, "LOCI_CLOUD_SPACE": "home"}
    with pytest.raises(ValueError):
        CloudConfig.from_env(env)
    cfg = CloudConfig.from_env({**env, "LOCI_CLOUD_KEY": "loci_sk_x", "LOCI_CLOUD_SYNC_S": "2"})
    assert cfg is not None and cfg.interval_s == 2.0
    with pytest.raises(ValueError):
        CloudConfig.from_env({**env, "LOCI_CLOUD_URL": "file:///etc/passwd", "LOCI_CLOUD_KEY": "k"})


def test_requests_send_an_explicit_user_agent(cloud) -> None:
    # Cloudflare's edge rejects the default "Python-urllib/x.y" agent with error 1010.
    LociCloudClient(secret_cfg(cloud.url)).digest()
    assert cloud.user_agents and all(ua.startswith("loci-cloud-sync/") for ua in cloud.user_agents)
