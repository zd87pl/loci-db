"""Tests for demo_spatial/app/cloud_sync.py against a fake LOCI Cloud v1 server."""

from __future__ import annotations

import json
import logging
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
from app.persistence import MemoryPersistence  # noqa: E402
from app.room_anchor import RoomMap  # noqa: E402
from app.spatial_memory import SpatialMemory  # noqa: E402

REF = "abcdefghij12"


class FakeCloud:
    """Records writes; issues device tokens; can be told to fail or expire tokens.

    Writes follow the Worker: deletes first, then inserts that ignore known ids, or with
    ``"upsert": true`` updates of known ids when label and room match and ``t`` does not
    go back (the Worker's guard). ``supports_upsert = False`` plays an older Worker,
    which ignores the flag and has no ``updated`` / ``conflicts`` in its reply.
    """

    def __init__(self) -> None:
        self.observations: dict[str, dict] = {}
        self.batches: list[dict] = []
        self.rev = 0
        self.tokens_issued = 0
        self.valid_tokens: set[str] = {"loci_sk_secret"}
        self.fail_next = 0
        self.user_agents: set[str] = set()
        self.supports_upsert = True

    def _write(self, body: dict) -> dict:
        obs, dels = body.get("observations", []), body.get("deletes", [])
        upsert = bool(body.get("upsert")) and self.supports_upsert
        for d in dels:
            self.observations.pop(d, None)
        accepted = duplicates = updated = conflicts = 0
        for o in obs:
            old = self.observations.get(o["id"])
            if old is None:
                self.observations[o["id"]] = o
                accepted += 1
            elif not upsert or old == o:
                duplicates += 1
            elif (old["label"], old.get("room")) != (o["label"], o.get("room")) or o["t"] < old[
                "t"
            ]:
                conflicts += 1
            else:
                self.observations[o["id"]] = o
                updated += 1
        self.rev += 1
        reply = {"accepted": accepted, "duplicates": duplicates, "deleted": len(dels)}
        if self.supports_upsert:
            reply.update(updated=updated, conflicts=conflicts)
        return {**reply, "rev": self.rev}

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
            return 201, self._write(body)
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


def secret_cfg(url: str, **limits) -> CloudConfig:
    return CloudConfig(
        base_url=url, project=REF, space="home", secret_key="loci_sk_secret", **limits
    )


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

    assert sync.sync_once() == {"sent": 0, "updated": 0, "deleted": 0, "held": 0, "rev": 3}

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
    assert out == {"sent": 0, "updated": 0, "deleted": 1, "held": 0, "rev": 2}
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
    assert (cfg.max_delete_fraction, cfg.max_delete_count) == (0.5, 200)
    limits = {"LOCI_CLOUD_MAX_DELETE_FRACTION": "0.2", "LOCI_CLOUD_MAX_DELETE_COUNT": "50"}
    tuned = CloudConfig.from_env({**env, "LOCI_CLOUD_KEY": "k", **limits})
    assert tuned is not None
    assert (tuned.max_delete_fraction, tuned.max_delete_count) == (0.2, 50)
    for bad in ({"LOCI_CLOUD_MAX_DELETE_FRACTION": "1.5"}, {"LOCI_CLOUD_MAX_DELETE_COUNT": "-1"}):
        with pytest.raises(ValueError):
            CloudConfig.from_env({**env, "LOCI_CLOUD_KEY": "k", **bad})
    with pytest.raises(ValueError):
        CloudConfig.from_env({**env, "LOCI_CLOUD_URL": "file:///etc/passwd", "LOCI_CLOUD_KEY": "k"})


def test_requests_send_an_explicit_user_agent(cloud) -> None:
    # Cloudflare's edge rejects the default "Python-urllib/x.y" agent with error 1010.
    LociCloudClient(secret_cfg(cloud.url)).digest()
    assert cloud.user_agents and all(ua.startswith("loci-cloud-sync/") for ua in cloud.user_agents)


# ---------------------------------------------------------------------------
# Mass-delete breaker
# ---------------------------------------------------------------------------

LIVE = 1225  # acknowledged sightings in the live host's data dir


def _seeded(cloud, tmp_path: Path, n: int = LIVE) -> tuple[FakeMemory, Path]:
    """Upload ``n`` sightings through a first host run; return its memory and state file."""
    state = tmp_path / "cloud_sync.json"
    mem = FakeMemory([rec(i) for i in range(n)])
    CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state).sync_once()
    assert len(cloud.observations) == n
    cloud.batches.clear()
    return mem, state


def _deletes_sent(cloud) -> int:
    return sum(len(b.get("deletes", [])) for b in cloud.batches)


def test_empty_restore_sends_no_deletes_and_keeps_the_breaker_open(
    cloud, tmp_path: Path, caplog
) -> None:
    mem, state = _seeded(cloud, tmp_path)
    mem.records = []  # failed restore: the host came back with an empty memory
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    with caplog.at_level(logging.WARNING, logger="app.cloud_sync"):
        for _ in range(3):
            out = sync.sync_once()
            assert (out["sent"], out["deleted"], out["held"]) == (0, 0, LIVE)
    assert cloud.batches == [] and len(cloud.observations) == LIVE
    trips = [r.getMessage() for r in caplog.records if "breaker OPEN" in r.getMessage()]
    assert len(trips) == 1 and "local memory is empty" in trips[0]
    status = sync.status()
    assert (status["delete_breaker"], status["deletes_held"]) == ("open", LIVE)
    assert status["delete_breaker_since_ms"] is not None

    # new sightings still go up while the deletes wait
    mem.records = [rec(5000 + i) for i in range(3)]
    out = sync.sync_once()
    assert (out["sent"], out["deleted"], out["held"]) == (3, 0, LIVE)
    assert len(cloud.observations) == LIVE + 3

    # a restart does not close it
    restarted = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    assert restarted.status()["delete_breaker"] == "open"
    assert restarted.sync_once()["held"] == LIVE
    assert _deletes_sent(cloud) == 0


def test_restart_with_fresh_record_ids_keeps_the_cloud_ids(cloud, tmp_path: Path) -> None:
    # CLOUD-1: restore gives every record a new id but keeps its sighting id, so the cloud
    # sees the same 1,225 sightings after a restart: nothing to send, update or delete.
    mem, state = _seeded(cloud, tmp_path)
    mem.records = [
        {**rec(i), "id": f"new{i}", "metadata": {**rec(i)["metadata"], "sighting_id": f"r{i}"}}
        for i in range(LIVE)
    ]
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    out = sync.sync_once()
    assert (out["sent"], out["updated"], out["deleted"], out["held"]) == (0, 0, 0, 0)
    assert cloud.batches == [] and len(cloud.observations) == LIVE


@pytest.mark.parametrize(
    ("acked", "kept", "limits"),
    [
        (1000, 799, {}),  # 201 deletes in one pass: over the count limit
        (120, 50, {}),  # 70 of 120 (58 %): over the share limit
        (3, 0, {}),  # empty local memory, even below BREAKER_MIN_DELETES
        (120, 105, {"max_delete_count": 10}),  # tuned limit
    ],
)
def test_breaker_holds_large_or_total_deletes(cloud, acked, kept, limits) -> None:
    mem = FakeMemory([rec(i) for i in range(acked)])
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url, **limits)))
    sync.sync_once()
    mem.records = mem.records[:kept]
    out = sync.sync_once()
    assert (out["deleted"], out["held"]) == (0, acked - kept)
    assert sync.status()["delete_breaker"] == "open"
    assert len(cloud.observations) == acked


@pytest.mark.parametrize(
    ("acked", "kept"),
    [
        (LIVE, LIVE - 12),  # a few merges in a live-sized space
        (300, 150),  # exactly half is not more than half
        (6, 2),  # small space: 4 deletes are below BREAKER_MIN_DELETES
    ],
)
def test_normal_passes_still_delete(cloud, acked, kept) -> None:
    mem = FakeMemory([rec(i) for i in range(acked)])
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)))
    sync.sync_once()
    mem.records = mem.records[:kept]
    out = sync.sync_once()
    assert (out["deleted"], out["held"]) == (acked - kept, 0)
    assert sync.status()["delete_breaker"] == "closed"
    assert len(cloud.observations) == kept


def test_accepting_lets_the_large_delete_through(cloud, tmp_path: Path) -> None:
    mem, state = _seeded(cloud, tmp_path)
    mem.records = []
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    sync.sync_once()
    with pytest.raises(ValueError, match="1225 cloud deletes are held, not 1000"):
        sync.accept_deletes(expect=1000)
    assert sync.status()["delete_breaker"] == "open"
    assert sync.accept_deletes(expect=LIVE) == {"accepted": LIVE}
    assert sync.status()["delete_breaker"] == "closed"

    # the acceptance is saved: it holds even if the host restarts before the next pass
    restarted = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    out = restarted.sync_once()
    assert (out["deleted"], out["held"]) == (LIVE, 0)
    assert [len(b["deletes"]) for b in cloud.batches] == [200] * 6 + [25]
    assert cloud.observations == {}
    status = restarted.status()
    assert (status["pushed"], status["deletes_accepted"]) == (0, 0)
    with pytest.raises(ValueError, match="no cloud deletes are held"):
        restarted.accept_deletes()


def test_acceptance_covers_only_the_deletes_that_were_checked(cloud, tmp_path: Path) -> None:
    mem, state = _seeded(cloud, tmp_path, n=300)
    mem.records = mem.records[:50]
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    assert sync.sync_once()["held"] == 250
    sync.accept_deletes()
    mem.records = []  # the rest vanished after the human looked
    out = sync.sync_once()
    assert (out["deleted"], out["held"]) == (250, 50)
    assert sync.status()["delete_breaker"] == "open"
    assert len(cloud.observations) == 50


# ---------------------------------------------------------------------------
# Stable sighting ids (CLOUD-1)
# ---------------------------------------------------------------------------


def test_cloud_id_comes_from_the_sighting_id_and_the_room_is_kept() -> None:
    r = rec(0, frame="image")
    r["id"] = "r_after_merge"
    r["metadata"].update(sighting_id="first_id", last_room="kitchen")
    obs = record_to_observation(r)
    assert obs is not None and obs["id"] == "h_first_id"
    assert obs["metadata"]["last_room"] == "kitchen"


def test_a_changed_sighting_is_upserted_not_deleted_and_reinserted(cloud) -> None:
    mem = FakeMemory([rec(i) for i in range(3)])
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)))
    sync.sync_once()
    merged = rec(0, t=1_700_000_009_000)
    merged["id"] = "r0_merged"  # a merge re-ids the record ...
    merged["metadata"].update(sighting_id="r0", merge_count=1, pos_m=[-1.0, 0.8, 1.5])
    mem.records = [merged, *mem.records[1:]]
    out = sync.sync_once()
    assert (out["sent"], out["updated"], out["deleted"]) == (0, 1, 0)
    assert cloud.batches[-1]["upsert"] is True and "deletes" not in cloud.batches[-1]
    assert cloud.observations["h_r0"]["t"] == 1_700_000_009_000
    assert cloud.observations["h_r0"]["x"] == -1.0 and len(cloud.observations) == 3
    assert sync.sync_once() == {"sent": 0, "updated": 0, "deleted": 0, "held": 0, "rev": 2}
    assert sync.status()["updated_total"] == 1


def test_merge_and_restart_through_spatial_memory_keep_the_cloud_ids(cloud, tmp_path) -> None:
    now = 1_760_000_000_000

    def boot() -> SpatialMemory:
        p = MemoryPersistence(tmp_path / "mem", snapshot_interval_s=1e9, journal_fsync_s=0.02)
        mem = SpatialMemory(epoch_size_ms=5000, persistence=p, rooms=RoomMap.default())
        mem.restore()
        return mem

    def see(mem: SpatialMemory, label: str, x: float, t: int) -> None:
        mem.observe(
            label,
            cx=0.5,
            cy=0.5,
            confidence=0.9,
            timestamp_ms=t,
            min_confidence=0.0,
            pos_room_m=(x, 0.8, 1.0),
            room="kitchen",
            anchored=True,
            anchor_confidence=1.0,
        )

    state = tmp_path / "cloud_sync.json"
    mem = boot()
    see(mem, "cup", 1.0, now)
    see(mem, "bottle", -0.6, now + 1000)
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    assert sync.sync_once()["sent"] == 2
    ids = set(cloud.observations)
    see(mem, "cup", 1.1, now + 1500)  # merge: the record gets a new id
    out = sync.sync_once()
    assert (out["sent"], out["updated"], out["deleted"]) == (0, 1, 0)
    assert set(cloud.observations) == ids
    mem.close()

    mem2 = boot()  # restart: restore re-ids every record
    restarted = CloudSync(mem2, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    out = restarted.sync_once()
    assert (out["sent"], out["updated"], out["deleted"], out["held"]) == (0, 0, 0, 0)
    assert set(cloud.observations) == ids
    assert restarted.status()["delete_breaker"] == "closed"
    mem2.close()


def test_a_state_file_from_before_fingerprints_is_adopted_without_resending(
    cloud, tmp_path: Path
) -> None:
    state = tmp_path / "cloud_sync.json"
    mem = FakeMemory([rec(i) for i in range(5)])
    CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state).sync_once()
    data = json.loads(state.read_text())
    data["pushed"] = sorted(data["pushed"])  # the format before fingerprints: ids only
    state.write_text(json.dumps(data))
    cloud.batches.clear()
    mem.records[0]["metadata"]["last_room"] = "kitchen"  # content the old build never sent
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)), state_path=state)
    assert sync.sync_once() == {"sent": 0, "updated": 0, "deleted": 0, "held": 0, "rev": None}
    assert cloud.batches == []
    assert sorted(json.loads(state.read_text())["pushed"].values())[0]  # fingerprints now


def test_a_cloud_that_ignores_upserts_is_reported_and_tried_again_later(cloud) -> None:
    cloud.supports_upsert = False  # a Worker from before the upsert support
    mem = FakeMemory([rec(i) for i in range(2)])
    sync = CloudSync(mem, LociCloudClient(secret_cfg(cloud.url)))
    sync.sync_once()
    mem.records[0] = {**mem.records[0], "timestamp_ms": 1_700_000_009_000}
    assert sync.sync_once()["updated"] == 0
    assert "does not apply upserts" in (sync.status()["last_error"] or "")
    batches = len(cloud.batches)
    sync.sync_once()
    assert len(cloud.batches) == batches  # no re-send on every pass while it waits
    cloud.supports_upsert = True
    sync._upsert_retry_at = 0.0  # the hour is up
    assert sync.sync_once()["updated"] == 1
    assert sync.status()["last_error"] is None
    assert cloud.observations["h_r0"]["t"] == 1_700_000_009_000
