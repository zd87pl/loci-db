"""Cloud conformance: golden queries shared with the Cloudflare Worker.

``cloud/workers/test/conformance-golden.json`` lists states, queries and the
expected result keys in order.  This module checks that

* the local library (``LocalLociClient``) produces exactly those results, and
* ``LociClient`` / ``AsyncLociClient`` in cloud mode produce them over HTTP,
  against a fake Worker that speaks the compatibility contract of
  ``cloud/workers/src/compat.ts`` (``POST /p/<ref>/insert`` and ``/query``,
  float32 vectors, the same request and result keys).

``cloud/workers/test/conformance.test.ts`` runs the same file through the real
Worker under Miniflare, so the library and the Worker agree on ids, ordering
and filters.  After an intended change to the ranking, regenerate the
expectations from the library with::

    LOCI_REGEN_GOLDEN=1 pytest tests/test_cloud_conformance.py
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from loci import AsyncLociClient, LocalLociClient, LociClient, WorldState, cloud_transport
from loci.cloud_transport import CloudTransport, _insert_payload, _query_payload
from loci.temporal.consolidation import data_collection_name

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "cloud/workers/test/conformance-golden.json"
GOLDEN: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
REF = "conformance1"
KEY = f"loci_sk_{REF}_{'0' * 48}"
MISSING_ID = "00000000-0000-0000-0000-000000000000"


def _label(state: dict) -> str:
    """The Worker files a state under its normalised metadata label."""
    label = (state.get("metadata") or {}).get("label")
    return " ".join(label.strip().lower().split()) if isinstance(label, str) else ""


def _world_state(state: dict, now: int) -> WorldState:
    return WorldState(
        x=state["x"],
        y=state["y"],
        z=state["z"],
        timestamp_ms=now + state["dt_ms"],
        vector=[float(v) for v in state["vector"]],
        scene_id=state["scene_id"],
        scale_level=state.get("scale_level", "patch"),
        confidence=state["confidence"],
        metadata=state.get("metadata", {}),
        prev_state_id=state.get("prev_state_id"),
        next_state_id=state.get("next_state_id"),
    )


def _query_kwargs(query: dict, now: int) -> dict[str, Any]:
    window = query.get("window_dt_ms")
    return {
        "vector": [float(v) for v in query["vector"]],
        "spatial_bounds": query.get("bounds"),
        "time_window_ms": (now + window[0], now + window[1]) if window else None,
        "limit": query["limit"],
        "min_confidence": query.get("min_confidence"),
    }


def _sdk_queries() -> list[dict]:
    return [q for q in GOLDEN["queries"] if "labels" not in q]


def _keys(results: list[WorldState], key_of: dict[str, str]) -> list[str]:
    return [key_of[s.id] for s in results]


def _links(client: Any, key_of: dict[str, str], n: int) -> dict[str, dict]:
    everything = client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=100)
    assert len(everything) == n
    return {
        key_of[s.id]: {
            "prev": key_of.get(s.prev_state_id, s.prev_state_id),
            "next": key_of.get(s.next_state_id, s.next_state_id),
        }
        for s in sorted(everything, key=lambda s: key_of[s.id])
    }


def _run_m1(client: Any, now: int, key_of: dict[str, str]) -> dict[str, Any]:
    """After the base states and queries: scores, the batch, trajectories, causal context."""
    q = next(q for q in GOLDEN["queries"] if q["name"] == GOLDEN["scored"]["query"])
    scores = [[key_of[r.state.id], r.score] for r in client.query_scored(**_query_kwargs(q, now))]
    batch = GOLDEN["batch"]["states"]
    ids = client.insert_batch([_world_state(s, now) for s in batch])
    key_of.update({state_id: s["key"] for state_id, s in zip(ids, batch, strict=True)})
    id_of = {key: state_id for state_id, key in key_of.items()}

    def anchor(key: str) -> str:
        return id_of.get(key, MISSING_ID)

    return {
        "scores": scores,
        "batch_links": _links(client, key_of, len(GOLDEN["states"]) + len(batch)),
        "trajectories": [
            _keys(
                client.get_trajectory(anchor(t["anchor"]), t["steps_back"], t["steps_forward"]),
                key_of,
            )
            for t in GOLDEN["trajectories"]
        ],
        "causal": [
            _keys(client.get_causal_context(anchor(c["anchor"]), c["window_ms"]), key_of)
            for c in GOLDEN["causal"]
        ],
    }


def _golden_m1() -> dict[str, Any]:
    return {
        "scores": GOLDEN["scored"]["scores"],
        "batch_links": GOLDEN["batch"]["links"],
        "trajectories": [t["expect"] for t in GOLDEN["trajectories"]],
        "causal": [c["expect"] for c in GOLDEN["causal"]],
    }


def _library_addresses() -> list[dict[str, int]]:
    """The 4D address the library stores for each address case."""
    spec = GOLDEN["addresses"]
    client = LocalLociClient(vector_size=1, epoch_size_ms=spec["epoch_ms"])
    out = []
    for a in spec["states"]:
        state = WorldState(
            x=a["x"], y=a["y"], z=a["z"], timestamp_ms=a["timestamp_ms"], vector=[1.0]
        )
        stored = client.store.retrieve(data_collection_name(""), [client.insert(state)])
        payload = stored[0]["payload"]
        address = {"epoch": a["timestamp_ms"] // spec["epoch_ms"]}
        out.append(address | {f"r{r}": payload[f"hilbert_r{r}"] for r in (4, 8, 12)})
    return out


def _assert_m1(got: dict[str, Any]) -> None:
    want = _golden_m1()
    assert [k for k, _ in got["scores"]] == [k for k, _ in want["scores"]]
    got_scores = [s for _, s in got["scores"]]
    assert got_scores == pytest.approx([s for _, s in want["scores"]], abs=1e-9)
    keys = ("batch_links", "trajectories", "causal")
    assert {k: got[k] for k in keys} == {k: want[k] for k in keys}


def _run_library(now: int) -> tuple[dict[str, list[str]], dict[str, dict], dict[str, Any]]:
    client = LocalLociClient(vector_size=GOLDEN["vector_size"])
    key_of = {client.insert(_world_state(s, now)): s["key"] for s in GOLDEN["states"]}
    results: dict[str, list[str]] = {}
    for q in GOLDEN["queries"]:
        if "labels" in q:
            # No label filter in the library: run the query on the labelled subset instead.
            subset = LocalLociClient(vector_size=GOLDEN["vector_size"])
            sub_key_of = {
                subset.insert(_world_state(s, now)): s["key"]
                for s in GOLDEN["states"]
                if _label(s) in q["labels"]
            }
            results[q["name"]] = _keys(subset.query(**_query_kwargs(q, now)), sub_key_of)
        else:
            results[q["name"]] = _keys(client.query(**_query_kwargs(q, now)), key_of)
    links = _links(client, key_of, len(GOLDEN["states"]))
    return results, links, _run_m1(client, now, key_of)


def _dump(v: Any, indent: str = "") -> str:
    """One state, query or link per line, so diffs of expectations stay readable."""
    inner = indent + "  "
    if isinstance(v, list) and v and all(isinstance(x, (dict, str)) for x in v):
        return "[\n" + ",\n".join(inner + json.dumps(x) for x in v) + "\n" + indent + "]"
    if isinstance(v, dict) and any(isinstance(x, (dict, list)) for x in v.values()):
        body = ",\n".join(f"{inner}{json.dumps(k)}: {_dump(x, inner)}" for k, x in v.items())
        return "{\n" + body + "\n" + indent + "}"
    return json.dumps(v)


def test_local_library_matches_golden():
    now = int(time.time() * 1000)
    results, links, m1 = _run_library(now)
    addresses = _library_addresses()
    if os.environ.get("LOCI_REGEN_GOLDEN"):
        for q in GOLDEN["queries"]:
            q["expect"] = results[q["name"]]
        GOLDEN["links"] = links
        GOLDEN["scored"]["scores"] = m1["scores"]
        GOLDEN["batch"]["links"] = m1["batch_links"]
        for t, keys in zip(GOLDEN["trajectories"], m1["trajectories"], strict=True):
            t["expect"] = keys
        for c, keys in zip(GOLDEN["causal"], m1["causal"], strict=True):
            c["expect"] = keys
        for a, address in zip(GOLDEN["addresses"]["states"], addresses, strict=True):
            a["expect"] = address
        GOLDEN_PATH.write_text(_dump(GOLDEN) + "\n")
    assert results == {q["name"]: q["expect"] for q in GOLDEN["queries"]}
    assert links == GOLDEN["links"]
    _assert_m1(m1)
    assert addresses == [a["expect"] for a in GOLDEN["addresses"]["states"]]


def test_wire_keys_match_the_sdk():
    wire = GOLDEN["wire"]
    full = WorldState(
        x=0.1,
        y=0.2,
        z=0.3,
        timestamp_ms=1,
        vector=[1.0],
        scene_id="s",
        metadata={"label": "cup"},
        prev_state_id="p",
        next_state_id="n",
    )
    assert sorted(_insert_payload(full)) == wire["insert_keys"]
    bounds = {k: 0.0 for k in ("x_min", "y_min", "z_min")} | {
        k: 1.0 for k in ("x_max", "y_max", "z_max")
    }
    query = _query_payload([1.0], bounds, (0, 1), 5, 1.2, False, 0.5)
    assert sorted(query) == wire["query_keys"]
    # The Worker returns every WorldState field; _parse_query_results reads them all.
    assert sorted(WorldState.__dataclass_fields__) == wire["result_keys"]


# ── Fake Worker: the compat contract of cloud/workers/src/compat.ts ─────────


def _f32(values: list[float]) -> list[float]:
    """The Worker stores SDK vectors as float32."""
    return list(struct.unpack(f"{len(values)}f", struct.pack(f"{len(values)}f", *values)))


class _FakeWorker:
    def __init__(self) -> None:
        self.store = LocalLociClient(vector_size=GOLDEN["vector_size"])
        self.requests: list[dict[str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "ua": self.headers.get("User-Agent", ""),
                        "body": body,
                    }
                )
                actions = "insert|insert_batch|query|trajectory|causal_context"
                m = re.fullmatch(rf"/p/{REF}/({actions})", self.path)
                if m is None:
                    return self._send(404, {"detail": "not found", "code": "not_found"})
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self._send(401, {"detail": "invalid key", "code": "unauthenticated"})
                return self._send(200, getattr(fake, m[1])(body))

            def _send(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/p/{REF}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def insert(self, body: dict) -> dict:
        assert set(body) <= set(GOLDEN["wire"]["insert_keys"])
        return {"id": self.store.insert(WorldState(**body))}

    def insert_batch(self, body: dict) -> dict:
        # One request holds the whole batch here, so there are no chains to continue.
        assert set(body) == {"states", "chains"} and body["chains"] == {}
        states = [WorldState(**st) for st in body["states"]]
        ids = self.store.insert_batch(states)
        order = sorted(
            range(len(states)), key=lambda i: (states[i].scene_id, states[i].timestamp_ms)
        )
        return {"ids": ids, "chains": {states[i].scene_id: ids[i] for i in order}}

    def _results(self, states: list[WorldState], body: dict) -> dict:
        results = []
        for s in states:
            r = {k: getattr(s, k) for k in GOLDEN["wire"]["result_keys"]}
            r["vector"] = _f32(s.vector) if body.get("include_vectors", True) else []
            results.append(r)
        return {"results": results}

    def trajectory(self, body: dict) -> dict:
        states = self.store.get_trajectory(
            body["state_id"], body["steps_back"], body["steps_forward"]
        )
        return self._results(states, body)

    def causal_context(self, body: dict) -> dict:
        states = self.store.get_causal_context(body["state_id"], body["window_ms"])
        return self._results(states, body)

    def query(self, body: dict) -> dict:
        assert set(body) <= set(GOLDEN["wire"]["query_keys"]) | {"include_scores"}
        has_window = body.get("time_start_ms") is not None and body.get("time_end_ms") is not None
        hits = self.store.query_scored(
            vector=body["vector"],
            spatial_bounds={
                k: body.get(k, 0.0 if k.endswith("min") else 1.0)
                for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
            },
            time_window_ms=(body["time_start_ms"], body["time_end_ms"]) if has_window else None,
            limit=body.get("limit", 10),
            min_confidence=body.get("min_confidence"),
        )
        out = self._results([h.state for h in hits], body)
        if body.get("include_scores"):
            for r, h in zip(out["results"], hits, strict=True):
                r |= {"score": h.score, "decayed_score": h.decayed_score}
        return out


@pytest.fixture
def fake_worker():
    fake = _FakeWorker()
    fake.thread.start()
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def test_sdk_cloud_mode_matches_golden(fake_worker):
    now = int(time.time() * 1000)
    client = LociClient(base_url=fake_worker.base_url, api_key=KEY, vector_size=4)
    inserted = {s["key"]: _world_state(s, now) for s in GOLDEN["states"]}
    key_of = {client.insert(state): key for key, state in inserted.items()}

    for q in _sdk_queries():
        assert _keys(client.query(**_query_kwargs(q, now)), key_of) == q["expect"], q["name"]
    assert _links(client, key_of, len(GOLDEN["states"])) == GOLDEN["links"]

    # Every WorldState field round-trips; vectors come back float32-rounded.
    for s in client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=100):
        sent = inserted[key_of[s.id]]
        assert (s.x, s.y, s.z, s.timestamp_ms) == (sent.x, sent.y, sent.z, sent.timestamp_ms)
        assert (s.scene_id, s.scale_level, s.confidence) == (
            sent.scene_id,
            sent.scale_level,
            sent.confidence,
        )
        assert s.metadata == sent.metadata
        assert s.vector == pytest.approx(sent.vector, abs=1e-6)

    # insert_batch, query_scored, get_trajectory and get_causal_context, as in the library.
    _assert_m1(_run_m1(client, now, key_of))

    paths = {r["path"] for r in fake_worker.requests}
    assert paths == {
        f"/p/{REF}/{action}"
        for action in ("insert", "insert_batch", "query", "trajectory", "causal_context")
    }
    assert all(r["ua"].startswith("loci-python/") for r in fake_worker.requests)


def test_sdk_splits_large_batches_and_chains_scenes(monkeypatch):
    """Requests go in (scene, time) order and carry each scene's last id to the next request."""
    monkeypatch.setattr(cloud_transport, "BATCH_STATES", 2)
    bodies: list[dict] = []

    def server(method: str, path: str, body: dict | None = None) -> dict:
        assert (method, path) == ("POST", "/insert_batch") and body is not None
        bodies.append(json.loads(json.dumps(body)))
        ids = [f"id{len(bodies)}-{n}" for n in range(len(body["states"]))]
        chains = {st["scene_id"]: i for st, i in zip(body["states"], ids, strict=True)}
        return {"ids": ids, "chains": chains}

    transport = CloudTransport("http://fake/p/ref", KEY)
    monkeypatch.setattr(transport, "_request", server)
    now = int(time.time() * 1000)
    ids = transport.insert_batch([_world_state(s, now) for s in GOLDEN["batch"]["states"]])

    sent = [(st["scene_id"], st["timestamp_ms"]) for b in bodies for st in b["states"]]
    assert [len(b["states"]) for b in bodies] == [2, 2, 1]
    assert sent == sorted(sent)
    assert bodies[0]["chains"] == {}
    assert bodies[1]["chains"] == {"hall": "id1-0", "kitchen": "id1-1"}
    assert bodies[2]["chains"] == {"hall": "id1-0", "kitchen": "id2-0", "roof": "id2-1"}
    # Ids come back in input order: j, k, l, i, m.
    assert ids == ["id2-0", "id1-0", "id3-0", "id1-1", "id2-1"]


def test_sdk_cloud_mode_without_vectors(fake_worker):
    client = LociClient(base_url=fake_worker.base_url, api_key=KEY, vector_size=4)
    client.insert(_world_state(GOLDEN["states"][0], int(time.time() * 1000)))
    transport = client._cloud
    assert transport is not None
    [hit] = transport.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5, include_vectors=False)
    assert hit.vector == []
    assert hit.metadata == GOLDEN["states"][0]["metadata"]


@pytest.mark.asyncio
async def test_async_sdk_cloud_mode_keeps_the_project_path(fake_worker):
    now = int(time.time() * 1000)
    async with AsyncLociClient(base_url=fake_worker.base_url, api_key=KEY, vector_size=4) as client:
        key_of = {await client.insert(_world_state(s, now)): s["key"] for s in GOLDEN["states"]}
        q = next(q for q in GOLDEN["queries"] if q["name"] == "combined")
        assert _keys(await client.query(**_query_kwargs(q, now)), key_of) == q["expect"]
        scored = await client.query_scored(**_query_kwargs(q, now))
        assert [key_of[h.state.id] for h in scored] == q["expect"]
        b = next(k for k, v in key_of.items() if v == "b")
        assert _keys(await client.get_trajectory(b, 1, 0), key_of) == ["e", "b"]
    paths = {r["path"] for r in fake_worker.requests}
    assert paths == {f"/p/{REF}/{a}" for a in ("insert", "query", "trajectory")}
    assert all(r["ua"].startswith("loci-python/") for r in fake_worker.requests)
