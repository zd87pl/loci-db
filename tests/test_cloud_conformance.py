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

from loci import AsyncLociClient, LocalLociClient, LociClient, WorldState
from loci.cloud_transport import _insert_payload, _query_payload

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "cloud/workers/test/conformance-golden.json"
GOLDEN: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
REF = "conformance1"
KEY = f"loci_sk_{REF}_{'0' * 48}"


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


def _links(client: Any, key_of: dict[str, str]) -> dict[str, dict]:
    everything = client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=100)
    assert len(everything) == len(GOLDEN["states"])
    return {
        key_of[s.id]: {
            "prev": key_of.get(s.prev_state_id, s.prev_state_id),
            "next": key_of.get(s.next_state_id, s.next_state_id),
        }
        for s in sorted(everything, key=lambda s: key_of[s.id])
    }


def _run_library(now: int) -> tuple[dict[str, list[str]], dict[str, dict]]:
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
    return results, _links(client, key_of)


def _dump_golden(golden: dict) -> str:
    """One state or query per line, so diffs of expectations stay readable."""
    lines = ["{"]
    items = list(golden.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if isinstance(v, list) and k in ("states", "queries", "about"):
            body = ",\n".join(f"    {json.dumps(x)}" for x in v)
            lines.append(f'  "{k}": [\n{body}\n  ]{comma}')
        elif isinstance(v, dict):
            body = ",\n".join(f"    {json.dumps(kk)}: {json.dumps(vv)}" for kk, vv in v.items())
            lines.append(f'  "{k}": {{\n{body}\n  }}{comma}')
        else:
            lines.append(f'  "{k}": {json.dumps(v)}{comma}')
    return "\n".join(lines) + "\n}\n"


def test_local_library_matches_golden():
    now = int(time.time() * 1000)
    results, links = _run_library(now)
    if os.environ.get("LOCI_REGEN_GOLDEN"):
        for q in GOLDEN["queries"]:
            q["expect"] = results[q["name"]]
        GOLDEN["links"] = links
        GOLDEN_PATH.write_text(_dump_golden(GOLDEN))
    assert results == {q["name"]: q["expect"] for q in GOLDEN["queries"]}
    assert links == GOLDEN["links"]


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
                m = re.fullmatch(rf"/p/{REF}/(insert|query)", self.path)
                if m is None:
                    return self._send(404, {"detail": "not found", "code": "not_found"})
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self._send(401, {"detail": "invalid key", "code": "unauthenticated"})
                return self._send(200, fake.insert(body) if m[1] == "insert" else fake.query(body))

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

    def query(self, body: dict) -> dict:
        assert set(body) <= set(GOLDEN["wire"]["query_keys"])
        has_window = body.get("time_start_ms") is not None and body.get("time_end_ms") is not None
        hits = self.store.query(
            vector=body["vector"],
            spatial_bounds={
                k: body.get(k, 0.0 if k.endswith("min") else 1.0)
                for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
            },
            time_window_ms=(body["time_start_ms"], body["time_end_ms"]) if has_window else None,
            limit=body.get("limit", 10),
            min_confidence=body.get("min_confidence"),
        )
        results = []
        for s in hits:
            r = {k: getattr(s, k) for k in GOLDEN["wire"]["result_keys"]}
            r["vector"] = _f32(s.vector) if body.get("include_vectors", True) else []
            results.append(r)
        return {"results": results}


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
    assert _links(client, key_of) == GOLDEN["links"]

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

    paths = {r["path"] for r in fake_worker.requests}
    assert paths == {f"/p/{REF}/insert", f"/p/{REF}/query"}
    assert all(r["ua"].startswith("loci-python/") for r in fake_worker.requests)


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
    assert {r["path"] for r in fake_worker.requests} == {f"/p/{REF}/insert", f"/p/{REF}/query"}
    assert all(r["ua"].startswith("loci-python/") for r in fake_worker.requests)
