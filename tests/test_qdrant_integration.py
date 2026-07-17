"""Integration tests for LociClient/AsyncLociClient against the REAL qdrant-client local engine.

Unlike tests/test_client.py (MagicMock-based), these tests run every operation through
``QdrantClient(location=":memory:")`` — qdrant-client's server-free local engine — and pin
fixes for bugs that the mocked tests could not catch: ID format round-trips, discovery of
collections created by other writers, unordered scroll pagination past 256 points,
min_confidence pushdown, euclidean score ordering, retention purges, and tenant isolation
via collection_prefix.

Notes on the local engine:

* Each ``:memory:`` instance is a private store, so LociClient instances that must see each
  other's data share ONE underlying Qdrant client (injected by patching the class the same
  way the mock-based tests do).
* KNOWN INCOMPATIBILITY (worked around, not fixed here): ``_ensure_collection`` probes with
  ``get_collection`` and handles only ``UnexpectedResponse(404)`` — the real HTTP server's
  contract. The local engine raises ``ValueError("Collection X not found")`` instead, so a
  stock LociClient crashes on first insert in local mode. ``_ServerLikeQdrantClient`` below
  translates ONLY that probe to the server's 404 contract; every other call goes to the
  unmodified local engine.
* Under cosine distance the local engine stores vectors unit-normalised (as the real server
  does), so vector round-trip assertions use unit vectors and approximate comparison
  (float32 storage).
"""

from __future__ import annotations

import contextlib
import math
import uuid
from unittest import mock

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from loci.async_client import AsyncLociClient
from loci.client import LociClient
from loci.schema import WorldState
from loci.temporal.consolidation import ConsolidationPolicy
from loci.temporal.retention import RetentionPolicy

VECTOR_SIZE = 8
EPOCH_MS = 5000

# The local engine warns that payload indexes are no-ops; irrelevant to these tests.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")


def _not_found() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=404, reason_phrase="Not Found", content=b"", headers=httpx.Headers()
    )


class _ServerLikeQdrantClient(QdrantClient):
    """Local-mode client that mimics the HTTP server's 404 contract on get_collection."""

    def get_collection(self, collection_name: str, **kwargs):
        try:
            return super().get_collection(collection_name, **kwargs)
        except ValueError as exc:
            if "not found" in str(exc):
                raise _not_found() from exc
            raise


class _ServerLikeAsyncQdrantClient(AsyncQdrantClient):
    """Async twin of :class:`_ServerLikeQdrantClient`."""

    async def get_collection(self, collection_name: str, **kwargs):
        try:
            return await super().get_collection(collection_name, **kwargs)
        except ValueError as exc:
            if "not found" in str(exc):
                raise _not_found() from exc
            raise


def _make_client(store: QdrantClient, **overrides) -> LociClient:
    """Build a LociClient whose ``_qdrant`` is the shared real local engine."""
    kwargs: dict = {
        "qdrant_url": "http://unused:6333",
        "epoch_size_ms": EPOCH_MS,
        "vector_size": VECTOR_SIZE,
        "decay_lambda": 0.0,  # deterministic ordering, independent of wall clock
    }
    kwargs.update(overrides)
    with mock.patch("loci.client.QdrantClient", return_value=store):
        return LociClient(**kwargs)


def _make_async_client(store: AsyncQdrantClient, **overrides) -> AsyncLociClient:
    kwargs: dict = {
        "qdrant_url": "http://unused:6333",
        "epoch_size_ms": EPOCH_MS,
        "vector_size": VECTOR_SIZE,
        "decay_lambda": 0.0,
    }
    kwargs.update(overrides)
    with mock.patch("loci.async_client.AsyncQdrantClient", return_value=store):
        return AsyncLociClient(**kwargs)


def _planar(theta: float) -> list[float]:
    """Unit vector at angle *theta* in the first two dimensions (cosine-safe)."""
    return [math.cos(theta), math.sin(theta)] + [0.0] * (VECTOR_SIZE - 2)


def _axis(value: float) -> list[float]:
    """Vector at euclidean distance ``value`` from the origin along dim 0."""
    return [value] + [0.0] * (VECTOR_SIZE - 1)


def _state(**overrides) -> WorldState:
    defaults: dict = {
        "x": 0.5,
        "y": 0.5,
        "z": 0.5,
        "timestamp_ms": 10_000,
        "vector": _planar(0.0),
        "scene_id": "scene",
    }
    defaults.update(overrides)
    return WorldState(**defaults)


@pytest.fixture()
def store() -> QdrantClient:
    return _ServerLikeQdrantClient(location=":memory:")


# ---------------------------------------------------------------------------
# 1. Insert -> query round trip with spatial bounds + time window
# ---------------------------------------------------------------------------


class TestRoundTrip:
    BOX = {"x_min": 0.2, "x_max": 0.6, "y_min": 0.2, "y_max": 0.6, "z_min": 0.2, "z_max": 0.6}
    WINDOW = (8_000, 14_000)

    def test_spatial_and_time_filtered_round_trip(self, store):
        client = _make_client(store)
        inside_a = _state(
            x=0.3,
            y=0.4,
            z=0.5,
            timestamp_ms=9_000,
            vector=_planar(0.0),
            scale_level="frame",
            confidence=0.7,
            metadata={"name": "inside_a", "n": 1},
        )
        inside_b = _state(
            x=0.5,
            y=0.3,
            z=0.4,
            timestamp_ms=13_000,
            vector=_planar(0.3),
            confidence=0.9,
            metadata={"name": "inside_b"},
        )
        outside_box = _state(
            x=0.95, y=0.95, z=0.95, timestamp_ms=9_500, metadata={"name": "outside_box"}
        )
        outside_window = _state(
            x=0.4, y=0.4, z=0.4, timestamp_ms=30_000, metadata={"name": "outside_window"}
        )
        for st in (inside_a, inside_b, outside_box, outside_window):
            client.insert(st)

        results = client.query(
            _planar(0.0), spatial_bounds=self.BOX, time_window_ms=self.WINDOW, limit=10
        )

        assert {r.metadata["name"] for r in results} == {"inside_a", "inside_b"}
        # Nearest (exact vector match) ranks first
        got = results[0]
        assert got.metadata == {"name": "inside_a", "n": 1}
        assert (got.x, got.y, got.z) == (inside_a.x, inside_a.y, inside_a.z)
        assert got.timestamp_ms == inside_a.timestamp_ms
        assert got.scale_level == "frame"
        assert got.confidence == pytest.approx(0.7, abs=1e-6)
        assert got.vector == pytest.approx(inside_a.vector, abs=1e-6)

    def test_unfiltered_query_sees_everything(self, store):
        client = _make_client(store)
        for i in range(4):
            client.insert(_state(timestamp_ms=10_000 + i, metadata={"i": i}))
        results = client.query(_planar(0.0), limit=10)
        assert {r.metadata["i"] for r in results} == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# 2. ID round trip + get_trajectory anchor matching
# ---------------------------------------------------------------------------


class TestIdRoundTrip:
    def test_ids_are_canonical_hyphenated_uuids(self, store):
        client = _make_client(store)
        state_id = client.insert(_state())
        # Real Qdrant canonicalises UUID ids into hyphenated form; the client
        # must generate that form up front (uuid4().hex used to break matching).
        assert "-" in state_id
        assert str(uuid.UUID(state_id)) == state_id
        # What Qdrant stores is byte-for-byte what the client returned.
        points, _ = store.scroll("loci_2", limit=10)
        assert [str(p.id) for p in points] == [state_id]
        # And query results carry the same id back.
        results = client.query(_planar(0.0), limit=1)
        assert results[0].id == state_id

    def test_get_trajectory_anchor_found_and_chain_walks(self, store):
        client = _make_client(store)
        ids = [
            client.insert(_state(timestamp_ms=10_000 + i * 100, scene_id="walk", metadata={"i": i}))
            for i in range(5)
        ]

        traj = client.get_trajectory(ids[2], steps_back=2, steps_forward=2)

        assert [s.id for s in traj] == ids  # anchor matched, ordered oldest→newest
        assert [s.metadata["i"] for s in traj] == [0, 1, 2, 3, 4]
        # Causal links round-trip through the real store.
        assert traj[0].prev_state_id is None
        for prev, cur in zip(traj, traj[1:], strict=False):
            assert cur.prev_state_id == prev.id
            assert prev.next_state_id == cur.id


# ---------------------------------------------------------------------------
# 3. Scroll pagination past 256 points
# ---------------------------------------------------------------------------


class TestScrollPagination:
    N = 300  # > the 256-point scroll page size that used to truncate results

    def _fill(self, client) -> list[str]:
        states = [
            _state(timestamp_ms=10_000 + i * 10, scene_id="big", vector=_planar(0.001 * i))
            for i in range(self.N)
        ]
        return client.insert_batch(states)

    def test_get_causal_context_returns_all_points(self, store):
        client = _make_client(store)
        ids = self._fill(client)
        context = client.get_causal_context(ids[150], window_ms=EPOCH_MS)
        assert len(context) == self.N
        assert {s.id for s in context} == set(ids)
        assert [s.timestamp_ms for s in context] == sorted(s.timestamp_ms for s in context)

    def test_get_trajectory_returns_all_points(self, store):
        client = _make_client(store)
        ids = self._fill(client)
        traj = client.get_trajectory(ids[150], steps_back=self.N, steps_forward=self.N)
        assert len(traj) == self.N
        assert {s.id for s in traj} == set(ids)


# ---------------------------------------------------------------------------
# 4. collection_prefix tenant isolation on a shared store
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    def test_prefixed_clients_never_see_each_other(self, store):
        tenant_a = _make_client(store, collection_prefix="tenant_a_")
        tenant_b = _make_client(store, collection_prefix="tenant_b_")

        a_ids = [
            tenant_a.insert(_state(timestamp_ms=10_000 + i, metadata={"tenant": "a"}))
            for i in range(3)
        ]
        b_ids = [
            tenant_b.insert(_state(timestamp_ms=10_000 + i, metadata={"tenant": "b"}))
            for i in range(3)
        ]

        # Both tenants share the same physical store...
        names = {c.name for c in store.get_collections().collections}
        assert {"tenant_a_loci_2", "tenant_b_loci_2"} <= names

        # ...but queries are namespaced.
        a_results = tenant_a.query(_planar(0.0), limit=10)
        b_results = tenant_b.query(_planar(0.0), limit=10)
        assert {s.metadata["tenant"] for s in a_results} == {"a"}
        assert {s.metadata["tenant"] for s in b_results} == {"b"}
        assert {s.id for s in a_results} == set(a_ids)
        assert {s.id for s in b_results} == set(b_ids)

        # Trajectory/context lookups cannot cross the namespace either.
        assert tenant_a.get_trajectory(b_ids[0]) == []
        assert tenant_a.get_causal_context(b_ids[0]) == []
        assert tenant_b.get_trajectory(a_ids[0]) == []


# ---------------------------------------------------------------------------
# 5. Retention purge + late insert into a purged epoch
# ---------------------------------------------------------------------------


class TestRetention:
    def test_max_epochs_purges_and_late_insert_does_not_crash(self, store):
        client = _make_client(store, retention_policy=RetentionPolicy(max_epochs=2))
        for ep in range(5):
            client.insert(_state(timestamp_ms=ep * EPOCH_MS + 2_500, scene_id="ret"))

        names = {c.name for c in store.get_collections().collections}
        assert names == {"loci_3", "loci_4"}  # only the two newest epochs survive

        # Late insert into the long-purged epoch 0: must not raise; the epoch is
        # either recreated or immediately re-purged.
        late_id = client.insert(_state(timestamp_ms=100, scene_id="ret"))
        assert late_id
        names = {c.name for c in store.get_collections().collections}
        assert {"loci_3", "loci_4"} <= names <= {"loci_0", "loci_3", "loci_4"}

        # Store stays consistent: the retained states are still queryable and
        # new inserts keep working.
        surviving = {s.timestamp_ms for s in client.query(_planar(0.0), limit=10)}
        assert {17_500, 22_500} <= surviving
        client.insert(_state(timestamp_ms=5 * EPOCH_MS + 2_500, scene_id="ret"))
        results = client.query(_planar(0.0), limit=10)
        assert 27_500 in {s.timestamp_ms for s in results}


# ---------------------------------------------------------------------------
# 6. Insert-before-query discovery of another writer's collections
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_writer_b_sees_writer_a_after_inserting_first(self, store):
        writer_a = _make_client(store)
        a_ids = [
            writer_a.insert(
                _state(timestamp_ms=10_000 + i * 100, scene_id="scene_a", metadata={"w": "a"})
            )
            for i in range(3)
        ]

        # A brand-new client over the same store inserts FIRST, then queries.
        # Discovery must still surface A's collections (they used to stay invisible).
        writer_b = _make_client(store)
        writer_b.insert(_state(timestamp_ms=20_000, scene_id="scene_b", metadata={"w": "b"}))

        everything = writer_b.query(_planar(0.0), limit=10)
        assert {s.metadata["w"] for s in everything} == {"a", "b"}

        windowed = writer_b.query(_planar(0.0), time_window_ms=(9_000, 11_000), limit=10)
        assert {s.id for s in windowed} == set(a_ids)

        # Anchor lookup across writers works too.
        traj = writer_b.get_trajectory(a_ids[1], steps_back=5, steps_forward=5)
        assert {s.id for s in traj} == set(a_ids)


# ---------------------------------------------------------------------------
# 7. min_confidence with more qualifying matches than limit
# ---------------------------------------------------------------------------


class TestMinConfidence:
    LIMIT = 5

    def test_returns_full_limit_above_threshold(self, store):
        client = _make_client(store)
        states = [
            # 3x limit HIGH-confidence states, all farther from the query vector...
            _state(
                timestamp_ms=10_000 + i,
                vector=_planar(0.8 + 0.001 * i),
                scene_id="mc",
                confidence=0.9,
            )
            for i in range(3 * self.LIMIT)
        ] + [
            # ...and `limit` LOW-confidence decoys sitting nearest to it.
            _state(
                timestamp_ms=12_000 + i,
                vector=_planar(0.01 * i),
                scene_id="mc",
                confidence=0.2,
            )
            for i in range(self.LIMIT)
        ]
        client.insert_batch(states)

        # Sanity: without the filter, the nearby low-confidence decoys win —
        # exactly the setup where post-hoc filtering used to return 0 results.
        unfiltered = client.query(_planar(0.0), limit=self.LIMIT)
        assert all(s.confidence == pytest.approx(0.2) for s in unfiltered)

        filtered = client.query(_planar(0.0), limit=self.LIMIT, min_confidence=0.5)
        assert len(filtered) == self.LIMIT
        assert all(s.confidence == pytest.approx(0.9) for s in filtered)

        scored = client.query_scored(_planar(0.0), limit=self.LIMIT, min_confidence=0.5)
        assert len(scored) == self.LIMIT
        assert all(s.state.confidence == pytest.approx(0.9) for s in scored)


# ---------------------------------------------------------------------------
# 8. distance="euclidean" ordering
# ---------------------------------------------------------------------------


class TestEuclideanOrdering:
    def test_nearest_point_ranks_first(self, store):
        client = _make_client(store, distance="euclidean")
        for i, dist in enumerate([0.9, 0.1, 0.5]):
            client.insert(_state(timestamp_ms=10_000 + i, vector=_axis(dist), metadata={"d": dist}))

        scored = client.query_scored(_axis(0.0), limit=3)
        assert [s.state.metadata["d"] for s in scored] == [0.1, 0.5, 0.9]
        # Scores are negated distances: higher-is-better, strictly descending.
        assert [s.score for s in scored] == pytest.approx([-0.1, -0.5, -0.9], abs=1e-6)
        assert scored[0].decayed_score >= scored[1].decayed_score >= scored[2].decayed_score

        plain = client.query(_axis(0.0), limit=3)
        assert [s.metadata["d"] for s in plain] == [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# 9. AsyncLociClient against AsyncQdrantClient(":memory:")
# ---------------------------------------------------------------------------


class TestAsyncClient:
    @pytest.mark.asyncio
    async def test_round_trip_with_bounds_and_window(self):
        store = _ServerLikeAsyncQdrantClient(location=":memory:")
        client = _make_async_client(store)
        inside = _state(
            x=0.4, y=0.4, z=0.4, timestamp_ms=9_000, confidence=0.7, metadata={"name": "inside"}
        )
        outside_box = _state(x=0.95, y=0.95, z=0.95, timestamp_ms=9_500)
        outside_window = _state(x=0.4, y=0.4, z=0.4, timestamp_ms=30_000)
        for st in (inside, outside_box, outside_window):
            await client.insert(st)

        results = await client.query(
            _planar(0.0),
            spatial_bounds=TestRoundTrip.BOX,
            time_window_ms=TestRoundTrip.WINDOW,
            limit=10,
        )
        assert len(results) == 1
        got = results[0]
        assert got.metadata == {"name": "inside"}
        assert (got.x, got.y, got.z) == (0.4, 0.4, 0.4)
        assert got.timestamp_ms == 9_000
        assert got.confidence == pytest.approx(0.7, abs=1e-6)
        assert got.vector == pytest.approx(inside.vector, abs=1e-6)
        assert str(uuid.UUID(got.id)) == got.id  # hyphenated UUID round trip

    @pytest.mark.asyncio
    async def test_prefix_isolation_on_shared_store(self):
        store = _ServerLikeAsyncQdrantClient(location=":memory:")
        tenant_a = _make_async_client(store, collection_prefix="tenant_a_")
        tenant_b = _make_async_client(store, collection_prefix="tenant_b_")

        a_ids = await tenant_a.insert_batch(
            [_state(timestamp_ms=10_000 + i, metadata={"tenant": "a"}) for i in range(2)]
        )
        b_ids = await tenant_b.insert_batch(
            [_state(timestamp_ms=10_000 + i, metadata={"tenant": "b"}) for i in range(2)]
        )

        a_results = await tenant_a.query(_planar(0.0), limit=10)
        b_results = await tenant_b.query(_planar(0.0), limit=10)
        assert {s.id for s in a_results} == set(a_ids)
        assert {s.id for s in b_results} == set(b_ids)
        assert {s.metadata["tenant"] for s in a_results} == {"a"}
        assert {s.metadata["tenant"] for s in b_results} == {"b"}
        assert await tenant_a.get_trajectory(b_ids[0]) == []

    @pytest.mark.asyncio
    async def test_euclidean_nearest_first(self):
        store = _ServerLikeAsyncQdrantClient(location=":memory:")
        client = _make_async_client(store, distance="euclidean")
        for i, dist in enumerate([0.9, 0.1, 0.5]):
            await client.insert(
                _state(timestamp_ms=10_000 + i, vector=_axis(dist), metadata={"d": dist})
            )

        scored = await client.query_scored(_axis(0.0), limit=3)
        assert [s.state.metadata["d"] for s in scored] == [0.1, 0.5, 0.9]
        assert [s.score for s in scored] == pytest.approx([-0.1, -0.5, -0.9], abs=1e-6)

        plain = await client.query(_axis(0.0), limit=3)
        assert [s.metadata["d"] for s in plain] == [0.1, 0.5, 0.9]

    @pytest.mark.asyncio
    async def test_scroll_pagination_past_256(self):
        store = _ServerLikeAsyncQdrantClient(location=":memory:")
        client = _make_async_client(store)
        n = 300
        ids = await client.insert_batch(
            [
                _state(timestamp_ms=10_000 + i * 10, scene_id="big", vector=_planar(0.001 * i))
                for i in range(n)
            ]
        )
        context = await client.get_causal_context(ids[150], window_ms=EPOCH_MS)
        assert len(context) == n
        assert {s.id for s in context} == set(ids)
        traj = await client.get_trajectory(ids[150], steps_back=n, steps_forward=n)
        assert len(traj) == n


# ---------------------------------------------------------------------------
# 10. Memory consolidation end-to-end (real engine, injected timestamps)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _pinned_now(module: str, ts_ms: int):
    """Pin the client module's wall clock (maintenance + decay) to *ts_ms*."""
    with mock.patch(f"{module}.time.time", return_value=ts_ms / 1000.0):
        yield


def _scene_planar(scene: str, i: int) -> list[float]:
    """Unit vectors clustered per scene: scene 'a' near theta=0, 'b' near 1.5."""
    base = {"a": 0.0, "b": 1.5}[scene]
    return _planar(base + 0.02 * i)


class TestConsolidation:
    """Insert 10 epochs x 2 scenes x 5 states, advancing the pinned clock.

    With raw_window_epochs=2 and summary_epoch_ratio=4, epochs 0-7 end up
    consolidated into loci_sum_0 (epochs 0-3) and loci_sum_1 (epochs 4-7),
    leaving epochs 8 and 9 raw.
    """

    N_EPOCHS = 10
    STATES_PER_SCENE_PER_EPOCH = 5
    POLICY = ConsolidationPolicy(raw_window_epochs=2, summary_epoch_ratio=4, max_states_per_scene=3)

    def _fill(self, store: QdrantClient) -> tuple[LociClient, dict[int, list[str]]]:
        client = _make_client(store, consolidation_policy=self.POLICY)
        ids_by_epoch: dict[int, list[str]] = {}
        for e in range(self.N_EPOCHS):
            for scene in ("a", "b"):
                for i in range(self.STATES_PER_SCENE_PER_EPOCH):
                    ts = e * EPOCH_MS + i * 100
                    with _pinned_now("loci.client", ts):
                        sid = client.insert(
                            _state(timestamp_ms=ts, scene_id=scene, vector=_scene_planar(scene, i))
                        )
                    ids_by_epoch.setdefault(e, []).append(sid)
        return client, ids_by_epoch

    def test_raw_beyond_window_dropped_summaries_bounded(self, store):
        client, _ = self._fill(store)

        names = {c.name for c in store.get_collections().collections}
        assert names == {"loci_8", "loci_9", "loci_sum_0", "loci_sum_1"}
        assert client._list_active_epochs() == [8, 9]

        # Bounded: at most max_states_per_scene per scene per coarse
        # collection, every summary flagged as consolidated.
        summary_points = 0
        for name in ("loci_sum_0", "loci_sum_1"):
            points, _ = store.scroll(name, limit=100)
            assert 0 < len(points) <= 2 * self.POLICY.max_states_per_scene
            assert all(p.payload["metadata"]["consolidated"] is True for p in points)
            summary_points += len(points)

        # Total resident points stay bounded: 200 inserted, <= 20 raw + 12 summaries.
        raw_points = sum(len(store.scroll(f"loci_{e}", limit=100)[0]) for e in (8, 9))
        assert raw_points == 2 * 2 * self.STATES_PER_SCENE_PER_EPOCH
        assert raw_points + summary_points <= 20 + 12

    def test_old_data_findable_via_summaries(self, store):
        client, _ = self._fill(store)
        old_window = (0, 4 * EPOCH_MS - 1)  # epochs 0-3, all raw collections dropped
        with _pinned_now("loci.client", 9 * EPOCH_MS + 400):
            results = client.query(_scene_planar("a", 0), time_window_ms=old_window, limit=10)
        assert results
        for s in results:
            assert s.metadata["consolidated"] is True
            assert old_window[0] <= s.timestamp_ms <= old_window[1]
            assert s.scene_id in {"a", "b"}
            assert s.metadata["source_count"] >= 1

    def test_recent_data_returns_raw(self, store):
        client, ids_by_epoch = self._fill(store)
        recent_window = (8 * EPOCH_MS, 10 * EPOCH_MS)
        with _pinned_now("loci.client", 9 * EPOCH_MS + 400):
            results = client.query(_scene_planar("a", 0), time_window_ms=recent_window, limit=20)
        assert results
        raw_ids = set(ids_by_epoch[8]) | set(ids_by_epoch[9])
        for s in results:
            assert not s.metadata.get("consolidated")
            assert s.id in raw_ids

    def test_unwindowed_query_spans_raw_and_summaries(self, store):
        client, _ = self._fill(store)
        with _pinned_now("loci.client", 9 * EPOCH_MS + 400):
            results = client.query(_scene_planar("a", 0), limit=50)
        flags = {bool(s.metadata.get("consolidated")) for s in results}
        assert flags == {True, False}

    def test_trajectory_ignores_summaries(self, store):
        client, ids_by_epoch = self._fill(store)
        anchor_id = ids_by_epoch[9][0]  # scene "a", epoch 9
        trajectory = client.get_trajectory(anchor_id, steps_back=100, steps_forward=100)
        assert trajectory
        for s in trajectory:
            assert not s.metadata.get("consolidated")
            assert s.timestamp_ms >= 8 * EPOCH_MS  # only raw epochs remain


class TestAsyncConsolidation:
    @pytest.mark.asyncio
    async def test_consolidation_end_to_end(self):
        store = _ServerLikeAsyncQdrantClient(location=":memory:")
        policy = ConsolidationPolicy(
            raw_window_epochs=1, summary_epoch_ratio=2, max_states_per_scene=2
        )
        client = _make_async_client(store, consolidation_policy=policy)
        ids_by_epoch: dict[int, list[str]] = {}
        for e in range(4):
            for i in range(3):
                ts = e * EPOCH_MS + i * 100
                with _pinned_now("loci.async_client", ts):
                    sid = await client.insert(
                        _state(timestamp_ms=ts, scene_id="s", vector=_planar(0.05 * (3 * e + i)))
                    )
                ids_by_epoch.setdefault(e, []).append(sid)

        # raw_window_epochs=1 keeps only the current epoch raw; epochs 0-1
        # fold into loci_sum_0 and epoch 2 into loci_sum_1.
        names = {c.name for c in (await store.get_collections()).collections}
        assert names == {"loci_3", "loci_sum_0", "loci_sum_1"}
        assert client._list_active_epochs() == [3]

        for name in ("loci_sum_0", "loci_sum_1"):
            points, _ = await store.scroll(name, limit=100)
            assert 0 < len(points) <= policy.max_states_per_scene
            assert all(p.payload["metadata"]["consolidated"] is True for p in points)

        # Old data findable via summaries within a time window.
        with _pinned_now("loci.async_client", 3 * EPOCH_MS + 200):
            old = await client.query(_planar(0.0), time_window_ms=(0, 2 * EPOCH_MS - 1), limit=10)
        assert old
        assert all(s.metadata["consolidated"] is True for s in old)

        # Recent data stays raw; trajectory ignores summaries.
        with _pinned_now("loci.async_client", 3 * EPOCH_MS + 200):
            recent = await client.query(
                _planar(0.05 * 9), time_window_ms=(3 * EPOCH_MS, 4 * EPOCH_MS - 1), limit=10
            )
        assert recent
        assert all(not s.metadata.get("consolidated") for s in recent)
        assert {s.id for s in recent} == set(ids_by_epoch[3])

        traj = await client.get_trajectory(ids_by_epoch[3][0], steps_back=50, steps_forward=50)
        assert traj
        assert all(not s.metadata.get("consolidated") for s in traj)
        assert all(s.timestamp_ms >= 3 * EPOCH_MS for s in traj)
