"""Parity tests: RustMemoryStore (loci_core.LociStore) vs the Python MemoryStore.

Contract-style suite: every operation runs against BOTH stores side by side
and the results are asserted equal (scores and vectors with approx — the
native store holds float32).  ``tests/test_memory_store.py`` remains the
reference suite for the Python store and is not modified.

The whole module skips cleanly when the loci_core extension (or its
LociStore class) is unavailable.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any
from unittest.mock import patch

import pytest

from loci.backends.memory import MemoryStore

pytest.importorskip(
    "loci.backends.rust_store",
    reason="loci_core native store unavailable; install with `uv sync --group native`",
)

from loci.backends.rust_store import RustMemoryStore  # noqa: E402
from loci.local_client import LocalLociClient  # noqa: E402
from loci.schema import WorldState  # noqa: E402
from loci.temporal.consolidation import ConsolidationPolicy  # noqa: E402
from loci.temporal.retention import RetentionPolicy  # noqa: E402

APPROX: dict[str, float] = {"rel": 1e-5, "abs": 1e-6}


# ---------------------------------------------------------------------------
# Side-by-side harness
# ---------------------------------------------------------------------------


def _assert_hits_equal(
    py_hits: list[dict],
    rs_hits: list[dict],
    *,
    with_score: bool = False,
    ordered: bool = True,
) -> None:
    if not ordered:
        py_hits = sorted(py_hits, key=lambda p: p["id"])
        rs_hits = sorted(rs_hits, key=lambda p: p["id"])
    assert [h["id"] for h in rs_hits] == [h["id"] for h in py_hits]
    for py_hit, rs_hit in zip(py_hits, rs_hits, strict=True):
        assert rs_hit["vector"] == pytest.approx(py_hit["vector"], **APPROX)
        assert rs_hit["payload"] == py_hit["payload"]
        if with_score:
            assert rs_hit["score"] == pytest.approx(py_hit["score"], **APPROX)


class StorePair:
    """Mirrors every operation onto both stores and asserts equal results."""

    def __init__(self) -> None:
        self.py = MemoryStore()
        self.rs = RustMemoryStore()
        self.both = (self.py, self.rs)

    # -- mutations (no interesting return value) --

    def create_collection(self, *args: Any, **kwargs: Any) -> None:
        for s in self.both:
            s.create_collection(*args, **kwargs)

    def delete_collection(self, *args: Any) -> None:
        for s in self.both:
            s.delete_collection(*args)

    def create_payload_index(self, *args: Any) -> None:
        for s in self.both:
            s.create_payload_index(*args)

    def upsert(self, *args: Any, **kwargs: Any) -> None:
        for s in self.both:
            s.upsert(*args, **kwargs)

    def set_payload(self, *args: Any) -> None:
        for s in self.both:
            s.set_payload(*args)

    # -- operations whose results are compared --

    def collection_exists(self, name: str) -> bool:
        py_r, rs_r = (s.collection_exists(name) for s in self.both)
        assert rs_r == py_r
        return rs_r

    def delete_points(self, *args: Any) -> int:
        py_r, rs_r = (s.delete_points(*args) for s in self.both)
        assert rs_r == py_r
        return rs_r

    def delete_points_in_time_range(self, *args: Any, **kwargs: Any) -> int:
        py_r, rs_r = (s.delete_points_in_time_range(*args, **kwargs) for s in self.both)
        assert rs_r == py_r
        return rs_r

    def retrieve(self, *args: Any) -> list[dict]:
        py_r, rs_r = (s.retrieve(*args) for s in self.both)
        _assert_hits_equal(py_r, rs_r)
        return rs_r

    def search(self, *args: Any, ordered: bool = True, **kwargs: Any) -> list[dict]:
        """Pass ``ordered=False`` when equal scores make the hit order a tie
        (tie order is arbitrary in both stores)."""
        py_r, rs_r = (s.search(*args, **kwargs) for s in self.both)
        _assert_hits_equal(py_r, rs_r, with_score=True, ordered=ordered)
        return rs_r

    def scroll(self, *args: Any, ordered: bool = True, **kwargs: Any) -> list[dict]:
        py_r, rs_r = (s.scroll(*args, **kwargs) for s in self.both)
        _assert_hits_equal(py_r, rs_r, ordered=ordered)
        return rs_r

    def payload_value_range(self, *args: Any) -> tuple[Any, Any] | None:
        py_r, rs_r = (s.payload_value_range(*args) for s in self.both)
        assert rs_r == py_r
        return rs_r

    @property
    def total_points(self) -> int:
        py_r, rs_r = (s.total_points for s in self.both)
        assert rs_r == py_r
        return rs_r

    def collection_count(self, name: str) -> int:
        py_r, rs_r = (s.collection_count(name) for s in self.both)
        assert rs_r == py_r
        return rs_r


@pytest.fixture()
def pair() -> StorePair:
    p = StorePair()
    p.create_collection("test", vector_size=4, distance="cosine")
    return p


def _vec(x: float, y: float = 0.0, z: float = 0.0, w: float = 0.0) -> list[float]:
    return [x, y, z, w]


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------


class TestLifecycleParity:
    def test_create_exists_delete(self):
        p = StorePair()
        assert not p.collection_exists("foo")
        p.create_collection("foo", vector_size=4)
        assert p.collection_exists("foo")
        p.create_collection("foo", vector_size=4)  # idempotent
        p.upsert("foo", [{"id": "a", "vector": _vec(1.0), "payload": {}}])
        assert p.collection_count("foo") == 1
        p.delete_collection("foo")
        assert not p.collection_exists("foo")
        assert p.total_points == 0

    def test_create_payload_index_is_accepted(self, pair):
        pair.create_payload_index("test", "my_field")
        pair.create_payload_index("nonexistent", "my_field")  # silent no-op

    def test_empty_counts(self, pair):
        assert pair.total_points == 0
        assert pair.collection_count("test") == 0
        assert pair.collection_count("nonexistent") == 0


# ---------------------------------------------------------------------------
# Upsert / retrieve
# ---------------------------------------------------------------------------


class TestUpsertRetrieveParity:
    def test_roundtrip_with_nested_payload(self, pair):
        payload = {
            "x": 0.25,
            "timestamp_ms": 1234,
            "scene_id": "kitchen",
            "confidence": 0.75,
            "prev_state_id": None,
            "metadata": {"labels": ["door", "wall"], "flag": True, "depth": {"m": 1.5}},
        }
        pair.upsert("test", [{"id": "a", "vector": _vec(1.0, 0.5), "payload": payload}])
        results = pair.retrieve("test", ["a"])
        assert results[0]["payload"] == payload

    def test_overwrite_same_id(self, pair):
        pair.upsert("test", [{"id": "a", "vector": _vec(1.0), "payload": {"x": 1}}])
        pair.upsert("test", [{"id": "a", "vector": _vec(2.0), "payload": {"x": 2}}])
        assert pair.collection_count("test") == 1
        results = pair.retrieve("test", ["a"])
        assert results[0]["payload"] == {"x": 2}

    def test_missing_id_and_collection(self, pair):
        assert pair.retrieve("test", ["nope"]) == []
        assert pair.retrieve("nonexistent", ["a"]) == []

    def test_retrieve_preserves_request_order(self, pair):
        pair.upsert(
            "test",
            [{"id": f"p{i}", "vector": _vec(float(i)), "payload": {"i": i}} for i in range(4)],
        )
        results = pair.retrieve("test", ["p2", "missing", "p0", "p3"])
        assert [r["id"] for r in results] == ["p2", "p0", "p3"]


class TestErrorParity:
    def test_upsert_wrong_dimension_raises_value_error(self, pair):
        for s in pair.both:
            with pytest.raises(ValueError, match="dimension"):
                s.upsert("test", [{"id": "a", "vector": [1.0, 2.0], "payload": {}}])
        assert pair.collection_count("test") == 0

    def test_bad_vector_does_not_poison_search(self, pair):
        pair.upsert("test", [{"id": "a", "vector": _vec(1.0), "payload": {}}])
        for s in pair.both:
            with pytest.raises(ValueError):
                s.upsert("test", [{"id": "b", "vector": [1.0], "payload": {}}])
        results = pair.search("test", _vec(1.0), limit=5)
        assert [r["id"] for r in results] == ["a"]

    def test_upsert_missing_collection_raises_key_error(self, pair):
        for s in pair.both:
            with pytest.raises(KeyError):
                s.upsert("nonexistent", [{"id": "a", "vector": _vec(1.0), "payload": {}}])

    def test_set_payload_missing_collection_raises_key_error(self, pair):
        for s in pair.both:
            with pytest.raises(KeyError):
                s.set_payload("nonexistent", "a", {"y": 2})


# ---------------------------------------------------------------------------
# Copy-on-write / copy-on-read (the FFI boundary must isolate references)
# ---------------------------------------------------------------------------


class TestRustReferenceIsolation:
    @pytest.fixture()
    def store(self) -> RustMemoryStore:
        s = RustMemoryStore()
        s.create_collection("test", vector_size=4, distance="cosine")
        return s

    def test_caller_mutation_does_not_corrupt_store(self, store):
        vector = _vec(1.0)
        payload = {"metadata": {"label": "door"}}
        store.upsert("test", [{"id": "a", "vector": vector, "payload": payload}])
        vector[0] = 999.0
        payload["metadata"]["label"] = "corrupted"

        stored = store.retrieve("test", ["a"])[0]
        assert stored["vector"] == [1.0, 0.0, 0.0, 0.0]
        assert stored["payload"]["metadata"] == {"label": "door"}

    def test_mutating_results_does_not_corrupt_store(self, store):
        store.upsert(
            "test",
            [{"id": "a", "vector": _vec(1.0), "payload": {"metadata": {"k": "v"}, "x": 1}}],
        )
        for result in (
            store.retrieve("test", ["a"])[0],
            store.search("test", _vec(1.0), limit=1)[0],
            store.scroll("test")[0],
        ):
            result["payload"]["x"] = 999
            result["payload"]["metadata"]["k"] = "corrupted"
            result["vector"][0] = 123.0

        fresh = store.retrieve("test", ["a"])[0]
        assert fresh["payload"]["x"] == 1
        assert fresh["payload"]["metadata"] == {"k": "v"}
        assert fresh["vector"] == [1.0, 0.0, 0.0, 0.0]

    def test_set_payload_copies_input(self, store):
        store.upsert("test", [{"id": "a", "vector": _vec(1.0), "payload": {}}])
        update = {"metadata": {"k": "v"}}
        store.set_payload("test", "a", update)
        update["metadata"]["k"] = "corrupted"
        assert store.retrieve("test", ["a"])[0]["payload"]["metadata"] == {"k": "v"}


# ---------------------------------------------------------------------------
# set_payload
# ---------------------------------------------------------------------------


class TestSetPayloadParity:
    def test_merges_top_level_keys(self, pair):
        pair.upsert("test", [{"id": "a", "vector": _vec(1.0), "payload": {"x": 1, "m": {"a": 1}}}])
        pair.set_payload("test", "a", {"y": 2, "m": {"b": 2}})
        results = pair.retrieve("test", ["a"])
        # dict.update semantics: top-level replacement, not deep merge.
        assert results[0]["payload"] == {"x": 1, "y": 2, "m": {"b": 2}}

    def test_missing_point_is_noop(self, pair):
        pair.set_payload("test", "nope", {"y": 2})
        assert pair.total_points == 0


# ---------------------------------------------------------------------------
# Search: metrics, score conventions, ordering
# ---------------------------------------------------------------------------


class TestSearchParity:
    def test_cosine_ordering(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {}},
                {"id": "c", "vector": [0.9, 0.1, 0, 0], "payload": {}},
            ],
        )
        results = pair.search("test", [1, 0, 0, 0], limit=2)
        assert [r["id"] for r in results] == ["a", "c"]

    def test_dot_product_scores(self):
        p = StorePair()
        p.create_collection("dots", vector_size=2, distance="dot")
        p.upsert(
            "dots",
            [
                {"id": "a", "vector": [3, 0], "payload": {}},
                {"id": "b", "vector": [1, 1], "payload": {}},
            ],
        )
        results = p.search("dots", [1, 0], limit=2)
        assert results[0]["id"] == "a"
        assert results[0]["score"] == 3.0

    def test_euclidean_negated_distance(self):
        p = StorePair()
        p.create_collection("euc", vector_size=2, distance="euclidean")
        p.upsert(
            "euc",
            [
                {"id": "a", "vector": [1, 0], "payload": {}},
                {"id": "b", "vector": [10, 10], "payload": {}},
            ],
        )
        results = p.search("euc", [1, 0], limit=2)
        assert results[0]["id"] == "a"
        assert results[0]["score"] == 0.0
        assert results[1]["score"] < 0.0  # higher-is-better: farther is more negative

    def test_zero_query_vector_cosine(self, pair):
        pair.upsert("test", [{"id": "a", "vector": [1, 0, 0, 0], "payload": {}}])
        results = pair.search("test", [0, 0, 0, 0])
        assert results[0]["score"] == 0.0

    def test_zero_stored_vector_norm_substitution(self, pair):
        pair.upsert("test", [{"id": "z", "vector": [0, 0, 0, 0], "payload": {}}])
        results = pair.search("test", [1, 0, 0, 0])
        assert results[0]["score"] == 0.0

    def test_empty_and_missing_collection(self, pair):
        assert pair.search("test", [1, 0, 0, 0]) == []
        assert pair.search("nonexistent", [1, 0, 0, 0]) == []

    @pytest.mark.parametrize("distance", ["cosine", "dot", "euclidean"])
    def test_randomized_scores_match_per_id(self, distance):
        """Full-collection score parity on seeded pseudo-random vectors."""
        import random

        rng = random.Random(42)
        dim = 8
        p = StorePair()
        p.create_collection("rand", vector_size=dim, distance=distance)
        p.upsert(
            "rand",
            [
                {
                    "id": f"p{i}",
                    "vector": [round(rng.uniform(-1, 1), 3) for _ in range(dim)],
                    "payload": {"i": i},
                }
                for i in range(50)
            ],
        )
        for _ in range(5):
            query = [round(rng.uniform(-1, 1), 3) for _ in range(dim)]
            py_hits = p.py.search("rand", query, limit=50)
            rs_hits = p.rs.search("rand", query, limit=50)
            py_scores = {h["id"]: h["score"] for h in py_hits}
            rs_scores = {h["id"]: h["score"] for h in rs_hits}
            assert rs_scores.keys() == py_scores.keys()
            for pid, score in py_scores.items():
                assert rs_scores[pid] == pytest.approx(score, **APPROX)
            # Descending score order on both sides.
            for hits in (py_hits, rs_hits):
                scores = [h["score"] for h in hits]
                assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Payload filters
# ---------------------------------------------------------------------------


class TestFilterParity:
    def test_exact_match(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"scene": "s1"}},
                {"id": "b", "vector": [0.9, 0.1, 0, 0], "payload": {"scene": "s2"}},
            ],
        )
        results = pair.search("test", [1, 0, 0, 0], payload_filter={"scene": "s1"})
        assert [r["id"] for r in results] == ["a"]

    def test_any_membership(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"hid": 5}},
                {"id": "b", "vector": [0.9, 0.1, 0, 0], "payload": {"hid": 10}},
                {"id": "c", "vector": [0.8, 0.2, 0, 0], "payload": {"hid": 15}},
            ],
        )
        results = pair.search("test", [1, 0, 0, 0], payload_filter={"hid": {"any": [5, 15]}})
        assert {r["id"] for r in results} == {"a", "c"}

    def test_any_membership_numeric_equivalence(self, pair):
        # Python semantics: 5.0 in [5] is True.
        pair.upsert("test", [{"id": "a", "vector": [1, 0, 0, 0], "payload": {"hid": 5.0}}])
        results = pair.search("test", [1, 0, 0, 0], payload_filter={"hid": {"any": [5]}})
        assert [r["id"] for r in results] == ["a"]

    def test_range_gte_lte(self, pair):
        pair.upsert(
            "test",
            [
                {"id": f"p{i}", "vector": [1, 0, 0, 0], "payload": {"ts": i * 100}}
                for i in range(10)
            ],
        )
        results = pair.search(
            "test", [1, 0, 0, 0], payload_filter={"ts": {"gte": 300, "lte": 500}}, ordered=False
        )
        assert {r["id"] for r in results} == {"p3", "p4", "p5"}

    def test_range_gt_lt(self, pair):
        pair.upsert(
            "test",
            [{"id": f"p{i}", "vector": [1, 0, 0, 0], "payload": {"val": i}} for i in range(5)],
        )
        results = pair.search(
            "test", [1, 0, 0, 0], payload_filter={"val": {"gt": 1, "lt": 4}}, ordered=False
        )
        assert {r["id"] for r in results} == {"p2", "p3"}

    def test_combined_filters(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"hid": 5, "ts": 100}},
                {"id": "b", "vector": [0.9, 0.1, 0, 0], "payload": {"hid": 5, "ts": 200}},
                {"id": "c", "vector": [0.8, 0.2, 0, 0], "payload": {"hid": 10, "ts": 100}},
            ],
        )
        results = pair.search(
            "test", [1, 0, 0, 0], payload_filter={"hid": {"any": [5]}, "ts": {"gte": 150}}
        )
        assert [r["id"] for r in results] == ["b"]

    def test_missing_field_excluded_from_ranges(self, pair):
        pair.upsert("test", [{"id": "a", "vector": [1, 0, 0, 0], "payload": {}}])
        assert pair.search("test", [1, 0, 0, 0], payload_filter={"ts": {"gte": 0}}) == []

    def test_empty_condition_dict_matches_everything(self, pair):
        pair.upsert("test", [{"id": "a", "vector": [1, 0, 0, 0], "payload": {}}])
        results = pair.search("test", [1, 0, 0, 0], payload_filter={"ts": {}})
        assert [r["id"] for r in results] == ["a"]


# ---------------------------------------------------------------------------
# Scroll
# ---------------------------------------------------------------------------


class TestScrollParity:
    def test_unordered_full_scan(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"ts": 100}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {"ts": 200}},
            ],
        )
        results = pair.scroll("test", limit=10, ordered=False)
        assert len(results) == 2

    def test_filtered_scroll(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"scene": "s1"}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {"scene": "s2"}},
            ],
        )
        results = pair.scroll("test", payload_filter={"scene": "s1"}, ordered=False)
        assert [r["id"] for r in results] == ["a"]

    def test_order_by_ascending_with_limit(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"ts": 100}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {"ts": 300}},
                {"id": "c", "vector": [0, 0, 1, 0], "payload": {"ts": 200}},
            ],
        )
        results = pair.scroll("test", order_by="ts", limit=2)
        assert [r["id"] for r in results] == ["a", "c"]

    def test_order_by_missing_field_sorts_as_zero(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"ts": 100}},
                {"id": "d", "vector": [0, 1, 0, 0], "payload": {}},
            ],
        )
        results = pair.scroll("test", order_by="ts", limit=10)
        assert [r["id"] for r in results] == ["d", "a"]

    def test_unordered_early_exit_lengths_match(self, pair):
        pair.upsert(
            "test",
            [{"id": f"p{i}", "vector": [1, 0, 0, 0], "payload": {"i": i}} for i in range(10)],
        )
        py_r = pair.py.scroll("test", payload_filter={"i": {"gte": 4}}, limit=3)
        rs_r = pair.rs.scroll("test", payload_filter={"i": {"gte": 4}}, limit=3)
        assert len(py_r) == len(rs_r) == 3

    def test_missing_collection(self, pair):
        assert pair.scroll("nonexistent") == []


# ---------------------------------------------------------------------------
# Deletes (tombstones on the Rust side)
# ---------------------------------------------------------------------------


class TestDeleteParity:
    def test_delete_points_by_id(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(1.0), "payload": {}},
                {"id": "b", "vector": _vec(2.0), "payload": {}},
            ],
        )
        assert pair.delete_points("test", ["a", "missing"]) == 1
        assert pair.collection_count("test") == 1
        assert pair.retrieve("test", ["a"]) == []
        assert [r["id"] for r in pair.retrieve("test", ["b"])] == ["b"]

    def test_delete_points_missing_collection(self, pair):
        assert pair.delete_points("nonexistent", ["a"]) == 0
        assert pair.delete_points_in_time_range("nonexistent", 0, 1000) == 0

    def test_time_range_delete_end_exclusive(self, pair):
        pair.upsert(
            "test",
            [
                {"id": f"p{i}", "vector": _vec(1.0), "payload": {"timestamp_ms": i * 100}}
                for i in range(5)
            ],
        )
        assert pair.delete_points_in_time_range("test", 100, 300) == 2
        remaining = {
            r["payload"]["timestamp_ms"] for r in pair.scroll("test", limit=10, ordered=False)
        }
        assert remaining == {0, 300, 400}

    def test_time_range_delete_skips_missing_field(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(1.0), "payload": {"timestamp_ms": 100}},
                {"id": "b", "vector": _vec(2.0), "payload": {}},
            ],
        )
        assert pair.delete_points_in_time_range("test", 0, 1000) == 1
        assert [r["id"] for r in pair.scroll("test", limit=10)] == ["b"]

    def test_delete_then_reinsert_and_search(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"k": "a"}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {"k": "b"}},
            ],
        )
        pair.delete_points("test", ["a"])
        results = pair.search("test", [1, 0, 0, 0], limit=10)
        assert [r["id"] for r in results] == ["b"]

        # Reuse after tombstoning (fresh id lands in the freed row on rust).
        pair.upsert("test", [{"id": "c", "vector": [1, 0, 0, 0], "payload": {"k": "c"}}])
        results = pair.search("test", [1, 0, 0, 0], limit=10)
        assert results[0]["id"] == "c"
        assert pair.collection_count("test") == 2
        # Re-deleting an already-deleted id is a no-op on both.
        assert pair.delete_points("test", ["a"]) == 0


# ---------------------------------------------------------------------------
# Stats: payload_value_range / total_points
# ---------------------------------------------------------------------------


class TestStatsParity:
    def test_min_max_over_field(self, pair):
        for i, ts in enumerate((300, 100, 200)):
            pair.upsert(
                "test",
                [{"id": f"p{i}", "vector": _vec(1.0), "payload": {"timestamp_ms": ts}}],
            )
        assert pair.payload_value_range("test", "timestamp_ms") == (100, 300)

    def test_float_values(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(1.0), "payload": {"conf": 0.25}},
                {"id": "b", "vector": _vec(1.0), "payload": {"conf": 0.75}},
            ],
        )
        assert pair.payload_value_range("test", "conf") == (0.25, 0.75)

    def test_empty_and_missing(self, pair):
        assert pair.payload_value_range("test", "timestamp_ms") is None
        assert pair.payload_value_range("nonexistent", "timestamp_ms") is None

    def test_points_missing_field_are_skipped(self, pair):
        pair.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(1.0), "payload": {"timestamp_ms": 42}},
                {"id": "b", "vector": _vec(2.0), "payload": {}},
            ],
        )
        assert pair.payload_value_range("test", "timestamp_ms") == (42, 42)

    def test_total_points_across_collections(self):
        p = StorePair()
        p.create_collection("c1", vector_size=2)
        p.create_collection("c2", vector_size=2)
        p.upsert("c1", [{"id": "a", "vector": [1, 0], "payload": {}}])
        p.upsert("c2", [{"id": "b", "vector": [1, 0], "payload": {}}])
        assert p.total_points == 2
        p.delete_collection("c1")
        assert p.total_points == 1


# ---------------------------------------------------------------------------
# LocalLociClient(backend="rust") end-to-end
# ---------------------------------------------------------------------------

VEC_SIZE = 4
EPOCH_MS = 5000


def _state(
    ts: int,
    scene: str = "a",
    vector: list[float] | None = None,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.5,
) -> WorldState:
    return WorldState(
        x=x,
        y=y,
        z=z,
        timestamp_ms=ts,
        vector=vector if vector is not None else [1.0, 0.0, 0.0, 0.0],
        scene_id=scene,
    )


@contextlib.contextmanager
def _now(ts_ms: int):
    """Pin the client's wall clock (maintenance + decay) to *ts_ms*."""
    with patch("loci.local_client.time.time", return_value=ts_ms / 1000.0):
        yield


class TestRustBackendClient:
    def test_backend_param_validation(self):
        with pytest.raises(ValueError, match="backend"):
            LocalLociClient(vector_size=VEC_SIZE, backend="bogus")

    def test_rust_backend_uses_rust_store(self):
        client = LocalLociClient(vector_size=VEC_SIZE, backend="rust")
        assert isinstance(client.store, RustMemoryStore)
        # The default stays pure Python.
        assert isinstance(LocalLociClient(vector_size=VEC_SIZE).store, MemoryStore)

    def test_query_round_trip_matches_python_backend(self):
        """Same inserts, same query (spatial + time filters) -> same results."""
        clients = {
            name: LocalLociClient(
                vector_size=VEC_SIZE, epoch_size_ms=EPOCH_MS, decay_lambda=0.0, backend=name
            )
            for name in ("python", "rust")
        }
        states = [
            _state(
                ts=1000 + 17 * i,
                scene=f"s{i % 3}",
                vector=[1.0, 0.01 * i, 0.02 * (i % 5), 0.0],
                x=0.1 + 0.025 * i,
                y=0.2,
                z=0.3,
            )
            for i in range(30)
        ]
        for client in clients.values():
            with _now(2000):
                client.insert_batch(states)

        bounds = {
            "x_min": 0.0,
            "x_max": 0.45,
            "y_min": 0.0,
            "y_max": 0.5,
            "z_min": 0.0,
            "z_max": 0.5,
        }
        window = (1000, 1300)
        scored = {}
        for name, client in clients.items():
            with _now(2000):
                scored[name] = client.query_scored(
                    vector=[1.0, 0.1, 0.0, 0.0],
                    spatial_bounds=bounds,
                    time_window_ms=window,
                    limit=10,
                )
        assert scored["python"], "sanity: the query must match something"

        def key(s):
            return (s.state.timestamp_ms, s.state.scene_id, round(s.state.x, 9))

        assert [key(s) for s in scored["rust"]] == [key(s) for s in scored["python"]]
        for py_s, rs_s in zip(scored["python"], scored["rust"], strict=True):
            assert rs_s.score == pytest.approx(py_s.score, **APPROX)
        for s in scored["rust"]:
            assert bounds["x_min"] <= s.state.x <= bounds["x_max"]
            assert window[0] <= s.state.timestamp_ms <= window[1]

    def test_euclid_ordering(self):
        client = LocalLociClient(
            vector_size=2, distance="euclidean", decay_lambda=0.0, backend="rust"
        )
        with _now(1000):
            client.insert(_state(1000, vector=[0.0, 0.0]))
            client.insert(_state(1001, vector=[3.0, 0.0]))
            client.insert(_state(1002, vector=[10.0, 0.0]))
            scored = client.query_scored(vector=[0.0, 0.0], limit=3)
        assert [s.state.vector[0] for s in scored] == pytest.approx([0.0, 3.0, 10.0])
        scores = [s.score for s in scored]
        assert scores[0] == pytest.approx(0.0)
        assert scores == sorted(scores, reverse=True)  # higher-is-better
        assert scores[2] == pytest.approx(-10.0, rel=1e-5)

    def test_consolidation_and_retention_maintenance(self):
        """The client drives delete_points_in_time_range etc. on the rust store."""
        policy = ConsolidationPolicy(
            raw_window_epochs=2, summary_epoch_ratio=4, max_states_per_scene=3
        )
        clients = {
            name: LocalLociClient(
                vector_size=VEC_SIZE,
                epoch_size_ms=EPOCH_MS,
                decay_lambda=0.0,
                consolidation_policy=policy,
                retention_policy=RetentionPolicy(max_epochs=20),
                backend=name,
            )
            for name in ("python", "rust")
        }
        for client in clients.values():
            for e in range(10):
                for scene in ("a", "b"):
                    for i in range(3):
                        ts = e * EPOCH_MS + i * 100
                        with _now(ts):
                            client.insert(_state(ts, scene=scene, vector=[1.0, 0.1 * i, 0.0, 0.0]))

        for name, client in clients.items():
            store = client.store
            # Raw window: only epochs 8 and 9 stay raw (6 points each).
            assert store.collection_count("loci_data") == 12, name
            assert (
                store.scroll("loci_data", {"timestamp_ms": {"lt": 8 * EPOCH_MS}}, limit=100) == []
            ), name
            # Summaries exist, bounded per scene per coarse group.
            assert store.collection_exists("loci_summary"), name
            summaries = store.scroll("loci_summary", limit=1000)
            assert all(s["payload"]["metadata"]["consolidated"] for s in summaries), name
            assert 0 < len(summaries) <= 2 * 2 * 3, name  # 2 groups x 2 scenes x k

        # Both backends converge to the same maintenance state.
        assert clients["rust"].store.collection_count("loci_summary") == clients[
            "python"
        ].store.collection_count("loci_summary")
        rng_rs = clients["rust"].store.payload_value_range("loci_data", "timestamp_ms")
        rng_py = clients["python"].store.payload_value_range("loci_data", "timestamp_ms")
        assert rng_rs == rng_py

        # Old data remains findable through summaries.
        with _now(9 * EPOCH_MS + 400):
            results = clients["rust"].query(
                vector=[1.0, 0.1, 0.0, 0.0], time_window_ms=(0, 4 * EPOCH_MS - 1), limit=10
            )
        assert results
        assert all(s.metadata["consolidated"] for s in results)

    def test_causal_trajectory(self):
        client = LocalLociClient(vector_size=VEC_SIZE, decay_lambda=0.0, backend="rust")
        ids = []
        with _now(1000):
            for i in range(6):
                ids.append(client.insert(_state(1000 + i * 10, scene="walk")))

        # Causal links were written through set_payload on the rust store.
        anchor = client.store.retrieve("loci_data", [ids[2]])[0]
        assert anchor["payload"]["prev_state_id"] == ids[1]
        assert anchor["payload"]["next_state_id"] == ids[3]

        trajectory = client.get_trajectory(ids[2], steps_back=2, steps_forward=2)
        assert [s.id for s in trajectory] == ids[0:5]
        timestamps = [s.timestamp_ms for s in trajectory]
        assert timestamps == sorted(timestamps)

        context = client.get_causal_context(ids[2], window_ms=15)
        assert [s.id for s in context] == ids[1:4]

    def test_predict_and_retrieve_novelty_sanity(self):
        client = LocalLociClient(vector_size=VEC_SIZE, decay_lambda=0.0, backend="rust")
        with _now(1000):
            for i in range(5):
                client.insert(_state(1000 + i, scene="s", vector=[1.0, 0.05 * i, 0.0, 0.0]))

            # Predictor lands on a stored vector: an analog exists, low novelty.
            result = client.predict_and_retrieve(
                context_vector=[1.0, 0.0, 0.0, 0.0],
                predictor_fn=lambda v: [1.0, 0.0, 0.0, 0.0],
                current_position=(0.5, 0.5, 0.5),
                current_timestamp_ms=1010,
                limit=3,
            )
        assert result.results
        assert 0.0 <= result.prediction_novelty <= 1.0
        assert result.prediction_novelty == pytest.approx(0.0, abs=1e-4)

        # Orthogonal prediction: no analog, high novelty.
        with _now(1000):
            result = client.predict_and_retrieve(
                context_vector=[1.0, 0.0, 0.0, 0.0],
                predictor_fn=lambda v: [0.0, 0.0, 0.0, 1.0],
                current_position=(0.5, 0.5, 0.5),
                current_timestamp_ms=1010,
                limit=3,
            )
        assert result.prediction_novelty > 0.9


# ---------------------------------------------------------------------------
# Micro-benchmark (opt-in, matching test_hilbert.py's LOCI_PERF_TESTS gate)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("LOCI_PERF_TESTS") != "1",
    reason="Wall-clock perf test; opt-in via LOCI_PERF_TESTS=1",
)
def test_rust_insert_throughput_beats_python() -> None:
    """The native store must out-insert the Python store (RFC-0001 R5 target: 5x).

    Wall-clock assertions are intrinsically flaky on shared CI runners, so
    the hard gate is only "faster than Python"; the 5x goal is tracked in
    the reported numbers.
    """
    dim = 64
    n = 5000
    vectors = [[(i * 31 + j) % 97 / 97.0 for j in range(dim)] for i in range(n)]
    payloads = [
        {"timestamp_ms": i, "scene_id": f"s{i % 7}", "x": 0.5, "y": 0.5, "z": 0.5} for i in range(n)
    ]

    def run(store: MemoryStore | RustMemoryStore) -> float:
        store.create_collection("bench", vector_size=dim)
        start = time.perf_counter()
        for i in range(n):
            store.upsert("bench", [{"id": f"p{i}", "vector": vectors[i], "payload": payloads[i]}])
        return time.perf_counter() - start

    py_elapsed = run(MemoryStore())
    rs_elapsed = run(RustMemoryStore())
    assert rs_elapsed < py_elapsed, (
        f"rust insert slower than python: {rs_elapsed:.3f}s vs {py_elapsed:.3f}s"
    )
