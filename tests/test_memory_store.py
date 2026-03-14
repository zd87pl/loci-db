"""Tests for the pure-Python in-memory vector store."""

from __future__ import annotations

import pytest

from engram.backends.memory import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    s = MemoryStore()
    s.create_collection("test", vector_size=4, distance="cosine")
    return s


def _vec(x: float, y: float = 0.0, z: float = 0.0, w: float = 0.0) -> list[float]:
    """Helper to create a 4D vector."""
    return [x, y, z, w]


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------


class TestCollectionLifecycle:
    def test_create_and_exists(self):
        s = MemoryStore()
        assert not s.collection_exists("foo")
        s.create_collection("foo", vector_size=4)
        assert s.collection_exists("foo")

    def test_create_idempotent(self, store):
        store.create_collection("test", vector_size=4)
        assert store.collection_exists("test")

    def test_create_payload_index(self, store):
        store.create_payload_index("test", "my_field")
        # No error, index recorded internally

    def test_total_points_empty(self, store):
        assert store.total_points == 0

    def test_collection_count_empty(self, store):
        assert store.collection_count("test") == 0
        assert store.collection_count("nonexistent") == 0


# ---------------------------------------------------------------------------
# Upsert and retrieve
# ---------------------------------------------------------------------------


class TestUpsertRetrieve:
    def test_upsert_and_retrieve(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(1.0), "payload": {"x": 1}},
            ],
        )
        results = store.retrieve("test", ["a"])
        assert len(results) == 1
        assert results[0]["id"] == "a"
        assert results[0]["payload"]["x"] == 1

    def test_upsert_overwrites(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(1.0), "payload": {"x": 1}},
            ],
        )
        store.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(2.0), "payload": {"x": 2}},
            ],
        )
        results = store.retrieve("test", ["a"])
        assert results[0]["payload"]["x"] == 2

    def test_retrieve_missing_id(self, store):
        assert store.retrieve("test", ["nope"]) == []

    def test_retrieve_missing_collection(self, store):
        assert store.retrieve("nonexistent", ["a"]) == []

    def test_total_points_after_upsert(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(1.0), "payload": {}},
                {"id": "b", "vector": _vec(2.0), "payload": {}},
            ],
        )
        assert store.total_points == 2
        assert store.collection_count("test") == 2


# ---------------------------------------------------------------------------
# set_payload
# ---------------------------------------------------------------------------


class TestSetPayload:
    def test_set_payload_updates(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": _vec(1.0), "payload": {"x": 1}},
            ],
        )
        store.set_payload("test", "a", {"y": 2})
        results = store.retrieve("test", ["a"])
        assert results[0]["payload"]["x"] == 1
        assert results[0]["payload"]["y"] == 2

    def test_set_payload_missing_point(self, store):
        # Should not raise
        store.set_payload("test", "nope", {"y": 2})


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_cosine_search_basic(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {}},
                {"id": "c", "vector": [0.9, 0.1, 0, 0], "payload": {}},
            ],
        )
        results = store.search("test", [1, 0, 0, 0], limit=2)
        assert len(results) == 2
        assert results[0]["id"] == "a"  # exact match
        assert results[1]["id"] == "c"  # closest

    def test_search_with_limit(self, store):
        for i in range(20):
            store.upsert(
                "test",
                [
                    {"id": f"p{i}", "vector": [float(i), 0, 0, 0], "payload": {}},
                ],
            )
        results = store.search("test", [10, 0, 0, 0], limit=5)
        assert len(results) == 5

    def test_search_empty_collection(self, store):
        assert store.search("test", [1, 0, 0, 0]) == []

    def test_search_missing_collection(self, store):
        assert store.search("nonexistent", [1, 0, 0, 0]) == []

    def test_dot_product_search(self):
        s = MemoryStore()
        s.create_collection("dots", vector_size=2, distance="dot")
        s.upsert(
            "dots",
            [
                {"id": "a", "vector": [3, 0], "payload": {}},
                {"id": "b", "vector": [1, 1], "payload": {}},
            ],
        )
        results = s.search("dots", [1, 0], limit=2)
        assert results[0]["id"] == "a"
        assert results[0]["score"] == 3.0

    def test_euclidean_search(self):
        s = MemoryStore()
        s.create_collection("euc", vector_size=2, distance="euclidean")
        s.upsert(
            "euc",
            [
                {"id": "a", "vector": [1, 0], "payload": {}},
                {"id": "b", "vector": [10, 10], "payload": {}},
            ],
        )
        results = s.search("euc", [1, 0], limit=2)
        assert results[0]["id"] == "a"
        assert results[0]["score"] == 0.0  # neg euclidean, 0 distance

    def test_search_zero_vector(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {}},
            ],
        )
        results = store.search("test", [0, 0, 0, 0])
        assert len(results) == 1
        assert results[0]["score"] == 0.0  # cosine with zero = 0


# ---------------------------------------------------------------------------
# Payload filter matching
# ---------------------------------------------------------------------------


class TestPayloadFilters:
    def test_exact_match_filter(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"scene": "s1"}},
                {"id": "b", "vector": [0.9, 0.1, 0, 0], "payload": {"scene": "s2"}},
            ],
        )
        results = store.search("test", [1, 0, 0, 0], payload_filter={"scene": "s1"})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_range_filter_gte_lte(self, store):
        for i in range(10):
            store.upsert(
                "test",
                [
                    {"id": f"p{i}", "vector": [1, 0, 0, 0], "payload": {"ts": i * 100}},
                ],
            )
        results = store.search(
            "test",
            [1, 0, 0, 0],
            payload_filter={"ts": {"gte": 300, "lte": 500}},
        )
        ids = {r["id"] for r in results}
        assert ids == {"p3", "p4", "p5"}

    def test_range_filter_lt_gt(self, store):
        for i in range(5):
            store.upsert(
                "test",
                [
                    {"id": f"p{i}", "vector": [1, 0, 0, 0], "payload": {"val": i}},
                ],
            )
        results = store.search(
            "test",
            [1, 0, 0, 0],
            payload_filter={"val": {"gt": 1, "lt": 4}},
        )
        ids = {r["id"] for r in results}
        assert ids == {"p2", "p3"}

    def test_any_filter(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"hid": 5}},
                {"id": "b", "vector": [0.9, 0.1, 0, 0], "payload": {"hid": 10}},
                {"id": "c", "vector": [0.8, 0.2, 0, 0], "payload": {"hid": 15}},
            ],
        )
        results = store.search(
            "test",
            [1, 0, 0, 0],
            payload_filter={"hid": {"any": [5, 15]}},
        )
        ids = {r["id"] for r in results}
        assert ids == {"a", "c"}

    def test_combined_filters(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"hid": 5, "ts": 100}},
                {"id": "b", "vector": [0.9, 0.1, 0, 0], "payload": {"hid": 5, "ts": 200}},
                {"id": "c", "vector": [0.8, 0.2, 0, 0], "payload": {"hid": 10, "ts": 100}},
            ],
        )
        results = store.search(
            "test",
            [1, 0, 0, 0],
            payload_filter={"hid": {"any": [5]}, "ts": {"gte": 150}},
        )
        assert len(results) == 1
        assert results[0]["id"] == "b"

    def test_filter_none_value_excluded(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {}},
            ],
        )
        results = store.search(
            "test",
            [1, 0, 0, 0],
            payload_filter={"ts": {"gte": 0}},
        )
        assert len(results) == 0  # None < 0 → excluded


# ---------------------------------------------------------------------------
# Scroll
# ---------------------------------------------------------------------------


class TestScroll:
    def test_scroll_basic(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"ts": 100}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {"ts": 200}},
            ],
        )
        results = store.scroll("test")
        assert len(results) == 2

    def test_scroll_with_filter(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"scene": "s1"}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {"scene": "s2"}},
            ],
        )
        results = store.scroll("test", payload_filter={"scene": "s1"})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_scroll_with_order_by(self, store):
        store.upsert(
            "test",
            [
                {"id": "a", "vector": [1, 0, 0, 0], "payload": {"ts": 100}},
                {"id": "b", "vector": [0, 1, 0, 0], "payload": {"ts": 300}},
                {"id": "c", "vector": [0, 0, 1, 0], "payload": {"ts": 200}},
            ],
        )
        results = store.scroll("test", order_by="ts", limit=2)
        assert results[0]["id"] == "b"
        assert results[1]["id"] == "c"

    def test_scroll_missing_collection(self, store):
        assert store.scroll("nonexistent") == []
