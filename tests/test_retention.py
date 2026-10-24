"""Tests for temporal epoch retention management."""

import contextlib
import time

import pytest

from loci.temporal.retention import RetentionManager, RetentionPolicy, epochs_to_drop


class FakeStore:
    """In-memory stand-in for a Qdrant/Memory backend."""

    def __init__(self):
        self.collections: set[str] = set()

    def add(self, name: str):
        self.collections.add(name)

    def delete(self, name: str):
        self.collections.discard(name)

    def list_active(self) -> list[int]:
        epochs = []
        for col in self.collections:
            if col.startswith("loci_"):
                with contextlib.suppress(ValueError):
                    epochs.append(int(col.split("_", 1)[1]))
        return sorted(epochs)


def test_epochs_to_drop_respects_max_epochs():
    store = FakeStore()
    for e in range(100):
        store.add(f"loci_{e}")

    policy = RetentionPolicy(max_epochs=50)
    to_drop = epochs_to_drop(
        store.list_active(),
        now_ms=int(time.time() * 1000),
        epoch_size_ms=5000,
        policy=policy,
    )
    assert len(to_drop) == 50
    assert to_drop == list(range(50))


def test_epochs_to_drop_respects_max_age():
    now = int(time.time() * 1000)
    epoch_size = 5000
    # Create epochs that are ~10s, ~20s, ~30s old
    epochs = [
        (now - 10_000) // epoch_size,
        (now - 20_000) // epoch_size,
        (now - 30_000) // epoch_size,
    ]
    store = FakeStore()
    for ep in epochs:
        store.add(f"loci_{ep}")

    policy = RetentionPolicy(max_age_ms=15_000)
    to_drop = epochs_to_drop(
        store.list_active(),
        now_ms=now,
        epoch_size_ms=epoch_size,
        policy=policy,
    )
    # The ~20s and ~30s old epochs should be dropped
    assert len(to_drop) == 2


def test_retention_manager_drops_oldest():
    store = FakeStore()
    for e in range(10):
        store.add(f"loci_{e}")

    policy = RetentionPolicy(max_epochs=5)
    mgr = RetentionManager(policy, epoch_size_ms=5000)
    dropped = mgr.maybe_purge(
        active_epochs=store.list_active(),
        now_ms=int(time.time() * 1000),
        delete_fn=store.delete,
    )
    assert dropped == ["loci_0", "loci_1", "loci_2", "loci_3", "loci_4"]
    assert store.collections == {f"loci_{e}" for e in range(5, 10)}


def test_retention_manager_no_policy():
    store = FakeStore()
    for e in range(10):
        store.add(f"loci_{e}")

    mgr = RetentionManager(RetentionPolicy(max_epochs=999), epoch_size_ms=5000)
    dropped = mgr.maybe_purge(
        active_epochs=store.list_active(),
        now_ms=int(time.time() * 1000),
        delete_fn=store.delete,
    )
    assert dropped == []
    assert len(store.collections) == 10


def test_retention_manager_custom_callback():
    store = FakeStore()
    archived = []

    def archive_then_delete(ep: int, col: str):
        archived.append(col)

    for e in range(10):
        store.add(f"loci_{e}")

    policy = RetentionPolicy(max_epochs=5, archive_callback=archive_then_delete)
    mgr = RetentionManager(policy, epoch_size_ms=5000)
    dropped = mgr.maybe_purge(
        active_epochs=store.list_active(),
        now_ms=int(time.time() * 1000),
        delete_fn=store.delete,
    )
    assert dropped == ["loci_0", "loci_1", "loci_2", "loci_3", "loci_4"]
    assert archived == dropped


@pytest.mark.asyncio
async def test_retention_manager_async():
    store = FakeStore()
    for e in range(10):
        store.add(f"loci_{e}")

    policy = RetentionPolicy(max_epochs=5)
    mgr = RetentionManager(policy, epoch_size_ms=5000)
    dropped = await mgr.maybe_purge_async(
        active_epochs=store.list_active(),
        now_ms=int(time.time() * 1000),
        delete_fn=store.delete,
    )
    assert dropped == ["loci_0", "loci_1", "loci_2", "loci_3", "loci_4"]


def _state_at(timestamp_ms):
    from loci.schema import WorldState

    return WorldState(
        x=0.5, y=0.5, z=0.5, timestamp_ms=timestamp_ms, vector=[1.0, 0.0], scene_id=""
    )


@pytest.mark.parametrize("prefix", ["", "tenant_a_"])
def test_local_reinsert_into_purged_epoch_recreates_collection(prefix):
    from loci.local_client import LocalLociClient

    client = LocalLociClient(
        vector_size=2, collection_prefix=prefix, retention_policy=RetentionPolicy(max_epochs=1)
    )
    client.insert(_state_at(0))
    newest = client.insert(_state_at(5000))
    assert client._known_collections == {f"{prefix}loci_1"}
    assert not client.store.collection_exists(f"{prefix}loci_0")
    # An out-of-order arrival must be inserted safely, then expire under policy.
    client.insert(_state_at(1000))
    assert client._known_collections == {f"{prefix}loci_1"}
    assert [hit.id for hit in client.query([1.0, 0.0])] == [newest]


def test_local_failed_purge_keeps_cached_collection():
    from unittest.mock import patch

    from loci.local_client import LocalLociClient

    client = LocalLociClient(vector_size=2)
    client.insert(_state_at(0))
    client.insert(_state_at(5000))
    client._retention = RetentionManager(RetentionPolicy(max_epochs=1), epoch_size_ms=5000)
    with patch.object(client.store, "delete_collection", side_effect=RuntimeError("unavailable")):
        client._maybe_purge()
    assert client._known_collections == {"loci_0", "loci_1"}
    assert client.store.collection_exists("loci_0")


@pytest.mark.parametrize("prefix", ["", "tenant_a_"])
def test_qdrant_reinsert_into_purged_epoch_recreates_collection(prefix):
    from unittest.mock import patch

    from qdrant_client import QdrantClient

    from loci.client import LociClient

    with contextlib.closing(QdrantClient(":memory:")) as backend:
        with patch("loci.client.QdrantClient", return_value=backend):
            client = LociClient(
                qdrant_url="http://unused",
                vector_size=2,
                collection_prefix=prefix,
                retention_policy=RetentionPolicy(max_epochs=1),
            )
        client.insert(_state_at(0))
        newest = client.insert(_state_at(5000))
        assert client._known_collections == {f"{prefix}loci_1"}
        assert not backend.collection_exists(f"{prefix}loci_0")
        client.insert(_state_at(1000))
        assert client._known_collections == {f"{prefix}loci_1"}
        assert [hit.id for hit in client.query([1.0, 0.0])] == [newest]


@pytest.mark.parametrize("prefix", ["", "tenant_a_"])
@pytest.mark.asyncio
async def test_async_qdrant_reinsert_into_purged_epoch_recreates_collection(prefix):
    from unittest.mock import patch

    from qdrant_client import AsyncQdrantClient

    from loci.async_client import AsyncLociClient

    backend = AsyncQdrantClient(":memory:")
    try:
        with patch("loci.async_client.AsyncQdrantClient", return_value=backend):
            client = AsyncLociClient(
                qdrant_url="http://unused",
                vector_size=2,
                collection_prefix=prefix,
                retention_policy=RetentionPolicy(max_epochs=1),
            )
        await client.insert(_state_at(0))
        newest = await client.insert(_state_at(5000))
        assert client._known_collections == {f"{prefix}loci_1"}
        assert not await backend.collection_exists(f"{prefix}loci_0")
        await client.insert(_state_at(1000))
        assert client._known_collections == {f"{prefix}loci_1"}
        hits = await client.query([1.0, 0.0])
        assert [hit.id for hit in hits] == [newest]
    finally:
        await backend.close()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["raise", "false"])
@pytest.mark.asyncio
async def test_failed_backend_purge_keeps_cached_collection(async_mode, failure):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from loci.async_client import AsyncLociClient
    from loci.client import LociClient

    backend = MagicMock()
    listing = SimpleNamespace(
        collections=[SimpleNamespace(name=f"tenant_a_loci_{e}") for e in (0, 1)]
    )
    delete_options = (
        {"side_effect": RuntimeError("unavailable")}
        if failure == "raise"
        else {
            "return_value": False,
        }
    )
    if async_mode:
        backend.get_collections = AsyncMock(return_value=listing)
        backend.delete_collection = AsyncMock(**delete_options)
        with patch("loci.async_client.AsyncQdrantClient", return_value=backend):
            client = AsyncLociClient(
                qdrant_url="http://unused",
                collection_prefix="tenant_a_",
                retention_policy=RetentionPolicy(max_epochs=1),
            )
        await client._maybe_purge()
    else:
        backend.get_collections.return_value = listing
        backend.delete_collection = MagicMock(**delete_options)
        with patch("loci.client.QdrantClient", return_value=backend):
            client = LociClient(
                qdrant_url="http://unused",
                collection_prefix="tenant_a_",
                retention_policy=RetentionPolicy(max_epochs=1),
            )
        client._maybe_purge()
    assert client._known_collections == {"tenant_a_loci_0", "tenant_a_loci_1"}
    backend.delete_collection.assert_called_once_with("tenant_a_loci_0")
