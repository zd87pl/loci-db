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


@pytest.mark.asyncio
async def test_retention_manager_async_awaits_future_returning_deleter():
    """Deleters returning Futures/Tasks (not bare coroutines) must be awaited."""
    import asyncio

    store = FakeStore()
    for e in range(6):
        store.add(f"loci_{e}")

    async def _delete_coro(name: str) -> None:
        store.delete(name)

    def future_deleter(name: str) -> asyncio.Task:
        # Returns a Task (awaitable but not a coroutine object).
        return asyncio.ensure_future(_delete_coro(name))

    policy = RetentionPolicy(max_epochs=3)
    mgr = RetentionManager(policy, epoch_size_ms=5000)
    dropped = await mgr.maybe_purge_async(
        active_epochs=store.list_active(),
        now_ms=int(time.time() * 1000),
        delete_fn=future_deleter,
    )

    assert dropped == ["loci_0", "loci_1", "loci_2"]
    # The deletions actually ran (the Tasks were awaited, not abandoned).
    assert store.collections == {"loci_3", "loci_4", "loci_5"}


# ---------------------------------------------------------------------------
# Client-level cache invalidation after purge
# ---------------------------------------------------------------------------


def test_local_client_late_insert_into_purged_epoch_recreates():
    """A late insert into a purged epoch must recreate the collection."""
    from loci.local_client import LocalLociClient
    from loci.schema import WorldState

    def _state(ts: int) -> WorldState:
        return WorldState(
            x=0.5, y=0.5, z=0.5, timestamp_ms=ts, vector=[1.0, 0.0, 0.0, 0.0], scene_id=""
        )

    client = LocalLociClient(
        vector_size=4,
        epoch_size_ms=5000,
        decay_lambda=0.0,
        retention_policy=RetentionPolicy(max_epochs=1),
    )
    client.insert(_state(1000))  # loci_0
    client.insert(_state(6000))  # loci_1 → loci_0 purged

    assert "loci_0" not in client._known_collections
    assert not client.store.collection_exists("loci_0")

    # This previously raised KeyError because the client still believed the
    # purged collection existed. The insert recreates the collection; the
    # retention pass at the end of insert() then re-purges the stale epoch
    # (policy-consistent), leaving cache and store in sync.
    state_id = client.insert(_state(2000))
    assert isinstance(state_id, str)
    assert client.store.collection_exists("loci_0") == ("loci_0" in client._known_collections)


def test_local_client_purge_forgets_dropped_collections():
    from loci.local_client import LocalLociClient
    from loci.schema import WorldState

    client = LocalLociClient(
        vector_size=4,
        epoch_size_ms=5000,
        decay_lambda=0.0,
        retention_policy=RetentionPolicy(max_epochs=2),
    )
    for ts in (1000, 6000, 11_000, 16_000):
        client.insert(
            WorldState(
                x=0.5, y=0.5, z=0.5, timestamp_ms=ts, vector=[1.0, 0.0, 0.0, 0.0], scene_id=""
            )
        )

    assert client._known_collections == {"loci_2", "loci_3"}
    assert client._list_active_epochs() == [2, 3]
