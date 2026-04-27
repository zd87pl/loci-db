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
