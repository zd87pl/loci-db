"""Tests for cutoff-based temporal retention management."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from loci.temporal.retention import RetentionManager, RetentionPolicy, retention_cutoff_ms

EPOCH_MS = 5000


class FakeStore:
    """In-memory stand-in for a raw data collection keyed by timestamp."""

    def __init__(self):
        self.timestamps: list[int] = []
        self.delete_calls: list[int] = []

    def add(self, ts: int):
        self.timestamps.append(ts)

    def delete_before(self, cutoff_ms: int) -> int:
        self.delete_calls.append(cutoff_ms)
        kept = [ts for ts in self.timestamps if ts >= cutoff_ms]
        deleted = len(self.timestamps) - len(kept)
        self.timestamps = kept
        return deleted


# ---------------------------------------------------------------------------
# RetentionPolicy validation
# ---------------------------------------------------------------------------


class TestPolicyValidation:
    def test_requires_at_least_one_knob(self):
        with pytest.raises(ValueError, match="max_epochs or max_age_ms"):
            RetentionPolicy()

    @pytest.mark.parametrize("kwargs", [{"max_epochs": 0}, {"max_age_ms": 0}])
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(ValueError):
            RetentionPolicy(**kwargs)


# ---------------------------------------------------------------------------
# retention_cutoff_ms
# ---------------------------------------------------------------------------


class TestRetentionCutoff:
    def test_max_age_cutoff_aligned_down(self):
        policy = RetentionPolicy(max_age_ms=15_000)
        now = 10 * EPOCH_MS + 400  # mid-epoch 10
        # now - 15s = epoch 7 territory, aligned down to the epoch boundary.
        assert retention_cutoff_ms(now, EPOCH_MS, policy) == 7 * EPOCH_MS

    def test_max_epochs_keeps_that_many_epoch_slots(self):
        # max_epochs=N retains N epoch-wide slots including the current
        # (possibly partial) epoch: cutoff is the start of epoch now-N+1.
        policy = RetentionPolicy(max_epochs=3)
        now = 10 * EPOCH_MS + 400
        assert retention_cutoff_ms(now, EPOCH_MS, policy) == 8 * EPOCH_MS

    def test_max_epochs_one_keeps_only_current_epoch(self):
        policy = RetentionPolicy(max_epochs=1)
        now = 10 * EPOCH_MS + 400
        assert retention_cutoff_ms(now, EPOCH_MS, policy) == 10 * EPOCH_MS

    def test_never_splits_an_epoch(self):
        policy = RetentionPolicy(max_age_ms=7_777)
        for now in (10 * EPOCH_MS, 10 * EPOCH_MS + 1, 11 * EPOCH_MS - 1):
            assert retention_cutoff_ms(now, EPOCH_MS, policy) % EPOCH_MS == 0

    def test_most_aggressive_knob_wins(self):
        now = 100 * EPOCH_MS
        loose_age = RetentionPolicy(max_epochs=2, max_age_ms=50 * EPOCH_MS)
        tight_age = RetentionPolicy(max_epochs=50, max_age_ms=2 * EPOCH_MS)
        assert retention_cutoff_ms(now, EPOCH_MS, loose_age) == 99 * EPOCH_MS
        assert retention_cutoff_ms(now, EPOCH_MS, tight_age) == 98 * EPOCH_MS

    def test_clamped_to_zero_at_beginning_of_time(self):
        policy = RetentionPolicy(max_age_ms=10**9)
        assert retention_cutoff_ms(1000, EPOCH_MS, policy) == 0


# ---------------------------------------------------------------------------
# RetentionManager
# ---------------------------------------------------------------------------


class TestRetentionManager:
    def test_purges_points_below_cutoff(self):
        store = FakeStore()
        for e in range(10):
            store.add(e * EPOCH_MS + 100)

        mgr = RetentionManager(RetentionPolicy(max_epochs=5), epoch_size_ms=EPOCH_MS)
        deleted = mgr.maybe_purge(9 * EPOCH_MS + 400, store.delete_before)

        assert deleted == 5  # epochs 0-4 purged, epochs 5-9 kept
        assert store.timestamps == [e * EPOCH_MS + 100 for e in range(5, 10)]

    def test_no_purge_inside_retention_window(self):
        store = FakeStore()
        store.add(100)
        mgr = RetentionManager(RetentionPolicy(max_epochs=999), epoch_size_ms=EPOCH_MS)
        assert mgr.maybe_purge(9 * EPOCH_MS, store.delete_before) is None
        assert store.delete_calls == []
        assert store.timestamps == [100]

    def test_trigger_cadence_throttled_until_cutoff_advances(self):
        store = FakeStore()
        for e in range(4):
            store.add(e * EPOCH_MS)
        mgr = RetentionManager(RetentionPolicy(max_epochs=2), epoch_size_ms=EPOCH_MS)

        mgr.maybe_purge(3 * EPOCH_MS + 100, store.delete_before)
        # Same epoch again: cutoff unchanged, deleter must not run again.
        assert mgr.maybe_purge(3 * EPOCH_MS + 200, store.delete_before) is None
        assert len(store.delete_calls) == 1
        # Next epoch: cutoff advances, deleter runs once more.
        store.add(4 * EPOCH_MS)
        assert mgr.maybe_purge(4 * EPOCH_MS + 100, store.delete_before) == 1
        assert store.delete_calls == [2 * EPOCH_MS, 3 * EPOCH_MS]

    def test_failed_delete_retries_on_next_call(self):
        calls: list[int] = []

        def flaky_delete(cutoff_ms: int) -> int:
            calls.append(cutoff_ms)
            if len(calls) == 1:
                raise RuntimeError("backend down")
            return 7

        mgr = RetentionManager(RetentionPolicy(max_epochs=1), epoch_size_ms=EPOCH_MS)
        with pytest.raises(RuntimeError):
            mgr.maybe_purge(5 * EPOCH_MS, flaky_delete)
        # The cutoff was not marked applied, so the same purge retries.
        assert mgr.maybe_purge(5 * EPOCH_MS, flaky_delete) == 7
        assert calls == [5 * EPOCH_MS, 5 * EPOCH_MS]


@pytest.mark.asyncio
async def test_retention_manager_async():
    store = FakeStore()
    for e in range(10):
        store.add(e * EPOCH_MS + 100)

    mgr = RetentionManager(RetentionPolicy(max_epochs=5), epoch_size_ms=EPOCH_MS)
    deleted = await mgr.maybe_purge_async(9 * EPOCH_MS + 400, store.delete_before)

    assert deleted == 5
    assert store.timestamps == [e * EPOCH_MS + 100 for e in range(5, 10)]


@pytest.mark.asyncio
async def test_retention_manager_async_awaits_future_returning_deleter():
    """Deleters returning Futures/Tasks (not bare coroutines) must be awaited."""
    import asyncio

    store = FakeStore()
    for e in range(6):
        store.add(e * EPOCH_MS)

    async def _delete_coro(cutoff_ms: int) -> int:
        return store.delete_before(cutoff_ms)

    def future_deleter(cutoff_ms: int) -> asyncio.Task:
        # Returns a Task (awaitable but not a coroutine object).
        return asyncio.ensure_future(_delete_coro(cutoff_ms))

    mgr = RetentionManager(RetentionPolicy(max_epochs=3), epoch_size_ms=EPOCH_MS)
    deleted = await mgr.maybe_purge_async(5 * EPOCH_MS + 1, store.delete_before)

    assert deleted == 3
    # The deletions actually ran (the Task was awaited, not abandoned).
    assert store.timestamps == [3 * EPOCH_MS, 4 * EPOCH_MS, 5 * EPOCH_MS]

    # And a Future-returning deleter is awaited on the next advance.
    store.add(6 * EPOCH_MS)
    deleted = await mgr.maybe_purge_async(6 * EPOCH_MS + 1, future_deleter)
    assert deleted == 1
    assert store.timestamps == [4 * EPOCH_MS, 5 * EPOCH_MS, 6 * EPOCH_MS]


# ---------------------------------------------------------------------------
# Client-level retention over the bounded layout
# ---------------------------------------------------------------------------


def _state(ts: int):
    from loci.schema import WorldState

    return WorldState(
        x=0.5, y=0.5, z=0.5, timestamp_ms=ts, vector=[1.0, 0.0, 0.0, 0.0], scene_id=""
    )


def _client(max_epochs: int):
    from loci.local_client import LocalLociClient

    return LocalLociClient(
        vector_size=4,
        epoch_size_ms=EPOCH_MS,
        decay_lambda=0.0,
        retention_policy=RetentionPolicy(max_epochs=max_epochs),
    )


def _pinned(ts_ms: int):
    return patch("loci.local_client.time.time", return_value=ts_ms / 1000.0)


def test_local_client_purges_expired_raw_points():
    client = _client(max_epochs=2)
    for ts in (1000, 6000, 11_000, 16_000):
        with _pinned(ts):
            client.insert(_state(ts))

    # Epoch 3 is current; max_epochs=2 keeps epochs 2-3 only.
    kept = [p["payload"]["timestamp_ms"] for p in client.store.scroll("loci_data", limit=100)]
    assert sorted(kept) == [11_000, 16_000]


def test_local_client_purge_never_splits_an_epoch():
    client = _client(max_epochs=1)
    with _pinned(1000):
        client.insert(_state(1000))
    with _pinned(6100):
        client.insert(_state(6000))  # crosses into epoch 1: epoch 0 purged whole

    kept = [p["payload"]["timestamp_ms"] for p in client.store.scroll("loci_data", limit=100)]
    assert kept == [6000]


def test_local_client_late_insert_into_purged_range_still_works():
    """A late insert older than the applied cutoff must not crash.

    The point lands in the (single) data collection and is swept up when
    the cutoff next advances — there is no per-epoch collection to
    recreate any more.
    """
    client = _client(max_epochs=1)
    with _pinned(1000):
        client.insert(_state(1000))
    with _pinned(6100):
        client.insert(_state(6000))

    with _pinned(6200):
        state_id = client.insert(_state(2000))  # older than the applied cutoff
    assert isinstance(state_id, str)

    # The straggler is purged as soon as the cutoff advances again.
    with _pinned(11_000):
        client.insert(_state(11_000))
    kept = [p["payload"]["timestamp_ms"] for p in client.store.scroll("loci_data", limit=100)]
    assert sorted(kept) == [11_000]
