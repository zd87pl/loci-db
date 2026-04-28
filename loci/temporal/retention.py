"""Temporal epoch retention management.

Long-running deployments (drones, vehicles, wearables) generate one Qdrant
collection per epoch. Without retention, collection counts grow unbounded,
degrading discovery performance and consuming storage.

This module provides configurable policies to drop or archive old epochs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from loci.temporal.sharding import collection_name, epoch_id

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Policy controlling how old temporal epochs are purged.

    At least one of *max_epochs* or *max_age_ms* must be set.

    Attributes:
        max_epochs: Maximum number of epochs to retain. Oldest are dropped first.
        max_age_ms: Maximum age in milliseconds. Epochs older than
            ``now_ms - max_age_ms`` are dropped.
        archive_callback: Optional async callback ``fn(epoch_id, collection_name)``
            invoked before a collection is dropped. If the callback raises,
            the drop is aborted.
    """

    max_epochs: int | None = None
    max_age_ms: int | None = None
    archive_callback: Callable[[int, str], object] | None = None

    def __post_init__(self) -> None:
        if self.max_epochs is None and self.max_age_ms is None:
            raise ValueError("RetentionPolicy requires max_epochs or max_age_ms")
        if self.max_epochs is not None and self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.max_age_ms is not None and self.max_age_ms < 1:
            raise ValueError("max_age_ms must be >= 1")


def epochs_to_drop(
    active_epochs: list[int],
    *,
    now_ms: int,
    epoch_size_ms: int,
    policy: RetentionPolicy,
) -> list[int]:
    """Return the list of epoch IDs that should be purged under *policy*.

    Args:
        active_epochs: Sorted list of epoch IDs currently stored.
        now_ms: Current timestamp in milliseconds.
        epoch_size_ms: Width of each epoch in milliseconds.
        policy: Retention policy to apply.

    Returns:
        Sorted list of epoch IDs that exceed the retention limits.
    """
    if not active_epochs:
        return []

    to_drop: set[int] = set()

    if policy.max_age_ms is not None:
        cutoff_epoch = epoch_id(now_ms - policy.max_age_ms, epoch_size_ms)
        to_drop.update(ep for ep in active_epochs if ep < cutoff_epoch)

    if policy.max_epochs is not None and len(active_epochs) > policy.max_epochs:
        # Keep the *most recent* max_epochs
        sorted_epochs = sorted(active_epochs, reverse=True)
        to_drop.update(sorted_epochs[policy.max_epochs :])

    return sorted(to_drop)


class RetentionManager:
    """Manages lifecycle of temporal epoch collections.

    Works with both Qdrant-backed and in-memory backends via injected
    delete callbacks.
    """

    def __init__(
        self,
        policy: RetentionPolicy,
        *,
        epoch_size_ms: int,
        collection_prefix: str = "",
    ) -> None:
        self._policy = policy
        self._epoch_size_ms = epoch_size_ms
        self._collection_prefix = collection_prefix

    def _col_name(self, ep: int) -> str:
        base = collection_name(ep)
        return f"{self._collection_prefix}{base}" if self._collection_prefix else base

    def maybe_purge(
        self,
        active_epochs: list[int],
        now_ms: int,
        delete_fn: Callable[[str], object],
    ) -> list[str]:
        """Evaluate retention policy and delete expired collections.

        Args:
            active_epochs: Currently-known epoch IDs.
            now_ms: Current timestamp in milliseconds.
            delete_fn: Callable that deletes a collection by name.
                Should raise on failure.

        Returns:
            List of collection names that were successfully dropped.
        """
        dropped: list[str] = []
        for ep in epochs_to_drop(
            active_epochs,
            now_ms=now_ms,
            epoch_size_ms=self._epoch_size_ms,
            policy=self._policy,
        ):
            col = self._col_name(ep)
            if self._policy.archive_callback is not None:
                try:
                    self._policy.archive_callback(ep, col)
                except Exception:
                    logger.warning(
                        "Archive callback failed for %s; skipping drop",
                        col,
                        exc_info=True,
                    )
                    continue
            try:
                delete_fn(col)
                dropped.append(col)
                logger.info("Dropped expired collection %s (epoch %d)", col, ep)
            except Exception:
                logger.warning("Failed to drop collection %s", col, exc_info=True)
        return dropped

    async def maybe_purge_async(
        self,
        active_epochs: list[int],
        now_ms: int,
        delete_fn: Callable[[str], object],
    ) -> list[str]:
        """Async variant of :meth:`maybe_purge`.

        *delete_fn* may be a coroutine or a regular callable.
        """
        import asyncio
        import inspect

        dropped: list[str] = []
        for ep in epochs_to_drop(
            active_epochs,
            now_ms=now_ms,
            epoch_size_ms=self._epoch_size_ms,
            policy=self._policy,
        ):
            col = self._col_name(ep)
            if self._policy.archive_callback is not None:
                try:
                    cb_result = self._policy.archive_callback(ep, col)
                    if inspect.isawaitable(cb_result):
                        await cb_result
                except Exception:
                    logger.warning(
                        "Archive callback failed for %s; skipping drop",
                        col,
                        exc_info=True,
                    )
                    continue
            try:
                del_result = delete_fn(col)
                if asyncio.iscoroutine(del_result):
                    await del_result
                dropped.append(col)
                logger.info("Dropped expired collection %s (epoch %d)", col, ep)
            except Exception:
                logger.warning("Failed to drop collection %s", col, exc_info=True)
        return dropped
