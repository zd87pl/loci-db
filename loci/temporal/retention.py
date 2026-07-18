"""Temporal retention management — cutoff-based purging of old raw points.

Long-running deployments (drones, vehicles, wearables) write raw points
forever.  Without retention, the raw data collection grows unbounded.

Retention reduces to a single cutoff timestamp: raw points with
``timestamp_ms < cutoff`` expire.  Both policy knobs (``max_epochs``,
``max_age_ms``) compute a cutoff, the most aggressive wins, and the cutoff
is aligned down to an epoch boundary so retention never splits an epoch.
Purging only ever touches the raw data collection — consolidated summaries
are never purged.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Policy controlling how old raw points are purged.

    At least one of *max_epochs* or *max_age_ms* must be set.

    Attributes:
        max_epochs: Number of epoch-wide time slots to retain, counting the
            current (possibly partial) epoch.  Raw points older than the
            resulting window are purged.
        max_age_ms: Maximum age in milliseconds.  Raw points older than
            ``now_ms - max_age_ms`` (aligned down to an epoch boundary)
            are purged.
    """

    max_epochs: int | None = None
    max_age_ms: int | None = None

    def __post_init__(self) -> None:
        if self.max_epochs is None and self.max_age_ms is None:
            raise ValueError("RetentionPolicy requires max_epochs or max_age_ms")
        if self.max_epochs is not None and self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.max_age_ms is not None and self.max_age_ms < 1:
            raise ValueError("max_age_ms must be >= 1")


def retention_cutoff_ms(
    now_ms: int,
    epoch_size_ms: int,
    policy: RetentionPolicy,
) -> int:
    """Return the expiry cutoff: raw points with ``timestamp_ms < cutoff`` expire.

    Each configured knob reduces to a cutoff timestamp:

    - ``max_age_ms``: ``now_ms - max_age_ms``;
    - ``max_epochs``: ``now_ms - (max_epochs - 1) * epoch_size_ms``, which
      retains exactly ``max_epochs`` epoch-wide slots including the current
      (possibly partial) epoch.

    The most aggressive (largest) cutoff wins, aligned DOWN to an epoch
    boundary so retention never splits an epoch, and clamped to 0 while the
    retention window still reaches back to the beginning of time.
    """
    candidates: list[int] = []
    if policy.max_age_ms is not None:
        candidates.append(now_ms - policy.max_age_ms)
    if policy.max_epochs is not None:
        candidates.append(now_ms - (policy.max_epochs - 1) * epoch_size_ms)
    aligned = (max(candidates) // epoch_size_ms) * epoch_size_ms
    return max(0, aligned)


class RetentionManager:
    """Applies a :class:`RetentionPolicy` through an injected deleter.

    Works with any backend via ``delete_before(cutoff_ms)`` — a callable
    that deletes raw points with ``timestamp_ms < cutoff_ms`` from the raw
    data collection and returns the deleted count (or ``None`` when the
    backend cannot count cheaply).  The summary collection must never be
    touched by the deleter.

    Trigger-cadence throttling: clients call :meth:`maybe_purge` on every
    insert, but the deleter only runs when the cutoff has advanced past the
    last applied cutoff — i.e. at most once per epoch boundary crossing.
    """

    def __init__(self, policy: RetentionPolicy, *, epoch_size_ms: int) -> None:
        self._policy = policy
        self._epoch_size_ms = epoch_size_ms
        self._last_cutoff_ms = 0

    def maybe_purge(
        self,
        now_ms: int,
        delete_before: Callable[[int], int | None],
    ) -> int | None:
        """Evaluate the policy and delete expired raw points.

        Args:
            now_ms: Current timestamp in milliseconds.
            delete_before: Callable that deletes raw points with
                ``timestamp_ms`` strictly below the given cutoff.  Should
                raise on failure so the cutoff can be retried later.

        Returns:
            The deleted count from *delete_before* when a purge ran
            (``None`` if the backend does not count), or ``None`` when the
            cutoff had not advanced and no purge was needed.
        """
        cutoff = retention_cutoff_ms(now_ms, self._epoch_size_ms, self._policy)
        if cutoff <= self._last_cutoff_ms:
            return None
        deleted = delete_before(cutoff)
        self._last_cutoff_ms = cutoff
        if deleted:
            logger.info("Retention purged %s raw points below t=%d", deleted, cutoff)
        return deleted

    async def maybe_purge_async(
        self,
        now_ms: int,
        delete_before: Callable[[int], int | None | Awaitable[int | None]],
    ) -> int | None:
        """Async variant of :meth:`maybe_purge`.

        *delete_before* may return an awaitable (coroutine, Task, Future)
        or a plain value.
        """
        cutoff = retention_cutoff_ms(now_ms, self._epoch_size_ms, self._policy)
        if cutoff <= self._last_cutoff_ms:
            return None
        result = delete_before(cutoff)
        deleted = await result if inspect.isawaitable(result) else result
        self._last_cutoff_ms = cutoff
        if deleted:
            logger.info("Retention purged %s raw points below t=%d", deleted, cutoff)
        return deleted
