"""Retry logic with exponential backoff for transient Qdrant failures."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions considered transient (worth retrying)
_TRANSIENT_CODES = {502, 503, 504, 429}

# Optional dependencies: qdrant-client wraps httpx transport failures in
# ResponseHandlingException (which has no status_code and a class name that
# matches none of the keyword heuristics), so both must be handled explicitly.
try:  # pragma: no cover - import guard
    from qdrant_client.http.exceptions import ResponseHandlingException as _QdrantWrapped
except ImportError:  # pragma: no cover
    _QdrantWrapped = None

try:  # pragma: no cover - import guard
    import httpx as _httpx
except ImportError:  # pragma: no cover
    _httpx = None


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is likely transient and worth retrying."""
    # qdrant_client wraps the underlying httpx transport error; unwrap it so
    # connection failures against a restarting Qdrant are retried.
    if _QdrantWrapped is not None and isinstance(exc, _QdrantWrapped):
        source = getattr(exc, "source", None)
        if isinstance(source, Exception):
            return _is_transient(source)
    # httpx transport failures (ConnectError, ReadTimeout, ...) are transient.
    if _httpx is not None and isinstance(exc, _httpx.TransportError):
        return True
    # qdrant_client raises UnexpectedResponse with a status_code attribute
    status = getattr(exc, "status_code", None)
    if status is not None and status in _TRANSIENT_CODES:
        return True
    # Connection errors
    exc_name = type(exc).__name__
    return bool(any(keyword in exc_name for keyword in ("Connection", "Timeout", "Socket")))


def with_retry(
    max_retries: int = 3,
    backoff_base: float = 0.5,
) -> Callable:
    """Decorator that retries a function on transient failures.

    Uses exponential backoff: sleep = backoff_base * 2^attempt.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds (doubles each retry).
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if not _is_transient(exc) or attempt == max_retries:
                        raise
                    last_exc = exc
                    delay = backoff_base * (2**attempt)
                    logger.warning(
                        "Transient error on %s (attempt %d/%d), retrying in %.1fs: %s",
                        getattr(fn, "__name__", repr(fn)),
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            assert last_exc is not None  # pragma: no cover
            raise last_exc

        return wrapper

    return decorator


async def async_with_retry(
    fn: Callable,
    *args: Any,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    **kwargs: Any,
) -> Any:
    """Retry an async callable on transient failures.

    Args:
        fn: Async function to call.
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds.
    """
    import asyncio

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            if not _is_transient(exc) or attempt == max_retries:
                raise
            last_exc = exc
            delay = backoff_base * (2**attempt)
            logger.warning(
                "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None  # pragma: no cover
    raise last_exc
