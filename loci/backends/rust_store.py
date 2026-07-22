"""Rust-native in-memory vector store — RFC-0001 R5 stage (a).

:class:`RustMemoryStore` is a behavioral drop-in for
:class:`loci.backends.memory.MemoryStore`, backed by the
``loci_core.LociStore`` native engine (contiguous f32 arena per
collection, tombstone deletes, brute-force SIMD-friendly scoring).

Semantics mirrored from the Python store:

- Scores are always higher-is-better (euclidean = negative L2).
- Filters: exact match, ``{"any": [...]}`` membership, and
  ``gte``/``lte``/``gt``/``lt`` numeric ranges.
- Copy-on-write / copy-on-read: points, payloads, and filters cross the
  FFI boundary by value (native dict/list <-> ``serde_json::Value``
  conversion inside the extension), so returned objects are always fresh
  Python objects and caller-side mutation can never corrupt stored points.

Differences from the Python store (documented, not observable through
``LocalLociClient``):

- Vectors are stored as float32, so values round-trip with f32 precision.
- Payloads must be JSON-representable (None/bool/int/float/str/list/
  tuple/dict) — the same constraint the Qdrant backend imposes;
  non-finite floats are rejected.
- Physical iteration order of unordered scrolls may differ after deletes
  (tombstoned rows are reused); ordered scrolls sort identically.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = (
    "install the native extension with `uv sync --group native` (or `pip install -e ./loci-core`)"
)

try:
    import loci_core as _core
except ImportError as _exc:  # pragma: no cover - exercised only without the extension
    raise ImportError(
        f"RustMemoryStore requires the loci_core native extension; {_INSTALL_HINT}"
    ) from _exc

if not hasattr(_core, "LociStore"):  # pragma: no cover - exercised only with stale wheels
    raise ImportError(
        f"the installed loci_core extension predates the native store "
        f"(no LociStore class); rebuild it: {_INSTALL_HINT}"
    )


class RustMemoryStore:
    """In-memory vector store with the same semantics as :class:`MemoryStore`.

    Thin delegation wrapper over ``loci_core.LociStore``.
    """

    def __init__(self) -> None:
        self._inner = _core.LociStore()

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def create_collection(self, name: str, vector_size: int, distance: str = "cosine") -> None:
        self._inner.create_collection(name, vector_size, distance)

    def collection_exists(self, name: str) -> bool:
        return self._inner.collection_exists(name)

    def delete_collection(self, name: str) -> None:
        """Remove a collection and all its points."""
        self._inner.delete_collection(name)

    def create_payload_index(self, collection: str, field_name: str) -> None:
        self._inner.create_payload_index(collection, field_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, collection: str, points: list[dict]) -> None:
        """Insert or update points (``id`` + ``vector`` + ``payload`` dicts).

        Raises:
            KeyError: If the collection does not exist.
            ValueError: If any vector does not match the collection's
                configured ``vector_size`` (earlier points in the batch
                stay inserted, matching the Python store).
        """
        self._inner.upsert(collection, points)

    def set_payload(self, collection: str, point_id: str, payload: dict) -> None:
        self._inner.set_payload(collection, str(point_id), payload)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_points(self, collection: str, ids: list[str]) -> int:
        """Delete points by id.  Returns the number actually removed."""
        return self._inner.delete_points(collection, [str(i) for i in ids])

    def delete_points_in_time_range(
        self,
        collection: str,
        start_ms: int,
        end_ms_exclusive: int,
        *,
        field: str = "timestamp_ms",
    ) -> int:
        """Delete points with ``start_ms <= payload[field] < end_ms_exclusive``."""
        return self._inner.delete_points_in_time_range(
            collection, start_ms, end_ms_exclusive, field
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(self, collection: str, ids: list[str]) -> list[dict]:
        return self._inner.retrieve(collection, [str(i) for i in ids])

    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        payload_filter: dict | None = None,
    ) -> list[dict]:
        """Brute-force ANN search with optional payload filtering.

        Same conventions as the Python store: results sorted by score
        descending, scores higher-is-better (euclidean negated).
        """
        return self._inner.search(collection, query_vector, limit, payload_filter)

    def scroll(
        self,
        collection: str,
        payload_filter: dict | None = None,
        limit: int = 10,
        order_by: str | None = None,
    ) -> list[dict]:
        return self._inner.scroll(collection, payload_filter, limit, order_by)

    @property
    def total_points(self) -> int:
        return self._inner.total_points

    def collection_count(self, name: str) -> int:
        return self._inner.collection_count(name)

    def payload_value_range(self, collection: str, field: str) -> tuple[Any, Any] | None:
        """Return ``(min, max)`` of a payload field across a collection."""
        return self._inner.payload_value_range(collection, field)
