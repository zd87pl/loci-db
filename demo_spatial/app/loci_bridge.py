"""Thin bridge around the private ``LocalLociClient`` surface used by the demo.

Design decision D10 / risk 11 of ``docs/HANDHELD_DEMO_DESIGN.md``: the only
private client member the demo needs is the name of the raw data collection
(``client._data_collection``), which the keyed merge uses to delete the old
point before re-inserting the merged one.  Every touch of that private surface
lives here, so a rename upstream breaks exactly one small module.
"""

from __future__ import annotations

from typing import Any

from loci import LocalLociClient

__all__ = [
    "data_collection",
    "delete_point",
    "delete_points",
    "get_payload",
    "make_client",
    "point_count",
]


def data_collection(client: LocalLociClient) -> str:
    """Name of the raw data collection that holds every inserted point."""
    return client._data_collection


def delete_point(client: LocalLociClient, state_id: str) -> bool:
    """Remove one raw point; True when it existed."""
    return client.store.delete_points(data_collection(client), [state_id]) > 0


def delete_points(client: LocalLociClient, state_ids: list[str]) -> int:
    """Remove several raw points; returns how many were actually removed."""
    if not state_ids:
        return 0
    return client.store.delete_points(data_collection(client), list(state_ids))


def point_count(client: LocalLociClient) -> int:
    """Number of live raw points (summaries excluded)."""
    return client.store.collection_count(data_collection(client))


def get_payload(client: LocalLociClient, state_id: str) -> dict[str, Any] | None:
    """Payload of one raw point (``None`` when it does not exist)."""
    hits = client.store.retrieve(data_collection(client), [state_id])
    if not hits:
        return None
    payload = hits[0].get("payload")
    return dict(payload) if isinstance(payload, dict) else None


def make_client(epoch_size_ms: int, vector_size: int) -> LocalLociClient:
    """The client every ``SpatialMemory`` uses: cosine distance, default decay."""
    return LocalLociClient(
        epoch_size_ms=epoch_size_ms,
        vector_size=vector_size,
        distance="cosine",
    )
