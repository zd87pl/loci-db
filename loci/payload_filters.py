"""Helpers for translating internal payload filters across backends."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, MatchAny, MatchValue


def extra_filter_to_conditions(extra_filter: dict[str, Any] | None) -> list[FieldCondition]:
    """Convert internal extra payload filters to Qdrant conditions."""
    if not extra_filter:
        return []

    conditions: list[FieldCondition] = []
    for key, value in extra_filter.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return conditions


def extra_filter_to_memory(extra_filter: dict[str, Any] | None) -> dict[str, Any]:
    """Convert internal extra payload filters to MemoryStore semantics."""
    if not extra_filter:
        return {}

    payload_filter: dict[str, Any] = {}
    for key, value in extra_filter.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            payload_filter[key] = {"any": list(value)}
        else:
            payload_filter[key] = value
    return payload_filter
