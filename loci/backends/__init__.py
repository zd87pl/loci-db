"""Pluggable storage backends for Loci.

``MemoryStore`` (pure Python, always available) is the reference
implementation.  ``RustMemoryStore`` is its native drop-in and is loaded
lazily: importing it raises ``ImportError`` unless the ``loci_core``
extension is installed (``uv sync --group native``).
"""

from __future__ import annotations

from typing import Any

from loci.backends.memory import MemoryStore

__all__ = ["MemoryStore", "RustMemoryStore"]


def __getattr__(name: str) -> Any:
    if name == "RustMemoryStore":
        from loci.backends.rust_store import RustMemoryStore

        return RustMemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
