"""Engram — 4D spatiotemporal vector database for AI world models."""

from engram.async_client import AsyncEngramClient
from engram.client import EngramClient
from engram.local_client import LocalEngramClient, QueryStats
from engram.schema import WorldState
from engram.spatial.adaptive import DensityStats

__all__ = [
    "AsyncEngramClient",
    "DensityStats",
    "EngramClient",
    "LocalEngramClient",
    "QueryStats",
    "WorldState",
]
__version__ = "0.3.0"
