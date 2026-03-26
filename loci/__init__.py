"""Loci — 4D spatiotemporal vector database for AI world models."""

from loci.async_client import AsyncLociClient
from loci.client import LociClient
from loci.local_client import LocalLociClient, QueryStats
from loci.retrieval.predict import PredictRetrieveResult, PredictThenRetrieve
from loci.schema import ScoredWorldState, WorldState
from loci.spatial.adaptive import DensityStats
from loci.spatial.hilbert import HilbertIndex, SpatialBounds

__all__ = [
    "AsyncLociClient",
    "DensityStats",
    "HilbertIndex",
    "LociClient",
    "LocalLociClient",
    "PredictRetrieveResult",
    "PredictThenRetrieve",
    "QueryStats",
    "ScoredWorldState",
    "SpatialBounds",
    "WorldState",
]
__version__ = "0.3.0"
