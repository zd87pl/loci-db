"""World model adapters for LOCI.

Ready-to-use integrations for converting world model outputs
into WorldState objects for storage in LOCI.
"""

from loci.adapters.dreamer import DreamerV3Adapter
from loci.adapters.generic import GenericAdapter
from loci.adapters.vjepa2 import VJEPA2Adapter

__all__ = [
    "DreamerV3Adapter",
    "GenericAdapter",
    "VJEPA2Adapter",
]
