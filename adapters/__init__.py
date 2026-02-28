"""
CMPLX Adapters
==============
Bridge existing CMPLX components to the MCP architecture.
Allows gradual migration while maintaining backward compatibility.
"""

from .base import MCPAdapter, AdapterRegistry
from .morphonic import MorphonicAdapter
from .geometric import GeometricAdapter
from .operational import OperationalAdapter

__all__ = [
    "MCPAdapter",
    "AdapterRegistry", 
    "MorphonicAdapter",
    "GeometricAdapter",
    "OperationalAdapter"
]
