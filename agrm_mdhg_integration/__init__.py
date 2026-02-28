"""
AGRM + MDHG + MMDB Integration
==============================

The complete geometric routing and caching layer for CMPLX MCP OS.

Provides:
- MDHGMultiScale: Multi-timescale geometric cache
- CAFieldMultiScale: Self-regulating cellular automata dynamics
- AGRMRouter: Inter-planet routing using GR sweeps
- Planet: Self-contained unit combining all three
- PlanetNetwork: Multi-planet system with ribbon communication

Integration with Universal System:
- Crystals are admitted to MDHG as 24D vectors
- CA channels reflect crystal properties
- AGRM routes queries between planets
- MMDB provides long-term persistence
"""

from .mdhg_ca import MDHGCache, MDHGMultiScale, CAField, CAFieldMultiScale, WolframAssignment
from .agrm_router import AGRMRouter, AGRMRoute
from .planet import Planet, PlanetConfig
from .network import PlanetNetwork, Ribbon

__all__ = [
    "MDHGCache",
    "MDHGMultiScale", 
    "CAField",
    "CAFieldMultiScale",
    "WolframAssignment",
    "AGRMRouter",
    "AGRMRoute",
    "Planet",
    "PlanetConfig",
    "PlanetNetwork",
    "Ribbon"
]
