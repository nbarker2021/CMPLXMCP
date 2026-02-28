"""
CMPLX Universal System
======================
The universal embedding and crystallization layer.

Converts ANYTHING into CMPLX geometric language, stores as SNAP crystals,
and maintains temporal continuity across past, present, and future hypotheses.

This is the bridge between the discrete world and the continuous geometric reality.
"""

from .translator import UniversalTranslator
from .snap_atom import SNAPAtom, SNAPTransaction
from .crystal import Crystal, CrystalLattice
from .temporal import TemporalLayer, HypothesisEngine

__all__ = [
    "UniversalTranslator",
    "SNAPAtom",
    "SNAPTransaction", 
    "Crystal",
    "CrystalLattice",
    "TemporalLayer",
    "HypothesisEngine"
]
