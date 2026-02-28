"""
Geometric Adapter (Layer 2)
===========================
Bridges Layer 2 geometric operations to MCP server.
Heavy lattice operations happen server-side.
"""

import numpy as np
from typing import Any

from .base import MCPAdapter


class GeometricAdapter(MCPAdapter):
    """Adapter for Layer 2: Geometric Engine."""
    
    @property
    def layer(self) -> int:
        return 2
    
    async def initialize(self) -> bool:
        """Initialize geometric systems."""
        # NO heavy data loaded locally
        # E8 roots, Leech lattice - all server-side
        return True
    
    async def project_e8(self, vector: list[float], use_server: bool = True) -> dict:
        """
        Project to E8 lattice.
        
        By default uses server (where E8 roots are stored).
        Can do local approximation if needed.
        """
        if use_server and self._client:
            return await self._client.project_e8(vector)
        
        # Local approximation (no E8 roots loaded)
        v = np.array(vector)
        v = v / np.linalg.norm(v)
        return {
            "projected": v.tolist(),
            "approximation": True,
            "note": "For true E8 projection, use server"
        }
    
    async def nearest_leech(self, vector: list[float]) -> dict:
        """
        Find nearest Leech lattice point.
        
        ALWAYS uses server - Leech lattice is 196,560 vectors.
        Never load this locally.
        """
        if not self._client:
            raise RuntimeError("Leech lattice requires MCP server connection")
        
        return await self._client.nearest_leech(vector)
    
    async def navigate_weyl(self, position: list[float], 
                           target_root: list[float] | None = None) -> dict:
        """
        Navigate Weyl chambers.
        
        696,729,600 chambers - must use server.
        """
        if not self._client:
            raise RuntimeError("Weyl navigation requires MCP server")
        
        return await self._client.navigate_weyl(position, target_root)
    
    async def classify_niemeier(self, vector: list[float]) -> dict:
        """Classify against 24 Niemeier lattices."""
        if self._client:
            return await self._client.classify_niemeier(vector)
        
        # Local approximation
        return {"classification": "approximation", "local": True}
