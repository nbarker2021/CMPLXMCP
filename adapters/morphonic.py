"""
Morphonic Adapter (Layer 1)
============================
Bridges Layer 1 morphonic operations to MCP server.
"""

import numpy as np
from typing import Any

from .base import MCPAdapter


class MorphonicAdapter(MCPAdapter):
    """Adapter for Layer 1: Morphonic Foundation."""
    
    @property
    def layer(self) -> int:
        return 1
    
    async def initialize(self) -> bool:
        """Initialize morphonic systems."""
        # Pre-compute common morphons locally (lightweight)
        for digit in range(10):
            morphon = self._generate_local_morphon(digit)
            self._cache_set(f"morphon_{digit}", morphon)
        return True
    
    def _generate_local_morphon(self, digit: int) -> dict:
        """Generate morphon locally (lightweight computation)."""
        return {
            "seed": digit,
            "digital_root": digit % 9 or 9,
            "resonance": np.exp(2j * np.pi * digit / 9).real
        }
    
    async def generate_morphon(self, seed: str, use_mcp: bool = False) -> dict:
        """
        Generate morphon from seed.
        
        Args:
            seed: Single digit seed
            use_mcp: If True, always use MCP server (for distributed ops)
        
        Returns:
            Morphon data (handle if using MCP)
        """
        digit = int(seed[0]) if seed else 0
        
        # Check local cache first
        cached = self._cache_get(f"morphon_{digit}")
        if cached and not use_mcp:
            return cached
        
        # Use MCP server
        if self._client:
            return await self._client.generate_morphon(seed)
        
        # Fallback to local
        return self._generate_local_morphon(digit)
    
    async def execute_mglc(self, expression: str, context: dict | None = None) -> dict:
        """Execute MGLC expression."""
        if self._client:
            return await self._client.execute_mglc(expression, context)
        
        # Local fallback
        return {"expression": expression, "status": "local_execution"}
    
    async def expand_seed(self, digit: int, dimensions: int = 24) -> dict:
        """Expand seed to substrate."""
        if self._client:
            return await self._client.expand_seed(digit, dimensions)
        
        # Local lightweight generation
        np.random.seed(digit)
        substrate = np.random.randn(dimensions)
        substrate = substrate / np.linalg.norm(substrate)
        
        return {
            "seed": digit,
            "dimensions": dimensions,
            "norm": float(np.linalg.norm(substrate)),
            "local": True
        }
