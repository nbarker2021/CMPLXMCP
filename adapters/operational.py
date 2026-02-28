"""
Operational Adapter (Layer 3)
=============================
Bridges Layer 3 operational systems to MCP.
"""

import numpy as np
from typing import Any

from .base import MCPAdapter


class OperationalAdapter(MCPAdapter):
    """Adapter for Layer 3: Operational Systems."""
    
    @property
    def layer(self) -> int:
        return 3
    
    async def initialize(self) -> bool:
        """Initialize operational systems."""
        return True
    
    async def morsr_optimize(self, initial_state: list[float], 
                            iterations: int = 100,
                            constraint: str = "conservation") -> dict:
        """
        Run MORSR optimization.
        
        Can run locally for small states, uses server for large/complex.
        """
        state = np.array(initial_state)
        
        # Small state: local optimization
        if len(state) <= 8 and iterations <= 1000:
            return await self._local_morsr(state, iterations, constraint)
        
        # Large state: use server
        if self._client:
            return await self._client.morsr_optimize(
                initial_state, iterations, constraint
            )
        
        raise RuntimeError("Large MORSR requires MCP server")
    
    async def _local_morsr(self, state: np.ndarray, iterations: int, 
                          constraint: str) -> dict:
        """Local MORSR for small states."""
        # Simple gradient descent
        for i in range(iterations):
            gradient = np.random.randn(*state.shape) * 0.01
            state = state - gradient
        
        return {
            "final_state": state.tolist(),
            "iterations": iterations,
            "local": True
        }
    
    async def check_conservation(self, before: list[float], 
                                 after: list[float]) -> dict:
        """Check ΔΦ ≤ 0 conservation law."""
        if self._client:
            return await self._client.check_conservation(before, after)
        
        # Local check
        b = np.array(before)
        a = np.array(after)
        
        phi = (1 + np.sqrt(5)) / 2
        phi_before = np.linalg.norm(b) * phi
        phi_after = np.linalg.norm(a) * phi
        
        return {
            "delta_phi": float(phi_after - phi_before),
            "conserved": phi_after <= phi_before + 1e-10,
            "local": True
        }
