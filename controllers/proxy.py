"""
Controller Proxy Classes
========================
These provide the same interface as the original controllers but
all heavy operations are delegated to the MCP server.
"""

from abc import ABC, abstractmethod
from typing import Any


class ControllerProxy(ABC):
    """
    Base controller proxy.
    
    Maintains the same API as original controllers for compatibility,
    but all operations go through MCP server.
    """
    
    def __init__(self, client=None):
        self._client = client
        self._handle_cache = {}
    
    @property
    @abstractmethod
    def layer_name(self) -> str:
        pass
    
    def _store_handle(self, key: str, handle_data: dict):
        """Cache a handle locally."""
        self._handle_cache[key] = handle_data.get("handle")
    
    def _get_handle(self, key: str) -> str | None:
        """Retrieve cached handle."""
        return self._handle_cache.get(key)


class Layer1Proxy(ControllerProxy):
    """Proxy for Layer 1: Morphonic Foundation."""
    
    @property
    def layer_name(self) -> str:
        return "layer1_morphonic"
    
    async def generate_morphon(self, seed: str) -> dict:
        """Generate universal morphon."""
        if self._client:
            return await self._client.generate_morphon(seed)
        raise RuntimeError("No MCP client configured")
    
    async def execute_mglc(self, expression: str, context: dict | None = None) -> dict:
        """Execute MGLC expression."""
        if self._client:
            return await self._client.execute_mglc(expression, context)
        raise RuntimeError("No MCP client configured")
    
    async def expand_seed(self, digit: int, dimensions: int = 24) -> dict:
        """Expand seed to substrate."""
        if self._client:
            return await self._client.expand_seed(digit, dimensions)
        raise RuntimeError("No MCP client configured")


class Layer2Proxy(ControllerProxy):
    """Proxy for Layer 2: Geometric Engine."""
    
    @property
    def layer_name(self) -> str:
        return "layer2_geometric"
    
    async def project_e8(self, vector: list[float]) -> dict:
        """Project to E8 lattice."""
        if self._client:
            result = await self._client.project_e8(vector)
            self._store_handle("last_e8", result)
            return result
        raise RuntimeError("No MCP client configured")
    
    async def nearest_leech(self, vector: list[float]) -> dict:
        """Find nearest Leech lattice point."""
        if self._client:
            result = await self._client.nearest_leech(vector)
            self._store_handle("last_leech", result)
            return result
        raise RuntimeError("No MCP client configured")
    
    async def navigate_weyl(self, position: list[float], 
                           target: list[float] | None = None) -> dict:
        """Navigate Weyl chambers."""
        if self._client:
            return await self._client.navigate_weyl(position, target)
        raise RuntimeError("No MCP client configured")
    
    async def classify_niemeier(self, vector: list[float]) -> dict:
        """Classify against Niemeier lattices."""
        if self._client:
            return await self._client.classify_niemeier(vector)
        raise RuntimeError("No MCP client configured")


class Layer3Proxy(ControllerProxy):
    """Proxy for Layer 3: Operational Systems."""
    
    @property
    def layer_name(self) -> str:
        return "layer3_operational"
    
    async def morsr_optimize(self, initial_state: list[float], 
                            iterations: int = 100) -> dict:
        """Run MORSR optimization."""
        if self._client:
            result = await self._client.morsr_optimize(initial_state, iterations)
            self._store_handle("last_morsr", result)
            return result
        raise RuntimeError("No MCP client configured")
    
    async def check_conservation(self, before: list[float], 
                                 after: list[float]) -> dict:
        """Check conservation law."""
        if self._client:
            return await self._client.check_conservation(before, after)
        raise RuntimeError("No MCP client configured")


class Layer4Proxy(ControllerProxy):
    """Proxy for Layer 4: Governance."""
    
    @property
    def layer_name(self) -> str:
        return "layer4_governance"
    
    async def digital_root(self, number: int) -> dict:
        """Calculate digital root."""
        if self._client:
            return await self._client.digital_root(number)
        raise RuntimeError("No MCP client configured")
    
    async def seven_witness(self, artifact: dict) -> dict:
        """Run seven-witness validation."""
        if self._client:
            return await self._client.seven_witness(artifact)
        raise RuntimeError("No MCP client configured")
    
    async def policy_check(self, artifact_id: str, tier: int = 1) -> dict:
        """Check against policy hierarchy."""
        if self._client:
            return await self._client.policy_check(artifact_id, tier)
        raise RuntimeError("No MCP client configured")


class Layer5Proxy(ControllerProxy):
    """Proxy for Layer 5: Interface."""
    
    @property
    def layer_name(self) -> str:
        return "layer5_interface"
    
    async def embed(self, content: str, domain: str = "text") -> dict:
        """Embed content to E8."""
        if self._client:
            result = await self._client.embed(content, domain)
            self._store_handle(f"embed_{domain}", result)
            return result
        raise RuntimeError("No MCP client configured")
    
    async def query_similar(self, handle: str, top_k: int = 10) -> dict:
        """Query similar overlays."""
        if self._client:
            return await self._client.query_similar(handle, top_k)
        raise RuntimeError("No MCP client configured")
    
    async def transform(self, handle: str, operator: str, 
                       params: dict | None = None) -> dict:
        """Apply transformation."""
        if self._client:
            return await self._client.transform(handle, operator, params)
        raise RuntimeError("No MCP client configured")
