"""
Base Adapter Class
==================
Abstract base for all CMPLX-MCP adapters.
"""

from abc import ABC, abstractmethod
from typing import Any


class MCPAdapter(ABC):
    """
    Base class for adapters that bridge CMPLX components to MCP.
    
    Adapters provide:
    1. Local lightweight interface
    2. MCP server communication when needed
    3. Caching for frequently accessed data
    """
    
    def __init__(self, client=None):
        self._client = client
        self._local_cache = {}
    
    @property
    @abstractmethod
    def layer(self) -> int:
        """Return the CMPLX layer this adapter serves."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the adapter."""
        pass
    
    def _cache_get(self, key: str) -> Any:
        """Get from local cache."""
        return self._local_cache.get(key)
    
    def _cache_set(self, key: str, value: Any):
        """Set in local cache."""
        self._local_cache[key] = value
    
    def _cache_clear(self):
        """Clear local cache."""
        self._local_cache.clear()


class AdapterRegistry:
    """Registry for all adapters."""
    
    _adapters: dict[int, MCPAdapter] = {}
    
    @classmethod
    def register(cls, layer: int, adapter: MCPAdapter):
        """Register an adapter for a layer."""
        cls._adapters[layer] = adapter
    
    @classmethod
    def get(cls, layer: int) -> MCPAdapter | None:
        """Get adapter for a layer."""
        return cls._adapters.get(layer)
    
    @classmethod
    async def initialize_all(cls):
        """Initialize all registered adapters."""
        for layer, adapter in sorted(cls._adapters.items()):
            await adapter.initialize()
