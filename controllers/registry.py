"""
Proxy Registry
==============
Manages all controller proxies and their lifecycle.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import CMPLXClient

from .proxy import Layer1Proxy, Layer2Proxy, Layer3Proxy, Layer4Proxy, Layer5Proxy
from .family_manager import FamilyControllerManager


class ProxyRegistry:
    """
    Registry for controller proxies.
    
    Provides easy access to all layers through a single interface.
    """
    
    def __init__(self, client: "CMPLXClient | None" = None):
        self._client = client
        self._proxies = {}
        self._family_manager = FamilyControllerManager()
        
        if client:
            self._initialize_proxies()
    
    def _initialize_proxies(self):
        """Create all layer proxies."""
        self._proxies = {
            1: Layer1Proxy(self._client),
            2: Layer2Proxy(self._client),
            3: Layer3Proxy(self._client),
            4: Layer4Proxy(self._client),
            5: Layer5Proxy(self._client),
        }
    
    @property
    def l1(self) -> Layer1Proxy:
        """Layer 1: Morphonic Foundation."""
        return self._proxies[1]
    
    @property
    def l2(self) -> Layer2Proxy:
        """Layer 2: Geometric Engine."""
        return self._proxies[2]
    
    @property
    def l3(self) -> Layer3Proxy:
        """Layer 3: Operational Systems."""
        return self._proxies[3]
    
    @property
    def l4(self) -> Layer4Proxy:
        """Layer 4: Governance."""
        return self._proxies[4]
    
    @property
    def l5(self) -> Layer5Proxy:
        """Layer 5: Interface."""
        return self._proxies[5]
    
    def get(self, layer: int):
        """Get proxy by layer number."""
        return self._proxies.get(layer)

    @property
    def families(self) -> FamilyControllerManager:
        """Family-based controller manager for donor/historical build wrappers."""
        return self._family_manager
