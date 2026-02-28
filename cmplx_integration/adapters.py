"""
CMPLX Component Adapters
=========================

High-level adapters for CMPLX toolkit components.

These provide a simplified interface for using CMPLX tools
through the MCP protocol.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..controllers.family_manager import FamilyControllerManager

logger = logging.getLogger("cmplx.mcp.adapters")


@dataclass
class QuorumResult:
    """Simplified Quorum result."""
    synthesis: str
    confidence: float
    agent_responses: Dict[str, Dict]
    consensus_areas: List[str]
    disagreements: List[str]
    deliberation_time_ms: float


class QuorumAdapter:
    """Adapter for Quorum deliberation system."""
    
    def __init__(self, registry):
        self.registry = registry
    
    async def deliberate(
        self,
        question: str,
        roles: List[str] = None,
        use_tools: bool = True,
        use_cache: bool = True
    ) -> QuorumResult:
        """Run quorum deliberation."""
        
        result = await self.registry.call("quorum_deliberate", {
            "question": question,
            "roles": roles or ["planner", "implementer", "critic", "researcher"],
            "use_tools": use_tools,
            "use_cache": use_cache
        })
        
        if "error" in result:
            raise RuntimeError(f"Quorum failed: {result['error']}")
        
        return QuorumResult(
            synthesis=result.get("synthesis", ""),
            confidence=result.get("synthesis_confidence", 0.0),
            agent_responses=result.get("role_responses", {}),
            consensus_areas=result.get("consensus_areas", []),
            disagreements=result.get("disagreements", []),
            deliberation_time_ms=result.get("deliberation_time_ms", 0)
        )
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.registry.call("quorum_check_cache", {})
    
    async def clear_cache(self):
        """Clear deliberation cache."""
        return await self.registry.call("quorum_clear_cache", {})


class ThinkTankAdapter:
    """Adapter for Think Tank autonomous system."""
    
    def __init__(self, registry):
        self.registry = registry
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Think Tank status."""
        return await self.registry.call("think_tank_status", {})
    
    async def start(self) -> bool:
        """Start Think Tank."""
        result = await self.registry.call("think_tank_start", {})
        return result.get("started", False)
    
    async def stop(self) -> bool:
        """Stop Think Tank."""
        result = await self.registry.call("think_tank_stop", {})
        return result.get("stopped", False)
    
    async def run_session(self) -> Dict[str, Any]:
        """Force immediate session."""
        return await self.registry.call("think_tank_run_session", {})
    
    async def get_proposals(self) -> List[Dict]:
        """Get pending proposals."""
        result = await self.registry.call("think_tank_get_proposals", {})
        return result.get("proposals", [])
    
    async def approve_proposal(self, proposal_id: str) -> bool:
        """Approve a proposal."""
        result = await self.registry.call("think_tank_approve_proposal", {
            "proposal_id": proposal_id
        })
        return result.get("approved", False)
    
    async def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        """Reject a proposal."""
        result = await self.registry.call("think_tank_reject_proposal", {
            "proposal_id": proposal_id,
            "reason": reason
        })
        return result.get("rejected", False)
    
    async def get_history(self, limit: int = 10) -> List[Dict]:
        """Get session history."""
        result = await self.registry.call("think_tank_get_history", {"limit": limit})
        return result.get("history", [])
    
    @property
    def is_running(self) -> bool:
        """Check if Think Tank is running."""
        import asyncio
        try:
            status = asyncio.run(self.get_status())
            return status.get("running", False)
        except:
            return False


class PlanetaryDBAdapter:
    """Adapter for Planetary Database."""
    
    def __init__(self, registry):
        self.registry = registry
    
    async def admit(
        self,
        content: str,
        layer: str = "fast",
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Admit content to database."""
        return await self.registry.call("planetary_admit", {
            "content": content,
            "layer": layer,
            "tags": tags or []
        })
    
    async def query(
        self,
        query: str,
        use_quorum: bool = False,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Query database."""
        return await self.registry.call("planetary_query", {
            "query": query,
            "use_quorum": use_quorum,
            "limit": limit
        })
    
    async def store_crystal(
        self,
        content: str,
        name: str = None,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Store as crystal."""
        return await self.registry.call("planetary_store_crystal", {
            "content": content,
            "name": name,
            "tags": tags or []
        })
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return await self.registry.call("planetary_get_stats", {})


class ReceiptAdapter:
    """Adapter for Receipt system."""
    
    def __init__(self, registry):
        self.registry = registry
    
    async def verify_chain(self) -> Dict[str, Any]:
        """Verify receipt chain integrity."""
        return await self.registry.call("receipt_verify_chain", {})
    
    async def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get recent receipts."""
        result = await self.registry.call("receipt_get_recent", {"limit": limit})
        return result.get("receipts", [])


class HealthAdapter:
    """Adapter for Health monitoring."""
    
    def __init__(self, registry):
        self.registry = registry
    
    async def check_all(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        return await self.registry.call("health_check", {})
    
    async def check_component(self, component: str) -> Dict[str, Any]:
        """Check specific component."""
        return await self.registry.call("health_component", {"component": component})
    
    @property
    def is_healthy(self) -> bool:
        """Quick health check."""
        import asyncio
        try:
            result = asyncio.run(self.check_all())
            return result.get("healthy", False)
        except:
            return False


class TMNAdapter:
    """Adapter for Triadic Manifold Network."""
    
    def __init__(self, registry):
        self.registry = registry
    
    async def learn(self, input_code: str, output_code: str) -> Dict[str, Any]:
        """Train TMN on code pair."""
        return await self.registry.call("tmn_learn", {
            "input_code": input_code,
            "output_code": output_code
        })
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return await self.registry.call("tmn_state", {})
    
    async def save(self) -> Dict[str, Any]:
        """Save state."""
        return await self.registry.call("tmn_save", {})
    
    async def load(self) -> bool:
        """Load state."""
        result = await self.registry.call("tmn_load", {})
        return result.get("loaded", False)


class FamilyAdapter:
    """Adapter for family-organized donor build wrappers."""

    def __init__(self, source_root: str | None = None):
        self.manager = FamilyControllerManager(source_root=source_root)

    async def list_families(self) -> List[Dict[str, Any]]:
        return self.manager.list_families()

    async def get_family(self, family_name: str) -> Dict[str, Any]:
        result = self.manager.get_family(family_name)
        if result is None:
            raise KeyError(f"Unknown family: {family_name}")
        return result

    async def wrappers(self, family_name: str) -> List[Dict[str, Any]]:
        return self.manager.build_wrappers_for_family(family_name)


class CMPLXAdapterBundle:
    """
    Bundle of all CMPLX adapters for easy access.
    
    Usage:
        bundle = CMPLXAdapterBundle(registry)
        
        # Use any adapter
        result = await bundle.quorum.deliberate("What is X?")
        status = await bundle.think_tank.get_status()
        stats = await bundle.planetary.get_stats()
    """
    
    def __init__(self, registry):
        self.registry = registry
        self.quorum = QuorumAdapter(registry)
        self.think_tank = ThinkTankAdapter(registry)
        self.planetary = PlanetaryDBAdapter(registry)
        self.receipts = ReceiptAdapter(registry)
        self.health = HealthAdapter(registry)
        self.tmn = TMNAdapter(registry)
        self.families = FamilyAdapter()
