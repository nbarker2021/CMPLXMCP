"""
Universal System MCP Tools
==========================
Tools for the Universal Translator, Crystal Storage, and Identity Family.

These tools enable:
- Converting anything to geometric form
- Storing as crystals with full provenance
- Querying by resonance (not just exact match)
- Temporal queries (past/present/future)
- Hypothesis generation
"""

import logging
from pathlib import Path
from typing import Any

from .tools import ToolRegistry, _generate_handle, _resolve_handle

logger = logging.getLogger("cmplx.mcp.universal")


class UniversalTools(ToolRegistry):
    """
    Universal System tools for MCP.
    
    These connect to the Universal Translator, Crystal Lattice,
    Temporal Layer, and Identity Family systems.
    """
    
    def __init__(self):
        # These would be initialized with actual instances
        self._translator = None
        self._lattice = None
        self._temporal = None
        self._identity_family = None
    
    def _init_systems(self, data_root: Path):
        """Initialize universal systems (lazy loading)."""
        if self._translator is None:
            from ..universal import UniversalTranslator, CrystalLattice, TemporalLayer, IdentityFamily
            
            self._translator = UniversalTranslator()
            self._lattice = CrystalLattice(data_root / "crystal_lattice.db")
            self._temporal = TemporalLayer(self._lattice, data_root / "temporal_layer.db")
            self._identity_family = IdentityFamily(self._lattice, self._temporal, 
                                                   data_root / "identity_family.db")
            
            logger.info("Universal systems initialized")
    
    async def handle(self, name: str, arguments: dict, data_root: Path) -> dict:
        """Route to appropriate handler."""
        self._init_systems(data_root)
        return await super().handle(name, arguments, data_root)
    
    # ===== Universal Translation =====
    
    async def _universal_translate(self, args: dict, data_root: Path) -> dict:
        """Translate any content to geometric form."""
        content = args.get("content", "")
        content_type = args.get("content_type")  # Auto-detect if None
        identity = args.get("identity", "anonymous")
        
        form = await self._translator.translate(
            content, 
            content_type=content_type,
            identity=identity
        )
        
        # Store form server-side
        handle = _generate_handle("form", form.to_dict())
        
        return {
            "handle": handle,
            "content_type": form.envelope.get("content_type", "unknown"),
            "atom_count": len(form.atoms),
            "bond_count": len(form.bonds),
            "symmetry_signature": form.symmetry_signature,
            "lightweight": True
        }
    
    # ===== Crystal Operations =====
    
    async def _crystal_store(self, args: dict, data_root: Path) -> dict:
        """Store a geometric form as a crystal."""
        form_handle = args.get("form_handle")
        name = args.get("name", "")
        identity = args.get("identity", "anonymous")
        temporal_phase = args.get("temporal_phase", "present")
        tags = args.get("tags", [])
        
        # Resolve form
        form_data = _resolve_handle(form_handle)
        
        # Reconstruct form
        from ..universal.translator import GeometricForm
        from ..universal.snap_atom import SNAPAtom
        
        form = GeometricForm(
            atoms=[SNAPAtom(**a) for a in form_data.get("atoms", [])],
            bonds=form_data.get("bonds", []),
            envelope=form_data.get("envelope", {}),
            symmetry_signature=form_data.get("symmetry_signature", "")
        )
        
        # Execute atomic action
        result = await self._identity_family.atomic_action(
            identity_id=identity,
            action_type="crystal_store",
            geometric_form=form,
            description=f"Stored crystal: {name}",
            temporal_phase=temporal_phase
        )
        
        # Add tags
        if tags:
            crystal = self._lattice.retrieve(result["crystal_id"])
            if crystal:
                crystal.tags.extend(tags)
                self._lattice.store(crystal)
        
        return {
            "crystal_id": result["crystal_id"],
            "tx_id": result["tx_id"],
            "receipt_id": result["receipt_id"],
            "resonance_signature": result["resonance_signature"],
            "verified": result["verified"],
            "temporal_phase": temporal_phase
        }
    
    async def _crystal_retrieve(self, args: dict, data_root: Path) -> dict:
        """Retrieve a crystal by handle."""
        crystal_id = args.get("crystal_id")
        
        crystal = self._lattice.retrieve(crystal_id)
        if not crystal:
            return {"error": "Crystal not found"}
        
        # Return lightweight metadata only
        return {
            "crystal_id": crystal.crystal_id,
            "name": crystal.name,
            "atom_count": len(crystal.atoms),
            "bond_count": len(crystal.bonds),
            "resonance_signature": crystal.resonance_signature,
            "temporal_phase": crystal.temporal_phase,
            "tags": crystal.tags,
            "created_at": crystal.created_at,
            "access_count": crystal.access_count
        }
    
    async def _crystal_resonance_query(self, args: dict, data_root: Path) -> dict:
        """Find crystals by resonance (similarity)."""
        crystal_id = args.get("crystal_id")
        threshold = args.get("threshold", 0.7)
        limit = args.get("limit", 10)
        
        query_crystal = self._lattice.retrieve(crystal_id)
        if not query_crystal:
            return {"error": "Query crystal not found"}
        
        results = self._lattice.find_by_resonance(query_crystal, threshold, limit)
        
        return {
            "query_crystal": crystal_id,
            "threshold": threshold,
            "results": [
                {
                    "crystal_id": c.crystal_id,
                    "name": c.name,
                    "resonance": float(score),
                    "temporal_phase": c.temporal_phase
                }
                for c, score in results
            ]
        }
    
    async def _crystal_merge(self, args: dict, data_root: Path) -> dict:
        """Merge multiple crystals into one."""
        crystal_ids = args.get("crystal_ids", [])
        name = args.get("name", "")
        
        crystals = []
        for cid in crystal_ids:
            c = self._lattice.retrieve(cid)
            if c:
                crystals.append(c)
        
        if len(crystals) < 2:
            return {"error": "Need at least 2 crystals to merge"}
        
        from ..universal.crystal import CrystalFactory
        
        merged = CrystalFactory.merge(crystals, name)
        merged_id = self._lattice.store(merged)
        
        return {
            "merged_crystal_id": merged_id,
            "source_crystals": crystal_ids,
            "atom_count": len(merged.atoms),
            "name": merged.name
        }
    
    # ===== Temporal Operations =====
    
    async def _temporal_query(self, args: dict, data_root: Path) -> dict:
        """Query crystals by temporal phase."""
        phase = args.get("phase", "present")
        limit = args.get("limit", 100)
        
        crystals = list(self._lattice.find_by_temporal_phase(phase))
        
        return {
            "phase": phase,
            "total": len(crystals),
            "crystals": [
                {
                    "crystal_id": c.crystal_id,
                    "name": c.name,
                    "resonance_signature": c.resonance_signature[:16] + "..."
                }
                for c in crystals[:limit]
            ]
        }
    
    async def _temporal_remember(self, args: dict, data_root: Path) -> dict:
        """Convert a crystal to a memory (past phase)."""
        crystal_id = args.get("crystal_id")
        description = args.get("description", "")
        reliability = args.get("reliability", 1.0)
        
        crystal = self._lattice.retrieve(crystal_id)
        if not crystal:
            return {"error": "Crystal not found"}
        
        memory = self._temporal.remember(crystal, description, reliability)
        
        return {
            "memory_id": memory.memory_id,
            "crystal_id": crystal_id,
            "reliability": memory.current_reliability,
            "temporal_phase": "past"
        }
    
    async def _hypothesis_generate(self, args: dict, data_root: Path) -> dict:
        """Generate hypotheses from a context crystal."""
        crystal_id = args.get("crystal_id")
        num_hypotheses = args.get("num_hypotheses", 3)
        description = args.get("description", "")
        
        context = self._lattice.retrieve(crystal_id)
        if not context:
            return {"error": "Context crystal not found"}
        
        hypotheses = self._temporal.hypothesize(context, description, num_hypotheses)
        
        return {
            "context_crystal": crystal_id,
            "hypotheses": [
                {
                    "hypothesis_id": h.hypothesis_id,
                    "description": h.description,
                    "probability": h.posterior_probability,
                    "status": h.status,
                    "temporal_phase": "future"
                }
                for h in hypotheses
            ]
        }
    
    async def _hypothesis_validate(self, args: dict, data_root: Path) -> dict:
        """Validate a hypothesis against actual outcome."""
        hypothesis_id = args.get("hypothesis_id")
        actual_crystal_id = args.get("actual_crystal_id")
        
        actual = self._lattice.retrieve(actual_crystal_id)
        if not actual:
            return {"error": "Actual outcome crystal not found"}
        
        confirmed = self._temporal.hypothesis_engine.validate(hypothesis_id, actual)
        
        return {
            "hypothesis_id": hypothesis_id,
            "confirmed": confirmed,
            "actual_crystal": actual_crystal_id
        }
    
    async def _temporal_counterfactual(self, args: dict, data_root: Path) -> dict:
        """Generate counterfactual scenario."""
        crystal_id = args.get("crystal_id")
        changes = args.get("changes", {})
        
        actual = self._lattice.retrieve(crystal_id)
        if not actual:
            return {"error": "Crystal not found"}
        
        cf = self._temporal.counterfactual(actual, changes)
        
        return {
            "counterfactual_id": cf.crystal_id,
            "based_on": crystal_id,
            "changes": changes,
            "temporal_phase": "future"
        }
    
    # ===== Identity Operations =====
    
    async def _identity_register(self, args: dict, data_root: Path) -> dict:
        """Register a new identity."""
        name = args.get("name")
        identity_id = args.get("identity_id")  # Optional
        
        identity = self._identity_family.register_identity(name, identity_id)
        
        return {
            "identity_id": identity.identity_id,
            "name": identity.name,
            "public_key": identity.public_key,
            "created_at": identity.created_at
        }
    
    async def _identity_history(self, args: dict, data_root: Path) -> dict:
        """Get identity's complete history."""
        identity_id = args.get("identity_id")
        
        history = self._identity_family.get_identity_history(identity_id)
        
        return history
    
    async def _audit_provenance(self, args: dict, data_root: Path) -> dict:
        """Audit full provenance of a crystal."""
        crystal_id = args.get("crystal_id")
        
        audit = self._identity_family.audit(crystal_id)
        
        return audit
    
    async def _verify_receipt(self, args: dict, data_root: Path) -> dict:
        """Verify a receipt's authenticity."""
        receipt_id = args.get("receipt_id")
        
        verified = self._identity_family.verify_receipt(receipt_id)
        
        return {
            "receipt_id": receipt_id,
            "verified": verified
        }
    
    # ===== System Stats =====
    
    async def _universal_stats(self, args: dict, data_root: Path) -> dict:
        """Get universal system statistics."""
        return {
            "identity_family": self._identity_family.stats(),
            "crystal_lattice": self._lattice.stats(),
            "temporal_layer": self._temporal.get_timeline()
        }


# Global instance
UNIVERSAL_TOOLS = UniversalTools()
