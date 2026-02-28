"""
Planet Abstraction
==================
A self-contained unit combining MDHG, CA, and AGRM routing.

Planets are the primary nodes in the CMPLX geometric network.
Each planet has:
- MDHG cache (fast/med/slow scales)
- CA field (self-regulating dynamics)
- AGRM router interface
- Receipt ledger
"""

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .mdhg_ca import MDHGMultiScale, CAFieldMultiScale, WolframAssignment
from .agrm_router import AGRMRouter, AGRMNode


@dataclass
class PlanetConfig:
    """Configuration for a planet."""
    name: str
    grid_side: int = 12
    cap_per_slot: int = 6
    bins: int = 16
    seed: int = 42
    
    # AGRM position (24D)
    position: List[float] = field(default_factory=lambda: [0.0] * 24)
    
    # CA defaults
    default_wolfram_class: str = "II"  # Oscillating
    default_kernel: str = "oscillate"


@dataclass
class Receipt:
    """Receipt for atomic actions on a planet."""
    receipt_id: str
    timestamp: float
    action: str
    slot: str
    layer: str
    crystal_id: Optional[str]
    metadata: Dict[str, Any]
    signature: str


class Planet:
    """
    A self-contained geometric processing unit.
    
    Combines:
    - MDHGMultiScale: Geometric caching at three timescales
    - CAFieldMultiScale: Self-regulating dynamics
    - Ledger: Immutable receipt log
    """
    
    def __init__(self, config: PlanetConfig, router: Optional[AGRMRouter] = None):
        self.config = config
        self.name = config.name
        self.planet_id = f"planet_{hashlib.sha256(config.name.encode()).hexdigest()[:12]}"
        
        # Initialize MDHG + CA
        self.mdhg = MDHGMultiScale(
            grid_side=config.grid_side,
            cap_per_slot=config.cap_per_slot,
            bins=config.bins
        )
        
        self.ca = CAFieldMultiScale(
            w=config.grid_side,
            h=config.grid_side,
            seed=config.seed
        )
        
        # Receipt ledger
        self._ledger: List[Receipt] = []
        self._crystal_id_to_slot: Dict[str, Tuple[str, str]] = {}  # crystal_id -> (layer, slot)
        
        # Router connection
        self._router = router
        if router:
            self._register_with_router()
    
    def _register_with_router(self):
        """Register this planet with the AGRM router."""
        if self._router:
            # Compute resonance signature from position
            pos_str = json.dumps(self.config.position, sort_keys=True)
            resonance = hashlib.sha256(pos_str.encode()).hexdigest()[:32]
            
            self._router.register_node(
                node_id=self.planet_id,
                position=self.config.position,
                resonance=resonance,
                metadata={
                    "name": self.name,
                    "grid_side": self.config.grid_side,
                    "planet_type": "standard"
                }
            )
    
    def admit_crystal(self, v24: List[float], crystal_id: str,
                     meta: Dict[str, Any],
                     layer: str = "fast",
                     generate_receipt: bool = True) -> Dict[str, Any]:
        """
        Admit a crystal to this planet's MDHG cache.
        
        This is the primary entry point for storing geometric data.
        """
        # Enrich metadata
        full_meta = {
            **meta,
            "crystal_id": crystal_id,
            "planet_id": self.planet_id,
            "admitted_at": time.time()
        }
        
        # Admit to MDHG
        result = self.mdhg.admit(v24, full_meta, layer)
        
        # Track crystal location
        slot = result.get("slot")
        if slot:
            self._crystal_id_to_slot[crystal_id] = (layer, slot)
        
        # Update CA field
        self.ca.apply_mdhg_admission(layer, slot, full_meta)
        
        # Generate receipt
        if generate_receipt:
            receipt = self._create_receipt("admit", result, crystal_id)
            result["receipt_id"] = receipt.receipt_id
        
        return result
    
    def admit_crystal_all_layers(self, v24: List[float], crystal_id: str,
                                 meta: Dict[str, Any]) -> Dict[str, Any]:
        """Admit crystal to all three MDHG layers."""
        full_meta = {
            **meta,
            "crystal_id": crystal_id,
            "planet_id": self.planet_id,
            "admitted_at": time.time()
        }
        
        # Admit to all layers
        results = self.mdhg.admit_all_layers(v24, full_meta)
        
        # Update CA for each layer
        for layer, result in results.items():
            slot = result.get("slot")
            if slot:
                self._crystal_id_to_slot[f"{crystal_id}:{layer}"] = (layer, slot)
                self.ca.apply_mdhg_admission(layer, slot, full_meta)
        
        # Generate master receipt
        receipt = self._create_receipt("admit_all", {
            "slots": {k: v.get("slot") for k, v in results.items()}
        }, crystal_id)
        
        return {
            "results": results,
            "receipt_id": receipt.receipt_id
        }
    
    def query_resonance(self, v24: List[float], threshold: float = 0.7,
                       max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query for similar crystals by resonance.
        
        Looks in MDHG slots near the query vector.
        """
        from .mdhg_ca import quantize, slot_of
        
        q = quantize(v24, bins=self.mdhg.fast.bins)
        query_slot = slot_of(q, grid_side=self.mdhg.fast.grid_side)
        
        # Get nearby slots (include neighbors)
        try:
            x, y = query_slot.split(",")
            x, y = int(x), int(y)
            
            nearby_slots = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx = (x + dx) % self.mdhg.fast.grid_side
                    ny = (y + dy) % self.mdhg.fast.grid_side
                    nearby_slots.append(f"{nx:02d},{ny:02d}")
        except ValueError:
            nearby_slots = [query_slot]
        
        # Collect candidates from nearby slots
        candidates = []
        for layer in ["fast", "med", "slow"]:
            cache = getattr(self.mdhg, layer)
            for slot in nearby_slots:
                entries = cache.get_slot_contents(slot)
                for entry in entries:
                    # Compute distance
                    dist = sum(1 for a, b in zip(q, entry.q24) if a != b)
                    max_dist = len(q)
                    similarity = 1.0 - (dist / max_dist)
                    
                    if similarity >= threshold:
                        candidates.append({
                            "layer": layer,
                            "slot": slot,
                            "key": entry.key,
                            "similarity": similarity,
                            "meta": entry.meta,
                            "hits": entry.hits
                        })
        
        # Sort by similarity
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:max_results]
    
    def step_dynamics(self) -> List[Dict[str, Any]]:
        """
        Step CA field dynamics.
        
        Call periodically to allow self-regulation.
        """
        diagnostics = self.ca.step()
        
        # Log significant diagnostics
        for diag in diagnostics:
            if diag.get("kind") == "risk_hotspots":
                self._create_receipt("ca_diag", diag, None)
        
        return diagnostics
    
    def get_cell_state(self, layer: str, x: int, y: int) -> Dict[str, Any]:
        """Get CA cell state at specific location."""
        field = getattr(self.ca, layer)
        return field.get_cell_state(x, y)
    
    def get_planet_state(self) -> Dict[str, Any]:
        """Get complete planet state."""
        mdhg_stats = self.mdhg.get_stats()
        ca_grids = self.ca.scalar_grids()
        
        # Compute overall "health" from CA channels
        health = self._compute_health()
        
        return {
            "planet_id": self.planet_id,
            "name": self.name,
            "mdhg": mdhg_stats,
            "ca_grids": ca_grids,
            "health": health,
            "ledger_size": len(self._ledger),
            "crystals_tracked": len(self._crystal_id_to_slot)
        }
    
    def _compute_health(self) -> Dict[str, float]:
        """Compute planet health metrics from CA state."""
        # Aggregate CA channels across all layers
        total_pressure = 0
        total_risk = 0
        total_trust = 0
        cell_count = 0
        
        for layer in ["fast", "med", "slow"]:
            field = getattr(self.ca, layer)
            for row in field.grid:
                for cell in row:
                    total_pressure += cell.get("pressure")
                    total_risk += cell.get("risk")
                    total_trust += cell.get("trust")
                    cell_count += 1
        
        if cell_count == 0:
            return {"pressure": 0.5, "risk": 0.5, "trust": 0.5}
        
        return {
            "pressure": min(1.0, total_pressure / (cell_count * 15)),
            "risk": min(1.0, total_risk / (cell_count * 15)),
            "trust": min(1.0, total_trust / (cell_count * 15))
        }
    
    def _create_receipt(self, action: str, result: Dict[str, Any],
                       crystal_id: Optional[str]) -> Receipt:
        """Create and store a receipt."""
        receipt_data = {
            "planet": self.planet_id,
            "action": action,
            "result": result,
            "crystal_id": crystal_id,
            "timestamp": time.time()
        }
        
        sig = hashlib.sha256(json.dumps(receipt_data, sort_keys=True).encode()).hexdigest()[:16]
        
        receipt = Receipt(
            receipt_id=f"rcpt_{sig}",
            timestamp=receipt_data["timestamp"],
            action=action,
            slot=result.get("slot", "unknown"),
            layer=result.get("layer", "unknown"),
            crystal_id=crystal_id,
            metadata=result,
            signature=sig
        )
        
        self._ledger.append(receipt)
        return receipt
    
    def get_ledger(self, since: Optional[float] = None) -> List[Receipt]:
        """Get receipt ledger, optionally filtered by time."""
        if since is None:
            return self._ledger.copy()
        return [r for r in self._ledger if r.timestamp >= since]
    
    def find_crystal(self, crystal_id: str) -> Optional[Tuple[str, str]]:
        """Find which layer/slot a crystal is stored in."""
        # Check all layers
        for layer in ["fast", "med", "slow"]:
            key = f"{crystal_id}:{layer}"
            if key in self._crystal_id_to_slot:
                return self._crystal_id_to_slot[key]
        
        # Check fast layer only (default)
        return self._crystal_id_to_slot.get(crystal_id)
    
    def snapshot(self) -> Dict[str, Any]:
        """Full snapshot for persistence."""
        return {
            "config": {
                "name": self.config.name,
                "grid_side": self.config.grid_side,
                "cap_per_slot": self.config.cap_per_slot,
                "bins": self.config.bins,
                "seed": self.config.seed,
                "position": self.config.position
            },
            "mdhg": self.mdhg.snapshot(),
            "ledger": [
                {
                    "receipt_id": r.receipt_id,
                    "timestamp": r.timestamp,
                    "action": r.action,
                    "slot": r.slot,
                    "layer": r.layer,
                    "crystal_id": r.crystal_id,
                    "signature": r.signature
                }
                for r in self._ledger
            ]
        }
