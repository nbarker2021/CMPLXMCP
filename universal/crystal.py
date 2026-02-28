"""
Crystal Storage System
======================
A non-discrete database that stores geometric forms as "crystals".

Unlike traditional databases (rows in tables), crystals are:
- Continuous geometric structures
- Self-similar at different scales
- Connected by resonance/symmetry
- Queryable by proximity, not just equality

You can USE them discretely (query by handle), but the underlying
structure is continuous and holographic.
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Iterator
import numpy as np

from .snap_atom import SNAPAtom, SNAPBond


@dataclass
class Crystal:
    """
    A crystal is a stored geometric form.
    
    Crystals have:
    - A unique identity (handle)
    - A geometric form (atoms + bonds)
    - Temporal context (when it exists)
    - Resonance signature (how it vibrates)
    - Parent/child relationships (lineage)
    """
    
    # Identity
    crystal_id: str  # Unique handle
    name: str = ""  # Human-readable name
    
    # Geometric content
    atoms: list[SNAPAtom] = field(default_factory=list)
    bonds: list[SNAPBond] = field(default_factory=list)
    
    # Symmetry
    resonance_signature: str = ""  # Computed from structure
    symmetry_group: str = ""  # Classification
    
    # Temporal
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    temporal_phase: str = "present"  # past, present, future
    
    # Provenance
    parent_crystal_ids: list[str] = field(default_factory=list)
    child_crystal_ids: list[str] = field(default_factory=list)
    snap_tx_id: str = ""  # Link to SNAP transaction
    
    # Metadata
    envelope: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    # State
    access_count: int = 0
    last_accessed: Optional[str] = None
    
    def __post_init__(self):
        if not self.resonance_signature:
            self.resonance_signature = self._compute_resonance()
    
    def _compute_resonance(self) -> str:
        """Compute resonance signature from geometric structure."""
        # Hash of atom positions and bond topology
        atom_positions = sorted([(a.identity, tuple(a.position)) for a in self.atoms])
        bond_data = sorted([(b.source_id, b.target_id, b.strength) for b in self.bonds])
        
        content = json.dumps({
            "atoms": atom_positions,
            "bonds": bond_data
        }, sort_keys=True)
        
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def vibrate(self, frequency: float) -> list[float]:
        """
        Compute crystal's response to a frequency.
        
        This simulates how the crystal would "ring" if excited.
        Used for resonance-based querying.
        """
        # Simple: sum of atom responses
        response = []
        for atom in self.atoms:
            # Each atom responds based on its charge and position
            phase = sum(p * frequency for p in atom.position[:3])  # Use first 3 dims
            amplitude = atom.charge * np.sin(phase)
            response.append(amplitude)
        
        return response
    
    def resonance_with(self, other: 'Crystal') -> float:
        """
        Compute resonance between two crystals.
        
        Returns 0-1 similarity score.
        """
        if not self.atoms or not other.atoms:
            return 0.0
        
        # Compare vibration patterns at multiple frequencies
        frequencies = [0.1, 0.5, 1.0, 2.0, 5.0]
        similarities = []
        
        for freq in frequencies:
            v1 = np.array(self.vibrate(freq))
            v2 = np.array(other.vibrate(freq))
            
            if len(v1) > 0 and len(v2) > 0:
                # Cosine similarity
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm > 0:
                    similarities.append(dot / norm)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "crystal_id": self.crystal_id,
            "name": self.name,
            "atoms": [a.to_dict() for a in self.atoms],
            "bonds": [b.to_dict() for b in self.bonds],
            "resonance_signature": self.resonance_signature,
            "symmetry_group": self.symmetry_group,
            "created_at": self.created_at,
            "temporal_phase": self.temporal_phase,
            "parent_crystal_ids": self.parent_crystal_ids,
            "snap_tx_id": self.snap_tx_id,
            "envelope": self.envelope,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Crystal':
        """Deserialize from dictionary."""
        crystal = cls(
            crystal_id=data["crystal_id"],
            name=data.get("name", ""),
            resonance_signature=data.get("resonance_signature", ""),
            symmetry_group=data.get("symmetry_group", ""),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            temporal_phase=data.get("temporal_phase", "present"),
            parent_crystal_ids=data.get("parent_crystal_ids", []),
            snap_tx_id=data.get("snap_tx_id", ""),
            envelope=data.get("envelope", {}),
            tags=data.get("tags", []),
        )
        
        # Deserialize atoms and bonds
        crystal.atoms = [SNAPAtom(**a) for a in data.get("atoms", [])]
        crystal.bonds = [SNAPBond(**b) for b in data.get("bonds", [])]
        
        return crystal


class CrystalLattice:
    """
    A lattice of crystals - the "database".
    
    Crystals are stored in a continuous geometric space.
    They can be queried by:
    - Exact handle (discrete query)
    - Resonance similarity (continuous query)
    - Temporal phase (past/present/future)
    - Symmetry group (classification)
    - Tags (categorical)
    
    This is NOT a relational database. It's a holographic storage system.
    """
    
    def __init__(self, db_path: Path | str = "crystal_lattice.db"):
        self.db_path = Path(db_path)
        self._init_db()
        self._memory_cache: dict[str, Crystal] = {}
    
    def _init_db(self):
        """Initialize crystal storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS crystals (
                    crystal_id TEXT PRIMARY KEY,
                    resonance_signature TEXT NOT NULL,
                    symmetry_group TEXT,
                    temporal_phase TEXT,
                    created_at TEXT,
                    snap_tx_id TEXT,
                    json_data TEXT,
                    tags TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_resonance ON crystals(resonance_signature);
                CREATE INDEX IF NOT EXISTS idx_symmetry ON crystals(symmetry_group);
                CREATE INDEX IF NOT EXISTS idx_temporal ON crystals(temporal_phase);
                CREATE INDEX IF NOT EXISTS idx_tags ON crystals(tags);
                
                CREATE TABLE IF NOT EXISTS crystal_lineage (
                    parent_id TEXT,
                    child_id TEXT,
                    relationship_type TEXT,
                    PRIMARY KEY (parent_id, child_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_parent ON crystal_lineage(parent_id);
                CREATE INDEX IF NOT EXISTS idx_child ON crystal_lineage(child_id);
            """)
            conn.commit()
    
    def store(self, crystal: Crystal) -> str:
        """
        Store a crystal in the lattice.
        
        Returns the crystal's handle.
        """
        # Serialize
        json_data = json.dumps(crystal.to_dict())
        
        # Store
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO crystals 
                   (crystal_id, resonance_signature, symmetry_group, temporal_phase,
                    created_at, snap_tx_id, json_data, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    crystal.crystal_id,
                    crystal.resonance_signature,
                    crystal.symmetry_group,
                    crystal.temporal_phase,
                    crystal.created_at,
                    crystal.snap_tx_id,
                    json_data,
                    json.dumps(crystal.tags)
                )
            )
            
            # Store lineage
            for parent_id in crystal.parent_crystal_ids:
                conn.execute(
                    """INSERT OR REPLACE INTO crystal_lineage 
                       (parent_id, child_id, relationship_type)
                       VALUES (?, ?, 'parent_of')""",
                    (parent_id, crystal.crystal_id)
                )
            
            conn.commit()
        
        # Cache in memory
        self._memory_cache[crystal.crystal_id] = crystal
        
        return crystal.crystal_id
    
    def retrieve(self, crystal_id: str) -> Optional[Crystal]:
        """
        Retrieve a crystal by exact handle.
        
        This is the DISCRETE query method.
        """
        # Check cache first
        if crystal_id in self._memory_cache:
            crystal = self._memory_cache[crystal_id]
            crystal.access_count += 1
            crystal.last_accessed = datetime.utcnow().isoformat()
            return crystal
        
        # Load from DB
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT json_data FROM crystals WHERE crystal_id = ?",
                (crystal_id,)
            ).fetchone()
            
            if row:
                data = json.loads(row[0])
                crystal = Crystal.from_dict(data)
                crystal.access_count += 1
                crystal.last_accessed = datetime.utcnow().isoformat()
                self._memory_cache[crystal_id] = crystal
                return crystal
        
        return None
    
    def find_by_resonance(self, query_crystal: Crystal, threshold: float = 0.7, 
                         limit: int = 10) -> list[tuple[Crystal, float]]:
        """
        Find crystals that resonate with query.
        
        This is the CONTINUOUS query method.
        Returns crystals sorted by resonance strength.
        """
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Load all crystals (in production, use vector index)
            rows = conn.execute(
                "SELECT json_data FROM crystals"
            ).fetchall()
            
            for row in rows:
                data = json.loads(row[0])
                crystal = Crystal.from_dict(data)
                
                # Skip self
                if crystal.crystal_id == query_crystal.crystal_id:
                    continue
                
                # Compute resonance
                resonance = query_crystal.resonance_with(crystal)
                
                if resonance >= threshold:
                    results.append((crystal, resonance))
        
        # Sort by resonance (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def find_by_signature(self, signature: str) -> Optional[Crystal]:
        """Find crystal by exact resonance signature."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT json_data FROM crystals WHERE resonance_signature = ?",
                (signature,)
            ).fetchone()
            
            if row:
                return Crystal.from_dict(json.loads(row[0]))
        return None
    
    def find_by_temporal_phase(self, phase: str) -> Iterator[Crystal]:
        """Find all crystals in a temporal phase (past/present/future)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT json_data FROM crystals WHERE temporal_phase = ?",
                (phase,)
            )
            
            for row in rows:
                yield Crystal.from_dict(json.loads(row[0]))
    
    def find_by_tags(self, tags: list[str], match_all: bool = False) -> Iterator[Crystal]:
        """Find crystals by tags."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT json_data, tags FROM crystals")
            
            for row in rows:
                crystal_tags = json.loads(row[1])
                
                if match_all:
                    if all(tag in crystal_tags for tag in tags):
                        yield Crystal.from_dict(json.loads(row[0]))
                else:
                    if any(tag in crystal_tags for tag in tags):
                        yield Crystal.from_dict(json.loads(row[0]))
    
    def get_lineage(self, crystal_id: str) -> dict:
        """Get complete lineage (ancestors and descendants)."""
        ancestors = []
        descendants = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Ancestors
            current = crystal_id
            while True:
                row = conn.execute(
                    """SELECT parent_id FROM crystal_lineage 
                       WHERE child_id = ? AND relationship_type = 'parent_of'""",
                    (current,)
                ).fetchone()
                
                if not row:
                    break
                
                ancestors.append(row[0])
                current = row[0]
            
            # Descendants
            rows = conn.execute(
                """SELECT child_id FROM crystal_lineage
                   WHERE parent_id = ? AND relationship_type = 'parent_of'""",
                (crystal_id,)
            ).fetchall()
            
            descendants = [r[0] for r in rows]
        
        return {
            "crystal_id": crystal_id,
            "ancestors": list(reversed(ancestors)),
            "descendants": descendants
        }
    
    def stats(self) -> dict:
        """Get lattice statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM crystals").fetchone()[0]
            
            phases = conn.execute(
                "SELECT temporal_phase, COUNT(*) FROM crystals GROUP BY temporal_phase"
            ).fetchall()
            
            groups = conn.execute(
                "SELECT symmetry_group, COUNT(*) FROM crystals GROUP BY symmetry_group"
            ).fetchall()
            
            return {
                "total_crystals": total,
                "by_phase": {p: c for p, c in phases},
                "by_symmetry": {g: c for g, c in groups},
                "cached_in_memory": len(self._memory_cache),
            }


class CrystalFactory:
    """
    Factory for creating crystals from various sources.
    """
    
    @staticmethod
    def from_geometric_form(form, crystal_id: Optional[str] = None,
                           name: str = "", temporal_phase: str = "present",
                           tags: list[str] = None) -> Crystal:
        """Create crystal from geometric form."""
        from .translator import GeometricForm
        
        crystal = Crystal(
            crystal_id=crystal_id or f"cryst_{form.symmetry_signature[:16]}",
            name=name,
            temporal_phase=temporal_phase,
            tags=tags or [],
        )
        
        # Copy atoms and bonds
        crystal.atoms = form.atoms
        crystal.bonds = [
            SNAPBond(
                source_id=form.atoms[b[0]].atom_id if b[0] < len(form.atoms) else "",
                target_id=form.atoms[b[1]].atom_id if b[1] < len(form.atoms) else "",
                strength=b[2]
            )
            for b in form.bonds
        ]
        
        # Copy envelope
        crystal.envelope = form.envelope
        
        return crystal
    
    @staticmethod
    def merge(crystals: list[Crystal], name: str = "") -> Crystal:
        """
        Merge multiple crystals into one.
        
        Creates a super-crystal containing all atoms and bonds.
        """
        merged = Crystal(
            crystal_id=f"merged_{datetime.utcnow().timestamp()}",
            name=name or f"Merge of {len(crystals)} crystals",
        )
        
        # Collect all atoms and bonds
        all_atoms = []
        all_bonds = []
        atom_id_map = {}  # Old ID -> new index
        
        for crystal in crystals:
            for atom in crystal.atoms:
                new_idx = len(all_atoms)
                atom_id_map[atom.atom_id] = new_idx
                all_atoms.append(atom)
        
        # Remap bonds
        for crystal in crystals:
            for bond in crystal.bonds:
                if bond.source_id in atom_id_map and bond.target_id in atom_id_map:
                    all_bonds.append(SNAPBond(
                        source_id=all_atoms[atom_id_map[bond.source_id]].atom_id,
                        target_id=all_atoms[atom_id_map[bond.target_id]].atom_id,
                        strength=bond.strength,
                        bond_type=bond.bond_type
                    ))
        
        merged.atoms = all_atoms
        merged.bonds = all_bonds
        merged.parent_crystal_ids = [c.crystal_id for c in crystals]
        
        return merged
