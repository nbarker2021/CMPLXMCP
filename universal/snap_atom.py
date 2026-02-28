"""
SNAP Atom System
================
SNAP = Semantic Network Atomic Protocol

Every atomic action in the system generates a SNAP transaction.
These form an immutable chain of everything that happens.

SNAP atoms are the universal transaction primitive.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional
import uuid


@dataclass
class SNAPAtom:
    """
    A SNAP atom is the smallest unit of semantic meaning.
    
    Atoms have:
    - Identity (who created it)
    - Position (where in geometric space)
    - Charge (how much energy/significance)
    - Content (what it represents)
    - Type (what kind of atom)
    - Morphon seed (which fundamental pattern)
    """
    
    identity: str  # Creator identity
    morphon_seed: int  # 0-9 digital root foundation
    position: list[float]  # Coordinates in lattice space
    charge: float  # Energy/significance (0-1)
    content: str  # Human-readable content
    atom_type: str  # Classification
    
    # Optional fields
    atom_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    provenance: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute content hash for verification."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    @property
    def digital_root(self) -> int:
        """Compute digital root of this atom."""
        # DR of position + charge + seed
        position_dr = sum(int(abs(p) * 1000) for p in self.position) % 9 or 9
        charge_dr = int(self.charge * 9) % 9 or 9
        combined = (position_dr + charge_dr + self.morphon_seed) % 9 or 9
        return combined


@dataclass
class SNAPBond:
    """
    A bond connects two atoms with a strength.
    
    Bonds represent relationships:
    - Spatial (proximity)
    - Temporal (sequence)
    - Semantic (meaning)
    - Causal (dependency)
    """
    
    source_id: str  # Source atom ID
    target_id: str  # Target atom ID
    strength: float  # Bond strength (0-1)
    bond_type: str = "generic"  # Type of relationship
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SNAPTransaction:
    """
    A SNAP transaction records an atomic action.
    
    Every action in the system generates a transaction:
    - Tool calls
    - Data transformations
    - Geometric projections
    - Hypothesis generations
    - Memory formations
    
    Transactions form an immutable DAG (directed acyclic graph).
    """
    
    # Core fields
    tx_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    identity: str = "anonymous"
    
    # Action description
    action_type: str = ""  # What happened
    action_description: str = ""  # Human-readable
    
    # Inputs/Outputs
    input_handles: list[str] = field(default_factory=list)  # Input data handles
    output_handle: str = ""  # Output data handle
    
    # Geometric context
    input_signature: str = ""  # Hash of input geometric form
    output_signature: str = ""  # Hash of output geometric form
    
    # Provenance
    parent_tx_ids: list[str] = field(default_factory=list)  # Parent transactions
    atoms_created: list[str] = field(default_factory=list)  # Atoms created
    bonds_created: list[str] = field(default_factory=list)  # Bonds created
    
    # Verification
    receipt_hash: str = ""  # Cryptographic receipt
    digital_root: int = 0  # Transaction's DR
    
    # Context
    temporal_context: str = "present"  # past, present, future
    layer: int = 0  # Which CMPLX layer
    
    def __post_init__(self):
        if not self.receipt_hash:
            self.receipt_hash = self._compute_receipt()
        if not self.digital_root:
            self.digital_root = self._compute_dr()
    
    def _compute_receipt(self) -> str:
        """Compute cryptographic receipt for this transaction."""
        # Include all immutable fields
        data = {
            "tx_id": self.tx_id,
            "timestamp": self.timestamp,
            "identity": self.identity,
            "action": self.action_type,
            "input_sig": self.input_signature,
            "output_sig": self.output_signature,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _compute_dr(self) -> int:
        """Compute digital root of transaction."""
        # Hash-based DR
        hash_val = int(self.receipt_hash[:8], 16)
        dr = hash_val % 9
        return dr if dr != 0 else 9
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def verify(self) -> bool:
        """Verify transaction integrity."""
        return self.receipt_hash == self._compute_receipt()


class SNAPChain:
    """
    A chain of SNAP transactions.
    
    This is the immutable history of everything that happens.
    Like a blockchain but for geometric operations.
    """
    
    def __init__(self):
        self.transactions: dict[str, SNAPTransaction] = {}
        self.by_identity: dict[str, list[str]] = {}  # identity -> tx_ids
        self.by_signature: dict[str, str] = {}  # signature -> tx_id
    
    def add(self, tx: SNAPTransaction) -> bool:
        """Add transaction to chain."""
        if not tx.verify():
            return False
        
        self.transactions[tx.tx_id] = tx
        
        # Index by identity
        if tx.identity not in self.by_identity:
            self.by_identity[tx.identity] = []
        self.by_identity[tx.identity].append(tx.tx_id)
        
        # Index by output signature
        if tx.output_signature:
            self.by_signature[tx.output_signature] = tx.tx_id
        
        return True
    
    def get(self, tx_id: str) -> Optional[SNAPTransaction]:
        """Get transaction by ID."""
        return self.transactions.get(tx_id)
    
    def get_by_identity(self, identity: str) -> list[SNAPTransaction]:
        """Get all transactions by identity."""
        tx_ids = self.by_identity.get(identity, [])
        return [self.transactions[tx_id] for tx_id in tx_ids if tx_id in self.transactions]
    
    def get_by_signature(self, signature: str) -> Optional[SNAPTransaction]:
        """Get transaction by output signature."""
        tx_id = self.by_signature.get(signature)
        if tx_id:
            return self.transactions.get(tx_id)
        return None
    
    def lineage(self, tx_id: str) -> list[SNAPTransaction]:
        """
        Get lineage (ancestors) of a transaction.
        Follows parent_tx_ids recursively.
        """
        result = []
        visited = set()
        
        def traverse(tid: str):
            if tid in visited or tid not in self.transactions:
                return
            visited.add(tid)
            tx = self.transactions[tid]
            result.append(tx)
            for parent in tx.parent_tx_ids:
                traverse(parent)
        
        traverse(tx_id)
        return list(reversed(result))  # Oldest first
    
    def stats(self) -> dict:
        """Get chain statistics."""
        return {
            "total_transactions": len(self.transactions),
            "unique_identities": len(self.by_identity),
            "unique_signatures": len(self.by_signature),
        }


class SNAPLedger:
    """
    The global SNAP ledger.
    
    Singleton that records all atomic actions across the entire system.
    This is the "source of truth" for what happened.
    """
    
    _instance: Optional['SNAPLedger'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._chain = SNAPChain()
            cls._instance._pending: list[SNAPTransaction] = []
        return cls._instance
    
    @property
    def chain(self) -> SNAPChain:
        return self._chain
    
    def record(self, action_type: str, identity: str = "anonymous",
               input_handles: list[str] = None,
               output_handle: str = "",
               input_signature: str = "",
               output_signature: str = "",
               parent_txs: list[str] = None,
               atoms: list[str] = None,
               bonds: list[str] = None,
               temporal: str = "present",
               layer: int = 0,
               description: str = "") -> SNAPTransaction:
        """
        Record an atomic action.
        
        This is THE function that should be called for EVERY atomic action.
        """
        tx = SNAPTransaction(
            identity=identity,
            action_type=action_type,
            action_description=description,
            input_handles=input_handles or [],
            output_handle=output_handle,
            input_signature=input_signature,
            output_signature=output_signature,
            parent_tx_ids=parent_txs or [],
            atoms_created=atoms or [],
            bonds_created=bonds or [],
            temporal_context=temporal,
            layer=layer
        )
        
        self._chain.add(tx)
        return tx
    
    def get_stats(self) -> dict:
        """Get ledger statistics."""
        return self._chain.stats()
