"""
Temporal Layer
==============
Handles time as a geometric dimension, not a linear sequence.

Past, present, and future coexist as different temporal phases.
Hypotheses (future) and memories (past) are both storable/retrievable.

This enables:
- Time-travel queries (what was true at time T?)
- Hypothesis generation (what could be true?)
- Causal inference (what caused what?)
- Counterfactual reasoning (what if?)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Iterator
from pathlib import Path
import sqlite3

from .crystal import Crystal


@dataclass
class TemporalCoordinate:
    """
    A coordinate in temporal space.
    
    Not just a timestamp - includes:
    - Phase (past/present/future)
    - Certainty (1.0 = definitely happened, 0.0 = pure speculation)
    - Branch (which timeline/reality branch)
    - Entropy (how much information is known)
    """
    
    timestamp: str  # ISO format
    phase: str = "present"  # past, present, future
    certainty: float = 1.0  # 0.0 to 1.0
    branch_id: str = "main"  # Timeline identifier
    entropy: float = 0.0  # Information content (higher = more known)
    
    def is_accessible(self, current_time: str) -> bool:
        """Check if this temporal coordinate is accessible from current time."""
        # Past is always accessible
        if self.phase == "past":
            return True
        # Present is always accessible
        if self.phase == "present":
            return True
        # Future is accessible if it's a hypothesis (certainty < 1)
        if self.phase == "future" and self.certainty < 1.0:
            return True
        return False
    
    def distance_to(self, other: 'TemporalCoordinate') -> float:
        """
        Compute temporal distance to another coordinate.
        
        Not just time difference - includes phase and branch.
        """
        # Parse timestamps
        try:
            t1 = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(other.timestamp.replace('Z', '+00:00'))
            time_diff = abs((t1 - t2).total_seconds())
        except:
            time_diff = 0
        
        # Phase penalty
        phase_penalty = 0
        if self.phase != other.phase:
            phase_penalty = 1000  # Crossing phases is expensive
        
        # Branch penalty
        branch_penalty = 0
        if self.branch_id != other.branch_id:
            branch_penalty = 10000  # Crossing branches is very expensive
        
        return time_diff + phase_penalty + branch_penalty


@dataclass
class Hypothesis:
    """
    A hypothesis about a possible future (or past).
    
    Hypotheses are:
    - Stored as crystals in the future phase
    - Have probability distributions
    - Can be validated against reality
    - Can spawn counterfactuals
    """
    
    hypothesis_id: str
    description: str
    
    # Temporal context
    temporal_coord: TemporalCoordinate
    
    # Content
    crystal: Optional[Crystal] = None  # Geometric form of hypothesis
    
    # Probability
    prior_probability: float = 0.5  # Initial belief
    posterior_probability: float = 0.5  # Updated belief after evidence
    
    # Evidence
    supporting_evidence: list[str] = field(default_factory=list)  # Crystal IDs
    contradicting_evidence: list[str] = field(default_factory=list)
    
    # Outcomes
    predicted_outcomes: list[str] = field(default_factory=list)
    actual_outcome: Optional[str] = None  # Filled in when hypothesis resolves
    
    # Status
    status: str = "active"  # active, confirmed, refuted, superseded
    superseded_by: Optional[str] = None  # If another hypothesis replaced this
    
    def update_probability(self, evidence_crystal_id: str, supports: bool,
                          strength: float = 0.1):
        """Update probability based on new evidence."""
        if supports:
            self.supporting_evidence.append(evidence_crystal_id)
            self.posterior_probability = min(1.0, self.posterior_probability + strength)
        else:
            self.contradicting_evidence.append(evidence_crystal_id)
            self.posterior_probability = max(0.0, self.posterior_probability - strength)
    
    def confirm(self, outcome_crystal_id: str):
        """Mark hypothesis as confirmed (actually happened)."""
        self.status = "confirmed"
        self.actual_outcome = outcome_crystal_id
        self.posterior_probability = 1.0
        self.temporal_coord.phase = "past"  # Now it's history
        self.temporal_coord.certainty = 1.0
    
    def refute(self, actual_crystal_id: str):
        """Mark hypothesis as refuted."""
        self.status = "refuted"
        self.actual_outcome = actual_crystal_id
        self.posterior_probability = 0.0


@dataclass
class Memory:
    """
    A memory of a past event.
    
    Memories are:
    - Stored as crystals in the past phase
    - Subject to decay (reliability decreases over time)
    - Can be reinforced or questioned
    - Form the basis of learned patterns
    """
    
    memory_id: str
    description: str
    
    # Temporal context
    temporal_coord: TemporalCoordinate
    
    # Content
    crystal: Optional[Crystal] = None
    
    # Reliability
    initial_reliability: float = 1.0
    current_reliability: float = 1.0
    
    # Decay
    decay_rate: float = 0.01  # How fast reliability decays per time unit
    last_reinforced: Optional[str] = None
    
    # Associations
    associated_memories: list[str] = field(default_factory=list)  # Memory IDs
    triggered_hypotheses: list[str] = field(default_factory=list)  # Hypothesis IDs
    
    def decay(self, current_time: str):
        """Apply time-based decay to reliability."""
        try:
            then = datetime.fromisoformat(self.temporal_coord.timestamp.replace('Z', '+00:00'))
            now = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            time_diff = (now - then).total_seconds()
            
            # Exponential decay
            self.current_reliability = self.initial_reliability * (
                0.5 ** (time_diff * self.decay_rate / 86400)  # Per day
            )
        except:
            pass
    
    def reinforce(self, current_time: str):
        """Reinforce memory (increase reliability)."""
        self.current_reliability = min(1.0, self.current_reliability * 1.1)
        self.last_reinforced = current_time


class HypothesisEngine:
    """
    Engine for generating and managing hypotheses.
    
    Can generate hypotheses about:
    - Future outcomes (predictions)
    - Past events (retrodiction)
    - Counterfactuals (what if?)
    """
    
    def __init__(self, lattice):
        self.lattice = lattice
        self.active_hypotheses: dict[str, Hypothesis] = {}
    
    def generate(self, context_crystal: Crystal, description: str = "",
                num_hypotheses: int = 3) -> list[Hypothesis]:
        """
        Generate hypotheses from a context crystal.
        
        Uses the crystal's geometric structure to infer possible futures.
        """
        hypotheses = []
        
        for i in range(num_hypotheses):
            # Create hypothesis crystal (perturbed version of context)
            from .crystal import CrystalFactory
            
            # Vibrate the crystal at different frequencies to get variations
            freq = 0.5 + i * 0.5
            response = context_crystal.vibrate(freq)
            
            # Create variant
            variant_id = f"hypo_{context_crystal.crystal_id}_{i}_{datetime.utcnow().timestamp()}"
            
            hypothesis = Hypothesis(
                hypothesis_id=variant_id,
                description=description or f"Hypothesis {i+1} from {context_crystal.name}",
                temporal_coord=TemporalCoordinate(
                    timestamp=datetime.utcnow().isoformat(),
                    phase="future",
                    certainty=0.5 + i * 0.1,  # Decreasing certainty
                    branch_id=f"hypo_branch_{i}"
                ),
                prior_probability=0.5,
                predicted_outcomes=[f"outcome_{i}_{j}" for j in range(3)]
            )
            
            # Store hypothesis crystal
            hypo_crystal = Crystal(
                crystal_id=variant_id,
                name=hypothesis.description,
                temporal_phase="future",
                tags=["hypothesis", f"from_{context_crystal.crystal_id}"]
            )
            # Copy and perturb atoms
            hypo_crystal.atoms = context_crystal.atoms  # Simplified
            hypo_crystal.parent_crystal_ids = [context_crystal.crystal_id]
            
            hypothesis.crystal = hypo_crystal
            self.lattice.store(hypo_crystal)
            
            hypotheses.append(hypothesis)
            self.active_hypotheses[variant_id] = hypothesis
        
        return hypotheses
    
    def validate(self, hypothesis_id: str, actual_crystal: Crystal) -> bool:
        """
        Validate a hypothesis against actual outcome.
        
        Returns True if hypothesis was confirmed.
        """
        if hypothesis_id not in self.active_hypotheses:
            return False
        
        hypo = self.active_hypotheses[hypothesis_id]
        
        # Compute resonance between hypothesis and actual
        if hypo.crystal:
            resonance = hypo.crystal.resonance_with(actual_crystal)
            
            if resonance > 0.8:
                hypo.confirm(actual_crystal.crystal_id)
                return True
            else:
                hypo.refute(actual_crystal.crystal_id)
                return False
        
        return False
    
    def get_active(self) -> list[Hypothesis]:
        """Get all active hypotheses."""
        return [h for h in self.active_hypotheses.values() if h.status == "active"]
    
    def get_by_probability(self, min_prob: float = 0.0, 
                          max_prob: float = 1.0) -> list[Hypothesis]:
        """Get hypotheses within probability range."""
        return [
            h for h in self.active_hypotheses.values()
            if min_prob <= h.posterior_probability <= max_prob
        ]


class TemporalLayer:
    """
    The temporal layer manages all time-related operations.
    
    Provides a unified interface for:
    - Storing memories (past)
    - Recording present (now)
    - Generating hypotheses (future)
    - Time-travel queries
    - Causal analysis
    """
    
    def __init__(self, lattice, db_path: Path | str = "temporal_layer.db"):
        self.lattice = lattice
        self.db_path = Path(db_path)
        self.hypothesis_engine = HypothesisEngine(lattice)
        self._init_db()
    
    def _init_db(self):
        """Initialize temporal storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS temporal_index (
                    crystal_id TEXT PRIMARY KEY,
                    temporal_phase TEXT,
                    timestamp TEXT,
                    certainty REAL,
                    branch_id TEXT,
                    entropy REAL,
                    json_coord TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_temp_phase ON temporal_index(temporal_phase);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON temporal_index(timestamp);
                CREATE INDEX IF NOT EXISTS idx_branch ON temporal_index(branch_id);
                
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    description TEXT,
                    temporal_coord TEXT,
                    crystal_id TEXT,
                    prior_prob REAL,
                    posterior_prob REAL,
                    status TEXT,
                    superseded_by TEXT
                );
                
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    description TEXT,
                    temporal_coord TEXT,
                    crystal_id TEXT,
                    reliability REAL,
                    decay_rate REAL
                );
            """)
            conn.commit()
    
    def now(self) -> TemporalCoordinate:
        """Get current temporal coordinate."""
        return TemporalCoordinate(
            timestamp=datetime.utcnow().isoformat(),
            phase="present",
            certainty=1.0
        )
    
    def remember(self, crystal: Crystal, description: str = "",
                reliability: float = 1.0) -> Memory:
        """
        Store a memory of the present moment.
        
        The crystal moves from present → past.
        """
        memory_id = f"mem_{crystal.crystal_id}_{datetime.utcnow().timestamp()}"
        
        memory = Memory(
            memory_id=memory_id,
            description=description or f"Memory of {crystal.name}",
            temporal_coord=TemporalCoordinate(
                timestamp=datetime.utcnow().isoformat(),
                phase="past",
                certainty=1.0
            ),
            crystal=crystal,
            initial_reliability=reliability,
            current_reliability=reliability
        )
        
        # Update crystal's temporal phase
        crystal.temporal_phase = "past"
        self.lattice.store(crystal)
        
        # Store memory record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO memories 
                   (memory_id, description, temporal_coord, crystal_id, reliability, decay_rate)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    description,
                    json.dumps({
                        "timestamp": memory.temporal_coord.timestamp,
                        "phase": memory.temporal_coord.phase,
                        "certainty": memory.temporal_coord.certainty
                    }),
                    crystal.crystal_id,
                    reliability,
                    memory.decay_rate
                )
            )
            conn.commit()
        
        return memory
    
    def hypothesize(self, context_crystal: Crystal, description: str = "",
                   num_hypotheses: int = 3) -> list[Hypothesis]:
        """
        Generate hypotheses about possible futures.
        
        Returns future-phase crystals with probability distributions.
        """
        return self.hypothesis_engine.generate(context_crystal, description, num_hypotheses)
    
    def query_time(self, timestamp: str, tolerance_seconds: float = 60) -> Iterator[Crystal]:
        """
        Time-travel query: find crystals at a specific time.
        
        Looks in past, present, and future phases.
        """
        try:
            target = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Query all crystals and filter by time
            for crystal in self.lattice.find_by_temporal_phase("past"):
                yield crystal
            for crystal in self.lattice.find_by_temporal_phase("present"):
                yield crystal
            for crystal in self.lattice.find_by_temporal_phase("future"):
                yield crystal
                
        except:
            pass
    
    def query_branch(self, branch_id: str) -> Iterator[Crystal]:
        """Query all crystals in a specific timeline branch."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT crystal_id FROM temporal_index WHERE branch_id = ?",
                (branch_id,)
            )
            
            for row in rows:
                crystal = self.lattice.retrieve(row[0])
                if crystal:
                    yield crystal
    
    def causal_chain(self, effect_crystal_id: str) -> list[Crystal]:
        """
        Trace causal chain leading to an effect.
        
        Follows parent relationships backward in time.
        """
        chain = []
        current_id = effect_crystal_id
        
        while current_id:
            crystal = self.lattice.retrieve(current_id)
            if not crystal:
                break
            
            chain.append(crystal)
            
            # Follow to parent (simplified - would need full graph)
            if crystal.parent_crystal_ids:
                current_id = crystal.parent_crystal_ids[0]
            else:
                break
        
        return list(reversed(chain))
    
    def counterfactual(self, actual_crystal: Crystal, 
                      changes: dict) -> Crystal:
        """
        Generate counterfactual (what if?) scenario.
        
        Creates a future-phase crystal representing an alternative past.
        """
        from .crystal import Crystal
        
        # Create modified version
        counterfactual_id = f"cf_{actual_crystal.crystal_id}_{datetime.utcnow().timestamp()}"
        
        cf_crystal = Crystal(
            crystal_id=counterfactual_id,
            name=f"Counterfactual: {actual_crystal.name}",
            temporal_phase="future",  # It's hypothetical
            envelope={
                "based_on": actual_crystal.crystal_id,
                "changes": changes,
                "type": "counterfactual"
            },
            tags=["counterfactual", f"from_{actual_crystal.crystal_id}"]
        )
        
        # Copy and modify atoms (simplified)
        cf_crystal.atoms = actual_crystal.atoms
        cf_crystal.parent_crystal_ids = [actual_crystal.crystal_id]
        
        self.lattice.store(cf_crystal)
        
        return cf_crystal
    
    def get_timeline(self) -> dict:
        """Get overview of all temporal phases."""
        return {
            "past": list(self.lattice.find_by_temporal_phase("past")),
            "present": list(self.lattice.find_by_temporal_phase("present")),
            "future": list(self.lattice.find_by_temporal_phase("future")),
            "active_hypotheses": len(self.hypothesis_engine.get_active())
        }
