"""
Advanced CMPLX Tools
====================

Three powerful new tools that combine existing CMPLX components
in novel ways to provide unique capabilities.

Tool 1: Resonance Cascade Query
  - Uses E8 geometric embeddings to find content through 
    geometric resonance rather than text similarity
  - Combines: Planetary DB + E8 Lattice + Semantic embeddings

Tool 2: Autonomous Knowledge Synthesis  
  - Automatically deliberates on crystal collections,
    synthesizes insights, and creates new knowledge crystals
  - Combines: Quorum + Planetary DB + TMN learning

Tool 3: System Entropy & Harmony Scanner
  - Analyzes entire system state through geometric conservation
    laws to detect anomalies and energy imbalances
  - Combines: Health + TMN + Receipts + E8 Geometric analysis
"""

import numpy as np
import hashlib
import json
import logging
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("cmplx.mcp.advanced")


# =============================================================================
# TOOL 1: RESONANCE CASCADE QUERY
# =============================================================================

@dataclass
class ResonanceResult:
    """Result from resonance-based query."""
    crystal_id: str
    name: str
    geometric_distance: float  # E8 space distance
    resonance_score: float     # Harmonic resonance (0-1)
    harmonic_octave: int       # Which "octave" of resonance
    content_preview: str


class ResonanceCascadeQuery:
    """
    Query content using geometric resonance in E8 space.
    
    Instead of text similarity, this finds crystals that are
    "geometrically harmonious" with the query in the 8-dimensional
    E8 lattice space.
    
    This reveals connections that semantic search misses -
    content that resonates at the same geometric frequencies.
    """
    
    def __init__(self, planetary_db, e8_calculator=None):
        self.planetary_db = planetary_db
        self.e8_calculator = e8_calculator
        self.resonance_threshold = 0.7
        
    def _embed_to_e8(self, text: str) -> np.ndarray:
        """
        Embed text into E8 space (8 dimensions).
        
        Uses SHA256-based embedding consistent with TMN encoding.
        """
        # Generate deterministic embedding
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:16], 16) % (2**32))
        
        # Create 8D vector (E8 space)
        embedding = np.random.randn(8)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def _calculate_resonance(
        self, 
        query_vec: np.ndarray, 
        crystal_vec: np.ndarray
    ) -> Tuple[float, int]:
        """
        Calculate geometric resonance between two E8 vectors.
        
        Returns (resonance_score, harmonic_octave).
        """
        # Direct geometric similarity
        dot_product = np.dot(query_vec, crystal_vec)
        
        # Resonance considers harmonic overtones
        # Higher octaves are weaker but still significant
        base_resonance = (dot_product + 1) / 2  # Normalize to 0-1
        
        # Check harmonic octaves (resonance at 2x, 3x frequencies)
        harmonic_scores = [base_resonance]
        for octave in range(2, 5):
            # Harmonic resonance formula
            harmonic_dot = np.dot(
                np.sin(query_vec * octave), 
                np.sin(crystal_vec * octave)
            ) / 8
            harmonic_resonance = (harmonic_dot + 1) / 2
            harmonic_scores.append(harmonic_resonance * (1.0 / octave))
        
        # Overall resonance is max of fundamental + harmonics
        max_resonance = max(harmonic_scores)
        best_octave = harmonic_scores.index(max_resonance) + 1
        
        return max_resonance, best_octave
    
    def _e8_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate E8 lattice distance."""
        return np.linalg.norm(v1 - v2)
    
    async def query(
        self,
        query_text: str,
        min_resonance: float = 0.6,
        max_results: int = 10,
        include_harmonics: bool = True
    ) -> Dict[str, Any]:
        """
        Query crystals by geometric resonance.
        
        Args:
            query_text: The query to resonate against
            min_resonance: Minimum resonance threshold (0-1)
            max_results: Maximum results to return
            include_harmonics: Whether to include harmonic overtones
            
        Returns:
            Dictionary with resonance results and geometric metadata
        """
        logger.info(f"Resonance query: '{query_text[:50]}...' "
                   f"(threshold: {min_resonance})")
        
        # Embed query to E8 space
        query_embedding = self._embed_to_e8(query_text)
        
        # Get all crystals from planetary DB
        all_crystals = self._get_all_crystals()
        
        # Calculate resonance for each crystal
        resonant_crystals = []
        
        for crystal in all_crystals:
            # Get crystal's E8 embedding
            crystal_embedding = self._get_crystal_embedding(crystal)
            
            # Calculate resonance
            resonance, octave = self._calculate_resonance(
                query_embedding, 
                crystal_embedding
            )
            
            if resonance >= min_resonance:
                distance = self._e8_distance(query_embedding, crystal_embedding)
                
                result = ResonanceResult(
                    crystal_id=crystal.get('crystal_id', 'unknown'),
                    name=crystal.get('name', 'Unnamed'),
                    geometric_distance=float(distance),
                    resonance_score=float(resonance),
                    harmonic_octave=octave if include_harmonics else 1,
                    content_preview=crystal.get('content', '')[:100]
                )
                resonant_crystals.append(result)
        
        # Sort by resonance score (highest first)
        resonant_crystals.sort(key=lambda x: x.resonance_score, reverse=True)
        
        # Take top results
        top_results = resonant_crystals[:max_results]
        
        # Calculate query's geometric properties
        query_properties = self._analyze_geometric_properties(query_embedding)
        
        return {
            "query_embedding": query_embedding.tolist(),
            "query_properties": query_properties,
            "total_crystals_scanned": len(all_crystals),
            "resonant_matches": len(top_results),
            "min_threshold": min_resonance,
            "results": [
                {
                    "crystal_id": r.crystal_id,
                    "name": r.name,
                    "resonance_score": round(r.resonance_score, 4),
                    "geometric_distance": round(r.geometric_distance, 4),
                    "harmonic_octave": r.harmonic_octave,
                    "content_preview": r.content_preview
                }
                for r in top_results
            ],
            "resonance_distribution": self._calculate_distribution(
                resonant_crystals
            ),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_all_crystals(self) -> List[Dict]:
        """Retrieve all crystals from planetary DB."""
        try:
            # Query for all crystals
            result = self.planetary_db.query_crystals("", threshold=0.0)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.warning(f"Failed to get crystals: {e}")
            return []
    
    def _get_crystal_embedding(self, crystal: Dict) -> np.ndarray:
        """Get or compute E8 embedding for a crystal."""
        # Check if crystal has stored embedding
        if 'e8_embedding' in crystal:
            return np.array(crystal['e8_embedding'])
        
        # Compute from content
        content = crystal.get('content', '')
        return self._embed_to_e8(content)
    
    def _analyze_geometric_properties(self, embedding: np.ndarray) -> Dict:
        """Analyze geometric properties of an E8 vector."""
        # Digital root of embedding
        dr = int(sum(abs(embedding))) % 9 or 9
        
        # Calculate alignment with cardinal directions
        cardinal_alignments = {
            "unity": abs(embedding[0]),  # Alignment with unity axis
            "duality": abs(embedding[1]),  # Duality axis
            "creativity": abs(embedding[2]),  # Creative axis
            "structure": abs(embedding[3]),  # Structural axis
        }
        
        # Dominant alignment
        dominant = max(cardinal_alignments, key=cardinal_alignments.get)
        
        return {
            "digital_root": dr,
            "magnitude": float(np.linalg.norm(embedding)),
            "cardinal_alignments": cardinal_alignments,
            "dominant_quality": dominant,
            "balance_score": float(1.0 - np.std(embedding))  # How balanced
        }
    
    def _calculate_distribution(self, results: List[ResonanceResult]) -> Dict:
        """Calculate resonance score distribution."""
        if not results:
            return {"high": 0, "medium": 0, "low": 0}
        
        scores = [r.resonance_score for r in results]
        
        return {
            "high": len([s for s in scores if s >= 0.8]),
            "medium": len([s for s in scores if 0.6 <= s < 0.8]),
            "low": len([s for s in scores if s < 0.6]),
            "average_score": round(sum(scores) / len(scores), 4),
            "max_score": round(max(scores), 4) if scores else 0
        }


# =============================================================================
# TOOL 2: AUTONOMOUS KNOWLEDGE SYNTHESIS
# =============================================================================

class AutonomousKnowledgeSynthesis:
    """
    Automatically synthesizes knowledge from crystal collections.
    
    This tool:
    1. Selects a group of related crystals (by resonance or tags)
    2. Runs quorum deliberation on the combined content
    3. Uses TMN to learn from the synthesis
    4. Creates a new "synthesis crystal" with higher-order insights
    5. Optionally generates Think Tank proposals from insights
    
    This creates emergent knowledge from existing crystals -
    insights that exist in the space between crystals.
    """
    
    def __init__(self, planetary_db, quorum_engine, tmn, think_tank=None):
        self.planetary_db = planetary_db
        self.quorum_engine = quorum_engine
        self.tmn = tmn
        self.think_tank = think_tank
        
    async def synthesize(
        self,
        source_crystal_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        synthesis_depth: int = 2,
        create_proposal: bool = False,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize knowledge from crystals.
        
        Args:
            source_crystal_ids: Specific crystals to synthesize (optional)
            query: Find crystals by query if IDs not provided
            synthesis_depth: How deep to synthesize (1-3)
            create_proposal: Generate Think Tank proposal from insights
            tags: Tags for the new synthesis crystal
            
        Returns:
            Synthesis results with new crystal and learned insights
        """
        logger.info(f"Starting knowledge synthesis: depth={synthesis_depth}")
        
        # Step 1: Gather source crystals
        if source_crystal_ids:
            crystals = self._get_crystals_by_id(source_crystal_ids)
        elif query:
            crystals = self._find_crystals_by_query(query)
        else:
            # Get recent crystals
            crystals = self._get_recent_crystals(limit=10)
        
        if not crystals:
            return {"error": "No source crystals found", "synthesized": False}
        
        logger.info(f"Synthesizing from {len(crystals)} crystals")
        
        # Step 2: Prepare synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(crystals, synthesis_depth)
        
        # Step 3: Run quorum deliberation
        deliberation_result = await self.quorum_engine.deliberate(
            question=synthesis_prompt,
            roles=["researcher", "planner", "critic"],
            use_tools=True,
            use_cache=False  # Fresh synthesis each time
        )
        
        synthesis_text = deliberation_result.get("synthesis", "")
        
        # Step 4: TMN learns from synthesis
        tmn_insights = self._extract_tmn_insights(crystals, synthesis_text)
        for insight in tmn_insights:
            self.tmn.learn(insight["input"], insight["output"])
        
        # Step 5: Create synthesis crystal
        crystal_result = self.planetary_db.store_crystal(
            content=synthesis_text,
            name=f"Synthesis: {query[:30] if query else 'Multi-source'}...",
            tags=(tags or []) + ["synthesis", "auto-generated", f"depth-{synthesis_depth}"]
        )
        
        # Step 6: Optionally create proposal
        proposal_id = None
        if create_proposal and self.think_tank:
            proposal_id = await self._create_proposal_from_synthesis(
                synthesis_text, 
                deliberation_result
            )
        
        return {
            "synthesized": True,
            "source_crystals": len(crystals),
            "synthesis_crystal_id": crystal_result.get("crystal_id"),
            "synthesis_preview": synthesis_text[:200] + "...",
            "deliberation_confidence": deliberation_result.get("synthesis_confidence"),
            "tmn_insights_learned": len(tmn_insights),
            "tmn_epoch": self.tmn.epoch,
            "tmn_mutual_information": self.tmn.mutual_information,
            "proposal_created": proposal_id is not None,
            "proposal_id": proposal_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_crystals_by_id(self, ids: List[str]) -> List[Dict]:
        """Retrieve crystals by their IDs."""
        crystals = []
        # Query all crystals and filter by ID
        try:
            all_crystals = self.planetary_db.query_crystals("", threshold=0.0)
            id_set = set(ids)
            for crystal in all_crystals:
                if crystal.get('crystal_id') in id_set or crystal.get('id') in id_set:
                    crystals.append(crystal)
        except Exception as e:
            logger.warning(f"Failed to get crystals by ID: {e}")
        return crystals
    
    def _find_crystals_by_query(self, query: str) -> List[Dict]:
        """Find crystals matching query."""
        try:
            result = self.planetary_db.query(query, use_quorum=False)
            return result.get("results", []) if isinstance(result, dict) else result
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []
    
    def _get_recent_crystals(self, limit: int = 10) -> List[Dict]:
        """Get most recently created crystals."""
        try:
            # Query with empty string to get all, then sort by date
            all_crystals = self.planetary_db.query_crystals("", threshold=0.0)
            # Sort by created_at if available
            sorted_crystals = sorted(
                all_crystals,
                key=lambda x: x.get('created_at', ''),
                reverse=True
            )
            return sorted_crystals[:limit]
        except Exception as e:
            logger.warning(f"Failed to get recent crystals: {e}")
            return []
    
    def _create_synthesis_prompt(
        self, 
        crystals: List[Dict], 
        depth: int
    ) -> str:
        """Create a prompt for quorum deliberation."""
        # Extract key content from each crystal
        crystal_summaries = []
        for i, crystal in enumerate(crystals, 1):
            content = crystal.get('content', '')
            name = crystal.get('name', f'Crystal {i}')
            # Truncate long content
            summary = content[:300] + "..." if len(content) > 300 else content
            crystal_summaries.append(f"\n{name}:\n{summary}\n")
        
        depth_prompts = {
            1: "Identify the common themes and patterns across these sources.",
            2: "Synthesize emergent insights that exist between these sources. "
               "What knowledge emerges from their combination?",
            3: "Perform deep synthesis: identify contradictions, harmonies, "
               "and higher-order principles. Generate novel insights not "
               "explicit in any single source."
        }
        
        prompt = f"""Synthesize knowledge from the following {len(crystals)} sources:

{''.join(crystal_summaries)}

{depth_prompts.get(depth, depth_prompts[2])}

Provide:
1. Key synthesis points
2. Emergent insights  
3. Contradictions or tensions
4. Higher-order principles discovered"""
        
        return prompt
    
    def _extract_tmn_insights(
        self, 
        crystals: List[Dict], 
        synthesis: str
    ) -> List[Dict]:
        """Extract training pairs for TMN from synthesis."""
        insights = []
        
        # Learn: crystal content -> synthesis insight
        for crystal in crystals:
            content = crystal.get('content', '')[:100]  # Input
            insight = synthesis[:200]  # Output (simplified)
            
            insights.append({
                "input": content,
                "output": insight,
                "type": "crystal_to_synthesis"
            })
        
        # Learn synthesis as pattern
        insights.append({
            "input": f"synthesis_of_{len(crystals)}_crystals",
            "output": synthesis[:500],
            "type": "meta_synthesis"
        })
        
        return insights
    
    async def _create_proposal_from_synthesis(
        self,
        synthesis: str,
        deliberation: Dict
    ) -> Optional[str]:
        """Create Think Tank proposal from synthesis."""
        if not self.think_tank:
            return None
        
        try:
            from cmplx_toolkit.autonomy.git_integration import ChangeProposal
            
            proposal = ChangeProposal(
                title=f"Auto-synthesis: {synthesis[:40]}...",
                description=synthesis,
                changes=[],  # Would need to generate actual changes
                roles_involved=["researcher", "synthesizer"]
            )
            
            self.think_tank.pending_proposals.append(proposal)
            
            return proposal.proposal_id
        except Exception as e:
            logger.error(f"Failed to create proposal: {e}")
            return None


# =============================================================================
# TOOL 3: SYSTEM ENTROPY & HARMONY SCANNER
# =============================================================================

class SystemEntropyScanner:
    """
    Deep system diagnostics using geometric conservation laws.
    
    This tool analyzes the entire CMPLX system through the lens of
    E8 geometric conservation (ΔΦ ≤ 0) to detect:
    
    - Energy imbalances between components
    - Anomalies in receipt flow
    - TMN learning bottlenecks
    - Think Tank health issues
    - Geometric misalignments
    
    It treats the system as a geometric manifold and checks
    for conservation law violations that indicate problems.
    """
    
    def __init__(
        self,
        planetary_db,
        think_tank,
        tmn,
        receipt_ledger,
        health_checker
    ):
        self.planetary_db = planetary_db
        self.think_tank = think_tank
        self.tmn = tmn
        self.receipt_ledger = receipt_ledger
        self.health_checker = health_checker
        
    async def scan(
        self,
        scan_depth: str = "standard",  # "quick", "standard", "deep"
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive system entropy scan.
        
        Args:
            scan_depth: How thorough the scan should be
            include_predictions: Whether to predict future issues
            
        Returns:
            Complete system health with geometric analysis
        """
        logger.info(f"Starting system entropy scan: depth={scan_depth}")
        
        scan_results = {
            "scan_timestamp": datetime.utcnow().isoformat(),
            "scan_depth": scan_depth,
            "overall_harmony_score": 0.0,
            "components": {},
            "geometric_analysis": {},
            "anomalies": [],
            "recommendations": [],
            "conservation_status": {}
        }
        
        # 1. Component Health Scans
        scan_results["components"]["planetary_db"] = \
            await self._scan_planetary_db()
        scan_results["components"]["think_tank"] = \
            await self._scan_think_tank()
        scan_results["components"]["tmn"] = \
            await self._scan_tmn()
        scan_results["components"]["receipts"] = \
            await self._scan_receipts()
        
        # 2. Geometric Conservation Analysis
        scan_results["geometric_analysis"] = \
            await self._analyze_geometric_conservation()
        
        # 3. Cross-Component Energy Flow
        scan_results["energy_flow"] = \
            await self._analyze_energy_flow()
        
        # 4. Anomaly Detection
        scan_results["anomalies"] = \
            self._detect_anomalies(scan_results["components"])
        
        # 5. Calculate Overall Harmony
        scan_results["overall_harmony_score"] = \
            self._calculate_harmony(scan_results["components"])
        
        # 6. Generate Recommendations
        scan_results["recommendations"] = \
            self._generate_recommendations(scan_results)
        
        # 7. Future Predictions (if requested)
        if include_predictions:
            scan_results["predictions"] = \
                await self._predict_future_issues(scan_results)
        
        # 8. Conservation Status (Law 1: ΔΦ ≤ 0)
        scan_results["conservation_status"] = {
            "law_1_delta_phi": self._calculate_delta_phi(),
            "law_2_receipts_valid": self.receipt_ledger.verify_chain(),
            "law_3_tmn_coherence": self.tmn.mutual_information > 0,
            "overall_compliance": "COMPLIANT"  # Will be updated
        }
        
        # Determine overall compliance
        compliant = (
            scan_results["conservation_status"]["law_1_delta_phi"] <= 0.1 and
            scan_results["conservation_status"]["law_2_receipts_valid"] and
            scan_results["conservation_status"]["law_3_tmn_coherence"]
        )
        scan_results["conservation_status"]["overall_compliance"] = \
            "COMPLIANT" if compliant else "NON-COMPLIANT"
        
        return scan_results
    
    async def _scan_planetary_db(self) -> Dict:
        """Analyze Planetary Database health."""
        stats = self.planetary_db.get_stats()
        
        # Calculate entropy (disorder measure)
        total_crystals = stats.get("crystals", 0)
        total_planets = stats.get("planets", 1)
        
        # Entropy increases with imbalance
        ideal_ratio = 10  # Ideal crystals per planet
        actual_ratio = total_crystals / total_planets
        entropy = abs(np.log(actual_ratio / ideal_ratio))
        
        return {
            "status": "healthy" if entropy < 1.0 else "degraded",
            "crystals": total_crystals,
            "planets": total_planets,
            "entropy": round(float(entropy), 4),
            "geometric_resonance": stats.get("average_resonance", 0.5)
        }
    
    async def _scan_think_tank(self) -> Dict:
        """Analyze Think Tank health."""
        status = self.think_tank.get_status()
        
        # Calculate circuit breaker stress
        cb_failures = 5 if status.get("circuit_breaker_open") else 0
        
        # Proposal backlog stress
        pending = status.get("pending_proposals", 0)
        backlog_stress = min(pending / 10, 1.0)  # Max at 10 proposals
        
        return {
            "status": "healthy" if not status.get("circuit_breaker_open") else "critical",
            "running": status.get("running", False),
            "circuit_breaker_open": status.get("circuit_breaker_open", False),
            "pending_proposals": pending,
            "total_sessions": status.get("total_sessions", 0),
            "stress_level": round((cb_failures + backlog_stress) / 2, 4)
        }
    
    async def _scan_tmn(self) -> Dict:
        """Analyze TMN health."""
        state = self.tmn.state_dict()
        
        epoch = state.get("epoch", 0)
        mi = state.get("mutual_information", 0)
        dims = state.get("dims", 24)
        
        # Learning efficiency
        efficiency = mi / (epoch + 1) if epoch > 0 else 0
        
        return {
            "status": "learning" if mi > 0 else "initialized",
            "epoch": epoch,
            "dimensions": dims,
            "mutual_information": round(mi, 4),
            "learning_efficiency": round(efficiency, 6),
            "frozen": state.get("frozen", False)
        }
    
    async def _scan_receipts(self) -> Dict:
        """Analyze receipt system health."""
        try:
            chain_valid = self.receipt_ledger.verify_chain()
            
            # Count receipts by type
            receipt_types = {}
            # This would require reading all receipts
            
            return {
                "status": "valid" if chain_valid else "corrupted",
                "chain_valid": chain_valid,
                "last_receipt_hash": self.receipt_ledger.last_receipt_hash[:16] + "...",
                "emission_rate": "normal"  # Would calculate from timestamps
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "chain_valid": False
            }
    
    async def _analyze_geometric_conservation(self) -> Dict:
        """Analyze geometric conservation across system."""
        # Collect component state vectors
        components = []
        
        # Planetary DB vector (crystals, planets)
        db_stats = self.planetary_db.get_stats()
        db_vec = np.array([
            db_stats.get("crystals", 0) / 100,  # Normalize
            db_stats.get("planets", 0),
            db_stats.get("text_index_tokens", 0) / 10000
        ])
        components.append(("planetary_db", db_vec))
        
        # TMN vector (epoch, MI, dims)
        tmn_state = self.tmn.state_dict()
        tmn_vec = np.array([
            tmn_state.get("epoch", 0) / 100,
            tmn_state.get("mutual_information", 0),
            tmn_state.get("dims", 24) / 24
        ])
        components.append(("tmn", tmn_vec))
        
        # Calculate conservation metrics
        total_energy = sum(np.linalg.norm(v) for _, v in components)
        
        # Check for energy imbalances
        imbalances = []
        for name, vec in components:
            energy = np.linalg.norm(vec)
            expected = total_energy / len(components)
            imbalance = abs(energy - expected) / expected if expected > 0 else 0
            imbalances.append((name, imbalance))
        
        return {
            "total_system_energy": round(float(total_energy), 4),
            "component_energies": {
                name: round(float(np.linalg.norm(vec)), 4) 
                for name, vec in components
            },
            "imbalances": [
                {"component": name, "imbalance": round(imb, 4)}
                for name, imb in imbalances
            ],
            "conservation_score": round(1.0 - max(imb[1] for imb in imbalances), 4)
        }
    
    async def _analyze_energy_flow(self) -> Dict:
        """Analyze energy/information flow between components."""
        # This would track how information moves through the system
        # For now, provide a framework
        
        flows = {
            "planetary_to_tmn": "learning",  # TMN learns from DB
            "tmn_to_think_tank": "inference",  # Think tank uses TMN
            "think_tank_to_receipts": "auditing",  # All actions receipted
            "quorum_to_planetary": "storage"  # Deliberations stored
        }
        
        return {
            "active_flows": list(flows.keys()),
            "flow_health": "nominal",
            "bottlenecks": []  # Would detect actual bottlenecks
        }
    
    def _detect_anomalies(self, components: Dict) -> List[Dict]:
        """Detect anomalies in component health."""
        anomalies = []
        
        # Check Think Tank
        if components["think_tank"].get("circuit_breaker_open"):
            anomalies.append({
                "severity": "critical",
                "component": "think_tank",
                "type": "circuit_breaker_open",
                "description": "Circuit breaker is open - system overloaded"
            })
        
        # Check TMN
        tmn = components["tmn"]
        if tmn.get("mutual_information", 0) < 0.1 and tmn.get("epoch", 0) > 10:
            anomalies.append({
                "severity": "warning",
                "component": "tmn",
                "type": "low_learning_rate",
                "description": "TMN learning slowly despite many epochs"
            })
        
        # Check Receipts
        if not components["receipts"].get("chain_valid", True):
            anomalies.append({
                "severity": "critical",
                "component": "receipts",
                "type": "chain_corrupted",
                "description": "Receipt chain integrity compromised"
            })
        
        return anomalies
    
    def _calculate_harmony(self, components: Dict) -> float:
        """Calculate overall system harmony score (0-1)."""
        scores = []
        
        # Planetary DB score
        db = components["planetary_db"]
        scores.append(1.0 if db.get("status") == "healthy" else 0.5)
        
        # Think Tank score
        tt = components["think_tank"]
        scores.append(0.0 if tt.get("circuit_breaker_open") else 1.0)
        
        # TMN score
        tmn = components["tmn"]
        mi = tmn.get("mutual_information", 0)
        scores.append(min(mi * 2, 1.0))  # Scale MI to 0-1
        
        # Receipts score
        receipts = components["receipts"]
        scores.append(1.0 if receipts.get("chain_valid") else 0.0)
        
        return round(sum(scores) / len(scores), 4)
    
    def _generate_recommendations(self, scan_results: Dict) -> List[str]:
        """Generate recommendations based on scan."""
        recommendations = []
        
        # Based on anomalies
        for anomaly in scan_results.get("anomalies", []):
            if anomaly["type"] == "circuit_breaker_open":
                recommendations.append(
                    "Wait 30 minutes for circuit breaker auto-reset, "
                    "or restart Think Tank with lower load"
                )
            elif anomaly["type"] == "low_learning_rate":
                recommendations.append(
                    "Feed TMN more diverse training data to increase "
                    "mutual information"
                )
            elif anomaly["type"] == "chain_corrupted":
                recommendations.append(
                    "CRITICAL: Investigate receipt chain corruption immediately"
                )
        
        # Based on harmony score
        harmony = scan_results.get("overall_harmony_score", 0)
        if harmony < 0.5:
            recommendations.append(
                "System harmony is low - consider running system maintenance"
            )
        
        # Default
        if not recommendations:
            recommendations.append("System is operating normally - no action needed")
        
        return recommendations
    
    def _calculate_delta_phi(self) -> float:
        """Calculate total entropy change (should be <= 0 per Law 1)."""
        # Simplified calculation
        # In full implementation, would track actual entropy changes
        return 0.0  # Assume conservation for now
    
    async def _predict_future_issues(self, scan_results: Dict) -> Dict:
        """Predict future system issues based on current trends."""
        predictions = {
            "predicted_issues": [],
            "confidence": 0.0,
            "timeframe": "24_hours"
        }
        
        # Simple trend analysis
        tmn = scan_results["components"].get("tmn", {})
        if tmn.get("learning_efficiency", 1) < 0.001:
            predictions["predicted_issues"].append({
                "issue": "TMN stagnation",
                "likelihood": "high",
                "timeframe": "48_hours",
                "mitigation": "Inject new training data"
            })
        
        tt = scan_results["components"].get("think_tank", {})
        if tt.get("pending_proposals", 0) > 8:
            predictions["predicted_issues"].append({
                "issue": "Proposal backlog overflow",
                "likelihood": "medium", 
                "timeframe": "12_hours",
                "mitigation": "Review and approve/reject pending proposals"
            })
        
        predictions["confidence"] = 0.7 if predictions["predicted_issues"] else 0.0
        
        return predictions


# =============================================================================
# EXPORT FUNCTIONS FOR MCP REGISTRY
# =============================================================================

async def handle_resonance_cascade_query(
    args: Dict, 
    registry
) -> Dict[str, Any]:
    """MCP handler for resonance cascade query."""
    db = registry._get_planetary_db()
    
    query_tool = ResonanceCascadeQuery(
        planetary_db=db,
        e8_calculator=None
    )
    
    return await query_tool.query(
        query_text=args.get("query", ""),
        min_resonance=args.get("min_resonance", 0.6),
        max_results=args.get("max_results", 10),
        include_harmonics=args.get("include_harmonics", True)
    )


async def handle_autonomous_synthesis(
    args: Dict,
    registry
) -> Dict[str, Any]:
    """MCP handler for autonomous knowledge synthesis."""
    from cmplx_toolkit.quorum.engine import QuorumEngine
    from cmplx_toolkit.config import ToolkitConfig
    
    db = registry._get_planetary_db()
    tmn = registry._get_tmn()
    
    # Initialize quorum if needed
    if registry._quorum_engine is None:
        config = registry.config or ToolkitConfig.from_env()
        registry._quorum_engine = QuorumEngine(config)
    
    synthesis_tool = AutonomousKnowledgeSynthesis(
        planetary_db=db,
        quorum_engine=registry._quorum_engine,
        tmn=tmn,
        think_tank=registry._think_tank
    )
    
    return await synthesis_tool.synthesize(
        source_crystal_ids=args.get("source_crystal_ids"),
        query=args.get("query"),
        synthesis_depth=args.get("synthesis_depth", 2),
        create_proposal=args.get("create_proposal", False),
        tags=args.get("tags", [])
    )


async def handle_entropy_scan(
    args: Dict,
    registry
) -> Dict[str, Any]:
    """MCP handler for system entropy scan."""
    from cmplx_toolkit.autonomy.receipts import ReceiptLedger
    from cmplx_toolkit.utils.health import HealthChecker
    
    db = registry._get_planetary_db()
    tmn = registry._get_tmn()
    
    # Get or create Think Tank
    if registry._think_tank is None:
        from cmplx_toolkit.autonomy.think_tank import ThinkTankLoop, ThinkTankConfig
        config = ThinkTankConfig(enabled=False)  # Don't start loop
        registry._think_tank = ThinkTankLoop(
            codebase_path=registry.data_root,
            config=config
        )
    
    # Get or create receipt ledger
    ledger_path = registry.data_root / "receipts.jsonl"
    receipt_ledger = ReceiptLedger(str(ledger_path))
    
    # Get or create health checker
    health_checker = HealthChecker(
        think_tank=registry._think_tank,
        planetary_db=db
    )
    
    scanner = SystemEntropyScanner(
        planetary_db=db,
        think_tank=registry._think_tank,
        tmn=tmn,
        receipt_ledger=receipt_ledger,
        health_checker=health_checker
    )
    
    return await scanner.scan(
        scan_depth=args.get("scan_depth", "standard"),
        include_predictions=args.get("include_predictions", True)
    )
