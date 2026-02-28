"""
AGRM Router
===========
Adaptive Geometric Resonance Matrix routing engine.

Ports AGRM TSP concepts to general geometric routing between planets.
Uses Golden Ratio sweeps, zone classification, and midpoint unlocking.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum


class ZoneDensity(Enum):
    """Zone density classification."""
    SPARSE = "sparse"
    MEDIUM = "medium"
    DENSE = "dense"


@dataclass
class AGRMNode:
    """A node in AGRM routing space (represents a planet or data point)."""
    node_id: str
    position: List[float]  # 24D position
    resonance_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # AGRM classification
    shell_index: int = 0
    sector: int = 0
    quadrant: int = 0
    density: ZoneDensity = ZoneDensity.MEDIUM
    
    def distance_to(self, other: 'AGRMNode') -> float:
        """Euclidean distance to another node."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other.position)))


@dataclass
class AGRMRoute:
    """A route between nodes."""
    path: List[str]  # Node IDs in order
    total_distance: float
    legs: List[Tuple[str, str, float]]  # (from, to, distance)
    quality_score: float = 0.0  # 0-1, higher is better
    
    def __len__(self):
        return len(self.path)


@dataclass
class SweepResult:
    """Result of GR sweep."""
    ranked_nodes: List[Tuple[AGRMNode, float]]  # Node + GR score
    center: List[float]
    max_radius: float


class AGRMSweepScanner:
    """
    Golden Ratio sweep scanner.
    
    Ranks nodes by their alignment with GR-based spiral sweep pattern.
    """
    
    PHI = (1 + math.sqrt(5)) / 2
    
    def __init__(self, dimensions: int = 24):
        self.dimensions = dimensions
    
    def _compute_center(self, nodes: List[AGRMNode]) -> List[float]:
        """Compute centroid of nodes."""
        if not nodes:
            return [0.0] * self.dimensions
        
        center = []
        for d in range(self.dimensions):
            coords = [n.position[d] for n in nodes]
            center.append(sum(coords) / len(coords))
        return center
    
    def _gr_spiral_score(self, node: AGRMNode, center: List[float], 
                         radius: float, index: int) -> float:
        """
        Compute Golden Ratio spiral score for node.
        
        Higher score = better alignment with GR sweep pattern.
        """
        # Distance from center
        dist = math.sqrt(sum((node.position[d] - center[d]) ** 2 
                            for d in range(self.dimensions)))
        
        if dist > radius or dist < 0.0001:
            return 0.0
        
        # Normalize to 0-1
        normalized_dist = dist / radius
        
        # Golden ratio spiral position
        spiral_pos = (index * self.PHI) % 1.0
        
        # Score based on how close node is to ideal spiral position
        score = 1.0 - abs(normalized_dist - spiral_pos)
        
        # Bonus for nodes in sparse zones (exploration)
        if node.density == ZoneDensity.SPARSE:
            score *= 1.2
        
        return max(0.0, score)
    
    def sweep(self, nodes: List[AGRMNode], 
              center: Optional[List[float]] = None) -> SweepResult:
        """
        Perform GR sweep over nodes.
        
        Returns nodes ranked by sweep score.
        """
        if not nodes:
            return SweepResult([], [0.0] * self.dimensions, 0.0)
        
        if center is None:
            center = self._compute_center(nodes)
        
        # Compute max radius
        max_radius = max(
            math.sqrt(sum((n.position[d] - center[d]) ** 2 
                         for d in range(self.dimensions)))
            for n in nodes
        )
        
        # Score each node
        scored = []
        for i, node in enumerate(nodes):
            score = self._gr_spiral_score(node, center, max_radius, i)
            scored.append((node, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return SweepResult(scored, center, max_radius)


class AGRMZoneClassifier:
    """
    Classifies zones by density for intelligent routing.
    """
    
    def __init__(self, dimensions: int = 24):
        self.dimensions = dimensions
    
    def classify(self, node: AGRMNode, all_nodes: List[AGRMNode], 
                 radius: float) -> ZoneDensity:
        """Classify density of zone around node."""
        # Count neighbors within radius
        neighbors = sum(
            1 for n in all_nodes
            if n.node_id != node.node_id and node.distance_to(n) < radius
        )
        
        # Classify
        if neighbors < 3:
            return ZoneDensity.SPARSE
        elif neighbors < 8:
            return ZoneDensity.MEDIUM
        else:
            return ZoneDensity.DENSE
    
    def assign_shells(self, nodes: List[AGRMNode], 
                      center: List[float],
                      num_shells: int = 5) -> Dict[str, int]:
        """Assign nodes to concentric shells."""
        # Compute distances from center
        distances = []
        for n in nodes:
            dist = math.sqrt(sum((n.position[d] - center[d]) ** 2 
                                for d in range(self.dimensions)))
            distances.append((n, dist))
        
        if not distances:
            return {}
        
        max_dist = max(d for _, d in distances)
        shell_width = max_dist / num_shells if max_dist > 0 else 1.0
        
        assignments = {}
        for node, dist in distances:
            shell = min(int(dist / shell_width), num_shells - 1)
            node.shell_index = shell
            assignments[node.node_id] = shell
        
        return assignments


class AGRMPathBuilder:
    """
    Builds paths between nodes using AGRM principles.
    """
    
    def __init__(self, dimensions: int = 24):
        self.dimensions = dimensions
        self.scanner = AGRMSweepScanner(dimensions)
        self.classifier = AGRMZoneClassifier(dimensions)
    
    def build_path(self, start: AGRMNode, end: AGRMNode,
                   candidates: List[AGRMNode],
                   max_hops: int = 5) -> AGRMRoute:
        """
        Build path from start to end using GR-guided traversal.
        
        Uses midpoint unlocking for complex routes.
        """
        if start.node_id == end.node_id:
            return AGRMRoute([start.node_id], 0.0, [])
        
        # Filter candidates
        relevant = [n for n in candidates 
                   if n.node_id not in (start.node_id, end.node_id)]
        
        # Try direct path first
        direct_dist = start.distance_to(end)
        
        # Find intermediate hops using GR sweep from midpoint
        if len(relevant) > 0 and direct_dist > 0:
            # Midpoint
            midpoint = [
                (start.position[d] + end.position[d]) / 2
                for d in range(self.dimensions)
            ]
            
            # Sweep from midpoint to find good intermediates
            sweep = self.scanner.sweep(relevant, center=midpoint)
            
            # Build path through top candidates
            path = [start.node_id]
            current = start
            total_dist = 0.0
            legs = []
            
            for node, score in sweep.ranked_nodes[:max_hops]:
                if score < 0.3:  # Quality threshold
                    continue
                
                dist = current.distance_to(node)
                legs.append((current.node_id, node.node_id, dist))
                total_dist += dist
                path.append(node.node_id)
                current = node
                
                # Check if close enough to end
                if current.distance_to(end) < direct_dist * 0.5:
                    break
            
            # Final leg to end
            final_dist = current.distance_to(end)
            legs.append((current.node_id, end.node_id, final_dist))
            total_dist += final_dist
            path.append(end.node_id)
            
            # Quality score (lower distance + more hops = better exploration)
            quality = 1.0 / (1.0 + total_dist / max(len(legs), 1))
            
            return AGRMRoute(path, total_dist, legs, quality)
        
        # Direct path
        return AGRMRoute(
            [start.node_id, end.node_id],
            direct_dist,
            [(start.node_id, end.node_id, direct_dist)],
            0.5
        )


class AGRMRouter:
    """
    Main AGRM router for inter-planet communication.
    
    Routes queries between planets using:
    1. GR sweep to rank candidate planets
    2. Zone classification for density-aware routing
    3. Midpoint unlocking for complex distributed queries
    4. Path validation and quality scoring
    """
    
    def __init__(self, dimensions: int = 24):
        self.dimensions = dimensions
        self.scanner = AGRMSweepScanner(dimensions)
        self.classifier = AGRMZoneClassifier(dimensions)
        self.builder = AGRMPathBuilder(dimensions)
        
        self._nodes: Dict[str, AGRMNode] = {}
        self._routes: Dict[Tuple[str, str], AGRMRoute] = {}
    
    def register_node(self, node_id: str, position: List[float],
                     resonance: str, metadata: Dict[str, Any] = None):
        """Register a node (planet) with the router."""
        node = AGRMNode(
            node_id=node_id,
            position=position[:self.dimensions],
            resonance_signature=resonance,
            metadata=metadata or {}
        )
        self._nodes[node_id] = node
        
        # Recompute classifications
        self._reclassify()
    
    def _reclassify(self):
        """Recompute zone classifications for all nodes."""
        nodes = list(self._nodes.values())
        if not nodes:
            return
        
        center = self.scanner._compute_center(nodes)
        max_radius = max(
            math.sqrt(sum((n.position[d] - center[d]) ** 2 
                         for d in range(self.dimensions)))
            for n in nodes
        ) if nodes else 1.0
        
        # Classify each node
        for node in nodes:
            node.density = self.classifier.classify(node, nodes, max_radius * 0.2)
        
        # Assign shells
        self.classifier.assign_shells(nodes, center, num_shells=5)
    
    def find_nearest(self, position: List[float], n: int = 5) -> List[Tuple[AGRMNode, float]]:
        """Find n nearest nodes to position."""
        position = position[:self.dimensions]
        
        distances = []
        for node in self._nodes.values():
            dist = math.sqrt(sum((node.position[d] - position[d]) ** 2 
                                for d in range(self.dimensions)))
            distances.append((node, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:n]
    
    def sweep_from(self, center_node_id: str, 
                   predicate: Optional[Callable[[AGRMNode], bool]] = None) -> SweepResult:
        """
        Perform GR sweep from a center node.
        
        Optionally filter nodes by predicate.
        """
        center_node = self._nodes.get(center_node_id)
        if not center_node:
            return SweepResult([], [0.0] * self.dimensions, 0.0)
        
        candidates = list(self._nodes.values())
        if predicate:
            candidates = [n for n in candidates if predicate(n)]
        
        return self.scanner.sweep(candidates, center_node.position)
    
    def route(self, from_node_id: str, to_node_id: str,
              max_hops: int = 5) -> Optional[AGRMRoute]:
        """
        Find route between two nodes.
        
        Uses cached route if available, otherwise builds new path.
        """
        # Check cache
        cache_key = (from_node_id, to_node_id)
        if cache_key in self._routes:
            return self._routes[cache_key]
        
        start = self._nodes.get(from_node_id)
        end = self._nodes.get(to_node_id)
        
        if not start or not end:
            return None
        
        candidates = [n for n in self._nodes.values() 
                     if n.node_id not in (from_node_id, to_node_id)]
        
        route = self.builder.build_path(start, end, candidates, max_hops)
        
        # Cache
        self._routes[cache_key] = route
        
        return route
    
    def route_query(self, from_node_id: str, 
                    target_resonance: str,
                    threshold: float = 0.7,
                    max_results: int = 3) -> List[Tuple[str, AGRMRoute]]:
        """
        Route a query to nodes with similar resonance.
        
        Uses GR sweep to find candidates, then builds optimal routes.
        """
        from_node = self._nodes.get(from_node_id)
        if not from_node:
            return []
        
        # Find nodes with similar resonance
        def similar_resonance(n: AGRMNode) -> bool:
            # Simple: compare first 8 chars of resonance signature
            return n.resonance_signature[:8] == target_resonance[:8]
        
        sweep = self.sweep_from(from_node_id, similar_resonance)
        
        results = []
        for node, score in sweep.ranked_nodes:
            if score < threshold:
                continue
            
            route = self.route(from_node_id, node.node_id)
            if route:
                results.append((node.node_id, route))
            
            if len(results) >= max_results:
                break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Router statistics."""
        density_counts = {"sparse": 0, "medium": 0, "dense": 0}
        for node in self._nodes.values():
            density_counts[node.density.value] += 1
        
        return {
            "nodes": len(self._nodes),
            "cached_routes": len(self._routes),
            "density_distribution": density_counts,
            "dimensions": self.dimensions
        }
