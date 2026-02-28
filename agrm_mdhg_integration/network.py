"""
Planet Network
==============
Multi-planet system with ribbon communication.

Ribbons are the communication channels between planets.
AGRM routes queries across the network.
"""

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .planet import Planet, PlanetConfig
from .agrm_router import AGRMRouter, AGRMRoute


@dataclass
class Ribbon:
    """
    A communication channel between two planets.
    
    Ribbons have:
    - Bandwidth (capacity)
    - Latency (delay)
    - Resonance signature (compatibility)
    - Drift tracking (how much data changes)
    """
    ribbon_id: str
    planet_a: str
    planet_b: str
    
    # Metrics
    bandwidth: float = 1.0  # Messages per tick
    latency: float = 0.1    # Delay in seconds
    resonance: float = 1.0  # 0-1 compatibility
    
    # State
    message_count: int = 0
    total_bytes: int = 0
    established_at: float = field(default_factory=lambda: time.time())
    
    def compute_cost(self) -> float:
        """Compute routing cost (lower is better)."""
        # Cost = latency / (bandwidth * resonance)
        if self.bandwidth <= 0 or self.resonance <= 0:
            return float('inf')
        return self.latency / (self.bandwidth * self.resonance)


@dataclass
class NetworkQuery:
    """A query being routed through the network."""
    query_id: str
    origin_planet: str
    target_criteria: Dict[str, Any]
    
    # Routing
    current_planet: Optional[str] = None
    visited_planets: List[str] = field(default_factory=list)
    route: List[str] = field(default_factory=list)
    
    # Results
    results: List[Dict[str, Any]] = field(default_factory=list)
    hops: int = 0
    max_hops: int = 5
    
    # Timing
    started_at: float = field(default_factory=lambda: time.time())
    completed_at: Optional[float] = None


class PlanetNetwork:
    """
    Network of planets with AGRM routing.
    
    Manages:
    - Multiple planets
    - Ribbon connections
    - Cross-planet queries
    - Network-wide statistics
    """
    
    def __init__(self, network_name: str = "cmplx_network"):
        self.network_name = network_name
        self.network_id = f"net_{hashlib.sha256(network_name.encode()).hexdigest()[:12]}"
        
        # Components
        self._planets: Dict[str, Planet] = {}
        self._ribbons: Dict[str, Ribbon] = {}
        self._router = AGRMRouter(dimensions=24)
        
        # Query tracking
        self._queries: Dict[str, NetworkQuery] = {}
        self._completed_queries: List[NetworkQuery] = []
        
        # Metrics
        self._start_time = time.time()
    
    def create_planet(self, config: PlanetConfig) -> Planet:
        """
        Create and register a new planet.
        
        Automatically connects to router.
        """
        planet = Planet(config, router=self._router)
        self._planets[planet.planet_id] = planet
        
        print(f"[Network] Created planet: {planet.name} ({planet.planet_id})")
        return planet
    
    def get_planet(self, planet_id: str) -> Optional[Planet]:
        """Get planet by ID."""
        return self._planets.get(planet_id)
    
    def list_planets(self) -> List[str]:
        """List all planet IDs."""
        return list(self._planets.keys())
    
    def connect_planets(self, planet_a_id: str, planet_b_id: str,
                       bandwidth: float = 1.0, latency: float = 0.1) -> Optional[Ribbon]:
        """
        Create a ribbon (connection) between two planets.
        
        Computes resonance based on planet positions.
        """
        planet_a = self._planets.get(planet_a_id)
        planet_b = self._planets.get(planet_b_id)
        
        if not planet_a or not planet_b:
            return None
        
        # Compute resonance from position similarity
        pos_a = planet_a.config.position
        pos_b = planet_b.config.position
        
        # Cosine similarity of positions
        dot = sum(a * b for a, b in zip(pos_a, pos_b))
        norm_a = sum(a * a for a in pos_a) ** 0.5
        norm_b = sum(b * b for b in pos_b) ** 0.5
        
        if norm_a > 0 and norm_b > 0:
            resonance = 0.5 + 0.5 * (dot / (norm_a * norm_b))  # Normalize to 0.5-1.0
        else:
            resonance = 0.5
        
        # Create ribbon
        ribbon_id = f"ribbon_{min(planet_a_id, planet_b_id)}_{max(planet_a_id, planet_b_id)}"
        
        ribbon = Ribbon(
            ribbon_id=ribbon_id,
            planet_a=planet_a_id,
            planet_b=planet_b_id,
            bandwidth=bandwidth,
            latency=latency,
            resonance=resonance
        )
        
        self._ribbons[ribbon_id] = ribbon
        
        print(f"[Network] Connected {planet_a.name} <-> {planet_b.name} "
              f"(resonance: {resonance:.2f})")
        
        return ribbon
    
    def get_ribbon(self, planet_a_id: str, planet_b_id: str) -> Optional[Ribbon]:
        """Get ribbon between two planets."""
        ribbon_id = f"ribbon_{min(planet_a_id, planet_b_id)}_{max(planet_a_id, planet_b_id)}"
        return self._ribbons.get(ribbon_id)
    
    def route_query(self, from_planet_id: str, 
                   target_resonance: str,
                   threshold: float = 0.7,
                   max_results: int = 10) -> NetworkQuery:
        """
        Route a query from one planet to find similar crystals across network.
        
        Uses AGRM to find optimal route to planets with matching resonance.
        """
        query_id = f"qry_{hashlib.sha256(f'{from_planet_id}:{time.time()}'.encode()).hexdigest()[:12]}"
        
        query = NetworkQuery(
            query_id=query_id,
            origin_planet=from_planet_id,
            target_criteria={"resonance": target_resonance, "threshold": threshold},
            current_planet=from_planet_id,
            max_hops=5
        )
        
        self._queries[query_id] = query
        
        # Start with local query
        origin = self._planets.get(from_planet_id)
        if origin:
            # Create dummy 24D vector from resonance signature
            v24 = [int(target_resonance[i:i+2], 16) / 255.0 for i in range(0, 48, 2)]
            v24 = v24[:24] if len(v24) >= 24 else v24 + [0.5] * (24 - len(v24))
            
            local_results = origin.query_resonance(v24, threshold, max_results)
            query.results.extend([
                {"planet": from_planet_id, **r} for r in local_results
            ])
        
        # Use AGRM to route to other planets
        routes = self._router.route_query(
            from_planet_id,
            target_resonance,
            threshold,
            max_results=3  # Top 3 planets
        )
        
        # Query remote planets
        for planet_id, route in routes:
            if planet_id == from_planet_id:
                continue
            
            planet = self._planets.get(planet_id)
            if not planet:
                continue
            
            # Check ribbon quality
            ribbon = self.get_ribbon(from_planet_id, planet_id)
            if ribbon and ribbon.resonance < 0.3:
                continue  # Skip poor connections
            
            # Query remote planet
            v24 = [int(target_resonance[i:i+2], 16) / 255.0 for i in range(0, 48, 2)]
            v24 = v24[:24] if len(v24) >= 24 else v24 + [0.5] * (24 - len(v24))
            
            remote_results = planet.query_resonance(v24, threshold, max_results // 2)
            query.results.extend([
                {"planet": planet_id, "route_quality": route.quality_score, **r} 
                for r in remote_results
            ])
            
            query.hops += 1
            query.visited_planets.append(planet_id)
            
            if ribbon:
                ribbon.message_count += 1
        
        # Sort by similarity
        query.results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        query.results = query.results[:max_results]
        
        # Complete
        query.completed_at = time.time()
        self._completed_queries.append(query)
        del self._queries[query_id]
        
        return query
    
    def broadcast(self, from_planet_id: str, message: Dict[str, Any],
                 max_hops: int = 3) -> List[str]:
        """
        Broadcast message to all reachable planets.
        
        Returns list of planet IDs that received the message.
        """
        reached = [from_planet_id]
        to_visit = [(from_planet_id, 0)]
        visited = {from_planet_id}
        
        while to_visit:
            current_id, hop_count = to_visit.pop(0)
            
            if hop_count >= max_hops:
                continue
            
            # Find connected planets via ribbons
            for ribbon in self._ribbons.values():
                if ribbon.planet_a == current_id:
                    next_id = ribbon.planet_b
                elif ribbon.planet_b == current_id:
                    next_id = ribbon.planet_a
                else:
                    continue
                
                if next_id not in visited:
                    visited.add(next_id)
                    reached.append(next_id)
                    to_visit.append((next_id, hop_count + 1))
                    
                    # Update ribbon stats
                    ribbon.message_count += 1
        
        return reached
    
    def step_all_dynamics(self):
        """Step CA dynamics on all planets."""
        all_diagnostics = []
        for planet in self._planets.values():
            diagnostics = planet.step_dynamics()
            all_diagnostics.extend([
                {"planet": planet.planet_id, **d} for d in diagnostics
            ])
        return all_diagnostics
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get complete network state."""
        planet_states = {
            pid: planet.get_planet_state()
            for pid, planet in self._planets.items()
        }
        
        ribbon_states = [
            {
                "ribbon_id": r.ribbon_id,
                "planets": [r.planet_a, r.planet_b],
                "bandwidth": r.bandwidth,
                "latency": r.latency,
                "resonance": r.resonance,
                "messages": r.message_count,
                "cost": r.compute_cost()
            }
            for r in self._ribbons.values()
        ]
        
        # Compute network health
        total_health = sum(
            ps["health"].get("trust", 0.5) 
            for ps in planet_states.values()
        ) / max(len(planet_states), 1)
        
        return {
            "network_id": self.network_id,
            "name": self.network_name,
            "planets": planet_states,
            "ribbons": ribbon_states,
            "router_stats": self._router.get_stats(),
            "queries_completed": len(self._completed_queries),
            "uptime": time.time() - self._start_time,
            "network_health": total_health
        }
    
    def find_optimal_path(self, from_planet_id: str, to_planet_id: str) -> Optional[AGRMRoute]:
        """Find optimal path between two planets using AGRM."""
        return self._router.route(from_planet_id, to_planet_id)
    
    def save_snapshot(self, path: Path):
        """Save network snapshot to disk."""
        snapshot = {
            "network_id": self.network_id,
            "name": self.network_name,
            "planets": {
                pid: planet.snapshot()
                for pid, planet in self._planets.items()
            },
            "ribbons": [
                {
                    "ribbon_id": r.ribbon_id,
                    "planet_a": r.planet_a,
                    "planet_b": r.planet_b,
                    "bandwidth": r.bandwidth,
                    "latency": r.latency,
                    "resonance": r.resonance,
                    "message_count": r.message_count
                }
                for r in self._ribbons.values()
            ],
            "queries": [
                {
                    "query_id": q.query_id,
                    "origin": q.origin_planet,
                    "results": len(q.results),
                    "hops": q.hops,
                    "duration": (q.completed_at or time.time()) - q.started_at
                }
                for q in self._completed_queries[-100:]  # Last 100
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"[Network] Snapshot saved to {path}")
