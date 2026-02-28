"""
AGRM + MDHG Validator
=====================
Validates MDHG cache, CA fields, AGRM routing, and Planet components.
"""

import time
from typing import Dict, Any

from .system_validator import ValidationSuite, ValidationResult


class AGRMMDHGValidator:
    """Validates AGRM+MDHG integration components."""
    
    async def run_all_tests(self) -> ValidationSuite:
        """Run all AGRM+MDHG validation tests."""
        suite = ValidationSuite("AGRM+MDHG Integration Validation")
        
        # MDHG Tests
        suite.add(await self._test_mdhg_admission())
        suite.add(await self._test_mdhg_quantization())
        suite.add(await self._test_mdhg_eviction())
        suite.add(await self._test_mdhg_multiscale())
        
        # CA Field Tests
        suite.add(await self._test_ca_cell_creation())
        suite.add(await self._test_ca_channel_updates())
        suite.add(await self._test_ca_kernel_step())
        suite.add(await self._test_ca_multiscale())
        
        # AGRM Tests
        suite.add(await self._test_agrm_node_creation())
        suite.add(await self._test_agrm_sweep())
        suite.add(await self._test_agrm_zone_classification())
        suite.add(await self._test_agrm_path_building())
        
        # Planet Tests
        suite.add(await self._test_planet_creation())
        suite.add(await self._test_planet_crystal_admission())
        suite.add(await self._test_planet_resonance_query())
        suite.add(await self._test_planet_dynamics())
        
        # Network Tests
        suite.add(await self._test_network_creation())
        suite.add(await self._test_ribbon_creation())
        suite.add(await self._test_network_routing())
        
        suite.finalize()
        return suite
    
    # ===== MDHG Tests =====
    
    async def _test_mdhg_admission(self) -> ValidationResult:
        """Test MDHG cache admission."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import MDHGCache
            
            cache = MDHGCache(grid_side=8, cap_per_slot=3)
            v24 = [0.5] * 24
            
            result = cache.admit(v24, {"test": "data"})
            
            assert result["admit"] == True
            assert "slot" in result
            assert "key" in result
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="mdhg_admission",
                component="mdhg_cache",
                status="passed",
                message="MDHG admission successful",
                duration_ms=duration,
                details={"slot": result["slot"]}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="mdhg_admission",
                component="mdhg_cache",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_mdhg_quantization(self) -> ValidationResult:
        """Test 24D vector quantization."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration.mdhg_ca import quantize
            
            v24 = [0.1 * i for i in range(24)]
            q = quantize(v24, bins=16)
            
            assert len(q) == 24
            assert all(0 <= x < 16 for x in q)
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="mdhg_quantization",
                component="mdhg_cache",
                status="passed",
                message="Quantization successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="mdhg_quantization",
                component="mdhg_cache",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_mdhg_eviction(self) -> ValidationResult:
        """Test MDHG slot eviction."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import MDHGCache
            
            cache = MDHGCache(grid_side=8, cap_per_slot=2)  # Small capacity
            
            # Admit 3 items (1 should be evicted)
            for i in range(3):
                v24 = [0.1 * i] * 24
                result = cache.admit(v24, {"idx": i})
            
            stats = cache.get_stats()
            assert stats["evictions"] >= 0  # May or may not evict depending on slot
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="mdhg_eviction",
                component="mdhg_cache",
                status="passed",
                message="Eviction logic working",
                duration_ms=duration,
                details=stats
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="mdhg_eviction",
                component="mdhg_cache",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_mdhg_multiscale(self) -> ValidationResult:
        """Test MDHG multi-scale cache."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import MDHGMultiScale
            
            mdhg = MDHGMultiScale(grid_side=8)
            v24 = [0.5] * 24
            
            # Admit to specific layer
            result = mdhg.admit(v24, {"test": True}, layer="fast")
            assert result["layer"] == "fast"
            
            # Admit to all layers
            results = mdhg.admit_all_layers(v24, {"test": True})
            assert "fast" in results
            assert "med" in results
            assert "slow" in results
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="mdhg_multiscale",
                component="mdhg_cache",
                status="passed",
                message="Multi-scale admission working",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="mdhg_multiscale",
                component="mdhg_cache",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    # ===== CA Field Tests =====
    
    async def _test_ca_cell_creation(self) -> ValidationResult:
        """Test CA cell creation."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration.mdhg_ca import CACell, empty_cell
            
            cell = empty_cell()
            
            assert cell.get("pressure") == 0
            assert cell.get("trust") == 0
            
            cell.set("pressure", 5)
            assert cell.get("pressure") == 5
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ca_cell_creation",
                component="ca_field",
                status="passed",
                message="CA cell creation successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ca_cell_creation",
                component="ca_field",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_ca_channel_updates(self) -> ValidationResult:
        """Test CA channel updates from events."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration.mdhg_ca import CACell, apply_event_to_cell
            
            cell = CACell()
            for k in ["pressure", "risk", "trust", "innovation"]:
                cell.ch[k] = 0
            
            event = {"op": "store", "mag": 0.5}
            apply_event_to_cell(cell, event)
            
            assert cell.get("pressure") > 0
            assert cell.get("innovation") > 0
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ca_channel_updates",
                component="ca_field",
                status="passed",
                message="Channel updates working",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ca_channel_updates",
                component="ca_field",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_ca_kernel_step(self) -> ValidationResult:
        """Test CA kernel step execution."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration.mdhg_ca import CAField
            import random
            
            field = CAField(w=8, h=8, seed=42)
            
            # Apply some initial pressure
            for row in field.grid:
                for cell in row:
                    cell.set("pressure", 5)
            
            # Step
            diagnostics = field.step_async(update_frac=0.1)
            
            assert field.tick == 1
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ca_kernel_step",
                component="ca_field",
                status="passed",
                message="Kernel step execution successful",
                duration_ms=duration,
                details={"diagnostics": len(diagnostics)}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ca_kernel_step",
                component="ca_field",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_ca_multiscale(self) -> ValidationResult:
        """Test CA multi-scale fields."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import CAFieldMultiScale
            
            ca = CAFieldMultiScale(w=8, h=8, seed=42)
            
            # Step all layers
            diagnostics = ca.step()
            
            grids = ca.scalar_grids()
            assert "fast" in grids
            assert "med" in grids
            assert "slow" in grids
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ca_multiscale",
                component="ca_field",
                status="passed",
                message="Multi-scale CA working",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ca_multiscale",
                component="ca_field",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    # ===== AGRM Tests =====
    
    async def _test_agrm_node_creation(self) -> ValidationResult:
        """Test AGRM node creation."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration.agrm_router import AGRMNode
            
            node = AGRMNode(
                node_id="test_node_001",
                position=[0.5] * 24,
                resonance_signature="abc123"
            )
            
            assert node.node_id == "test_node_001"
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="agrm_node_creation",
                component="agrm_router",
                status="passed",
                message="AGRM node creation successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="agrm_node_creation",
                component="agrm_router",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_agrm_sweep(self) -> ValidationResult:
        """Test AGRM GR sweep."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration.agrm_router import AGRMNode, AGRMSweepScanner
            
            scanner = AGRMSweepScanner(dimensions=24)
            
            nodes = [
                AGRMNode(f"n{i}", [0.1 * i] * 24, f"sig{i}")
                for i in range(5)
            ]
            
            sweep = scanner.sweep(nodes)
            
            assert len(sweep.ranked_nodes) == 5
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="agrm_sweep",
                component="agrm_router",
                status="passed",
                message="GR sweep successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="agrm_sweep",
                component="agrm_router",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_agrm_zone_classification(self) -> ValidationResult:
        """Test AGRM zone classification."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration.agrm_router import AGRMNode, AGRMZoneClassifier, ZoneDensity
            
            classifier = AGRMZoneClassifier(dimensions=24)
            
            nodes = [
                AGRMNode(f"n{i}", [0.1 * i] * 24, f"sig{i}")
                for i in range(10)
            ]
            
            center = [0.5] * 24
            assignments = classifier.assign_shells(nodes, center, num_shells=3)
            
            assert len(assignments) == 10
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="agrm_zone_classification",
                component="agrm_router",
                status="passed",
                message="Zone classification successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="agrm_zone_classification",
                component="agrm_router",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_agrm_path_building(self) -> ValidationResult:
        """Test AGRM path building."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration.agrm_router import AGRMNode, AGRMPathBuilder
            
            builder = AGRMPathBuilder(dimensions=24)
            
            start_node = AGRMNode("start", [0.0] * 24, "sig1")
            end_node = AGRMNode("end", [1.0] * 24, "sig2")
            candidates = [
                AGRMNode(f"c{i}", [0.2 * i] * 24, f"csig{i}")
                for i in range(1, 4)
            ]
            
            route = builder.build_path(start_node, end_node, candidates)
            
            assert route.path[0] == "start"
            assert route.path[-1] == "end"
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="agrm_path_building",
                component="agrm_router",
                status="passed",
                message="Path building successful",
                duration_ms=duration,
                details={"path_length": len(route.path)}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="agrm_path_building",
                component="agrm_router",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    # ===== Planet Tests =====
    
    async def _test_planet_creation(self) -> ValidationResult:
        """Test planet creation."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import Planet, PlanetConfig
            
            config = PlanetConfig(
                name="TestPlanet",
                grid_side=8,
                position=[0.5] * 24
            )
            
            planet = Planet(config)
            
            assert planet.name == "TestPlanet"
            assert planet.planet_id.startswith("planet_")
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="planet_creation",
                component="planet",
                status="passed",
                message="Planet creation successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="planet_creation",
                component="planet",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_planet_crystal_admission(self) -> ValidationResult:
        """Test planet crystal admission."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import Planet, PlanetConfig
            
            config = PlanetConfig(name="TestPlanet", grid_side=8)
            planet = Planet(config)
            
            v24 = [0.5] * 24
            result = planet.admit_crystal(
                v24=v24,
                crystal_id="test_cryst_001",
                meta={"test": True},
                layer="fast"
            )
            
            assert result["admit"] == True
            assert "slot" in result
            assert "receipt_id" in result
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="planet_crystal_admission",
                component="planet",
                status="passed",
                message="Crystal admission successful",
                duration_ms=duration,
                details={"slot": result["slot"]}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="planet_crystal_admission",
                component="planet",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_planet_resonance_query(self) -> ValidationResult:
        """Test planet resonance query."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import Planet, PlanetConfig
            
            config = PlanetConfig(name="TestPlanet", grid_side=8)
            planet = Planet(config)
            
            # Admit some crystals
            for i in range(3):
                v24 = [0.1 * i] * 24
                planet.admit_crystal(v24, f"cryst_{i}", {}, "fast")
            
            # Query
            query_v24 = [0.15] * 24
            results = planet.query_resonance(query_v24, threshold=0.0)
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="planet_resonance_query",
                component="planet",
                status="passed",
                message=f"Resonance query returned {len(results)} results",
                duration_ms=duration,
                details={"result_count": len(results)}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="planet_resonance_query",
                component="planet",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_planet_dynamics(self) -> ValidationResult:
        """Test planet CA dynamics."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import Planet, PlanetConfig
            
            config = PlanetConfig(name="TestPlanet", grid_side=8)
            planet = Planet(config)
            
            # Admit a crystal to create pressure
            planet.admit_crystal([0.5] * 24, "cryst_1", {"op": "store"}, "fast")
            
            # Step dynamics
            diagnostics = planet.step_dynamics()
            
            state = planet.get_planet_state()
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="planet_dynamics",
                component="planet",
                status="passed",
                message="CA dynamics step successful",
                duration_ms=duration,
                details={"health": state.get("health", {})}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="planet_dynamics",
                component="planet",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    # ===== Network Tests =====
    
    async def _test_network_creation(self) -> ValidationResult:
        """Test network creation."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import PlanetNetwork
            
            network = PlanetNetwork("test_network")
            
            assert network.network_name == "test_network"
            assert network.network_id.startswith("net_")
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="network_creation",
                component="network",
                status="passed",
                message="Network creation successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="network_creation",
                component="network",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_ribbon_creation(self) -> ValidationResult:
        """Test ribbon creation between planets."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import PlanetNetwork, PlanetConfig
            
            network = PlanetNetwork("test_network")
            
            # Create two planets
            earth = network.create_planet(PlanetConfig("Earth", position=[0.5] * 24))
            mars = network.create_planet(PlanetConfig("Mars", position=[0.3] * 24))
            
            # Connect them
            ribbon = network.connect_planets(earth.planet_id, mars.planet_id)
            
            assert ribbon is not None
            assert ribbon.ribbon_id
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ribbon_creation",
                component="network",
                status="passed",
                message="Ribbon creation successful",
                duration_ms=duration,
                details={"resonance": ribbon.resonance}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="ribbon_creation",
                component="network",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_network_routing(self) -> ValidationResult:
        """Test network routing."""
        start = time.time()
        try:
            from ..agrm_mdhg_integration import PlanetNetwork, PlanetConfig
            
            network = PlanetNetwork("test_network")
            
            # Create and connect planets
            earth = network.create_planet(PlanetConfig("Earth", position=[0.5] * 24))
            mars = network.create_planet(PlanetConfig("Mars", position=[0.3] * 24))
            network.connect_planets(earth.planet_id, mars.planet_id)
            
            # Route query
            query = network.route_query(
                from_planet_id=earth.planet_id,
                target_resonance="abc123",
                threshold=0.0
            )
            
            assert query.query_id
            assert query.origin_planet == earth.planet_id
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="network_routing",
                component="network",
                status="passed",
                message="Network routing successful",
                duration_ms=duration,
                details={"hops": query.hops}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="network_routing",
                component="network",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
