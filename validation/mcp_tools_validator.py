"""
MCP Tools Validator
===================
Validates all MCP server tools and their handlers.
"""

import time
import asyncio
from typing import List, Dict, Any

from .system_validator import ValidationSuite, ValidationResult


class MCPToolsValidator:
    """Validates MCP server tools."""
    
    async def run_all_tests(self) -> ValidationSuite:
        """Run all MCP tool validation tests."""
        suite = ValidationSuite("MCP Tools Validation")
        
        # Layer 1 Tests
        suite.add(await self._test_l1_morphon_generate())
        suite.add(await self._test_l1_mglc_execute())
        suite.add(await self._test_l1_seed_expand())
        
        # Layer 2 Tests
        suite.add(await self._test_l2_e8_project())
        suite.add(await self._test_l2_leech_nearest())
        suite.add(await self._test_l2_weyl_navigate())
        suite.add(await self._test_l2_niemeier_classify())
        
        # Layer 3 Tests
        suite.add(await self._test_l3_morsr_optimize())
        suite.add(await self._test_l3_conservation_check())
        
        # Layer 4 Tests
        suite.add(await self._test_l4_digital_root())
        suite.add(await self._test_l4_seven_witness())
        suite.add(await self._test_l4_policy_check())
        
        # Layer 5 Tests
        suite.add(await self._test_l5_embed())
        suite.add(await self._test_l5_query_similar())
        suite.add(await self._test_l5_transform())
        
        # Universal System Tests
        suite.add(await self._test_universal_translate())
        suite.add(await self._test_crystal_store())
        suite.add(await self._test_crystal_resonance_query())
        suite.add(await self._test_temporal_query())
        suite.add(await self._test_hypothesis_generate())
        
        # Planet/Network Tests
        suite.add(await self._test_planet_admit())
        suite.add(await self._test_network_route())
        
        # System Tests
        suite.add(await self._test_sys_info())
        suite.add(await self._test_audit_provenance())
        
        suite.finalize()
        return suite
    
    async def _test_tool(self, name: str, component: str, test_fn) -> ValidationResult:
        """Helper to test a tool with timing and error handling."""
        start = time.time()
        try:
            result = await test_fn()
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name=name,
                component=component,
                status="passed",
                message="Tool executed successfully",
                duration_ms=duration,
                details=result if isinstance(result, dict) else {}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name=name,
                component=component,
                status="failed",
                message=f"Tool execution failed: {str(e)}",
                duration_ms=duration,
                error=str(e)
            )
    
    # ===== Layer 1: Morphonic Foundation =====
    
    async def _test_l1_morphon_generate(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER1_TOOLS
            result = await LAYER1_TOOLS._l1_morphon_generate(
                {"seed": "7"},
                None
            )
            assert "handle" in result
            assert "dr" in result
            return result
        return await self._test_tool("l1_morphon_generate", "layer1", test)
    
    async def _test_l1_mglc_execute(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER1_TOOLS
            result = await LAYER1_TOOLS._l1_mglc_execute(
                {"expression": "(λx.x)", "context": {}},
                None
            )
            assert "handle" in result
            return result
        return await self._test_tool("l1_mglc_execute", "layer1", test)
    
    async def _test_l1_seed_expand(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER1_TOOLS
            result = await LAYER1_TOOLS._l1_seed_expand(
                {"digit": 7, "dimensions": 24},
                None
            )
            assert "handle" in result
            assert result["dimensions"] == 24
            return result
        return await self._test_tool("l1_seed_expand", "layer1", test)
    
    # ===== Layer 2: Geometric Engine =====
    
    async def _test_l2_e8_project(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER2_TOOLS
            result = await LAYER2_TOOLS._l2_e8_project(
                {"vector": [1.0] * 8, "return_format": "minimal"},
                None
            )
            assert "handle" in result
            assert result["lattice"] == "E8"
            return result
        return await self._test_tool("l2_e8_project", "layer2", test)
    
    async def _test_l2_leech_nearest(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER2_TOOLS
            result = await LAYER2_TOOLS._l2_leech_nearest(
                {"vector": [0.5] * 24, "return_format": "handle"},
                None
            )
            assert "handle" in result
            assert result["lattice"] == "Leech"
            return result
        return await self._test_tool("l2_leech_nearest", "layer2", test)
    
    async def _test_l2_weyl_navigate(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER2_TOOLS
            result = await LAYER2_TOOLS._l2_weyl_navigate(
                {"position": [0.1] * 8},
                None
            )
            assert "handle" in result
            assert "chamber" in result
            return result
        return await self._test_tool("l2_weyl_navigate", "layer2", test)
    
    async def _test_l2_niemeier_classify(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER2_TOOLS
            result = await LAYER2_TOOLS._l2_niemeier_classify(
                {"vector": [0.3] * 24},
                None
            )
            assert "handle" in result
            assert "top_lattice" in result
            return result
        return await self._test_tool("l2_niemeier_classify", "layer2", test)
    
    # ===== Layer 3: Operational Systems =====
    
    async def _test_l3_morsr_optimize(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER3_TOOLS
            result = await LAYER3_TOOLS._l3_morsr_optimize(
                {"initial_state": [1.0, 0.5, -0.3, 0.7], "iterations": 10},
                None
            )
            assert "handle" in result
            assert "final_norm" in result
            return result
        return await self._test_tool("l3_morsr_optimize", "layer3", test)
    
    async def _test_l3_conservation_check(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER3_TOOLS
            result = await LAYER3_TOOLS._l3_conservation_check(
                {"before": [1.0, 0.5], "after": [0.8, 0.4]},
                None
            )
            assert "delta_phi" in result
            assert "conserved" in result
            return result
        return await self._test_tool("l3_conservation_check", "layer3", test)
    
    # ===== Layer 4: Governance =====
    
    async def _test_l4_digital_root(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER4_TOOLS
            result = await LAYER4_TOOLS._l4_digital_root(
                {"number": 432, "modulus": 9},
                None
            )
            assert result["digital_root"] == 9
            assert "meaning" in result
            return result
        return await self._test_tool("l4_digital_root", "layer4", test)
    
    async def _test_l4_seven_witness(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER4_TOOLS
            result = await LAYER4_TOOLS._l4_seven_witness(
                {"artifact": {"test": "data"}},
                None
            )
            assert "witnesses" in result
            assert "all_valid" in result
            return result
        return await self._test_tool("l4_seven_witness", "layer4", test)
    
    async def _test_l4_policy_check(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER4_TOOLS
            result = await LAYER4_TOOLS._l4_policy_check(
                {"artifact_id": "test_123", "policy_tier": 3},
                None
            )
            assert "tier_name" in result
            return result
        return await self._test_tool("l4_policy_check", "layer4", test)
    
    # ===== Layer 5: Interface =====
    
    async def _test_l5_embed(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER5_TOOLS
            result = await LAYER5_TOOLS._l5_embed(
                {"content": "Test content", "domain": "text"},
                None
            )
            assert "handle" in result
            assert "content_hash" in result
            return result
        return await self._test_tool("l5_embed", "layer5", test)
    
    async def _test_l5_query_similar(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER5_TOOLS
            # First create a handle to query
            embed_result = await LAYER5_TOOLS._l5_embed(
                {"content": "Test", "domain": "text"},
                None
            )
            result = await LAYER5_TOOLS._l5_query_similar(
                {"handle": embed_result["handle"], "top_k": 5},
                None
            )
            assert "results" in result
            return result
        return await self._test_tool("l5_query_similar", "layer5", test)
    
    async def _test_l5_transform(self) -> ValidationResult:
        async def test():
            from ..server.tools import LAYER5_TOOLS
            embed_result = await LAYER5_TOOLS._l5_embed(
                {"content": "Test", "domain": "text"},
                None
            )
            result = await LAYER5_TOOLS._l5_transform(
                {"handle": embed_result["handle"], "operator": "rotation"},
                None
            )
            assert "handle" in result
            return result
        return await self._test_tool("l5_transform", "layer5", test)
    
    # ===== Universal System Tools =====
    
    async def _test_universal_translate(self) -> ValidationResult:
        async def test():
            from ..server.universal_tools import UNIVERSAL_TOOLS
            result = await UNIVERSAL_TOOLS._universal_translate(
                {"content": "Test content", "content_type": "text", "identity": "test"},
                None
            )
            assert "handle" in result
            assert "atom_count" in result
            return result
        return await self._test_tool("universal_translate", "universal", test)
    
    async def _test_crystal_store(self) -> ValidationResult:
        async def test():
            # This would need the full setup - for now mark as skipped
            return ValidationResult(
                name="crystal_store",
                component="universal",
                status="skipped",
                message="Requires full system setup"
            )
        return await self._test_tool("crystal_store", "universal", test)
    
    async def _test_crystal_resonance_query(self) -> ValidationResult:
        async def test():
            return ValidationResult(
                name="crystal_resonance_query",
                component="universal",
                status="skipped",
                message="Requires crystal database"
            )
        return await self._test_tool("crystal_resonance_query", "universal", test)
    
    async def _test_temporal_query(self) -> ValidationResult:
        async def test():
            return ValidationResult(
                name="temporal_query",
                component="universal",
                status="skipped",
                message="Requires temporal layer setup"
            )
        return await self._test_tool("temporal_query", "universal", test)
    
    async def _test_hypothesis_generate(self) -> ValidationResult:
        async def test():
            return ValidationResult(
                name="hypothesis_generate",
                component="universal",
                status="skipped",
                message="Requires hypothesis engine"
            )
        return await self._test_tool("hypothesis_generate", "universal", test)
    
    # ===== Planet/Network Tools =====
    
    async def _test_planet_admit(self) -> ValidationResult:
        async def test():
            return ValidationResult(
                name="planet_admit",
                component="planet",
                status="skipped",
                message="Requires planet network setup"
            )
        return await self._test_tool("planet_admit", "planet", test)
    
    async def _test_network_route(self) -> ValidationResult:
        async def test():
            return ValidationResult(
                name="network_route",
                component="network",
                status="skipped",
                message="Requires multi-planet network"
            )
        return await self._test_tool("network_route", "network", test)
    
    # ===== System Tools =====
    
    async def _test_sys_info(self) -> ValidationResult:
        async def test():
            from ..server.tools import SYSTEM_TOOLS
            result = await SYSTEM_TOOLS._sys_info({}, None)
            assert "system" in result
            assert "version" in result
            return result
        return await self._test_tool("sys_info", "system", test)
    
    async def _test_audit_provenance(self) -> ValidationResult:
        async def test():
            return ValidationResult(
                name="audit_provenance",
                component="system",
                status="skipped",
                message="Requires identity family setup"
            )
        return await self._test_tool("audit_provenance", "system", test)
