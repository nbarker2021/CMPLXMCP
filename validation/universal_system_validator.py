"""
Universal System Validator
==========================
Validates Universal Translator, Crystal Storage, and related components.
"""

import time
from typing import Dict, Any

from .system_validator import ValidationSuite, ValidationResult


class UniversalSystemValidator:
    """Validates Universal System components."""
    
    async def run_all_tests(self) -> ValidationSuite:
        """Run all Universal System validation tests."""
        suite = ValidationSuite("Universal System Validation")
        
        # Translator Tests
        suite.add(await self._test_translator_text())
        suite.add(await self._test_translator_code())
        suite.add(await self._test_translator_math())
        suite.add(await self._test_translator_data())
        
        # Crystal Tests
        suite.add(await self._test_crystal_creation())
        suite.add(await self._test_crystal_resonance())
        suite.add(await self._test_crystal_merge())
        
        # SNAP Tests
        suite.add(await self._test_snap_transaction())
        suite.add(await self._test_snap_chain())
        
        # Temporal Tests
        suite.add(await self._test_temporal_coordinate())
        suite.add(await self._test_hypothesis_generation())
        suite.add(await self._test_memory_creation())
        
        # Identity Tests
        suite.add(await self._test_identity_registration())
        suite.add(await self._test_receipt_generation())
        suite.add(await self._test_provenance_audit())
        
        suite.finalize()
        return suite
    
    async def _test_translator_text(self) -> ValidationResult:
        """Test text translation to geometric form."""
        start = time.time()
        try:
            from ..universal import UniversalTranslator
            
            translator = UniversalTranslator()
            form = await translator.translate(
                "Quantum consciousness",
                content_type="text",
                identity="test"
            )
            
            assert len(form.atoms) > 0
            assert len(form.bonds) > 0
            assert form.symmetry_signature
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="translator_text",
                component="universal_translator",
                status="passed",
                message="Text translation successful",
                duration_ms=duration,
                details={"atom_count": len(form.atoms)}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="translator_text",
                component="universal_translator",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_translator_code(self) -> ValidationResult:
        """Test code translation."""
        start = time.time()
        try:
            from ..universal import UniversalTranslator
            
            translator = UniversalTranslator()
            code = "def hello(): return 'world'"
            form = await translator.translate(code, content_type="code")
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="translator_code",
                component="universal_translator",
                status="passed",
                message="Code translation successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="translator_code",
                component="universal_translator",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_translator_math(self) -> ValidationResult:
        """Test math expression translation."""
        start = time.time()
        try:
            from ..universal import UniversalTranslator
            
            translator = UniversalTranslator()
            math = "E = mc^2"
            form = await translator.translate(math, content_type="math")
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="translator_math",
                component="universal_translator",
                status="passed",
                message="Math translation successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="translator_math",
                component="universal_translator",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_translator_data(self) -> ValidationResult:
        """Test data structure translation."""
        start = time.time()
        try:
            from ..universal import UniversalTranslator
            
            translator = UniversalTranslator()
            data = {"key": "value", "nested": {"x": 1}}
            form = await translator.translate(data, content_type="json")
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="translator_data",
                component="universal_translator",
                status="passed",
                message="Data translation successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="translator_data",
                component="universal_translator",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_crystal_creation(self) -> ValidationResult:
        """Test crystal creation from geometric form."""
        start = time.time()
        try:
            from ..universal import Crystal, GeometricForm, SNAPAtom
            
            # Create a simple geometric form
            atoms = [SNAPAtom(
                identity="test_atom",
                morphon_seed=7,
                position=[0.5] * 24,
                charge=0.8,
                content="test",
                atom_type="test"
            )]
            
            form = GeometricForm(
                atoms=atoms,
                bonds=[],
                envelope={"test": True}
            )
            
            crystal = Crystal(
                crystal_id="test_cryst_001",
                name="Test Crystal",
                atoms=atoms,
                bonds=[],
                temporal_phase="present"
            )
            
            assert crystal.resonance_signature
            assert crystal.crystal_id == "test_cryst_001"
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="crystal_creation",
                component="crystal_storage",
                status="passed",
                message="Crystal creation successful",
                duration_ms=duration,
                details={"resonance_sig": crystal.resonance_signature[:16] + "..."}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="crystal_creation",
                component="crystal_storage",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_crystal_resonance(self) -> ValidationResult:
        """Test crystal resonance calculation."""
        start = time.time()
        try:
            from ..universal import Crystal, SNAPAtom
            
            # Create two similar crystals
            atoms1 = [SNAPAtom("a", 7, [0.5]*24, 0.8, "x", "test")]
            atoms2 = [SNAPAtom("b", 7, [0.51]*24, 0.81, "y", "test")]
            
            c1 = Crystal("c1", "C1", atoms=atoms1, bonds=[])
            c2 = Crystal("c2", "C2", atoms=atoms2, bonds=[])
            
            resonance = c1.resonance_with(c2)
            
            assert 0 <= resonance <= 1
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="crystal_resonance",
                component="crystal_storage",
                status="passed",
                message=f"Resonance calculated: {resonance:.3f}",
                duration_ms=duration,
                details={"resonance": resonance}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="crystal_resonance",
                component="crystal_storage",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_crystal_merge(self) -> ValidationResult:
        """Test crystal merging."""
        start = time.time()
        try:
            from ..universal.crystal import Crystal, CrystalFactory, SNAPAtom
            
            atoms1 = [SNAPAtom("a", 7, [0.5]*24, 0.8, "x", "test")]
            atoms2 = [SNAPAtom("b", 8, [0.6]*24, 0.7, "y", "test")]
            
            c1 = Crystal("c1", "C1", atoms=atoms1, bonds=[])
            c2 = Crystal("c2", "C2", atoms=atoms2, bonds=[])
            
            merged = CrystalFactory.merge([c1, c2], "Merged")
            
            assert len(merged.atoms) == 2
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="crystal_merge",
                component="crystal_storage",
                status="passed",
                message="Crystal merge successful",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="crystal_merge",
                component="crystal_storage",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_snap_transaction(self) -> ValidationResult:
        """Test SNAP transaction creation."""
        start = time.time()
        try:
            from ..universal import SNAPTransaction
            
            tx = SNAPTransaction(
                action_type="test_action",
                identity="test_user",
                output_handle="test_output"
            )
            
            assert tx.tx_id
            assert tx.receipt_hash
            assert tx.verify()
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="snap_transaction",
                component="snap_ledger",
                status="passed",
                message="SNAP transaction created and verified",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="snap_transaction",
                component="snap_ledger",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_snap_chain(self) -> ValidationResult:
        """Test SNAP chain operations."""
        start = time.time()
        try:
            from ..universal import SNAPChain, SNAPTransaction
            
            chain = SNAPChain()
            
            tx1 = SNAPTransaction(action_type="action1", identity="user1")
            tx2 = SNAPTransaction(action_type="action2", identity="user2")
            
            assert chain.add(tx1)
            assert chain.add(tx2)
            assert chain.get(tx1.tx_id) == tx1
            
            stats = chain.stats()
            assert stats["total_transactions"] == 2
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="snap_chain",
                component="snap_ledger",
                status="passed",
                message="SNAP chain operations successful",
                duration_ms=duration,
                details=stats
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="snap_chain",
                component="snap_ledger",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_temporal_coordinate(self) -> ValidationResult:
        """Test temporal coordinate creation."""
        start = time.time()
        try:
            from ..universal import TemporalCoordinate
            
            coord = TemporalCoordinate(
                timestamp="2026-02-20T10:00:00",
                phase="present",
                certainty=1.0
            )
            
            assert coord.phase == "present"
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="temporal_coordinate",
                component="temporal_layer",
                status="passed",
                message="Temporal coordinate created",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="temporal_coordinate",
                component="temporal_layer",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_hypothesis_generation(self) -> ValidationResult:
        """Test hypothesis generation."""
        start = time.time()
        try:
            # This would need the full temporal layer setup
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="hypothesis_generation",
                component="temporal_layer",
                status="skipped",
                message="Requires full temporal layer setup",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="hypothesis_generation",
                component="temporal_layer",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_memory_creation(self) -> ValidationResult:
        """Test memory creation."""
        start = time.time()
        try:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="memory_creation",
                component="temporal_layer",
                status="skipped",
                message="Requires temporal layer setup",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="memory_creation",
                component="temporal_layer",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_identity_registration(self) -> ValidationResult:
        """Test identity registration."""
        start = time.time()
        try:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="identity_registration",
                component="identity_family",
                status="skipped",
                message="Requires identity family setup",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="identity_registration",
                component="identity_family",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_receipt_generation(self) -> ValidationResult:
        """Test receipt generation."""
        start = time.time()
        try:
            from ..universal.identity_family import SpeedlightReceipt
            
            receipt = SpeedlightReceipt(
                receipt_id="test_rcpt_001",
                tx_id="test_tx_001",
                input_hash="abc123",
                output_hash="def456",
                action_hash="ghi789"
            )
            
            assert receipt.verify()
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="receipt_generation",
                component="identity_family",
                status="passed",
                message="Receipt created and verified",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="receipt_generation",
                component="identity_family",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
    
    async def _test_provenance_audit(self) -> ValidationResult:
        """Test provenance audit."""
        start = time.time()
        try:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="provenance_audit",
                component="identity_family",
                status="skipped",
                message="Requires full identity family setup",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="provenance_audit",
                component="identity_family",
                status="failed",
                message=str(e),
                duration_ms=duration,
                error=str(e)
            )
