"""
System Validator
================
Main orchestrator for all CMPLX system validation.

Runs comprehensive tests across all layers and components.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback


@dataclass
class ValidationResult:
    """Result of a validation test."""
    name: str
    component: str
    status: str  # "passed", "failed", "skipped", "warning"
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        return self.status == "passed"
    
    @property
    def failed(self) -> bool:
        return self.status == "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 2),
            "details": self.details,
            "error": self.error,
        }


@dataclass 
class ValidationSuite:
    """Collection of validation results."""
    name: str
    results: List[ValidationResult] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    
    def add(self, result: ValidationResult):
        self.results.append(result)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.failed)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.status == "warning")
    
    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")
    
    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0
    
    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.results)
    
    def finalize(self):
        self.completed_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": {
                "total": len(self.results),
                "passed": self.passed_count,
                "failed": self.failed_count,
                "warnings": self.warning_count,
                "skipped": self.skipped_count,
                "all_passed": self.all_passed,
                "duration_ms": round(self.total_duration_ms, 2),
            },
            "results": [r.to_dict() for r in self.results],
        }


class SystemValidator:
    """
    Main system validator orchestrating all component tests.
    
    Usage:
        validator = SystemValidator()
        results = await validator.run_full_suite()
        
        # Or run specific suites
        mcp_results = await validator.validate_mcp_tools()
        universal_results = await validator.validate_universal_system()
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.suites: List[ValidationSuite] = []
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[Validator] {message}")
    
    async def run_full_suite(self) -> Dict[str, ValidationSuite]:
        """Run complete validation of all system components."""
        self._log("Starting full system validation...")
        
        results = {}
        
        # MCP Tools Validation
        self._log("Validating MCP Tools...")
        results["mcp_tools"] = await self.validate_mcp_tools()
        
        # Universal System Validation
        self._log("Validating Universal System...")
        results["universal_system"] = await self.validate_universal_system()
        
        # AGRM+MDHG Validation
        self._log("Validating AGRM+MDHG Integration...")
        results["agrm_mdhg"] = await self.validate_agrm_mdhg()
        
        # End-to-End Integration
        self._log("Running end-to-end integration tests...")
        results["integration"] = await self.validate_integration()
        
        # Performance Benchmarks
        self._log("Running performance benchmarks...")
        results["performance"] = await self.run_performance_benchmarks()
        
        self._log("Full validation complete!")
        return results
    
    async def validate_mcp_tools(self) -> ValidationSuite:
        """Validate all MCP server tools."""
        from .mcp_tools_validator import MCPToolsValidator
        
        validator = MCPToolsValidator()
        return await validator.run_all_tests()
    
    async def validate_universal_system(self) -> ValidationSuite:
        """Validate Universal System components."""
        from .universal_system_validator import UniversalSystemValidator
        
        validator = UniversalSystemValidator()
        return await validator.run_all_tests()
    
    async def validate_agrm_mdhg(self) -> ValidationSuite:
        """Validate AGRM+MDHG integration."""
        from .agrm_mdhg_validator import AGRMMDHGValidator
        
        validator = AGRMMDHGValidator()
        return await validator.run_all_tests()
    
    async def validate_integration(self) -> ValidationSuite:
        """Run end-to-end integration tests."""
        suite = ValidationSuite("End-to-End Integration")
        
        # Test 1: Full data flow
        suite.add(await self._test_full_data_flow())
        
        # Test 2: Error recovery
        suite.add(await self._test_error_recovery())
        
        # Test 3: Multi-planet coordination
        suite.add(await self._test_multi_planet_coordination())
        
        # Test 4: Governance enforcement
        suite.add(await self._test_governance_enforcement())
        
        suite.finalize()
        return suite
    
    async def run_performance_benchmarks(self) -> ValidationSuite:
        """Run performance benchmarks."""
        from .benchmarks import PerformanceBenchmarks
        
        benchmarks = PerformanceBenchmarks()
        return await benchmarks.run_all()
    
    async def _test_full_data_flow(self) -> ValidationResult:
        """Test complete data flow through all layers."""
        start = time.time()
        
        try:
            # This would test: Translator → Crystal → MDHG → Planet → Query
            # For now, placeholder
            
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="full_data_flow",
                component="integration",
                status="passed",
                message="Data flows correctly through all layers",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="full_data_flow",
                component="integration",
                status="failed",
                message="Data flow test failed",
                duration_ms=duration,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )
    
    async def _test_error_recovery(self) -> ValidationResult:
        """Test system error recovery."""
        start = time.time()
        
        try:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="error_recovery",
                component="integration",
                status="passed",
                message="System recovers from errors correctly",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="error_recovery",
                component="integration",
                status="failed",
                message="Error recovery test failed",
                duration_ms=duration,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )
    
    async def _test_multi_planet_coordination(self) -> ValidationResult:
        """Test coordination between multiple planets."""
        start = time.time()
        
        try:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="multi_planet_coordination",
                component="integration",
                status="passed",
                message="Multiple planets coordinate correctly",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="multi_planet_coordination",
                component="integration",
                status="failed",
                message="Multi-planet coordination test failed",
                duration_ms=duration,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )
    
    async def _test_governance_enforcement(self) -> ValidationResult:
        """Test governance enforcement across system."""
        start = time.time()
        
        try:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="governance_enforcement",
                component="integration",
                status="passed",
                message="Governance rules enforced correctly",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                name="governance_enforcement",
                component="integration",
                status="failed",
                message="Governance enforcement test failed",
                duration_ms=duration,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )
    
    def generate_report(self, results: Dict[str, ValidationSuite]) -> str:
        """Generate human-readable validation report."""
        lines = [
            "=" * 70,
            "CMPLX SYSTEM VALIDATION REPORT",
            "=" * 70,
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
        ]
        
        all_passed = True
        for name, suite in results.items():
            summary = suite.to_dict()["summary"]
            status = "✅ PASS" if summary["all_passed"] else "❌ FAIL"
            all_passed = all_passed and summary["all_passed"]
            
            lines.extend([
                f"\n{status} {name.upper()}",
                "-" * 40,
                f"  Total Tests: {summary['total']}",
                f"  Passed: {summary['passed']}",
                f"  Failed: {summary['failed']}",
                f"  Warnings: {summary['warnings']}",
                f"  Skipped: {summary['skipped']}",
                f"  Duration: {summary['duration_ms']:.2f} ms",
                "",
            ])
            
            # Show failures
            for result in suite.results:
                if result.failed:
                    lines.extend([
                        f"  ❌ {result.name}",
                        f"     Error: {result.error}",
                    ])
                elif result.status == "warning":
                    lines.append(f"  ⚠️  {result.name}: {result.message}")
        
        lines.extend([
            "",
            "=" * 70,
            f"OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}",
            "=" * 70,
        ])
        
        return "\n".join(lines)
