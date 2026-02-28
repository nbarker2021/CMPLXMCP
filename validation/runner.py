"""
Validation Runner CLI
=====================
Command-line interface for running MCP OS validation suites.

Usage:
    python -m mcp_os.validation.runner --all
    python -m mcp_os.validation.runner --universal --mdhg --agrm
    python -m mcp_os.validation.runner --mcp --quick

Options:
    --all           Run all validation suites
    --universal     Run Universal System validation
    --mdhg          Run MDHG validation
    --agrm          Run AGRM validation  
    --planet        Run Planet validation
    --network       Run Network validation
    --mcp           Run MCP server/client validation
    --quick         Run quick smoke tests only
    --verbose       Verbose output
    --json          Output as JSON
    --output FILE   Save results to file
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import List, Optional

from .system_validator import ValidationRegistry, ValidationSuite
from .universal_system_validator import UniversalSystemValidator
from .agrm_mdhg_validator import AGRMMDHGValidator


class ValidationRunner:
    """Main validation runner."""
    
    def __init__(self):
        self.registry = ValidationRegistry()
        self.results: List[ValidationSuite] = []
        self.verbose = False
    
    async def run_all(self) -> ValidationSuite:
        """Run all validation suites."""
        master_suite = ValidationSuite("MCP OS Complete Validation")
        
        # Universal System
        universal = UniversalSystemValidator()
        universal_suite = await universal.run_all_tests()
        master_suite.merge(universal_suite)
        self.results.append(universal_suite)
        
        # AGRM + MDHG
        agrm_mdhg = AGRMMDHGValidator()
        agrm_suite = await agrm_mdhg.run_all_tests()
        master_suite.merge(agrm_suite)
        self.results.append(agrm_suite)
        
        master_suite.finalize()
        return master_suite
    
    async def run_universal(self) -> ValidationSuite:
        """Run Universal System validation only."""
        validator = UniversalSystemValidator()
        suite = await validator.run_all_tests()
        self.results.append(suite)
        return suite
    
    async def run_agrm_mdhg(self) -> ValidationSuite:
        """Run AGRM+MDHG validation only."""
        validator = AGRMMDHGValidator()
        suite = await validator.run_all_tests()
        self.results.append(suite)
        return suite
    
    def print_results(self, suite: ValidationSuite, json_output: bool = False):
        """Print validation results."""
        if json_output:
            print(json.dumps(suite.to_dict(), indent=2))
            return
        
        print()
        print("=" * 80)
        print(f"  {suite.name}")
        print("=" * 80)
        print()
        
        # Group by component
        by_component = {}
        for result in suite.results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)
        
        for component, results in sorted(by_component.items()):
            print(f"\n📦 {component.upper()}")
            print("-" * 40)
            
            for r in results:
                icon = "✅" if r.status == "passed" else ("⚠️" if r.status == "skipped" else "❌")
                print(f"  {icon} {r.name}: {r.message}")
                
                if self.verbose and r.details:
                    for k, v in r.details.items():
                        print(f"     {k}: {v}")
                
                if r.status == "failed" and r.error:
                    print(f"     Error: {r.error[:100]}...")
        
        print()
        print("=" * 80)
        print(f"  SUMMARY")
        print("=" * 80)
        print(f"  Total:    {suite.total}")
        print(f"  Passed:   {suite.passed} ✅")
        print(f"  Failed:   {suite.failed} ❌")
        print(f"  Skipped:  {suite.skipped} ⚠️")
        print(f"  Duration: {suite.duration_ms:.2f} ms")
        print()
        
        if suite.failed > 0:
            print(f"  Status: FAILED - {suite.failed} test(s) failed")
        else:
            print(f"  Status: PASSED - All tests passed!")
        
        print("=" * 80)
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MCP OS Validation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mcp_os.validation.runner --all
  python -m mcp_os.validation.runner --universal --verbose
  python -m mcp_os.validation.runner --agrm --mdhg --json --output results.json
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run all validation suites")
    parser.add_argument("--universal", action="store_true", help="Run Universal System validation")
    parser.add_argument("--mdhg", action="store_true", help="Run MDHG validation")
    parser.add_argument("--agrm", action="store_true", help="Run AGRM validation")
    parser.add_argument("--planet", action="store_true", help="Run Planet validation")
    parser.add_argument("--network", action="store_true", help="Run Network validation")
    parser.add_argument("--mcp", action="store_true", help="Run MCP server validation")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, help="Save results to file")
    
    args = parser.parse_args()
    
    # If no specific tests selected, default to --all
    if not any([args.all, args.universal, args.mdhg, args.agrm, 
                args.planet, args.network, args.mcp]):
        args.all = True
    
    runner = ValidationRunner()
    runner.verbose = args.verbose
    
    async def run():
        try:
            if args.all:
                suite = await runner.run_all()
            elif args.universal:
                suite = await runner.run_universal()
            elif args.mdhg or args.agrm or args.planet or args.network:
                suite = await runner.run_agrm_mdhg()
            else:
                print("No validation suite selected.")
                return 1
            
            runner.print_results(suite, json_output=args.json)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(suite.to_dict(), f, indent=2)
                print(f"Results saved to: {args.output}")
            
            return 0 if suite.failed == 0 else 1
            
        except Exception as e:
            print(f"Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
