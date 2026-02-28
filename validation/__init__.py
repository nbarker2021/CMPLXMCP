"""
MCP OS Validation Framework
===========================

Comprehensive validation and testing framework for all MCP OS components.

Usage:
    # Run all validations
    python -m mcp_os.validation.runner --all
    
    # Run specific suites
    python -m mcp_os.validation.runner --universal
    python -m mcp_os.validation.runner --agrm --mdhg
    
    # Get JSON output
    python -m mcp_os.validation.runner --all --json --output results.json
"""

from .system_validator import (
    ValidationRegistry,
    ValidationResult,
    ValidationSuite,
)
from .universal_system_validator import UniversalSystemValidator
from .agrm_mdhg_validator import AGRMMDHGValidator
from .runner import ValidationRunner, main as run_validation

__all__ = [
    "ValidationRegistry",
    "ValidationResult",
    "ValidationSuite",
    "UniversalSystemValidator",
    "AGRMMDHGValidator",
    "ValidationRunner",
    "run_validation",
]

__version__ = "1.0.0"
