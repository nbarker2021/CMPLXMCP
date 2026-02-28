"""
CMPLX MCP OS
============
A Model Context Protocol-based operating system for CMPLX.

Heavy data and computation happen server-side.
Local runtime is lightweight with only handles and adapters.

Usage:
    from mcp_os import create_client, ProxyRegistry
    
    async with create_client() as client:
        registry = ProxyRegistry(client)
        
        # Use any layer
        e8_result = await registry.l2.project_e8([1,2,3,4,5,6,7,8])
        dr = await registry.l4.digital_root(432)
"""

# Apply constants fix BEFORE any other imports
import sys
from pathlib import Path

# Import and apply constants fix
from . import constants_fix

__version__ = "1.0.0"

from .client import CMPLXClient, create_client
from .controllers import ProxyRegistry

__all__ = ["CMPLXClient", "create_client", "ProxyRegistry"]
