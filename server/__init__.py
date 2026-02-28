"""
CMPLX MCP OS Server
===================
The central MCP server that exposes all CMPLX tools as callable functions.
Organized by layers matching the controller hierarchy.

Heavy data operations happen here. Local clients get lightweight responses.
"""

__version__ = "1.0.0"
__all__ = ["CMPLXMCPServer", "create_server"]

from .server import CMPLXMCPServer, create_server
