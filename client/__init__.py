"""
CMPLX MCP OS Client
===================
Lightweight local runtime that communicates with the MCP server.
All heavy operations are proxied to the server.
"""

from .client import CMPLXClient, create_client

__all__ = ["CMPLXClient", "create_client"]
