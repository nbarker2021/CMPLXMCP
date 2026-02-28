"""
CMPLX MCP Server - Direct Entry Point
======================================

Allows running the server via:
    python -m mcp_os.server

Delegates to the existing mcp_os entry point.
"""

import asyncio

from .server import create_server


def main():
    """Start the MCP server."""
    server = create_server()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
