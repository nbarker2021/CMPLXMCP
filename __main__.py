"""
CMPLX MCP OS - Main Entry Point
===============================

Run server:
    python -m mcp_os server

Run client example:
    python -m mcp_os client
"""

import asyncio
import sys


async def run_server():
    """Run the MCP server."""
    from .server import create_server
    
    server = create_server()
    await server.run()


async def run_client_example():
    """Run a client example."""
    from .client import create_client
    from .controllers import ProxyRegistry
    
    async with create_client() as client:
        print("Connected to CMPLX MCP OS")
        
        # Get system info
        info = await client.sys_info()
        print(f"System: {info}")
        
        # Use proxy registry
        registry = ProxyRegistry(client)
        
        # Layer 1: Morphonic
        morphon = await registry.l1.generate_morphon("7")
        print(f"Morphon: {morphon}")
        
        # Layer 2: Geometric
        e8 = await registry.l2.project_e8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        print(f"E8 projection: {e8}")
        
        # Layer 4: Governance
        dr = await registry.l4.digital_root(432)
        print(f"Digital root: {dr}")
        
        # Layer 5: Interface
        embed = await registry.l5.embed("Quantum consciousness in geometric spaces")
        print(f"Embedding: {embed}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m mcp_os [server|client]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "server":
        asyncio.run(run_server())
    elif command == "client":
        asyncio.run(run_client_example())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
