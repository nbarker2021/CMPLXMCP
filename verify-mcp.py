#!/usr/bin/env python3
"""
CMPLX MCP OS Verification Script
=================================

Verifies that the MCP server is properly configured and can start.
Run this to test your MCP setup before using with Kimi Code CLI.
"""

import sys
import json
import asyncio
from pathlib import Path

# Ensure paths
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

async def verify_imports():
    """Verify all required imports work."""
    print("[VERIFY] Testing imports...")
    
    try:
        from mcp.server import Server
        print("  [OK] mcp.server imported")
    except ImportError as e:
        print(f"  [X] Failed to import mcp.server: {e}")
        return False
    
    try:
        from mcp_os.server.server import CMPLXMCPServer
        print("  [OK] CMPLXMCPServer imported")
    except ImportError as e:
        print(f"  [X] Failed to import CMPLXMCPServer: {e}")
        return False
    
    try:
        from mcp_os.server.tools import LAYER1_TOOLS, LAYER2_TOOLS
        print(f"  [OK] Tools imported: L1={type(LAYER1_TOOLS).__name__}, L2={type(LAYER2_TOOLS).__name__}")
    except ImportError as e:
        print(f"  [X] Failed to import tools: {e}")
        return False
    
    return True

async def verify_server_creation():
    """Verify server can be created."""
    print("\n[VERIFY] Testing server creation...")
    
    try:
        from mcp_os.server.server import CMPLXMCPServer
        
        server = CMPLXMCPServer()
        print(f"  [OK] Server created: {server.server}")
        return True
    except Exception as e:
        print(f"  [X] Failed to create server: {e}")
        return False

async def verify_config_files():
    """Verify MCP config files exist."""
    print("\n[VERIFY] Checking config files...")
    
    # Global config
    global_config = Path.home() / ".kimi" / "mcp.json"
    if global_config.exists():
        print(f"  [OK] Global config: {global_config}")
        try:
            with open(global_config) as f:
                config = json.load(f)
                servers = list(config.get("mcpServers", {}).keys())
                print(f"    Servers: {servers}")
        except Exception as e:
            print(f"    Warning: Could not parse: {e}")
    else:
        print(f"  [X] Global config not found: {global_config}")
    
    # Project config
    project_config = Path(__file__).parent.parent / ".kimi" / "mcp.json"
    if project_config.exists():
        print(f"  [OK] Project config: {project_config}")
    else:
        print(f"  ⚠ Project config not found: {project_config}")
    
    return True

async def main():
    """Run all verifications."""
    print("=" * 60)
    print("CMPLX MCP OS Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Check Python version
    print(f"[VERIFY] Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("  [X] Python 3.9+ required")
        results.append(False)
    else:
        print("  [OK] Python version OK")
        results.append(True)
    
    # Check paths
    print(f"\n[VERIFY] Python path:")
    for i, p in enumerate(sys.path[:3]):
        print(f"  [{i}] {p}")
    
    # Run verifications
    results.append(await verify_imports())
    results.append(await verify_server_creation())
    results.append(await verify_config_files())
    
    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("[OK] ALL CHECKS PASSED")
        print("\nYour CMPLX MCP OS is ready to use with Kimi Code CLI!")
        print("\nTest with:")
        print("  kimi mcp list")
        print("  kimi mcp test cmplx-os")
    else:
        print("[X] SOME CHECKS FAILED")
        print("\nPlease fix the issues above before using with Kimi.")
    print("=" * 60)
    
    return all(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
