#!/usr/bin/env python3
"""
CMPLX MCP OS Wrapper
====================

Ensures proper Python path setup before running MCP server.
This wrapper handles path resolution for Kimi Code CLI integration.
"""

import sys
import os
from pathlib import Path

def main():
    """Run MCP server with proper paths."""
    # Get the directory containing this script
    wrapper_dir = Path(__file__).parent.absolute()
    
    # Add to Python path
    if str(wrapper_dir) not in sys.path:
        sys.path.insert(0, str(wrapper_dir))
    
    # Also add parent (CMPLX-DevLab) to path
    parent_dir = wrapper_dir.parent.absolute()
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Set environment variables
    os.environ["CMPLX_ROOT"] = str(parent_dir)
    os.environ["PYTHONPATH"] = str(parent_dir)
    
    # Import and run server
    from mcp_os.__main__ import main as mcp_main
    
    # Override sys.argv to run server
    sys.argv = ["mcp_os", "server"]
    
    try:
        mcp_main()
    except KeyboardInterrupt:
        print("\n[CMPLX-MCP] Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"[CMPLX-MCP] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
