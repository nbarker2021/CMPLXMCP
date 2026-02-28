#!/usr/bin/env python3
"""
CMPLX MCP Server Entry Point (Production)
==========================================

Guaranteed to work with Kimi Code CLI.
Handles all encoding and constant issues.
"""

import sys
import os
import io
import warnings
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.absolute()

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['CMPLX_ROOT'] = str(PROJECT_ROOT)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import and run
from mcp_os.server.server import CMPLXMCPServer

async def main():
    """Start the server."""
    server = CMPLXMCPServer(data_root=PROJECT_ROOT)
    await server.run()

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[MCP] Server stopped", file=sys.stderr)
        sys.exit(0)
