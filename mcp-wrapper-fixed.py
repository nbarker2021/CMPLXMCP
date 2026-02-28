#!/usr/bin/env python3
"""
CMPLX MCP OS Wrapper (Fixed)
==============================

Fixed version with:
- UTF-8 encoding enforcement
- Windows console compatibility
- CHUNK_LIMIT workaround
"""

import sys
import os
import io
from pathlib import Path

def setup_encoding():
    """Force UTF-8 encoding for Windows."""
    if sys.platform == 'win32':
        # Force UTF-8 for stdout/stderr
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8', 
            errors='replace'
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, 
            encoding='utf-8', 
            errors='replace'
        )
        # Set environment variable
        os.environ['PYTHONIOENCODING'] = 'utf-8'

def fix_chunk_limit():
    """Workaround for CHUNK_LIMIT import issue."""
    # Define CHUNK_LIMIT if not already defined
    import builtins
    if not hasattr(builtins, 'CHUNK_LIMIT'):
        builtins.CHUNK_LIMIT = 1024 * 1024  # 1MB default

def main():
    """Run MCP server with fixes."""
    # Setup encoding first
    setup_encoding()
    
    # Fix CHUNK_LIMIT
    fix_chunk_limit()
    
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
    os.environ["CMPLX_MCP_DEBUG"] = "0"
    
    # Suppress the CHUNK_LIMIT warning
    import warnings
    warnings.filterwarnings('ignore', message='.*CHUNK_LIMIT.*')
    warnings.filterwarnings('ignore', message='.*Falling back to local.*')
    
    # Import and run server
    try:
        from mcp_os.__main__ import main as mcp_main
        
        # Override sys.argv to run server
        sys.argv = ["mcp_os", "server"]
        
        mcp_main()
    except KeyboardInterrupt:
        print("\n[CMPLX-MCP] Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"[CMPLX-MCP] Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
