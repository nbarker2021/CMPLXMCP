#!/usr/bin/env python3
"""
CHUNK_LIMIT Fix for CMPLX MCP OS
================================

Apply this fix to resolve the import warning.
"""

import sys
import builtins

# Define CHUNK_LIMIT at module level
CHUNK_LIMIT = 1024 * 1024  # 1MB

# Also inject into builtins to prevent import errors
if not hasattr(builtins, 'CHUNK_LIMIT'):
    builtins.CHUNK_LIMIT = CHUNK_LIMIT

# Make available for import
__all__ = ['CHUNK_LIMIT']
