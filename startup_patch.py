#!/usr/bin/env python3
"""
CMPLX Startup Patch
===================

This module MUST be imported first before any other CMPLX modules.
It patches critical issues:
1. CardType enum - adds lowercase aliases
2. Unicode encoding - fixes Windows console encoding issues

Usage:
    import mcp_os.startup_patch  # Must be FIRST import
    # ... rest of imports
"""

import sys
import os
import io
import logging

# ========================================================================
# PATCH 1: Import constants_fix to ensure all constants are defined
# ========================================================================
# This MUST run before any unified family imports
try:
    import mcp_os.constants_fix as _cf
    # Force all constants to be set
    CardType = _cf.CardType
    SafeSymbols = _cf.SafeSymbols
    safe_print = _cf.safe_print
    LANE_VERSION = _cf.builtins.LANE_VERSION if hasattr(_cf.builtins, 'LANE_VERSION') else '1.0.0'
except ImportError:
    try:
        import constants_fix as _cf
        CardType = _cf.CardType
        SafeSymbols = _cf.SafeSymbols
        safe_print = _cf.safe_print
        LANE_VERSION = _cf.builtins.LANE_VERSION if hasattr(_cf.builtins, 'LANE_VERSION') else '1.0.0'
    except ImportError:
        LANE_VERSION = '1.0.0'  # Fallback
        pass  # Will fail later with clear error

# ========================================================================
# PATCH 2: Fix Windows console encoding for unicode characters
# ========================================================================
def _patch_windows_console():
    """Patch console encoding on Windows to handle unicode."""
    if sys.platform != 'win32':
        return
    
    # Set environment variable for UTF-8 mode
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Try to set console code page to UTF-8
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # Set console input/output code pages to UTF-8 (65001)
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass  # Fail silently if we can't set code page
    
    # Reconfigure stdout/stderr if available (Python 3.7+)
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

_patch_windows_console()

# ========================================================================
# PATCH 3: Safe logging handler that replaces unicode on Windows
# ========================================================================
class SafeStreamHandler(logging.StreamHandler):
    """Logging handler that handles unicode encoding errors gracefully."""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Replace unicode checkmarks with safe ASCII
            msg = msg.replace('\u2713', '[OK]').replace('\u2717', '[X]')
            msg = msg.replace('\u2714', '[OK]').replace('\u2718', '[X]')
            
            stream = self.stream
            # Handle Windows encoding
            if sys.platform == 'win32' and hasattr(stream, 'encoding'):
                try:
                    stream.write(msg + self.terminator)
                except UnicodeEncodeError:
                    # Fall back to ASCII with replacements
                    safe_msg = msg.encode('ascii', 'replace').decode('ascii')
                    stream.write(safe_msg + self.terminator)
            else:
                stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def patch_logging():
    """Patch root logger to use SafeStreamHandler."""
    root_logger = logging.getLogger()
    
    # Replace existing StreamHandlers with SafeStreamHandler
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, SafeStreamHandler):
            # Create new safe handler with same settings
            new_handler = SafeStreamHandler(handler.stream)
            new_handler.setLevel(handler.level)
            new_handler.setFormatter(handler.formatter)
            
            root_logger.removeHandler(handler)
            root_logger.addHandler(new_handler)

# Apply logging patch
patch_logging()

# ========================================================================
# EXPORTS
# ========================================================================

__all__ = ['CardType', 'SafeSymbols', 'safe_print', 'SafeStreamHandler', 'patch_logging']

# Mark as loaded
PATCH_LOADED = True
print("[CMPLX-PATCH] Startup patches applied successfully")
