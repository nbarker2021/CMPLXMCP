#!/usr/bin/env python3
"""
CMPLX Constants Fix
===================

Defines all missing constants that unified families expect.
Import this BEFORE importing any unified family modules.
"""

import builtins
import numpy as np
from enum import Enum

# CHUNK_LIMIT for document chunking
if not hasattr(builtins, 'CHUNK_LIMIT'):
    builtins.CHUNK_LIMIT = 1024 * 1024  # 1MB

# CANONICAL_E8_LAPLACIAN for spectrum comparison
if not hasattr(builtins, 'CANONICAL_E8_LAPLACIAN'):
    class _Spectrum:
        def __init__(self):
            self.eigenvalues = np.array([2.0] * 240)
            self.multiplicities = np.ones(240)
    builtins.CANONICAL_E8_LAPLACIAN = _Spectrum()

# CardType enum
if not hasattr(builtins, 'CardType'):
    class CardType(Enum):
        FOUNDATION = "foundation"
        AGENT = "agent"
        ACTION = "action"
        ORACLE = "oracle"
        SCENE = "scene"
        STAKEHOLDER = "stakeholder"
        POLICY = "policy"
        GUARDRAIL = "guardrail"
        CHECKPOINT = "checkpoint"
        NOTE = "note"
        OBSERVATION = "observation"
        EVIDENCE = "evidence"
        CLAIM = "claim"
        HYPOTHESIS = "hypothesis"
        
        # Lowercase aliases for code compatibility
        foundation = FOUNDATION
        agent = AGENT
        action = ACTION
        oracle = ORACLE
        scene = SCENE
        stakeholder = STAKEHOLDER
        policy = POLICY
        guardrail = GUARDRAIL
        checkpoint = CHECKPOINT
        note = NOTE
        observation = OBSERVATION
        evidence = EVIDENCE
        claim = CLAIM
        hypothesis = HYPOTHESIS
    builtins.CardType = CardType

# Phase enum
if not hasattr(builtins, 'Phase'):
    class Phase(Enum):
        PAST = "past"
        PRESENT = "present"
        FUTURE = "future"
    builtins.Phase = Phase

# Spectrum class
if not hasattr(builtins, 'Spectrum'):
    class Spectrum:
        def __init__(self, eigenvalues=None, multiplicities=None):
            self.eigenvalues = eigenvalues or np.array([])
            self.multiplicities = multiplicities or np.array([])
    builtins.Spectrum = Spectrum

# Common constants
def ensure_constant(name, default_value):
    if not hasattr(builtins, name):
        setattr(builtins, name, default_value)
        return True
    return False

ensure_constant('MAX_EMBEDDING_DIM', 1536)
ensure_constant('DEFAULT_PHI', 1.618033988749895)
ensure_constant('PHI_TOLERANCE', 1e-10)
ensure_constant('E8_ROOT_COUNT', 240)
ensure_constant('LEECH_LATTICE_NORM', 4)
ensure_constant('DEFAULT_PRECISION', 'float64')
ensure_constant('RECEIPT_VERSION', '1.0')
ensure_constant('TMN_MAX_DEPTH', 24)
ensure_constant('PHI_METRIC_TOLERANCE', 0.01)
ensure_constant('LANE_VERSION', '1.0.0')  # Used by unified families for lane identification

# Unicode-safe symbols for Windows console compatibility
class SafeSymbols:
    """Unicode-safe symbols that work on all consoles including Windows."""
    CHECK = "[OK]"      # Replaces ✓ (\u2713)
    CROSS = "[X]"       # Replaces ✗ (\u2717)
    BULLET = "*"        # Replaces •
    ARROW = "->"        # Replaces →
    
    @classmethod
    def replace_unicode(cls, text: str) -> str:
        """Replace unicode characters with safe ASCII equivalents."""
        if not isinstance(text, str):
            return text
        replacements = {
            '\u2713': cls.CHECK,  # ✓
            '\u2714': cls.CHECK,  # ✔
            '\u2717': cls.CROSS,  # ✗
            '\u2718': cls.CROSS,  # ✘
            '\u2715': cls.CROSS,  # ✕
            '\u2716': cls.CROSS,  # ✖
        }
        for uni, safe in replacements.items():
            text = text.replace(uni, safe)
        return text

builtins.SafeSymbols = SafeSymbols


def safe_print(text: str, *args, **kwargs):
    """Print text with unicode characters replaced for console compatibility."""
    import sys
    safe_text = SafeSymbols.replace_unicode(str(text))
    # Handle encoding issues on Windows
    if sys.platform == 'win32':
        try:
            # Try to print with current encoding
            print(safe_text, *args, **kwargs)
        except UnicodeEncodeError:
            # Fall back to ASCII-only encoding
            safe_text = safe_text.encode('ascii', 'replace').decode('ascii')
            print(safe_text, *args, **kwargs)
    else:
        print(safe_text, *args, **kwargs)

builtins.safe_print = safe_print
