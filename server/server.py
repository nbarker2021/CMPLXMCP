"""
CMPLX MCP Server - Main Entry Point
====================================
Implements the Model Context Protocol server with all CMPLX tools organized
by their controller layers.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .tools import (LAYER1_TOOLS, LAYER2_TOOLS, LAYER3_TOOLS, LAYER4_TOOLS,
                    LAYER5_TOOLS, SYSTEM_TOOLS)
from .universal_tools import UNIVERSAL_TOOLS

logger = logging.getLogger("cmplx.mcp.server")


class CMPLXMCPServer:
    """
    CMPLX MCP Server - The OS Kernel
    
    Exposes all CMPLX capabilities through MCP protocol.
    Heavy data processing happens server-side.
    Clients receive lightweight references/handles.
    """
    
    def __init__(self, data_root: Path | None = None):
        self.data_root = data_root or Path("CMPLX-Build_v1.0_UNIFIED")
        self.server = Server("cmplx-os")
        self._register_handlers()
        
    def _register_handlers(self):
        """Register all tool handlers."""
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available tools organized by layer."""
            tools = []
            
            # Layer 1: Morphonic Foundation
            tools.extend([
                Tool(
                    name="l1_morphon_generate",
                    description="Generate a universal morphon from seed",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "seed": {"type": "string", "description": "Single digit seed (0-9)"}
                        },
                        "required": ["seed"]
                    }
                ),
                Tool(
                    name="l1_mglc_execute",
                    description="Execute Morphonic Lambda Calculus expression",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                            "context": {"type": "object"}
                        },
                        "required": ["expression"]
                    }
                ),
                Tool(
                    name="l1_seed_expand",
                    description="Expand single digit to 24D substrate",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "digit": {"type": "integer", "minimum": 0, "maximum": 9},
                            "dimensions": {"type": "integer", "default": 24}
                        },
                        "required": ["digit"]
                    }
                ),
            ])
            
            # Layer 2: Geometric Engine
            tools.extend([
                Tool(
                    name="l2_e8_project",
                    description="Project 8D vector to E8 lattice",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vector": {"type": "array", "items": {"type": "number"}, "minItems": 8, "maxItems": 8},
                            "return_format": {"type": "string", "enum": ["minimal", "full"], "default": "minimal"}
                        },
                        "required": ["vector"]
                    }
                ),
                Tool(
                    name="l2_leech_nearest",
                    description="Find nearest point in Leech lattice",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vector": {"type": "array", "items": {"type": "number"}, "minItems": 24, "maxItems": 24},
                            "return_format": {"type": "string", "enum": ["handle", "coordinates"], "default": "handle"}
                        },
                        "required": ["vector"]
                    }
                ),
                Tool(
                    name="l2_weyl_navigate",
                    description="Navigate Weyl chambers from position",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "position": {"type": "array", "items": {"type": "number"}},
                            "target_root": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["position"]
                    }
                ),
                Tool(
                    name="l2_niemeier_classify",
                    description="Classify vector against Niemeier lattices",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vector": {"type": "array", "items": {"type": "number"}, "minItems": 24, "maxItems": 24}
                        },
                        "required": ["vector"]
                    }
                ),
            ])
            
            # Layer 3: Operational Systems
            tools.extend([
                Tool(
                    name="l3_morsr_optimize",
                    description="Run MORSR optimization on state",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "initial_state": {"type": "array", "items": {"type": "number"}},
                            "iterations": {"type": "integer", "default": 100},
                            "constraint": {"type": "string", "enum": ["conservation", "phi_metric", "none"], "default": "conservation"}
                        },
                        "required": ["initial_state"]
                    }
                ),
                Tool(
                    name="l3_conservation_check",
                    description="Check if transformation satisfies ΔΦ ≤ 0",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "before": {"type": "array", "items": {"type": "number"}},
                            "after": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["before", "after"]
                    }
                ),
            ])
            
            # Layer 4: Governance
            tools.extend([
                Tool(
                    name="l4_digital_root",
                    description="Calculate digital root (gravitational anchor)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "number": {"type": "integer"},
                            "modulus": {"type": "integer", "default": 9}
                        },
                        "required": ["number"]
                    }
                ),
                Tool(
                    name="l4_seven_witness",
                    description="Run seven-witness validation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "artifact": {"type": "object"},
                            "perspectives": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["artifact"]
                    }
                ),
                Tool(
                    name="l4_policy_check",
                    description="Check artifact against policy hierarchy",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "artifact_id": {"type": "string"},
                            "policy_tier": {"type": "integer", "minimum": 1, "maximum": 7}
                        },
                        "required": ["artifact_id"]
                    }
                ),
            ])
            
            # Layer 5: Interface
            tools.extend([
                Tool(
                    name="l5_embed",
                    description="Embed content into E8 space",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "domain": {"type": "string", "enum": ["text", "math", "audio", "visual", "code"]},
                            "return_handle": {"type": "boolean", "default": True}
                        },
                        "required": ["content", "domain"]
                    }
                ),
                Tool(
                    name="l5_query_similar",
                    description="Find similar overlays by handle",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "handle": {"type": "string"},
                            "top_k": {"type": "integer", "default": 10}
                        },
                        "required": ["handle"]
                    }
                ),
                Tool(
                    name="l5_transform",
                    description="Apply geometric transformation to overlay",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "handle": {"type": "string"},
                            "operator": {"type": "string", "enum": ["rotation", "reflection", "scale", "translate"]},
                            "params": {"type": "object"}
                        },
                        "required": ["handle", "operator"]
                    }
                ),
                Tool(
                    name="l5_atomic_compose",
                    description="Atomicize text into atoms and adjacency edges (atomic interaction map)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to atomicize"},
                            "max_atoms": {"type": "integer", "default": 128}
                        },
                        "required": ["text"]
                    }
                ),
            ])
            
            # System Tools
            tools.extend([
                Tool(
                    name="sys_info",
                    description="Get system information and status",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="sys_cache_stats",
                    description="Get cache statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="sys_resolve_handle",
                    description="Resolve a handle to full data (heavy operation)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "handle": {"type": "string"},
                            "max_size_mb": {"type": "number", "default": 10}
                        },
                        "required": ["handle"]
                    }
                ),
            ])
            
            # Universal System Tools
            tools.extend([
                Tool(
                    name="universal_translate",
                    description="Translate any content to geometric form (atoms + bonds)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Content to translate"},
                            "content_type": {"type": "string", "description": "Type: text, code, math, audio, image, etc. (auto-detected if null)"},
                            "identity": {"type": "string", "default": "anonymous"}
                        },
                        "required": ["content"]
                    }
                ),
                Tool(
                    name="crystal_store",
                    description="Store geometric form as labeled crystal with full provenance",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "form_handle": {"type": "string", "description": "Handle from universal_translate"},
                            "name": {"type": "string"},
                            "identity": {"type": "string", "default": "anonymous"},
                            "temporal_phase": {"type": "string", "enum": ["past", "present", "future"], "default": "present"},
                            "tags": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["form_handle"]
                    }
                ),
                Tool(
                    name="crystal_retrieve",
                    description="Retrieve crystal metadata by handle",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crystal_id": {"type": "string"}
                        },
                        "required": ["crystal_id"]
                    }
                ),
                Tool(
                    name="crystal_resonance_query",
                    description="Find crystals by resonance (geometric similarity, not exact match)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crystal_id": {"type": "string", "description": "Crystal to find resonant matches for"},
                            "threshold": {"type": "number", "default": 0.7, "description": "Minimum resonance score (0-1)"},
                            "limit": {"type": "integer", "default": 10}
                        },
                        "required": ["crystal_id"]
                    }
                ),
                Tool(
                    name="crystal_merge",
                    description="Merge multiple crystals into super-crystal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crystal_ids": {"type": "array", "items": {"type": "string"}},
                            "name": {"type": "string"}
                        },
                        "required": ["crystal_ids"]
                    }
                ),
                Tool(
                    name="temporal_query",
                    description="Query crystals by temporal phase (past/present/future)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "phase": {"type": "string", "enum": ["past", "present", "future"]},
                            "limit": {"type": "integer", "default": 100}
                        },
                        "required": ["phase"]
                    }
                ),
                Tool(
                    name="temporal_remember",
                    description="Convert crystal to memory (move to past phase)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crystal_id": {"type": "string"},
                            "description": {"type": "string"},
                            "reliability": {"type": "number", "default": 1.0}
                        },
                        "required": ["crystal_id"]
                    }
                ),
                Tool(
                    name="hypothesis_generate",
                    description="Generate future hypotheses from context crystal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crystal_id": {"type": "string", "description": "Context to generate hypotheses from"},
                            "num_hypotheses": {"type": "integer", "default": 3},
                            "description": {"type": "string"}
                        },
                        "required": ["crystal_id"]
                    }
                ),
                Tool(
                    name="hypothesis_validate",
                    description="Validate hypothesis against actual outcome",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hypothesis_id": {"type": "string"},
                            "actual_crystal_id": {"type": "string"}
                        },
                        "required": ["hypothesis_id", "actual_crystal_id"]
                    }
                ),
                Tool(
                    name="temporal_counterfactual",
                    description="Generate counterfactual (what if?) scenario",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crystal_id": {"type": "string", "description": "Base reality"},
                            "changes": {"type": "object", "description": "What to change"}
                        },
                        "required": ["crystal_id"]
                    }
                ),
                Tool(
                    name="identity_register",
                    description="Register new identity in the system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "identity_id": {"type": "string"}
                        },
                        "required": ["name"]
                    }
                ),
                Tool(
                    name="identity_history",
                    description="Get complete history of an identity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "identity_id": {"type": "string"}
                        },
                        "required": ["identity_id"]
                    }
                ),
                Tool(
                    name="audit_provenance",
                    description="Audit full provenance chain of a crystal",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "crystal_id": {"type": "string"}
                        },
                        "required": ["crystal_id"]
                    }
                ),
                Tool(
                    name="verify_receipt",
                    description="Verify Speedlight receipt authenticity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "receipt_id": {"type": "string"}
                        },
                        "required": ["receipt_id"]
                    }
                ),
                Tool(
                    name="universal_stats",
                    description="Get universal system statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
            ])
            
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Route tool calls to appropriate handler."""
            logger.info(f"Tool call: {name} with args: {arguments}")
            
            try:
                # Route by layer prefix
                if name.startswith("l1_"):
                    result = await LAYER1_TOOLS.handle(name, arguments, self.data_root)
                elif name.startswith("l2_"):
                    result = await LAYER2_TOOLS.handle(name, arguments, self.data_root)
                elif name.startswith("l3_"):
                    result = await LAYER3_TOOLS.handle(name, arguments, self.data_root)
                elif name.startswith("l4_"):
                    result = await LAYER4_TOOLS.handle(name, arguments, self.data_root)
                elif name.startswith("l5_"):
                    result = await LAYER5_TOOLS.handle(name, arguments, self.data_root)
                elif name.startswith("sys_"):
                    result = await SYSTEM_TOOLS.handle(name, arguments, self.data_root)
                elif name in ["universal_translate", "crystal_store", "crystal_retrieve",
                             "crystal_resonance_query", "crystal_merge", "temporal_query",
                             "temporal_remember", "hypothesis_generate", "hypothesis_validate",
                             "temporal_counterfactual", "identity_register", "identity_history",
                             "audit_provenance", "verify_receipt", "universal_stats"]:
                    result = await UNIVERSAL_TOOLS.handle(name, arguments, self.data_root)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "error": str(e),
                    "tool": name,
                    "status": "failed"
                }, indent=2))]
    
    async def run(self):
        async with stdio_server() as (read, write):
            await self.server.run(read, write, self.server.create_initialization_options())


def create_server(data_root: Path | None = None) -> CMPLXMCPServer:
    """Factory function to create a configured server."""
    return CMPLXMCPServer(data_root=data_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server = create_server()
    asyncio.run(server.run())
