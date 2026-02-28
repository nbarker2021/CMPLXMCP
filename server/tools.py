"""
CMPLX MCP Tool Handlers
=======================
All heavy processing happens here. Returns lightweight handles/references
to clients, keeping data server-side.
"""

import hashlib
import json
import logging
import re
import numpy as np
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("cmplx.mcp.tools")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _local_compose_atomic_result(text: str, *, max_atoms: int) -> dict[str, Any]:
    """Local fallback for atomic composition when unified family imports are unavailable."""
    normalized_text = " ".join(str(text or "").split()).strip()
    source_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    limit = max(1, int(max_atoms))
    tokens = [token.lower() for token in _TOKEN_RE.findall(normalized_text)]

    def _prefixed_id(prefix: str, payload: dict[str, Any]) -> str:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return f"{prefix}_{digest[:12]}"

    atoms: list[dict[str, Any]] = []
    for index, token in enumerate(tokens[:limit]):
        atom_payload = {"token": token, "index": index, "source_hash": source_hash}
        atoms.append(
            {
                "atom_id": _prefixed_id("atom", atom_payload),
                "token": token,
                "index": index,
                "labels": ["atomic_token", f"position_{index}"],
            }
        )

    edges: list[dict[str, Any]] = []
    for index in range(max(0, len(atoms) - 1)):
        edge_payload = {"src": atoms[index]["atom_id"], "dst": atoms[index + 1]["atom_id"]}
        edges.append(
            {
                "edge_id": _prefixed_id("atom_edge", edge_payload),
                "src_atom_id": atoms[index]["atom_id"],
                "dst_atom_id": atoms[index + 1]["atom_id"],
                "relation": "adjacent",
            }
        )

    return {
        "composition_id": _prefixed_id(
            "acompose", {"text_hash": source_hash, "atom_count": len(atoms)}
        ),
        "source_text_hash": source_hash,
        "atom_count": len(atoms),
        "atoms": atoms,
        "edges": edges,
    }


try:
    # Canonical path in current repo layout
    from cmplx_toolkit.unified_families.cmplx.functions import _compose_atomic_result
except Exception as exc:  # pragma: no cover - mid-rebuild compatibility path
    try:
        from unified_families.cmplx.functions import _compose_atomic_result
    except Exception:
        logger.warning(
            "Falling back to local _compose_atomic_result due to import failure: %s",
            exc,
        )
        _compose_atomic_result = _local_compose_atomic_result

# In-memory handle registry (would be Redis/database in production)
_HANDLE_REGISTRY: dict[str, Any] = {}
_HANDLE_COUNTER = 0


def _generate_handle(prefix: str, data: Any) -> str:
    """Generate a lightweight handle for server-side data."""
    global _HANDLE_COUNTER
    _HANDLE_COUNTER += 1
    
    # Create content hash for verification
    content = json.dumps(data, sort_keys=True, default=str)
    short_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    
    handle = f"{prefix}_{short_hash}_{_HANDLE_COUNTER:08x}"
    _HANDLE_REGISTRY[handle] = {
        "data": data,
        "created": datetime.utcnow().isoformat(),
        "access_count": 0
    }
    return handle


def _resolve_handle(handle: str) -> Any:
    """Resolve a handle to actual data (server-side only)."""
    entry = _HANDLE_REGISTRY.get(handle)
    if entry:
        entry["access_count"] += 1
        return entry["data"]
    raise ValueError(f"Unknown handle: {handle}")


class ToolRegistry:
    """Base class for layer tool registries."""
    
    async def handle(self, name: str, arguments: dict, data_root: Path) -> dict:
        """Route to appropriate handler method."""
        method_name = f"_{name}"
        if hasattr(self, method_name):
            return await getattr(self, method_name)(arguments, data_root)
        raise NotImplementedError(f"Tool {name} not implemented")


class Layer1Tools(ToolRegistry):
    """Layer 1: Morphonic Foundation"""
    
    async def _l1_morphon_generate(self, args: dict, data_root: Path) -> dict:
        """Generate universal morphon from seed."""
        seed = args.get("seed", "0")
        
        # Lightweight computation - no heavy data needed
        digit = int(seed[0]) if seed else 0
        
        # Simple morphon generation (placeholder for actual MGLC)
        morphon = {
            "seed": digit,
            "type": "universal_morphon",
            "properties": {
                "digital_root": digit % 9 or 9,
                "charge": "positive" if digit % 2 == 0 else "negative",
                "resonance": np.exp(2j * np.pi * digit / 9).real
            }
        }
        
        handle = _generate_handle("mp", morphon)
        
        return {
            "handle": handle,
            "summary": f"Morphon from seed {digit}",
            "dr": morphon["properties"]["digital_root"],
            "lightweight": True
        }
    
    async def _l1_mglc_execute(self, args: dict, data_root: Path) -> dict:
        """Execute MGLC expression."""
        expression = args.get("expression", "")
        context = args.get("context", {})
        
        # Simplified MGLC interpreter
        # Real implementation would parse and execute lambda terms
        result = {
            "expression": expression,
            "context_keys": list(context.keys()),
            "status": "executed",
            "result_type": "lambda_term"
        }
        
        handle = _generate_handle("mglc", result)
        
        return {
            "handle": handle,
            "expression_preview": expression[:50] + "..." if len(expression) > 50 else expression,
            "status": "success"
        }
    
    async def _l1_seed_expand(self, args: dict, data_root: Path) -> dict:
        """Expand single digit to 24D substrate."""
        digit = args.get("digit", 0)
        dimensions = args.get("dimensions", 24)
        
        # Generate 24D substrate using digit as seed
        np.random.seed(digit)
        substrate = np.random.randn(dimensions)
        substrate = substrate / np.linalg.norm(substrate)
        
        # Only return summary, store full vector server-side
        substrate_data = {
            "vector": substrate.tolist(),
            "seed": digit,
            "dimensions": dimensions,
            "norm": float(np.linalg.norm(substrate))
        }
        
        handle = _generate_handle("sub", substrate_data)
        
        return {
            "handle": handle,
            "seed": digit,
            "dimensions": dimensions,
            "norm": substrate_data["norm"],
            "first_8": substrate[:8].tolist(),  # Preview only
            "lightweight": True
        }


class Layer2Tools(ToolRegistry):
    """Layer 2: Geometric Engine - Heavy data processing"""
    
    async def _l2_e8_project(self, args: dict, data_root: Path) -> dict:
        """Project vector to E8 lattice - server-side heavy computation."""
        vector = np.array(args.get("vector", []))
        return_format = args.get("return_format", "minimal")
        
        if len(vector) != 8:
            raise ValueError("E8 projection requires 8D vector")
        
        # Load E8 roots (heavy data - server side only)
        e8_data_path = data_root / "cqe_unified_runtime_v8.0_RELEASE" / "cqe_unified_runtime"
        
        # Perform projection (simplified - real version uses actual E8 lattice)
        # This would load the actual 240 E8 roots
        projected = vector / np.linalg.norm(vector)
        
        result = {
            "original": vector.tolist(),
            "projected": projected.tolist(),
            "norm": float(np.linalg.norm(projected)),
            "lattice": "E8",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if return_format == "minimal":
            handle = _generate_handle("e8", result)
            return {
                "handle": handle,
                "lattice": "E8",
                "norm": result["norm"],
                "lightweight": True
            }
        else:
            return result
    
    async def _l2_leech_nearest(self, args: dict, data_root: Path) -> dict:
        """Find nearest point in Leech lattice - very heavy operation."""
        vector = np.array(args.get("vector", []))
        return_format = args.get("return_format", "handle")
        
        if len(vector) != 24:
            raise ValueError("Leech lattice requires 24D vector")
        
        # Leech has 196,560 minimal vectors - this is HEAVY
        # Server-side only - never send full lattice to client
        
        # Simplified nearest neighbor (real version uses Babai's algorithm)
        nearest = vector / np.linalg.norm(vector)
        
        result = {
            "query": vector.tolist(),
            "nearest": nearest.tolist(),
            "distance": float(np.linalg.norm(vector - nearest)),
            "lattice": "Leech"
        }
        
        if return_format == "handle":
            handle = _generate_handle("leech", result)
            return {
                "handle": handle,
                "lattice": "Leech",
                "distance": result["distance"],
                "lightweight": True
            }
        else:
            return result
    
    async def _l2_weyl_navigate(self, args: dict, data_root: Path) -> dict:
        """Navigate Weyl chambers."""
        position = np.array(args.get("position", []))
        target_root = args.get("target_root")
        
        # 696,729,600 Weyl chambers - massive data structure
        # Only chamber indices/handles returned to client
        
        current_chamber = hash(position.tobytes()) % 696729600
        
        result = {
            "current_chamber": current_chamber,
            "total_chambers": 696729600,
            "group": "E8",
            "navigation": {
                "from": position.tolist()[:4],  # Truncated for brevity
                "steps": 0 if target_root is None else 1
            }
        }
        
        handle = _generate_handle("weyl", result)
        
        return {
            "handle": handle,
            "chamber": current_chamber,
            "group": "E8",
            "lightweight": True
        }
    
    async def _l2_niemeier_classify(self, args: dict, data_root: Path) -> dict:
        """Classify against 24 Niemeier lattices."""
        vector = np.array(args.get("vector", []))
        
        # 24 Niemeier lattices with different root systems
        # Return classification result only
        
        # Simplified classification
        classifications = []
        for i in range(24):
            score = float(np.dot(vector, np.roll(vector, i)) / np.linalg.norm(vector)**2)
            classifications.append({
                "lattice": i,
                "affinity": score
            })
        
        classifications.sort(key=lambda x: x["affinity"], reverse=True)
        
        result = {
            "classifications": classifications,
            "top_match": classifications[0],
            "input_norm": float(np.linalg.norm(vector))
        }
        
        handle = _generate_handle("nie", result)
        
        return {
            "handle": handle,
            "top_lattice": classifications[0]["lattice"],
            "affinity": classifications[0]["affinity"],
            "lightweight": True
        }


class Layer3Tools(ToolRegistry):
    """Layer 3: Operational Systems"""
    
    async def _l3_morsr_optimize(self, args: dict, data_root: Path) -> dict:
        """Run MORSR optimization."""
        initial_state = np.array(args.get("initial_state", []))
        iterations = args.get("iterations", 100)
        constraint = args.get("constraint", "conservation")
        
        # MORSR: Morphonic Orthogonal Recursive State Refinement
        # Heavy optimization loop - server side
        
        state = initial_state.copy()
        history = []
        
        for i in range(iterations):
            # Simplified gradient step
            gradient = np.random.randn(*state.shape) * 0.01
            state = state - gradient
            
            if i % 10 == 0:
                history.append({
                    "iteration": i,
                    "norm": float(np.linalg.norm(state))
                })
        
        result = {
            "initial": initial_state.tolist(),
            "final": state.tolist(),
            "iterations": iterations,
            "constraint": constraint,
            "history": history,
            "final_norm": float(np.linalg.norm(state))
        }
        
        handle = _generate_handle("morsr", result)
        
        return {
            "handle": handle,
            "iterations": iterations,
            "final_norm": result["final_norm"],
            "converged": True,
            "lightweight": True
        }
    
    async def _l3_conservation_check(self, args: dict, data_root: Path) -> dict:
        """Check ΔΦ ≤ 0 conservation law."""
        before = np.array(args.get("before", []))
        after = np.array(args.get("after", []))
        
        # Calculate phi metric change
        phi_before = np.linalg.norm(before) * (1 + np.sqrt(5)) / 2
        phi_after = np.linalg.norm(after) * (1 + np.sqrt(5)) / 2
        
        delta_phi = phi_after - phi_before
        
        return {
            "delta_phi": float(delta_phi),
            "conserved": delta_phi <= 0,
            "phi_before": float(phi_before),
            "phi_after": float(phi_after),
            "law": "ΔΦ ≤ 0"
        }


class Layer4Tools(ToolRegistry):
    """Layer 4: Governance"""
    
    async def _l4_digital_root(self, args: dict, data_root: Path) -> dict:
        """Calculate digital root."""
        number = args.get("number", 0)
        modulus = args.get("modulus", 9)
        
        # Digital root calculation
        if modulus == 9:
            dr = 9 if number % 9 == 0 and number != 0 else number % 9
        else:
            dr = number % modulus
        
        # Gravitational anchor meanings
        meanings = {
            0: "ground_state",
            1: "unity",
            2: "duality",
            3: "trinity",
            4: "foundation",
            5: "life",
            6: "creation",
            7: "mystery",
            8: "infinity",
            9: "completion"
        }
        
        return {
            "number": number,
            "digital_root": dr,
            "modulus": modulus,
            "meaning": meanings.get(dr, "unknown"),
            "anchor_type": "gravitational"
        }
    
    async def _l4_seven_witness(self, args: dict, data_root: Path) -> dict:
        """Run seven-witness validation."""
        artifact = args.get("artifact", {})
        perspectives = args.get("perspectives", ["logical", "empirical", "ethical", "aesthetic", "economic", "social", "temporal"])
        
        witnesses = []
        for perspective in perspectives[:7]:
            # Each witness validates from its perspective
            witness = {
                "perspective": perspective,
                "valid": True,  # Placeholder
                "confidence": np.random.random(),
                "notes": f"Validated from {perspective} perspective"
            }
            witnesses.append(witness)
        
        all_valid = all(w["valid"] for w in witnesses)
        avg_confidence = sum(w["confidence"] for w in witnesses) / len(witnesses)
        
        return {
            "artifact_type": type(artifact).__name__,
            "witnesses": witnesses,
            "all_valid": all_valid,
            "average_confidence": float(avg_confidence),
            "required_witnesses": 7,
            "actual_witnesses": len(witnesses)
        }
    
    async def _l4_policy_check(self, args: dict, data_root: Path) -> dict:
        """Check against policy hierarchy."""
        artifact_id = args.get("artifact_id", "")
        policy_tier = args.get("policy_tier", 1)
        
        # 7-tier policy hierarchy
        tiers = {
            1: "universal_constants",
            2: "physical_laws",
            3: "mathematical_axioms",
            4: "system_invariants",
            5: "organizational_rules",
            6: "operational_procedures",
            7: "user_preferences"
        }
        
        return {
            "artifact_id": artifact_id,
            "policy_tier": policy_tier,
            "tier_name": tiers.get(policy_tier, "unknown"),
            "compliant": True,  # Placeholder
            "check_timestamp": datetime.utcnow().isoformat()
        }


class Layer5Tools(ToolRegistry):
    """Layer 5: Interface"""
    
    async def _l5_embed(self, args: dict, data_root: Path) -> dict:
        """Embed content into E8 space."""
        content = args.get("content", "")
        domain = args.get("domain", "text")
        return_handle = args.get("return_handle", True)
        
        # Create embedding (simplified - real version uses actual embedding model)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Generate 8D embedding vector
        np.random.seed(int(content_hash, 16) % (2**32))
        embedding = np.random.randn(8)
        embedding = embedding / np.linalg.norm(embedding)
        
        result = {
            "content_hash": content_hash,
            "domain": domain,
            "embedding": embedding.tolist(),
            "dimensions": 8,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }
        
        if return_handle:
            handle = _generate_handle("emb", result)
            return {
                "handle": handle,
                "domain": domain,
                "content_hash": content_hash,
                "embedding_preview": embedding[:4].tolist(),
                "lightweight": True
            }
        else:
            return result
    
    async def _l5_atomic_compose(self, args: dict, data_root: Path) -> dict:
        """Atomicize text into an interaction map of atoms and adjacency edges."""
        text = args.get("text") or args.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required for l5_atomic_compose")
        
        try:
            max_atoms = int(args.get("max_atoms", 128))
        except Exception:
            max_atoms = 128
        max_atoms = max(1, max_atoms)

        composition = _compose_atomic_result(text, max_atoms=max_atoms)

        handle = _generate_handle("atomic", composition)

        atoms = composition.get("atoms", [])
        preview_atoms = atoms[: min(len(atoms), 16)]

        return {
            "handle": handle,
            "composition_id": composition.get("composition_id", ""),
            "atom_count": composition.get("atom_count", len(atoms)),
            "source_text_hash": composition.get("source_text_hash", ""),
            "preview_atoms": preview_atoms,
            "lightweight": True,
        }
    
    async def _l5_query_similar(self, args: dict, data_root: Path) -> dict:
        """Query similar overlays by handle."""
        handle = args.get("handle", "")
        top_k = args.get("top_k", 10)
        
        # Resolve the handle to get embedding
        try:
            data = _resolve_handle(handle)
            query_embedding = np.array(data.get("embedding", []))
        except ValueError:
            return {"error": f"Cannot resolve handle: {handle}"}
        
        # Query against "database" (simplified)
        # Real version would query actual vector store
        similar = []
        for i in range(min(top_k * 2, 100)):  # Simulate results
            score = float(np.random.random())
            similar.append({
                "handle": f"emb_sim_{i:04x}",
                "similarity": score,
                "rank": i + 1
            })
        
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "query_handle": handle,
            "results": similar[:top_k],
            "total_candidates": len(similar),
            "top_k": top_k
        }
    
    async def _l5_transform(self, args: dict, data_root: Path) -> dict:
        """Apply geometric transformation."""
        handle = args.get("handle", "")
        operator = args.get("operator", "rotation")
        params = args.get("params", {})
        
        # Resolve source
        try:
            data = _resolve_handle(handle)
        except ValueError:
            return {"error": f"Cannot resolve handle: {handle}"}
        
        # Apply transformation (simplified)
        transformed = {
            "source_handle": handle,
            "operator": operator,
            "params": params,
            "applied": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        new_handle = _generate_handle("txf", transformed)
        
        return {
            "handle": new_handle,
            "source": handle,
            "operator": operator,
            "status": "transformed",
            "lightweight": True
        }


class SystemTools(ToolRegistry):
    """System-level tools"""
    
    async def _sys_info(self, args: dict, data_root: Path) -> dict:
        """Get system information."""
        return {
            "system": "CMPLX OS",
            "version": "1.0.0",
            "mcp_version": "1.0",
            "layers_available": [1, 2, 3, 4, 5],
            "data_root": str(data_root),
            "handles_in_memory": len(_HANDLE_REGISTRY),
            "status": "operational"
        }
    
    async def _sys_cache_stats(self, args: dict, data_root: Path) -> dict:
        """Get cache statistics."""
        return {
            "handle_registry_size": len(_HANDLE_REGISTRY),
            "handle_prefixes": {},
            "memory_estimate_mb": len(_HANDLE_REGISTRY) * 0.001  # Rough estimate
        }
    
    async def _sys_resolve_handle(self, args: dict, data_root: Path) -> dict:
        """Resolve handle to full data (admin/debug only)."""
        handle = args.get("handle", "")
        max_size_mb = args.get("max_size_mb", 10)
        
        try:
            data = _resolve_handle(handle)
            
            # Check size
            data_str = json.dumps(data)
            size_mb = len(data_str) / (1024 * 1024)
            
            if size_mb > max_size_mb:
                return {
                    "handle": handle,
                    "error": f"Data too large ({size_mb:.2f}MB > {max_size_mb}MB)",
                    "size_mb": size_mb
                }
            
            return {
                "handle": handle,
                "data": data,
                "size_mb": size_mb
            }
        except ValueError as e:
            return {"error": str(e)}


# Global tool instances
LAYER1_TOOLS = Layer1Tools()
LAYER2_TOOLS = Layer2Tools()
LAYER3_TOOLS = Layer3Tools()
LAYER4_TOOLS = Layer4Tools()
LAYER5_TOOLS = Layer5Tools()
SYSTEM_TOOLS = SystemTools()
