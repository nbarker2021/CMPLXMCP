"""
Universal Translator
====================
Converts ANY input into CMPLX geometric atoms.

The universal language is symmetry. Everything maps to:
1. A set of morphons (atoms)
2. Their geometric relationships (bonds)
3. Their temporal context (when/why)
4. Their identity provenance (who)
"""

import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, TextIO, Union
import asyncio

from .snap_atom import SNAPAtom


@dataclass
class GeometricForm:
    """
    Any data expressed as geometric form.
    
    This is the universal intermediate representation.
    From here, we can project to any lattice (E8, Leech, Barnes-Wells, etc.)
    """
    atoms: list[SNAPAtom]
    bonds: list[tuple[int, int, float]]  # (atom_idx_1, atom_idx_2, strength)
    envelope: dict  # Metadata about the form
    symmetry_signature: str = ""  # Computed from atoms+bonds
    
    def compute_signature(self) -> str:
        """Compute symmetry signature from geometric structure."""
        # Hash of sorted atom positions + bond topology
        atom_data = sorted([(a.position, a.charge) for a in self.atoms])
        bond_data = sorted(self.bonds)
        
        content = json.dumps({"atoms": atom_data, "bonds": bond_data}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def __post_init__(self):
        if not self.symmetry_signature:
            self.symmetry_signature = self.compute_signature()


class UniversalTranslator:
    """
    Universal Translator - converts ANYTHING to geometric form.
    
    Supported input types:
    - Text (any language, any length)
    - Code (any language)
    - Math (equations, proofs)
    - Audio (waveforms → frequency atoms)
    - Images (pixels → geometric patterns)
    - Video (temporal sequences)
    - 3D Models (mesh → atoms)
    - Data structures (JSON, XML, etc.)
    - File systems (directory trees)
    - Processes (workflows, algorithms)
    - Concepts (abstract ideas via embedding)
    - Hypotheses (future possibilities)
    - Memories (past events)
    """
    
    def __init__(self, lattice_dimension: int = 24):
        self.lattice_dim = lattice_dimension
        self._embedders = {}
        self._init_embedders()
    
    def _init_embedders(self):
        """Initialize type-specific embedders."""
        self._embedders = {
            "text": self._embed_text,
            "code": self._embed_code,
            "math": self._embed_math,
            "audio": self._embed_audio,
            "image": self._embed_image,
            "video": self._embed_video,
            "mesh": self._embed_mesh,
            "json": self._embed_data,
            "filesystem": self._embed_filesystem,
            "process": self._embed_process,
            "concept": self._embed_concept,
            "hypothesis": self._embed_hypothesis,
            "memory": self._embed_memory,
        }
    
    async def translate(self, data: Any, content_type: str | None = None, 
                       identity: str = "anonymous",
                       context: dict | None = None) -> GeometricForm:
        """
        Universal translation entry point.
        
        Args:
            data: The content to translate
            content_type: Explicit type hint (auto-detected if None)
            identity: Who is creating this geometric form
            context: Additional context (why, when, where)
        
        Returns:
            GeometricForm ready for crystallization
        """
        # Auto-detect type if not specified
        if content_type is None:
            content_type = self._detect_type(data)
        
        # Get appropriate embedder
        embedder = self._embedders.get(content_type, self._embed_generic)
        
        # Create geometric form
        form = await embedder(data, identity, context or {})
        
        # Add provenance
        form.envelope["identity"] = identity
        form.envelope["content_type"] = content_type
        form.envelope["translated_at"] = datetime.utcnow().isoformat()
        form.envelope["context"] = context
        
        return form
    
    def _detect_type(self, data: Any) -> str:
        """Auto-detect content type."""
        if isinstance(data, str):
            # Check if it's code
            code_indicators = ['def ', 'class ', 'function', '{', '}', ';']
            if any(ind in data for ind in code_indicators):
                return "code"
            # Check if it's math
            math_indicators = ['=', '+', '-', '*', '/', '∫', '∑', '√']
            if any(ind in data for ind in math_indicators):
                return "math"
            return "text"
        
        elif isinstance(data, (dict, list)):
            return "json"
        
        elif isinstance(data, bytes):
            # Could be audio, image, etc. - need magic numbers
            return "binary"
        
        elif isinstance(data, Path):
            return "filesystem"
        
        else:
            return "concept"
    
    # ===== Type-Specific Embedders =====
    
    async def _embed_text(self, text: str, identity: str, context: dict) -> GeometricForm:
        """
        Embed text as geometric atoms.
        
        Strategy:
        1. Tokenize into semantic units
        2. Each token → morphon (atom)
        3. Syntactic relationships → bonds
        4. Semantic coherence → envelope
        """
        # Split into tokens/words
        tokens = text.split()
        
        atoms = []
        for i, token in enumerate(tokens):
            # Each word becomes an atom
            # Position based on semantic embedding
            seed = sum(ord(c) for c in token) % 10
            position = self._seed_to_position(seed, i, len(tokens))
            
            atom = SNAPAtom(
                identity=f"{identity}_token_{i}",
                morphon_seed=seed,
                position=position,
                charge=self._text_charge(token),
                content=token,
                atom_type="text_token"
            )
            atoms.append(atom)
        
        # Create bonds between adjacent tokens
        bonds = [(i, i+1, 1.0) for i in range(len(atoms)-1)]
        
        # Add semantic bonds (words with similar hashes)
        for i in range(len(atoms)):
            for j in range(i+1, min(i+5, len(atoms))):
                similarity = self._token_similarity(atoms[i], atoms[j])
                if similarity > 0.7:
                    bonds.append((i, j, similarity))
        
        return GeometricForm(
            atoms=atoms,
            bonds=bonds,
            envelope={
                "token_count": len(tokens),
                "char_count": len(text),
                "semantic_density": len(bonds) / max(len(atoms), 1)
            }
        )
    
    async def _embed_code(self, code: str, identity: str, context: dict) -> GeometricForm:
        """
        Embed code as geometric atoms.
        
        Strategy:
        1. Parse AST (abstract syntax tree)
        2. Each node → atom
        3. Call graph + data flow → bonds
        4. Type signatures → envelope
        """
        # Simplified: treat as text with code-specific positioning
        form = await self._embed_text(code, identity, context)
        
        # Add code-specific envelope
        form.envelope["language"] = context.get("language", "unknown")
        form.envelope["complexity"] = code.count('def ') + code.count('class ')
        
        return form
    
    async def _embed_math(self, math: str, identity: str, context: dict) -> GeometricForm:
        """Embed mathematical expression."""
        # Math has inherent symmetry - leverage it
        form = await self._embed_text(math, identity, context)
        form.envelope["math_type"] = "expression"
        form.envelope["symmetry"] = "inherent"
        return form
    
    async def _embed_audio(self, audio: bytes, identity: str, context: dict) -> GeometricForm:
        """
        Embed audio as geometric atoms.
        
        Strategy:
        1. FFT → frequency spectrum
        2. Peaks → atoms (positions = frequencies)
        3. Temporal continuity → bonds
        """
        # Placeholder: would use actual FFT
        np.random.seed(hash(audio) % 2**32)
        
        atoms = []
        for i in range(24):  # 24 frequency bands
            position = np.random.randn(self.lattice_dim)
            position = position / np.linalg.norm(position)
            
            atom = SNAPAtom(
                identity=f"{identity}_freq_{i}",
                morphon_seed=i % 10,
                position=position.tolist(),
                charge=float(np.random.random()),
                content=f"freq_band_{i}",
                atom_type="audio_frequency"
            )
            atoms.append(atom)
        
        # Bonds between adjacent frequencies
        bonds = [(i, i+1, 0.9) for i in range(len(atoms)-1)]
        
        return GeometricForm(
            atoms=atoms,
            bonds=bonds,
            envelope={"duration_ms": context.get("duration", 0), "sample_rate": context.get("sample_rate", 44100)}
        )
    
    async def _embed_image(self, image: Any, identity: str, context: dict) -> GeometricForm:
        """
        Embed image as geometric atoms.
        
        Strategy:
        1. Edge detection → key points
        2. Key points → atoms
        3. Spatial proximity → bonds
        """
        # Placeholder
        atoms = [SNAPAtom(
            identity=f"{identity}_pixel_{i}",
            morphon_seed=i % 10,
            position=np.random.randn(self.lattice_dim).tolist(),
            charge=0.5,
            atom_type="image_feature"
        ) for i in range(24)]
        
        return GeometricForm(atoms=atoms, bonds=[], envelope={"width": 0, "height": 0})
    
    async def _embed_video(self, video: Any, identity: str, context: dict) -> GeometricForm:
        """Embed video (temporal sequence of images)."""
        # Combine image atoms with temporal bonds
        form = await self._embed_image(None, identity, context)
        form.envelope["frames"] = context.get("frames", 0)
        form.envelope["fps"] = context.get("fps", 30)
        return form
    
    async def _embed_mesh(self, mesh: Any, identity: str, context: dict) -> GeometricForm:
        """Embed 3D mesh."""
        # Vertices → atoms, edges → bonds (natural mapping!)
        vertices = context.get("vertices", [])
        faces = context.get("faces", [])
        
        atoms = []
        for i, v in enumerate(vertices[:100]):  # Limit for now
            atom = SNAPAtom(
                identity=f"{identity}_vert_{i}",
                morphon_seed=i % 10,
                position=v if len(v) == self.lattice_dim else (v + [0] * self.lattice_dim)[:self.lattice_dim],
                charge=1.0,
                atom_type="mesh_vertex"
            )
            atoms.append(atom)
        
        # Bonds from faces (edges)
        bonds = []
        for face in faces:
            for i in range(len(face)):
                bonds.append((face[i], face[(i+1) % len(face)], 1.0))
        
        return GeometricForm(atoms=atoms, bonds=bonds, envelope={"vertex_count": len(vertices)})
    
    async def _embed_data(self, data: Union[dict, list], identity: str, context: dict) -> GeometricForm:
        """Embed structured data (JSON, etc.)."""
        # Flatten to key-value pairs
        flattened = self._flatten_data(data)
        
        atoms = []
        for i, (key, value) in enumerate(flattened.items()):
            seed = sum(ord(c) for c in str(key)) % 10
            atom = SNAPAtom(
                identity=f"{identity}_field_{i}",
                morphon_seed=seed,
                position=self._seed_to_position(seed, i, len(flattened)),
                charge=self._value_charge(value),
                content=f"{key}={value}",
                atom_type="data_field"
            )
            atoms.append(atom)
        
        # Bonds between parent-child
        bonds = []
        # ... would need to track hierarchy
        
        return GeometricForm(
            atoms=atoms,
            bonds=bonds,
            envelope={"field_count": len(flattened), "depth": self._data_depth(data)}
        )
    
    async def _embed_filesystem(self, path: Path, identity: str, context: dict) -> GeometricForm:
        """Embed directory tree."""
        # Tree structure → hierarchical atoms
        files = list(path.rglob("*")) if path.is_dir() else [path]
        
        atoms = []
        for i, f in enumerate(files[:100]):  # Limit
            seed = sum(ord(c) for c in str(f)) % 10
            atom = SNAPAtom(
                identity=f"{identity}_file_{i}",
                morphon_seed=seed,
                position=self._seed_to_position(seed, i, len(files)),
                charge=1.0 if f.is_file() else 0.5,
                content=str(f.relative_to(path) if path.is_dir() else f),
                atom_type="filesystem_node"
            )
            atoms.append(atom)
        
        return GeometricForm(atoms=atoms, bonds=[], envelope={"file_count": len(files)})
    
    async def _embed_process(self, process: Any, identity: str, context: dict) -> GeometricForm:
        """Embed workflow/process."""
        # Steps → atoms, transitions → bonds
        steps = context.get("steps", [])
        
        atoms = []
        for i, step in enumerate(steps):
            atom = SNAPAtom(
                identity=f"{identity}_step_{i}",
                morphon_seed=i % 10,
                position=self._seed_to_position(i % 10, i, len(steps)),
                charge=step.get("weight", 1.0),
                content=step.get("name", f"step_{i}"),
                atom_type="process_step"
            )
            atoms.append(atom)
        
        bonds = [(i, i+1, 1.0) for i in range(len(atoms)-1)]
        
        return GeometricForm(atoms=atoms, bonds=bonds, envelope={"step_count": len(steps)})
    
    async def _embed_concept(self, concept: Any, identity: str, context: dict) -> GeometricForm:
        """Embed abstract concept via semantic embedding."""
        # Use description/embeddings
        description = str(concept)
        return await self._embed_text(description, identity, context)
    
    async def _embed_hypothesis(self, hypothesis: Any, identity: str, context: dict) -> GeometricForm:
        """
        Embed future hypothesis.
        
        Hypotheses are NOT real - they are potential futures.
        They get stored but marked with temporal type = "future"
        """
        form = await self._embed_concept(hypothesis, identity, context)
        form.envelope["temporal_type"] = "future"
        form.envelope["probability"] = context.get("probability", 0.5)
        form.envelope["outcome_space"] = context.get("outcomes", [])
        return form
    
    async def _embed_memory(self, memory: Any, identity: str, context: dict) -> GeometricForm:
        """
        Embed past memory.
        
        Memories are real but past - marked with temporal type = "past"
        """
        form = await self._embed_concept(memory, identity, context)
        form.envelope["temporal_type"] = "past"
        form.envelope["timestamp"] = context.get("timestamp", datetime.utcnow().isoformat())
        form.envelope["reliability"] = context.get("reliability", 1.0)  # Memory decay
        return form
    
    async def _embed_generic(self, data: Any, identity: str, context: dict) -> GeometricForm:
        """Fallback embedder for unknown types."""
        return await self._embed_text(str(data), identity, context)
    
    # ===== Helper Methods =====
    
    def _seed_to_position(self, seed: int, index: int, total: int) -> list[float]:
        """Convert seed to position in lattice space."""
        np.random.seed(seed + index * 1000)
        pos = np.random.randn(self.lattice_dim)
        return (pos / np.linalg.norm(pos)).tolist()
    
    def _text_charge(self, text: str) -> float:
        """Compute charge from text properties."""
        # Length + complexity
        return min(1.0, len(text) / 100 + text.count(' ') / 10)
    
    def _token_similarity(self, atom1: SNAPAtom, atom2: SNAPAtom) -> float:
        """Compute similarity between two text atoms."""
        # Simple: same morphon seed = similar
        if atom1.morphon_seed == atom2.morphon_seed:
            return 0.8
        return 0.0
    
    def _flatten_data(self, data: Any, prefix: str = "") -> dict:
        """Flatten nested data structure."""
        result = {}
        if isinstance(data, dict):
            for k, v in data.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    result.update(self._flatten_data(v, key))
                else:
                    result[key] = v
        elif isinstance(data, list):
            for i, v in enumerate(data):
                key = f"{prefix}[{i}]"
                if isinstance(v, (dict, list)):
                    result.update(self._flatten_data(v, key))
                else:
                    result[key] = v
        else:
            result[prefix] = data
        return result
    
    def _data_depth(self, data: Any) -> int:
        """Compute nesting depth of data."""
        if isinstance(data, dict):
            return 1 + max((self._data_depth(v) for v in data.values()), default=0)
        elif isinstance(data, list):
            return 1 + max((self._data_depth(v) for v in data), default=0)
        return 0
    
    def _value_charge(self, value: Any) -> float:
        """Compute charge from value."""
        if isinstance(value, (int, float)):
            return min(1.0, abs(value) / 1000)
        return 0.5
