"""
MDHG + CA Integration
=====================
Multi-scale geometric cache with self-regulating cellular automata dynamics.

Adapted from cqe_civ to integrate with CMPLX Universal System.
"""

import math
import time
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional


def _h(x: Any) -> str:
    """Hash any object to hex string."""
    b = json.dumps(x, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def quantize(v24: List[float], bins=16) -> Tuple[int, ...]:
    """Quantize 24D vector to discrete bins."""
    out = []
    for x in v24:
        k = int(max(0, min(bins - 1, math.floor(float(x) * bins))))
        out.append(k)
    return tuple(out)


def slot_of(q24: Tuple[int, ...], grid_side=12) -> str:
    """Map quantized 24D to 2D slot via double hashing."""
    h1 = int(hashlib.sha256(("A" + str(q24)).encode()).hexdigest(), 16)
    h2 = int(hashlib.sha256(("B" + str(q24)).encode()).hexdigest(), 16)
    x = h1 % grid_side
    y = h2 % grid_side
    return f"{x:02d},{y:02d}"


@dataclass
class SlotEntry:
    """Entry in an MDHG slot."""
    key: str
    q24: Tuple[int, ...]
    meta: Dict[str, Any]
    last: float = field(default_factory=lambda: time.time())
    hits: int = 0


class MDHGCache:
    """
    MDHG geometric cache: 24D → 2D slot grid → per-slot eviction.
    
    Slot occupancy creates a "shape" that the CA field responds to.
    """
    
    def __init__(self, grid_side=12, cap_per_slot=6, bins=16, layer_name="default"):
        self.grid_side = grid_side
        self.cap_per_slot = cap_per_slot
        self.bins = bins
        self.layer_name = layer_name
        self.slots: Dict[str, List[SlotEntry]] = {}
        self._admission_count = 0
        self._eviction_count = 0
    
    def admit(self, v24: List[float], meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Admit a 24D vector into the cache.
        
        Returns admission result with slot assignment.
        """
        q = quantize(v24, bins=self.bins)
        slot = slot_of(q, grid_side=self.grid_side)
        
        # Create key from content
        key = _h({"q": q, "meta": {k: meta.get(k) for k in sorted(meta)[:12]}})[:16]
        
        arr = self.slots.setdefault(slot, [])
        
        # Check for existing (cache hit)
        for e in arr:
            if e.key == key:
                e.hits += 1
                e.last = time.time()
                return {
                    "admit": True,
                    "hit": True,
                    "slot": slot,
                    "distance": 0.0,
                    "key": key,
                    "q24": q,
                    "layer": self.layer_name
                }
        
        # Calculate distance to existing entries
        dist = 0.0
        if arr:
            distances = [sum(1 for a, b in zip(q, e.q24) if a != b) for e in arr]
            dist = float(min(distances))
        
        # Evict if necessary (LRU-like)
        evicted = None
        if len(arr) >= self.cap_per_slot:
            # Sort by (hits, last_access) - evict least used
            cand = sorted(arr, key=lambda e: (e.hits, e.last))[0]
            evicted = {
                "key": cand.key,
                "hits": cand.hits,
                "last": cand.last,
                "meta": cand.meta
            }
            arr.remove(cand)
            self._eviction_count += 1
        
        # Admit new entry
        arr.append(SlotEntry(key=key, q24=q, meta=meta))
        self._admission_count += 1
        
        return {
            "admit": True,
            "hit": False,
            "slot": slot,
            "distance": dist,
            "key": key,
            "evicted": evicted,
            "q24": q,
            "layer": self.layer_name
        }
    
    def get_slot_contents(self, slot: str) -> List[SlotEntry]:
        """Get all entries in a slot."""
        return self.slots.get(slot, [])
    
    def occupancy_grid(self) -> List[List[int]]:
        """Return 2D occupancy grid (0-9 scale)."""
        g = [[0 for _ in range(self.grid_side)] for _ in range(self.grid_side)]
        for s, arr in self.slots.items():
            try:
                x, y = s.split(",")
                g[int(y)][int(x)] = min(9, len(arr))
            except (ValueError, IndexError):
                continue
        return g
    
    def get_stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        total_entries = sum(len(arr) for arr in self.slots.values())
        return {
            "layer": self.layer_name,
            "slots_used": len(self.slots),
            "total_entries": total_entries,
            "admissions": self._admission_count,
            "evictions": self._eviction_count,
            "grid_side": self.grid_side,
            "cap_per_slot": self.cap_per_slot
        }
    
    def snapshot(self) -> Dict[str, Any]:
        """Full snapshot for persistence."""
        return {
            "grid_side": self.grid_side,
            "cap_per_slot": self.cap_per_slot,
            "bins": self.bins,
            "layer_name": self.layer_name,
            "slots": {
                k: [{"key": e.key, "hits": e.hits, "last": e.last, "meta": e.meta} for e in v]
                for k, v in self.slots.items()
            }
        }


class MDHGMultiScale:
    """
    Three MDHG caches at different timescales:
    - fast: high churn, micro-events (present)
    - med: medium churn, policy outcomes (recent past)
    - slow: low churn, identity/structure (deep past)
    """
    
    def __init__(self, grid_side=12, cap_per_slot=6, bins=16):
        self.fast = MDHGCache(grid_side, cap_per_slot, bins, "fast")
        self.med = MDHGCache(grid_side, cap_per_slot, bins, "med")
        self.slow = MDHGCache(grid_side, cap_per_slot, bins, "slow")
        
        # Drift tracking: slot -> rolling hash
        self._drift: Dict[str, Dict[str, str]] = {
            "fast": {},
            "med": {},
            "slow": {}
        }
    
    def admit(self, v24: List[float], meta: Dict[str, Any], layer="fast") -> Dict[str, Any]:
        """Admit to specific layer."""
        cache = getattr(self, layer)
        res = cache.admit(v24, meta)
        
        # Update drift signature
        slot = res.get("slot")
        if slot:
            q = res.get("q24")
            if q is not None:
                sig = _h({"q24": list(q)[:8], "meta_keys": sorted(list(meta.keys()))[:12]})
                dmap = self._drift[layer]
                prev = dmap.get(slot)
                dmap[slot] = sig
                if prev and prev != sig:
                    res["drift"] = True
        
        return res
    
    def admit_all_layers(self, v24: List[float], meta: Dict[str, Any]) -> Dict[str, Any]:
        """Admit to all three layers (for important data)."""
        return {
            "fast": self.admit(v24, meta, "fast"),
            "med": self.admit(v24, meta, "med"),
            "slow": self.admit(v24, meta, "slow")
        }
    
    def occupancy(self, layer="fast") -> List[List[int]]:
        """Get occupancy grid for layer."""
        return getattr(self, layer).occupancy_grid()
    
    def get_stats(self) -> Dict[str, Any]:
        """Stats for all layers."""
        return {
            "fast": self.fast.get_stats(),
            "med": self.med.get_stats(),
            "slow": self.slow.get_stats()
        }
    
    def snapshot(self) -> Dict[str, Any]:
        """Snapshot of all layers."""
        return {
            "fast": self.fast.snapshot(),
            "med": self.med.snapshot(),
            "slow": self.slow.snapshot()
        }


# ===== CA Field Integration =====

def clampi(x: int, lo: int, hi: int) -> int:
    """Clamp integer to range."""
    return lo if x < lo else hi if x > hi else x


@dataclass
class WolframAssignment:
    """
    Wolfram-style rule assignment for CA cells.
    
    Classes:
    - I: Stable/fixed (relax)
    - II: Oscillating/periodic (oscillate)
    - III: Chaotic (amplify)
    - IV: Complex/localized structures (complex)
    """
    wolfram_class: str  # "I", "II", "III", "IV"
    kernel: str  # "relax", "oscillate", "amplify", "complex"
    params: Dict[str, float] = field(default_factory=dict)
    rule_id: str = ""
    
    def finalize(self):
        """Generate stable rule ID."""
        if not self.rule_id:
            blob = {"c": self.wolfram_class, "k": self.kernel, "p": self.params}
            h = hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]
            self.rule_id = f"W{self.wolfram_class}-{self.kernel}-{h}"
        return self


@dataclass
class CACell:
    """Multi-state CA cell with typed channels (0-15, 4-bit each)."""
    ch: Dict[str, int] = field(default_factory=dict)
    assignment: Optional[WolframAssignment] = None
    phase: int = 0
    last_meta: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, k: str) -> int:
        return int(self.ch.get(k, 0))
    
    def set(self, k: str, v: int):
        self.ch[k] = clampi(int(v), 0, 15)


# Default channels map to crystal properties
DEFAULT_CHANNELS = [
    "pressure",    # Access frequency
    "risk",        # Uncertainty/entropy
    "trust",       # Verification status
    "food",        # Resource availability (data richness)
    "energy",      # Processing power available
    "water",       # Bandwidth/flow
    "debt",        # Dependency chain depth
    "innovation",  # Novelty of content
    "info",        # Information density
    "harm"         # Error/conflict count
]


def empty_cell() -> CACell:
    """Create empty CA cell."""
    c = CACell()
    for k in DEFAULT_CHANNELS:
        c.ch[k] = 0
    return c


def neighborhood_stats(grid: List[List[CACell]], x: int, y: int) -> Dict[str, float]:
    """Von Neumann + self neighborhood statistics."""
    h = len(grid)
    w = len(grid[0])
    coords = [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    
    out = {k: 0.0 for k in DEFAULT_CHANNELS}
    cnt = 0
    
    for cx, cy in coords:
        cx %= w
        cy %= h
        cell = grid[cy][cx]
        for k in DEFAULT_CHANNELS:
            out[k] += cell.get(k)
        cnt += 1
    
    for k in DEFAULT_CHANNELS:
        out[k] /= max(1, cnt)
    
    return out


def crystal_to_event(crystal_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert crystal metadata to CA event.
    
    Maps crystal properties to CA channel changes.
    """
    event = {
        "op": crystal_meta.get("action_type", "store"),
        "mag": crystal_meta.get("confidence", 0.5),
        "crystal_id": crystal_meta.get("crystal_id"),
        "resonance": crystal_meta.get("resonance_signature", "")[:8]
    }
    
    # Map crystal temporal phase to channel emphasis
    phase = crystal_meta.get("temporal_phase", "present")
    if phase == "past":
        event["scarcity_energy"] = 0.3  # Past = lower energy
    elif phase == "future":
        event["innovation_boost"] = 0.5  # Future = innovation
    
    return event


def apply_event_to_cell(cell: CACell, event: Dict[str, Any]):
    """Apply event to CA cell - determines channel deltas."""
    op = event.get("op") or event.get("act_op") or event.get("kind") or "event"
    mag = float(event.get("mag", 0.5))
    
    # Base pressure from any activity
    cell.set("pressure", cell.get("pressure") + int(2 * mag))
    
    # Map operations to channel changes
    if op in ("store", "embed", "create"):
        # Creation increases resources
        cell.set("innovation", cell.get("innovation") + int(2 * mag))
        cell.set("food", cell.get("food") + int(1 * mag))
        cell.set("energy", cell.get("energy") + int(1 * mag))
        cell.set("trust", cell.get("trust") + 1)
    
    elif op in ("query", "retrieve", "search"):
        # Queries increase pressure and info
        cell.set("pressure", cell.get("pressure") + int(3 * mag))
        cell.set("risk", cell.get("risk") + int(1 * mag))
        cell.set("info", cell.get("info") + 1)
    
    elif op in ("validate", "verify", "confirm"):
        # Validation reduces risk, increases trust
        cell.set("risk", cell.get("risk") - int(2 * mag))
        cell.set("pressure", cell.get("pressure") - int(1 * mag))
        cell.set("trust", cell.get("trust") + 1)
    
    elif op in ("merge", "combine", "integrate"):
        # Merging increases complexity
        cell.set("info", cell.get("info") + int(2 * mag))
        cell.set("risk", cell.get("risk") + int(1 * mag))
    
    elif op in ("evict", "delete", "remove"):
        # Removal = shock
        cell.set("harm", cell.get("harm") + int(2 * mag))
        cell.set("risk", cell.get("risk") + int(1 * mag))
        cell.set("trust", cell.get("trust") - 1)
    
    # Handle specific encodings
    if "debt_stress" in event:
        cell.set("debt", cell.get("debt") + int(4 * float(event["debt_stress"])))
    
    for rk, ch in [
        ("scarcity_food", "food"),
        ("scarcity_energy", "energy"),
        ("scarcity_water", "water")
    ]:
        if rk in event:
            cell.set(ch, cell.get(ch) + int(4 * float(event[rk])))


def kernel_step(cell: CACell, nb: Dict[str, float], rng: random.Random):
    """Update cell using its Wolfram assignment."""
    a = cell.assignment
    if a is None:
        # Default: oscillating (Class II)
        a = WolframAssignment("II", "oscillate", {
            "diffuse": 0.15,
            "noise": 0.01,
            "inertia": 0.4,
            "amp": 0.25
        }).finalize()
        cell.assignment = a
    
    diffuse = float(a.params.get("diffuse", 0.15))
    noise_p = float(a.params.get("noise", 0.01))
    inertia = float(a.params.get("inertia", 0.4))
    amp = float(a.params.get("amp", 0.25))
    nonlin = float(a.params.get("nonlin", 0.35))
    
    if a.kernel == "relax":
        # Class I: converge to stable state
        for k in DEFAULT_CHANNELS:
            cur = cell.get(k)
            tgt = nb[k]
            nxt = (1 - inertia) * cur + inertia * tgt
            nxt = nxt * (1 - diffuse) + diffuse * tgt
            if rng.random() < noise_p:
                nxt += rng.choice([-1, 1])
            cell.set(k, int(round(nxt)))
    
    elif a.kernel == "oscillate":
        # Class II: periodic behavior
        cell.phase = (cell.phase + 1) % 8
        osc = math.sin(2 * math.pi * (cell.phase / 8.0))
        for k in DEFAULT_CHANNELS:
            cur = cell.get(k)
            tgt = nb[k]
            nxt = (1 - inertia) * cur + inertia * tgt + amp * osc
            nxt = nxt * (1 - diffuse) + diffuse * tgt
            if rng.random() < noise_p:
                nxt += rng.choice([-1, 1])
            cell.set(k, int(round(nxt)))
    
    elif a.kernel == "amplify":
        # Class III: chaotic amplification
        for k in DEFAULT_CHANNELS:
            cur = cell.get(k)
            mean = nb[k]
            dev = cur - mean
            nxt = cur + amp * dev
            nxt = nxt * (1 - diffuse * 0.5) + (diffuse * 0.5) * mean
            if rng.random() < max(noise_p, 0.05):
                nxt += rng.choice([-2, -1, 1, 2])
            cell.set(k, int(round(nxt)))
    
    else:  # "complex"
        # Class IV: complex localized structures
        cell.phase = (cell.phase + 1) % 16
        osc = math.sin(2 * math.pi * (cell.phase / 16.0))
        for k in DEFAULT_CHANNELS:
            cur = cell.get(k)
            mean = nb[k]
            dev = cur - mean
            kick = 0.0
            if abs(dev) > (nonlin * 4.0):
                kick = amp * math.copysign(1.0, dev)
            nxt = cur * (1 - inertia) + inertia * mean + kick + 0.15 * osc
            nxt = nxt * (1 - diffuse) + diffuse * mean
            if rng.random() < noise_p:
                nxt += rng.choice([-1, 1])
            cell.set(k, int(round(nxt)))


@dataclass
class CAField:
    """
    Cellular Automaton field running over MDHG grid.
    
    Provides self-regulating dynamics that respond to MDHG admissions.
    """
    w: int
    h: int
    seed: int = 0
    grid: List[List[CACell]] = field(default_factory=list)
    tick: int = 0
    
    def __post_init__(self):
        if not self.grid:
            self.grid = [[empty_cell() for _ in range(self.w)] for __ in range(self.h)]
        self.rng = random.Random(self.seed)
    
    def apply_mdhg_admission(self, slot: str, meta: Dict[str, Any], 
                            assignment: Optional[WolframAssignment] = None):
        """
        Respond to MDHG admission.
        
        Converts slot coordinates to cell coordinates and applies event.
        """
        try:
            x_str, y_str = slot.split(",")
            x = int(x_str) % self.w
            y = int(y_str) % self.h
            
            cell = self.grid[y][x]
            event = crystal_to_event(meta)
            apply_event_to_cell(cell, event)
            
            if assignment:
                cell.assignment = assignment.finalize()
            
            cell.last_meta = meta
        except (ValueError, IndexError):
            pass
    
    def step_async(self, update_frac: float = 0.10) -> List[Dict[str, Any]]:
        """
        Update subset of cells asynchronously.
        
        Returns diagnostic events.
        """
        self.tick += 1
        n = int(self.w * self.h * max(0.01, min(1.0, update_frac)))
        
        for _ in range(n):
            x = self.rng.randrange(self.w)
            y = self.rng.randrange(self.h)
            cell = self.grid[y][x]
            nb = neighborhood_stats(self.grid, x, y)
            kernel_step(cell, nb, self.rng)
        
        # Diagnostics
        diagnostics = []
        hot = 0
        for row in self.grid:
            for c in row:
                if c.get("risk") >= 12 or c.get("harm") >= 10:
                    hot += 1
        
        if hot > (self.w * self.h * 0.10):
            diagnostics.append({
                "type": "ca_diag",
                "tick": self.tick,
                "kind": "risk_hotspots",
                "count": hot
            })
        
        return diagnostics
    
    def get_cell_state(self, x: int, y: int) -> Dict[str, Any]:
        """Get state of specific cell."""
        cell = self.grid[y % self.h][x % self.w]
        return {
            "channels": {k: cell.get(k) for k in DEFAULT_CHANNELS},
            "assignment": cell.assignment.rule_id if cell.assignment else None,
            "phase": cell.phase
        }
    
    def occupancy_scalar(self) -> List[List[int]]:
        """Scalar grid for visualization."""
        out = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                c = self.grid[y][x]
                v = int(round((c.get("pressure") + c.get("risk") + c.get("harm")) / 3))
                row.append(clampi(v, 0, 15))
            out.append(row)
        return out


@dataclass
class CAFieldMultiScale:
    """Three CA fields aligned with MDHG multi-scale caches."""
    w: int
    h: int
    seed: int = 0
    fast: CAField = None
    med: CAField = None
    slow: CAField = None
    
    def __post_init__(self):
        self.fast = CAField(self.w, self.h, seed=self.seed + 11)
        self.med = CAField(self.w, self.h, seed=self.seed + 22)
        self.slow = CAField(self.w, self.h, seed=self.seed + 33)
    
    def apply_mdhg_admission(self, layer: str, slot: str, meta: Dict[str, Any]):
        """Apply admission to specific layer's CA field."""
        field = getattr(self, layer)
        field.apply_mdhg_admission(slot, meta)
    
    def step(self) -> List[Dict[str, Any]]:
        """Step all three fields."""
        out = []
        out += self.fast.step_async(update_frac=0.18)
        out += self.med.step_async(update_frac=0.10)
        out += self.slow.step_async(update_frac=0.04)
        return out
    
    def scalar_grids(self) -> Dict[str, List[List[int]]]:
        """Get scalar grids for all layers."""
        return {
            "fast": self.fast.occupancy_scalar(),
            "med": self.med.occupancy_scalar(),
            "slow": self.slow.occupancy_scalar()
        }
