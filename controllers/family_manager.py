"""Family controller manager for historical and donor build ingestion.

Organizes external build archives by family name and exposes a normalized
controller-facing interface for adapters/wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FamilyBuild:
    family_key: str
    family_name: str
    build_name: str
    build_path: str
    tags: List[str] = field(default_factory=list)


class FamilyControllerManager:
    """Discovers and manages family-grouped donor builds.

    The manager is read-only by default and designed to wrap each family into
    controller + adapter workflows while preserving source structure.
    """

    DEFAULT_SOURCE = Path(
        r"D:\Work Files\CMPLX Retool-Main\Builds for inclusion"
        r"\Review dataset, all files read only and portable via retool"
        r"\Historical and donor builds"
    )

    def __init__(self, source_root: str | Path | None = None):
        self.source_root = Path(source_root) if source_root else self.DEFAULT_SOURCE
        self._families: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def normalize_family_name(name: str) -> str:
        key = name.strip().lower().replace("_", " ").replace("-", " ")
        key = " ".join(key.split())
        if key.endswith(" family"):
            key = key[:-7].strip()
        return key.replace(" ", "_")

    @staticmethod
    def family_language_profile(family_key: str) -> Dict[str, Any]:
        profiles = {
            "agrm_mdhg": {"controller_layer": 3, "adapter": "operational_router", "keywords": ["agrm", "mdhg", "router", "network"]},
            "aletheia": {"controller_layer": 1, "adapter": "morphonic_lambda", "keywords": ["aletheia", "lambda", "hieroglyph"]},
            "cmplx": {"controller_layer": 5, "adapter": "cmplx_interface", "keywords": ["cmplx", "api", "sdk"]},
            "complex_t": {"controller_layer": 3, "adapter": "operational_router", "keywords": ["complex", "quadratic", "runtime"]},
            "cqe": {"controller_layer": 2, "adapter": "geometric_core", "keywords": ["cqe", "e8", "leech", "niemeier"]},
            "e8": {"controller_layer": 2, "adapter": "geometric_core", "keywords": ["e8", "root", "weyl"]},
            "eqai": {"controller_layer": 5, "adapter": "cmplx_interface", "keywords": ["eqai", "agent", "interface"]},
            "lattice": {"controller_layer": 2, "adapter": "geometric_core", "keywords": ["lattice", "leech", "niemeier"]},
            "lfai": {"controller_layer": 3, "adapter": "operational_router", "keywords": ["lfai", "autonomy", "agent"]},
            "lsdt": {"controller_layer": 4, "adapter": "governance_validation", "keywords": ["lsdt", "validation", "policy"]},
            "morphonic": {"controller_layer": 1, "adapter": "morphonic_lambda", "keywords": ["morphonic", "mglc", "seed", "lambda"]},
            "promutate_construct": {"controller_layer": 5, "adapter": "cmplx_interface", "keywords": ["construct", "promutate", "analysis"]},
            "quadratic_frame": {"controller_layer": 2, "adapter": "geometric_core", "keywords": ["quadratic", "frame", "helix"]},
            "quorum": {"controller_layer": 3, "adapter": "quorum_deliberation", "keywords": ["quorum", "deliberation", "consensus"]},
            "snap": {"controller_layer": 5, "adapter": "snap_translation", "keywords": ["snap", "crystal", "atom"]},
            "snaplat": {"controller_layer": 2, "adapter": "geometric_core", "keywords": ["snaplat", "lattice", "embedding"]},
            "tarpit": {"controller_layer": 1, "adapter": "morphonic_lambda", "keywords": ["tarpit", "glyph", "jot"]},
            "uhp": {"controller_layer": 4, "adapter": "governance_validation", "keywords": ["uhp", "anchor", "validation"]},
        }
        return profiles.get(family_key, {"controller_layer": 5, "adapter": "cmplx_interface", "keywords": [family_key]})

    def discover_families(self, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        if self._families and not refresh:
            return self._families

        families: Dict[str, Dict[str, Any]] = {}
        if not self.source_root.exists():
            self._families = families
            return families

        for family_dir in sorted([p for p in self.source_root.iterdir() if p.is_dir()]):
            family_key = self.normalize_family_name(family_dir.name)
            profile = self.family_language_profile(family_key)

            builds: List[FamilyBuild] = []
            for build_dir in sorted([p for p in family_dir.iterdir() if p.is_dir()]):
                tags = [family_key] + profile.get("keywords", [])
                builds.append(
                    FamilyBuild(
                        family_key=family_key,
                        family_name=family_dir.name,
                        build_name=build_dir.name,
                        build_path=str(build_dir),
                        tags=sorted(set(tags)),
                    )
                )

            families[family_key] = {
                "family_key": family_key,
                "family_name": family_dir.name,
                "path": str(family_dir),
                "controller_layer": profile["controller_layer"],
                "adapter": profile["adapter"],
                "keywords": profile.get("keywords", []),
                "builds": [b.__dict__ for b in builds],
            }

        self._families = families
        return families

    def list_families(self) -> List[Dict[str, Any]]:
        families = self.discover_families()
        return [
            {
                "family_key": meta["family_key"],
                "family_name": meta["family_name"],
                "controller_layer": meta["controller_layer"],
                "adapter": meta["adapter"],
                "build_count": len(meta["builds"]),
            }
            for _, meta in sorted(families.items())
        ]

    def get_family(self, family_key: str) -> Optional[Dict[str, Any]]:
        families = self.discover_families()
        normalized = self.normalize_family_name(family_key)
        return families.get(normalized)

    def build_wrappers_for_family(self, family_key: str) -> List[Dict[str, Any]]:
        family = self.get_family(family_key)
        if not family:
            return []

        wrappers: List[Dict[str, Any]] = []
        for build in family["builds"]:
            wrappers.append(
                {
                    "wrapper_id": f"{family['family_key']}::{build['build_name']}",
                    "family_key": family["family_key"],
                    "build_name": build["build_name"],
                    "build_path": build["build_path"],
                    "controller_layer": family["controller_layer"],
                    "adapter": family["adapter"],
                    "translation_mode": "read_only_adapt",
                    "status": "discoverable",
                }
            )
        return wrappers
