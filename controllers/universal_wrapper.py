"""Universal wrapper for legacy/off-runpath controller candidates.

Provides a single adapter surface to:
1) normalize wrapper metadata,
2) emit wrapper records into MMDB+MDHG+SpeedLight,
3) run a ThinkTank iteration and retrieve proposal hints.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from cmplx_toolkit.controllers.memory_pipeline import \
    MMDBMDHGSpeedLightPipeline

from .family_manager import FamilyControllerManager


@dataclass
class WrappedControllerSpec:
    wrapper_id: str
    family_key: str
    class_name: str
    source_registry: str
    build_path: str = ""
    controller_layer: int = 5
    adapter: str = "cmplx_interface"
    translation_mode: str = "read_only_adapt"
    tags: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wrapper_id": self.wrapper_id,
            "family_key": self.family_key,
            "class_name": self.class_name,
            "source_registry": self.source_registry,
            "build_path": self.build_path,
            "controller_layer": self.controller_layer,
            "adapter": self.adapter,
            "translation_mode": self.translation_mode,
            "tags": self.tags,
            "conflict_flags": self.conflict_flags,
        }


class UniversalControllerWrapper:
    """Single wrapper surface for broad controller onboarding."""

    def __init__(
        self,
        *,
        family_manager: Optional[FamilyControllerManager] = None,
        pipeline: Optional[MMDBMDHGSpeedLightPipeline] = None,
        db_path: str = ".cmplx/mmdb/mmdb.sqlite3",
    ) -> None:
        self.family_manager = family_manager or FamilyControllerManager()
        self.db_path = db_path
        self.pipeline = pipeline

    def _ensure_pipeline(self) -> MMDBMDHGSpeedLightPipeline:
        if self.pipeline is None:
            self.pipeline = MMDBMDHGSpeedLightPipeline(
                db_path=self.db_path,
                data_root=str(Path(self.db_path).parent),
                speedlight_language="base100",
            )
        return self.pipeline

    def build_wrapper(
        self,
        *,
        family_key: str,
        class_name: str,
        source_registry: str,
        build_path: str = "",
    ) -> WrappedControllerSpec:
        normalized_family = self.family_manager.normalize_family_name(
            family_key,
        )
        profile = self.family_manager.family_language_profile(
            normalized_family,
        )

        conflict_flags = self.detect_conflicts(
            class_name=class_name,
            build_path=build_path,
        )

        return WrappedControllerSpec(
            wrapper_id=(
                f"uw:{normalized_family}:{class_name}:{uuid.uuid4().hex[:8]}"
            ),
            family_key=normalized_family,
            class_name=class_name,
            source_registry=source_registry,
            build_path=build_path,
            controller_layer=int(profile.get("controller_layer", 5)),
            adapter=str(profile.get("adapter", "cmplx_interface")),
            tags=list(profile.get("keywords", [])),
            conflict_flags=conflict_flags,
        )

    @staticmethod
    def detect_conflicts(
        *,
        class_name: str,
        build_path: str = "",
    ) -> List[str]:
        flags: List[str] = []

        if re.search(r"(^|_)test|^test", class_name, flags=re.IGNORECASE):
            flags.append("test_artifact")
        if class_name.startswith("_"):
            flags.append("private_symbol")

        if build_path:
            path = Path(build_path)
            if not path.exists():
                flags.append("missing_build_path")

        return flags

    async def emit_to_system(
        self,
        *,
        spec: WrappedControllerSpec,
        design_need: str,
        level: str = "planetary",
    ) -> Dict[str, Any]:
        pipeline = self._ensure_pipeline()

        payload = {
            "wrapper": spec.to_dict(),
            "design_need": design_need,
            "suggested_controller": {
                "layer": spec.controller_layer,
                "adapter": spec.adapter,
            },
        }

        return await pipeline.ingest(
            {
                "level": level,
                "corpus_label": "universal_wrapper",
                "source_path": (
                    f"wrapper://{spec.family_key}/{spec.class_name}"
                ),
                "content": json.dumps(payload, ensure_ascii=False),
                "metadata": {
                    "family_key": spec.family_key,
                    "class_name": spec.class_name,
                    "wrapper_id": spec.wrapper_id,
                    "controller_layer": spec.controller_layer,
                    "adapter": spec.adapter,
                    "conflicts": spec.conflict_flags,
                    "design_need": design_need,
                },
                "terms": [
                    spec.family_key,
                    spec.class_name,
                    spec.adapter,
                    "universal_wrapper",
                    "thinktank",
                ],
            }
        )

    @staticmethod
    async def run_thinktank_iteration(
        cmplx_registry: Any,
        top_k: int = 5,
        timeout_sec: float = 20.0,
    ) -> Dict[str, Any]:
        async def _call_with_timeout(coro: Any) -> Any:
            return await asyncio.wait_for(coro, timeout=timeout_sec)

        timed_out = False
        status: Dict[str, Any] = {}
        session: Dict[str, Any] = {}
        proposals: Dict[str, Any] = {}

        try:
            status = await _call_with_timeout(
                cmplx_registry._handle_think_tank_status(
                    {},
                    cmplx_registry,
                )
            )
            session = await _call_with_timeout(
                cmplx_registry._handle_think_tank_run_session(
                    {},
                    cmplx_registry,
                )
            )
            proposals = await _call_with_timeout(
                cmplx_registry._handle_think_tank_get_proposals(
                    {},
                    cmplx_registry,
                )
            )
        except TimeoutError:
            timed_out = True
            proposals = {"proposals": []}

        items = proposals.get("proposals", [])
        summary = []
        for proposal in items[:top_k]:
            summary.append(
                {
                    "id": proposal.get("id"),
                    "title": proposal.get("title"),
                    "type": proposal.get("type"),
                    "severity": proposal.get("severity"),
                }
            )

        return {
            "timed_out": timed_out,
            "timeout_sec": timeout_sec,
            "thinktank_status": status,
            "session": session,
            "proposal_count": len(items),
            "proposals": summary,
        }
