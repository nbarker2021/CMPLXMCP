"""
Pipeline Module
===============
Data flow pipeline between work, database, and tools.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""
    data: Any
    metadata: dict = field(default_factory=dict)
    handles: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def with_handle(self, name: str, handle: str):
        """Add a handle to context."""
        self.handles[name] = handle
        return self


class PipelineStage(ABC):
    """Base class for pipeline stages."""
    
    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext:
        pass


class IntakeStage(PipelineStage):
    """Stage 1: Intake data from files/API."""
    
    def __init__(self, watch_path: Path | None = None):
        self.watch_path = watch_path
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """Intake raw data."""
        # Validate and normalize intake
        context.metadata["intake_time"] = datetime.utcnow().isoformat()
        context.metadata["intake_type"] = type(context.data).__name__
        return context


class GeometricStage(PipelineStage):
    """Stage 2: Project to geometric space."""
    
    def __init__(self, client=None):
        self._client = client
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """Project data to E8/Leech space."""
        if not self._client:
            return context
        
        # Embed content
        if isinstance(context.data, str):
            result = await self._client.embed(context.data, domain="text")
            context.with_handle("embedding", result.get("handle"))
            context.metadata["embedded"] = True
        
        return context


class ValidationStage(PipelineStage):
    """Stage 3: Validate through governance."""
    
    def __init__(self, client=None):
        self._client = client
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """Run seven-witness validation."""
        if not self._client:
            return context
        
        artifact = {
            "handles": context.handles,
            "metadata": context.metadata
        }
        
        result = await self._client.seven_witness(artifact)
        context.metadata["validation"] = result
        context.metadata["valid"] = result.get("all_valid", False)
        
        return context


class StorageStage(PipelineStage):
    """Stage 4: Store to database."""
    
    def __init__(self, db_manager=None):
        self._db = db_manager
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """Store handles and metadata to database."""
        if not self._db:
            return context
        
        # Store lightweight metadata, not heavy data
        record = {
            "handles": context.handles,
            "metadata": context.metadata,
            "stored_at": datetime.utcnow().isoformat()
        }
        
        record_id = await self._db.store(record)
        context.metadata["record_id"] = record_id
        
        return context


class Pipeline:
    """
    Data processing pipeline.
    
    Flow: Intake -> Geometric -> Validation -> Storage
    """
    
    def __init__(self, client=None, db_manager=None):
        self._client = client
        self._db = db_manager
        self._stages: list[PipelineStage] = []
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build default pipeline."""
        self._stages = [
            IntakeStage(),
            GeometricStage(self._client),
            ValidationStage(self._client),
            StorageStage(self._db)
        ]
    
    async def process(self, data: Any) -> PipelineContext:
        """Process data through all stages."""
        context = PipelineContext(data=data)
        
        for stage in self._stages:
            context = await stage.process(context)
        
        return context
    
    def add_stage(self, stage: PipelineStage):
        """Add a custom stage."""
        self._stages.append(stage)
