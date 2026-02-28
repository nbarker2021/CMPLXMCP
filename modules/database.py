"""
Database Module
===============
Lightweight database interface for the MCP OS.
Stores handles and metadata, not heavy data.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


class DatabaseManager:
    """
    Database manager for CMPLX MCP OS.
    
    Stores:
    - Handles (lightweight references)
    - Metadata
    - Pipeline state
    - Audit logs
    
    Does NOT store:
    - Full E8/Leech lattice data
    - Large embeddings
    - Generated artifacts (those stay server-side)
    """
    
    def __init__(self, db_path: Path | str = "cmplx_os.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS handles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    handle TEXT UNIQUE NOT NULL,
                    handle_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS pipelines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    stages TEXT,
                    result_handles TEXT
                );
                
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    tool_name TEXT,
                    handle TEXT,
                    success BOOLEAN,
                    details TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_handles_type ON handles(handle_type);
                CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_log(tool_name);
            """)
            conn.commit()
    
    @contextmanager
    def _connect(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    async def store(self, record: dict) -> int:
        """Store a record and return its ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO handles (handle, handle_type, created_at, metadata)
                   VALUES (?, ?, ?, ?)""",
                (
                    record.get("handles", {}).get("embedding", "unknown"),
                    "pipeline_result",
                    datetime.utcnow().isoformat(),
                    json.dumps(record.get("metadata", {}))
                )
            )
            conn.commit()
            return cursor.lastrowid
    
    async def log_action(self, action: str, tool_name: str | None = None,
                        handle: str | None = None, success: bool = True,
                        details: dict | None = None):
        """Log an action to audit log."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO audit_log (timestamp, action, tool_name, handle, success, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    action,
                    tool_name,
                    handle,
                    success,
                    json.dumps(details or {})
                )
            )
            conn.commit()
    
    async def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connect() as conn:
            handles = conn.execute("SELECT COUNT(*) FROM handles").fetchone()[0]
            pipelines = conn.execute("SELECT COUNT(*) FROM pipelines").fetchone()[0]
            audits = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            
            return {
                "handles": handles,
                "pipelines": pipelines,
                "audit_entries": audits,
                "db_path": str(self.db_path),
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
            }
