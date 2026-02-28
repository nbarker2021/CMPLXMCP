"""
CMPLX Encoder/Decoder
=====================
Efficient serialization for lightweight client-server communication.
Handles are encoded with metadata; full data stays server-side.
"""

import json
import struct
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class HandleMetadata:
    """Metadata for a handle - sent to client."""
    handle: str
    data_type: str
    size_bytes: int
    checksum: str
    created_timestamp: str
    access_count: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


class CMPLXEncoder:
    """Encode data for transmission to client."""
    
    @staticmethod
    def encode_for_wire(data: Any, handle: str) -> bytes:
        """
        Encode data for wire transmission.
        For handles, only metadata is sent.
        For small data, full content may be sent.
        """
        # Check size
        json_data = json.dumps(data, default=str)
        size = len(json_data.encode())
        
        if size > 1024:  # 1KB threshold for handle-only mode
            # Large data: send handle reference only
            metadata = HandleMetadata(
                handle=handle,
                data_type=type(data).__name__,
                size_bytes=size,
                checksum="",
                created_timestamp=""
            )
            return json.dumps({
                "_handle": handle,
                "_metadata": metadata.to_dict(),
                "_lightweight": True
            }).encode()
        else:
            # Small data: send full content
            return json_data.encode()
    
    @staticmethod
    def encode_vector_preview(vector: list[float], full_dim: int, preview_dim: int = 8) -> dict:
        """Encode vector with preview - full data stays server-side."""
        return {
            "dimensions": full_dim,
            "preview": vector[:preview_dim],
            "norm": sum(x**2 for x in vector) ** 0.5,
            "_truncated": full_dim > preview_dim
        }


class CMPLXDecoder:
    """Decode data received from server."""
    
    @staticmethod
    def decode_from_wire(data: bytes) -> Any:
        """Decode wire data."""
        decoded = json.loads(data.decode())
        
        if isinstance(decoded, dict) and decoded.get("_lightweight"):
            # This is a handle reference
            return HandleReference(decoded["_handle"], decoded.get("_metadata", {}))
        
        return decoded


class HandleReference:
    """Local reference to server-side data."""
    
    def __init__(self, handle: str, metadata: dict):
        self.handle = handle
        self.metadata = HandleMetadata(**metadata)
    
    def __repr__(self):
        return f"HandleReference({self.handle}, {self.metadata.data_type})"
    
    def __str__(self):
        return self.handle


def encode_handle(handle: str, metadata: dict) -> str:
    """Quick encode handle to string."""
    return json.dumps({
        "handle": handle,
        "type": metadata.get("type", "unknown"),
        "lightweight": True
    })


def decode_handle(encoded: str) -> tuple[str, dict]:
    """Decode handle string to tuple."""
    data = json.loads(encoded)
    return data.get("handle"), data
