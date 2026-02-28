"""
CMPLX MCP Client - Lightweight Local Runtime
=============================================
Thin client that proxies all operations to the MCP server.
No heavy data stored locally - only handles and references.
"""

import asyncio
import json
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client

try:
    from mcp.client.stdio import StdioServerParameters
except Exception:
    StdioServerParameters = None


class CMPLXClient:
    """
    Lightweight CMPLX client for local use.
    
    All operations are sent to the MCP server.
    Only lightweight handles are stored locally.
    """
    
    def __init__(self, server_command: list[str] | None = None):
        self.server_command = (
            server_command or [sys.executable, "-m", "mcp_os", "server"]
        )
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._read = None
        self._write = None
    
    async def connect(self):
        """Connect to the MCP server."""
        # Start server process and connect via stdio
        server_params: Any = self.server_command
        if isinstance(self.server_command, list):
            command = self.server_command[0] if self.server_command else "python"
            args = self.server_command[1:] if len(self.server_command) > 1 else []

            if StdioServerParameters is not None:
                server_params = StdioServerParameters(command=command, args=args)
            else:
                class _CompatServerParams:
                    def __init__(self, command: str, args: list[str]):
                        self.command = command
                        self.args = args

                server_params = _CompatServerParams(command=command, args=args)

        self._stdio_ctx = stdio_client(server_params)
        self._read, self._write = await self._stdio_ctx.__aenter__()
        self._session = await ClientSession(self._read, self._write).__aenter__()
        await self._session.initialize()
    
    async def disconnect(self):
        """Disconnect from server."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._stdio_ctx is not None:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None
            self._read = None
            self._write = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def call(self, tool: str, **kwargs) -> dict:
        """Call a tool on the MCP server."""
        if not self._session:
            raise RuntimeError("Not connected to server")
        
        result = await self._session.call_tool(tool, kwargs)
        
        # Parse JSON response
        if result.content and len(result.content) > 0:
            text = result.content[0].text
            return json.loads(text)
        return {}
    
    # ===== Layer 1: Morphonic Foundation =====
    
    async def generate_morphon(self, seed: str) -> dict:
        """Generate a universal morphon from seed."""
        return await self.call("l1_morphon_generate", seed=seed)
    
    async def execute_mglc(self, expression: str, context: dict | None = None) -> dict:
        """Execute MGLC expression."""
        return await self.call(
            "l1_mglc_execute",
            expression=expression,
            context=context or {},
        )
    
    async def expand_seed(self, digit: int, dimensions: int = 24) -> dict:
        """Expand single digit to 24D substrate."""
        return await self.call("l1_seed_expand", digit=digit, dimensions=dimensions)
    
    # ===== Layer 2: Geometric Engine =====
    
    async def project_e8(self, vector: list[float], return_format: str = "minimal") -> dict:
        """Project 8D vector to E8 lattice."""
        return await self.call(
            "l2_e8_project",
            vector=vector,
            return_format=return_format,
        )
    
    async def nearest_leech(self, vector: list[float], return_format: str = "handle") -> dict:
        """Find nearest point in Leech lattice."""
        return await self.call(
            "l2_leech_nearest",
            vector=vector,
            return_format=return_format,
        )
    
    async def navigate_weyl(
        self,
        position: list[float],
        target_root: list[float] | None = None,
    ) -> dict:
        """Navigate Weyl chambers."""
        kwargs = {"position": position}
        if target_root:
            kwargs["target_root"] = target_root
        return await self.call("l2_weyl_navigate", **kwargs)
    
    async def classify_niemeier(self, vector: list[float]) -> dict:
        """Classify against Niemeier lattices."""
        return await self.call("l2_niemeier_classify", vector=vector)
    
    # ===== Layer 3: Operational Systems =====
    
    async def morsr_optimize(
        self,
        initial_state: list[float],
        iterations: int = 100,
        constraint: str = "conservation",
    ) -> dict:
        """Run MORSR optimization."""
        return await self.call(
            "l3_morsr_optimize",
            initial_state=initial_state,
            iterations=iterations,
            constraint=constraint,
        )
    
    async def check_conservation(self, before: list[float], after: list[float]) -> dict:
        """Check ΔΦ ≤ 0 conservation law."""
        return await self.call("l3_conservation_check", before=before, after=after)
    
    # ===== Layer 4: Governance =====
    
    async def digital_root(self, number: int, modulus: int = 9) -> dict:
        """Calculate digital root."""
        return await self.call("l4_digital_root", number=number, modulus=modulus)
    
    async def seven_witness(
        self,
        artifact: dict,
        perspectives: list[str] | None = None,
    ) -> dict:
        """Run seven-witness validation."""
        kwargs = {"artifact": artifact}
        if perspectives:
            kwargs["perspectives"] = perspectives
        return await self.call("l4_seven_witness", **kwargs)
    
    async def policy_check(self, artifact_id: str, policy_tier: int = 1) -> dict:
        """Check against policy hierarchy."""
        return await self.call(
            "l4_policy_check",
            artifact_id=artifact_id,
            policy_tier=policy_tier,
        )
    
    # ===== Layer 5: Interface =====
    
    async def embed(self, content: str, domain: str = "text", return_handle: bool = True) -> dict:
        """Embed content into E8 space."""
        return await self.call(
            "l5_embed",
            content=content,
            domain=domain,
            return_handle=return_handle,
        )
    
    async def query_similar(self, handle: str, top_k: int = 10) -> dict:
        """Query similar overlays."""
        return await self.call("l5_query_similar", handle=handle, top_k=top_k)
    
    async def transform(self, handle: str, operator: str, params: dict | None = None) -> dict:
        """Apply geometric transformation."""
        return await self.call(
            "l5_transform",
            handle=handle,
            operator=operator,
            params=params or {},
        )
    
    # ===== System Tools =====
    
    async def sys_info(self) -> dict:
        """Get system information."""
        return await self.call("sys_info")
    
    async def cache_stats(self) -> dict:
        """Get cache statistics."""
        return await self.call("sys_cache_stats")
    
    async def resolve_handle(self, handle: str, max_size_mb: float = 10) -> dict:
        """Resolve handle to full data."""
        return await self.call("sys_resolve_handle", handle=handle, max_size_mb=max_size_mb)


def create_client(server_command: list[str] | None = None) -> CMPLXClient:
    """Factory function to create a client."""
    return CMPLXClient(server_command=server_command)


# Example usage
async def main():
    """Example of using the client."""
    async with create_client() as client:
        # System info
        info = await client.sys_info()
        print(f"Connected to: {info}")
        
        # Generate morphon
        morphon = await client.generate_morphon("7")
        print(f"Generated: {morphon}")
        
        # Project to E8
        e8_result = await client.project_e8([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        print(f"E8 projection: {e8_result}")
        
        # Digital root
        dr = await client.digital_root(432)
        print(f"Digital root: {dr}")


if __name__ == "__main__":
    asyncio.run(main())
