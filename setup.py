"""
CMPLX MCP OS Setup
==================
"""

from setuptools import setup, find_packages

setup(
    name="cmplx-mcp-os",
    version="1.0.0",
    description="CMPLX MCP Operating System",
    packages=find_packages(),
    install_requires=[
        "mcp>=1.0.0",
        "uvicorn>=0.24.0",
        "numpy>=1.21.0",
        "anyio>=4.0.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "black", "mypy"],
        "db": ["aiosqlite>=0.19.0"],
    },
    entry_points={
        "console_scripts": [
            "cmplx-mcp-server=mcp_os.__main__:main",
        ],
    },
    python_requires=">=3.10",
)
