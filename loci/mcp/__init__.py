"""MCP (Model Context Protocol) server exposing Loci as spatial memory for agents.

Run with ``loci-mcp`` (stdio transport).  Requires the ``mcp`` extra:
``pip install "loci-stdb[mcp]"``.  See ``docs/MCP_SERVER.md`` for the full
tool reference and Claude Desktop / Claude Code configuration.
"""

from loci.mcp.server import build_server, get_client, main, reset_client

__all__ = [
    "build_server",
    "get_client",
    "main",
    "reset_client",
]
