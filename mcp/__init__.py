"""
Model Context Protocol (MCP) integration for Agentix.

This module provides MCP client support, tool integration, and server discovery
for building MCP-compatible AI agents.
"""

from .client import MCPClient, MCPClientConfig
from .tools import MCPToolNode, MCPToolRegistry
from .server import MCPServerManager, MCPServerConfig
from .discovery import MCPDiscovery, discover_mcp_servers

__all__ = [
    # Core MCP client
    "MCPClient",
    "MCPClientConfig",
    
    # Tool integration
    "MCPToolNode", 
    "MCPToolRegistry",
    
    # Server management
    "MCPServerManager",
    "MCPServerConfig",
    
    # Discovery
    "MCPDiscovery",
    "discover_mcp_servers"
]
