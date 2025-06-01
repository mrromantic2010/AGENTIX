"""
MCP tool integration for Agentix seven-node blueprint.

This module provides MCPToolNode for integrating MCP tools into the
Agentix agent execution graph.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from ..core.nodes.base import BaseNode, NodeConfig, NodeState
from ..core.nodes.tool import ToolNode, ToolNodeConfig
from ..utils.exceptions import NodeExecutionError
from .client import MCPClient, MCPClientConfig, MCPTool, MCPError


@dataclass
class MCPToolNodeConfig(NodeConfig):
    """Configuration for MCP tool node."""
    
    node_type: str = "mcp_tool"
    mcp_servers: List[MCPClientConfig] = field(default_factory=list)
    tool_filter: Optional[List[str]] = None  # Filter specific tools
    auto_discover: bool = True
    timeout: int = 30
    max_retries: int = 3


class MCPToolRegistry:
    """Registry for managing MCP tools across multiple servers."""
    
    def __init__(self):
        """Initialize MCP tool registry."""
        self.logger = logging.getLogger("agentix.mcp.registry")
        self.clients: Dict[str, MCPClient] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.server_tools: Dict[str, List[str]] = {}
    
    async def register_server(self, config: MCPClientConfig) -> bool:
        """
        Register an MCP server.
        
        Args:
            config: MCP client configuration
            
        Returns:
            True if registration successful
        """
        
        try:
            client = MCPClient(config)
            
            # Connect to server
            if await client.connect():
                self.clients[config.name] = client
                
                # Discover tools
                tools = await client.list_tools()
                tool_names = []
                
                for tool in tools:
                    # Use server-prefixed name to avoid conflicts
                    tool_key = f"{config.name}:{tool.name}"
                    self.tools[tool_key] = tool
                    tool_names.append(tool.name)
                
                self.server_tools[config.name] = tool_names
                
                self.logger.info(f"Registered MCP server '{config.name}' with {len(tools)} tools")
                return True
            else:
                self.logger.error(f"Failed to connect to MCP server: {config.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to register MCP server {config.name}: {str(e)}")
            return False
    
    async def unregister_server(self, server_name: str):
        """Unregister an MCP server."""
        
        if server_name in self.clients:
            await self.clients[server_name].disconnect()
            del self.clients[server_name]
        
        # Remove tools from this server
        if server_name in self.server_tools:
            for tool_name in self.server_tools[server_name]:
                tool_key = f"{server_name}:{tool_name}"
                if tool_key in self.tools:
                    del self.tools[tool_key]
            del self.server_tools[server_name]
        
        self.logger.info(f"Unregistered MCP server: {server_name}")
    
    def list_tools(self, server_filter: Optional[str] = None) -> List[MCPTool]:
        """
        List available MCP tools.
        
        Args:
            server_filter: Optional server name filter
            
        Returns:
            List of available tools
        """
        
        if server_filter:
            return [tool for key, tool in self.tools.items() 
                   if key.startswith(f"{server_filter}:")]
        
        return list(self.tools.values())
    
    def get_tool(self, tool_name: str, server_name: Optional[str] = None) -> Optional[MCPTool]:
        """
        Get a specific tool.
        
        Args:
            tool_name: Name of the tool
            server_name: Optional server name
            
        Returns:
            MCPTool if found, None otherwise
        """
        
        if server_name:
            tool_key = f"{server_name}:{tool_name}"
            return self.tools.get(tool_key)
        
        # Search across all servers
        for key, tool in self.tools.items():
            if tool.name == tool_name:
                return tool
        
        return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], 
                       server_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            server_name: Optional server name
            
        Returns:
            Tool execution result
        """
        
        tool = self.get_tool(tool_name, server_name)
        if not tool:
            raise MCPError(f"Tool not found: {tool_name}")
        
        client = self.clients.get(tool.server)
        if not client:
            raise MCPError(f"Server not connected: {tool.server}")
        
        return await client.call_tool(tool_name, arguments)
    
    async def shutdown(self):
        """Shutdown all MCP connections."""
        
        for client in self.clients.values():
            await client.disconnect()
        
        self.clients.clear()
        self.tools.clear()
        self.server_tools.clear()


class MCPToolNode(BaseNode):
    """
    MCP tool node for the seven-node blueprint.
    
    This node integrates MCP tools into the Agentix execution graph,
    allowing agents to use tools from multiple MCP servers.
    """
    
    def __init__(self, config: MCPToolNodeConfig):
        """Initialize MCP tool node."""
        super().__init__(config)
        self.config = config
        self.registry = MCPToolRegistry()
        self.initialized = False
    
    async def initialize(self):
        """Initialize MCP connections."""
        
        if self.initialized:
            return
        
        self.logger.info("Initializing MCP tool node...")
        
        # Register all configured servers
        for server_config in self.config.mcp_servers:
            await self.registry.register_server(server_config)
        
        self.initialized = True
        self.logger.info(f"MCP tool node initialized with {len(self.registry.clients)} servers")
    
    async def execute(self, state: NodeState) -> NodeState:
        """
        Execute MCP tool operations.
        
        Args:
            state: Current node state
            
        Returns:
            Updated node state
        """
        
        try:
            # Initialize if needed
            if not self.initialized:
                await self.initialize()
            
            # Extract tool call information from state
            tool_call = self._extract_tool_call(state)
            
            if not tool_call:
                # No tool call requested, pass through
                state.data["mcp_result"] = {"status": "no_tool_call"}
                return state
            
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            server_name = tool_call.get("server")
            
            if not tool_name:
                raise NodeExecutionError("No tool name specified")
            
            # Check if tool is available
            tool = self.registry.get_tool(tool_name, server_name)
            if not tool:
                available_tools = [t.name for t in self.registry.list_tools()]
                raise NodeExecutionError(
                    f"Tool '{tool_name}' not found. Available tools: {available_tools}"
                )
            
            # Validate arguments against tool schema
            self._validate_arguments(tool, arguments)
            
            # Execute tool
            self.logger.info(f"Executing MCP tool: {tool_name} on server: {tool.server}")
            
            result = await self.registry.call_tool(tool_name, arguments, server_name)
            
            # Store result in state
            state.data["mcp_result"] = {
                "status": "success",
                "tool_name": tool_name,
                "server": tool.server,
                "arguments": arguments,
                "result": result,
                "metadata": {
                    "execution_time": state.execution_time,
                    "tool_description": tool.description
                }
            }
            
            self.logger.info(f"MCP tool '{tool_name}' executed successfully")
            return state
            
        except Exception as e:
            error_msg = f"MCP tool execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            state.data["mcp_result"] = {
                "status": "error",
                "error": error_msg,
                "tool_name": tool_call.get("name") if tool_call else None
            }
            
            if self.config.strict_mode:
                raise NodeExecutionError(error_msg)
            
            return state
    
    def _extract_tool_call(self, state: NodeState) -> Optional[Dict[str, Any]]:
        """Extract tool call information from state."""
        
        # Check for explicit MCP tool call
        if "mcp_tool_call" in state.data:
            return state.data["mcp_tool_call"]
        
        # Check for general tool call that might be MCP
        if "tool_call" in state.data:
            tool_call = state.data["tool_call"]
            
            # Check if this is an MCP tool
            tool_name = tool_call.get("name")
            if tool_name and self.registry.get_tool(tool_name):
                return tool_call
        
        # Check for LLM function call
        if "function_call" in state.data:
            function_call = state.data["function_call"]
            function_name = function_call.get("name")
            
            if function_name and self.registry.get_tool(function_name):
                return {
                    "name": function_name,
                    "arguments": json.loads(function_call.get("arguments", "{}"))
                }
        
        return None
    
    def _validate_arguments(self, tool: MCPTool, arguments: Dict[str, Any]):
        """Validate tool arguments against schema."""
        
        # Basic validation - could be enhanced with jsonschema
        schema = tool.input_schema
        
        if "required" in schema:
            for required_field in schema["required"]:
                if required_field not in arguments:
                    raise NodeExecutionError(
                        f"Missing required argument '{required_field}' for tool '{tool.name}'"
                    )
        
        # Type validation could be added here
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available MCP tools for LLM function calling.
        
        Returns:
            List of tool definitions for LLM
        """
        
        if not self.initialized:
            await self.initialize()
        
        tools = []
        
        for tool in self.registry.list_tools():
            # Apply tool filter if configured
            if self.config.tool_filter and tool.name not in self.config.tool_filter:
                continue
            
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                },
                "metadata": {
                    "server": tool.server,
                    "mcp_tool": True
                }
            }
            
            tools.append(tool_def)
        
        return tools
    
    async def shutdown(self):
        """Shutdown MCP connections."""
        await self.registry.shutdown()
        self.initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get node status information."""
        
        return {
            "node_type": self.config.node_type,
            "initialized": self.initialized,
            "servers": list(self.registry.clients.keys()),
            "tools_count": len(self.registry.tools),
            "server_status": {
                name: client.connected 
                for name, client in self.registry.clients.items()
            }
        }


# Global MCP tool registry for convenience
_global_mcp_registry: Optional[MCPToolRegistry] = None


def get_global_mcp_registry() -> MCPToolRegistry:
    """Get the global MCP tool registry."""
    global _global_mcp_registry
    if _global_mcp_registry is None:
        _global_mcp_registry = MCPToolRegistry()
    return _global_mcp_registry


async def register_mcp_server(config: MCPClientConfig) -> bool:
    """Register an MCP server globally."""
    registry = get_global_mcp_registry()
    return await registry.register_server(config)


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any], 
                       server_name: Optional[str] = None) -> Dict[str, Any]:
    """Call an MCP tool globally."""
    registry = get_global_mcp_registry()
    return await registry.call_tool(tool_name, arguments, server_name)


def list_mcp_tools(server_filter: Optional[str] = None) -> List[MCPTool]:
    """List available MCP tools globally."""
    registry = get_global_mcp_registry()
    return registry.list_tools(server_filter)
