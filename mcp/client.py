"""
Model Context Protocol (MCP) client implementation for Agentix.

This module provides a client for connecting to MCP servers and executing
tools through the MCP protocol, following the official MCP Python SDK patterns.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
import websockets

from ..utils.exceptions import AgentixError


class MCPError(AgentixError):
    """Exception raised when MCP operations fail."""

    def __init__(self, message: str, error_code: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.error_code = error_code
        if error_code:
            self.context['error_code'] = error_code


@dataclass
class MCPClientConfig:
    """Configuration for MCP client connections."""

    name: str
    transport: str = "stdio"  # stdio, websocket, http
    endpoint: Optional[str] = None
    command: Optional[List[str]] = None
    env: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    auth: Optional[Dict[str, Any]] = None
    capabilities: List[str] = field(default_factory=lambda: ["tools", "resources"])


@dataclass
class MCPTool:
    """Represents an MCP tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    server: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResource:
    """Represents an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None
    server: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPClient:
    """
    Model Context Protocol client for connecting to MCP servers.

    Supports multiple transport mechanisms:
    - stdio: Process-based communication
    - websocket: WebSocket connections
    - http: HTTP-based communication
    """

    def __init__(self, config: MCPClientConfig):
        """Initialize MCP client."""
        self.config = config
        self.logger = logging.getLogger(f"agentix.mcp.client.{config.name}")

        # Connection state
        self.connected = False
        self.session_id = str(uuid.uuid4())
        self.request_id = 0

        # Transport-specific connections
        self.process = None
        self.websocket = None
        self.http_session = None

        # Server capabilities and tools
        self.server_info = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}

        # Message handling
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.message_handlers = {
            "tools/list": self._handle_tools_list,
            "resources/list": self._handle_resources_list,
            "tools/call": self._handle_tool_call,
            "resources/read": self._handle_resource_read
        }

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connection successful, False otherwise
        """

        try:
            if self.config.transport == "stdio":
                await self._connect_stdio()
            elif self.config.transport == "websocket":
                await self._connect_websocket()
            elif self.config.transport == "http":
                await self._connect_http()
            else:
                raise MCPError(f"Unsupported transport: {self.config.transport}")

            # Initialize connection
            await self._initialize_connection()

            # Discover capabilities
            await self._discover_capabilities()

            self.connected = True
            self.logger.info(f"Connected to MCP server: {self.config.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {str(e)}")
            await self.disconnect()
            return False

    async def disconnect(self):
        """Disconnect from the MCP server."""

        self.connected = False

        # Close transport connections
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except:
                pass
            self.process = None

        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

        if self.http_session:
            try:
                await self.http_session.close()
            except:
                pass
            self.http_session = None

        # Cancel pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        self.pending_requests.clear()

        self.logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def list_tools(self) -> List[MCPTool]:
        """
        List available tools from the MCP server.

        Returns:
            List of available tools
        """

        if not self.connected:
            raise MCPError("Not connected to MCP server")

        try:
            response = await self._send_request("tools/list", {})

            tools = []
            for tool_data in response.get("tools", []):
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server=self.config.name,
                    metadata=tool_data.get("metadata", {})
                )
                tools.append(tool)
                self.tools[tool.name] = tool

            self.logger.info(f"Discovered {len(tools)} tools from {self.config.name}")
            return tools

        except Exception as e:
            raise MCPError(f"Failed to list tools: {str(e)}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """

        if not self.connected:
            raise MCPError("Not connected to MCP server")

        if tool_name not in self.tools:
            raise MCPError(f"Tool not found: {tool_name}")

        try:
            response = await self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })

            return response

        except Exception as e:
            raise MCPError(f"Failed to call tool {tool_name}: {str(e)}")

    async def list_resources(self) -> List[MCPResource]:
        """
        List available resources from the MCP server.

        Returns:
            List of available resources
        """

        if not self.connected:
            raise MCPError("Not connected to MCP server")

        try:
            response = await self._send_request("resources/list", {})

            resources = []
            for resource_data in response.get("resources", []):
                resource = MCPResource(
                    uri=resource_data["uri"],
                    name=resource_data.get("name", ""),
                    description=resource_data.get("description", ""),
                    mime_type=resource_data.get("mimeType"),
                    server=self.config.name,
                    metadata=resource_data.get("metadata", {})
                )
                resources.append(resource)
                self.resources[resource.uri] = resource

            self.logger.info(f"Discovered {len(resources)} resources from {self.config.name}")
            return resources

        except Exception as e:
            raise MCPError(f"Failed to list resources: {str(e)}")

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI to read

        Returns:
            Resource content
        """

        if not self.connected:
            raise MCPError("Not connected to MCP server")

        try:
            response = await self._send_request("resources/read", {
                "uri": uri
            })

            return response

        except Exception as e:
            raise MCPError(f"Failed to read resource {uri}: {str(e)}")

    async def _connect_stdio(self):
        """Connect using stdio transport."""

        if not self.config.command:
            raise MCPError("Command required for stdio transport")

        self.process = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.config.env}
        )

        # Start message reading task
        asyncio.create_task(self._read_stdio_messages())

    async def _connect_websocket(self):
        """Connect using WebSocket transport."""

        if not self.config.endpoint:
            raise MCPError("Endpoint required for websocket transport")

        headers = {}
        if self.config.auth:
            if "bearer" in self.config.auth:
                headers["Authorization"] = f"Bearer {self.config.auth['bearer']}"

        self.websocket = await websockets.connect(
            self.config.endpoint,
            extra_headers=headers,
            timeout=self.config.timeout
        )

        # Start message reading task
        asyncio.create_task(self._read_websocket_messages())

    async def _connect_http(self):
        """Connect using HTTP transport."""

        if not self.config.endpoint:
            raise MCPError("Endpoint required for HTTP transport")

        headers = {"Content-Type": "application/json"}
        if self.config.auth:
            if "bearer" in self.config.auth:
                headers["Authorization"] = f"Bearer {self.config.auth['bearer']}"

        self.http_session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )

    async def _initialize_connection(self):
        """Initialize the MCP connection."""

        init_message = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"listChanged": True, "subscribe": True}
                },
                "clientInfo": {
                    "name": "Agentix",
                    "version": "1.0.0"
                }
            }
        }

        response = await self._send_message(init_message)
        self.server_info = response.get("result", {})

    async def _discover_capabilities(self):
        """Discover server capabilities."""

        # List tools if supported
        if "tools" in self.config.capabilities:
            try:
                await self.list_tools()
            except Exception as e:
                self.logger.warning(f"Failed to list tools: {str(e)}")

        # List resources if supported
        if "resources" in self.config.capabilities:
            try:
                await self.list_resources()
            except Exception as e:
                self.logger.warning(f"Failed to list resources: {str(e)}")

    def _next_request_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request and wait for response."""

        message = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params
        }

        response = await self._send_message(message)

        if "error" in response:
            error = response["error"]
            raise MCPError(
                error.get("message", "Unknown MCP error"),
                error_code=error.get("code")
            )

        return response.get("result", {})

    async def _send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and wait for response."""

        request_id = message.get("id")
        if request_id:
            future = asyncio.Future()
            self.pending_requests[request_id] = future

        try:
            # Send message based on transport
            if self.config.transport == "stdio":
                await self._send_stdio_message(message)
            elif self.config.transport == "websocket":
                await self._send_websocket_message(message)
            elif self.config.transport == "http":
                return await self._send_http_message(message)

            # Wait for response
            if request_id:
                response = await asyncio.wait_for(future, timeout=self.config.timeout)
                return response

        finally:
            if request_id and request_id in self.pending_requests:
                del self.pending_requests[request_id]

        return {}

    async def _send_stdio_message(self, message: Dict[str, Any]):
        """Send message via stdio."""

        if not self.process or not self.process.stdin:
            raise MCPError("No stdio connection")

        data = json.dumps(message) + "\n"
        self.process.stdin.write(data.encode())
        await self.process.stdin.drain()

    async def _send_websocket_message(self, message: Dict[str, Any]):
        """Send message via WebSocket."""

        if not self.websocket:
            raise MCPError("No WebSocket connection")

        data = json.dumps(message)
        await self.websocket.send(data)

    async def _send_http_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message via HTTP."""

        if not self.http_session:
            raise MCPError("No HTTP session")

        async with self.http_session.post(self.config.endpoint, json=message) as response:
            if response.status != 200:
                raise MCPError(f"HTTP error {response.status}")

            return await response.json()

    async def _read_stdio_messages(self):
        """Read messages from stdio."""

        if not self.process or not self.process.stdout:
            return

        try:
            while self.connected and self.process.returncode is None:
                line = await self.process.stdout.readline()
                if not line:
                    break

                try:
                    message = json.loads(line.decode().strip())
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            self.logger.error(f"Error reading stdio messages: {str(e)}")

    async def _read_websocket_messages(self):
        """Read messages from WebSocket."""

        if not self.websocket:
            return

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            self.logger.error(f"Error reading WebSocket messages: {str(e)}")

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message."""

        message_id = message.get("id")

        # Handle response to pending request
        if message_id and message_id in self.pending_requests:
            future = self.pending_requests[message_id]
            if not future.done():
                future.set_result(message)
            return

        # Handle notifications and other messages
        method = message.get("method")
        if method in self.message_handlers:
            await self.message_handlers[method](message)

    async def _handle_tools_list(self, message: Dict[str, Any]):
        """Handle tools list notification."""
        # Refresh tools list
        await self.list_tools()

    async def _handle_resources_list(self, message: Dict[str, Any]):
        """Handle resources list notification."""
        # Refresh resources list
        await self.list_resources()

    async def _handle_tool_call(self, message: Dict[str, Any]):
        """Handle tool call notification."""
        # This would be for server-initiated tool calls
        pass

    async def _handle_resource_read(self, message: Dict[str, Any]):
        """Handle resource read notification."""
        # This would be for server-initiated resource reads
        pass
