"""
MCP server discovery for Agentix.

This module provides discovery mechanisms for finding and configuring
MCP servers and tools.
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import aiohttp

from .server import MCPServerConfig, MCPServerManager


class MCPDiscovery:
    """
    MCP server discovery service.
    
    Discovers available MCP servers through various mechanisms:
    - Local system scanning
    - Package manager integration
    - Registry lookups
    - Network discovery
    """
    
    def __init__(self):
        """Initialize MCP discovery."""
        self.logger = logging.getLogger("agentix.mcp.discovery")
        self.discovered_servers: Dict[str, MCPServerConfig] = {}
    
    async def discover_all(self) -> List[MCPServerConfig]:
        """
        Discover all available MCP servers.
        
        Returns:
            List of discovered server configurations
        """
        
        self.logger.info("Starting MCP server discovery...")
        
        # Discover from various sources
        await asyncio.gather(
            self._discover_system_commands(),
            self._discover_npm_packages(),
            self._discover_python_packages(),
            self._discover_registry(),
            return_exceptions=True
        )
        
        servers = list(self.discovered_servers.values())
        self.logger.info(f"Discovered {len(servers)} MCP servers")
        
        return servers
    
    async def _discover_system_commands(self):
        """Discover MCP servers from system commands."""
        
        try:
            # Common MCP command patterns
            patterns = [
                "mcp-*",
                "*-mcp",
                "agentix-mcp-*"
            ]
            
            # Search in PATH
            path_dirs = []
            import os
            for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                if path_dir and Path(path_dir).exists():
                    path_dirs.append(Path(path_dir))
            
            for path_dir in path_dirs:
                for pattern in patterns:
                    for executable in path_dir.glob(pattern):
                        if executable.is_file() and os.access(executable, os.X_OK):
                            await self._analyze_executable(executable)
            
        except Exception as e:
            self.logger.error(f"Error discovering system commands: {str(e)}")
    
    async def _discover_npm_packages(self):
        """Discover MCP servers from npm packages."""
        
        try:
            # Check if npm is available
            result = subprocess.run(
                ["npm", "list", "-g", "--depth=0", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                dependencies = data.get("dependencies", {})
                
                for package_name, package_info in dependencies.items():
                    if "mcp" in package_name.lower():
                        await self._analyze_npm_package(package_name, package_info)
            
        except Exception as e:
            self.logger.debug(f"Error discovering npm packages: {str(e)}")
    
    async def _discover_python_packages(self):
        """Discover MCP servers from Python packages."""
        
        try:
            # Check installed packages
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                
                for package in packages:
                    package_name = package.get("name", "")
                    if "mcp" in package_name.lower():
                        await self._analyze_python_package(package_name, package.get("version"))
            
        except Exception as e:
            self.logger.debug(f"Error discovering Python packages: {str(e)}")
    
    async def _discover_registry(self):
        """Discover MCP servers from online registry."""
        
        try:
            # Check Agentix MCP registry
            registry_url = "https://registry.agentix.dev/mcp/servers"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(registry_url, timeout=10) as response:
                    if response.status == 200:
                        registry_data = await response.json()
                        
                        for server_data in registry_data.get("servers", []):
                            self._add_discovered_server(server_data)
            
        except Exception as e:
            self.logger.debug(f"Error discovering from registry: {str(e)}")
    
    async def _analyze_executable(self, executable: Path):
        """Analyze an executable to determine if it's an MCP server."""
        
        try:
            # Try to get MCP server info
            result = subprocess.run(
                [str(executable), "--mcp-info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                try:
                    info = json.loads(result.stdout)
                    self._create_server_from_executable(executable, info)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: create basic config
            else:
                self._create_basic_server_config(executable)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing executable {executable}: {str(e)}")
    
    async def _analyze_npm_package(self, package_name: str, package_info: Dict[str, Any]):
        """Analyze an npm package for MCP server capabilities."""
        
        try:
            # Get package.json info
            result = subprocess.run(
                ["npm", "view", package_name, "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                package_data = json.loads(result.stdout)
                
                # Check for MCP keywords
                keywords = package_data.get("keywords", [])
                if "mcp" in keywords or "model-context-protocol" in keywords:
                    self._create_server_from_npm_package(package_name, package_data)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing npm package {package_name}: {str(e)}")
    
    async def _analyze_python_package(self, package_name: str, version: str):
        """Analyze a Python package for MCP server capabilities."""
        
        try:
            # Try to import and check for MCP server
            import importlib
            
            module = importlib.import_module(package_name.replace("-", "_"))
            
            # Check for MCP server attributes
            if hasattr(module, "__mcp_server__") or hasattr(module, "mcp_main"):
                self._create_server_from_python_package(package_name, version, module)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing Python package {package_name}: {str(e)}")
    
    def _create_server_from_executable(self, executable: Path, info: Dict[str, Any]):
        """Create server config from executable info."""
        
        server_name = info.get("name", executable.stem)
        
        config = MCPServerConfig(
            name=server_name,
            description=info.get("description", f"MCP server: {server_name}"),
            transport="stdio",
            command=[str(executable)],
            capabilities=info.get("capabilities", ["tools"]),
            metadata={
                "discovered": True,
                "source": "executable",
                "path": str(executable),
                "version": info.get("version")
            }
        )
        
        self.discovered_servers[server_name] = config
    
    def _create_basic_server_config(self, executable: Path):
        """Create basic server config for unknown executable."""
        
        server_name = executable.stem
        
        config = MCPServerConfig(
            name=server_name,
            description=f"Discovered MCP server: {server_name}",
            transport="stdio",
            command=[str(executable)],
            enabled=False,  # Disabled by default for unknown servers
            metadata={
                "discovered": True,
                "source": "executable",
                "path": str(executable),
                "verified": False
            }
        )
        
        self.discovered_servers[server_name] = config
    
    def _create_server_from_npm_package(self, package_name: str, package_data: Dict[str, Any]):
        """Create server config from npm package."""
        
        # Determine command
        bin_info = package_data.get("bin", {})
        if isinstance(bin_info, dict):
            command_name = list(bin_info.keys())[0] if bin_info else package_name
        else:
            command_name = package_name
        
        config = MCPServerConfig(
            name=package_name,
            description=package_data.get("description", f"npm MCP server: {package_name}"),
            transport="stdio",
            command=[command_name],
            metadata={
                "discovered": True,
                "source": "npm",
                "package": package_name,
                "version": package_data.get("version"),
                "homepage": package_data.get("homepage")
            }
        )
        
        self.discovered_servers[package_name] = config
    
    def _create_server_from_python_package(self, package_name: str, version: str, module):
        """Create server config from Python package."""
        
        # Try to get MCP info from module
        mcp_info = getattr(module, "__mcp_server__", {})
        
        config = MCPServerConfig(
            name=package_name,
            description=mcp_info.get("description", f"Python MCP server: {package_name}"),
            transport="stdio",
            command=["python", "-m", package_name.replace("-", "_")],
            capabilities=mcp_info.get("capabilities", ["tools"]),
            metadata={
                "discovered": True,
                "source": "python",
                "package": package_name,
                "version": version,
                "module": module.__name__
            }
        )
        
        self.discovered_servers[package_name] = config
    
    def _add_discovered_server(self, server_data: Dict[str, Any]):
        """Add a server from registry data."""
        
        try:
            config = MCPServerConfig(**server_data)
            config.metadata["discovered"] = True
            config.metadata["source"] = "registry"
            
            self.discovered_servers[config.name] = config
            
        except Exception as e:
            self.logger.error(f"Invalid server data from registry: {str(e)}")
    
    def get_discovered_servers(self) -> List[MCPServerConfig]:
        """Get all discovered servers."""
        return list(self.discovered_servers.values())
    
    def get_server_by_category(self, category: str) -> List[MCPServerConfig]:
        """Get servers by category."""
        return [
            server for server in self.discovered_servers.values()
            if server.metadata.get("category") == category
        ]


async def discover_mcp_servers() -> List[MCPServerConfig]:
    """
    Discover available MCP servers.
    
    Returns:
        List of discovered server configurations
    """
    
    discovery = MCPDiscovery()
    return await discovery.discover_all()


async def install_mcp_server(server_name: str, source: str = "auto") -> bool:
    """
    Install an MCP server.
    
    Args:
        server_name: Name of the server to install
        source: Installation source (npm, pip, auto)
        
    Returns:
        True if installation successful
    """
    
    logger = logging.getLogger("agentix.mcp.install")
    
    try:
        if source == "auto":
            # Try npm first, then pip
            sources = ["npm", "pip"]
        else:
            sources = [source]
        
        for install_source in sources:
            if install_source == "npm":
                result = subprocess.run(
                    ["npm", "install", "-g", server_name],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    logger.info(f"Installed MCP server '{server_name}' via npm")
                    return True
            
            elif install_source == "pip":
                result = subprocess.run(
                    ["pip", "install", server_name],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    logger.info(f"Installed MCP server '{server_name}' via pip")
                    return True
        
        logger.error(f"Failed to install MCP server: {server_name}")
        return False
        
    except Exception as e:
        logger.error(f"Error installing MCP server {server_name}: {str(e)}")
        return False


def create_mcp_server_template(name: str, category: str = "general") -> MCPServerConfig:
    """
    Create a template MCP server configuration.
    
    Args:
        name: Server name
        category: Server category
        
    Returns:
        Template server configuration
    """
    
    templates = {
        "filesystem": MCPServerConfig(
            name=name,
            description="Filesystem access server",
            transport="stdio",
            command=["agentix-mcp-filesystem"],
            capabilities=["tools", "resources"]
        ),
        "database": MCPServerConfig(
            name=name,
            description="Database access server",
            transport="stdio",
            command=["agentix-mcp-database"],
            env={"DATABASE_URL": "${DATABASE_URL}"},
            capabilities=["tools", "resources"]
        ),
        "api": MCPServerConfig(
            name=name,
            description="HTTP API client server",
            transport="stdio",
            command=["agentix-mcp-http"],
            capabilities=["tools"]
        ),
        "websocket": MCPServerConfig(
            name=name,
            description="WebSocket MCP server",
            transport="websocket",
            endpoint="ws://localhost:8080/mcp",
            capabilities=["tools", "resources"]
        )
    }
    
    if category in templates:
        config = templates[category]
        config.name = name
        return config
    
    # Default template
    return MCPServerConfig(
        name=name,
        description=f"MCP server: {name}",
        transport="stdio",
        command=[f"agentix-mcp-{name}"],
        capabilities=["tools"]
    )
