"""
MCP server management for Agentix.

This module provides server configuration management and discovery
for MCP-compatible tools and services.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
import yaml

from .client import MCPClientConfig


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    
    name: str
    description: str = ""
    transport: str = "stdio"
    command: Optional[List[str]] = None
    endpoint: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    auth: Optional[Dict[str, Any]] = None
    capabilities: List[str] = field(default_factory=lambda: ["tools", "resources"])
    enabled: bool = True
    auto_start: bool = True
    timeout: int = 30
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_client_config(self) -> MCPClientConfig:
        """Convert to MCPClientConfig."""
        return MCPClientConfig(
            name=self.name,
            transport=self.transport,
            endpoint=self.endpoint,
            command=self.command,
            env=self.env,
            timeout=self.timeout,
            max_retries=self.max_retries,
            auth=self.auth,
            capabilities=self.capabilities
        )


class MCPServerManager:
    """
    Manager for MCP server configurations.
    
    Handles loading, saving, and managing MCP server configurations
    from various sources (files, environment, etc.).
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize MCP server manager."""
        self.logger = logging.getLogger("agentix.mcp.server_manager")
        
        # Configuration directory
        if config_dir is None:
            config_dir = Path.home() / ".agentix" / "mcp"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Server configurations
        self.servers: Dict[str, MCPServerConfig] = {}
        
        # Load configurations
        self._load_configurations()
    
    def _load_configurations(self):
        """Load MCP server configurations from files."""
        
        # Load from main config file
        main_config_file = self.config_dir / "servers.yaml"
        if main_config_file.exists():
            self._load_config_file(main_config_file)
        
        # Load from individual server files
        servers_dir = self.config_dir / "servers"
        if servers_dir.exists():
            for config_file in servers_dir.glob("*.yaml"):
                self._load_config_file(config_file)
            
            for config_file in servers_dir.glob("*.json"):
                self._load_config_file(config_file)
        
        # Load from environment
        self._load_from_environment()
        
        self.logger.info(f"Loaded {len(self.servers)} MCP server configurations")
    
    def _load_config_file(self, config_file: Path):
        """Load configuration from a file."""
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Handle single server or multiple servers
            if isinstance(data, dict):
                if "servers" in data:
                    # Multiple servers in one file
                    for server_data in data["servers"]:
                        self._add_server_config(server_data)
                else:
                    # Single server
                    self._add_server_config(data)
            elif isinstance(data, list):
                # List of servers
                for server_data in data:
                    self._add_server_config(server_data)
            
            self.logger.debug(f"Loaded MCP config from: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load MCP config from {config_file}: {str(e)}")
    
    def _add_server_config(self, server_data: Dict[str, Any]):
        """Add a server configuration."""
        
        try:
            config = MCPServerConfig(**server_data)
            self.servers[config.name] = config
        except Exception as e:
            self.logger.error(f"Invalid server config: {str(e)}")
    
    def _load_from_environment(self):
        """Load MCP server configurations from environment variables."""
        
        # Look for AGENTIX_MCP_SERVERS environment variable
        env_servers = os.getenv("AGENTIX_MCP_SERVERS")
        if env_servers:
            try:
                servers_data = json.loads(env_servers)
                for server_data in servers_data:
                    self._add_server_config(server_data)
            except Exception as e:
                self.logger.error(f"Failed to parse AGENTIX_MCP_SERVERS: {str(e)}")
        
        # Look for individual server environment variables
        for env_var in os.environ:
            if env_var.startswith("AGENTIX_MCP_"):
                server_name = env_var[12:].lower()  # Remove AGENTIX_MCP_ prefix
                if server_name and server_name not in self.servers:
                    # Try to parse as JSON
                    try:
                        server_data = json.loads(os.environ[env_var])
                        server_data["name"] = server_name
                        self._add_server_config(server_data)
                    except:
                        # Treat as simple command
                        self._add_server_config({
                            "name": server_name,
                            "transport": "stdio",
                            "command": [os.environ[env_var]]
                        })
    
    def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get a server configuration by name."""
        return self.servers.get(name)
    
    def list_servers(self, enabled_only: bool = True) -> List[MCPServerConfig]:
        """
        List server configurations.
        
        Args:
            enabled_only: Only return enabled servers
            
        Returns:
            List of server configurations
        """
        
        servers = list(self.servers.values())
        
        if enabled_only:
            servers = [s for s in servers if s.enabled]
        
        return servers
    
    def add_server(self, config: MCPServerConfig, save: bool = True) -> bool:
        """
        Add a new server configuration.
        
        Args:
            config: Server configuration
            save: Whether to save to file
            
        Returns:
            True if added successfully
        """
        
        try:
            self.servers[config.name] = config
            
            if save:
                self.save_server(config)
            
            self.logger.info(f"Added MCP server: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add server {config.name}: {str(e)}")
            return False
    
    def remove_server(self, name: str, delete_file: bool = True) -> bool:
        """
        Remove a server configuration.
        
        Args:
            name: Server name
            delete_file: Whether to delete the config file
            
        Returns:
            True if removed successfully
        """
        
        if name not in self.servers:
            return False
        
        try:
            del self.servers[name]
            
            if delete_file:
                config_file = self.config_dir / "servers" / f"{name}.yaml"
                if config_file.exists():
                    config_file.unlink()
            
            self.logger.info(f"Removed MCP server: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove server {name}: {str(e)}")
            return False
    
    def save_server(self, config: MCPServerConfig):
        """Save a server configuration to file."""
        
        try:
            servers_dir = self.config_dir / "servers"
            servers_dir.mkdir(exist_ok=True)
            
            config_file = servers_dir / f"{config.name}.yaml"
            
            with open(config_file, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False, indent=2)
            
            self.logger.debug(f"Saved MCP server config: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save server config {config.name}: {str(e)}")
    
    def save_all_servers(self):
        """Save all server configurations."""
        
        try:
            main_config = {
                "servers": [asdict(config) for config in self.servers.values()]
            }
            
            config_file = self.config_dir / "servers.yaml"
            
            with open(config_file, 'w') as f:
                yaml.dump(main_config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Saved all MCP server configs to: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save server configs: {str(e)}")
    
    def get_client_configs(self, enabled_only: bool = True) -> List[MCPClientConfig]:
        """
        Get client configurations for all servers.
        
        Args:
            enabled_only: Only return enabled servers
            
        Returns:
            List of client configurations
        """
        
        servers = self.list_servers(enabled_only)
        return [server.to_client_config() for server in servers]
    
    def create_builtin_servers(self):
        """Create built-in server configurations."""
        
        builtin_servers = [
            MCPServerConfig(
                name="filesystem",
                description="Local filesystem access",
                transport="stdio",
                command=["agentix-mcp-filesystem"],
                capabilities=["tools", "resources"],
                metadata={
                    "category": "filesystem",
                    "builtin": True
                }
            ),
            MCPServerConfig(
                name="web_search",
                description="Web search capabilities",
                transport="stdio", 
                command=["agentix-mcp-websearch"],
                env={"SEARCH_API_KEY": "${GOOGLE_API_KEY}"},
                capabilities=["tools"],
                metadata={
                    "category": "search",
                    "builtin": True
                }
            ),
            MCPServerConfig(
                name="database",
                description="Database access",
                transport="stdio",
                command=["agentix-mcp-database"],
                env={"DATABASE_URL": "${DATABASE_URL}"},
                capabilities=["tools", "resources"],
                metadata={
                    "category": "database",
                    "builtin": True
                }
            ),
            MCPServerConfig(
                name="http_api",
                description="HTTP API client",
                transport="stdio",
                command=["agentix-mcp-http"],
                capabilities=["tools"],
                metadata={
                    "category": "api",
                    "builtin": True
                }
            ),
            MCPServerConfig(
                name="email",
                description="Email operations",
                transport="stdio",
                command=["agentix-mcp-email"],
                env={
                    "SMTP_HOST": "${SMTP_HOST}",
                    "SMTP_USER": "${SMTP_USER}",
                    "SMTP_PASS": "${SMTP_PASS}"
                },
                capabilities=["tools"],
                metadata={
                    "category": "communication",
                    "builtin": True
                }
            )
        ]
        
        for server in builtin_servers:
            if server.name not in self.servers:
                self.add_server(server, save=False)
        
        self.logger.info(f"Created {len(builtin_servers)} built-in server configurations")


# Global server manager instance
_global_server_manager: Optional[MCPServerManager] = None


def get_server_manager() -> MCPServerManager:
    """Get the global MCP server manager."""
    global _global_server_manager
    if _global_server_manager is None:
        _global_server_manager = MCPServerManager()
    return _global_server_manager


def list_mcp_servers(enabled_only: bool = True) -> List[MCPServerConfig]:
    """List available MCP servers."""
    manager = get_server_manager()
    return manager.list_servers(enabled_only)


def get_mcp_server(name: str) -> Optional[MCPServerConfig]:
    """Get an MCP server configuration."""
    manager = get_server_manager()
    return manager.get_server(name)


def add_mcp_server(config: MCPServerConfig) -> bool:
    """Add an MCP server configuration."""
    manager = get_server_manager()
    return manager.add_server(config)
