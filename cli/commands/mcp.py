"""
MCP (Model Context Protocol) CLI commands for Agentix.
"""

import asyncio
import click
import json
import yaml
from pathlib import Path
from typing import List, Optional

from ...mcp.discovery import discover_mcp_servers, install_mcp_server, create_mcp_server_template
from ...mcp.server import get_server_manager, MCPServerConfig
from ...mcp.tools import get_global_mcp_registry, register_mcp_server
from ...mcp.client import MCPClient


@click.group()
def mcp_command():
    """Model Context Protocol (MCP) management commands."""
    pass


@mcp_command.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'yaml']), 
              default='table', help='Output format')
@click.option('--category', '-c', help='Filter by category')
@click.option('--enabled-only', is_flag=True, help='Show only enabled servers')
def list(format, category, enabled_only):
    """List available MCP servers."""
    
    manager = get_server_manager()
    servers = manager.list_servers(enabled_only=enabled_only)
    
    if category:
        servers = [s for s in servers if s.metadata.get('category') == category]
    
    if format == 'table':
        _display_servers_table(servers)
    elif format == 'json':
        _display_servers_json(servers)
    elif format == 'yaml':
        _display_servers_yaml(servers)


@mcp_command.command()
@click.option('--source', '-s', type=click.Choice(['auto', 'npm', 'pip', 'registry']), 
              default='auto', help='Discovery source')
@click.option('--save', is_flag=True, help='Save discovered servers to config')
def discover(source, save):
    """Discover available MCP servers."""
    
    click.echo("🔍 Discovering MCP servers...")
    
    async def _discover():
        servers = await discover_mcp_servers()
        
        if not servers:
            click.echo("No MCP servers discovered.")
            return
        
        click.echo(f"\n✅ Discovered {len(servers)} MCP servers:")
        _display_servers_table(servers)
        
        if save:
            manager = get_server_manager()
            saved_count = 0
            
            for server in servers:
                if manager.add_server(server, save=True):
                    saved_count += 1
            
            click.echo(f"\n💾 Saved {saved_count} server configurations")
    
    asyncio.run(_discover())


@mcp_command.command()
@click.argument('server_name')
@click.option('--source', '-s', type=click.Choice(['auto', 'npm', 'pip']), 
              default='auto', help='Installation source')
@click.option('--enable', is_flag=True, help='Enable server after installation')
def install(server_name, source, enable):
    """Install an MCP server."""
    
    click.echo(f"📦 Installing MCP server: {server_name}")
    
    async def _install():
        success = await install_mcp_server(server_name, source)
        
        if success:
            click.echo(f"✅ Successfully installed: {server_name}")
            
            if enable:
                # Try to discover and enable the server
                servers = await discover_mcp_servers()
                for server in servers:
                    if server.name == server_name:
                        server.enabled = True
                        manager = get_server_manager()
                        manager.add_server(server, save=True)
                        click.echo(f"✅ Enabled server: {server_name}")
                        break
        else:
            click.echo(f"❌ Failed to install: {server_name}")
            return 1
    
    return asyncio.run(_install())


@mcp_command.command()
@click.argument('server_name')
@click.option('--force', is_flag=True, help='Force removal without confirmation')
def remove(server_name, force):
    """Remove an MCP server configuration."""
    
    manager = get_server_manager()
    server = manager.get_server(server_name)
    
    if not server:
        click.echo(f"❌ Server not found: {server_name}")
        return 1
    
    if not force:
        if not click.confirm(f"Remove MCP server '{server_name}'?"):
            click.echo("Cancelled.")
            return
    
    if manager.remove_server(server_name, delete_file=True):
        click.echo(f"✅ Removed server: {server_name}")
    else:
        click.echo(f"❌ Failed to remove server: {server_name}")
        return 1


@mcp_command.command()
@click.argument('server_name')
@click.option('--enabled/--disabled', default=True, help='Enable or disable server')
def enable(server_name, enabled):
    """Enable or disable an MCP server."""
    
    manager = get_server_manager()
    server = manager.get_server(server_name)
    
    if not server:
        click.echo(f"❌ Server not found: {server_name}")
        return 1
    
    server.enabled = enabled
    manager.add_server(server, save=True)
    
    status = "enabled" if enabled else "disabled"
    click.echo(f"✅ Server '{server_name}' {status}")


@mcp_command.command()
@click.argument('server_name')
@click.option('--timeout', '-t', default=30, help='Connection timeout')
def test(server_name, timeout):
    """Test connection to an MCP server."""
    
    manager = get_server_manager()
    server = manager.get_server(server_name)
    
    if not server:
        click.echo(f"❌ Server not found: {server_name}")
        return 1
    
    click.echo(f"🔌 Testing connection to: {server_name}")
    
    async def _test():
        try:
            client_config = server.to_client_config()
            client_config.timeout = timeout
            
            client = MCPClient(client_config)
            
            if await client.connect():
                click.echo(f"✅ Connection successful")
                
                # List tools
                tools = await client.list_tools()
                click.echo(f"🔧 Available tools: {len(tools)}")
                
                for tool in tools[:5]:  # Show first 5 tools
                    click.echo(f"   • {tool.name}: {tool.description}")
                
                if len(tools) > 5:
                    click.echo(f"   ... and {len(tools) - 5} more")
                
                # List resources
                try:
                    resources = await client.list_resources()
                    click.echo(f"📁 Available resources: {len(resources)}")
                except:
                    pass
                
                await client.disconnect()
            else:
                click.echo(f"❌ Connection failed")
                return 1
                
        except Exception as e:
            click.echo(f"❌ Test failed: {str(e)}")
            return 1
    
    return asyncio.run(_test())


@mcp_command.command()
@click.argument('name')
@click.option('--category', '-c', default='general', help='Server category')
@click.option('--transport', '-t', type=click.Choice(['stdio', 'websocket', 'http']), 
              default='stdio', help='Transport type')
@click.option('--command', help='Command to run server')
@click.option('--endpoint', help='Server endpoint (for websocket/http)')
@click.option('--save', is_flag=True, help='Save configuration')
def create(name, category, transport, command, endpoint, save):
    """Create a new MCP server configuration."""
    
    # Create template
    config = create_mcp_server_template(name, category)
    
    # Override with provided options
    config.transport = transport
    
    if command:
        config.command = command.split()
    
    if endpoint:
        config.endpoint = endpoint
    
    # Display configuration
    click.echo(f"📝 Created MCP server configuration:")
    click.echo(f"   Name: {config.name}")
    click.echo(f"   Transport: {config.transport}")
    click.echo(f"   Command: {config.command}")
    click.echo(f"   Endpoint: {config.endpoint}")
    click.echo(f"   Capabilities: {config.capabilities}")
    
    if save:
        manager = get_server_manager()
        if manager.add_server(config, save=True):
            click.echo(f"✅ Saved configuration for: {name}")
        else:
            click.echo(f"❌ Failed to save configuration")
            return 1


@mcp_command.command()
@click.argument('server_name')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), 
              default='yaml', help='Output format')
def show(server_name, format):
    """Show detailed information about an MCP server."""
    
    manager = get_server_manager()
    server = manager.get_server(server_name)
    
    if not server:
        click.echo(f"❌ Server not found: {server_name}")
        return 1
    
    if format == 'json':
        from dataclasses import asdict
        click.echo(json.dumps(asdict(server), indent=2))
    else:
        from dataclasses import asdict
        click.echo(yaml.dump(asdict(server), default_flow_style=False, indent=2))


@mcp_command.command()
@click.option('--server', '-s', help='Filter by server')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
def tools(server, format):
    """List available MCP tools."""
    
    click.echo("🔧 Loading MCP tools...")
    
    async def _list_tools():
        registry = get_global_mcp_registry()
        
        # Register servers
        manager = get_server_manager()
        servers = manager.list_servers(enabled_only=True)
        
        for server_config in servers:
            if server and server_config.name != server:
                continue
            
            client_config = server_config.to_client_config()
            await registry.register_server(client_config)
        
        # List tools
        tools = registry.list_tools(server_filter=server)
        
        if not tools:
            click.echo("No MCP tools available.")
            return
        
        if format == 'table':
            _display_tools_table(tools)
        else:
            tools_data = [
                {
                    'name': tool.name,
                    'description': tool.description,
                    'server': tool.server,
                    'schema': tool.input_schema
                }
                for tool in tools
            ]
            click.echo(json.dumps(tools_data, indent=2))
    
    asyncio.run(_list_tools())


def _display_servers_table(servers: List[MCPServerConfig]):
    """Display servers in table format."""
    
    if not servers:
        click.echo("No servers found.")
        return
    
    # Header
    click.echo(f"{'Name':<20} {'Transport':<10} {'Enabled':<8} {'Category':<12} {'Description'}")
    click.echo("-" * 80)
    
    # Servers
    for server in servers:
        enabled = "✅" if server.enabled else "❌"
        category = server.metadata.get('category', 'general')
        description = server.description[:30] + "..." if len(server.description) > 30 else server.description
        
        click.echo(f"{server.name:<20} {server.transport:<10} {enabled:<8} {category:<12} {description}")


def _display_servers_json(servers: List[MCPServerConfig]):
    """Display servers in JSON format."""
    
    from dataclasses import asdict
    servers_data = [asdict(server) for server in servers]
    click.echo(json.dumps(servers_data, indent=2))


def _display_servers_yaml(servers: List[MCPServerConfig]):
    """Display servers in YAML format."""
    
    from dataclasses import asdict
    servers_data = [asdict(server) for server in servers]
    click.echo(yaml.dump(servers_data, default_flow_style=False, indent=2))


def _display_tools_table(tools):
    """Display tools in table format."""
    
    if not tools:
        click.echo("No tools found.")
        return
    
    # Header
    click.echo(f"{'Tool Name':<25} {'Server':<15} {'Description'}")
    click.echo("-" * 70)
    
    # Tools
    for tool in tools:
        description = tool.description[:30] + "..." if len(tool.description) > 30 else tool.description
        click.echo(f"{tool.name:<25} {tool.server:<15} {description}")


if __name__ == '__main__':
    mcp_command()
