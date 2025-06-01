"""
Run command for Agentix CLI.
"""

import click


@click.command()
@click.argument('agent_file')
@click.option('--port', '-p', default=8000, help='Port to run on')
@click.option('--host', '-h', default='localhost', help='Host to bind to')
def run_command(agent_file, port, host):
    """Run an Agentix agent."""
    
    click.echo(f"🚀 Running agent from: {agent_file}")
    click.echo(f"🌐 Server: http://{host}:{port}")
    
    # This would implement the actual agent running logic
    click.echo("⚠️  Run command not yet implemented")
    click.echo("Use 'python your_agent.py' for now")
