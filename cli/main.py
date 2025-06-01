"""
Main CLI entry point for Agentix.
"""

import click
import asyncio
import logging
from pathlib import Path

from .commands.init import init_command
from .commands.run import run_command
from .commands.test import test_command
from .commands.deploy import deploy_command
from .commands.mcp import mcp_command
from ..utils.logging import setup_logging


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """
    Agentix - FastAPI for AI Agents
    
    Build production-ready AI agents with progressive disclosure.
    """
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config


# Add command groups
cli.add_command(init_command)
cli.add_command(run_command)
cli.add_command(test_command)
cli.add_command(deploy_command)
cli.add_command(mcp_command)


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()
