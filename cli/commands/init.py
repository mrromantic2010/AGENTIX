"""
Init command for Agentix CLI.
"""

import click
from pathlib import Path


@click.command()
@click.argument('project_name')
@click.option('--template', '-t', default='basic', help='Project template')
def init_command(project_name, template):
    """Initialize a new Agentix project."""
    
    project_path = Path(project_name)
    
    if project_path.exists():
        click.echo(f"❌ Directory '{project_name}' already exists")
        return 1
    
    click.echo(f"🚀 Creating Agentix project: {project_name}")
    
    # Create project structure
    project_path.mkdir(parents=True)
    
    # Create basic files
    (project_path / "agent.py").write_text(f'''"""
{project_name} - Agentix Agent

A simple agent created with Agentix.
"""

import agentix

# Create your agent
agent = agentix.create_agent(
    name="{project_name}",
    tools=["search"],
    memory=True
)

if __name__ == "__main__":
    # Test the agent
    response = agent("Hello! What can you help me with?")
    print(response)
''')
    
    (project_path / "requirements.txt").write_text('''agentix
''')
    
    (project_path / "README.md").write_text(f'''# {project_name}

An AI agent built with Agentix.

## Quick Start

```bash
pip install -r requirements.txt
python agent.py
```

## Usage

```python
import agentix

agent = agentix.create_agent(
    name="{project_name}",
    tools=["search"],
    memory=True
)

response = agent("Your question here")
print(response)
```
''')
    
    click.echo(f"✅ Project '{project_name}' created successfully!")
    click.echo(f"📁 Location: {project_path.absolute()}")
    click.echo("\n🚀 Next steps:")
    click.echo(f"   cd {project_name}")
    click.echo("   pip install -r requirements.txt")
    click.echo("   python agent.py")
