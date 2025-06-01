"""
Agentix CLI for agent template scaffolding and management.

This provides the `agentix` command-line interface for:
- Creating new agent projects from templates
- Managing agent configurations
- Development server and tools
"""

import click
import os
import shutil
from pathlib import Path
from typing import Dict, Any
import yaml
import json

from .progressive import ConfigTemplates, ProgressiveAgentBuilder
from .utils.logging import setup_logging


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Agentix CLI - FastAPI for AI Agents"""
    pass


@cli.command()
@click.argument('template', type=click.Choice([
    'web_search_bot', 'summarizer', 'form_filler',
    'email_responder', 'multi_step_planner'
]))
@click.argument('project_name')
@click.option('--output-dir', '-o', default='.', help='Output directory for the project')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Configuration file format')
def init(template: str, project_name: str, output_dir: str, format: str):
    """
    Initialize a new agent project from a template.

    TEMPLATE: The template to use (web_search_bot, summarizer, etc.)
    PROJECT_NAME: Name of the new project
    """

    click.echo(f"🚀 Creating new Agentix project: {project_name}")
    click.echo(f"📋 Using template: {template}")

    # Create project directory
    project_path = Path(output_dir) / project_name
    project_path.mkdir(parents=True, exist_ok=True)

    # Get template configuration
    template_func = getattr(ConfigTemplates, template)
    agent_def = template_func()
    agent_def.name = project_name

    # Create project structure
    _create_project_structure(project_path, agent_def, template, format)

    click.echo(f"✅ Project created at: {project_path}")
    click.echo(f"\nNext steps:")
    click.echo(f"  cd {project_name}")
    click.echo(f"  pip install -r requirements.txt")
    click.echo(f"  python main.py")


def _create_project_structure(project_path: Path, agent_def, template: str, format: str):
    """Create the complete project structure."""

    # 1. Configuration file
    config_file = f"agent.{format}"
    config_path = project_path / config_file

    if format == 'yaml':
        with open(config_path, 'w') as f:
            yaml.dump(agent_def.model_dump(), f, default_flow_style=False, indent=2)
    else:
        with open(config_path, 'w') as f:
            json.dump(agent_def.model_dump(), f, indent=2)

    # 2. Main application file
    main_content = _generate_main_file(agent_def.name, config_file, template)
    with open(project_path / "main.py", 'w', encoding='utf-8') as f:
        f.write(main_content)

    # 3. Requirements file
    requirements_content = _generate_requirements()
    with open(project_path / "requirements.txt", 'w') as f:
        f.write(requirements_content)

    # 4. README file
    readme_content = _generate_readme(agent_def.name, template)
    with open(project_path / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)

    # 5. Example usage file
    example_content = _generate_example_file(agent_def.name, template)
    with open(project_path / "example.py", 'w', encoding='utf-8') as f:
        f.write(example_content)

    # 6. Test file
    test_content = _generate_test_file(agent_def.name, config_file)
    with open(project_path / "test_agent.py", 'w') as f:
        f.write(test_content)

    # 7. Docker files
    dockerfile_content = _generate_dockerfile()
    with open(project_path / "Dockerfile", 'w') as f:
        f.write(dockerfile_content)

    docker_compose_content = _generate_docker_compose(agent_def.name)
    with open(project_path / "docker-compose.yml", 'w') as f:
        f.write(docker_compose_content)

    # 8. Environment file template
    env_content = _generate_env_template()
    with open(project_path / ".env.example", 'w') as f:
        f.write(env_content)


def _generate_main_file(agent_name: str, config_file: str, template: str) -> str:
    """Generate the main application file."""

    return f'''#!/usr/bin/env python3
"""
{agent_name} - Agentix Agent

This is a {template.replace('_', ' ').title()} agent created with Agentix.
"""

import asyncio
import logging
from agentix.progressive import config_agent

def main():
    """Main application entry point."""

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("{agent_name.lower()}")

    logger.info("🚀 Starting {agent_name}")

    # Create agent from configuration
    agent = config_agent("{config_file}")

    # Example usage
    test_queries = {_get_template_queries(template)}

    print(f"\\n🤖 {agent_name} is ready!")
    print("Try these example queries:")

    for i, query in enumerate(test_queries, 1):
        print(f"  {{i}}. {{query}}")

    print("\\nEnter your queries (type 'quit' to exit):")

    while True:
        try:
            user_input = input("\\n> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("🤔 Processing...")
            response = agent(user_input)
            print(f"🤖 {{response}}")

        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {{e}}")

async def async_example():
    """Example of async usage."""

    agent = config_agent("{config_file}")

    queries = {_get_template_queries(template)}

    for query in queries:
        print(f"\\n🔍 Query: {{query}}")
        response = await agent.arun(query)
        print(f"🤖 Response: {{response}}")

if __name__ == "__main__":
    main()
'''


def _get_template_queries(template: str) -> str:
    """Get example queries for each template."""

    queries = {
        'web_search_bot': [
            "What are the latest developments in AI?",
            "Find information about renewable energy trends",
            "Search for Python programming best practices"
        ],
        'summarizer': [
            "Summarize this article: [paste article text]",
            "Create a summary of the key points from this document",
            "Provide an executive summary of this report"
        ],
        'form_filler': [
            "Fill out a job application form with my information",
            "Complete a customer survey form",
            "Help me fill out a registration form"
        ],
        'email_responder': [
            "Draft a response to this customer inquiry",
            "Write a professional follow-up email",
            "Compose a meeting request email"
        ],
        'multi_step_planner': [
            "Plan a marketing campaign for a new product",
            "Create a project timeline for website development",
            "Plan a research study on user behavior"
        ]
    }

    return str(queries.get(template, ["What can you help me with?"]))


def _generate_requirements() -> str:
    """Generate requirements.txt content."""

    return """# Agentix framework
agentix>=0.1.0

# Core dependencies
pydantic>=2.0.0
aiohttp>=3.8.0
pyyaml>=6.0

# Optional: LLM providers (uncomment as needed)
# openai>=1.0.0
# anthropic>=0.7.0

# Optional: Additional tools
# beautifulsoup4>=4.12.0
# requests>=2.31.0

# Development dependencies
pytest>=7.0.0
pytest-asyncio>=0.21.0
"""


def _generate_readme(agent_name: str, template: str) -> str:
    """Generate README.md content."""

    return f"""# {agent_name}

A {template.replace('_', ' ').title()} agent built with [Agentix](https://github.com/AP3X-Dev/agentix).

## Features

- 🧠 **Seven-Node Blueprint**: Complete agent architecture with LLM, tools, memory, and guardrails
- 🕒 **Temporal Knowledge**: Dynamic memory that evolves over time
- 🛡️ **Safety Guardrails**: Input/output validation and content filtering
- 📊 **Performance Monitoring**: Real-time execution tracking and metrics

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the agent:**
   ```bash
   python main.py
   ```

## Configuration

The agent is configured via `agent.yaml`. You can modify:

- **LLM settings**: Model, temperature, max tokens
- **Tools**: Enable/disable web search, file operations, etc.
- **Memory**: Temporal graph, episodic memory settings
- **Guardrails**: Safety and validation rules

## Usage Examples

### Basic Usage
```python
from agentix.progressive import config_agent

agent = config_agent("agent.yaml")
response = agent("Your query here")
print(response)
```

### Async Usage
```python
import asyncio
from agentix.progressive import config_agent

async def main():
    agent = config_agent("agent.yaml")
    response = await agent.arun("Your query here")
    print(response)

asyncio.run(main())
```

## Deployment

### Docker
```bash
docker build -t {agent_name.lower()} .
docker run -p 8000:8000 {agent_name.lower()}
```

### Docker Compose
```bash
docker-compose up
```

## Development

### Running Tests
```bash
pytest test_agent.py
```

### Monitoring
The agent includes built-in performance monitoring. Access metrics via:
```python
stats = agent.get_stats()
print(stats)
```

## Customization

### Progressive Disclosure
Agentix supports three levels of complexity:

1. **Simple**: One-liner agent creation
2. **Intermediate**: YAML/JSON configuration (current)
3. **Advanced**: Full graph programming

To upgrade to advanced mode:
```python
from agentix.progressive import builder

# Extract the underlying graph for customization
graph = builder.upgrade_to_graph(agent)
# Modify graph as needed...
```

## Support

- 📖 [Documentation](https://docs.agentix.dev)
- 💬 [Discord Community](https://discord.gg/agentix)
- 🐛 [Issues](https://github.com/AP3X-Dev/agentix/issues)

## License

MIT License - see LICENSE file for details.
"""


def _generate_example_file(agent_name: str, template: str) -> str:
    """Generate example usage file."""

    return f'''"""
Example usage for {agent_name}.

This file demonstrates different ways to use your agent.
"""

import asyncio
from agentix.progressive import config_agent

def basic_example():
    """Basic synchronous usage."""

    print("🔄 Basic Example")
    print("-" * 20)

    agent = config_agent("agent.yaml")

    queries = {_get_template_queries(template)}

    for query in queries[:2]:  # Just first 2 for demo
        print(f"\\n🔍 Query: {{query}}")
        response = agent(query)
        print(f"🤖 Response: {{response[:200]}}...")

async def async_example():
    """Asynchronous usage example."""

    print("\\n⚡ Async Example")
    print("-" * 20)

    agent = config_agent("agent.yaml")

    # Process multiple queries concurrently
    queries = {_get_template_queries(template)}

    tasks = [agent.arun(query) for query in queries[:2]]
    responses = await asyncio.gather(*tasks)

    for query, response in zip(queries[:2], responses):
        print(f"\\n🔍 Query: {{query}}")
        print(f"🤖 Response: {{response[:200]}}...")

def stats_example():
    """Example of getting agent statistics."""

    print("\\n📊 Statistics Example")
    print("-" * 20)

    agent = config_agent("agent.yaml")

    # Run a few queries
    agent("Test query 1")
    agent("Test query 2")

    # Get statistics
    stats = agent.get_stats()
    print(f"Agent stats: {{stats}}")

if __name__ == "__main__":
    basic_example()
    asyncio.run(async_example())
    stats_example()
'''


def _generate_test_file(agent_name: str, config_file: str) -> str:
    """Generate test file."""

    return f'''"""
Tests for {agent_name}.
"""

import pytest
import asyncio
from agentix.progressive import config_agent

@pytest.fixture
def agent():
    """Create agent for testing."""
    return config_agent("{config_file}")

def test_agent_creation(agent):
    """Test that agent can be created."""
    assert agent is not None
    assert agent.name == "{agent_name}"

def test_basic_query(agent):
    """Test basic query functionality."""
    response = agent("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_async_query(agent):
    """Test async query functionality."""
    response = await agent.arun("What is AI?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_agent_stats(agent):
    """Test agent statistics."""
    # Run a query first
    agent("Test query")

    stats = agent.get_stats()
    assert isinstance(stats, dict)

def test_multiple_queries(agent):
    """Test multiple queries in sequence."""
    queries = [
        "First query",
        "Second query",
        "Third query"
    ]

    responses = []
    for query in queries:
        response = agent(query)
        responses.append(response)
        assert isinstance(response, str)

    # All responses should be different (basic check)
    assert len(set(responses)) > 1

@pytest.mark.asyncio
async def test_concurrent_queries(agent):
    """Test concurrent query processing."""
    queries = [
        "Query 1",
        "Query 2",
        "Query 3"
    ]

    tasks = [agent.arun(query) for query in queries]
    responses = await asyncio.gather(*tasks)

    assert len(responses) == len(queries)
    for response in responses:
        assert isinstance(response, str)
        assert len(response) > 0
'''


def _generate_dockerfile() -> str:
    """Generate Dockerfile."""

    return '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
'''


def _generate_docker_compose(agent_name: str) -> str:
    """Generate docker-compose.yml."""

    return f'''version: '3.8'

services:
  {agent_name.lower()}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AGENTIX_ENVIRONMENT=production
      - AGENTIX_LOG_LEVEL=INFO
    env_file:
      - .env
    restart: unless-stopped

  # Optional: Add monitoring
  # prometheus:
  #   image: prom/prometheus
  #   ports:
  #     - "9090:9090"
  #   volumes:
  #     - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # Optional: Add database for memory persistence
  # postgres:
  #   image: postgres:15
  #   environment:
  #     POSTGRES_DB: agentix
  #     POSTGRES_USER: agentix
  #     POSTGRES_PASSWORD: password
  #   volumes:
  #     - postgres_data:/var/lib/postgresql/data
  #   ports:
  #     - "5432:5432"

# volumes:
#   postgres_data:
'''


def _generate_env_template() -> str:
    """Generate .env.example file."""

    return '''# Agentix Configuration
AGENTIX_ENVIRONMENT=development
AGENTIX_LOG_LEVEL=INFO

# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Search Engine API Keys (optional)
GOOGLE_API_KEY=your_google_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
BING_API_KEY=your_bing_key_here

# Database Configuration (optional)
DATABASE_URL=postgresql://user:password@localhost/agentix

# Monitoring (optional)
ENABLE_METRICS=true
METRICS_PORT=9090
'''


@cli.command()
@click.argument('config_file')
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def dev(config_file: str, host: str, port: int, reload: bool):
    """
    Start development server for an agent.

    CONFIG_FILE: Path to agent configuration file
    """

    click.echo(f"🚀 Starting Agentix development server")
    click.echo(f"📋 Config: {config_file}")
    click.echo(f"🌐 Server: http://{host}:{port}")

    # This would start a development server with hot reload
    # For now, we'll just show the concept
    click.echo("Development server would start here...")
    click.echo("Features:")
    click.echo("  ✅ Hot reload on config changes")
    click.echo("  ✅ Interactive agent testing")
    click.echo("  ✅ Real-time performance monitoring")
    click.echo("  ✅ Graph visualization")


@cli.command()
@click.argument('template', type=click.Choice([
    'web_search_bot', 'summarizer', 'form_filler',
    'email_responder', 'multi_step_planner'
]))
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml')
def template(template: str, format: str):
    """
    Show a template configuration.

    TEMPLATE: The template to display
    """

    template_func = getattr(ConfigTemplates, template)
    agent_def = template_func()

    if format == 'yaml':
        click.echo(yaml.dump(agent_def.model_dump(), default_flow_style=False, indent=2))
    else:
        click.echo(json.dumps(agent_def.model_dump(), indent=2))


if __name__ == '__main__':
    cli()
