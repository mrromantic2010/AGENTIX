# 🚀 Agentix - FastAPI for AI Agents

**Build production-ready AI agents with progressive disclosure**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub issues](https://img.shields.io/github/issues/AP3X-Dev/agentix)](https://github.com/AP3X-Dev/agentix/issues)
[![GitHub stars](https://img.shields.io/github/stars/AP3X-Dev/agentix)](https://github.com/AP3X-Dev/agentix/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/AP3X-Dev/agentix)](https://github.com/AP3X-Dev/agentix/network)

Agentix is the leading framework for building AI agents with **progressive disclosure** - start simple, scale to enterprise. From zero-config agents to sophisticated multi-agent systems with **Anthropic Claude integration** and **Model Context Protocol (MCP) support**.

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Architecture](#️-architecture)
- [Documentation](#-documentation)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Contributing](#-contributing)
  - [Development Setup](#development-setup)
  - [Testing](#testing)
- [Security](#-security)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Support](#-support)

## ✨ **Key Features**

### 🎯 **Progressive Disclosure**
- **Zero-config**: `agentix.agent("MyBot")` - instant agent creation
- **Configuration-based**: YAML/JSON for structured development
- **Graph-based**: Full control with seven-node architecture
- **Enterprise-ready**: Production deployment and monitoring

### 🤖 **Best-in-Class LLM Support**
- **Anthropic Claude**: Direct API integration (Claude-3.5 Sonnet, Opus, Haiku)
- **OpenRouter**: 100+ models from multiple providers
- **OpenAI**: GPT-4, GPT-3.5 with function calling
- **Streaming**: Real-time responses across all providers

### 🔧 **Model Context Protocol (MCP)**
- **Tool Ecosystem**: Filesystem, web search, database, HTTP API, email
- **Server Discovery**: Automatic MCP server detection and installation
- **Cross-Agent Sharing**: Tools and memory across agent instances
- **CLI Management**: `agentix mcp` commands for server management

### 🧠 **Advanced Memory System**
- **Temporal Knowledge Graphs**: Graphiti-powered memory
- **Cross-Agent Memory**: Shared memory via MCP protocol
- **Memory Scoping**: Per-user, per-session, global memory
- **Real-time Sync**: Memory drift tracking and visualization

## 🚀 **Quick Start**

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Basic installation
pip install agentix

# With MCP support (recommended)
pip install agentix[mcp]

# Full installation with all features
pip install agentix[all]

# Development installation
git clone https://github.com/AP3X-Dev/agentix.git
cd agentix
pip install -e ".[dev]"
```

### Environment Setup

Create a `.env` file or set environment variables:

```bash
# Required for Claude integration
export ANTHROPIC_API_KEY="your_anthropic_key"

# Optional for other providers
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

### Zero-Config Agent
```python
import agentix

# Create an agent in one line
agent = agentix.agent("MyBot")
response = agent("What's the weather like?")
print(response)
```

### Claude Agent with MCP Tools
```python
import agentix

# Claude agent with filesystem and web search tools
agent = agentix.anthropic_agent(
    name="ClaudeBot",
    model="claude-3-5-sonnet-20241022",
    mcp_servers=["filesystem", "web_search"]
)

response = agent("""
Search for information about quantum computing,
save the results to a file, and summarize the key points.
""")
```

### Multi-Provider Comparison
```python
import agentix

# Test the same query across different models
models = [
    ("Claude-3.5 Sonnet", "claude-3-5-sonnet-20241022"),
    ("GPT-4 Turbo", "openai/gpt-4-turbo"),
    ("Gemini Pro", "google/gemini-pro")
]

for name, model in models:
    agent = agentix.create_agent(f"{name}Agent", llm_model=model)
    response = agent("Explain machine learning in simple terms")
    print(f"{name}: {response[:100]}...")
```

### Temporal Knowledge Graph

```python
from agentix.memory import TemporalKnowledgeGraph, TemporalNode, TemporalEdge
from datetime import datetime

# Create temporal knowledge graph
tkg = TemporalKnowledgeGraph()

# Add temporal nodes
ai_node = TemporalNode(
    node_type="concept",
    label="Artificial Intelligence",
    properties={"definition": "Machine intelligence"},
    created_at=datetime.now()
)

# Add to graph
tkg.add_node(ai_node)

# Query with temporal constraints
from agentix.memory import TemporalQuery

query = TemporalQuery(
    query_type="search",
    node_types=["concept"],
    time_range=(datetime(2024, 1, 1), datetime.now())
)

results = tkg.query(query)
```

### Tool Integration

```python
from agentix.tools import WebSearchTool, WebSearchConfig

# Configure web search tool
search_config = WebSearchConfig(
    name="web_search",
    description="Web search with content extraction",
    search_engine="duckduckgo",
    max_results=5,
    extract_content=True
)

# Create and use tool
search_tool = WebSearchTool(search_config)

async def search_example():
    result = await search_tool.run({
        "query": "latest AI research",
        "max_results": 3
    })
    return result
```

### Guardrails & Safety

```python
from agentix.guardrails import InputValidator, SafetyChecker
from agentix.guardrails import InputValidationConfig, SafetyConfig

# Input validation
input_config = InputValidationConfig(
    max_input_length=1000,
    block_personal_info=True,
    validate_urls=True
)

validator = InputValidator(input_config)
validation_result = validator.validate("User input text")

# Safety checking
safety_config = SafetyConfig(
    check_harmful_content=True,
    check_personal_info=True,
    safety_threshold=0.8
)

safety_checker = SafetyChecker(safety_config)
safety_result = safety_checker.check_safety("Content to check")
```

## 🏛️ Architecture

### Seven-Node Blueprint

1. **LLM Nodes**: Primary reasoning and text generation
2. **Tool Nodes**: External action execution (APIs, databases, etc.)
3. **Control Nodes**: Flow control and decision making
4. **Memory Nodes**: Temporal knowledge management
5. **Guardrail Nodes**: Safety and validation
6. **Fallback Nodes**: Error handling and recovery
7. **Human Input Nodes**: Human-in-the-loop integration

### Temporal Knowledge Graphs

Unlike static RAG systems, Agentix uses temporal knowledge graphs that:
- Track knowledge validity over time
- Support dynamic relationship updates
- Enable temporal reasoning and queries
- Provide automatic knowledge consolidation

## 📚 Documentation

- [Contributing Guide](CONTRIBUTING.md) - How to contribute to Agentix
- [Changelog](CHANGELOG.md) - Version history and changes
- [Examples Directory](examples/) - Working code examples
- [Demo Applications](demo/) - Complete demo applications
- [License](LICENSE) - MIT License details

> 📖 **Full documentation coming soon!** We're working on comprehensive docs including API reference, tutorials, and guides.

## 🔧 Configuration

### Environment Variables

```bash
# LLM Provider API Keys
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"

# Search Engine API Keys
export GOOGLE_API_KEY="your_google_key"
export GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id"
export BING_API_KEY="your_bing_key"

# Database Configuration
export DATABASE_URL="postgresql://user:pass@localhost/agentix"

# Framework Configuration
export AGENTIX_ENVIRONMENT="development"
export AGENTIX_LOG_LEVEL="INFO"
```

### Configuration Files

```yaml
# agentix_config.yaml
framework_version: "0.1.0"
environment: "development"
log_level: "INFO"

memory_config:
  enable_temporal_graph: true
  enable_episodic_memory: true
  auto_consolidation: true

tool_config:
  default_timeout: 30
  max_retries: 3
  enable_validation: true

security_config:
  enable_guardrails: true
  validate_inputs: true
  validate_outputs: true
```

## 🧪 Examples

See the `examples/` directory for comprehensive examples:

- [`basic_agent_example.py`](examples/basic_agent_example.py) - Complete agent with seven-node blueprint
- [`claude_mcp_demo.py`](examples/claude_mcp_demo.py) - Claude integration with MCP tools
- [`openrouter_demo.py`](examples/openrouter_demo.py) - OpenRouter multi-model examples
- [`progressive_disclosure_demo.py`](examples/progressive_disclosure_demo.py) - Progressive disclosure patterns

### Demo Applications

Check out the [`demo/`](demo/) directory for complete applications:

- [`simple_demo.py`](demo/simple_demo.py) - Basic agent demonstration
- [`intelligent_research_assistant.py`](demo/intelligent_research_assistant.py) - Research assistant with web search
- [`run_demo.py`](demo/run_demo.py) - Interactive demo runner

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/AP3X-Dev/agentix.git
cd agentix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import agentix; print('Agentix installed successfully!')"

# Run tests
pytest

# Run code quality checks
black agentix/
isort agentix/
mypy agentix/
flake8 agentix/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentix --cov-report=html

# Run specific test file
pytest tests/test_agent.py

# Run integration tests
pytest tests/integration/
```

## 🔒 Security

### API Key Management

- **Never commit API keys** to version control
- Use environment variables or `.env` files
- Rotate keys regularly
- Use different keys for development and production

### Reporting Security Issues

If you discover a security vulnerability, please email [GuerrillaMedia702@gmail.com](mailto:GuerrillaMedia702@gmail.com) instead of creating a public issue.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the LangGraph framework
- Built with Pydantic for type safety
- Temporal knowledge graph concepts from academic research
- Community feedback and contributions

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/AP3X-Dev/agentix/issues) - Bug reports and feature requests
- 💬 **Discussions**: [GitHub Discussions](https://github.com/AP3X-Dev/agentix/discussions) - Community Q&A
- 📧 **Email**: [GuerrillaMedia702@gmail.com](mailto:GuerrillaMedia702@gmail.com) - Direct support
- 📖 **Documentation**: Coming soon - Comprehensive guides and API reference

### Getting Help

1. **Check existing issues** - Your question might already be answered
2. **Search discussions** - Community knowledge base
3. **Create an issue** - For bugs or feature requests
4. **Start a discussion** - For questions and ideas

---

**Agentix** - Building the future of AI agents with temporal intelligence.
