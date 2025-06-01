"""
Agentix: A comprehensive framework for building powerful AI agents with temporal knowledge graphs.

This framework implements the seven-node blueprint for agent construction with:
- LLM Nodes for reasoning
- Tool Nodes for external actions
- Control Nodes for flow management
- Memory Nodes with temporal awareness
- Guardrail Nodes for safety
- Fallback Nodes for error handling
- Human Input Nodes for oversight
"""

__version__ = "0.1.0"
__author__ = "AP3X"
__description__ = "A comprehensive framework for building powerful AI agents with temporal knowledge graphs"

# Core components
from .core.agent import Agent, AgentConfig
from .core.graph import AgentGraph, GraphState
from .core.nodes import (
    LLMNode,
    ToolNode,
    ControlNode,
    MemoryNode,
    GuardrailNode,
    FallbackNode,
    HumanInputNode,
    NodeConfig,
    NodeResult
)

# Memory system
from .memory.temporal_graph import TemporalKnowledgeGraph
from .memory.memory_manager import MemoryManager

# Tool system
from .tools.base import BaseTool, ToolRegistry
from .tools.web_search import WebSearchTool
from .tools.database import DatabaseTool

# Guardrails
from .guardrails.input_validation import InputValidator
from .guardrails.output_validation import OutputValidator
from .guardrails.safety_checker import SafetyChecker

# Configuration
from .config.settings import AgentixConfig
from .config.llm_config import LLMConfig

# Utilities
from .utils.logging import setup_logging
from .utils.exceptions import AgentixError, NodeExecutionError, ValidationError

# Progressive Disclosure & Zero-Config
from .decorators import (
    agent,
    create_agent,
    search_agent,
    chat_agent,
    research_agent,
    openrouter_agent,
    anthropic_agent,
    AgentixAgent
)

from .progressive import (
    ProgressiveAgentBuilder,
    AgentDefinition,
    ConfigTemplates,
    simple_agent,
    config_agent,
    graph_agent,
    template_agent
)

# MCP Integration
from .mcp import (
    MCPClient,
    MCPClientConfig,
    MCPToolNode,
    MCPToolRegistry,
    MCPServerManager,
    MCPServerConfig,
    discover_mcp_servers
)

# CLI and Development Tools
from .cli.main import main as cli_main

# Public API exports
__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "AgentGraph",
    "GraphState",

    # Nodes
    "LLMNode",
    "ToolNode",
    "ControlNode",
    "MemoryNode",
    "GuardrailNode",
    "FallbackNode",
    "HumanInputNode",
    "NodeConfig",
    "NodeResult",

    # Memory
    "TemporalKnowledgeGraph",
    "MemoryManager",

    # Tools
    "BaseTool",
    "ToolRegistry",
    "WebSearchTool",
    "DatabaseTool",

    # Guardrails
    "InputValidator",
    "OutputValidator",
    "SafetyChecker",

    # Config
    "AgentixConfig",
    "LLMConfig",

    # Utils
    "setup_logging",
    "AgentixError",
    "NodeExecutionError",
    "ValidationError",

    # Progressive Disclosure & Zero-Config
    "agent",
    "create_agent",
    "search_agent",
    "chat_agent",
    "research_agent",
    "openrouter_agent",
    "anthropic_agent",
    "AgentixAgent",
    "ProgressiveAgentBuilder",
    "AgentDefinition",
    "ConfigTemplates",
    "simple_agent",
    "config_agent",
    "graph_agent",
    "template_agent",

    # MCP Integration
    "MCPClient",
    "MCPClientConfig",
    "MCPToolNode",
    "MCPToolRegistry",
    "MCPServerManager",
    "MCPServerConfig",
    "discover_mcp_servers",

    # CLI
    "cli_main"
]

# Framework metadata
FRAMEWORK_INFO = {
    "name": "Agentix",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "architecture": "Seven-Node Blueprint",
    "features": [
        "Temporal Knowledge Graphs",
        "LangGraph Integration",
        "Pydantic Type Safety",
        "Multi-Agent Coordination",
        "Human-in-the-Loop",
        "Dynamic RAG",
        "Safety Guardrails"
    ]
}

def get_framework_info():
    """Get information about the Agentix framework."""
    return FRAMEWORK_INFO.copy()
