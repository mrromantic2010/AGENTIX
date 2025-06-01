"""
Zero-config agent creation decorators for Agentix.

This module provides the "FastAPI for AI Agents" experience with simple decorators
that hide the complexity of the seven-node blueprint while maintaining full power.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from functools import wraps
from datetime import datetime

from .core.agent import Agent, AgentConfig
from .core.graph import AgentGraph
from .core.nodes import (
    LLMNode, ToolNode, ControlNode, MemoryNode, GuardrailNode, FallbackNode,
    LLMNodeConfig, ToolNodeConfig, ControlNodeConfig, MemoryNodeConfig,
    GuardrailNodeConfig, FallbackNodeConfig
)
from .tools.web_search import WebSearchTool, WebSearchConfig
from .tools.base import ToolRegistry
from .memory.memory_manager import MemoryManager, MemoryConfig
from .config.llm_config import LLMConfig, LLMProvider
from .utils.logging import get_logger


class AgentixAgent:
    """
    Zero-config agent wrapper that provides a simple interface
    while using the full seven-node blueprint internally.
    """

    def __init__(self,
                 name: str = "AgentixAgent",
                 llm_model: str = "gpt-4",
                 temperature: float = 0.7,
                 tools: List[str] = None,
                 memory: bool = True,
                 guardrails: bool = True,
                 **kwargs):

        self.name = name
        self.llm_model = llm_model
        self.temperature = temperature
        self.tools = tools or ["search"]
        self.memory_enabled = memory
        self.guardrails_enabled = guardrails
        self.kwargs = kwargs

        self.logger = get_logger(f"agentix.agent.{name}")
        self.agent: Optional[Agent] = None
        self.tool_registry = ToolRegistry()

        # Initialize the agent
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the underlying Agent with seven-node blueprint."""

        # Create agent graph with seven-node blueprint
        graph = self._create_default_graph()

        # Create agent configuration
        agent_config = AgentConfig(
            name=self.name,
            description=f"Zero-config agent: {self.name}",
            version="1.0.0",
            graph=graph,
            memory_config={
                "enable_temporal_graph": self.memory_enabled,
                "enable_episodic_memory": self.memory_enabled,
                "auto_consolidation": True
            } if self.memory_enabled else {},
            max_iterations=10,
            timeout_seconds=120,
            enable_guardrails=self.guardrails_enabled,
            enable_monitoring=True
        )

        # Create the agent
        self.agent = Agent(agent_config)
        self.logger.info(f"✅ Zero-config agent '{self.name}' initialized")

    def _create_default_graph(self) -> AgentGraph:
        """Create a default seven-node blueprint graph."""

        graph = AgentGraph()

        # 1. Input Validation (if guardrails enabled)
        if self.guardrails_enabled:
            input_validation_config = GuardrailNodeConfig(
                node_type="guardrail",
                name="input_validator",
                description="Validate user input",
                validation_type="input",
                rules=[
                    {"name": "length_check", "type": "length", "min_length": 1, "max_length": 2000}
                ],
                strict_mode=False
            )
            input_validator = GuardrailNode(input_validation_config)
            graph.add_node("input_validation", input_validator)

        # 2. Memory Retrieval (if memory enabled)
        if self.memory_enabled:
            memory_config = MemoryNodeConfig(
                node_type="memory",
                name="memory_retriever",
                description="Retrieve relevant context",
                operation="search",
                memory_type="temporal",
                query_params={"max_results": 5}
            )
            memory_node = MemoryNode(memory_config)
            graph.add_node("memory_retrieval", memory_node)

        # 3. Main LLM Processing
        llm_config = LLMNodeConfig(
            node_type="llm",
            name="main_llm",
            description="Main language model processing",
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=1000,
            prompt_template=self._get_default_prompt_template()
        )
        main_llm = LLMNode(llm_config)
        graph.add_node("main_processing", main_llm)

        # 4. Tool Control (if tools enabled)
        if self.tools:
            control_config = ControlNodeConfig(
                node_type="control",
                name="tool_controller",
                description="Decide whether to use tools",
                condition_type="simple",
                conditions=[
                    {"name": "needs_tools", "condition": "tool_required", "path": "tool_execution"}
                ],
                default_path="output_validation" if self.guardrails_enabled else "end"
            )
            tool_controller = ControlNode(control_config)
            graph.add_node("tool_control", tool_controller)

            # 5. Tool Execution
            self._setup_tools()
            tool_config = ToolNodeConfig(
                node_type="tool",
                name="tool_executor",
                description="Execute tools as needed",
                tool_name="search" if "search" in self.tools else self.tools[0],
                validate_input=True,
                validate_output=True
            )
            tool_executor = ToolNode(tool_config)
            graph.add_node("tool_execution", tool_executor)

        # 6. Memory Storage (if memory enabled)
        if self.memory_enabled:
            memory_storage_config = MemoryNodeConfig(
                node_type="memory",
                name="memory_storer",
                description="Store conversation and results",
                operation="store",
                memory_type="temporal"
            )
            memory_storer = MemoryNode(memory_storage_config)
            graph.add_node("memory_storage", memory_storer)

        # 7. Output Validation (if guardrails enabled)
        if self.guardrails_enabled:
            output_validation_config = GuardrailNodeConfig(
                node_type="guardrail",
                name="output_validator",
                description="Validate output quality and safety",
                validation_type="output",
                rules=[
                    {"name": "quality_check", "type": "quality", "min_quality_score": 0.6}
                ]
            )
            output_validator = GuardrailNode(output_validation_config)
            graph.add_node("output_validation", output_validator)

        # Define execution flow
        self._connect_graph_nodes(graph)

        return graph

    def _connect_graph_nodes(self, graph: AgentGraph):
        """Connect the graph nodes based on enabled features."""

        # Determine start node
        if self.guardrails_enabled:
            start_node = "input_validation"
            current_node = "input_validation"
        elif self.memory_enabled:
            start_node = "memory_retrieval"
            current_node = "memory_retrieval"
        else:
            start_node = "main_processing"
            current_node = "main_processing"

        graph.set_start_node(start_node)

        # Connect the flow
        if self.guardrails_enabled and self.memory_enabled:
            graph.add_edge("input_validation", "memory_retrieval")
            current_node = "memory_retrieval"

        if self.memory_enabled:
            graph.add_edge(current_node, "main_processing")
            current_node = "main_processing"

        if self.tools:
            graph.add_edge(current_node, "tool_control")
            graph.add_edge("tool_control", "tool_execution")
            graph.add_edge("tool_control", "output_validation" if self.guardrails_enabled else "end")
            current_node = "tool_execution"

        if self.memory_enabled:
            graph.add_edge(current_node, "memory_storage")
            current_node = "memory_storage"

        if self.guardrails_enabled:
            graph.add_edge(current_node, "output_validation")
            graph.add_end_node("output_validation")
        else:
            graph.add_end_node(current_node)

    def _setup_tools(self):
        """Setup default tools."""

        if "search" in self.tools:
            search_config = WebSearchConfig(
                name="search",
                description="Web search tool",
                search_engine="duckduckgo",
                max_results=3,
                extract_content=True
            )
            search_tool = WebSearchTool(search_config)
            self.tool_registry.register_tool(search_tool)

    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template."""

        template = "You are a helpful AI assistant. "

        if self.memory_enabled:
            template += "Use the provided context from memory to inform your response. "

        if self.tools:
            template += "You have access to tools for additional information. "

        template += """

User Query: {query}
"""

        if self.memory_enabled:
            template += "Context: {memory_context}\n"

        template += """
Provide a helpful, accurate, and concise response to the user's query.
"""

        return template

    def __call__(self, query: str, **kwargs) -> str:
        """
        Simple synchronous interface for agent execution.

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            Agent response as string
        """
        return asyncio.run(self.arun(query, **kwargs))

    async def arun(self, query: str, **kwargs) -> str:
        """
        Asynchronous interface for agent execution.

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            Agent response as string
        """

        if not self.agent:
            raise RuntimeError("Agent not initialized")

        # Prepare input data
        input_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "user_id": kwargs.get("user_id", "default_user"),
            **kwargs
        }

        try:
            # Execute the agent
            result = await self.agent.run(input_data)

            # Extract the response
            if "main_processing_result" in result:
                return result["main_processing_result"].get("response", "No response generated")
            elif "output_validation_result" in result:
                return result["output_validation_result"].get("validated_content", "No response generated")
            else:
                return "No response generated"

        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}")
            return f"Error: {str(e)}"

    def chat(self, query: str, **kwargs) -> str:
        """Alias for __call__ to provide chat-like interface."""
        return self(query, **kwargs)

    async def achat(self, query: str, **kwargs) -> str:
        """Async alias for arun to provide chat-like interface."""
        return await self.arun(query, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        if self.agent:
            return self.agent.get_status()
        return {}


def agent(name: str = None,
          llm_model: str = "gpt-4",
          temperature: float = 0.7,
          tools: List[str] = None,
          memory: bool = True,
          guardrails: bool = True,
          **kwargs):
    """
    Decorator for zero-config agent creation.

    Usage:
        @agentix.agent
        def my_agent():
            pass

        response = my_agent("What's the weather?")

    Args:
        name: Agent name (defaults to function name)
        llm_model: LLM model to use
        temperature: LLM temperature
        tools: List of tools to enable
        memory: Enable temporal memory
        guardrails: Enable safety guardrails
        **kwargs: Additional configuration
    """

    def decorator(func: Callable):
        agent_name = name or func.__name__

        # Create the agent instance
        agent_instance = AgentixAgent(
            name=agent_name,
            llm_model=llm_model,
            temperature=temperature,
            tools=tools,
            memory=memory,
            guardrails=guardrails,
            **kwargs
        )

        @wraps(func)
        def wrapper(query: str, **call_kwargs):
            return agent_instance(query, **call_kwargs)

        # Add async method
        wrapper.arun = agent_instance.arun
        wrapper.achat = agent_instance.achat
        wrapper.get_stats = agent_instance.get_stats
        wrapper._agent = agent_instance

        return wrapper

    return decorator


def create_agent(name: str = "QuickAgent", **kwargs) -> AgentixAgent:
    """
    Functional interface for creating agents.

    Usage:
        agent = agentix.create_agent("MyAgent", tools=["search"])
        response = agent("What's the weather in Paris?")

    Args:
        name: Agent name
        **kwargs: Agent configuration

    Returns:
        AgentixAgent instance
    """
    return AgentixAgent(name=name, **kwargs)


# Convenience functions for common agent types
def search_agent(name: str = "SearchAgent", **kwargs) -> AgentixAgent:
    """Create an agent optimized for web search tasks."""
    return create_agent(
        name=name,
        tools=["search"],
        memory=True,
        guardrails=True,
        temperature=0.3,
        **kwargs
    )


def chat_agent(name: str = "ChatAgent", **kwargs) -> AgentixAgent:
    """Create an agent optimized for conversational tasks."""
    return create_agent(
        name=name,
        tools=[],
        memory=True,
        guardrails=True,
        temperature=0.7,
        **kwargs
    )


def research_agent(name: str = "ResearchAgent", **kwargs) -> AgentixAgent:
    """Create an agent optimized for research tasks."""
    return create_agent(
        name=name,
        tools=["search"],
        memory=True,
        guardrails=True,
        temperature=0.4,
        llm_model="gpt-4",
        **kwargs
    )


def openrouter_agent(name: str = "OpenRouterAgent",
                    model: str = "openai/gpt-4-turbo",
                    **kwargs) -> AgentixAgent:
    """Create an agent using OpenRouter for access to multiple LLM providers."""
    return create_agent(
        name=name,
        llm_model=model,
        tools=["search"],
        memory=True,
        guardrails=True,
        temperature=0.7,
        **kwargs
    )


def anthropic_agent(name: str = "ClaudeAgent",
                   model: str = "claude-3-5-sonnet-20241022",
                   **kwargs) -> AgentixAgent:
    """Create an agent using Anthropic Claude models."""
    return create_agent(
        name=name,
        llm_model=model,
        tools=["search"],
        memory=True,
        guardrails=True,
        temperature=0.7,
        **kwargs
    )
