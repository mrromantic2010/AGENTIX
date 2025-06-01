"""
Progressive disclosure architecture for Agentix.

This module provides three levels of complexity:
1. Beginner: One-liner agent creation with decorators
2. Intermediate: YAML/JSON configuration files
3. Advanced: Full AgentGraph programming with custom nodes
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field

from .decorators import AgentixAgent, create_agent
from .core.agent import Agent, AgentConfig
from .core.graph import AgentGraph
from .core.nodes import *
from .config.settings import AgentixConfig
from .utils.logging import get_logger


class AgentDefinition(BaseModel):
    """
    Intermediate-level agent definition for YAML/JSON configuration.
    """
    
    # Basic configuration
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # LLM configuration
    llm: Dict[str, Any] = Field(default_factory=lambda: {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    })
    
    # Tools configuration
    tools: List[str] = Field(default_factory=lambda: ["search"])
    
    # Memory configuration
    memory: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "temporal_graph": True,
        "episodic": True,
        "auto_consolidation": True
    })
    
    # Guardrails configuration
    guardrails: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "input_validation": True,
        "output_validation": True,
        "content_filtering": True
    })
    
    # Custom prompt template
    prompt_template: Optional[str] = None
    
    # Advanced configuration
    graph_config: Optional[Dict[str, Any]] = None
    custom_nodes: Optional[Dict[str, Any]] = None
    
    # Deployment configuration
    deployment: Dict[str, Any] = Field(default_factory=lambda: {
        "timeout": 120,
        "max_iterations": 10,
        "enable_monitoring": True
    })


class ProgressiveAgentBuilder:
    """
    Builder class that supports progressive disclosure from simple to complex agents.
    """
    
    def __init__(self):
        self.logger = get_logger("agentix.progressive")
    
    def from_decorator(self, **kwargs) -> AgentixAgent:
        """
        Level 1: Create agent using decorator-style configuration.
        
        Usage:
            builder = ProgressiveAgentBuilder()
            agent = builder.from_decorator(name="MyAgent", tools=["search"])
        """
        return create_agent(**kwargs)
    
    def from_config(self, config: Union[str, Path, Dict[str, Any], AgentDefinition]) -> AgentixAgent:
        """
        Level 2: Create agent from YAML/JSON configuration.
        
        Usage:
            builder = ProgressiveAgentBuilder()
            agent = builder.from_config("my_agent.yaml")
        """
        
        # Parse configuration
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            agent_def = AgentDefinition(**config_data)
        
        elif isinstance(config, dict):
            agent_def = AgentDefinition(**config)
        
        elif isinstance(config, AgentDefinition):
            agent_def = config
        
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
        
        # Create agent from definition
        return self._create_from_definition(agent_def)
    
    def from_graph(self, graph: AgentGraph, config: Optional[Dict[str, Any]] = None) -> Agent:
        """
        Level 3: Create agent from full AgentGraph programming.
        
        Usage:
            builder = ProgressiveAgentBuilder()
            graph = AgentGraph()
            # ... build custom graph ...
            agent = builder.from_graph(graph)
        """
        
        config = config or {}
        
        agent_config = AgentConfig(
            name=config.get("name", "CustomAgent"),
            description=config.get("description", "Custom agent with full graph"),
            version=config.get("version", "1.0.0"),
            graph=graph,
            **config
        )
        
        return Agent(agent_config)
    
    def _create_from_definition(self, agent_def: AgentDefinition) -> AgentixAgent:
        """Create agent from AgentDefinition."""
        
        # Extract configuration
        kwargs = {
            "name": agent_def.name,
            "llm_model": agent_def.llm.get("model", "gpt-4"),
            "temperature": agent_def.llm.get("temperature", 0.7),
            "tools": agent_def.tools,
            "memory": agent_def.memory.get("enabled", True),
            "guardrails": agent_def.guardrails.get("enabled", True)
        }
        
        # Add custom prompt template if provided
        if agent_def.prompt_template:
            kwargs["prompt_template"] = agent_def.prompt_template
        
        # Create the agent
        agent = create_agent(**kwargs)
        
        self.logger.info(f"Created agent '{agent_def.name}' from configuration")
        return agent
    
    def upgrade_to_config(self, agent: AgentixAgent, output_path: str):
        """
        Upgrade a decorator-created agent to configuration file.
        
        This allows users to start simple and progressively add complexity.
        """
        
        # Extract current configuration
        config = AgentDefinition(
            name=agent.name,
            description=f"Agent: {agent.name}",
            llm={
                "model": agent.llm_model,
                "temperature": agent.temperature
            },
            tools=agent.tools,
            memory={"enabled": agent.memory_enabled},
            guardrails={"enabled": agent.guardrails_enabled}
        )
        
        # Save to file
        output_path = Path(output_path)
        
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(config.dict(), f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(config.dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        self.logger.info(f"Upgraded agent configuration saved to: {output_path}")
    
    def upgrade_to_graph(self, agent: AgentixAgent) -> AgentGraph:
        """
        Upgrade a simple agent to full AgentGraph for advanced customization.
        
        This extracts the underlying graph for manual modification.
        """
        
        if agent.agent and agent.agent.config.graph:
            return agent.agent.config.graph
        
        raise ValueError("Agent does not have an underlying graph to extract")


class ConfigTemplates:
    """
    Pre-built configuration templates for common agent types.
    """
    
    @staticmethod
    def web_search_bot() -> AgentDefinition:
        """Template for web search bot."""
        return AgentDefinition(
            name="WebSearchBot",
            description="Agent specialized in web search and information retrieval",
            llm={
                "model": "gpt-4",
                "temperature": 0.3,
                "max_tokens": 1500
            },
            tools=["search"],
            memory={
                "enabled": True,
                "temporal_graph": True,
                "episodic": True
            },
            guardrails={
                "enabled": True,
                "input_validation": True,
                "output_validation": True
            },
            prompt_template="""
You are a web search specialist. Your job is to find accurate, up-to-date information from the web.

User Query: {query}
Search Results: {search_results}
Context: {memory_context}

Provide a comprehensive answer based on the search results. Include:
1. Direct answer to the query
2. Supporting evidence from sources
3. Source citations
4. Any limitations or caveats
"""
        )
    
    @staticmethod
    def summarizer() -> AgentDefinition:
        """Template for document summarizer."""
        return AgentDefinition(
            name="Summarizer",
            description="Agent specialized in document summarization",
            llm={
                "model": "gpt-4",
                "temperature": 0.2,
                "max_tokens": 1000
            },
            tools=[],
            memory={
                "enabled": True,
                "temporal_graph": False,
                "episodic": True
            },
            prompt_template="""
You are a professional summarizer. Create concise, accurate summaries.

Document/Text: {query}
Previous Summaries: {memory_context}

Create a summary that:
1. Captures key points and main ideas
2. Maintains important details
3. Uses clear, concise language
4. Preserves the original tone and intent
"""
        )
    
    @staticmethod
    def form_filler() -> AgentDefinition:
        """Template for form filling agent."""
        return AgentDefinition(
            name="FormFiller",
            description="Agent specialized in filling out forms and structured data",
            llm={
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 800
            },
            tools=["search"],
            memory={
                "enabled": True,
                "temporal_graph": True,
                "episodic": True
            },
            prompt_template="""
You are a form filling specialist. Extract and organize information to fill forms accurately.

Form Fields: {query}
Available Information: {search_results}
Previous Forms: {memory_context}

Fill out the form with:
1. Accurate information from available sources
2. Consistent formatting
3. Complete all required fields
4. Flag any missing information
"""
        )
    
    @staticmethod
    def email_responder() -> AgentDefinition:
        """Template for email response agent."""
        return AgentDefinition(
            name="EmailResponder",
            description="Agent specialized in email responses",
            llm={
                "model": "gpt-4",
                "temperature": 0.6,
                "max_tokens": 1200
            },
            tools=["search"],
            memory={
                "enabled": True,
                "temporal_graph": True,
                "episodic": True
            },
            prompt_template="""
You are a professional email assistant. Craft appropriate email responses.

Email Content: {query}
Context: {memory_context}
Additional Info: {search_results}

Write a response that:
1. Addresses all points in the original email
2. Maintains appropriate tone and professionalism
3. Provides helpful information
4. Includes clear next steps if needed
"""
        )
    
    @staticmethod
    def multi_step_planner() -> AgentDefinition:
        """Template for multi-step planning agent."""
        return AgentDefinition(
            name="MultiStepPlanner",
            description="Agent specialized in breaking down complex tasks into steps",
            llm={
                "model": "gpt-4",
                "temperature": 0.4,
                "max_tokens": 2000
            },
            tools=["search"],
            memory={
                "enabled": True,
                "temporal_graph": True,
                "episodic": True
            },
            prompt_template="""
You are a strategic planning expert. Break down complex tasks into actionable steps.

Task/Goal: {query}
Available Resources: {search_results}
Previous Plans: {memory_context}

Create a detailed plan with:
1. Clear, sequential steps
2. Resource requirements for each step
3. Timeline estimates
4. Success criteria
5. Potential obstacles and mitigation strategies
"""
        )


# Global builder instance for convenience
builder = ProgressiveAgentBuilder()

# Convenience functions for each level
def simple_agent(**kwargs) -> AgentixAgent:
    """Level 1: Create simple agent with decorator-style config."""
    return builder.from_decorator(**kwargs)

def config_agent(config_path: str) -> AgentixAgent:
    """Level 2: Create agent from configuration file."""
    return builder.from_config(config_path)

def graph_agent(graph: AgentGraph, **kwargs) -> Agent:
    """Level 3: Create agent from custom graph."""
    return builder.from_graph(graph, kwargs)

def template_agent(template_name: str) -> AgentDefinition:
    """Get a pre-built agent template."""
    templates = {
        "web_search_bot": ConfigTemplates.web_search_bot,
        "summarizer": ConfigTemplates.summarizer,
        "form_filler": ConfigTemplates.form_filler,
        "email_responder": ConfigTemplates.email_responder,
        "multi_step_planner": ConfigTemplates.multi_step_planner
    }
    
    if template_name not in templates:
        available = ", ".join(templates.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")
    
    return templates[template_name]()
