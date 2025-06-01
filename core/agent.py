"""
Core Agent implementation with comprehensive configuration and execution capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

from .graph import AgentGraph, GraphState
from ..memory.memory_manager import MemoryManager
from ..config.settings import AgentixConfig
from ..utils.exceptions import AgentixError, NodeExecutionError
from ..utils.logging import setup_logging


class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AgentConfig(BaseModel):
    """Configuration for an Agentix agent."""

    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    version: str = Field(default="1.0.0", description="Agent version")

    # Core components
    graph: Optional[AgentGraph] = Field(None, description="Agent execution graph")
    memory_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Memory configuration")

    # Execution settings
    max_iterations: int = Field(default=100, description="Maximum execution iterations")
    timeout_seconds: int = Field(default=300, description="Execution timeout in seconds")
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")

    # Safety settings
    enable_guardrails: bool = Field(default=True, description="Enable safety guardrails")
    require_human_approval: List[str] = Field(default_factory=list, description="Node types requiring human approval")

    # Logging and monitoring
    log_level: str = Field(default="INFO", description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class AgentMetrics(BaseModel):
    """Agent execution metrics."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    total_nodes_executed: int = 0
    memory_operations: int = 0
    tool_calls: int = 0
    guardrail_triggers: int = 0
    human_interventions: int = 0

    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions


class Agent:
    """
    Main Agent class implementing the seven-node blueprint architecture.

    This agent provides:
    - Graph-based execution flow
    - Temporal memory management
    - Safety guardrails
    - Human-in-the-loop capabilities
    - Comprehensive monitoring
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.description = config.description
        self.version = config.version

        # Initialize logging
        self.logger = setup_logging(
            name=f"agentix.agent.{self.name}",
            level=config.log_level
        )

        # Core components
        self.graph = config.graph or AgentGraph()
        self.memory_manager = MemoryManager(config.memory_config)

        # State management
        self.status = AgentStatus.IDLE
        self.current_state: Optional[GraphState] = None
        self.execution_history: List[Dict[str, Any]] = []

        # Metrics
        self.metrics = AgentMetrics()

        # Runtime state
        self._execution_start_time: Optional[datetime] = None
        self._current_iteration = 0

        self.logger.info(f"Agent '{self.name}' initialized with {len(self.graph.nodes)} nodes")

    async def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with the given input data.

        Args:
            input_data: Initial input data for the agent
            **kwargs: Additional execution parameters

        Returns:
            Final execution result
        """
        execution_id = kwargs.get('execution_id', f"{self.name}_{datetime.now().isoformat()}")

        try:
            self.logger.info(f"Starting agent execution: {execution_id}")
            self._execution_start_time = datetime.now()
            self.status = AgentStatus.RUNNING
            self._current_iteration = 0

            # Initialize graph state
            initial_state = GraphState(
                data=input_data.copy(),
                metadata={
                    'execution_id': execution_id,
                    'agent_name': self.name,
                    'start_time': self._execution_start_time.isoformat(),
                    'iteration': 0
                }
            )

            # Execute the graph
            result = await self._execute_graph(initial_state)

            # Update metrics
            execution_time = (datetime.now() - self._execution_start_time).total_seconds()
            self._update_metrics(True, execution_time)

            self.status = AgentStatus.COMPLETED
            self.logger.info(f"Agent execution completed: {execution_id}")

            return result

        except Exception as e:
            execution_time = (datetime.now() - self._execution_start_time).total_seconds() if self._execution_start_time else 0
            self._update_metrics(False, execution_time)
            self.status = AgentStatus.FAILED
            self.logger.error(f"Agent execution failed: {execution_id} - {str(e)}")
            raise AgentixError(f"Agent execution failed: {str(e)}") from e

    async def _execute_graph(self, initial_state: GraphState) -> Dict[str, Any]:
        """Execute the agent graph with the given initial state."""
        current_state = initial_state

        while self._current_iteration < self.config.max_iterations:
            self._current_iteration += 1
            current_state.metadata['iteration'] = self._current_iteration

            try:
                # Execute one iteration of the graph
                result = await self.graph.execute_step(current_state)

                # Check if execution is complete
                if result.is_complete:
                    return result.data

                # Update current state
                current_state = result

                # Store execution step in history
                self.execution_history.append({
                    'iteration': self._current_iteration,
                    'timestamp': datetime.now().isoformat(),
                    'state_data': current_state.data.copy(),
                    'metadata': current_state.metadata.copy()
                })

            except Exception as e:
                self.logger.error(f"Error in iteration {self._current_iteration}: {str(e)}")
                raise NodeExecutionError(f"Graph execution failed at iteration {self._current_iteration}: {str(e)}") from e

        # Max iterations reached
        self.logger.warning(f"Agent reached maximum iterations ({self.config.max_iterations})")
        return current_state.data

    def _update_metrics(self, success: bool, execution_time: float):
        """Update agent execution metrics."""
        self.metrics.total_executions += 1

        if success:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1

        # Update average execution time
        total_time = self.metrics.average_execution_time * (self.metrics.total_executions - 1) + execution_time
        self.metrics.average_execution_time = total_time / self.metrics.total_executions

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            'name': self.name,
            'status': self.status.value,
            'current_iteration': self._current_iteration,
            'metrics': self.metrics.dict(),
            'graph_info': {
                'total_nodes': len(self.graph.nodes),
                'total_edges': len(self.graph.edges)
            }
        }

    def reset(self):
        """Reset agent to initial state."""
        self.status = AgentStatus.IDLE
        self.current_state = None
        self.execution_history.clear()
        self._current_iteration = 0
        self._execution_start_time = None
        self.logger.info(f"Agent '{self.name}' reset to initial state")

    def add_node(self, node_id: str, node_func, node_config: Optional[Dict[str, Any]] = None):
        """Add a node to the agent graph."""
        self.graph.add_node(node_id, node_func, node_config)
        self.logger.info(f"Added node '{node_id}' to agent '{self.name}'")

    def add_edge(self, from_node: str, to_node: str, condition: Optional[callable] = None):
        """Add an edge to the agent graph."""
        self.graph.add_edge(from_node, to_node, condition)
        self.logger.info(f"Added edge from '{from_node}' to '{to_node}' in agent '{self.name}'")
