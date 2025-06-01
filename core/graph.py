"""
Graph orchestration system for agent execution flow.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from ..utils.exceptions import NodeExecutionError, AgentixError


class GraphState(BaseModel):
    """State object passed between nodes in the agent graph."""
    
    data: Dict[str, Any] = Field(default_factory=dict, description="Main data payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    is_complete: bool = Field(default=False, description="Whether execution is complete")
    next_node: Optional[str] = Field(None, description="Next node to execute")
    error: Optional[str] = Field(None, description="Error message if any")
    
    def copy_with_updates(self, **updates) -> 'GraphState':
        """Create a copy of the state with updates."""
        new_data = self.dict()
        new_data.update(updates)
        return GraphState(**new_data)


class EdgeCondition(BaseModel):
    """Condition for graph edge traversal."""
    
    condition_func: Callable[[GraphState], bool]
    description: str = ""
    
    def evaluate(self, state: GraphState) -> bool:
        """Evaluate the condition against the current state."""
        try:
            return self.condition_func(state)
        except Exception as e:
            logging.error(f"Edge condition evaluation failed: {str(e)}")
            return False


class GraphNode(BaseModel):
    """Node in the agent execution graph."""
    
    node_id: str
    node_func: Callable[[GraphState], Union[GraphState, Dict[str, Any]]]
    config: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    node_type: str = "generic"
    
    class Config:
        arbitrary_types_allowed = True
    
    async def execute(self, state: GraphState) -> GraphState:
        """Execute the node function with the given state."""
        try:
            logging.info(f"Executing node: {self.node_id}")
            
            # Execute the node function
            if asyncio.iscoroutinefunction(self.node_func):
                result = await self.node_func(state)
            else:
                result = self.node_func(state)
            
            # Handle different return types
            if isinstance(result, GraphState):
                return result
            elif isinstance(result, dict):
                # Merge result into state data
                new_state = state.copy_with_updates(
                    data={**state.data, **result}
                )
                return new_state
            else:
                # Assume single value result
                new_state = state.copy_with_updates(
                    data={**state.data, f"{self.node_id}_result": result}
                )
                return new_state
                
        except Exception as e:
            logging.error(f"Node execution failed: {self.node_id} - {str(e)}")
            raise NodeExecutionError(f"Node '{self.node_id}' execution failed: {str(e)}") from e


class AgentGraph:
    """
    Graph orchestration system for agent execution.
    
    Implements the execution flow between different node types in the seven-node blueprint.
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[Dict[str, Any]]] = {}
        self.start_node: Optional[str] = None
        self.end_nodes: List[str] = []
        
        self.logger = logging.getLogger("agentix.graph")
    
    def add_node(self, node_id: str, node_func: Callable, config: Optional[Dict[str, Any]] = None, 
                 description: str = "", node_type: str = "generic"):
        """Add a node to the graph."""
        if node_id in self.nodes:
            raise AgentixError(f"Node '{node_id}' already exists in graph")
        
        self.nodes[node_id] = GraphNode(
            node_id=node_id,
            node_func=node_func,
            config=config or {},
            description=description,
            node_type=node_type
        )
        
        # Set as start node if it's the first node
        if not self.start_node:
            self.start_node = node_id
        
        self.logger.info(f"Added node '{node_id}' of type '{node_type}'")
    
    def add_edge(self, from_node: str, to_node: str, condition: Optional[Callable[[GraphState], bool]] = None,
                 description: str = ""):
        """Add an edge between two nodes."""
        if from_node not in self.nodes:
            raise AgentixError(f"Source node '{from_node}' does not exist")
        if to_node not in self.nodes:
            raise AgentixError(f"Target node '{to_node}' does not exist")
        
        edge_info = {
            'to_node': to_node,
            'condition': EdgeCondition(
                condition_func=condition or (lambda state: True),
                description=description
            )
        }
        
        if from_node not in self.edges:
            self.edges[from_node] = []
        
        self.edges[from_node].append(edge_info)
        self.logger.info(f"Added edge from '{from_node}' to '{to_node}'")
    
    def set_start_node(self, node_id: str):
        """Set the starting node for graph execution."""
        if node_id not in self.nodes:
            raise AgentixError(f"Node '{node_id}' does not exist")
        self.start_node = node_id
        self.logger.info(f"Set start node to '{node_id}'")
    
    def add_end_node(self, node_id: str):
        """Mark a node as an end node."""
        if node_id not in self.nodes:
            raise AgentixError(f"Node '{node_id}' does not exist")
        if node_id not in self.end_nodes:
            self.end_nodes.append(node_id)
        self.logger.info(f"Added end node '{node_id}'")
    
    async def execute_step(self, state: GraphState) -> GraphState:
        """Execute a single step in the graph."""
        if not self.start_node:
            raise AgentixError("No start node defined for graph")
        
        # Determine current node
        current_node_id = state.next_node or self.start_node
        
        if current_node_id not in self.nodes:
            raise AgentixError(f"Node '{current_node_id}' does not exist in graph")
        
        # Execute current node
        current_node = self.nodes[current_node_id]
        result_state = await current_node.execute(state)
        
        # Update metadata
        result_state.metadata.update({
            'last_executed_node': current_node_id,
            'execution_timestamp': datetime.now().isoformat()
        })
        
        # Determine next node
        next_node = self._get_next_node(current_node_id, result_state)
        result_state.next_node = next_node
        
        # Check if execution is complete
        if next_node is None or current_node_id in self.end_nodes:
            result_state.is_complete = True
        
        return result_state
    
    def _get_next_node(self, current_node_id: str, state: GraphState) -> Optional[str]:
        """Determine the next node to execute based on current state."""
        # Check if next_node is explicitly set in state
        if state.next_node and state.next_node in self.nodes:
            return state.next_node
        
        # Check edges from current node
        if current_node_id not in self.edges:
            return None
        
        # Evaluate edge conditions
        for edge_info in self.edges[current_node_id]:
            condition = edge_info['condition']
            if condition.evaluate(state):
                return edge_info['to_node']
        
        # No valid edge found
        return None
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete graph with the given input data."""
        if not self.start_node:
            raise AgentixError("No start node defined for graph")
        
        # Initialize state
        state = GraphState(
            data=input_data.copy(),
            metadata={
                'start_time': datetime.now().isoformat(),
                'graph_execution': True
            }
        )
        
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while not state.is_complete and iteration < max_iterations:
            iteration += 1
            state.metadata['iteration'] = iteration
            
            try:
                state = await self.execute_step(state)
            except Exception as e:
                self.logger.error(f"Graph execution failed at iteration {iteration}: {str(e)}")
                raise
        
        if iteration >= max_iterations:
            raise AgentixError(f"Graph execution exceeded maximum iterations ({max_iterations})")
        
        return state.data
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph structure."""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': sum(len(edges) for edges in self.edges.values()),
            'start_node': self.start_node,
            'end_nodes': self.end_nodes.copy(),
            'nodes': {
                node_id: {
                    'type': node.node_type,
                    'description': node.description
                }
                for node_id, node in self.nodes.items()
            },
            'edges': {
                from_node: [
                    {
                        'to_node': edge['to_node'],
                        'description': edge['condition'].description
                    }
                    for edge in edges
                ]
                for from_node, edges in self.edges.items()
            }
        }
    
    def validate_graph(self) -> List[str]:
        """Validate the graph structure and return any issues."""
        issues = []
        
        if not self.start_node:
            issues.append("No start node defined")
        
        if not self.end_nodes:
            issues.append("No end nodes defined")
        
        # Check for unreachable nodes
        reachable = set()
        if self.start_node:
            self._find_reachable_nodes(self.start_node, reachable)
        
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            issues.append(f"Unreachable nodes: {list(unreachable)}")
        
        return issues
    
    def _find_reachable_nodes(self, node_id: str, reachable: set):
        """Recursively find all reachable nodes from a given node."""
        if node_id in reachable:
            return
        
        reachable.add(node_id)
        
        if node_id in self.edges:
            for edge_info in self.edges[node_id]:
                self._find_reachable_nodes(edge_info['to_node'], reachable)
