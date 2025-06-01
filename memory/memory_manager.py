"""
Memory Manager for coordinating different types of memory in Agentix agents.

This module provides a unified interface for managing:
- Temporal knowledge graphs
- Episodic memory
- Semantic memory
- Working memory
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum

from .temporal_graph import TemporalKnowledgeGraph, TemporalQuery, TemporalNode, TemporalEdge
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from ..utils.exceptions import AgentixError


class MemoryType(str, Enum):
    """Types of memory systems."""
    TEMPORAL = "temporal"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


class MemoryOperation(str, Enum):
    """Memory operations."""
    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    CONSOLIDATE = "consolidate"


class MemoryConfig(BaseModel):
    """Configuration for the memory manager."""

    # Temporal knowledge graph settings
    enable_temporal_graph: bool = True
    temporal_graph_config: Dict[str, Any] = Field(default_factory=dict)

    # Episodic memory settings
    enable_episodic_memory: bool = True
    episodic_memory_config: Dict[str, Any] = Field(default_factory=dict)

    # Semantic memory settings
    enable_semantic_memory: bool = True
    semantic_memory_config: Dict[str, Any] = Field(default_factory=dict)

    # Working memory settings
    working_memory_size: int = 1000
    working_memory_ttl: int = 3600  # seconds

    # Consolidation settings
    auto_consolidation: bool = True
    consolidation_interval: int = 3600  # seconds

    # Performance settings
    max_query_results: int = 100
    query_timeout: int = 30  # seconds


class MemoryRequest(BaseModel):
    """Request for memory operations."""

    operation: MemoryOperation
    memory_type: MemoryType
    data: Dict[str, Any] = Field(default_factory=dict)
    query: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Response from memory operations."""

    success: bool
    operation: MemoryOperation
    memory_type: MemoryType
    results: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0


class MemoryManager:
    """
    Unified memory manager for Agentix agents.

    This manager coordinates different memory systems and provides:
    - Unified interface for all memory operations
    - Automatic memory consolidation
    - Cross-memory search and retrieval
    - Memory lifecycle management
    """

    def __init__(self, config: Union[MemoryConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = MemoryConfig(**config)

        self.config = config
        self.logger = logging.getLogger("agentix.memory.manager")

        # Initialize memory systems
        self.temporal_graph: Optional[TemporalKnowledgeGraph] = None
        self.episodic_memory: Optional[EpisodicMemory] = None
        self.semantic_memory: Optional[SemanticMemory] = None
        self.working_memory: Dict[str, Any] = {}

        # Working memory management
        self.working_memory_timestamps: Dict[str, datetime] = {}

        # Statistics
        self.stats = {
            'total_operations': 0,
            'operations_by_type': {},
            'memory_usage': {},
            'last_consolidation': None
        }

        self._initialize_memory_systems()

        # Start background tasks
        if config.auto_consolidation:
            try:
                asyncio.create_task(self._consolidation_loop())
            except RuntimeError:
                # No event loop running, will start consolidation when needed
                self.logger.debug("No event loop running, deferring consolidation task")

    def _initialize_memory_systems(self):
        """Initialize the configured memory systems."""

        if self.config.enable_temporal_graph:
            self.temporal_graph = TemporalKnowledgeGraph(self.config.temporal_graph_config)
            self.logger.info("Temporal knowledge graph initialized")

        if self.config.enable_episodic_memory:
            self.episodic_memory = EpisodicMemory(self.config.episodic_memory_config)
            self.logger.info("Episodic memory initialized")

        if self.config.enable_semantic_memory:
            self.semantic_memory = SemanticMemory(self.config.semantic_memory_config)
            self.logger.info("Semantic memory initialized")

        self.logger.info("Memory manager initialized with all configured systems")

    async def execute(self, request: MemoryRequest) -> MemoryResponse:
        """Execute a memory operation."""
        start_time = datetime.now()

        try:
            self.logger.debug(f"Executing {request.operation} on {request.memory_type} memory")

            # Route to appropriate memory system
            if request.memory_type == MemoryType.TEMPORAL:
                result = await self._execute_temporal(request)
            elif request.memory_type == MemoryType.EPISODIC:
                result = await self._execute_episodic(request)
            elif request.memory_type == MemoryType.SEMANTIC:
                result = await self._execute_semantic(request)
            elif request.memory_type == MemoryType.WORKING:
                result = await self._execute_working(request)
            else:
                raise ValueError(f"Unknown memory type: {request.memory_type}")

            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(request.operation, request.memory_type, execution_time)

            return MemoryResponse(
                success=True,
                operation=request.operation,
                memory_type=request.memory_type,
                results=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Memory operation failed: {str(e)}")

            return MemoryResponse(
                success=False,
                operation=request.operation,
                memory_type=request.memory_type,
                error=str(e),
                execution_time=execution_time
            )

    async def _execute_temporal(self, request: MemoryRequest) -> List[Dict[str, Any]]:
        """Execute operations on temporal knowledge graph."""
        if not self.temporal_graph:
            raise AgentixError("Temporal knowledge graph not initialized")

        if request.operation == MemoryOperation.STORE:
            return await self._store_temporal(request.data)
        elif request.operation == MemoryOperation.RETRIEVE:
            return await self._retrieve_temporal(request.query or {})
        elif request.operation == MemoryOperation.SEARCH:
            return await self._search_temporal(request.query or {})
        elif request.operation == MemoryOperation.UPDATE:
            return await self._update_temporal(request.data)
        else:
            raise ValueError(f"Unsupported temporal operation: {request.operation}")

    async def _store_temporal(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Store data in temporal knowledge graph."""
        results = []

        # Store nodes
        if 'nodes' in data:
            for node_data in data['nodes']:
                node = TemporalNode(**node_data)
                node_id = self.temporal_graph.add_node(node)
                results.append({'type': 'node', 'id': node_id, 'status': 'stored'})

        # Store edges
        if 'edges' in data:
            for edge_data in data['edges']:
                edge = TemporalEdge(**edge_data)
                edge_id = self.temporal_graph.add_edge(edge)
                results.append({'type': 'edge', 'id': edge_id, 'status': 'stored'})

        return results

    async def _retrieve_temporal(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve data from temporal knowledge graph."""
        temporal_query = TemporalQuery(query_type="retrieve", **query)
        return self.temporal_graph.query(temporal_query)

    async def _search_temporal(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search temporal knowledge graph."""
        temporal_query = TemporalQuery(query_type="search", **query)
        return self.temporal_graph.query(temporal_query)

    async def _update_temporal(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update temporal knowledge graph data."""
        results = []

        # Update nodes
        if 'nodes' in data:
            for node_data in data['nodes']:
                node_id = node_data.get('id')
                if node_id and node_id in self.temporal_graph.nodes:
                    node = self.temporal_graph.nodes[node_id]
                    for key, value in node_data.items():
                        if hasattr(node, key):
                            setattr(node, key, value)
                    node.updated_at = datetime.now()
                    results.append({'type': 'node', 'id': node_id, 'status': 'updated'})

        return results

    async def _execute_episodic(self, request: MemoryRequest) -> List[Dict[str, Any]]:
        """Execute operations on episodic memory."""
        if not self.episodic_memory:
            raise AgentixError("Episodic memory not initialized")

        # TODO: Implement episodic memory operations
        return [{'status': 'episodic_operation_placeholder'}]

    async def _execute_semantic(self, request: MemoryRequest) -> List[Dict[str, Any]]:
        """Execute operations on semantic memory."""
        if not self.semantic_memory:
            raise AgentixError("Semantic memory not initialized")

        # TODO: Implement semantic memory operations
        return [{'status': 'semantic_operation_placeholder'}]

    async def _execute_working(self, request: MemoryRequest) -> List[Dict[str, Any]]:
        """Execute operations on working memory."""

        if request.operation == MemoryOperation.STORE:
            return self._store_working(request.data)
        elif request.operation == MemoryOperation.RETRIEVE:
            return self._retrieve_working(request.query or {})
        elif request.operation == MemoryOperation.DELETE:
            return self._delete_working(request.query or {})
        else:
            raise ValueError(f"Unsupported working memory operation: {request.operation}")

    def _store_working(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Store data in working memory."""
        results = []

        for key, value in data.items():
            self.working_memory[key] = value
            self.working_memory_timestamps[key] = datetime.now()
            results.append({'key': key, 'status': 'stored'})

        # Clean up old entries
        self._cleanup_working_memory()

        return results

    def _retrieve_working(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve data from working memory."""
        results = []

        if 'keys' in query:
            for key in query['keys']:
                if key in self.working_memory:
                    results.append({
                        'key': key,
                        'value': self.working_memory[key],
                        'timestamp': self.working_memory_timestamps[key].isoformat()
                    })
        else:
            # Return all working memory
            for key, value in self.working_memory.items():
                results.append({
                    'key': key,
                    'value': value,
                    'timestamp': self.working_memory_timestamps[key].isoformat()
                })

        return results

    def _delete_working(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Delete data from working memory."""
        results = []

        if 'keys' in query:
            for key in query['keys']:
                if key in self.working_memory:
                    del self.working_memory[key]
                    del self.working_memory_timestamps[key]
                    results.append({'key': key, 'status': 'deleted'})

        return results

    def _cleanup_working_memory(self):
        """Clean up expired working memory entries."""
        current_time = datetime.now()
        expired_keys = []

        for key, timestamp in self.working_memory_timestamps.items():
            if (current_time - timestamp).total_seconds() > self.config.working_memory_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.working_memory[key]
            del self.working_memory_timestamps[key]

    def _update_stats(self, operation: MemoryOperation, memory_type: MemoryType, execution_time: float):
        """Update memory operation statistics."""
        self.stats['total_operations'] += 1

        op_key = f"{memory_type}_{operation}"
        if op_key not in self.stats['operations_by_type']:
            self.stats['operations_by_type'][op_key] = 0
        self.stats['operations_by_type'][op_key] += 1

        # Update memory usage stats
        self.stats['memory_usage'] = {
            'temporal_nodes': len(self.temporal_graph.nodes) if self.temporal_graph else 0,
            'temporal_edges': len(self.temporal_graph.edges) if self.temporal_graph else 0,
            'working_memory_items': len(self.working_memory)
        }

    async def _consolidation_loop(self):
        """Background task for memory consolidation."""
        while True:
            try:
                await asyncio.sleep(self.config.consolidation_interval)
                await self.consolidate_memory()
            except Exception as e:
                self.logger.error(f"Memory consolidation failed: {str(e)}")

    async def consolidate_memory(self):
        """Consolidate memory across different systems."""
        self.logger.info("Starting memory consolidation")

        # Clean up expired temporal graph entries
        if self.temporal_graph:
            self.temporal_graph.cleanup_expired_nodes()

        # Clean up working memory
        self._cleanup_working_memory()

        # TODO: Implement cross-memory consolidation logic
        # - Move important working memory to long-term storage
        # - Consolidate related episodic memories
        # - Update semantic memory based on patterns

        self.stats['last_consolidation'] = datetime.now()
        self.logger.info("Memory consolidation completed")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        return self.stats.copy()
