"""
Temporal Knowledge Graph implementation for dynamic agent memory.

This module implements the temporal knowledge graph system described in the blueprint,
providing dynamic, temporally-aware knowledge management that goes beyond static RAG systems.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import json
import uuid


class TemporalNodeType(str, Enum):
    """Types of nodes in the temporal knowledge graph."""
    ENTITY = "entity"
    CONCEPT = "concept"
    EVENT = "event"
    FACT = "fact"
    RELATIONSHIP = "relationship"


class TemporalEdgeType(str, Enum):
    """Types of edges in the temporal knowledge graph."""
    RELATES_TO = "relates_to"
    CAUSED_BY = "caused_by"
    LEADS_TO = "leads_to"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"


class TemporalNode(BaseModel):
    """A node in the temporal knowledge graph."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: TemporalNodeType
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    # Temporal information
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Confidence and relevance
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Source tracking
    source: Optional[str] = None
    source_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if the node is valid at the given timestamp."""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True
    
    def update_relevance(self, access_time: datetime, decay_factor: float = 0.95):
        """Update relevance score based on access patterns."""
        time_diff = (datetime.now() - access_time).total_seconds() / 3600  # hours
        self.relevance_score = self.relevance_score * (decay_factor ** time_diff)
        self.updated_at = datetime.now()


class TemporalEdge(BaseModel):
    """An edge in the temporal knowledge graph."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    edge_type: TemporalEdgeType
    source_node_id: str
    target_node_id: str
    
    # Edge properties
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, ge=0.0)
    
    # Temporal information
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Confidence
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if the edge is valid at the given timestamp."""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True


class TemporalQuery(BaseModel):
    """Query object for temporal knowledge graph operations."""
    
    query_type: str  # "retrieve", "search", "traverse", "temporal"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Temporal constraints
    time_range: Optional[Tuple[datetime, datetime]] = None
    temporal_context: Optional[datetime] = None
    
    # Search parameters
    node_types: List[TemporalNodeType] = Field(default_factory=list)
    edge_types: List[TemporalEdgeType] = Field(default_factory=list)
    
    # Filtering
    min_confidence: float = 0.0
    min_relevance: float = 0.0
    max_results: int = 100


class TemporalKnowledgeGraph:
    """
    Temporal Knowledge Graph for dynamic agent memory.
    
    This implementation provides:
    - Temporal awareness with time-based validity
    - Dynamic relationship tracking
    - Confidence and relevance scoring
    - Efficient querying and traversal
    - Memory decay and consolidation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.nodes: Dict[str, TemporalNode] = {}
        self.edges: Dict[str, TemporalEdge] = {}
        
        # Indexing for efficient queries
        self.node_type_index: Dict[TemporalNodeType, Set[str]] = {}
        self.edge_type_index: Dict[TemporalEdgeType, Set[str]] = {}
        self.temporal_index: Dict[str, List[str]] = {}  # Date -> Node IDs
        
        # Graph statistics
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'last_updated': datetime.now(),
            'query_count': 0
        }
        
        self.logger = logging.getLogger("agentix.memory.temporal_graph")
        self.logger.info("Temporal Knowledge Graph initialized")
    
    def add_node(self, node: TemporalNode) -> str:
        """Add a node to the temporal knowledge graph."""
        self.nodes[node.id] = node
        
        # Update indexes
        if node.node_type not in self.node_type_index:
            self.node_type_index[node.node_type] = set()
        self.node_type_index[node.node_type].add(node.id)
        
        # Update temporal index
        date_key = node.created_at.strftime("%Y-%m-%d")
        if date_key not in self.temporal_index:
            self.temporal_index[date_key] = []
        self.temporal_index[date_key].append(node.id)
        
        # Update stats
        self.stats['total_nodes'] += 1
        self.stats['last_updated'] = datetime.now()
        
        self.logger.debug(f"Added node: {node.id} ({node.node_type})")
        return node.id
    
    def add_edge(self, edge: TemporalEdge) -> str:
        """Add an edge to the temporal knowledge graph."""
        # Validate that source and target nodes exist
        if edge.source_node_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_node_id} does not exist")
        if edge.target_node_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_node_id} does not exist")
        
        self.edges[edge.id] = edge
        
        # Update indexes
        if edge.edge_type not in self.edge_type_index:
            self.edge_type_index[edge.edge_type] = set()
        self.edge_type_index[edge.edge_type].add(edge.id)
        
        # Update stats
        self.stats['total_edges'] += 1
        self.stats['last_updated'] = datetime.now()
        
        self.logger.debug(f"Added edge: {edge.id} ({edge.edge_type})")
        return edge.id
    
    def get_node(self, node_id: str) -> Optional[TemporalNode]:
        """Retrieve a node by ID."""
        node = self.nodes.get(node_id)
        if node:
            # Update relevance based on access
            node.update_relevance(datetime.now())
        return node
    
    def get_edge(self, edge_id: str) -> Optional[TemporalEdge]:
        """Retrieve an edge by ID."""
        return self.edges.get(edge_id)
    
    def query(self, query: TemporalQuery) -> List[Dict[str, Any]]:
        """Execute a query against the temporal knowledge graph."""
        self.stats['query_count'] += 1
        
        if query.query_type == "retrieve":
            return self._query_retrieve(query)
        elif query.query_type == "search":
            return self._query_search(query)
        elif query.query_type == "traverse":
            return self._query_traverse(query)
        elif query.query_type == "temporal":
            return self._query_temporal(query)
        else:
            raise ValueError(f"Unknown query type: {query.query_type}")
    
    def _query_retrieve(self, query: TemporalQuery) -> List[Dict[str, Any]]:
        """Retrieve specific nodes or edges."""
        results = []
        
        # Retrieve by node IDs
        if 'node_ids' in query.parameters:
            for node_id in query.parameters['node_ids']:
                node = self.get_node(node_id)
                if node and self._passes_filters(node, query):
                    results.append({
                        'type': 'node',
                        'data': node.dict(),
                        'relevance': node.relevance_score
                    })
        
        # Retrieve by edge IDs
        if 'edge_ids' in query.parameters:
            for edge_id in query.parameters['edge_ids']:
                edge = self.get_edge(edge_id)
                if edge and self._passes_temporal_filter(edge, query):
                    results.append({
                        'type': 'edge',
                        'data': edge.dict(),
                        'relevance': edge.confidence
                    })
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True)[:query.max_results]
    
    def _query_search(self, query: TemporalQuery) -> List[Dict[str, Any]]:
        """Search for nodes and edges based on criteria."""
        results = []
        
        # Search nodes
        for node_type in query.node_types or list(self.node_type_index.keys()):
            if node_type in self.node_type_index:
                for node_id in self.node_type_index[node_type]:
                    node = self.nodes[node_id]
                    if self._passes_filters(node, query):
                        results.append({
                            'type': 'node',
                            'data': node.dict(),
                            'relevance': node.relevance_score
                        })
        
        # Search edges
        for edge_type in query.edge_types or list(self.edge_type_index.keys()):
            if edge_type in self.edge_type_index:
                for edge_id in self.edge_type_index[edge_type]:
                    edge = self.edges[edge_id]
                    if self._passes_temporal_filter(edge, query):
                        results.append({
                            'type': 'edge',
                            'data': edge.dict(),
                            'relevance': edge.confidence
                        })
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True)[:query.max_results]
    
    def _query_traverse(self, query: TemporalQuery) -> List[Dict[str, Any]]:
        """Traverse the graph from starting nodes."""
        start_nodes = query.parameters.get('start_nodes', [])
        max_depth = query.parameters.get('max_depth', 3)
        
        results = []
        visited = set()
        
        for start_node_id in start_nodes:
            if start_node_id in self.nodes:
                path_results = self._traverse_from_node(start_node_id, max_depth, visited, query)
                results.extend(path_results)
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True)[:query.max_results]
    
    def _query_temporal(self, query: TemporalQuery) -> List[Dict[str, Any]]:
        """Query based on temporal constraints."""
        results = []
        
        if query.time_range:
            start_time, end_time = query.time_range
            
            # Find nodes valid in time range
            for node in self.nodes.values():
                if (node.created_at >= start_time and node.created_at <= end_time) or \
                   (node.valid_from and node.valid_from <= end_time and 
                    (not node.valid_until or node.valid_until >= start_time)):
                    if self._passes_filters(node, query):
                        results.append({
                            'type': 'node',
                            'data': node.dict(),
                            'relevance': node.relevance_score
                        })
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True)[:query.max_results]
    
    def _passes_filters(self, node: TemporalNode, query: TemporalQuery) -> bool:
        """Check if a node passes the query filters."""
        if node.confidence < query.min_confidence:
            return False
        if node.relevance_score < query.min_relevance:
            return False
        if query.temporal_context and not node.is_valid_at(query.temporal_context):
            return False
        return True
    
    def _passes_temporal_filter(self, edge: TemporalEdge, query: TemporalQuery) -> bool:
        """Check if an edge passes the temporal filters."""
        if edge.confidence < query.min_confidence:
            return False
        if query.temporal_context and not edge.is_valid_at(query.temporal_context):
            return False
        return True
    
    def _traverse_from_node(self, node_id: str, max_depth: int, visited: set, query: TemporalQuery) -> List[Dict[str, Any]]:
        """Traverse the graph from a specific node."""
        if max_depth <= 0 or node_id in visited:
            return []
        
        visited.add(node_id)
        results = []
        
        # Add current node if it passes filters
        node = self.nodes.get(node_id)
        if node and self._passes_filters(node, query):
            results.append({
                'type': 'node',
                'data': node.dict(),
                'relevance': node.relevance_score,
                'depth': max_depth
            })
        
        # Traverse connected nodes
        for edge in self.edges.values():
            if edge.source_node_id == node_id and self._passes_temporal_filter(edge, query):
                # Add edge
                results.append({
                    'type': 'edge',
                    'data': edge.dict(),
                    'relevance': edge.confidence,
                    'depth': max_depth
                })
                
                # Recursively traverse target node
                target_results = self._traverse_from_node(edge.target_node_id, max_depth - 1, visited, query)
                results.extend(target_results)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return self.stats.copy()
    
    def cleanup_expired_nodes(self, current_time: Optional[datetime] = None):
        """Remove nodes and edges that are no longer valid."""
        if not current_time:
            current_time = datetime.now()
        
        expired_nodes = []
        expired_edges = []
        
        # Find expired nodes
        for node_id, node in self.nodes.items():
            if node.valid_until and current_time > node.valid_until:
                expired_nodes.append(node_id)
        
        # Find expired edges
        for edge_id, edge in self.edges.items():
            if edge.valid_until and current_time > edge.valid_until:
                expired_edges.append(edge_id)
        
        # Remove expired items
        for node_id in expired_nodes:
            self._remove_node(node_id)
        
        for edge_id in expired_edges:
            self._remove_edge(edge_id)
        
        self.logger.info(f"Cleaned up {len(expired_nodes)} expired nodes and {len(expired_edges)} expired edges")
    
    def _remove_node(self, node_id: str):
        """Remove a node and update indexes."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Remove from type index
            if node.node_type in self.node_type_index:
                self.node_type_index[node.node_type].discard(node_id)
            
            # Remove from temporal index
            date_key = node.created_at.strftime("%Y-%m-%d")
            if date_key in self.temporal_index:
                if node_id in self.temporal_index[date_key]:
                    self.temporal_index[date_key].remove(node_id)
            
            del self.nodes[node_id]
            self.stats['total_nodes'] -= 1
    
    def _remove_edge(self, edge_id: str):
        """Remove an edge and update indexes."""
        if edge_id in self.edges:
            edge = self.edges[edge_id]
            
            # Remove from type index
            if edge.edge_type in self.edge_type_index:
                self.edge_type_index[edge.edge_type].discard(edge_id)
            
            del self.edges[edge_id]
            self.stats['total_edges'] -= 1
