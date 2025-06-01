"""
Semantic Memory implementation for Agentix agents.

Semantic memory stores general knowledge, concepts, and facts that are not tied
to specific experiences, enabling agents to maintain and access conceptual understanding.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class Concept(BaseModel):
    """A concept in semantic memory."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str
    
    # Concept properties
    properties: Dict[str, Any] = Field(default_factory=dict)
    attributes: List[str] = Field(default_factory=list)
    
    # Relationships
    is_a: List[str] = Field(default_factory=list)  # Parent concepts
    has_a: List[str] = Field(default_factory=list)  # Component concepts
    related_to: List[str] = Field(default_factory=list)  # Related concepts
    
    # Confidence and usage
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    usage_count: int = 0
    last_used: datetime = Field(default_factory=datetime.now)
    
    # Source information
    sources: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def use(self):
        """Mark this concept as used."""
        self.usage_count += 1
        self.last_used = datetime.now()


class Fact(BaseModel):
    """A fact in semantic memory."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str  # Concept ID or name
    predicate: str  # Relationship type
    object: str   # Concept ID, name, or value
    
    # Fact metadata
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    temporal_validity: Optional[tuple] = None  # (start_time, end_time)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Source and usage
    sources: List[str] = Field(default_factory=list)
    usage_count: int = 0
    last_used: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def use(self):
        """Mark this fact as used."""
        self.usage_count += 1
        self.last_used = datetime.now()
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if the fact is valid at the given timestamp."""
        if not self.temporal_validity:
            return True
        
        start_time, end_time = self.temporal_validity
        if start_time and timestamp < start_time:
            return False
        if end_time and timestamp > end_time:
            return False
        
        return True


class SemanticMemory:
    """
    Semantic Memory system for storing and retrieving general knowledge.
    
    This system provides:
    - Concept hierarchy management
    - Fact storage and retrieval
    - Knowledge inference
    - Concept similarity computation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Storage
        self.concepts: Dict[str, Concept] = {}
        self.facts: Dict[str, Fact] = {}
        
        # Indexing
        self.concept_name_index: Dict[str, str] = {}  # Name -> Concept ID
        self.category_index: Dict[str, List[str]] = {}  # Category -> Concept IDs
        self.predicate_index: Dict[str, List[str]] = {}  # Predicate -> Fact IDs
        self.subject_index: Dict[str, List[str]] = {}   # Subject -> Fact IDs
        
        # Configuration
        self.max_concepts = self.config.get('max_concepts', 50000)
        self.max_facts = self.config.get('max_facts', 100000)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        
        self.logger = logging.getLogger("agentix.memory.semantic")
        self.logger.info("Semantic memory initialized")
    
    def add_concept(self, concept: Concept) -> str:
        """Add a concept to semantic memory."""
        self.concepts[concept.id] = concept
        
        # Update indexes
        self.concept_name_index[concept.name.lower()] = concept.id
        
        if concept.category not in self.category_index:
            self.category_index[concept.category] = []
        self.category_index[concept.category].append(concept.id)
        
        self.logger.debug(f"Added concept: {concept.name} ({concept.category})")
        return concept.id
    
    def add_fact(self, fact: Fact) -> str:
        """Add a fact to semantic memory."""
        self.facts[fact.id] = fact
        
        # Update indexes
        if fact.predicate not in self.predicate_index:
            self.predicate_index[fact.predicate] = []
        self.predicate_index[fact.predicate].append(fact.id)
        
        if fact.subject not in self.subject_index:
            self.subject_index[fact.subject] = []
        self.subject_index[fact.subject].append(fact.id)
        
        self.logger.debug(f"Added fact: {fact.subject} {fact.predicate} {fact.object}")
        return fact.id
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID."""
        concept = self.concepts.get(concept_id)
        if concept:
            concept.use()
        return concept
    
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get a concept by name."""
        concept_id = self.concept_name_index.get(name.lower())
        if concept_id:
            return self.get_concept(concept_id)
        return None
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID."""
        fact = self.facts.get(fact_id)
        if fact:
            fact.use()
        return fact
    
    def search_concepts(self, 
                       name_pattern: Optional[str] = None,
                       category: Optional[str] = None,
                       attributes: Optional[List[str]] = None,
                       max_results: int = 100) -> List[Concept]:
        """Search for concepts based on criteria."""
        
        candidate_ids = set(self.concepts.keys())
        
        # Filter by category
        if category and category in self.category_index:
            candidate_ids &= set(self.category_index[category])
        
        # Filter by name pattern
        if name_pattern:
            name_filtered = []
            pattern_lower = name_pattern.lower()
            for concept_id in candidate_ids:
                concept = self.concepts[concept_id]
                if pattern_lower in concept.name.lower():
                    name_filtered.append(concept_id)
            candidate_ids = set(name_filtered)
        
        # Filter by attributes
        if attributes:
            attr_filtered = []
            for concept_id in candidate_ids:
                concept = self.concepts[concept_id]
                if any(attr in concept.attributes for attr in attributes):
                    attr_filtered.append(concept_id)
            candidate_ids = set(attr_filtered)
        
        # Get concepts and sort by usage
        results = []
        for concept_id in candidate_ids:
            concept = self.concepts[concept_id]
            concept.use()
            results.append(concept)
        
        results.sort(key=lambda c: c.usage_count, reverse=True)
        return results[:max_results]
    
    def search_facts(self,
                    subject: Optional[str] = None,
                    predicate: Optional[str] = None,
                    object_value: Optional[str] = None,
                    timestamp: Optional[datetime] = None,
                    max_results: int = 100) -> List[Fact]:
        """Search for facts based on criteria."""
        
        candidate_ids = set(self.facts.keys())
        
        # Filter by subject
        if subject and subject in self.subject_index:
            candidate_ids &= set(self.subject_index[subject])
        
        # Filter by predicate
        if predicate and predicate in self.predicate_index:
            candidate_ids &= set(self.predicate_index[predicate])
        
        # Filter by object and temporal validity
        filtered_facts = []
        for fact_id in candidate_ids:
            fact = self.facts[fact_id]
            
            # Check object match
            if object_value and object_value not in fact.object:
                continue
            
            # Check temporal validity
            if timestamp and not fact.is_valid_at(timestamp):
                continue
            
            fact.use()
            filtered_facts.append(fact)
        
        # Sort by confidence and usage
        filtered_facts.sort(key=lambda f: (f.confidence, f.usage_count), reverse=True)
        return filtered_facts[:max_results]
    
    def get_concept_hierarchy(self, concept_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """Get the hierarchy for a concept (parents and children)."""
        concept = self.get_concept(concept_id)
        if not concept:
            return {}
        
        hierarchy = {
            'concept': concept.dict(),
            'parents': [],
            'children': [],
            'related': []
        }
        
        # Get parents (is_a relationships)
        for parent_id in concept.is_a:
            if parent_id in self.concepts:
                parent_concept = self.get_concept(parent_id)
                hierarchy['parents'].append(parent_concept.dict())
        
        # Get children (concepts that have is_a relationship to this concept)
        for other_concept in self.concepts.values():
            if concept_id in other_concept.is_a:
                hierarchy['children'].append(other_concept.dict())
        
        # Get related concepts
        for related_id in concept.related_to:
            if related_id in self.concepts:
                related_concept = self.get_concept(related_id)
                hierarchy['related'].append(related_concept.dict())
        
        return hierarchy
    
    def infer_facts(self, concept_id: str) -> List[Fact]:
        """Infer new facts based on existing knowledge."""
        inferred_facts = []
        concept = self.get_concept(concept_id)
        
        if not concept:
            return inferred_facts
        
        # Inherit properties from parent concepts
        for parent_id in concept.is_a:
            parent_facts = self.search_facts(subject=parent_id)
            for parent_fact in parent_facts:
                # Create inferred fact
                inferred_fact = Fact(
                    subject=concept_id,
                    predicate=parent_fact.predicate,
                    object=parent_fact.object,
                    confidence=parent_fact.confidence * 0.8,  # Reduce confidence for inference
                    sources=[f"inferred_from_{parent_fact.id}"]
                )
                inferred_facts.append(inferred_fact)
        
        return inferred_facts
    
    def compute_concept_similarity(self, concept1_id: str, concept2_id: str) -> float:
        """Compute similarity between two concepts."""
        concept1 = self.get_concept(concept1_id)
        concept2 = self.get_concept(concept2_id)
        
        if not concept1 or not concept2:
            return 0.0
        
        similarity_score = 0.0
        
        # Category similarity
        if concept1.category == concept2.category:
            similarity_score += 0.3
        
        # Attribute overlap
        common_attributes = set(concept1.attributes) & set(concept2.attributes)
        total_attributes = set(concept1.attributes) | set(concept2.attributes)
        if total_attributes:
            attribute_similarity = len(common_attributes) / len(total_attributes)
            similarity_score += attribute_similarity * 0.4
        
        # Relationship overlap
        common_parents = set(concept1.is_a) & set(concept2.is_a)
        total_parents = set(concept1.is_a) | set(concept2.is_a)
        if total_parents:
            parent_similarity = len(common_parents) / len(total_parents)
            similarity_score += parent_similarity * 0.3
        
        return min(similarity_score, 1.0)
    
    def find_similar_concepts(self, concept_id: str, max_results: int = 10) -> List[tuple]:
        """Find concepts similar to the given concept."""
        similarities = []
        
        for other_id, other_concept in self.concepts.items():
            if other_id != concept_id:
                similarity = self.compute_concept_similarity(concept_id, other_id)
                if similarity >= self.similarity_threshold:
                    similarities.append((similarity, other_concept))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:max_results]
    
    def consolidate_knowledge(self):
        """Consolidate semantic knowledge by merging similar concepts and facts."""
        # Find and merge very similar concepts
        concepts_to_merge = []
        
        for concept_id, concept in self.concepts.items():
            similar_concepts = self.find_similar_concepts(concept_id, max_results=5)
            for similarity, similar_concept in similar_concepts:
                if similarity > 0.9:  # Very high similarity
                    concepts_to_merge.append((concept_id, similar_concept.id, similarity))
        
        # TODO: Implement concept merging logic
        # This would involve combining attributes, relationships, and updating references
        
        self.logger.info(f"Found {len(concepts_to_merge)} concept pairs for potential merging")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        total_usage = sum(concept.usage_count for concept in self.concepts.values())
        total_fact_usage = sum(fact.usage_count for fact in self.facts.values())
        
        return {
            'total_concepts': len(self.concepts),
            'total_facts': len(self.facts),
            'unique_categories': len(self.category_index),
            'unique_predicates': len(self.predicate_index),
            'average_concept_usage': total_usage / len(self.concepts) if self.concepts else 0,
            'average_fact_usage': total_fact_usage / len(self.facts) if self.facts else 0
        }
