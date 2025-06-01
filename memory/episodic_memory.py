"""
Episodic Memory implementation for Agentix agents.

Episodic memory stores specific experiences and events with temporal context,
enabling agents to remember what happened, when it happened, and in what context.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import uuid


class Episode(BaseModel):
    """An episode in episodic memory."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Episode content
    event_type: str
    description: str
    context: Dict[str, Any] = Field(default_factory=dict)
    participants: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    
    # Emotional and importance markers
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)  # -1 negative, +1 positive
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Memory consolidation
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.now)
    consolidation_level: int = 0  # 0=new, 1=short-term, 2=long-term
    
    # Relationships to other episodes
    related_episodes: List[str] = Field(default_factory=list)
    causal_predecessors: List[str] = Field(default_factory=list)
    causal_successors: List[str] = Field(default_factory=list)
    
    def access(self):
        """Mark this episode as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_age_hours(self) -> float:
        """Get the age of this episode in hours."""
        return (datetime.now() - self.timestamp).total_seconds() / 3600


class EpisodicMemory:
    """
    Episodic Memory system for storing and retrieving specific experiences.
    
    This system provides:
    - Temporal organization of experiences
    - Context-aware retrieval
    - Memory consolidation over time
    - Relationship tracking between episodes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.episodes: Dict[str, Episode] = {}
        
        # Indexing for efficient retrieval
        self.temporal_index: Dict[str, List[str]] = {}  # Date -> Episode IDs
        self.type_index: Dict[str, List[str]] = {}      # Event type -> Episode IDs
        self.participant_index: Dict[str, List[str]] = {} # Participant -> Episode IDs
        
        # Configuration
        self.max_episodes = self.config.get('max_episodes', 10000)
        self.consolidation_threshold = self.config.get('consolidation_threshold', 24)  # hours
        
        self.logger = logging.getLogger("agentix.memory.episodic")
        self.logger.info("Episodic memory initialized")
    
    def store_episode(self, episode: Episode) -> str:
        """Store an episode in episodic memory."""
        self.episodes[episode.id] = episode
        
        # Update indexes
        self._update_indexes(episode)
        
        # Check if we need to prune old episodes
        if len(self.episodes) > self.max_episodes:
            self._prune_episodes()
        
        self.logger.debug(f"Stored episode: {episode.id} ({episode.event_type})")
        return episode.id
    
    def retrieve_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve a specific episode by ID."""
        episode = self.episodes.get(episode_id)
        if episode:
            episode.access()
        return episode
    
    def search_episodes(self, 
                       event_type: Optional[str] = None,
                       participants: Optional[List[str]] = None,
                       time_range: Optional[tuple] = None,
                       min_importance: float = 0.0,
                       max_results: int = 100) -> List[Episode]:
        """Search for episodes based on criteria."""
        
        candidate_ids = set(self.episodes.keys())
        
        # Filter by event type
        if event_type and event_type in self.type_index:
            candidate_ids &= set(self.type_index[event_type])
        
        # Filter by participants
        if participants:
            for participant in participants:
                if participant in self.participant_index:
                    candidate_ids &= set(self.participant_index[participant])
        
        # Filter by time range
        if time_range:
            start_time, end_time = time_range
            time_filtered = []
            for episode_id in candidate_ids:
                episode = self.episodes[episode_id]
                if start_time <= episode.timestamp <= end_time:
                    time_filtered.append(episode_id)
            candidate_ids = set(time_filtered)
        
        # Get episodes and filter by importance
        results = []
        for episode_id in candidate_ids:
            episode = self.episodes[episode_id]
            if episode.importance >= min_importance:
                episode.access()
                results.append(episode)
        
        # Sort by importance and recency
        results.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
        
        return results[:max_results]
    
    def get_recent_episodes(self, hours: int = 24, max_results: int = 50) -> List[Episode]:
        """Get episodes from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_episodes = []
        for episode in self.episodes.values():
            if episode.timestamp >= cutoff_time:
                episode.access()
                recent_episodes.append(episode)
        
        # Sort by recency
        recent_episodes.sort(key=lambda e: e.timestamp, reverse=True)
        
        return recent_episodes[:max_results]
    
    def get_related_episodes(self, episode_id: str, max_results: int = 20) -> List[Episode]:
        """Get episodes related to a specific episode."""
        episode = self.episodes.get(episode_id)
        if not episode:
            return []
        
        related_ids = set()
        related_ids.update(episode.related_episodes)
        related_ids.update(episode.causal_predecessors)
        related_ids.update(episode.causal_successors)
        
        # Also find episodes with similar participants or context
        for other_episode in self.episodes.values():
            if other_episode.id == episode_id:
                continue
            
            # Check for common participants
            common_participants = set(episode.participants) & set(other_episode.participants)
            if common_participants:
                related_ids.add(other_episode.id)
            
            # Check for temporal proximity (within 1 hour)
            time_diff = abs((episode.timestamp - other_episode.timestamp).total_seconds())
            if time_diff < 3600:  # 1 hour
                related_ids.add(other_episode.id)
        
        # Get the actual episodes
        related_episodes = []
        for related_id in related_ids:
            if related_id in self.episodes:
                related_episode = self.episodes[related_id]
                related_episode.access()
                related_episodes.append(related_episode)
        
        # Sort by relevance (importance and recency)
        related_episodes.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
        
        return related_episodes[:max_results]
    
    def consolidate_memory(self):
        """Consolidate episodic memory by updating consolidation levels."""
        current_time = datetime.now()
        
        for episode in self.episodes.values():
            age_hours = episode.get_age_hours()
            
            # Update consolidation level based on age and access patterns
            if age_hours > self.consolidation_threshold and episode.consolidation_level == 0:
                # Move to short-term if accessed enough or important
                if episode.access_count > 1 or episode.importance > 0.7:
                    episode.consolidation_level = 1
                    self.logger.debug(f"Episode {episode.id} consolidated to short-term")
            
            elif age_hours > self.consolidation_threshold * 7 and episode.consolidation_level == 1:
                # Move to long-term if frequently accessed or very important
                if episode.access_count > 5 or episode.importance > 0.8:
                    episode.consolidation_level = 2
                    self.logger.debug(f"Episode {episode.id} consolidated to long-term")
    
    def _update_indexes(self, episode: Episode):
        """Update indexes when storing an episode."""
        
        # Temporal index
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        if date_key not in self.temporal_index:
            self.temporal_index[date_key] = []
        self.temporal_index[date_key].append(episode.id)
        
        # Type index
        if episode.event_type not in self.type_index:
            self.type_index[episode.event_type] = []
        self.type_index[episode.event_type].append(episode.id)
        
        # Participant index
        for participant in episode.participants:
            if participant not in self.participant_index:
                self.participant_index[participant] = []
            self.participant_index[participant].append(episode.id)
    
    def _prune_episodes(self):
        """Remove old, unimportant episodes to stay within memory limits."""
        # Sort episodes by importance and consolidation level
        episodes_to_consider = []
        
        for episode in self.episodes.values():
            # Never prune long-term consolidated memories
            if episode.consolidation_level < 2:
                score = episode.importance * (1 + episode.access_count * 0.1)
                episodes_to_consider.append((score, episode))
        
        # Sort by score (lowest first for removal)
        episodes_to_consider.sort(key=lambda x: x[0])
        
        # Remove the lowest scoring episodes
        num_to_remove = len(self.episodes) - self.max_episodes + 100  # Remove extra for buffer
        
        for i in range(min(num_to_remove, len(episodes_to_consider))):
            _, episode_to_remove = episodes_to_consider[i]
            self._remove_episode(episode_to_remove.id)
    
    def _remove_episode(self, episode_id: str):
        """Remove an episode and update indexes."""
        if episode_id not in self.episodes:
            return
        
        episode = self.episodes[episode_id]
        
        # Remove from temporal index
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        if date_key in self.temporal_index:
            if episode_id in self.temporal_index[date_key]:
                self.temporal_index[date_key].remove(episode_id)
        
        # Remove from type index
        if episode.event_type in self.type_index:
            if episode_id in self.type_index[episode.event_type]:
                self.type_index[episode.event_type].remove(episode_id)
        
        # Remove from participant index
        for participant in episode.participants:
            if participant in self.participant_index:
                if episode_id in self.participant_index[participant]:
                    self.participant_index[participant].remove(episode_id)
        
        # Remove the episode itself
        del self.episodes[episode_id]
        
        self.logger.debug(f"Removed episode: {episode_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        consolidation_counts = {0: 0, 1: 0, 2: 0}
        total_access_count = 0
        
        for episode in self.episodes.values():
            consolidation_counts[episode.consolidation_level] += 1
            total_access_count += episode.access_count
        
        return {
            'total_episodes': len(self.episodes),
            'consolidation_levels': consolidation_counts,
            'average_access_count': total_access_count / len(self.episodes) if self.episodes else 0,
            'unique_event_types': len(self.type_index),
            'unique_participants': len(self.participant_index)
        }
