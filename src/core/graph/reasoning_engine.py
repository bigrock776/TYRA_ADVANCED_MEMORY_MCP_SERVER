"""
Multi-Hop Graph Reasoning Engine for Advanced Knowledge Discovery.

This module provides comprehensive multi-hop reasoning capabilities using local networkx algorithms,
reasoning path scoring with local metrics, explanation generation using template-based NLG,
and reasoning depth controls with configurable limits. All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import json
import math
import heapq
import hashlib

# Graph and ML imports - all local
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ReasoningType(str, Enum):
    """Types of reasoning operations."""
    INDUCTIVE = "inductive"              # Inductive reasoning (specific to general)
    DEDUCTIVE = "deductive"              # Deductive reasoning (general to specific)
    ABDUCTIVE = "abductive"              # Abductive reasoning (best explanation)
    ANALOGICAL = "analogical"            # Analogical reasoning (similarity-based)
    CAUSAL = "causal"                    # Causal reasoning (cause-effect)
    TRANSITIVE = "transitive"            # Transitive reasoning (A->B, B->C => A->C)
    TEMPORAL = "temporal"                # Temporal reasoning (time-based)
    SPATIAL = "spatial"                  # Spatial reasoning (location-based)


class ReasoningStrategy(str, Enum):
    """Strategies for graph traversal and reasoning."""
    BREADTH_FIRST = "breadth_first"      # BFS traversal
    DEPTH_FIRST = "depth_first"          # DFS traversal
    BEST_FIRST = "best_first"            # Best-first search
    A_STAR = "a_star"                    # A* search algorithm
    DIJKSTRA = "dijkstra"                # Dijkstra's algorithm
    RANDOM_WALK = "random_walk"          # Random walk exploration
    PAGERANK = "pagerank"                # PageRank-based importance
    COMMUNITY_BASED = "community_based"  # Community detection based


class ConfidenceLevel(str, Enum):
    """Confidence levels for reasoning results."""
    VERY_HIGH = "very_high"      # 90%+ confidence
    HIGH = "high"                # 75-90% confidence
    MEDIUM = "medium"            # 50-75% confidence
    LOW = "low"                  # 25-50% confidence
    VERY_LOW = "very_low"        # <25% confidence


@dataclass
class ReasoningNode:
    """Represents a node in the reasoning graph."""
    node_id: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    importance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_similarity(self, other: 'ReasoningNode') -> float:
        """Calculate similarity with another node."""
        if self.embedding is not None and other.embedding is not None:
            return float(cosine_similarity(
                self.embedding.reshape(1, -1),
                other.embedding.reshape(1, -1)
            )[0, 0])
        
        # Fallback to property-based similarity
        common_props = set(self.properties.keys()).intersection(set(other.properties.keys()))
        if not common_props:
            return 0.0
        
        similarity_scores = []
        for prop in common_props:
            val1, val2 = self.properties[prop], other.properties[prop]
            if isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simplified Jaccard)
                words1, words2 = set(val1.lower().split()), set(val2.lower().split())
                union_size = len(words1.union(words2))
                if union_size > 0:
                    sim = len(words1.intersection(words2)) / union_size
                    similarity_scores.append(sim)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    sim = 1.0 - abs(val1 - val2) / max_val
                    similarity_scores.append(sim)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0


@dataclass
class ReasoningEdge:
    """Represents an edge in the reasoning graph."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    evidence: List[str] = field(default_factory=list)
    
    def get_weight(self) -> float:
        """Calculate edge weight for graph algorithms."""
        return self.strength * self.confidence


@dataclass
class ReasoningPath:
    """Represents a reasoning path through the graph."""
    path_id: str
    nodes: List[ReasoningNode]
    edges: List[ReasoningEdge]
    reasoning_type: ReasoningType
    confidence: float
    path_score: float
    explanation: str = ""
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_length(self) -> int:
        """Get path length."""
        return len(self.edges)
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get human-readable confidence level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass
class ReasoningQuery:
    """Represents a reasoning query."""
    query_id: str
    source_entities: List[str]
    target_entities: List[str]
    reasoning_types: List[ReasoningType]
    max_hops: int = 5
    min_confidence: float = 0.3
    strategy: ReasoningStrategy = ReasoningStrategy.BEST_FIRST
    constraints: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    query_id: str
    paths: List[ReasoningPath]
    total_paths_found: int
    processing_time: float
    confidence_distribution: Dict[ConfidenceLevel, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_best_path(self) -> Optional[ReasoningPath]:
        """Get the highest scoring path."""
        if not self.paths:
            return None
        return max(self.paths, key=lambda p: p.path_score)
    
    def get_paths_by_confidence(self, min_confidence: float) -> List[ReasoningPath]:
        """Get paths above confidence threshold."""
        return [path for path in self.paths if path.confidence >= min_confidence]


class PathScorer:
    """
    Advanced path scoring system for reasoning paths.
    
    Features:
    - Multiple scoring criteria (confidence, relevance, novelty, coherence)
    - Weighted scoring with adaptive weights
    - Context-aware scoring adjustments
    - Path diversity promotion
    """
    
    def __init__(
        self,
        scoring_weights: Optional[Dict[str, float]] = None,
        diversity_penalty: float = 0.1
    ):
        self.scoring_weights = scoring_weights or {
            'confidence': 0.3,
            'relevance': 0.25,
            'novelty': 0.2,
            'coherence': 0.15,
            'efficiency': 0.1
        }
        self.diversity_penalty = diversity_penalty
        
        # Path history for diversity calculation
        self.scored_paths: List[ReasoningPath] = []
    
    async def score_path(
        self,
        path: ReasoningPath,
        query: ReasoningQuery,
        graph: nx.Graph
    ) -> float:
        """Score a reasoning path based on multiple criteria."""
        
        scores = {}
        
        # Confidence score (weighted average of edge confidences)
        if path.edges:
            edge_confidences = [edge.confidence for edge in path.edges]
            scores['confidence'] = np.mean(edge_confidences)
        else:
            scores['confidence'] = 0.0
        
        # Relevance score (how well path connects source to target)
        scores['relevance'] = await self._calculate_relevance_score(path, query)
        
        # Novelty score (how unique this path is)
        scores['novelty'] = await self._calculate_novelty_score(path)
        
        # Coherence score (how well the path makes sense)
        scores['coherence'] = await self._calculate_coherence_score(path, graph)
        
        # Efficiency score (shorter paths preferred)
        scores['efficiency'] = await self._calculate_efficiency_score(path, query)
        
        # Calculate weighted score
        total_score = 0.0
        for criterion, weight in self.scoring_weights.items():
            if criterion in scores:
                total_score += weight * scores[criterion]
        
        # Apply diversity penalty if path is too similar to existing ones
        diversity_penalty = self._calculate_diversity_penalty(path)
        total_score *= (1.0 - diversity_penalty * self.diversity_penalty)
        
        # Store path for future diversity calculations
        self.scored_paths.append(path)
        if len(self.scored_paths) > 100:
            self.scored_paths.pop(0)  # Keep only recent paths
        
        return max(0.0, min(1.0, total_score))
    
    async def _calculate_relevance_score(
        self,
        path: ReasoningPath,
        query: ReasoningQuery
    ) -> float:
        """Calculate how relevant the path is to the query."""
        
        # Check if path connects source and target entities
        path_entities = {node.node_id for node in path.nodes}
        source_overlap = len(set(query.source_entities).intersection(path_entities))
        target_overlap = len(set(query.target_entities).intersection(path_entities))
        
        # Calculate overlap ratios
        source_ratio = source_overlap / len(query.source_entities) if query.source_entities else 0.0
        target_ratio = target_overlap / len(query.target_entities) if query.target_entities else 0.0
        
        # Relevance is geometric mean of source and target overlap
        relevance = math.sqrt(source_ratio * target_ratio) if source_ratio * target_ratio > 0 else 0.0
        
        return relevance
    
    async def _calculate_novelty_score(self, path: ReasoningPath) -> float:
        """Calculate how novel/unique this path is."""
        
        if not self.scored_paths:
            return 1.0  # First path is completely novel
        
        # Calculate similarity with existing paths
        similarities = []
        path_entities = {node.node_id for node in path.nodes}
        
        for existing_path in self.scored_paths[-20:]:  # Compare with recent paths
            existing_entities = {node.node_id for node in existing_path.nodes}
            
            # Jaccard similarity of entity sets
            if path_entities or existing_entities:
                intersection = len(path_entities.intersection(existing_entities))
                union = len(path_entities.union(existing_entities))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        return novelty
    
    async def _calculate_coherence_score(
        self,
        path: ReasoningPath,
        graph: nx.Graph
    ) -> float:
        """Calculate how coherent/logical the path is."""
        
        if len(path.edges) < 2:
            return 1.0  # Single edge is coherent
        
        coherence_scores = []
        
        # Check relationship type consistency
        relationship_types = [edge.relationship_type for edge in path.edges]
        type_transitions = []
        
        for i in range(len(relationship_types) - 1):
            current_type = relationship_types[i]
            next_type = relationship_types[i + 1]
            
            # Score based on how well relationship types flow together
            transition_score = self._score_relationship_transition(current_type, next_type)
            type_transitions.append(transition_score)
        
        if type_transitions:
            coherence_scores.append(np.mean(type_transitions))
        
        # Check entity type consistency
        entity_types = [node.entity_type for node in path.nodes]
        type_diversity = len(set(entity_types)) / len(entity_types) if entity_types else 0.0
        
        # Moderate diversity is preferred (not too homogeneous, not too chaotic)
        optimal_diversity = 0.5
        diversity_score = 1.0 - abs(type_diversity - optimal_diversity)
        coherence_scores.append(diversity_score)
        
        # Calculate temporal coherence if timestamps are available
        if all(hasattr(edge, 'created_at') and edge.created_at for edge in path.edges):
            temporal_score = self._calculate_temporal_coherence(path)
            coherence_scores.append(temporal_score)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _score_relationship_transition(self, type1: str, type2: str) -> float:
        """Score how well two relationship types transition."""
        
        # Define compatibility matrix for relationship types
        compatibility_rules = {
            ('causes', 'causes'): 0.9,      # Causal chains are coherent
            ('causes', 'influences'): 0.8,
            ('influences', 'causes'): 0.7,
            ('contains', 'contains'): 0.9,  # Hierarchical chains
            ('part_of', 'part_of'): 0.9,
            ('similar_to', 'similar_to'): 0.8,
            ('related_to', 'related_to'): 0.7,
            ('temporal_before', 'temporal_before'): 0.9,
            ('temporal_after', 'temporal_after'): 0.9,
        }
        
        # Check direct compatibility
        if (type1, type2) in compatibility_rules:
            return compatibility_rules[(type1, type2)]
        
        # Check reverse compatibility
        if (type2, type1) in compatibility_rules:
            return compatibility_rules[(type2, type1)] * 0.8  # Slightly lower for reverse
        
        # Default compatibility for unknown combinations
        if type1 == type2:
            return 0.6  # Same type has moderate compatibility
        else:
            return 0.4  # Different types have low compatibility
    
    def _calculate_temporal_coherence(self, path: ReasoningPath) -> float:
        """Calculate temporal coherence of the path."""
        
        timestamps = []
        for edge in path.edges:
            if hasattr(edge, 'created_at') and edge.created_at:
                timestamps.append(edge.created_at.timestamp())
        
        if len(timestamps) < 2:
            return 1.0
        
        # Check if timestamps are roughly in order (allowing some variance)
        temporal_scores = []
        for i in range(len(timestamps) - 1):
            time_diff = timestamps[i + 1] - timestamps[i]
            
            # Prefer forward temporal flow, but allow some backward flow
            if time_diff >= 0:
                temporal_scores.append(1.0)
            else:
                # Penalize backward flow based on magnitude
                penalty = min(1.0, abs(time_diff) / (24 * 3600))  # Normalize by day
                temporal_scores.append(max(0.0, 1.0 - penalty))
        
        return np.mean(temporal_scores) if temporal_scores else 1.0
    
    async def _calculate_efficiency_score(
        self,
        path: ReasoningPath,
        query: ReasoningQuery
    ) -> float:
        """Calculate efficiency score (shorter paths preferred)."""
        
        path_length = path.get_length()
        max_hops = query.max_hops
        
        if path_length == 0:
            return 1.0
        
        # Efficiency decreases exponentially with path length
        efficiency = math.exp(-path_length / max_hops) if max_hops > 0 else 0.0
        
        return efficiency
    
    def _calculate_diversity_penalty(self, path: ReasoningPath) -> float:
        """Calculate penalty for paths too similar to existing ones."""
        
        if not self.scored_paths:
            return 0.0
        
        # Calculate similarity with recent paths
        path_entities = {node.node_id for node in path.nodes}
        similarities = []
        
        for existing_path in self.scored_paths[-10:]:  # Check last 10 paths
            existing_entities = {node.node_id for node in existing_path.nodes}
            
            if path_entities or existing_entities:
                intersection = len(path_entities.intersection(existing_entities))
                union = len(path_entities.union(existing_entities))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        # Penalty is based on maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        
        # Apply penalty if similarity is high
        penalty = max(0.0, max_similarity - 0.7) / 0.3  # Penalty kicks in above 70% similarity
        
        return min(1.0, penalty)


class ExplanationGenerator:
    """
    Template-based natural language explanation generator.
    
    Features:
    - Template-based explanation generation
    - Context-aware template selection
    - Evidence integration
    - Confidence-based language adjustment
    """
    
    def __init__(self):
        # Explanation templates for different reasoning types
        self.templates = {
            ReasoningType.CAUSAL: [
                "{source} causes {target} because {evidence}.",
                "The causal relationship from {source} to {target} is supported by {evidence}.",
                "Evidence suggests that {source} leads to {target} through {intermediate_steps}."
            ],
            
            ReasoningType.TRANSITIVE: [
                "Since {source} relates to {intermediate} and {intermediate} relates to {target}, we can infer that {source} relates to {target}.",
                "Through transitive reasoning: {source} → {intermediate} → {target}.",
                "The connection from {source} to {target} is established via {intermediate}."
            ],
            
            ReasoningType.ANALOGICAL: [
                "{source} is similar to {target} based on {shared_properties}.",
                "By analogy, {source} and {target} share characteristics: {evidence}.",
                "The similarity between {source} and {target} suggests {conclusion}."
            ],
            
            ReasoningType.INDUCTIVE: [
                "Based on patterns observed in {evidence}, we can generalize that {conclusion}.",
                "The evidence {evidence} supports the general principle that {conclusion}.",
                "Inductive reasoning from {instances} leads to {conclusion}."
            ],
            
            ReasoningType.DEDUCTIVE: [
                "Given that {premises}, it follows that {conclusion}.",
                "Deductive reasoning from {general_rule} applied to {specific_case} yields {conclusion}.",
                "Logically, if {premise} then {conclusion}."
            ]
        }
        
        # Confidence-based language modifiers
        self.confidence_modifiers = {
            ConfidenceLevel.VERY_HIGH: ["clearly", "definitely", "certainly", "undoubtedly"],
            ConfidenceLevel.HIGH: ["likely", "probably", "strongly suggests", "indicates"],
            ConfidenceLevel.MEDIUM: ["suggests", "may indicate", "could mean", "appears to"],
            ConfidenceLevel.LOW: ["might suggest", "possibly indicates", "could potentially", "weakly suggests"],
            ConfidenceLevel.VERY_LOW: ["speculatively", "tentatively", "with low confidence", "uncertainly"]
        }
    
    async def generate_explanation(
        self,
        path: ReasoningPath,
        query: ReasoningQuery,
        include_evidence: bool = True
    ) -> str:
        """Generate natural language explanation for a reasoning path."""
        
        try:
            # Select appropriate template
            reasoning_type = path.reasoning_type
            templates = self.templates.get(reasoning_type, self.templates[ReasoningType.TRANSITIVE])
            template = templates[0]  # Use first template for now
            
            # Extract key entities
            source_entity = path.nodes[0].node_id if path.nodes else "unknown"
            target_entity = path.nodes[-1].node_id if path.nodes else "unknown"
            
            # Build intermediate steps
            intermediate_steps = []
            if len(path.nodes) > 2:
                for i in range(1, len(path.nodes) - 1):
                    intermediate_steps.append(path.nodes[i].node_id)
            
            intermediate_text = " → ".join(intermediate_steps) if intermediate_steps else "direct connection"
            
            # Gather evidence
            evidence_text = self._format_evidence(path.evidence) if include_evidence and path.evidence else "available data"
            
            # Get confidence modifier
            confidence_level = path.get_confidence_level()
            modifiers = self.confidence_modifiers.get(confidence_level, [""])
            modifier = modifiers[0] if modifiers else ""
            
            # Fill template
            explanation = template.format(
                source=source_entity,
                target=target_entity,
                intermediate=intermediate_steps[0] if intermediate_steps else "intermediate entity",
                intermediate_steps=intermediate_text,
                evidence=evidence_text,
                shared_properties="common characteristics",
                conclusion=f"there is a relationship between {source_entity} and {target_entity}",
                premises="the given facts",
                general_rule="the general principle",
                specific_case="this specific instance",
                premise="the premise holds",
                instances="the observed instances"
            )
            
            # Add confidence modifier
            if modifier:
                explanation = f"The evidence {modifier} that " + explanation.lower()
            
            # Add path details
            path_details = f" This reasoning path involves {len(path.edges)} connections with {confidence_level.value} confidence."
            explanation += path_details
            
            return explanation
            
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")
            # Fallback explanation
            return f"A reasoning path was found connecting the entities with {path.get_confidence_level().value} confidence."
    
    def _format_evidence(self, evidence: List[str]) -> str:
        """Format evidence list into readable text."""
        
        if not evidence:
            return "available data"
        
        if len(evidence) == 1:
            return evidence[0]
        elif len(evidence) == 2:
            return f"{evidence[0]} and {evidence[1]}"
        else:
            return f"{', '.join(evidence[:-1])}, and {evidence[-1]}"


class MultiHopReasoningEngine:
    """
    Advanced multi-hop reasoning engine for knowledge graph exploration.
    
    Features:
    - Multiple graph traversal algorithms (BFS, DFS, A*, Dijkstra)
    - Configurable reasoning depth controls
    - Path scoring with multiple criteria
    - Template-based explanation generation
    - Performance optimization with caching
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        max_concurrent_queries: int = 10,
        default_max_hops: int = 5
    ):
        self.redis_cache = redis_cache
        self.max_concurrent_queries = max_concurrent_queries
        self.default_max_hops = default_max_hops
        
        # Core components
        self.path_scorer = PathScorer()
        self.explanation_generator = ExplanationGenerator()
        
        # Graph state
        self.graph = nx.MultiDiGraph()  # Support multiple edges between nodes
        self.nodes: Dict[str, ReasoningNode] = {}
        self.edges: List[ReasoningEdge] = []
        
        # Query processing
        self.active_queries: Dict[str, ReasoningQuery] = {}
        self.query_semaphore = asyncio.Semaphore(max_concurrent_queries)
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_processing_time': 0.0,
            'total_paths_found': 0,
            'reasoning_type_usage': defaultdict(int)
        }
        
        logger.info(f"MultiHopReasoningEngine initialized with max {default_max_hops} hops")
    
    async def add_node(self, node: ReasoningNode):
        """Add a node to the reasoning graph."""
        
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            entity_type=node.entity_type,
            properties=node.properties,
            importance_score=node.importance_score,
            created_at=node.created_at
        )
        
        logger.debug(f"Added reasoning node: {node.node_id}")
    
    async def add_edge(self, edge: ReasoningEdge):
        """Add an edge to the reasoning graph."""
        
        # Ensure nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            logger.warning(f"Edge references unknown nodes: {edge.source_id} -> {edge.target_id}")
            return
        
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=f"{edge.relationship_type}_{len(self.edges)}",  # Unique key for multi-edges
            relationship_type=edge.relationship_type,
            strength=edge.strength,
            confidence=edge.confidence,
            weight=edge.get_weight(),
            properties=edge.properties,
            evidence=edge.evidence,
            created_at=edge.created_at
        )
        
        logger.debug(f"Added reasoning edge: {edge.source_id} -[{edge.relationship_type}]-> {edge.target_id}")
    
    async def execute_reasoning_query(self, query: ReasoningQuery) -> ReasoningResult:
        """Execute a multi-hop reasoning query."""
        
        async with self.query_semaphore:
            start_time = datetime.utcnow()
            
            try:
                # Store active query
                self.active_queries[query.query_id] = query
                
                # Find reasoning paths
                paths = await self._find_reasoning_paths(query)
                
                # Score and rank paths
                scored_paths = []
                for path in paths:
                    score = await self.path_scorer.score_path(path, query, self.graph)
                    path.path_score = score
                    
                    # Generate explanation
                    if not path.explanation:
                        path.explanation = await self.explanation_generator.generate_explanation(
                            path, query
                        )
                    
                    scored_paths.append(path)
                
                # Sort by score and filter by confidence
                scored_paths.sort(key=lambda p: p.path_score, reverse=True)
                filtered_paths = [
                    path for path in scored_paths 
                    if path.confidence >= query.min_confidence
                ]
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Create result
                result = ReasoningResult(
                    query_id=query.query_id,
                    paths=filtered_paths,
                    total_paths_found=len(paths),
                    processing_time=processing_time,
                    metadata={
                        'strategy': query.strategy.value,
                        'max_hops': query.max_hops,
                        'reasoning_types': [rt.value for rt in query.reasoning_types]
                    }
                )
                
                # Calculate confidence distribution
                for path in filtered_paths:
                    confidence_level = path.get_confidence_level()
                    if confidence_level not in result.confidence_distribution:
                        result.confidence_distribution[confidence_level] = 0
                    result.confidence_distribution[confidence_level] += 1
                
                # Update statistics
                self.stats['total_queries'] += 1
                self.stats['successful_queries'] += 1
                self.stats['total_paths_found'] += len(paths)
                self.stats['average_processing_time'] = (
                    (self.stats['average_processing_time'] * (self.stats['total_queries'] - 1) + processing_time) /
                    self.stats['total_queries']
                )
                
                for reasoning_type in query.reasoning_types:
                    self.stats['reasoning_type_usage'][reasoning_type.value] += 1
                
                logger.info(f"Reasoning query completed: {len(filtered_paths)} paths found in {processing_time:.3f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Reasoning query failed: {e}")
                self.stats['total_queries'] += 1
                raise
            
            finally:
                # Clean up
                if query.query_id in self.active_queries:
                    del self.active_queries[query.query_id]
    
    async def _find_reasoning_paths(self, query: ReasoningQuery) -> List[ReasoningPath]:
        """Find reasoning paths based on query strategy."""
        
        if query.strategy == ReasoningStrategy.BREADTH_FIRST:
            return await self._breadth_first_search(query)
        elif query.strategy == ReasoningStrategy.DEPTH_FIRST:
            return await self._depth_first_search(query)
        elif query.strategy == ReasoningStrategy.BEST_FIRST:
            return await self._best_first_search(query)
        elif query.strategy == ReasoningStrategy.A_STAR:
            return await self._a_star_search(query)
        elif query.strategy == ReasoningStrategy.DIJKSTRA:
            return await self._dijkstra_search(query)
        elif query.strategy == ReasoningStrategy.RANDOM_WALK:
            return await self._random_walk_search(query)
        elif query.strategy == ReasoningStrategy.PAGERANK:
            return await self._pagerank_search(query)
        else:
            # Default to best-first search
            return await self._best_first_search(query)
    
    async def _breadth_first_search(self, query: ReasoningQuery) -> List[ReasoningPath]:
        """Breadth-first search for reasoning paths."""
        
        paths = []
        
        for source in query.source_entities:
            if source not in self.graph:
                continue
            
            for target in query.target_entities:
                if target not in self.graph:
                    continue
                
                # Find all simple paths up to max_hops
                try:
                    simple_paths = list(nx.all_simple_paths(
                        self.graph, source, target, cutoff=query.max_hops
                    ))
                    
                    for path_nodes in simple_paths:
                        reasoning_path = await self._create_reasoning_path(
                            path_nodes, query
                        )
                        if reasoning_path:
                            paths.append(reasoning_path)
                
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.warning(f"BFS search failed for {source} -> {target}: {e}")
                    continue
        
        return paths
    
    async def _depth_first_search(self, query: ReasoningQuery) -> List[ReasoningPath]:
        """Depth-first search for reasoning paths."""
        
        paths = []
        
        async def dfs_recursive(current_node, target_nodes, current_path, visited, depth):
            if depth > query.max_hops:
                return
            
            if current_node in target_nodes and len(current_path) > 1:
                # Found a path to target
                path_copy = current_path.copy()
                paths.append(path_copy)
                return
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    current_path.append(neighbor)
                    await dfs_recursive(neighbor, target_nodes, current_path, visited, depth + 1)
                    current_path.pop()
                    visited.remove(neighbor)
        
        for source in query.source_entities:
            if source not in self.graph:
                continue
            
            visited = {source}
            current_path = [source]
            await dfs_recursive(source, set(query.target_entities), current_path, visited, 0)
        
        # Convert node paths to reasoning paths
        reasoning_paths = []
        for node_path in paths:
            reasoning_path = await self._create_reasoning_path(node_path, query)
            if reasoning_path:
                reasoning_paths.append(reasoning_path)
        
        return reasoning_paths
    
    async def _best_first_search(self, query: ReasoningQuery) -> List[ReasoningPath]:
        """Best-first search using heuristic scoring."""
        
        paths = []
        
        for source in query.source_entities:
            if source not in self.graph:
                continue
            
            # Priority queue: (negative_score, path_nodes)
            queue = [(0.0, [source])]
            visited_paths = set()
            
            while queue and len(paths) < 100:  # Limit number of paths
                current_score, current_path = heapq.heappop(queue)
                current_score = -current_score  # Convert back to positive
                
                current_node = current_path[-1]
                
                # Check if we reached a target
                if current_node in query.target_entities and len(current_path) > 1:
                    reasoning_path = await self._create_reasoning_path(current_path, query)
                    if reasoning_path:
                        paths.append(reasoning_path)
                    continue
                
                # Don't expand beyond max hops
                if len(current_path) > query.max_hops:
                    continue
                
                # Expand neighbors
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in current_path:  # Avoid cycles
                        new_path = current_path + [neighbor]
                        path_key = tuple(new_path)
                        
                        if path_key not in visited_paths:
                            visited_paths.add(path_key)
                            
                            # Calculate heuristic score
                            heuristic_score = await self._calculate_heuristic_score(
                                new_path, query
                            )
                            
                            heapq.heappush(queue, (-heuristic_score, new_path))
        
        return paths
    
    async def _a_star_search(self, query: ReasoningQuery) -> List[ReasoningPath]:
        """A* search algorithm for optimal paths."""
        
        paths = []
        
        for source in query.source_entities:
            if source not in self.graph:
                continue
            
            for target in query.target_entities:
                if target not in self.graph:
                    continue
                
                try:
                    # Use NetworkX A* with custom heuristic
                    def heuristic(node1, node2):
                        # Simple heuristic based on node similarity
                        if node1 in self.nodes and node2 in self.nodes:
                            return 1.0 - self.nodes[node1].calculate_similarity(self.nodes[node2])
                        return 1.0
                    
                    path_nodes = nx.astar_path(
                        self.graph, source, target, heuristic=heuristic, weight='weight'
                    )
                    
                    reasoning_path = await self._create_reasoning_path(path_nodes, query)
                    if reasoning_path:
                        paths.append(reasoning_path)
                
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.warning(f"A* search failed for {source} -> {target}: {e}")
                    continue
        
        return paths
    
    async def _dijkstra_search(self, query: ReasoningQuery) -> List[ReasoningPath]:
        """Dijkstra's algorithm for shortest weighted paths."""
        
        paths = []
        
        for source in query.source_entities:
            if source not in self.graph:
                continue
            
            for target in query.target_entities:
                if target not in self.graph:
                    continue
                
                try:
                    path_nodes = nx.dijkstra_path(self.graph, source, target, weight='weight')
                    
                    reasoning_path = await self._create_reasoning_path(path_nodes, query)
                    if reasoning_path:
                        paths.append(reasoning_path)
                
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.warning(f"Dijkstra search failed for {source} -> {target}: {e}")
                    continue
        
        return paths
    
    async def _random_walk_search(self, query: ReasoningQuery) -> List[ReasoningPath]:
        """Random walk exploration for path discovery."""
        
        paths = []
        num_walks = 50  # Number of random walks per source
        walk_length = query.max_hops
        
        for source in query.source_entities:
            if source not in self.graph:
                continue
            
            for _ in range(num_walks):
                current_node = source
                walk_path = [current_node]
                
                for step in range(walk_length):
                    neighbors = list(self.graph.neighbors(current_node))
                    if not neighbors:
                        break
                    
                    # Choose next node randomly (could be weighted by edge strength)
                    weights = []
                    for neighbor in neighbors:
                        edge_data = self.graph.get_edge_data(current_node, neighbor)
                        if edge_data:
                            # Get weight from first edge (in case of multi-edges)
                            weight = list(edge_data.values())[0].get('weight', 1.0)
                            weights.append(weight)
                        else:
                            weights.append(1.0)
                    
                    # Weighted random choice
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                        next_node = np.random.choice(neighbors, p=weights)
                    else:
                        next_node = np.random.choice(neighbors)
                    
                    walk_path.append(next_node)
                    current_node = next_node
                    
                    # Check if we hit a target
                    if current_node in query.target_entities:
                        reasoning_path = await self._create_reasoning_path(walk_path, query)
                        if reasoning_path:
                            paths.append(reasoning_path)
                        break
        
        return paths
    
    async def _pagerank_search(self, query: ReasoningQuery) -> List[ReasoningPath]:
        """PageRank-based importance search."""
        
        paths = []
        
        # Calculate PageRank scores
        try:
            pagerank_scores = nx.pagerank(self.graph, weight='weight')
        except:
            pagerank_scores = {node: 1.0 for node in self.graph.nodes()}
        
        # Find paths using importance-guided search
        for source in query.source_entities:
            if source not in self.graph:
                continue
            
            # Priority queue based on PageRank importance
            queue = [(pagerank_scores.get(source, 0.0), [source])]
            visited_paths = set()
            
            while queue and len(paths) < 50:
                current_importance, current_path = heapq.heappop(queue)
                current_node = current_path[-1]
                
                # Check if we reached a target
                if current_node in query.target_entities and len(current_path) > 1:
                    reasoning_path = await self._create_reasoning_path(current_path, query)
                    if reasoning_path:
                        paths.append(reasoning_path)
                    continue
                
                # Don't expand beyond max hops
                if len(current_path) > query.max_hops:
                    continue
                
                # Expand to high-importance neighbors
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in current_path:
                        new_path = current_path + [neighbor]
                        path_key = tuple(new_path)
                        
                        if path_key not in visited_paths:
                            visited_paths.add(path_key)
                            neighbor_importance = pagerank_scores.get(neighbor, 0.0)
                            heapq.heappush(queue, (-neighbor_importance, new_path))
        
        return paths
    
    async def _create_reasoning_path(
        self,
        node_path: List[str],
        query: ReasoningQuery
    ) -> Optional[ReasoningPath]:
        """Create a ReasoningPath object from a list of node IDs."""
        
        if len(node_path) < 2:
            return None
        
        # Get nodes
        path_nodes = []
        for node_id in node_path:
            if node_id in self.nodes:
                path_nodes.append(self.nodes[node_id])
            else:
                return None  # Invalid path
        
        # Get edges
        path_edges = []
        total_confidence = 0.0
        evidence = []
        
        for i in range(len(node_path) - 1):
            source_id = node_path[i]
            target_id = node_path[i + 1]
            
            # Find edge data
            edge_data = self.graph.get_edge_data(source_id, target_id)
            if not edge_data:
                return None  # No edge found
            
            # Get the first edge (in case of multi-edges)
            first_edge_data = list(edge_data.values())[0]
            
            # Create edge object
            edge = ReasoningEdge(
                source_id=source_id,
                target_id=target_id,
                relationship_type=first_edge_data.get('relationship_type', 'related'),
                strength=first_edge_data.get('strength', 1.0),
                confidence=first_edge_data.get('confidence', 1.0),
                properties=first_edge_data.get('properties', {}),
                evidence=first_edge_data.get('evidence', [])
            )
            
            path_edges.append(edge)
            total_confidence += edge.confidence
            evidence.extend(edge.evidence)
        
        # Calculate overall confidence
        avg_confidence = total_confidence / len(path_edges) if path_edges else 0.0
        
        # Determine reasoning type
        reasoning_type = self._infer_reasoning_type(path_edges, query)
        
        # Create path ID
        path_id = hashlib.md5(
            "->".join(node_path).encode()
        ).hexdigest()[:16]
        
        return ReasoningPath(
            path_id=path_id,
            nodes=path_nodes,
            edges=path_edges,
            reasoning_type=reasoning_type,
            confidence=avg_confidence,
            path_score=0.0,  # Will be calculated later
            evidence=evidence,
            metadata={'query_id': query.query_id}
        )
    
    def _infer_reasoning_type(
        self,
        path_edges: List[ReasoningEdge],
        query: ReasoningQuery
    ) -> ReasoningType:
        """Infer the reasoning type from path characteristics."""
        
        # Use query preferences if available
        if query.reasoning_types:
            return query.reasoning_types[0]  # Use first preference
        
        # Infer from edge types
        edge_types = [edge.relationship_type for edge in path_edges]
        
        if any('cause' in edge_type for edge_type in edge_types):
            return ReasoningType.CAUSAL
        elif any('temporal' in edge_type for edge_type in edge_types):
            return ReasoningType.TEMPORAL
        elif any('similar' in edge_type for edge_type in edge_types):
            return ReasoningType.ANALOGICAL
        elif len(path_edges) > 1:
            return ReasoningType.TRANSITIVE
        else:
            return ReasoningType.DEDUCTIVE
    
    async def _calculate_heuristic_score(
        self,
        path_nodes: List[str],
        query: ReasoningQuery
    ) -> float:
        """Calculate heuristic score for best-first search."""
        
        current_node = path_nodes[-1]
        
        # Distance to target heuristic
        min_distance_to_target = float('inf')
        for target in query.target_entities:
            if target in self.graph:
                try:
                    distance = nx.shortest_path_length(self.graph, current_node, target)
                    min_distance_to_target = min(min_distance_to_target, distance)
                except nx.NetworkXNoPath:
                    continue
        
        if min_distance_to_target == float('inf'):
            distance_score = 0.0
        else:
            distance_score = 1.0 / (1.0 + min_distance_to_target)
        
        # Path length penalty
        length_penalty = 1.0 / (1.0 + len(path_nodes))
        
        # Node importance (if available)
        node_importance = 0.0
        if current_node in self.nodes:
            node_importance = self.nodes[current_node].importance_score
        
        # Combine scores
        heuristic_score = 0.5 * distance_score + 0.3 * length_penalty + 0.2 * node_importance
        
        return heuristic_score
    
    async def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get comprehensive reasoning engine statistics."""
        
        try:
            connected_components = nx.number_connected_components(self.graph.to_undirected())
        except:
            connected_components = 0
            
        return {
            'graph_stats': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'graph_density': nx.density(self.graph) if len(self.nodes) > 0 else 0,
                'connected_components': connected_components
            },
            'query_stats': self.stats.copy(),
            'active_queries': len(self.active_queries),
            'path_scorer_stats': {
                'scored_paths': len(self.path_scorer.scored_paths)
            }
        }
    
    async def clear_cache(self):
        """Clear reasoning caches."""
        
        self.path_scorer.scored_paths.clear()
        
        if self.redis_cache:
            # Could implement Redis cache clearing here
            pass
        
        logger.info("Reasoning engine caches cleared")