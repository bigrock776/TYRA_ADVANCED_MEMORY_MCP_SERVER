"""
Local Auto-Generated Connections System.

This module provides intelligent auto-generation of connections between memories
using local relationship detection algorithms, graph analysis, confidence scoring,
validation systems, and refinement through local learning for optimal memory networks.
"""

import asyncio
import math
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
from collections import defaultdict, Counter
import spacy
import re

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ...core.embeddings.embedder import Embedder
from ...core.memory.manager import MemoryManager
from ...core.graph.neo4j_client import Neo4jClient
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ConnectionType(str, Enum):
    """Types of memory connections that can be auto-generated."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    ENTITY_MENTION = "entity_mention"
    TOPIC_OVERLAP = "topic_overlap"
    CONTRADICTION = "contradiction"
    ELABORATION = "elaboration"
    EXAMPLE_OF = "example_of"
    PREREQUISITE = "prerequisite"
    CONTINUATION = "continuation"


class ConnectionStrength(str, Enum):
    """Strength levels for memory connections."""
    VERY_STRONG = "very_strong"    # 0.9-1.0
    STRONG = "strong"              # 0.7-0.89
    MODERATE = "moderate"          # 0.5-0.69
    WEAK = "weak"                  # 0.3-0.49
    VERY_WEAK = "very_weak"        # 0.1-0.29


class ValidationStatus(str, Enum):
    """Validation status for connections."""
    VALIDATED = "validated"
    PENDING = "pending"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass
class ConnectionContext:
    """Context for connection generation."""
    user_id: str
    memory_ids: Optional[List[str]] = None  # Specific memories to analyze
    max_connections: int = 50
    min_confidence: float = 0.3
    include_weak_connections: bool = False
    validate_connections: bool = True
    connection_types: Optional[List[ConnectionType]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MemoryConnection(BaseModel):
    """Structured memory connection with comprehensive metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    connection_id: str = Field(..., description="Unique connection identifier")
    source_memory_id: str = Field(..., description="Source memory ID")
    target_memory_id: str = Field(..., description="Target memory ID")
    connection_type: ConnectionType = Field(..., description="Type of connection")
    strength: float = Field(..., ge=0.0, le=1.0, description="Connection strength")
    strength_category: ConnectionStrength = Field(..., description="Strength category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in connection")
    
    # Connection details
    explanation: str = Field(..., min_length=10, description="Why these memories are connected")
    evidence: List[str] = Field(..., description="Evidence supporting the connection")
    shared_elements: Dict[str, List[str]] = Field(..., description="Shared elements between memories")
    
    # Validation and quality
    validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING, description="Validation status")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    reliability_indicators: Dict[str, float] = Field(..., description="Reliability indicators")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_validated: Optional[datetime] = Field(None, description="Last validation timestamp")
    user_feedback: Optional[Dict[str, Any]] = Field(None, description="User feedback")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('strength')
    @classmethod
    def set_strength_category(cls, v):
        """Set strength category based on strength value."""
        return v
    
    def __post_init__(self):
        """Set strength category after initialization."""
        if self.strength >= 0.9:
            self.strength_category = ConnectionStrength.VERY_STRONG
        elif self.strength >= 0.7:
            self.strength_category = ConnectionStrength.STRONG
        elif self.strength >= 0.5:
            self.strength_category = ConnectionStrength.MODERATE
        elif self.strength >= 0.3:
            self.strength_category = ConnectionStrength.WEAK
        else:
            self.strength_category = ConnectionStrength.VERY_WEAK


class ConnectionResult(BaseModel):
    """Complete connection generation result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    connections: List[MemoryConnection] = Field(..., description="Generated connections")
    total_pairs_analyzed: int = Field(..., ge=0, description="Total memory pairs analyzed")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    connection_strategies: List[str] = Field(..., description="Strategies used for connection detection")
    network_analysis: Dict[str, Any] = Field(..., description="Network analysis results")
    quality_metrics: Dict[str, float] = Field(..., description="Overall quality metrics")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Result timestamp")


class LocalConnectionGenerator:
    """
    Local Auto-Generated Connections System.
    
    Automatically discovers and creates meaningful connections between memories
    using multiple local algorithms and validation systems.
    """
    
    def __init__(
        self,
        embedder: Embedder,
        memory_manager: MemoryManager,
        graph_client: Optional[Neo4jClient] = None
    ):
        """Initialize local connection generator."""
        self.embedder = embedder
        self.memory_manager = memory_manager
        self.graph_client = graph_client
        
        # NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic NLP")
            self.nlp = None
        
        # Machine learning models
        self.connection_classifier: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        
        # Connection detection patterns
        self.causal_patterns = [
            r'because\s+of\s+(.+?)[\.,]',
            r'due\s+to\s+(.+?)[\.,]',
            r'as\s+a\s+result\s+of\s+(.+?)[\.,]',
            r'caused\s+by\s+(.+?)[\.,]',
            r'leads\s+to\s+(.+?)[\.,]',
            r'results\s+in\s+(.+?)[\.,]',
            r'therefore\s+(.+?)[\.,]',
            r'consequently\s+(.+?)[\.,]'
        ]
        
        self.temporal_patterns = [
            r'after\s+(.+?)[\.,]',
            r'before\s+(.+?)[\.,]',
            r'during\s+(.+?)[\.,]',
            r'while\s+(.+?)[\.,]',
            r'then\s+(.+?)[\.,]',
            r'next\s+(.+?)[\.,]',
            r'previously\s+(.+?)[\.,]',
            r'subsequently\s+(.+?)[\.,]'
        ]
        
        self.elaboration_patterns = [
            r'specifically\s+(.+?)[\.,]',
            r'for\s+example\s+(.+?)[\.,]',
            r'in\s+particular\s+(.+?)[\.,]',
            r'namely\s+(.+?)[\.,]',
            r'such\s+as\s+(.+?)[\.,]',
            r'including\s+(.+?)[\.,]',
            r'especially\s+(.+?)[\.,]'
        ]
        
        # Caching
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.connection_cache: Dict[str, List[MemoryConnection]] = {}
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.connection_stats = {
            'total_connections_generated': 0,
            'validated_connections': 0,
            'rejected_connections': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'accuracy_feedback': []
        }
        
        logger.info("Initialized LocalConnectionGenerator")
    
    async def generate_connections(
        self,
        context: ConnectionContext
    ) -> ConnectionResult:
        """
        Generate connections between memories based on context.
        
        Args:
            context: Connection generation context
            
        Returns:
            Complete connection result with network analysis
        """
        start_time = datetime.utcnow()
        
        try:
            # Get memories to analyze
            memories = await self._get_memories_for_analysis(context)
            
            if len(memories) < 2:
                return ConnectionResult(
                    connections=[],
                    total_pairs_analyzed=0,
                    processing_time_ms=0.0,
                    connection_strategies=[],
                    network_analysis={"status": "insufficient_memories"},
                    quality_metrics={},
                    recommendations=["Add more memories to enable connection generation"]
                )
            
            # Generate all possible memory pairs
            memory_pairs = self._generate_memory_pairs(memories, context.max_connections * 2)
            
            # Detect connections using multiple strategies
            connections = []
            strategies_used = []
            
            # Strategy 1: Semantic similarity connections
            semantic_connections = await self._detect_semantic_connections(
                memory_pairs, context
            )
            connections.extend(semantic_connections)
            if semantic_connections:
                strategies_used.append("semantic_similarity")
            
            # Strategy 2: Entity-based connections
            entity_connections = await self._detect_entity_connections(
                memory_pairs, context
            )
            connections.extend(entity_connections)
            if entity_connections:
                strategies_used.append("entity_detection")
            
            # Strategy 3: Causal relationship detection
            causal_connections = await self._detect_causal_connections(
                memory_pairs, context
            )
            connections.extend(causal_connections)
            if causal_connections:
                strategies_used.append("causal_analysis")
            
            # Strategy 4: Temporal sequence detection
            temporal_connections = await self._detect_temporal_connections(
                memory_pairs, context
            )
            connections.extend(temporal_connections)
            if temporal_connections:
                strategies_used.append("temporal_analysis")
            
            # Strategy 5: Topic overlap detection
            topic_connections = await self._detect_topic_connections(
                memory_pairs, context
            )
            connections.extend(topic_connections)
            if topic_connections:
                strategies_used.append("topic_analysis")
            
            # Strategy 6: Contradiction detection
            contradiction_connections = await self._detect_contradictions(
                memory_pairs, context
            )
            connections.extend(contradiction_connections)
            if contradiction_connections:
                strategies_used.append("contradiction_detection")
            
            # Strategy 7: Elaboration pattern detection
            elaboration_connections = await self._detect_elaborations(
                memory_pairs, context
            )
            connections.extend(elaboration_connections)
            if elaboration_connections:
                strategies_used.append("elaboration_detection")
            
            # Remove duplicates and filter by confidence
            unique_connections = self._deduplicate_connections(connections)
            filtered_connections = [
                c for c in unique_connections 
                if c.confidence >= context.min_confidence
            ]
            
            # Apply ML-based validation if model available
            if self.connection_classifier and context.validate_connections:
                filtered_connections = await self._ml_validate_connections(
                    filtered_connections, memories
                )
                strategies_used.append("ml_validation")
            
            # Apply rule-based validation
            if context.validate_connections:
                filtered_connections = await self._rule_based_validation(
                    filtered_connections, memories
                )
                strategies_used.append("rule_validation")
            
            # Limit to max connections and sort by quality
            final_connections = self._rank_and_limit_connections(
                filtered_connections, context.max_connections
            )
            
            # Update connection categories
            for connection in final_connections:
                connection.__post_init__()
            
            # Perform network analysis
            network_analysis = await self._analyze_connection_network(
                final_connections, memories
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                final_connections, len(memory_pairs)
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                final_connections, quality_metrics, strategies_used
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update statistics
            self.connection_stats['total_connections_generated'] += len(final_connections)
            self.connection_stats['average_processing_time'] = (
                (self.connection_stats['average_processing_time'] * 
                 (self.connection_stats['total_connections_generated'] - len(final_connections)) + 
                 processing_time) /
                self.connection_stats['total_connections_generated']
            )
            
            # Store connections if graph client available
            if self.graph_client and final_connections:
                await self._store_connections_in_graph(final_connections)
            
            logger.info(
                "Generated memory connections",
                user_id=context.user_id,
                connections_count=len(final_connections),
                pairs_analyzed=len(memory_pairs),
                processing_time_ms=processing_time,
                strategies=strategies_used
            )
            
            return ConnectionResult(
                connections=final_connections,
                total_pairs_analyzed=len(memory_pairs),
                processing_time_ms=processing_time,
                connection_strategies=strategies_used,
                network_analysis=network_analysis,
                quality_metrics=quality_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error("Error generating connections", error=str(e))
            return ConnectionResult(
                connections=[],
                total_pairs_analyzed=0,
                processing_time_ms=0.0,
                connection_strategies=[],
                network_analysis={"error": str(e)},
                quality_metrics={},
                recommendations=["System error - please try again"]
            )
    
    async def _get_memories_for_analysis(self, context: ConnectionContext) -> List[Dict[str, Any]]:
        """Get memories for connection analysis."""
        try:
            if context.memory_ids:
                # Get specific memories
                memories = []
                for memory_id in context.memory_ids:
                    memory = await self.memory_manager.get_memory(memory_id)
                    if memory:
                        memories.append({
                            "id": memory.id,
                            "content": memory.content,
                            "metadata": memory.metadata,
                            "created_at": memory.created_at,
                            "updated_at": memory.updated_at,
                            "tags": getattr(memory, 'tags', []),
                            "entities": getattr(memory, 'entities', [])
                        })
            else:
                # Get all memories for user
                user_memories = await self.memory_manager.get_memories_for_user(
                    context.user_id,
                    limit=200  # Reasonable limit for connection analysis
                )
                
                memories = []
                for memory in user_memories:
                    memories.append({
                        "id": memory.id,
                        "content": memory.content,
                        "metadata": memory.metadata,
                        "created_at": memory.created_at,
                        "updated_at": memory.updated_at,
                        "tags": getattr(memory, 'tags', []),
                        "entities": getattr(memory, 'entities', [])
                    })
            
            logger.debug(f"Retrieved {len(memories)} memories for connection analysis")
            return memories
            
        except Exception as e:
            logger.error("Error retrieving memories for analysis", error=str(e))
            return []
    
    def _generate_memory_pairs(
        self,
        memories: List[Dict[str, Any]],
        max_pairs: int
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Generate memory pairs for connection analysis."""
        pairs = []
        
        # Generate all possible pairs
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                pairs.append((memories[i], memories[j]))
                
                # Limit pairs to avoid excessive computation
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        
        # Sort pairs by potential for connection (heuristic: similar creation times)
        pairs.sort(key=lambda p: abs((p[0]["created_at"] - p[1]["created_at"]).total_seconds()))
        
        return pairs[:max_pairs]
    
    async def _detect_semantic_connections(
        self,
        memory_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        context: ConnectionContext
    ) -> List[MemoryConnection]:
        """Detect semantic similarity connections."""
        connections = []
        
        for mem1, mem2 in memory_pairs:
            try:
                # Get embeddings
                emb1 = await self._get_or_compute_embedding(mem1["content"])
                emb2 = await self._get_or_compute_embedding(mem2["content"])
                
                if emb1 is None or emb2 is None:
                    continue
                
                # Compute cosine similarity
                similarity = float(cosine_similarity(
                    emb1.reshape(1, -1),
                    emb2.reshape(1, -1)
                )[0][0])
                
                # Threshold for semantic connection
                if similarity > 0.6:
                    connection = self._create_connection(
                        mem1, mem2,
                        ConnectionType.SEMANTIC_SIMILARITY,
                        strength=similarity,
                        confidence=similarity,
                        explanation=f"Semantically similar content (similarity: {similarity:.3f})",
                        evidence=[f"Cosine similarity: {similarity:.3f}"],
                        shared_elements={"similarity_score": [str(similarity)]}
                    )
                    connections.append(connection)
                    
            except Exception as e:
                logger.warning(f"Error detecting semantic connection: {e}")
                continue
        
        return connections
    
    async def _detect_entity_connections(
        self,
        memory_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        context: ConnectionContext
    ) -> List[MemoryConnection]:
        """Detect entity-based connections."""
        connections = []
        
        for mem1, mem2 in memory_pairs:
            try:
                # Extract entities
                entities1 = set(self._extract_entities(mem1["content"]))
                entities2 = set(self._extract_entities(mem2["content"]))
                
                # Add stored entities if available
                if "entities" in mem1:
                    entities1.update(mem1["entities"])
                if "entities" in mem2:
                    entities2.update(mem2["entities"])
                
                # Find shared entities
                shared_entities = entities1.intersection(entities2)
                
                if shared_entities:
                    # Calculate entity overlap strength
                    overlap_ratio = len(shared_entities) / len(entities1.union(entities2))
                    
                    # Boost based on number of shared entities
                    entity_boost = min(len(shared_entities) / 10.0, 0.3)
                    strength = min(overlap_ratio + entity_boost, 1.0)
                    
                    if strength > 0.3:
                        connection = self._create_connection(
                            mem1, mem2,
                            ConnectionType.ENTITY_MENTION,
                            strength=strength,
                            confidence=strength,
                            explanation=f"Share entities: {', '.join(list(shared_entities)[:3])}",
                            evidence=[f"Shared entities: {', '.join(shared_entities)}"],
                            shared_elements={"entities": list(shared_entities)}
                        )
                        connections.append(connection)
                
            except Exception as e:
                logger.warning(f"Error detecting entity connection: {e}")
                continue
        
        return connections
    
    async def _detect_causal_connections(
        self,
        memory_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        context: ConnectionContext
    ) -> List[MemoryConnection]:
        """Detect causal relationship connections."""
        connections = []
        
        for mem1, mem2 in memory_pairs:
            try:
                content1 = mem1["content"].lower()
                content2 = mem2["content"].lower()
                
                # Check for causal patterns in both directions
                causal_evidence = []
                causal_strength = 0.0
                
                # Check mem1 -> mem2 causality
                for pattern in self.causal_patterns:
                    matches1 = re.findall(pattern, content1)
                    matches2 = re.findall(pattern, content2)
                    
                    if matches1 or matches2:
                        causal_evidence.extend(matches1 + matches2)
                        causal_strength += 0.1
                
                # Look for cross-references
                if any(word in content2 for word in content1.split() if len(word) > 4):
                    cross_ref_strength = len([
                        word for word in content1.split() 
                        if len(word) > 4 and word in content2
                    ]) / max(len(content1.split()), 1)
                    
                    if cross_ref_strength > 0.2:
                        causal_strength += cross_ref_strength * 0.3
                        causal_evidence.append(f"Cross-reference strength: {cross_ref_strength:.3f}")
                
                # Create connection if strong enough
                if causal_strength > 0.4 and causal_evidence:
                    connection = self._create_connection(
                        mem1, mem2,
                        ConnectionType.CAUSAL_RELATIONSHIP,
                        strength=min(causal_strength, 1.0),
                        confidence=min(causal_strength * 0.8, 1.0),  # Lower confidence for causal
                        explanation=f"Potential causal relationship detected",
                        evidence=causal_evidence[:3],  # Limit evidence
                        shared_elements={"causal_indicators": causal_evidence}
                    )
                    connections.append(connection)
                
            except Exception as e:
                logger.warning(f"Error detecting causal connection: {e}")
                continue
        
        return connections
    
    async def _detect_temporal_connections(
        self,
        memory_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        context: ConnectionContext
    ) -> List[MemoryConnection]:
        """Detect temporal sequence connections."""
        connections = []
        
        for mem1, mem2 in memory_pairs:
            try:
                # Check temporal proximity
                time_diff = abs((mem1["created_at"] - mem2["created_at"]).total_seconds())
                
                # Strong temporal connection if within 24 hours
                if time_diff < 86400:  # 24 hours
                    temporal_strength = 1.0 - (time_diff / 86400)
                    
                    # Look for temporal language patterns
                    content1 = mem1["content"].lower()
                    content2 = mem2["content"].lower()
                    
                    temporal_evidence = []
                    pattern_strength = 0.0
                    
                    for pattern in self.temporal_patterns:
                        matches1 = re.findall(pattern, content1)
                        matches2 = re.findall(pattern, content2)
                        
                        if matches1 or matches2:
                            temporal_evidence.extend(matches1 + matches2)
                            pattern_strength += 0.1
                    
                    # Combine temporal proximity and pattern evidence
                    total_strength = (temporal_strength * 0.6) + (pattern_strength * 0.4)
                    
                    if total_strength > 0.3:
                        time_desc = self._humanize_time_diff(time_diff)
                        
                        connection = self._create_connection(
                            mem1, mem2,
                            ConnectionType.TEMPORAL_SEQUENCE,
                            strength=min(total_strength, 1.0),
                            confidence=temporal_strength,  # Higher confidence in time proximity
                            explanation=f"Temporal sequence - created {time_desc} apart",
                            evidence=[f"Time difference: {time_desc}"] + temporal_evidence[:2],
                            shared_elements={"temporal_indicators": temporal_evidence}
                        )
                        connections.append(connection)
                
            except Exception as e:
                logger.warning(f"Error detecting temporal connection: {e}")
                continue
        
        return connections
    
    async def _detect_topic_connections(
        self,
        memory_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        context: ConnectionContext
    ) -> List[MemoryConnection]:
        """Detect topic overlap connections."""
        connections = []
        
        for mem1, mem2 in memory_pairs:
            try:
                # Extract topics/keywords
                topics1 = set(self._extract_topics(mem1["content"]))
                topics2 = set(self._extract_topics(mem2["content"]))
                
                # Add stored tags as topics
                if "tags" in mem1:
                    topics1.update(mem1["tags"])
                if "tags" in mem2:
                    topics2.update(mem2["tags"])
                
                # Find shared topics
                shared_topics = topics1.intersection(topics2)
                
                if shared_topics:
                    # Calculate topic overlap strength
                    overlap_ratio = len(shared_topics) / len(topics1.union(topics2))
                    
                    # Boost based on number of shared topics
                    topic_boost = min(len(shared_topics) / 5.0, 0.4)
                    strength = min(overlap_ratio + topic_boost, 1.0)
                    
                    if strength > 0.4:
                        connection = self._create_connection(
                            mem1, mem2,
                            ConnectionType.TOPIC_OVERLAP,
                            strength=strength,
                            confidence=strength * 0.9,
                            explanation=f"Share topics: {', '.join(list(shared_topics)[:3])}",
                            evidence=[f"Common topics: {', '.join(shared_topics)}"],
                            shared_elements={"topics": list(shared_topics)}
                        )
                        connections.append(connection)
                
            except Exception as e:
                logger.warning(f"Error detecting topic connection: {e}")
                continue
        
        return connections
    
    async def _detect_contradictions(
        self,
        memory_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        context: ConnectionContext
    ) -> List[MemoryConnection]:
        """Detect contradiction connections."""
        connections = []
        
        # Contradiction patterns
        contradiction_patterns = [
            (r'is\s+(\w+)', r'is\s+not\s+\1'),
            (r'was\s+(\w+)', r'was\s+not\s+\1'),
            (r'will\s+(\w+)', r'will\s+not\s+\1'),
            (r'can\s+(\w+)', r'cannot\s+\1'),
            (r'should\s+(\w+)', r'should\s+not\s+\1'),
            (r'(\d+)', r'(?!\1)\d+'),  # Different numbers
        ]
        
        for mem1, mem2 in memory_pairs:
            try:
                content1 = mem1["content"].lower()
                content2 = mem2["content"].lower()
                
                contradiction_evidence = []
                contradiction_strength = 0.0
                
                # Check for direct contradictions
                for positive_pattern, negative_pattern in contradiction_patterns:
                    pos_matches1 = re.findall(positive_pattern, content1)
                    neg_matches2 = re.findall(negative_pattern, content2)
                    
                    pos_matches2 = re.findall(positive_pattern, content2)
                    neg_matches1 = re.findall(negative_pattern, content1)
                    
                    if (pos_matches1 and neg_matches2) or (pos_matches2 and neg_matches1):
                        contradiction_evidence.append("Direct contradiction pattern")
                        contradiction_strength += 0.3
                
                # Check for semantic contradictions using opposing keywords
                opposing_pairs = [
                    ("good", "bad"), ("positive", "negative"), ("increase", "decrease"),
                    ("up", "down"), ("yes", "no"), ("true", "false"), ("correct", "wrong"),
                    ("success", "failure"), ("win", "lose"), ("buy", "sell")
                ]
                
                for word1, word2 in opposing_pairs:
                    if word1 in content1 and word2 in content2:
                        contradiction_evidence.append(f"Opposing concepts: {word1} vs {word2}")
                        contradiction_strength += 0.2
                    elif word2 in content1 and word1 in content2:
                        contradiction_evidence.append(f"Opposing concepts: {word2} vs {word1}")
                        contradiction_strength += 0.2
                
                # Create contradiction connection if strong enough
                if contradiction_strength > 0.4 and contradiction_evidence:
                    connection = self._create_connection(
                        mem1, mem2,
                        ConnectionType.CONTRADICTION,
                        strength=min(contradiction_strength, 1.0),
                        confidence=contradiction_strength * 0.7,  # Lower confidence for contradictions
                        explanation=f"Potential contradiction detected",
                        evidence=contradiction_evidence[:3],
                        shared_elements={"contradiction_indicators": contradiction_evidence}
                    )
                    connections.append(connection)
                
            except Exception as e:
                logger.warning(f"Error detecting contradiction: {e}")
                continue
        
        return connections
    
    async def _detect_elaborations(
        self,
        memory_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        context: ConnectionContext
    ) -> List[MemoryConnection]:
        """Detect elaboration connections."""
        connections = []
        
        for mem1, mem2 in memory_pairs:
            try:
                content1 = mem1["content"].lower()
                content2 = mem2["content"].lower()
                
                elaboration_evidence = []
                elaboration_strength = 0.0
                
                # Check for elaboration patterns
                for pattern in self.elaboration_patterns:
                    matches1 = re.findall(pattern, content1)
                    matches2 = re.findall(pattern, content2)
                    
                    if matches1 or matches2:
                        elaboration_evidence.extend(matches1 + matches2)
                        elaboration_strength += 0.2
                
                # Check for example relationships
                if "example" in content1 and any(word in content2 for word in content1.split() if len(word) > 4):
                    elaboration_evidence.append("Example relationship detected")
                    elaboration_strength += 0.3
                
                # Check for detailed explanation patterns
                detail_keywords = ["detail", "specifically", "particularly", "furthermore", "moreover"]
                for keyword in detail_keywords:
                    if keyword in content1 or keyword in content2:
                        elaboration_evidence.append(f"Detail keyword: {keyword}")
                        elaboration_strength += 0.1
                
                # Create elaboration connection if strong enough
                if elaboration_strength > 0.3 and elaboration_evidence:
                    connection = self._create_connection(
                        mem1, mem2,
                        ConnectionType.ELABORATION,
                        strength=min(elaboration_strength, 1.0),
                        confidence=elaboration_strength * 0.8,
                        explanation=f"Elaboration relationship detected",
                        evidence=elaboration_evidence[:3],
                        shared_elements={"elaboration_indicators": elaboration_evidence}
                    )
                    connections.append(connection)
                
            except Exception as e:
                logger.warning(f"Error detecting elaboration: {e}")
                continue
        
        return connections
    
    def _create_connection(
        self,
        mem1: Dict[str, Any],
        mem2: Dict[str, Any],
        connection_type: ConnectionType,
        strength: float,
        confidence: float,
        explanation: str,
        evidence: List[str],
        shared_elements: Dict[str, List[str]]
    ) -> MemoryConnection:
        """Create a memory connection with proper metadata."""
        connection_id = hashlib.md5(
            f"{mem1['id']}-{mem2['id']}-{connection_type.value}".encode()
        ).hexdigest()[:16]
        
        # Calculate quality score
        quality_score = (strength * 0.4) + (confidence * 0.4) + (len(evidence) / 10.0 * 0.2)
        quality_score = min(quality_score, 1.0)
        
        # Calculate reliability indicators
        reliability_indicators = {
            "evidence_count": len(evidence),
            "shared_elements_count": sum(len(v) for v in shared_elements.values()),
            "strength_confidence_ratio": confidence / max(strength, 0.1),
            "type_reliability": self._get_type_reliability(connection_type)
        }
        
        return MemoryConnection(
            connection_id=connection_id,
            source_memory_id=mem1["id"],
            target_memory_id=mem2["id"],
            connection_type=connection_type,
            strength=strength,
            strength_category=ConnectionStrength.MODERATE,  # Will be updated in __post_init__
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            shared_elements=shared_elements,
            quality_score=quality_score,
            reliability_indicators=reliability_indicators
        )
    
    def _get_type_reliability(self, connection_type: ConnectionType) -> float:
        """Get reliability score for connection type."""
        type_reliability = {
            ConnectionType.SEMANTIC_SIMILARITY: 0.9,
            ConnectionType.ENTITY_MENTION: 0.85,
            ConnectionType.TOPIC_OVERLAP: 0.8,
            ConnectionType.TEMPORAL_SEQUENCE: 0.75,
            ConnectionType.ELABORATION: 0.7,
            ConnectionType.CAUSAL_RELATIONSHIP: 0.6,
            ConnectionType.CONTRADICTION: 0.5,
            ConnectionType.EXAMPLE_OF: 0.65,
            ConnectionType.PREREQUISITE: 0.55,
            ConnectionType.CONTINUATION: 0.7
        }
        return type_reliability.get(connection_type, 0.5)
    
    async def _ml_validate_connections(
        self,
        connections: List[MemoryConnection],
        memories: List[Dict[str, Any]]
    ) -> List[MemoryConnection]:
        """Validate connections using ML model."""
        if not self.connection_classifier or not connections:
            return connections
        
        try:
            # Extract features for each connection
            features = []
            for connection in connections:
                feature_vector = self._extract_connection_features(connection, memories)
                features.append(feature_vector)
            
            # Predict validity
            feature_matrix = np.array(features)
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
            
            predictions = self.connection_classifier.predict_proba(feature_matrix_scaled)
            
            # Update connections based on predictions
            validated_connections = []
            for i, connection in enumerate(connections):
                validity_prob = predictions[i][1]  # Probability of being valid
                
                if validity_prob > 0.6:  # Threshold for validation
                    connection.validation_status = ValidationStatus.VALIDATED
                    connection.confidence *= validity_prob  # Adjust confidence
                    validated_connections.append(connection)
                elif validity_prob > 0.3:
                    connection.validation_status = ValidationStatus.UNCERTAIN
                    connection.confidence *= validity_prob
                    validated_connections.append(connection)
                else:
                    connection.validation_status = ValidationStatus.REJECTED
            
            self.connection_stats['validated_connections'] += len([
                c for c in validated_connections 
                if c.validation_status == ValidationStatus.VALIDATED
            ])
            self.connection_stats['rejected_connections'] += len(connections) - len(validated_connections)
            
            return validated_connections
            
        except Exception as e:
            logger.error("Error in ML validation", error=str(e))
            return connections
    
    def _extract_connection_features(
        self,
        connection: MemoryConnection,
        memories: List[Dict[str, Any]]
    ) -> List[float]:
        """Extract feature vector for ML validation."""
        features = []
        
        # Basic connection features
        features.append(connection.strength)
        features.append(connection.confidence)
        features.append(connection.quality_score)
        features.append(len(connection.evidence))
        features.append(sum(len(v) for v in connection.shared_elements.values()))
        
        # Connection type (one-hot encoding)
        connection_types = list(ConnectionType)
        for ctype in connection_types:
            features.append(1.0 if connection.connection_type == ctype else 0.0)
        
        # Reliability indicators
        for indicator in connection.reliability_indicators.values():
            features.append(float(indicator))
        
        # Memory pair features
        source_mem = next((m for m in memories if m["id"] == connection.source_memory_id), None)
        target_mem = next((m for m in memories if m["id"] == connection.target_memory_id), None)
        
        if source_mem and target_mem:
            # Content length features
            features.append(len(source_mem["content"]) / 1000.0)  # Normalize
            features.append(len(target_mem["content"]) / 1000.0)
            
            # Time difference
            time_diff = abs((source_mem["created_at"] - target_mem["created_at"]).total_seconds())
            features.append(min(time_diff / 86400, 30))  # Days, capped at 30
            
            # Metadata similarity
            meta_similarity = self._calculate_metadata_similarity(
                source_mem.get("metadata", {}),
                target_mem.get("metadata", {})
            )
            features.append(meta_similarity)
        else:
            # Default values if memories not found
            features.extend([0.5, 0.5, 1.0, 0.0])
        
        return features
    
    def _calculate_metadata_similarity(self, meta1: Dict, meta2: Dict) -> float:
        """Calculate similarity between metadata dictionaries."""
        if not meta1 or not meta2:
            return 0.0
        
        common_keys = set(meta1.keys()).intersection(set(meta2.keys()))
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        for key in common_keys:
            if meta1[key] == meta2[key]:
                similarity_sum += 1.0
            elif isinstance(meta1[key], str) and isinstance(meta2[key], str):
                # Simple string similarity
                similarity_sum += len(set(meta1[key].split()).intersection(set(meta2[key].split()))) / max(
                    len(set(meta1[key].split()).union(set(meta2[key].split()))), 1
                )
        
        return similarity_sum / len(common_keys)
    
    async def _rule_based_validation(
        self,
        connections: List[MemoryConnection],
        memories: List[Dict[str, Any]]
    ) -> List[MemoryConnection]:
        """Apply rule-based validation to connections."""
        validated_connections = []
        
        for connection in connections:
            is_valid = True
            validation_notes = []
            
            # Rule 1: No self-connections
            if connection.source_memory_id == connection.target_memory_id:
                is_valid = False
                validation_notes.append("Self-connection not allowed")
            
            # Rule 2: Minimum confidence threshold
            if connection.confidence < 0.2:
                is_valid = False
                validation_notes.append("Confidence too low")
            
            # Rule 3: Evidence requirement
            if len(connection.evidence) == 0:
                is_valid = False
                validation_notes.append("No evidence provided")
            
            # Rule 4: Type-specific validation
            if connection.connection_type == ConnectionType.CONTRADICTION:
                # Contradictions need strong evidence
                if connection.confidence < 0.5:
                    is_valid = False
                    validation_notes.append("Contradiction needs higher confidence")
            
            # Rule 5: Temporal sequence validation
            if connection.connection_type == ConnectionType.TEMPORAL_SEQUENCE:
                source_mem = next((m for m in memories if m["id"] == connection.source_memory_id), None)
                target_mem = next((m for m in memories if m["id"] == connection.target_memory_id), None)
                
                if source_mem and target_mem:
                    time_diff = abs((source_mem["created_at"] - target_mem["created_at"]).total_seconds())
                    if time_diff > 604800:  # More than a week
                        connection.confidence *= 0.5  # Reduce confidence
                        validation_notes.append("Long temporal gap")
            
            # Apply validation result
            if is_valid:
                if connection.validation_status == ValidationStatus.PENDING:
                    connection.validation_status = ValidationStatus.VALIDATED
                connection.last_validated = datetime.utcnow()
                validated_connections.append(connection)
            else:
                connection.validation_status = ValidationStatus.REJECTED
                connection.metadata["validation_notes"] = validation_notes
        
        return validated_connections
    
    def _rank_and_limit_connections(
        self,
        connections: List[MemoryConnection],
        max_connections: int
    ) -> List[MemoryConnection]:
        """Rank connections by quality and limit to max count."""
        # Sort by composite score: (quality_score * 0.6) + (confidence * 0.4)
        connections.sort(
            key=lambda c: (c.quality_score * 0.6) + (c.confidence * 0.4),
            reverse=True
        )
        
        return connections[:max_connections]
    
    async def _analyze_connection_network(
        self,
        connections: List[MemoryConnection],
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the network structure of connections."""
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes (memories)
            for memory in memories:
                G.add_node(memory["id"], **memory)
            
            # Add edges (connections)
            for connection in connections:
                G.add_edge(
                    connection.source_memory_id,
                    connection.target_memory_id,
                    weight=connection.strength,
                    connection_type=connection.connection_type.value,
                    connection_id=connection.connection_id
                )
            
            # Calculate network metrics
            analysis = {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "density": nx.density(G),
                "is_connected": nx.is_connected(G),
                "connected_components": nx.number_connected_components(G),
                "average_clustering": nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0,
                "diameter": None,
                "average_path_length": None
            }
            
            # Calculate path metrics for connected graphs
            if analysis["is_connected"] and G.number_of_nodes() > 1:
                try:
                    analysis["diameter"] = nx.diameter(G)
                    analysis["average_path_length"] = nx.average_shortest_path_length(G)
                except:
                    pass
            
            # Centrality metrics for top nodes
            if G.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                
                # Get top central nodes
                top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                analysis["top_degree_central"] = [{"node": node, "centrality": cent} for node, cent in top_degree]
                analysis["top_betweenness_central"] = [{"node": node, "centrality": cent} for node, cent in top_betweenness]
            
            # Connection type distribution
            type_distribution = Counter(c.connection_type for c in connections)
            analysis["connection_type_distribution"] = dict(type_distribution)
            
            # Strength distribution
            strength_ranges = {
                "very_strong": len([c for c in connections if c.strength >= 0.9]),
                "strong": len([c for c in connections if 0.7 <= c.strength < 0.9]),
                "moderate": len([c for c in connections if 0.5 <= c.strength < 0.7]),
                "weak": len([c for c in connections if 0.3 <= c.strength < 0.5]),
                "very_weak": len([c for c in connections if c.strength < 0.3])
            }
            analysis["strength_distribution"] = strength_ranges
            
            return analysis
            
        except Exception as e:
            logger.error("Error in network analysis", error=str(e))
            return {"error": str(e)}
    
    def _calculate_quality_metrics(
        self,
        connections: List[MemoryConnection],
        total_pairs: int
    ) -> Dict[str, float]:
        """Calculate overall quality metrics for connections."""
        if not connections:
            return {
                "coverage": 0.0,
                "average_strength": 0.0,
                "average_confidence": 0.0,
                "average_quality": 0.0,
                "validation_rate": 0.0,
                "type_diversity": 0.0
            }
        
        # Calculate metrics
        avg_strength = sum(c.strength for c in connections) / len(connections)
        avg_confidence = sum(c.confidence for c in connections) / len(connections)
        avg_quality = sum(c.quality_score for c in connections) / len(connections)
        
        validated_count = len([c for c in connections if c.validation_status == ValidationStatus.VALIDATED])
        validation_rate = validated_count / len(connections)
        
        # Type diversity (Shannon entropy)
        type_counts = Counter(c.connection_type for c in connections)
        total_count = len(connections)
        type_diversity = -sum(
            (count / total_count) * math.log2(count / total_count)
            for count in type_counts.values()
        ) / math.log2(len(ConnectionType))  # Normalize by max entropy
        
        coverage = len(connections) / max(total_pairs, 1)
        
        return {
            "coverage": min(coverage, 1.0),
            "average_strength": avg_strength,
            "average_confidence": avg_confidence,
            "average_quality": avg_quality,
            "validation_rate": validation_rate,
            "type_diversity": type_diversity
        }
    
    def _generate_recommendations(
        self,
        connections: List[MemoryConnection],
        quality_metrics: Dict[str, float],
        strategies_used: List[str]
    ) -> List[str]:
        """Generate recommendations for improving connections."""
        recommendations = []
        
        # Coverage recommendations
        if quality_metrics.get("coverage", 0) < 0.1:
            recommendations.append("Consider adding more diverse memories to improve connection opportunities")
        
        # Quality recommendations
        if quality_metrics.get("average_quality", 0) < 0.6:
            recommendations.append("Focus on memories with richer content for better connection quality")
        
        # Validation recommendations
        if quality_metrics.get("validation_rate", 0) < 0.7:
            recommendations.append("Review and validate connections to improve reliability")
        
        # Diversity recommendations
        if quality_metrics.get("type_diversity", 0) < 0.5:
            recommendations.append("Expand memory content types to enable more diverse connections")
        
        # Strategy recommendations
        if len(strategies_used) < 4:
            recommendations.append("Enable more connection detection strategies for comprehensive analysis")
        
        # Connection count recommendations
        if len(connections) < 5:
            recommendations.append("Add more memories or lower confidence thresholds to increase connections")
        elif len(connections) > 100:
            recommendations.append("Consider raising confidence thresholds to focus on strongest connections")
        
        return recommendations
    
    def _deduplicate_connections(self, connections: List[MemoryConnection]) -> List[MemoryConnection]:
        """Remove duplicate connections, keeping the highest quality version."""
        connection_map = {}
        
        for connection in connections:
            # Create key for memory pair (order-independent)
            mem_pair = tuple(sorted([connection.source_memory_id, connection.target_memory_id]))
            key = f"{mem_pair[0]}-{mem_pair[1]}-{connection.connection_type.value}"
            
            if key not in connection_map:
                connection_map[key] = connection
            else:
                # Keep the connection with higher quality score
                existing = connection_map[key]
                if connection.quality_score > existing.quality_score:
                    connection_map[key] = connection
        
        return list(connection_map.values())
    
    async def _get_or_compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache or compute it."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            self.connection_stats['cache_hits'] += 1
            return self.embedding_cache[text_hash]
        
        try:
            embedding = await self.embedder.embed_text(text)
            
            # Cache the embedding
            self.embedding_cache[text_hash] = embedding
            self.connection_stats['cache_misses'] += 1
            
            # Limit cache size
            if len(self.embedding_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.embedding_cache.keys())[:100]
                for key in oldest_keys:
                    del self.embedding_cache[key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        if not self.nlp:
            # Fallback: simple capitalized word extraction
            words = text.split()
            entities = [word.strip('.,!?;:"()[]') for word in words if word[0].isupper() and len(word) > 2]
            return list(set(entities))
        
        try:
            doc = self.nlp(text)
            entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]]
            return list(set(entities))
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics/keywords from text."""
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [
            word.strip('.,!?;:"()[]') for word in words 
            if len(word) > 3 and word.lower() not in stop_words
        ]
        
        # Count frequency and return top keywords
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]
    
    def _humanize_time_diff(self, seconds: float) -> str:
        """Convert time difference to human-readable format."""
        if seconds < 3600:
            return f"{int(seconds // 60)} minutes"
        elif seconds < 86400:
            return f"{int(seconds // 3600)} hours"
        elif seconds < 604800:
            return f"{int(seconds // 86400)} days"
        elif seconds < 2592000:
            return f"{int(seconds // 604800)} weeks"
        else:
            return f"{int(seconds // 2592000)} months"
    
    async def _store_connections_in_graph(self, connections: List[MemoryConnection]) -> None:
        """Store connections in Neo4j graph database."""
        if not self.graph_client:
            return
        
        try:
            for connection in connections:
                # Create relationship in graph
                await self.graph_client.create_relationship(
                    source_id=connection.source_memory_id,
                    target_id=connection.target_memory_id,
                    relationship_type=connection.connection_type.value.upper(),
                    properties={
                        "strength": connection.strength,
                        "confidence": connection.confidence,
                        "quality_score": connection.quality_score,
                        "explanation": connection.explanation,
                        "evidence": connection.evidence,
                        "created_at": connection.created_at.isoformat(),
                        "validation_status": connection.validation_status.value
                    }
                )
            
            logger.info(f"Stored {len(connections)} connections in graph database")
            
        except Exception as e:
            logger.error("Error storing connections in graph", error=str(e))
    
    async def record_connection_feedback(
        self,
        connection_id: str,
        feedback_type: str,
        feedback_value: Union[float, bool, str],
        user_id: str
    ) -> None:
        """Record user feedback for connection quality."""
        feedback_record = {
            "connection_id": connection_id,
            "feedback_type": feedback_type,
            "feedback_value": feedback_value,
            "user_id": user_id,
            "timestamp": datetime.utcnow()
        }
        
        self.connection_stats['accuracy_feedback'].append(feedback_record)
        
        # Trigger model retraining if enough feedback collected
        if len(self.connection_stats['accuracy_feedback']) >= 100:
            await self._retrain_connection_classifier()
        
        logger.info(
            "Recorded connection feedback",
            connection_id=connection_id,
            feedback_type=feedback_type,
            user_id=user_id
        )
    
    async def _retrain_connection_classifier(self) -> None:
        """Retrain connection classification model using feedback."""
        try:
            feedback_data = self.connection_stats['accuracy_feedback']
            if len(feedback_data) < 50:
                return
            
            # This would implement actual model training
            # For now, just log the intent
            logger.info(
                "Would retrain connection classifier",
                feedback_samples=len(feedback_data)
            )
            
            # Clear old feedback to prevent memory growth
            self.connection_stats['accuracy_feedback'] = feedback_data[-500:]
            
        except Exception as e:
            logger.error("Error retraining connection classifier", error=str(e))
    
    def get_connection_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about connection generation."""
        return {
            "stats": self.connection_stats.copy(),
            "cache_efficiency": {
                "cache_hits": self.connection_stats['cache_hits'],
                "cache_misses": self.connection_stats['cache_misses'],
                "hit_rate": (
                    self.connection_stats['cache_hits'] / 
                    max(self.connection_stats['cache_hits'] + self.connection_stats['cache_misses'], 1)
                )
            },
            "model_status": {
                "classifier_trained": self.connection_classifier is not None,
                "feedback_count": len(self.connection_stats['accuracy_feedback'])
            },
            "performance": {
                "average_processing_time": self.connection_stats['average_processing_time'],
                "total_connections": self.connection_stats['total_connections_generated'],
                "validation_rate": (
                    self.connection_stats['validated_connections'] /
                    max(self.connection_stats['total_connections_generated'], 1)
                )
            }
        }


# Example usage
async def example_usage():
    """Example of using LocalConnectionGenerator."""
    from ...core.embeddings.embedder import Embedder
    from ...core.memory.manager import MemoryManager
    
    # Initialize components (mocked for example)
    embedder = Embedder()
    memory_manager = MemoryManager()
    
    # Create connection generator
    connector = LocalConnectionGenerator(embedder, memory_manager)
    
    # Create connection context
    context = ConnectionContext(
        user_id="user123",
        max_connections=20,
        min_confidence=0.3,
        validate_connections=True,
        include_weak_connections=False
    )
    
    # Generate connections
    result = await connector.generate_connections(context)
    
    print(f"Generated {len(result.connections)} connections")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Strategies used: {result.connection_strategies}")
    print(f"Network density: {result.network_analysis.get('density', 0):.3f}")
    
    for i, connection in enumerate(result.connections[:5], 1):
        print(f"\n{i}. {connection.source_memory_id} -> {connection.target_memory_id}")
        print(f"   Type: {connection.connection_type}")
        print(f"   Strength: {connection.strength:.3f} ({connection.strength_category})")
        print(f"   Explanation: {connection.explanation}")
        print(f"   Evidence: {connection.evidence[:2]}")
    
    # Record feedback
    if result.connections:
        await connector.record_connection_feedback(
            connection_id=result.connections[0].connection_id,
            feedback_type="accuracy",
            feedback_value=True,
            user_id="user123"
        )
    
    # Get analytics
    analytics = connector.get_connection_analytics()
    print(f"\nAnalytics: {analytics}")


if __name__ == "__main__":
    asyncio.run(example_usage())