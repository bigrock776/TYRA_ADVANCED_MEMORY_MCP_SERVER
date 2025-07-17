"""
Temporal Knowledge Evolution System for Dynamic Graph Learning.

This module provides comprehensive temporal knowledge evolution capabilities using local algorithms,
including concept drift detection, knowledge graph versioning, temporal embedding alignment,
and evolutionary pattern discovery. All processing is performed locally with zero external API calls.
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
import hashlib
import pickle
from pathlib import Path

# Graph and ML imports - all local
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, ks_2samp
import scipy.signal as signal

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings
from .reasoning_engine import ReasoningNode, ReasoningEdge, ReasoningPath
from .causal_inference import CausalClaim, CausalGraph

logger = structlog.get_logger(__name__)


class EvolutionType(str, Enum):
    """Types of knowledge evolution."""
    CONCEPT_EMERGENCE = "concept_emergence"          # New concepts appearing
    CONCEPT_DRIFT = "concept_drift"                  # Existing concepts changing
    RELATIONSHIP_CHANGE = "relationship_change"      # Relationship strength/type changes
    ENTITY_MERGE = "entity_merge"                    # Entities becoming equivalent
    ENTITY_SPLIT = "entity_split"                    # Entities becoming distinct
    KNOWLEDGE_DECAY = "knowledge_decay"              # Knowledge becoming outdated
    PATTERN_EMERGENCE = "pattern_emergence"          # New patterns emerging
    STRUCTURAL_CHANGE = "structural_change"          # Graph structure changes


class ChangeDetectionMethod(str, Enum):
    """Methods for detecting temporal changes."""
    STATISTICAL_DRIFT = "statistical_drift"         # Statistical change detection
    EMBEDDING_DRIFT = "embedding_drift"             # Embedding space changes
    GRAPH_STRUCTURE = "graph_structure"             # Graph topology changes
    CONCEPT_FREQUENCY = "concept_frequency"         # Concept occurrence frequency
    RELATIONSHIP_STRENGTH = "relationship_strength" # Edge weight changes
    CLUSTERING_EVOLUTION = "clustering_evolution"   # Cluster boundary changes
    ANOMALY_DETECTION = "anomaly_detection"         # Outlier-based detection


class TemporalGranularity(str, Enum):
    """Granularity levels for temporal analysis."""
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class TemporalSnapshot:
    """Represents a temporal snapshot of knowledge state."""
    timestamp: datetime
    entities: Dict[str, ReasoningNode]
    relationships: Dict[Tuple[str, str], ReasoningEdge]
    embeddings: Dict[str, np.ndarray]
    statistics: Dict[str, Any] = field(default_factory=dict)
    version_hash: str = ""
    
    def __post_init__(self):
        """Calculate version hash after initialization."""
        if not self.version_hash:
            self.version_hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate hash for this snapshot."""
        content = {
            'entities': sorted(self.entities.keys()),
            'relationships': sorted([f"{src}-{tgt}" for src, tgt in self.relationships.keys()]),
            'timestamp': self.timestamp.isoformat()
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class EvolutionEvent:
    """Represents a detected evolution event."""
    event_id: str
    evolution_type: EvolutionType
    detected_at: datetime
    affected_entities: List[str]
    affected_relationships: List[Tuple[str, str]]
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    before_snapshot: Optional[str] = None  # Hash of snapshot before change
    after_snapshot: Optional[str] = None   # Hash of snapshot after change
    description: str = ""
    
    def get_impact_score(self) -> float:
        """Calculate impact score for this evolution event."""
        entity_impact = len(self.affected_entities) / 10.0  # Normalize by expected size
        relationship_impact = len(self.affected_relationships) / 20.0
        
        # Weight by evolution type
        type_weights = {
            EvolutionType.CONCEPT_EMERGENCE: 0.8,
            EvolutionType.CONCEPT_DRIFT: 0.6,
            EvolutionType.RELATIONSHIP_CHANGE: 0.4,
            EvolutionType.ENTITY_MERGE: 0.7,
            EvolutionType.ENTITY_SPLIT: 0.7,
            EvolutionType.KNOWLEDGE_DECAY: 0.3,
            EvolutionType.PATTERN_EMERGENCE: 0.9,
            EvolutionType.STRUCTURAL_CHANGE: 1.0
        }
        
        type_weight = type_weights.get(self.evolution_type, 0.5)
        
        impact = (entity_impact + relationship_impact) * type_weight * self.confidence
        return min(1.0, impact)


@dataclass
class TemporalAlignment:
    """Represents alignment between temporal embeddings."""
    source_timestamp: datetime
    target_timestamp: datetime
    alignment_matrix: np.ndarray
    alignment_score: float
    aligned_entities: Dict[str, str]  # source -> target mappings
    drift_magnitude: float = 0.0


class ConceptDriftDetector:
    """
    Concept Drift Detection Engine.
    
    Detects when the meaning or relationships of concepts change over time.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.3,
        min_samples: int = 10
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        
        # Historical embeddings for comparison
        self.entity_embedding_history: Dict[str, List[Tuple[datetime, np.ndarray]]] = defaultdict(list)
        self.relationship_history: Dict[Tuple[str, str], List[Tuple[datetime, float]]] = defaultdict(list)
    
    async def detect_concept_drift(
        self,
        current_snapshot: TemporalSnapshot,
        historical_snapshots: List[TemporalSnapshot]
    ) -> List[EvolutionEvent]:
        """Detect concept drift between current and historical snapshots."""
        
        events = []
        
        if len(historical_snapshots) < 2:
            logger.warning("Insufficient historical data for drift detection")
            return events
        
        # Update embedding history
        await self._update_embedding_history(current_snapshot)
        
        # Detect embedding drift for each entity
        for entity_id, current_embedding in current_snapshot.embeddings.items():
            drift_event = await self._detect_entity_drift(entity_id, current_embedding, current_snapshot.timestamp)
            if drift_event:
                events.append(drift_event)
        
        # Detect relationship drift
        relationship_events = await self._detect_relationship_drift(current_snapshot, historical_snapshots)
        events.extend(relationship_events)
        
        # Detect structural drift
        structural_events = await self._detect_structural_drift(current_snapshot, historical_snapshots)
        events.extend(structural_events)
        
        return events
    
    async def _update_embedding_history(self, snapshot: TemporalSnapshot):
        """Update embedding history with current snapshot."""
        
        for entity_id, embedding in snapshot.embeddings.items():
            history = self.entity_embedding_history[entity_id]
            history.append((snapshot.timestamp, embedding))
            
            # Keep only recent history
            if len(history) > self.window_size:
                history.pop(0)
    
    async def _detect_entity_drift(
        self,
        entity_id: str,
        current_embedding: np.ndarray,
        timestamp: datetime
    ) -> Optional[EvolutionEvent]:
        """Detect drift for a specific entity."""
        
        history = self.entity_embedding_history[entity_id]
        
        if len(history) < self.min_samples:
            return None
        
        # Get recent embeddings for comparison
        recent_embeddings = [emb for ts, emb in history[-self.min_samples:]]
        
        if not recent_embeddings:
            return None
        
        # Calculate drift using multiple methods
        drift_scores = []
        
        # Method 1: Cosine similarity drift
        recent_similarities = []
        for hist_embedding in recent_embeddings:
            sim = cosine_similarity(
                current_embedding.reshape(1, -1),
                hist_embedding.reshape(1, -1)
            )[0, 0]
            recent_similarities.append(sim)
        
        avg_similarity = np.mean(recent_similarities)
        similarity_drift = 1.0 - avg_similarity
        drift_scores.append(similarity_drift)
        
        # Method 2: Statistical distance (KS test on embedding dimensions)
        try:
            current_flat = current_embedding.flatten()
            historical_flat = np.concatenate([emb.flatten() for emb in recent_embeddings])
            
            # Sample to avoid computational issues
            if len(historical_flat) > 1000:
                indices = np.random.choice(len(historical_flat), 1000, replace=False)
                historical_flat = historical_flat[indices]
            
            ks_stat, ks_p_value = ks_2samp(current_flat, historical_flat)
            statistical_drift = ks_stat
            drift_scores.append(statistical_drift)
            
        except Exception as e:
            logger.warning(f"Statistical drift calculation failed for {entity_id}: {e}")
            statistical_drift = 0.0
        
        # Method 3: Euclidean distance drift
        distances = []
        for hist_embedding in recent_embeddings:
            distance = np.linalg.norm(current_embedding - hist_embedding)
            distances.append(distance)
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Normalize distance (z-score)
        if std_distance > 0:
            distance_drift = min(1.0, abs(distances[-1] - mean_distance) / std_distance / 3.0)
        else:
            distance_drift = 0.0
        
        drift_scores.append(distance_drift)
        
        # Combine drift scores
        overall_drift = np.mean(drift_scores)
        
        if overall_drift > self.drift_threshold:
            return EvolutionEvent(
                event_id=f"drift_{entity_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                evolution_type=EvolutionType.CONCEPT_DRIFT,
                detected_at=timestamp,
                affected_entities=[entity_id],
                affected_relationships=[],
                confidence=min(1.0, overall_drift),
                evidence={
                    'similarity_drift': similarity_drift,
                    'statistical_drift': statistical_drift,
                    'distance_drift': distance_drift,
                    'overall_drift': overall_drift,
                    'avg_similarity': avg_similarity,
                    'sample_size': len(recent_embeddings)
                },
                description=f"Concept drift detected for entity '{entity_id}' with drift score {overall_drift:.3f}"
            )
        
        return None
    
    async def _detect_relationship_drift(
        self,
        current_snapshot: TemporalSnapshot,
        historical_snapshots: List[TemporalSnapshot]
    ) -> List[EvolutionEvent]:
        """Detect relationship strength/type changes."""
        
        events = []
        
        if len(historical_snapshots) < 2:
            return events
        
        # Compare with most recent historical snapshot
        previous_snapshot = historical_snapshots[-1]
        
        # Check for relationship strength changes
        for (src, tgt), current_edge in current_snapshot.relationships.items():
            if (src, tgt) in previous_snapshot.relationships:
                previous_edge = previous_snapshot.relationships[(src, tgt)]
                
                # Calculate strength change
                current_strength = current_edge.strength
                previous_strength = previous_edge.strength
                
                strength_change = abs(current_strength - previous_strength)
                
                if strength_change > 0.3:  # Significant change threshold
                    event = EvolutionEvent(
                        event_id=f"rel_change_{src}_{tgt}_{current_snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}",
                        evolution_type=EvolutionType.RELATIONSHIP_CHANGE,
                        detected_at=current_snapshot.timestamp,
                        affected_entities=[src, tgt],
                        affected_relationships=[(src, tgt)],
                        confidence=min(1.0, strength_change),
                        evidence={
                            'previous_strength': previous_strength,
                            'current_strength': current_strength,
                            'strength_change': strength_change,
                            'relationship_type': current_edge.relationship_type
                        },
                        before_snapshot=previous_snapshot.version_hash,
                        after_snapshot=current_snapshot.version_hash,
                        description=f"Relationship strength changed between '{src}' and '{tgt}' by {strength_change:.3f}"
                    )
                    events.append(event)
        
        # Check for new relationships (emergence)
        new_relationships = set(current_snapshot.relationships.keys()) - set(previous_snapshot.relationships.keys())
        if new_relationships:
            for src, tgt in new_relationships:
                event = EvolutionEvent(
                    event_id=f"rel_emerge_{src}_{tgt}_{current_snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    evolution_type=EvolutionType.PATTERN_EMERGENCE,
                    detected_at=current_snapshot.timestamp,
                    affected_entities=[src, tgt],
                    affected_relationships=[(src, tgt)],
                    confidence=0.8,
                    evidence={
                        'relationship_type': current_snapshot.relationships[(src, tgt)].relationship_type,
                        'relationship_strength': current_snapshot.relationships[(src, tgt)].strength
                    },
                    after_snapshot=current_snapshot.version_hash,
                    description=f"New relationship emerged between '{src}' and '{tgt}'"
                )
                events.append(event)
        
        # Check for disappeared relationships (decay)
        disappeared_relationships = set(previous_snapshot.relationships.keys()) - set(current_snapshot.relationships.keys())
        if disappeared_relationships:
            for src, tgt in disappeared_relationships:
                event = EvolutionEvent(
                    event_id=f"rel_decay_{src}_{tgt}_{current_snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    evolution_type=EvolutionType.KNOWLEDGE_DECAY,
                    detected_at=current_snapshot.timestamp,
                    affected_entities=[src, tgt],
                    affected_relationships=[(src, tgt)],
                    confidence=0.7,
                    evidence={
                        'previous_relationship_type': previous_snapshot.relationships[(src, tgt)].relationship_type,
                        'previous_strength': previous_snapshot.relationships[(src, tgt)].strength
                    },
                    before_snapshot=previous_snapshot.version_hash,
                    description=f"Relationship decayed between '{src}' and '{tgt}'"
                )
                events.append(event)
        
        return events
    
    async def _detect_structural_drift(
        self,
        current_snapshot: TemporalSnapshot,
        historical_snapshots: List[TemporalSnapshot]
    ) -> List[EvolutionEvent]:
        """Detect changes in graph structure."""
        
        events = []
        
        if len(historical_snapshots) < 2:
            return events
        
        previous_snapshot = historical_snapshots[-1]
        
        # Create graphs for comparison
        current_graph = nx.Graph()
        previous_graph = nx.Graph()
        
        # Add nodes and edges to current graph
        for entity_id in current_snapshot.entities:
            current_graph.add_node(entity_id)
        
        for (src, tgt), edge in current_snapshot.relationships.items():
            current_graph.add_edge(src, tgt, weight=edge.strength)
        
        # Add nodes and edges to previous graph
        for entity_id in previous_snapshot.entities:
            previous_graph.add_node(entity_id)
        
        for (src, tgt), edge in previous_snapshot.relationships.items():
            previous_graph.add_edge(src, tgt, weight=edge.strength)
        
        # Calculate structural metrics
        current_metrics = self._calculate_graph_metrics(current_graph)
        previous_metrics = self._calculate_graph_metrics(previous_graph)
        
        # Detect significant structural changes
        structural_changes = {}
        
        for metric, current_value in current_metrics.items():
            if metric in previous_metrics:
                previous_value = previous_metrics[metric]
                
                if previous_value != 0:
                    relative_change = abs(current_value - previous_value) / abs(previous_value)
                else:
                    relative_change = abs(current_value)
                
                if relative_change > 0.2:  # 20% change threshold
                    structural_changes[metric] = {
                        'previous': previous_value,
                        'current': current_value,
                        'relative_change': relative_change
                    }
        
        if structural_changes:
            confidence = min(1.0, np.mean([change['relative_change'] for change in structural_changes.values()]))
            
            event = EvolutionEvent(
                event_id=f"struct_change_{current_snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}",
                evolution_type=EvolutionType.STRUCTURAL_CHANGE,
                detected_at=current_snapshot.timestamp,
                affected_entities=list(current_snapshot.entities.keys()),
                affected_relationships=list(current_snapshot.relationships.keys()),
                confidence=confidence,
                evidence={
                    'structural_changes': structural_changes,
                    'current_metrics': current_metrics,
                    'previous_metrics': previous_metrics
                },
                before_snapshot=previous_snapshot.version_hash,
                after_snapshot=current_snapshot.version_hash,
                description=f"Significant structural changes detected in graph topology"
            )
            events.append(event)
        
        return events
    
    def _calculate_graph_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate structural metrics for a graph."""
        
        if len(graph.nodes()) == 0:
            return {}
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['num_nodes'] = graph.number_of_nodes()
            metrics['num_edges'] = graph.number_of_edges()
            metrics['density'] = nx.density(graph)
            
            # Connectivity metrics
            if nx.is_connected(graph):
                metrics['average_path_length'] = nx.average_shortest_path_length(graph)
                metrics['diameter'] = nx.diameter(graph)
            else:
                metrics['num_components'] = nx.number_connected_components(graph)
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                if len(subgraph.nodes()) > 1:
                    metrics['largest_component_avg_path'] = nx.average_shortest_path_length(subgraph)
            
            # Centrality metrics (averages)
            if len(graph.nodes()) > 1:
                degree_centrality = nx.degree_centrality(graph)
                betweenness_centrality = nx.betweenness_centrality(graph)
                closeness_centrality = nx.closeness_centrality(graph)
                
                metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
                metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
                metrics['avg_closeness_centrality'] = np.mean(list(closeness_centrality.values()))
            
            # Clustering coefficient
            metrics['average_clustering'] = nx.average_clustering(graph)
            
        except Exception as e:
            logger.warning(f"Graph metrics calculation failed: {e}")
        
        return metrics


class TemporalEmbeddingAligner:
    """
    Temporal Embedding Alignment Engine.
    
    Aligns embeddings across time to track concept evolution and maintain consistency.
    """
    
    def __init__(self, alignment_method: str = "procrustes"):
        self.alignment_method = alignment_method
        self.reference_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.alignment_history: List[TemporalAlignment] = []
    
    async def align_temporal_embeddings(
        self,
        source_embeddings: Dict[str, np.ndarray],
        target_embeddings: Dict[str, np.ndarray],
        source_timestamp: datetime,
        target_timestamp: datetime
    ) -> TemporalAlignment:
        """Align embeddings between two time periods."""
        
        # Find common entities
        common_entities = set(source_embeddings.keys()).intersection(set(target_embeddings.keys()))
        
        if len(common_entities) < 3:
            logger.warning("Too few common entities for reliable alignment")
            return TemporalAlignment(
                source_timestamp=source_timestamp,
                target_timestamp=target_timestamp,
                alignment_matrix=np.eye(list(source_embeddings.values())[0].shape[0]),
                alignment_score=0.0,
                aligned_entities={},
                drift_magnitude=1.0
            )
        
        # Extract embeddings for common entities
        source_matrix = np.array([source_embeddings[entity] for entity in common_entities])
        target_matrix = np.array([target_embeddings[entity] for entity in common_entities])
        
        # Perform alignment
        if self.alignment_method == "procrustes":
            alignment_result = await self._procrustes_alignment(source_matrix, target_matrix)
        elif self.alignment_method == "linear":
            alignment_result = await self._linear_alignment(source_matrix, target_matrix)
        else:
            alignment_result = await self._identity_alignment(source_matrix, target_matrix)
        
        # Calculate aligned entity mappings
        aligned_entities = {}
        aligned_source = source_matrix @ alignment_result['transformation_matrix']
        
        for i, entity in enumerate(common_entities):
            aligned_entities[entity] = entity  # Same entity across time
        
        # Calculate overall drift magnitude
        drift_magnitude = await self._calculate_drift_magnitude(aligned_source, target_matrix)
        
        alignment = TemporalAlignment(
            source_timestamp=source_timestamp,
            target_timestamp=target_timestamp,
            alignment_matrix=alignment_result['transformation_matrix'],
            alignment_score=alignment_result['alignment_score'],
            aligned_entities=aligned_entities,
            drift_magnitude=drift_magnitude
        )
        
        self.alignment_history.append(alignment)
        
        return alignment
    
    async def _procrustes_alignment(
        self,
        source_matrix: np.ndarray,
        target_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Perform Procrustes alignment between embedding matrices."""
        
        try:
            # Center the matrices
            source_centered = source_matrix - np.mean(source_matrix, axis=0)
            target_centered = target_matrix - np.mean(target_matrix, axis=0)
            
            # SVD for Procrustes alignment
            U, s, Vt = np.linalg.svd(target_centered.T @ source_centered)
            
            # Optimal rotation matrix
            R = U @ Vt
            
            # Apply alignment
            aligned_source = source_centered @ R
            
            # Calculate alignment score (negative of Frobenius norm of difference)
            alignment_score = 1.0 / (1.0 + np.linalg.norm(aligned_source - target_centered, 'fro'))
            
            return {
                'transformation_matrix': R,
                'alignment_score': alignment_score,
                'method': 'procrustes'
            }
            
        except Exception as e:
            logger.error(f"Procrustes alignment failed: {e}")
            return await self._identity_alignment(source_matrix, target_matrix)
    
    async def _linear_alignment(
        self,
        source_matrix: np.ndarray,
        target_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Perform linear transformation alignment."""
        
        try:
            # Solve for transformation matrix: target = source @ T
            # T = (source^T @ source)^-1 @ source^T @ target
            
            source_t = source_matrix.T
            gram_matrix = source_t @ source_matrix
            
            # Add regularization to prevent singular matrix
            reg_param = 1e-6
            gram_matrix += reg_param * np.eye(gram_matrix.shape[0])
            
            # Solve for transformation
            T = np.linalg.solve(gram_matrix, source_t @ target_matrix)
            
            # Apply transformation
            aligned_source = source_matrix @ T
            
            # Calculate alignment score
            residual = np.linalg.norm(aligned_source - target_matrix, 'fro')
            alignment_score = 1.0 / (1.0 + residual)
            
            return {
                'transformation_matrix': T,
                'alignment_score': alignment_score,
                'method': 'linear'
            }
            
        except Exception as e:
            logger.error(f"Linear alignment failed: {e}")
            return await self._identity_alignment(source_matrix, target_matrix)
    
    async def _identity_alignment(
        self,
        source_matrix: np.ndarray,
        target_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Fallback identity alignment."""
        
        # Identity transformation
        identity_matrix = np.eye(source_matrix.shape[1])
        
        # Calculate alignment score
        residual = np.linalg.norm(source_matrix - target_matrix, 'fro')
        alignment_score = 1.0 / (1.0 + residual)
        
        return {
            'transformation_matrix': identity_matrix,
            'alignment_score': alignment_score,
            'method': 'identity'
        }
    
    async def _calculate_drift_magnitude(
        self,
        aligned_source: np.ndarray,
        target_matrix: np.ndarray
    ) -> float:
        """Calculate magnitude of drift between aligned embeddings."""
        
        # Calculate per-entity drift
        entity_drifts = []
        
        for i in range(len(aligned_source)):
            source_vec = aligned_source[i]
            target_vec = target_matrix[i]
            
            # Cosine distance as drift measure
            similarity = cosine_similarity(
                source_vec.reshape(1, -1),
                target_vec.reshape(1, -1)
            )[0, 0]
            
            drift = 1.0 - similarity
            entity_drifts.append(drift)
        
        # Return average drift
        return np.mean(entity_drifts) if entity_drifts else 0.0
    
    async def get_temporal_consistency_score(
        self,
        embeddings_timeline: List[Tuple[datetime, Dict[str, np.ndarray]]]
    ) -> float:
        """Calculate overall temporal consistency score."""
        
        if len(embeddings_timeline) < 2:
            return 1.0
        
        consistency_scores = []
        
        for i in range(1, len(embeddings_timeline)):
            prev_timestamp, prev_embeddings = embeddings_timeline[i-1]
            curr_timestamp, curr_embeddings = embeddings_timeline[i]
            
            alignment = await self.align_temporal_embeddings(
                prev_embeddings, curr_embeddings, prev_timestamp, curr_timestamp
            )
            
            # Higher alignment score = higher consistency
            consistency_scores.append(alignment.alignment_score)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0


class EvolutionPatternDetector:
    """
    Evolution Pattern Detection Engine.
    
    Detects recurring patterns in knowledge evolution and predicts future changes.
    """
    
    def __init__(self):
        self.pattern_history: List[Dict[str, Any]] = []
        self.clustering_model = DBSCAN(eps=0.3, min_samples=3)
    
    async def detect_evolution_patterns(
        self,
        evolution_events: List[EvolutionEvent],
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Detect patterns in evolution events."""
        
        if len(evolution_events) < 5:
            logger.warning("Insufficient evolution events for pattern detection")
            return []
        
        # Filter events within lookback period
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        recent_events = [e for e in evolution_events if e.detected_at >= cutoff_date]
        
        if len(recent_events) < 3:
            return []
        
        # Extract features for clustering
        event_features = await self._extract_event_features(recent_events)
        
        if len(event_features) == 0:
            return []
        
        # Perform clustering to find patterns
        pattern_clusters = await self._cluster_events(event_features, recent_events)
        
        # Analyze temporal patterns
        temporal_patterns = await self._analyze_temporal_patterns(recent_events)
        
        # Combine results
        all_patterns = pattern_clusters + temporal_patterns
        
        return all_patterns
    
    async def _extract_event_features(self, events: List[EvolutionEvent]) -> np.ndarray:
        """Extract numerical features from evolution events."""
        
        features = []
        
        for event in events:
            feature_vector = [
                # Type encoding (one-hot)
                1.0 if event.evolution_type == EvolutionType.CONCEPT_EMERGENCE else 0.0,
                1.0 if event.evolution_type == EvolutionType.CONCEPT_DRIFT else 0.0,
                1.0 if event.evolution_type == EvolutionType.RELATIONSHIP_CHANGE else 0.0,
                1.0 if event.evolution_type == EvolutionType.ENTITY_MERGE else 0.0,
                1.0 if event.evolution_type == EvolutionType.ENTITY_SPLIT else 0.0,
                1.0 if event.evolution_type == EvolutionType.KNOWLEDGE_DECAY else 0.0,
                1.0 if event.evolution_type == EvolutionType.PATTERN_EMERGENCE else 0.0,
                1.0 if event.evolution_type == EvolutionType.STRUCTURAL_CHANGE else 0.0,
                
                # Numerical features
                event.confidence,
                len(event.affected_entities),
                len(event.affected_relationships),
                event.get_impact_score(),
                
                # Temporal features
                event.detected_at.hour / 24.0,  # Hour of day
                event.detected_at.weekday() / 6.0,  # Day of week
                event.detected_at.day / 31.0,  # Day of month
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    async def _cluster_events(
        self,
        event_features: np.ndarray,
        events: List[EvolutionEvent]
    ) -> List[Dict[str, Any]]:
        """Cluster events to find similar patterns."""
        
        patterns = []
        
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(event_features)
            
            # Perform clustering
            cluster_labels = self.clustering_model.fit_predict(features_scaled)
            
            # Analyze each cluster
            unique_labels = set(cluster_labels)
            unique_labels.discard(-1)  # Remove noise cluster
            
            for cluster_id in unique_labels:
                cluster_events = [events[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_events) < 2:
                    continue
                
                # Analyze cluster characteristics
                cluster_pattern = await self._analyze_cluster_pattern(cluster_events, cluster_id)
                patterns.append(cluster_pattern)
        
        except Exception as e:
            logger.error(f"Event clustering failed: {e}")
        
        return patterns
    
    async def _analyze_cluster_pattern(
        self,
        cluster_events: List[EvolutionEvent],
        cluster_id: int
    ) -> Dict[str, Any]:
        """Analyze the pattern within a cluster of events."""
        
        # Calculate cluster statistics
        evolution_types = [event.evolution_type for event in cluster_events]
        type_counts = Counter(evolution_types)
        
        avg_confidence = np.mean([event.confidence for event in cluster_events])
        avg_impact = np.mean([event.get_impact_score() for event in cluster_events])
        
        # Find common entities
        all_entities = []
        for event in cluster_events:
            all_entities.extend(event.affected_entities)
        
        entity_counts = Counter(all_entities)
        common_entities = [entity for entity, count in entity_counts.most_common(5)]
        
        # Temporal analysis
        timestamps = [event.detected_at for event in cluster_events]
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Hours
            time_diffs.append(diff)
        
        avg_time_interval = np.mean(time_diffs) if time_diffs else 0.0
        
        pattern = {
            'pattern_id': f"cluster_{cluster_id}",
            'pattern_type': 'behavioral_cluster',
            'event_count': len(cluster_events),
            'dominant_evolution_type': type_counts.most_common(1)[0][0].value,
            'evolution_type_distribution': {et.value: count for et, count in type_counts.items()},
            'average_confidence': avg_confidence,
            'average_impact': avg_impact,
            'common_entities': common_entities,
            'average_time_interval_hours': avg_time_interval,
            'first_occurrence': min(timestamps),
            'last_occurrence': max(timestamps),
            'description': f"Cluster of {len(cluster_events)} similar evolution events"
        }
        
        return pattern
    
    async def _analyze_temporal_patterns(self, events: List[EvolutionEvent]) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in evolution events."""
        
        patterns = []
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e.detected_at)
        
        # Detect periodic patterns
        timestamps = [event.detected_at for event in sorted_events]
        
        # Convert to time series (events per hour)
        if len(timestamps) < 10:
            return patterns
        
        # Create hourly time series
        start_time = timestamps[0].replace(minute=0, second=0, microsecond=0)
        end_time = timestamps[-1].replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        hourly_counts = defaultdict(int)
        current_time = start_time
        
        while current_time <= end_time:
            hourly_counts[current_time] = 0
            current_time += timedelta(hours=1)
        
        # Count events per hour
        for timestamp in timestamps:
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        # Convert to arrays for analysis
        time_points = sorted(hourly_counts.keys())
        counts = [hourly_counts[tp] for tp in time_points]
        
        if len(counts) < 24:  # Need at least 24 hours of data
            return patterns
        
        # Detect periodicity using FFT
        try:
            fft_result = np.fft.fft(counts)
            frequencies = np.fft.fftfreq(len(counts))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_result)
            dominant_freq_idx = np.argsort(power_spectrum)[-3:]  # Top 3 frequencies
            
            for idx in dominant_freq_idx:
                if idx == 0:  # Skip DC component
                    continue
                
                frequency = frequencies[idx]
                period_hours = 1.0 / abs(frequency) if frequency != 0 else float('inf')
                
                if 12 <= period_hours <= 168:  # Between 12 hours and 1 week
                    pattern = {
                        'pattern_id': f"temporal_period_{period_hours:.1f}h",
                        'pattern_type': 'temporal_periodicity',
                        'period_hours': period_hours,
                        'frequency': frequency,
                        'power': power_spectrum[idx],
                        'description': f"Periodic pattern with {period_hours:.1f} hour cycle"
                    }
                    patterns.append(pattern)
        
        except Exception as e:
            logger.warning(f"Temporal pattern analysis failed: {e}")
        
        # Detect burst patterns
        burst_pattern = await self._detect_burst_patterns(sorted_events)
        if burst_pattern:
            patterns.append(burst_pattern)
        
        return patterns
    
    async def _detect_burst_patterns(self, sorted_events: List[EvolutionEvent]) -> Optional[Dict[str, Any]]:
        """Detect burst patterns in evolution events."""
        
        if len(sorted_events) < 5:
            return None
        
        # Calculate inter-event times
        inter_event_times = []
        for i in range(1, len(sorted_events)):
            time_diff = (sorted_events[i].detected_at - sorted_events[i-1].detected_at).total_seconds() / 60  # Minutes
            inter_event_times.append(time_diff)
        
        # Detect bursts (sequences of events with short inter-event times)
        burst_threshold = np.percentile(inter_event_times, 25)  # Bottom quartile
        
        bursts = []
        current_burst = []
        
        for i, time_diff in enumerate(inter_event_times):
            if time_diff <= burst_threshold:
                if not current_burst:
                    current_burst = [sorted_events[i]]  # Start of burst
                current_burst.append(sorted_events[i+1])
            else:
                if len(current_burst) >= 3:  # Minimum burst size
                    bursts.append(current_burst)
                current_burst = []
        
        # Add final burst if exists
        if len(current_burst) >= 3:
            bursts.append(current_burst)
        
        if not bursts:
            return None
        
        # Analyze burst characteristics
        total_burst_events = sum(len(burst) for burst in bursts)
        avg_burst_size = np.mean([len(burst) for burst in bursts])
        
        return {
            'pattern_id': 'burst_pattern',
            'pattern_type': 'burst_activity',
            'num_bursts': len(bursts),
            'total_burst_events': total_burst_events,
            'average_burst_size': avg_burst_size,
            'burst_threshold_minutes': burst_threshold,
            'burst_frequency': len(bursts) / len(sorted_events),
            'description': f"Detected {len(bursts)} burst patterns with average size {avg_burst_size:.1f}"
        }


class TemporalKnowledgeEvolutionEngine:
    """
    Main Temporal Knowledge Evolution Engine.
    
    Coordinates concept drift detection, temporal alignment, and pattern analysis.
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        cache_ttl: int = 7200,
        snapshot_storage_path: Optional[str] = None
    ):
        self.redis_cache = redis_cache
        self.cache_ttl = cache_ttl
        
        # Snapshot storage
        self.snapshot_storage_path = Path(snapshot_storage_path) if snapshot_storage_path else Path("./temporal_snapshots")
        self.snapshot_storage_path.mkdir(exist_ok=True)
        
        # Components
        self.drift_detector = ConceptDriftDetector()
        self.embedding_aligner = TemporalEmbeddingAligner()
        self.pattern_detector = EvolutionPatternDetector()
        
        # State management
        self.snapshots: List[TemporalSnapshot] = []
        self.evolution_events: List[EvolutionEvent] = []
        
        # Statistics
        self.stats = {
            'total_snapshots': 0,
            'evolution_events_detected': 0,
            'drift_events': 0,
            'pattern_events': 0,
            'alignment_operations': 0
        }
    
    async def create_temporal_snapshot(
        self,
        memories: List[Memory],
        timestamp: Optional[datetime] = None
    ) -> TemporalSnapshot:
        """Create a temporal snapshot from current memory state."""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Extract entities and relationships from memories
        entities = {}
        relationships = {}
        embeddings = {}
        
        # Process memories to extract graph structure
        entity_mentions = defaultdict(list)
        relationship_patterns = []
        
        for memory in memories:
            # Extract entities (simplified - in practice use NER)
            content_lower = memory.content.lower()
            words = content_lower.split()
            
            # Simple entity extraction based on capitalization and common patterns
            import re
            entities_in_text = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', memory.content)
            
            for entity in entities_in_text:
                entity_mentions[entity].append(memory)
                
                # Create reasoning node if not exists
                if entity not in entities:
                    entities[entity] = ReasoningNode(
                        node_id=entity,
                        entity_type="extracted_entity",
                        properties={
                            'first_mentioned': memory.created_at.isoformat(),
                            'mention_count': 0
                        },
                        created_at=memory.created_at
                    )
                
                entities[entity].properties['mention_count'] = entities[entity].properties.get('mention_count', 0) + 1
                
                # Generate simple embedding (in practice, use actual embedding model)
                entity_text = entity.lower()
                simple_embedding = np.array([hash(entity_text) % 256 for _ in range(128)], dtype=np.float32)
                simple_embedding = simple_embedding / np.linalg.norm(simple_embedding)
                embeddings[entity] = simple_embedding
        
        # Create simple relationships based on co-occurrence
        entity_list = list(entities.keys())
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                # Check co-occurrence
                entity1_memories = set(entity_mentions[entity1])
                entity2_memories = set(entity_mentions[entity2])
                
                shared_memories = entity1_memories.intersection(entity2_memories)
                
                if len(shared_memories) > 0:
                    # Calculate relationship strength based on co-occurrence
                    strength = len(shared_memories) / min(len(entity1_memories), len(entity2_memories))
                    
                    relationship = ReasoningEdge(
                        source_id=entity1,
                        target_id=entity2,
                        relationship_type="co_occurrence",
                        strength=strength,
                        confidence=min(1.0, strength * 2),
                        evidence=[f"Co-occurred in {len(shared_memories)} memories"]
                    )
                    
                    relationships[(entity1, entity2)] = relationship
        
        # Calculate statistics
        statistics = {
            'entity_count': len(entities),
            'relationship_count': len(relationships),
            'avg_entity_mentions': np.mean([e.properties.get('mention_count', 0) for e in entities.values()]) if entities else 0,
            'memory_count': len(memories),
            'timestamp': timestamp.isoformat()
        }
        
        snapshot = TemporalSnapshot(
            timestamp=timestamp,
            entities=entities,
            relationships=relationships,
            embeddings=embeddings,
            statistics=statistics
        )
        
        # Store snapshot
        await self._store_snapshot(snapshot)
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots in memory
        if len(self.snapshots) > 100:
            self.snapshots.pop(0)
        
        self.stats['total_snapshots'] += 1
        
        logger.info(
            f"Created temporal snapshot",
            timestamp=timestamp,
            entities=len(entities),
            relationships=len(relationships),
            version_hash=snapshot.version_hash
        )
        
        return snapshot
    
    async def analyze_temporal_evolution(
        self,
        current_memories: List[Memory],
        lookback_days: int = 7
    ) -> List[EvolutionEvent]:
        """Analyze temporal evolution in knowledge."""
        
        # Create current snapshot
        current_snapshot = await self.create_temporal_snapshot(current_memories)
        
        # Get historical snapshots for comparison
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        historical_snapshots = [
            s for s in self.snapshots 
            if s.timestamp >= cutoff_date and s.timestamp < current_snapshot.timestamp
        ]
        
        if len(historical_snapshots) < 1:
            logger.warning("No historical snapshots available for evolution analysis")
            return []
        
        # Detect concept drift
        drift_events = await self.drift_detector.detect_concept_drift(
            current_snapshot, historical_snapshots
        )
        
        # Update evolution events
        new_events = drift_events
        self.evolution_events.extend(new_events)
        
        # Keep only recent events
        recent_cutoff = datetime.utcnow() - timedelta(days=30)
        self.evolution_events = [
            e for e in self.evolution_events 
            if e.detected_at >= recent_cutoff
        ]
        
        # Update statistics
        self.stats['evolution_events_detected'] += len(new_events)
        self.stats['drift_events'] += len([e for e in new_events if e.evolution_type == EvolutionType.CONCEPT_DRIFT])
        
        logger.info(
            f"Temporal evolution analysis completed",
            new_events=len(new_events),
            total_historical_snapshots=len(historical_snapshots)
        )
        
        return new_events
    
    async def align_temporal_embeddings(
        self,
        source_snapshot: TemporalSnapshot,
        target_snapshot: TemporalSnapshot
    ) -> TemporalAlignment:
        """Align embeddings between two temporal snapshots."""
        
        alignment = await self.embedding_aligner.align_temporal_embeddings(
            source_snapshot.embeddings,
            target_snapshot.embeddings,
            source_snapshot.timestamp,
            target_snapshot.timestamp
        )
        
        self.stats['alignment_operations'] += 1
        
        logger.info(
            f"Temporal embedding alignment completed",
            source_timestamp=source_snapshot.timestamp,
            target_timestamp=target_snapshot.timestamp,
            alignment_score=alignment.alignment_score,
            drift_magnitude=alignment.drift_magnitude
        )
        
        return alignment
    
    async def detect_evolution_patterns(
        self,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Detect patterns in evolution events."""
        
        patterns = await self.pattern_detector.detect_evolution_patterns(
            self.evolution_events, lookback_days
        )
        
        pattern_events = len([e for e in self.evolution_events if e.evolution_type == EvolutionType.PATTERN_EMERGENCE])
        self.stats['pattern_events'] = pattern_events
        
        logger.info(
            f"Evolution pattern detection completed",
            patterns_found=len(patterns),
            lookback_days=lookback_days
        )
        
        return patterns
    
    async def predict_future_evolution(
        self,
        prediction_horizon_days: int = 7
    ) -> List[Dict[str, Any]]:
        """Predict future evolution events based on historical patterns."""
        
        predictions = []
        
        if len(self.evolution_events) < 10:
            logger.warning("Insufficient historical data for evolution prediction")
            return predictions
        
        # Analyze recent trends
        recent_events = [
            e for e in self.evolution_events 
            if e.detected_at >= datetime.utcnow() - timedelta(days=30)
        ]
        
        if not recent_events:
            return predictions
        
        # Simple trend-based prediction
        event_types = [e.evolution_type for e in recent_events]
        type_counts = Counter(event_types)
        
        # Calculate event rates
        time_span = (recent_events[-1].detected_at - recent_events[0].detected_at).total_seconds() / (24 * 3600)  # Days
        
        if time_span <= 0:
            return predictions
        
        for evolution_type, count in type_counts.items():
            rate_per_day = count / time_span
            
            # Predict future occurrences
            predicted_count = rate_per_day * prediction_horizon_days
            
            if predicted_count >= 0.5:  # At least 50% chance of occurrence
                prediction = {
                    'prediction_id': f"pred_{evolution_type.value}_{datetime.utcnow().strftime('%Y%m%d')}",
                    'evolution_type': evolution_type.value,
                    'predicted_occurrences': predicted_count,
                    'confidence': min(0.8, count / 10.0),  # Confidence based on historical data
                    'prediction_horizon_days': prediction_horizon_days,
                    'based_on_events': count,
                    'historical_rate_per_day': rate_per_day,
                    'description': f"Predicted {predicted_count:.1f} {evolution_type.value} events in next {prediction_horizon_days} days"
                }
                predictions.append(prediction)
        
        logger.info(
            f"Future evolution prediction completed",
            predictions=len(predictions),
            horizon_days=prediction_horizon_days
        )
        
        return predictions
    
    async def _store_snapshot(self, snapshot: TemporalSnapshot):
        """Store temporal snapshot to disk."""
        
        try:
            filename = f"snapshot_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}_{snapshot.version_hash}.pkl"
            filepath = self.snapshot_storage_path / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(snapshot, f)
            
            # Cache in Redis if available
            if self.redis_cache:
                cache_key = f"temporal_snapshot:{snapshot.version_hash}"
                await self.redis_cache.set(
                    cache_key,
                    snapshot.__dict__,
                    ttl=self.cache_ttl
                )
            
        except Exception as e:
            logger.error(f"Failed to store temporal snapshot: {e}")
    
    async def load_historical_snapshots(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[TemporalSnapshot]:
        """Load historical snapshots from storage."""
        
        loaded_snapshots = []
        
        try:
            # Load from disk
            for filepath in self.snapshot_storage_path.glob("snapshot_*.pkl"):
                try:
                    with open(filepath, 'rb') as f:
                        snapshot = pickle.load(f)
                    
                    if start_date <= snapshot.timestamp <= end_date:
                        loaded_snapshots.append(snapshot)
                        
                except Exception as e:
                    logger.warning(f"Failed to load snapshot {filepath}: {e}")
            
            # Sort by timestamp
            loaded_snapshots.sort(key=lambda s: s.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to load historical snapshots: {e}")
        
        return loaded_snapshots
    
    async def get_evolution_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_events = [e for e in self.evolution_events if e.detected_at >= cutoff_date]
        
        # Event type distribution
        event_types = [e.evolution_type for e in recent_events]
        type_distribution = dict(Counter(event_types))
        
        # Impact analysis
        impact_scores = [e.get_impact_score() for e in recent_events]
        avg_impact = np.mean(impact_scores) if impact_scores else 0.0
        
        # Affected entities analysis
        all_affected_entities = []
        for event in recent_events:
            all_affected_entities.extend(event.affected_entities)
        
        entity_frequency = dict(Counter(all_affected_entities))
        
        # Confidence distribution
        confidence_scores = [e.confidence for e in recent_events]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Temporal patterns
        if recent_events:
            time_diffs = []
            sorted_events = sorted(recent_events, key=lambda e: e.detected_at)
            
            for i in range(1, len(sorted_events)):
                diff = (sorted_events[i].detected_at - sorted_events[i-1].detected_at).total_seconds() / 3600
                time_diffs.append(diff)
            
            avg_time_between_events = np.mean(time_diffs) if time_diffs else 0.0
        else:
            avg_time_between_events = 0.0
        
        return {
            'summary_period_days': days,
            'total_events': len(recent_events),
            'event_type_distribution': {et.value: count for et, count in type_distribution.items()},
            'average_impact_score': avg_impact,
            'average_confidence': avg_confidence,
            'most_affected_entities': dict(Counter(all_affected_entities).most_common(10)),
            'average_time_between_events_hours': avg_time_between_events,
            'high_impact_events': len([e for e in recent_events if e.get_impact_score() > 0.7]),
            'high_confidence_events': len([e for e in recent_events if e.confidence > 0.8]),
            'engine_statistics': self.stats.copy()
        }
    
    async def export_evolution_timeline(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Export evolution timeline for visualization."""
        
        # Get events in date range
        timeline_events = [
            e for e in self.evolution_events 
            if start_date <= e.detected_at <= end_date
        ]
        
        # Get snapshots in date range
        timeline_snapshots = [
            s for s in self.snapshots 
            if start_date <= s.timestamp <= end_date
        ]
        
        # Create timeline structure
        timeline = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'events': [
                {
                    'timestamp': event.detected_at.isoformat(),
                    'type': event.evolution_type.value,
                    'confidence': event.confidence,
                    'impact_score': event.get_impact_score(),
                    'affected_entities': event.affected_entities,
                    'description': event.description
                }
                for event in sorted(timeline_events, key=lambda e: e.detected_at)
            ],
            'snapshots': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'version_hash': snapshot.version_hash,
                    'entity_count': len(snapshot.entities),
                    'relationship_count': len(snapshot.relationships)
                }
                for snapshot in sorted(timeline_snapshots, key=lambda s: s.timestamp)
            ]
        }
        
        return timeline
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        
        return {
            'temporal_evolution_stats': self.stats.copy(),
            'current_state': {
                'snapshots_in_memory': len(self.snapshots),
                'evolution_events_tracked': len(self.evolution_events),
                'alignment_history_size': len(self.embedding_aligner.alignment_history),
                'pattern_history_size': len(self.pattern_detector.pattern_history)
            },
            'drift_detector_config': {
                'window_size': self.drift_detector.window_size,
                'drift_threshold': self.drift_detector.drift_threshold,
                'min_samples': self.drift_detector.min_samples
            }
        }