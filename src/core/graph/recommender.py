"""
Graph-Based Recommendation Engine for Intelligent Memory Suggestions.

This module provides comprehensive recommendation capabilities using local graph algorithms,
including collaborative filtering, content-based filtering, knowledge graph embeddings,
and hybrid recommendation strategies. All processing is performed locally with zero external API calls.
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
import heapq

# Graph and ML imports - all local
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings
from .reasoning_engine import ReasoningNode, ReasoningEdge, ReasoningPath
from .causal_inference import CausalClaim, CausalGraph

logger = structlog.get_logger(__name__)


class RecommendationType(str, Enum):
    """Types of recommendations."""
    MEMORY_SUGGESTION = "memory_suggestion"          # Suggest relevant memories
    KNOWLEDGE_GAP = "knowledge_gap"                  # Identify knowledge gaps
    RELATED_CONCEPT = "related_concept"              # Suggest related concepts
    CAUSAL_EXPLORATION = "causal_exploration"        # Suggest causal relationships to explore
    TEMPORAL_PATTERN = "temporal_pattern"            # Suggest temporal patterns
    ENTITY_CONNECTION = "entity_connection"          # Suggest entity connections
    LEARNING_PATH = "learning_path"                  # Suggest learning sequences
    ANOMALY_INVESTIGATION = "anomaly_investigation"  # Suggest anomalies to investigate


class RecommendationStrategy(str, Enum):
    """Strategies for generating recommendations."""
    COLLABORATIVE_FILTERING = "collaborative_filtering"  # User-item collaborative filtering
    CONTENT_BASED = "content_based"                      # Content similarity based
    GRAPH_EMBEDDING = "graph_embedding"                  # Graph embedding based
    RANDOM_WALK = "random_walk"                         # Random walk exploration
    PAGERANK_BASED = "pagerank_based"                   # PageRank importance based
    HYBRID_FUSION = "hybrid_fusion"                     # Combine multiple strategies
    DIVERSITY_PROMOTING = "diversity_promoting"         # Promote diverse recommendations
    NOVELTY_SEEKING = "novelty_seeking"                # Seek novel/unexpected items


class ConfidenceLevel(str, Enum):
    """Confidence levels for recommendations."""
    VERY_HIGH = "very_high"      # 90%+ confidence
    HIGH = "high"                # 75-90% confidence
    MEDIUM = "medium"            # 50-75% confidence
    LOW = "low"                  # 25-50% confidence
    VERY_LOW = "very_low"        # <25% confidence


@dataclass
class RecommendationItem:
    """Represents a single recommendation item."""
    item_id: str
    item_type: str  # 'memory', 'entity', 'relationship', etc.
    title: str
    description: str
    relevance_score: float
    confidence: ConfidenceLevel
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_display_score(self) -> float:
        """Get display-friendly score (0-100)."""
        return self.relevance_score * 100


@dataclass
class RecommendationSet:
    """Represents a set of recommendations for a query."""
    query_id: str
    recommendation_type: RecommendationType
    strategy_used: RecommendationStrategy
    items: List[RecommendationItem]
    generated_at: datetime
    diversity_score: float = 0.0
    novelty_score: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_items(self, n: int = 10) -> List[RecommendationItem]:
        """Get top N recommendation items."""
        sorted_items = sorted(self.items, key=lambda x: x.relevance_score, reverse=True)
        return sorted_items[:n]
    
    def get_average_confidence(self) -> ConfidenceLevel:
        """Get average confidence level."""
        if not self.items:
            return ConfidenceLevel.VERY_LOW
        
        confidence_values = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.VERY_HIGH: 0.95
        }
        
        avg_confidence = np.mean([confidence_values[item.confidence] for item in self.items])
        
        if avg_confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif avg_confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif avg_confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif avg_confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass
class UserProfile:
    """Represents a user's interaction profile."""
    user_id: str
    interaction_history: List[Tuple[str, str, float, datetime]]  # (item_id, action, weight, timestamp)
    preferences: Dict[str, float] = field(default_factory=dict)
    expertise_areas: List[str] = field(default_factory=list)
    learning_goals: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_interest_vector(self, all_items: List[str]) -> np.ndarray:
        """Generate interest vector for collaborative filtering."""
        interest_dict = defaultdict(float)
        
        # Decay weights based on recency
        now = datetime.utcnow()
        
        for item_id, action, weight, timestamp in self.interaction_history:
            time_decay = math.exp(-((now - timestamp).total_seconds() / (7 * 24 * 3600)))  # 7 day half-life
            
            action_multiplier = {
                'view': 1.0,
                'save': 2.0,
                'share': 3.0,
                'edit': 2.5,
                'search': 1.5
            }.get(action, 1.0)
            
            interest_dict[item_id] += weight * action_multiplier * time_decay
        
        # Create vector
        vector = np.array([interest_dict.get(item, 0.0) for item in all_items])
        
        # Normalize
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector


class ContentBasedRecommender:
    """
    Content-Based Recommendation Engine.
    
    Recommends items based on similarity to items the user has interacted with.
    """
    
    def __init__(self):
        self.item_features: Dict[str, np.ndarray] = {}
        self.feature_weights: Dict[str, float] = {
            'semantic_similarity': 0.4,
            'entity_overlap': 0.3,
            'topic_similarity': 0.2,
            'temporal_proximity': 0.1
        }
    
    async def generate_recommendations(
        self,
        user_profile: UserProfile,
        candidate_items: List[Memory],
        num_recommendations: int = 10
    ) -> List[RecommendationItem]:
        """Generate content-based recommendations."""
        
        if not user_profile.interaction_history:
            logger.warning("No interaction history for content-based recommendations")
            return []
        
        # Extract user preferences from history
        positive_items = [
            item_id for item_id, action, weight, _ in user_profile.interaction_history
            if weight > 0.5  # Positive interactions
        ]
        
        if not positive_items:
            return []
        
        # Calculate similarity scores for each candidate
        recommendations = []
        
        for memory in candidate_items:
            if memory.id in positive_items:
                continue  # Don't recommend items already interacted with
            
            similarity_score = await self._calculate_content_similarity(
                memory, positive_items, candidate_items
            )
            
            if similarity_score > 0.1:  # Minimum threshold
                confidence = self._map_score_to_confidence(similarity_score)
                
                item = RecommendationItem(
                    item_id=memory.id,
                    item_type="memory",
                    title=memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                    description=f"Similar to your previous interests",
                    relevance_score=similarity_score,
                    confidence=confidence,
                    reasoning=f"Content similarity score: {similarity_score:.3f}",
                    metadata={
                        'memory_created_at': memory.created_at.isoformat(),
                        'content_length': len(memory.content)
                    }
                )
                recommendations.append(item)
        
        # Sort by relevance and return top items
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        return recommendations[:num_recommendations]
    
    async def _calculate_content_similarity(
        self,
        target_memory: Memory,
        positive_item_ids: List[str],
        all_memories: List[Memory]
    ) -> float:
        """Calculate content similarity between target and user's positive items."""
        
        # Create lookup for memories
        memory_lookup = {m.id: m for m in all_memories}
        
        similarities = []
        
        for positive_id in positive_item_ids:
            if positive_id not in memory_lookup:
                continue
            
            positive_memory = memory_lookup[positive_id]
            
            # Semantic similarity (simplified - using text overlap)
            semantic_sim = self._calculate_text_similarity(
                target_memory.content, positive_memory.content
            )
            
            # Entity overlap
            entity_sim = self._calculate_entity_overlap(
                target_memory.content, positive_memory.content
            )
            
            # Topic similarity (using simple keyword matching)
            topic_sim = self._calculate_topic_similarity(
                target_memory.content, positive_memory.content
            )
            
            # Temporal proximity
            temporal_sim = self._calculate_temporal_proximity(
                target_memory.created_at, positive_memory.created_at
            )
            
            # Weighted combination
            overall_sim = (
                self.feature_weights['semantic_similarity'] * semantic_sim +
                self.feature_weights['entity_overlap'] * entity_sim +
                self.feature_weights['topic_similarity'] * topic_sim +
                self.feature_weights['temporal_proximity'] * temporal_sim
            )
            
            similarities.append(overall_sim)
        
        # Return average similarity
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard index."""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_entity_overlap(self, text1: str, text2: str) -> float:
        """Calculate entity overlap between texts."""
        
        # Simple entity extraction (capitalized words)
        import re
        entities1 = set(re.findall(r'\b[A-Z][a-z]+\b', text1))
        entities2 = set(re.findall(r'\b[A-Z][a-z]+\b', text2))
        
        if not entities1 or not entities2:
            return 0.0
        
        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_topic_similarity(self, text1: str, text2: str) -> float:
        """Calculate topic similarity using keyword matching."""
        
        # Extract keywords (words longer than 3 characters, excluding common words)
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that'}
        
        keywords1 = set([
            word.lower() for word in text1.split() 
            if len(word) > 3 and word.lower() not in common_words
        ])
        
        keywords2 = set([
            word.lower() for word in text2.split() 
            if len(word) > 3 and word.lower() not in common_words
        ])
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_temporal_proximity(self, time1: datetime, time2: datetime) -> float:
        """Calculate temporal proximity score."""
        
        time_diff = abs((time1 - time2).total_seconds())
        
        # Decay function: closer in time = higher similarity
        # 1 day = 0.9, 1 week = 0.5, 1 month = 0.1
        proximity = math.exp(-time_diff / (7 * 24 * 3600))  # 7 day half-life
        
        return proximity
    
    def _map_score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Map similarity score to confidence level."""
        
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class GraphEmbeddingRecommender:
    """
    Graph Embedding-Based Recommendation Engine.
    
    Uses graph structure and embeddings to find related items.
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.graph: nx.Graph = nx.Graph()
    
    async def generate_recommendations(
        self,
        user_profile: UserProfile,
        knowledge_graph: nx.Graph,
        num_recommendations: int = 10
    ) -> List[RecommendationItem]:
        """Generate recommendations using graph embeddings."""
        
        # Update graph and compute embeddings
        self.graph = knowledge_graph.copy()
        await self._compute_node_embeddings()
        
        if not user_profile.interaction_history:
            return await self._generate_popularity_based_recommendations(num_recommendations)
        
        # Get user's interacted nodes
        user_nodes = [
            item_id for item_id, action, weight, _ in user_profile.interaction_history
            if weight > 0.5 and item_id in self.node_embeddings
        ]
        
        if not user_nodes:
            return await self._generate_popularity_based_recommendations(num_recommendations)
        
        # Calculate user embedding as average of interacted nodes
        user_embedding = np.mean([self.node_embeddings[node] for node in user_nodes], axis=0)
        
        # Find similar nodes
        recommendations = []
        
        for node_id, node_embedding in self.node_embeddings.items():
            if node_id in user_nodes:
                continue  # Skip already interacted nodes
            
            # Calculate similarity
            similarity = cosine_similarity(
                user_embedding.reshape(1, -1),
                node_embedding.reshape(1, -1)
            )[0, 0]
            
            if similarity > 0.1:  # Minimum threshold
                confidence = self._map_score_to_confidence(similarity)
                
                # Get node information
                node_data = self.graph.nodes.get(node_id, {})
                node_type = node_data.get('type', 'unknown')
                
                item = RecommendationItem(
                    item_id=node_id,
                    item_type=node_type,
                    title=node_data.get('title', node_id),
                    description=f"Graph embedding similarity",
                    relevance_score=similarity,
                    confidence=confidence,
                    reasoning=f"Graph embedding similarity: {similarity:.3f}",
                    metadata={
                        'node_degree': self.graph.degree(node_id),
                        'node_type': node_type
                    }
                )
                recommendations.append(item)
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        return recommendations[:num_recommendations]
    
    async def _compute_node_embeddings(self):
        """Compute node embeddings using local methods."""
        
        if len(self.graph.nodes()) == 0:
            return
        
        try:
            # Method 1: Use node2vec-style random walks (simplified)
            embeddings = await self._node2vec_embeddings()
            
            if embeddings:
                self.node_embeddings = embeddings
                return
            
        except Exception as e:
            logger.warning(f"Node2vec embeddings failed: {e}")
        
        try:
            # Method 2: Use spectral embeddings
            embeddings = await self._spectral_embeddings()
            
            if embeddings:
                self.node_embeddings = embeddings
                return
                
        except Exception as e:
            logger.warning(f"Spectral embeddings failed: {e}")
        
        # Method 3: Fallback to degree-based embeddings
        await self._degree_based_embeddings()
    
    async def _node2vec_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """Simplified node2vec-style embeddings using random walks."""
        
        if len(self.graph.nodes()) < 2:
            return None
        
        walks = []
        walk_length = 10
        num_walks = 20
        
        # Generate random walks
        for node in self.graph.nodes():
            for _ in range(num_walks):
                walk = await self._random_walk(node, walk_length)
                if len(walk) > 1:
                    walks.append(walk)
        
        if not walks:
            return None
        
        # Create co-occurrence matrix
        vocab = list(self.graph.nodes())
        vocab_to_idx = {node: i for i, node in enumerate(vocab)}
        
        cooccurrence = np.zeros((len(vocab), len(vocab)))
        
        for walk in walks:
            for i, node1 in enumerate(walk):
                for j in range(max(0, i-2), min(len(walk), i+3)):  # Window size 2
                    if i != j:
                        node2 = walk[j]
                        idx1, idx2 = vocab_to_idx[node1], vocab_to_idx[node2]
                        cooccurrence[idx1, idx2] += 1
        
        # Apply SVD for dimensionality reduction
        try:
            U, s, Vt = np.linalg.svd(cooccurrence)
            
            # Take top dimensions
            embedding_matrix = U[:, :self.embedding_dim] * np.sqrt(s[:self.embedding_dim])
            
            # Create embedding dictionary
            embeddings = {}
            for i, node in enumerate(vocab):
                embeddings[node] = embedding_matrix[i]
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"SVD failed in node2vec: {e}")
            return None
    
    async def _random_walk(self, start_node: str, length: int) -> List[str]:
        """Generate a random walk starting from given node."""
        
        walk = [start_node]
        current_node = start_node
        
        for _ in range(length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            
            if not neighbors:
                break
            
            # Simple random selection (in full node2vec, this would be biased)
            next_node = np.random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
        
        return walk
    
    async def _spectral_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """Compute spectral embeddings using graph Laplacian."""
        
        try:
            # Convert to adjacency matrix
            adj_matrix = nx.adjacency_matrix(self.graph).astype(float)
            
            # Compute normalized Laplacian
            degree_matrix = np.diag(adj_matrix.sum(axis=1).A1)
            laplacian = degree_matrix - adj_matrix.toarray()
            
            # Normalize
            degree_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degree_matrix.diagonal(), 1e-12)))
            normalized_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv
            
            # Compute eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian)
            
            # Take smallest eigenvalues (excluding the first which is always 0)
            embedding_matrix = eigenvectors[:, 1:self.embedding_dim+1]
            
            # Create embedding dictionary
            embeddings = {}
            nodes = list(self.graph.nodes())
            
            for i, node in enumerate(nodes):
                embeddings[node] = embedding_matrix[i]
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"Spectral embeddings failed: {e}")
            return None
    
    async def _degree_based_embeddings(self):
        """Fallback: Simple degree-based embeddings."""
        
        nodes = list(self.graph.nodes())
        
        for node in nodes:
            # Create embedding based on node properties
            degree = self.graph.degree(node)
            node_data = self.graph.nodes.get(node, {})
            
            # Simple feature vector
            features = [
                degree,
                len(list(self.graph.neighbors(node))),
                hash(node) % 1000 / 1000.0,  # Node ID hash
                node_data.get('importance', 0.5)
            ]
            
            # Pad or truncate to embedding dimension
            if len(features) < self.embedding_dim:
                features.extend([0.0] * (self.embedding_dim - len(features)))
            else:
                features = features[:self.embedding_dim]
            
            self.node_embeddings[node] = np.array(features)
    
    async def _generate_popularity_based_recommendations(
        self,
        num_recommendations: int
    ) -> List[RecommendationItem]:
        """Generate popularity-based recommendations as fallback."""
        
        if not self.graph.nodes():
            return []
        
        # Calculate PageRank for popularity
        try:
            pagerank_scores = nx.pagerank(self.graph)
        except:
            # Fallback to degree centrality
            pagerank_scores = nx.degree_centrality(self.graph)
        
        # Sort by popularity
        popular_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        
        for node_id, score in popular_nodes[:num_recommendations]:
            node_data = self.graph.nodes.get(node_id, {})
            confidence = self._map_score_to_confidence(score)
            
            item = RecommendationItem(
                item_id=node_id,
                item_type=node_data.get('type', 'unknown'),
                title=node_data.get('title', node_id),
                description="Popular item",
                relevance_score=score,
                confidence=confidence,
                reasoning=f"Popularity score: {score:.3f}",
                metadata={
                    'pagerank_score': score,
                    'node_degree': self.graph.degree(node_id)
                }
            )
            recommendations.append(item)
        
        return recommendations
    
    def _map_score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Map similarity score to confidence level."""
        
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class DiversityOptimizer:
    """
    Diversity Optimization Engine.
    
    Ensures recommendation diversity to avoid filter bubbles.
    """
    
    def __init__(self, diversity_lambda: float = 0.5):
        self.diversity_lambda = diversity_lambda  # Balance between relevance and diversity
    
    async def optimize_diversity(
        self,
        recommendations: List[RecommendationItem],
        target_diversity: float = 0.7,
        max_items: int = 10
    ) -> List[RecommendationItem]:
        """Optimize recommendation list for diversity."""
        
        if len(recommendations) <= max_items:
            return recommendations
        
        # Use maximal marginal relevance (MMR) algorithm
        selected_items = []
        remaining_items = recommendations.copy()
        
        # Start with the highest relevance item
        if remaining_items:
            best_item = max(remaining_items, key=lambda x: x.relevance_score)
            selected_items.append(best_item)
            remaining_items.remove(best_item)
        
        # Iteratively select items balancing relevance and diversity
        while len(selected_items) < max_items and remaining_items:
            best_item = None
            best_score = -1
            
            for candidate in remaining_items:
                # Calculate MMR score
                relevance = candidate.relevance_score
                
                # Calculate maximum similarity to already selected items
                max_similarity = 0.0
                for selected_item in selected_items:
                    similarity = await self._calculate_item_similarity(candidate, selected_item)
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score: balance relevance and diversity
                mmr_score = (
                    self.diversity_lambda * relevance - 
                    (1 - self.diversity_lambda) * max_similarity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = candidate
            
            if best_item:
                selected_items.append(best_item)
                remaining_items.remove(best_item)
            else:
                break
        
        return selected_items
    
    async def _calculate_item_similarity(
        self,
        item1: RecommendationItem,
        item2: RecommendationItem
    ) -> float:
        """Calculate similarity between two recommendation items."""
        
        # Type similarity
        type_similarity = 1.0 if item1.item_type == item2.item_type else 0.0
        
        # Content similarity (simplified using title/description)
        content_similarity = self._text_similarity(
            item1.title + " " + item1.description,
            item2.title + " " + item2.description
        )
        
        # Metadata similarity
        metadata_similarity = self._metadata_similarity(item1.metadata, item2.metadata)
        
        # Weighted combination
        overall_similarity = (
            0.3 * type_similarity +
            0.5 * content_similarity +
            0.2 * metadata_similarity
        )
        
        return overall_similarity
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard index."""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _metadata_similarity(self, meta1: Dict[str, Any], meta2: Dict[str, Any]) -> float:
        """Calculate metadata similarity."""
        
        if not meta1 or not meta2:
            return 0.0
        
        common_keys = set(meta1.keys()).intersection(set(meta2.keys()))
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            val1, val2 = meta1[key], meta2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                sim = self._text_similarity(val1, val2)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                sim = 1.0 - abs(val1 - val2) / max_val if max_val > 0 else 1.0
            else:
                sim = 1.0 if val1 == val2 else 0.0
            
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0


class GraphBasedRecommendationEngine:
    """
    Main Graph-Based Recommendation Engine.
    
    Coordinates different recommendation strategies and provides unified interface.
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        cache_ttl: int = 3600
    ):
        self.redis_cache = redis_cache
        self.cache_ttl = cache_ttl
        
        # Recommendation engines
        self.content_recommender = ContentBasedRecommender()
        self.graph_recommender = GraphEmbeddingRecommender()
        self.diversity_optimizer = DiversityOptimizer()
        
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Statistics
        self.stats = {
            'total_recommendations': 0,
            'content_based_recs': 0,
            'graph_based_recs': 0,
            'hybrid_recs': 0,
            'user_profiles_created': 0
        }
    
    async def generate_recommendations(
        self,
        user_id: str,
        recommendation_type: RecommendationType,
        strategy: RecommendationStrategy = RecommendationStrategy.HYBRID_FUSION,
        num_recommendations: int = 10,
        memories: Optional[List[Memory]] = None,
        knowledge_graph: Optional[nx.Graph] = None
    ) -> RecommendationSet:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            recommendation_type: Type of recommendation to generate
            strategy: Recommendation strategy to use
            num_recommendations: Number of recommendations to return
            memories: Available memories for recommendation
            knowledge_graph: Knowledge graph for graph-based recommendations
        
        Returns:
            Set of recommendations with metadata
        """
        
        start_time = datetime.utcnow()
        query_id = f"rec_{user_id}_{recommendation_type.value}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Get or create user profile
        user_profile = await self._get_user_profile(user_id)
        
        # Cache key for this recommendation request
        cache_key = f"recommendations:{user_id}:{recommendation_type.value}:{strategy.value}:{num_recommendations}"
        
        # Check cache
        if self.redis_cache:
            cached_result = await self.redis_cache.get(cache_key)
            if cached_result:
                cached_rec_set = RecommendationSet(**cached_result)
                logger.info(f"Returning cached recommendations for user {user_id}")
                return cached_rec_set
        
        # Generate recommendations based on strategy
        if strategy == RecommendationStrategy.CONTENT_BASED:
            recommendations = await self._generate_content_based(
                user_profile, memories or [], num_recommendations
            )
            self.stats['content_based_recs'] += 1
            
        elif strategy == RecommendationStrategy.GRAPH_EMBEDDING:
            recommendations = await self._generate_graph_based(
                user_profile, knowledge_graph or nx.Graph(), num_recommendations
            )
            self.stats['graph_based_recs'] += 1
            
        elif strategy == RecommendationStrategy.HYBRID_FUSION:
            recommendations = await self._generate_hybrid(
                user_profile, memories or [], knowledge_graph or nx.Graph(), num_recommendations
            )
            self.stats['hybrid_recs'] += 1
            
        else:
            # Fallback to content-based
            recommendations = await self._generate_content_based(
                user_profile, memories or [], num_recommendations
            )
        
        # Apply diversity optimization
        if len(recommendations) > num_recommendations:
            recommendations = await self.diversity_optimizer.optimize_diversity(
                recommendations, max_items=num_recommendations
            )
        
        # Calculate diversity and novelty scores
        diversity_score = await self._calculate_diversity_score(recommendations)
        novelty_score = await self._calculate_novelty_score(recommendations, user_profile)
        
        # Create recommendation set
        rec_set = RecommendationSet(
            query_id=query_id,
            recommendation_type=recommendation_type,
            strategy_used=strategy,
            items=recommendations,
            generated_at=start_time,
            diversity_score=diversity_score,
            novelty_score=novelty_score,
            explanation=await self._generate_explanation(recommendation_type, strategy, len(recommendations))
        )
        
        # Cache result
        if self.redis_cache:
            await self.redis_cache.set(
                cache_key,
                rec_set.__dict__,
                ttl=self.cache_ttl
            )
        
        # Update statistics
        self.stats['total_recommendations'] += 1
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"Generated recommendations for user {user_id}",
            recommendation_type=recommendation_type.value,
            strategy=strategy.value,
            num_recommendations=len(recommendations),
            processing_time=processing_time,
            diversity_score=diversity_score,
            novelty_score=novelty_score
        )
        
        return rec_set
    
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                interaction_history=[]
            )
            self.stats['user_profiles_created'] += 1
        
        return self.user_profiles[user_id]
    
    async def _generate_content_based(
        self,
        user_profile: UserProfile,
        memories: List[Memory],
        num_recommendations: int
    ) -> List[RecommendationItem]:
        """Generate content-based recommendations."""
        
        return await self.content_recommender.generate_recommendations(
            user_profile, memories, num_recommendations
        )
    
    async def _generate_graph_based(
        self,
        user_profile: UserProfile,
        knowledge_graph: nx.Graph,
        num_recommendations: int
    ) -> List[RecommendationItem]:
        """Generate graph-based recommendations."""
        
        return await self.graph_recommender.generate_recommendations(
            user_profile, knowledge_graph, num_recommendations
        )
    
    async def _generate_hybrid(
        self,
        user_profile: UserProfile,
        memories: List[Memory],
        knowledge_graph: nx.Graph,
        num_recommendations: int
    ) -> List[RecommendationItem]:
        """Generate hybrid recommendations combining multiple strategies."""
        
        # Generate recommendations from both engines
        content_recs = await self._generate_content_based(
            user_profile, memories, num_recommendations * 2
        )
        
        graph_recs = await self._generate_graph_based(
            user_profile, knowledge_graph, num_recommendations * 2
        )
        
        # Combine and merge similar items
        all_recommendations = content_recs + graph_recs
        
        # Merge items with same ID (average scores)
        merged_recs = {}
        
        for rec in all_recommendations:
            if rec.item_id in merged_recs:
                existing = merged_recs[rec.item_id]
                # Average the scores
                new_score = (existing.relevance_score + rec.relevance_score) / 2
                existing.relevance_score = new_score
                existing.reasoning += f"; {rec.reasoning}"
            else:
                merged_recs[rec.item_id] = rec
        
        # Convert back to list and sort
        final_recs = list(merged_recs.values())
        final_recs.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return final_recs[:num_recommendations]
    
    async def _calculate_diversity_score(self, recommendations: List[RecommendationItem]) -> float:
        """Calculate diversity score for recommendation set."""
        
        if len(recommendations) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                similarity = await self.diversity_optimizer._calculate_item_similarity(
                    recommendations[i], recommendations[j]
                )
                similarities.append(similarity)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity
    
    async def _calculate_novelty_score(
        self,
        recommendations: List[RecommendationItem],
        user_profile: UserProfile
    ) -> float:
        """Calculate novelty score for recommendation set."""
        
        if not user_profile.interaction_history:
            return 1.0  # Everything is novel for new users
        
        # Get user's historical items
        historical_items = set([
            item_id for item_id, _, _, _ in user_profile.interaction_history
        ])
        
        # Calculate novelty as percentage of new items
        novel_items = [
            rec for rec in recommendations 
            if rec.item_id not in historical_items
        ]
        
        novelty_score = len(novel_items) / len(recommendations) if recommendations else 0.0
        return novelty_score
    
    async def _generate_explanation(
        self,
        recommendation_type: RecommendationType,
        strategy: RecommendationStrategy,
        num_items: int
    ) -> str:
        """Generate human-readable explanation for recommendations."""
        
        type_explanations = {
            RecommendationType.MEMORY_SUGGESTION: "memories that might interest you",
            RecommendationType.KNOWLEDGE_GAP: "knowledge gaps to explore",
            RecommendationType.RELATED_CONCEPT: "related concepts",
            RecommendationType.CAUSAL_EXPLORATION: "causal relationships to investigate",
            RecommendationType.TEMPORAL_PATTERN: "temporal patterns",
            RecommendationType.ENTITY_CONNECTION: "entity connections",
            RecommendationType.LEARNING_PATH: "learning paths",
            RecommendationType.ANOMALY_INVESTIGATION: "anomalies to investigate"
        }
        
        strategy_explanations = {
            RecommendationStrategy.CONTENT_BASED: "based on content similarity to your interests",
            RecommendationStrategy.GRAPH_EMBEDDING: "based on graph relationships",
            RecommendationStrategy.HYBRID_FUSION: "using multiple recommendation strategies",
            RecommendationStrategy.COLLABORATIVE_FILTERING: "based on similar users",
            RecommendationStrategy.PAGERANK_BASED: "based on item importance",
            RecommendationStrategy.DIVERSITY_PROMOTING: "promoting diversity"
        }
        
        type_desc = type_explanations.get(recommendation_type, "items")
        strategy_desc = strategy_explanations.get(strategy, "using our recommendation system")
        
        return f"Found {num_items} {type_desc} {strategy_desc}."
    
    async def update_user_interaction(
        self,
        user_id: str,
        item_id: str,
        action: str,
        weight: float = 1.0
    ):
        """Update user interaction history."""
        
        user_profile = await self._get_user_profile(user_id)
        
        # Add interaction to history
        interaction = (item_id, action, weight, datetime.utcnow())
        user_profile.interaction_history.append(interaction)
        
        # Keep only recent interactions (last 1000)
        if len(user_profile.interaction_history) > 1000:
            user_profile.interaction_history.pop(0)
        
        user_profile.last_updated = datetime.utcnow()
        
        logger.info(
            f"Updated user interaction",
            user_id=user_id,
            item_id=item_id,
            action=action,
            weight=weight
        )
    
    async def get_recommendation_explanation(
        self,
        recommendation_set: RecommendationSet,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """Generate detailed explanation for recommendations."""
        
        explanation = {
            'recommendation_type': recommendation_set.recommendation_type.value,
            'strategy_used': recommendation_set.strategy_used.value,
            'total_items': len(recommendation_set.items),
            'average_confidence': recommendation_set.get_average_confidence().value,
            'diversity_score': recommendation_set.diversity_score,
            'novelty_score': recommendation_set.novelty_score,
            'generated_at': recommendation_set.generated_at.isoformat()
        }
        
        if detailed:
            explanation['item_details'] = [
                {
                    'item_id': item.item_id,
                    'relevance_score': item.relevance_score,
                    'confidence': item.confidence.value,
                    'reasoning': item.reasoning
                }
                for item in recommendation_set.items
            ]
            
            explanation['score_distribution'] = {
                'high_confidence': len([i for i in recommendation_set.items if i.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]]),
                'medium_confidence': len([i for i in recommendation_set.items if i.confidence == ConfidenceLevel.MEDIUM]),
                'low_confidence': len([i for i in recommendation_set.items if i.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]])
            }
        
        return explanation
    
    async def get_trending_recommendations(
        self,
        time_window_hours: int = 24,
        num_recommendations: int = 10
    ) -> List[RecommendationItem]:
        """Get trending items based on recent interactions."""
        
        # Analyze recent interactions across all users
        now = datetime.utcnow()
        cutoff_time = now - timedelta(hours=time_window_hours)
        
        item_scores = defaultdict(float)
        
        for user_profile in self.user_profiles.values():
            for item_id, action, weight, timestamp in user_profile.interaction_history:
                if timestamp >= cutoff_time:
                    # Apply time decay
                    time_factor = 1.0 - ((now - timestamp).total_seconds() / (time_window_hours * 3600))
                    score = weight * time_factor
                    item_scores[item_id] += score
        
        # Sort by popularity
        trending_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        
        for item_id, score in trending_items[:num_recommendations]:
            confidence = self._map_score_to_confidence(score)
            
            item = RecommendationItem(
                item_id=item_id,
                item_type="trending",
                title=f"Trending item: {item_id}",
                description="Popular in the last 24 hours",
                relevance_score=score,
                confidence=confidence,
                reasoning=f"Trending score: {score:.3f}",
                metadata={
                    'trending_score': score,
                    'time_window_hours': time_window_hours
                }
            )
            recommendations.append(item)
        
        return recommendations
    
    def _map_score_to_confidence(self, score: float) -> ConfidenceLevel:
        """Map score to confidence level."""
        
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def export_user_profiles(self) -> Dict[str, Any]:
        """Export all user profiles for analysis."""
        
        return {
            'total_users': len(self.user_profiles),
            'profiles': {
                user_id: {
                    'interaction_count': len(profile.interaction_history),
                    'last_updated': profile.last_updated.isoformat(),
                    'expertise_areas': profile.expertise_areas,
                    'learning_goals': profile.learning_goals
                }
                for user_id, profile in self.user_profiles.items()
            }
        }
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        
        return {
            'recommendation_stats': self.stats.copy(),
            'user_stats': {
                'total_users': len(self.user_profiles),
                'active_users': len([
                    p for p in self.user_profiles.values() 
                    if p.last_updated >= datetime.utcnow() - timedelta(days=7)
                ])
            },
            'diversity_optimizer_config': {
                'diversity_lambda': self.diversity_optimizer.diversity_lambda
            }
        }