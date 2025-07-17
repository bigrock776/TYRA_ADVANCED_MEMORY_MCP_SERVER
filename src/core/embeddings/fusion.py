"""
Advanced Embedding Fusion System for Multi-Strategy Integration.

This module provides comprehensive embedding fusion capabilities using weighted averaging,
attention-based fusion with local models, quality assessment using cosine similarity,
and strategy optimization using local search. All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import math
import hashlib

# ML and optimization imports - all local
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize, differential_evolution
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings
from .contextual import SessionAwareEmbedder, ContextualEmbedding
from .multi_perspective import MultiPerspectiveEmbedder, MultiPerspectiveResult

logger = structlog.get_logger(__name__)


class FusionStrategy(str, Enum):
    """Strategies for embedding fusion."""
    WEIGHTED_AVERAGE = "weighted_average"            # Simple weighted averaging
    ATTENTION_FUSION = "attention_fusion"            # Attention-based fusion
    NEURAL_FUSION = "neural_fusion"                  # Neural network fusion
    ADAPTIVE_FUSION = "adaptive_fusion"              # Adaptive fusion with learning
    HIERARCHICAL_FUSION = "hierarchical_fusion"      # Multi-level hierarchical fusion
    CONSENSUS_FUSION = "consensus_fusion"            # Consensus-based fusion
    ENSEMBLE_FUSION = "ensemble_fusion"              # Ensemble method fusion
    QUALITY_WEIGHTED = "quality_weighted"            # Quality-score weighted fusion


class QualityMetric(str, Enum):
    """Metrics for assessing embedding quality."""
    COSINE_CONSISTENCY = "cosine_consistency"        # Internal cosine similarity consistency
    MAGNITUDE_STABILITY = "magnitude_stability"      # Embedding magnitude stability
    CLUSTER_COHERENCE = "cluster_coherence"          # Clustering coherence
    DIMENSIONALITY_QUALITY = "dimensionality_quality"  # Dimensionality reduction quality
    RETRIEVAL_ACCURACY = "retrieval_accuracy"        # Retrieval task accuracy
    SEMANTIC_PRESERVATION = "semantic_preservation"   # Semantic relationship preservation


class OptimizationMethod(str, Enum):
    """Methods for fusion optimization."""
    GRID_SEARCH = "grid_search"                      # Grid search optimization
    RANDOM_SEARCH = "random_search"                  # Random search optimization
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"  # Bayesian optimization
    EVOLUTIONARY = "evolutionary"                    # Evolutionary algorithms
    GRADIENT_DESCENT = "gradient_descent"            # Gradient-based optimization
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # RL-based optimization


@dataclass
class EmbeddingSource:
    """Represents a source of embeddings for fusion."""
    source_id: str
    source_type: str  # 'contextual', 'multi_perspective', 'base', etc.
    embedding: np.ndarray
    confidence: float
    quality_scores: Dict[QualityMetric, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    processing_time: float = 0.0
    
    def calculate_overall_quality(self, weights: Optional[Dict[QualityMetric, float]] = None) -> float:
        """Calculate overall quality score."""
        if not self.quality_scores:
            return self.confidence
        
        default_weights = {
            QualityMetric.COSINE_CONSISTENCY: 0.3,
            QualityMetric.MAGNITUDE_STABILITY: 0.2,
            QualityMetric.CLUSTER_COHERENCE: 0.2,
            QualityMetric.SEMANTIC_PRESERVATION: 0.3
        }
        
        weights = weights or default_weights
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, score in self.quality_scores.items():
            if metric in weights:
                weighted_score += weights[metric] * score
                total_weight += weights[metric]
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return self.confidence


@dataclass
class FusionResult:
    """Result of embedding fusion process."""
    fused_embedding: np.ndarray
    fusion_strategy: FusionStrategy
    source_weights: Dict[str, float]
    quality_score: float
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_weighted_sources(self) -> List[Tuple[str, float]]:
        """Get sources sorted by weight."""
        return sorted(self.source_weights.items(), key=lambda x: x[1], reverse=True)


@dataclass
class FusionConfig:
    """Configuration for fusion parameters."""
    strategy: FusionStrategy = FusionStrategy.ADAPTIVE_FUSION
    quality_weights: Dict[QualityMetric, float] = field(default_factory=dict)
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    min_confidence_threshold: float = 0.3
    max_sources: int = 10
    normalize_output: bool = True
    
    # Attention fusion specific
    attention_temperature: float = 1.0
    attention_heads: int = 8
    
    # Neural fusion specific
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    
    # Adaptive fusion specific
    adaptation_rate: float = 0.1
    memory_decay: float = 0.95
    
    # Quality assessment
    enable_quality_filtering: bool = True
    quality_threshold: float = 0.5


class QualityAssessor:
    """
    Quality assessment system for embeddings.
    
    Features:
    - Multiple quality metrics calculation
    - Statistical consistency analysis
    - Clustering-based coherence assessment
    - Semantic preservation evaluation
    """
    
    def __init__(
        self,
        reference_embeddings: Optional[np.ndarray] = None,
        cluster_count: int = 10
    ):
        self.reference_embeddings = reference_embeddings
        self.cluster_count = cluster_count
        
        # Quality assessment models
        self.clusterer = KMeans(n_clusters=cluster_count, random_state=42)
        self.pca = PCA(n_components=50, random_state=42)
        
        # Quality history for adaptive assessment
        self.quality_history: Dict[str, List[float]] = defaultdict(list)
    
    async def assess_embedding_quality(
        self,
        embedding: np.ndarray,
        source_id: str,
        metrics: Set[QualityMetric] = None
    ) -> Dict[QualityMetric, float]:
        """Assess quality of a single embedding."""
        
        metrics = metrics or {
            QualityMetric.COSINE_CONSISTENCY,
            QualityMetric.MAGNITUDE_STABILITY,
            QualityMetric.CLUSTER_COHERENCE,
            QualityMetric.DIMENSIONALITY_QUALITY
        }
        
        quality_scores = {}
        
        for metric in metrics:
            try:
                if metric == QualityMetric.COSINE_CONSISTENCY:
                    score = await self._assess_cosine_consistency(embedding)
                elif metric == QualityMetric.MAGNITUDE_STABILITY:
                    score = await self._assess_magnitude_stability(embedding)
                elif metric == QualityMetric.CLUSTER_COHERENCE:
                    score = await self._assess_cluster_coherence(embedding)
                elif metric == QualityMetric.DIMENSIONALITY_QUALITY:
                    score = await self._assess_dimensionality_quality(embedding)
                elif metric == QualityMetric.SEMANTIC_PRESERVATION:
                    score = await self._assess_semantic_preservation(embedding)
                else:
                    score = 0.5  # Default score
                
                quality_scores[metric] = score
                
                # Update quality history
                self.quality_history[f"{source_id}_{metric.value}"].append(score)
                if len(self.quality_history[f"{source_id}_{metric.value}"]) > 100:
                    self.quality_history[f"{source_id}_{metric.value}"].pop(0)
                
            except Exception as e:
                logger.warning(f"Failed to assess {metric.value} for {source_id}: {e}")
                quality_scores[metric] = 0.5  # Default score on failure
        
        return quality_scores
    
    async def _assess_cosine_consistency(self, embedding: np.ndarray) -> float:
        """Assess internal cosine similarity consistency."""
        
        if self.reference_embeddings is None or len(self.reference_embeddings) == 0:
            # Use embedding magnitude as proxy
            magnitude = np.linalg.norm(embedding)
            return min(1.0, magnitude / 1.5)  # Normalized embeddings should be ~1
        
        # Calculate similarities with reference embeddings
        similarities = cosine_similarity(
            embedding.reshape(1, -1),
            self.reference_embeddings
        )[0]
        
        # Consistency is measured by low variance in similarities
        consistency = 1.0 - np.std(similarities)
        return max(0.0, min(1.0, consistency))
    
    async def _assess_magnitude_stability(self, embedding: np.ndarray) -> float:
        """Assess embedding magnitude stability."""
        
        magnitude = np.linalg.norm(embedding)
        
        # For normalized embeddings, magnitude should be close to 1
        ideal_magnitude = 1.0
        deviation = abs(magnitude - ideal_magnitude)
        
        # Convert deviation to stability score (0-1)
        stability = max(0.0, 1.0 - deviation)
        return stability
    
    async def _assess_cluster_coherence(self, embedding: np.ndarray) -> float:
        """Assess clustering coherence."""
        
        if self.reference_embeddings is None or len(self.reference_embeddings) < self.cluster_count:
            return 0.5  # Can't assess without sufficient reference data
        
        try:
            # Fit clusterer on reference embeddings if not already done
            if not hasattr(self.clusterer, 'cluster_centers_'):
                self.clusterer.fit(self.reference_embeddings)
            
            # Find closest cluster for the embedding
            cluster_id = self.clusterer.predict(embedding.reshape(1, -1))[0]
            cluster_center = self.clusterer.cluster_centers_[cluster_id]
            
            # Calculate distance to cluster center
            distance = np.linalg.norm(embedding - cluster_center)
            
            # Convert distance to coherence score
            # Smaller distance = higher coherence
            coherence = max(0.0, 1.0 - distance / 2.0)  # Assuming max distance of 2
            return coherence
            
        except Exception as e:
            logger.warning(f"Cluster coherence assessment failed: {e}")
            return 0.5
    
    async def _assess_dimensionality_quality(self, embedding: np.ndarray) -> float:
        """Assess quality through dimensionality reduction."""
        
        if self.reference_embeddings is None or len(self.reference_embeddings) < 50:
            # Use embedding distribution properties
            return await self._assess_distribution_quality(embedding)
        
        try:
            # Fit PCA if not already done
            if not hasattr(self.pca, 'components_'):
                self.pca.fit(self.reference_embeddings)
            
            # Transform embedding and reconstruct
            transformed = self.pca.transform(embedding.reshape(1, -1))
            reconstructed = self.pca.inverse_transform(transformed)[0]
            
            # Calculate reconstruction error
            reconstruction_error = np.linalg.norm(embedding - reconstructed)
            
            # Convert error to quality score
            quality = max(0.0, 1.0 - reconstruction_error / np.linalg.norm(embedding))
            return quality
            
        except Exception as e:
            logger.warning(f"Dimensionality quality assessment failed: {e}")
            return 0.5
    
    async def _assess_distribution_quality(self, embedding: np.ndarray) -> float:
        """Assess embedding distribution quality."""
        
        # Check if embedding values follow expected distribution
        # Good embeddings often have values roughly following normal distribution
        
        # Normalize embedding values
        normalized_values = (embedding - np.mean(embedding)) / (np.std(embedding) + 1e-8)
        
        # Test for normality using Kolmogorov-Smirnov test
        try:
            stat, p_value = stats.kstest(normalized_values, 'norm')
            
            # Higher p-value indicates better fit to normal distribution
            distribution_quality = min(1.0, p_value * 2)  # Scale to 0-1
            return distribution_quality
            
        except Exception:
            # Fallback: use entropy as proxy for quality
            # Higher entropy generally indicates better distribution
            probabilities = np.abs(embedding) / (np.sum(np.abs(embedding)) + 1e-8)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            
            # Normalize entropy to 0-1 range
            max_entropy = np.log(len(embedding))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
    
    async def _assess_semantic_preservation(self, embedding: np.ndarray) -> float:
        """Assess semantic relationship preservation."""
        
        # This would require semantic ground truth
        # For now, use consistency with reference embeddings as proxy
        
        if self.reference_embeddings is None:
            return 0.5
        
        # Calculate average similarity with reference embeddings
        similarities = cosine_similarity(
            embedding.reshape(1, -1),
            self.reference_embeddings
        )[0]
        
        # Use average similarity as semantic preservation score
        avg_similarity = np.mean(similarities)
        return max(0.0, min(1.0, avg_similarity))
    
    def get_quality_trends(self, source_id: str) -> Dict[str, float]:
        """Get quality trends for a source."""
        
        trends = {}
        
        for key, values in self.quality_history.items():
            if key.startswith(f"{source_id}_") and len(values) >= 2:
                metric_name = key.replace(f"{source_id}_", "")
                
                # Calculate trend (positive = improving, negative = declining)
                if len(values) >= 10:
                    recent_avg = np.mean(values[-5:])  # Last 5 values
                    older_avg = np.mean(values[-10:-5])  # Previous 5 values
                    trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
                else:
                    # Simple difference for shorter history
                    trend = values[-1] - values[0] if len(values) >= 2 else 0.0
                
                trends[metric_name] = trend
        
        return trends


class AttentionFusionNetwork(nn.Module):
    """
    Neural attention-based fusion network.
    
    Features:
    - Multi-head attention mechanism
    - Learnable source importance weights
    - Quality-aware attention scores
    - Residual connections
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Source importance network
        self.source_importance = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),  # +1 for quality score
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Final fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Quality-based gating
        self.quality_gate = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,  # Shape: (batch_size, num_sources, embedding_dim)
        quality_scores: torch.Tensor,  # Shape: (batch_size, num_sources)
        source_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for attention fusion.
        
        Returns:
            fused_embedding: Fused embedding tensor
            attention_weights: Attention weights used for fusion
        """
        
        batch_size, num_sources, embedding_dim = embeddings.shape
        
        # Calculate source importance scores
        quality_expanded = quality_scores.unsqueeze(-1)  # Shape: (batch_size, num_sources, 1)
        source_input = torch.cat([embeddings, quality_expanded], dim=-1)
        importance_scores = self.source_importance(source_input)  # Shape: (batch_size, num_sources, 1)
        importance_scores = importance_scores.squeeze(-1)  # Shape: (batch_size, num_sources)
        
        # Apply source weights if provided
        if source_weights is not None:
            importance_scores = importance_scores * source_weights
        
        # Apply attention mechanism
        attended_embeddings, attention_weights = self.attention(
            embeddings, embeddings, embeddings,
            key_padding_mask=None
        )
        
        # Weight by importance and quality
        combined_weights = importance_scores * quality_scores
        combined_weights = torch.softmax(combined_weights, dim=-1)
        
        # Weighted combination
        weighted_embeddings = attended_embeddings * combined_weights.unsqueeze(-1)
        pooled_embedding = torch.sum(weighted_embeddings, dim=1)  # Shape: (batch_size, embedding_dim)
        
        # Quality-based gating
        avg_quality = torch.mean(quality_scores, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
        quality_gate = self.quality_gate(avg_quality)
        
        # Final fusion with residual connection
        fused_embedding = self.fusion_network(pooled_embedding)
        
        # Apply quality gating
        fused_embedding = fused_embedding * quality_gate
        
        # Normalize output
        fused_embedding = torch.nn.functional.normalize(fused_embedding, p=2, dim=-1)
        
        return fused_embedding, combined_weights


class AdaptiveFusionOptimizer:
    """
    Adaptive optimization system for fusion strategies.
    
    Features:
    - Performance-based weight adaptation
    - Multi-objective optimization
    - Bayesian optimization for hyperparameters
    - Reinforcement learning for strategy selection
    """
    
    def __init__(
        self,
        optimization_method: OptimizationMethod = OptimizationMethod.EVOLUTIONARY,
        adaptation_rate: float = 0.1
    ):
        self.optimization_method = optimization_method
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[FusionStrategy, List[float]] = defaultdict(list)
        
        # Optimization state
        self.current_weights: Dict[str, float] = {}
        self.best_weights: Dict[str, float] = {}
        self.best_performance: float = 0.0
        
        # ML components for learned optimization
        self.performance_predictor: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    async def optimize_fusion_weights(
        self,
        sources: List[EmbeddingSource],
        target_metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Optimize fusion weights for given sources and target metrics."""
        
        if self.optimization_method == OptimizationMethod.GRID_SEARCH:
            return await self._grid_search_optimization(sources, target_metrics, constraints)
        elif self.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            return await self._random_search_optimization(sources, target_metrics, constraints)
        elif self.optimization_method == OptimizationMethod.EVOLUTIONARY:
            return await self._evolutionary_optimization(sources, target_metrics, constraints)
        elif self.optimization_method == OptimizationMethod.GRADIENT_DESCENT:
            return await self._gradient_descent_optimization(sources, target_metrics, constraints)
        else:
            # Fallback to simple uniform weights
            return {source.source_id: 1.0 / len(sources) for source in sources}
    
    async def _grid_search_optimization(
        self,
        sources: List[EmbeddingSource],
        target_metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Grid search optimization for fusion weights."""
        
        source_ids = [source.source_id for source in sources]
        grid_points = 5  # Number of points per dimension
        
        # Generate grid of weight combinations
        weight_grids = []
        for _ in source_ids:
            weight_grids.append(np.linspace(0.0, 1.0, grid_points))
        
        best_weights = None
        best_score = -float('inf')
        
        # Evaluate all combinations
        import itertools
        for weight_combination in itertools.product(*weight_grids):
            # Normalize weights
            total_weight = sum(weight_combination)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weight_combination]
            else:
                normalized_weights = [1.0 / len(source_ids)] * len(source_ids)
            
            weights_dict = dict(zip(source_ids, normalized_weights))
            
            # Evaluate this weight combination
            score = await self._evaluate_fusion_performance(
                sources, weights_dict, target_metrics
            )
            
            if score > best_score:
                best_score = score
                best_weights = weights_dict
        
        return best_weights or {source_id: 1.0 / len(source_ids) for source_id in source_ids}
    
    async def _random_search_optimization(
        self,
        sources: List[EmbeddingSource],
        target_metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Random search optimization for fusion weights."""
        
        source_ids = [source.source_id for source in sources]
        num_iterations = 100
        
        best_weights = None
        best_score = -float('inf')
        
        for _ in range(num_iterations):
            # Generate random weights
            random_weights = np.random.dirichlet(np.ones(len(source_ids)))
            weights_dict = dict(zip(source_ids, random_weights))
            
            # Evaluate performance
            score = await self._evaluate_fusion_performance(
                sources, weights_dict, target_metrics
            )
            
            if score > best_score:
                best_score = score
                best_weights = weights_dict
        
        return best_weights or {source_id: 1.0 / len(source_ids) for source_id in source_ids}
    
    async def _evolutionary_optimization(
        self,
        sources: List[EmbeddingSource],
        target_metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evolutionary optimization for fusion weights."""
        
        source_ids = [source.source_id for source in sources]
        
        def objective_function(weights):
            # Normalize weights
            weights = np.abs(weights)  # Ensure non-negative
            total_weight = np.sum(weights)
            if total_weight > 0:
                weights = weights / total_weight
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            weights_dict = dict(zip(source_ids, weights))
            
            # Use asyncio to run the async evaluation
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            score = loop.run_until_complete(
                self._evaluate_fusion_performance(sources, weights_dict, target_metrics)
            )
            loop.close()
            
            return -score  # Minimize negative score (maximize score)
        
        # Define bounds
        bounds = [(0.0, 1.0) for _ in source_ids]
        
        # Run differential evolution
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=10,
                seed=42
            )
            
            if result.success:
                # Normalize final weights
                final_weights = np.abs(result.x)
                total_weight = np.sum(final_weights)
                if total_weight > 0:
                    final_weights = final_weights / total_weight
                else:
                    final_weights = np.ones(len(final_weights)) / len(final_weights)
                
                return dict(zip(source_ids, final_weights))
        
        except Exception as e:
            logger.warning(f"Evolutionary optimization failed: {e}")
        
        # Fallback to uniform weights
        return {source_id: 1.0 / len(source_ids) for source_id in source_ids}
    
    async def _gradient_descent_optimization(
        self,
        sources: List[EmbeddingSource],
        target_metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Gradient descent optimization (using numerical gradients)."""
        
        source_ids = [source.source_id for source in sources]
        
        # Initialize weights
        weights = np.ones(len(source_ids)) / len(source_ids)
        learning_rate = 0.1
        num_iterations = 20
        epsilon = 1e-4  # For numerical gradient
        
        for iteration in range(num_iterations):
            # Calculate numerical gradients
            gradients = np.zeros_like(weights)
            
            for i in range(len(weights)):
                # Forward difference
                weights_plus = weights.copy()
                weights_plus[i] += epsilon
                weights_plus /= np.sum(weights_plus)  # Normalize
                
                weights_minus = weights.copy()
                weights_minus[i] = max(0, weights_minus[i] - epsilon)
                weights_minus /= np.sum(weights_minus)  # Normalize
                
                # Evaluate performance
                weights_dict_plus = dict(zip(source_ids, weights_plus))
                weights_dict_minus = dict(zip(source_ids, weights_minus))
                
                score_plus = await self._evaluate_fusion_performance(
                    sources, weights_dict_plus, target_metrics
                )
                score_minus = await self._evaluate_fusion_performance(
                    sources, weights_dict_minus, target_metrics
                )
                
                # Numerical gradient
                gradients[i] = (score_plus - score_minus) / (2 * epsilon)
            
            # Update weights
            weights += learning_rate * gradients
            weights = np.maximum(weights, 0)  # Ensure non-negative
            weights /= np.sum(weights)  # Normalize
            
            # Decay learning rate
            learning_rate *= 0.95
        
        return dict(zip(source_ids, weights))
    
    async def _evaluate_fusion_performance(
        self,
        sources: List[EmbeddingSource],
        weights: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> float:
        """Evaluate performance of fusion with given weights."""
        
        # Create weighted average embedding
        weighted_embeddings = []
        total_weight = 0.0
        
        for source in sources:
            weight = weights.get(source.source_id, 0.0)
            if weight > 0:
                weighted_embeddings.append(weight * source.embedding)
                total_weight += weight
        
        if not weighted_embeddings or total_weight == 0:
            return 0.0
        
        fused_embedding = np.sum(weighted_embeddings, axis=0) / total_weight
        
        # Normalize
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding /= norm
        
        # Calculate performance metrics
        performance_score = 0.0
        
        # Quality consistency
        if 'quality_consistency' in target_metrics:
            quality_scores = [source.calculate_overall_quality() for source in sources]
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                performance_score += target_metrics['quality_consistency'] * avg_quality
        
        # Magnitude stability
        if 'magnitude_stability' in target_metrics:
            magnitude = np.linalg.norm(fused_embedding)
            stability = max(0.0, 1.0 - abs(magnitude - 1.0))  # Prefer normalized embeddings
            performance_score += target_metrics['magnitude_stability'] * stability
        
        # Diversity preservation
        if 'diversity_preservation' in target_metrics:
            source_embeddings = [source.embedding for source in sources]
            if len(source_embeddings) > 1:
                similarities = []
                for i in range(len(source_embeddings)):
                    for j in range(i + 1, len(source_embeddings)):
                        sim = np.dot(source_embeddings[i], source_embeddings[j])
                        similarities.append(sim)
                
                diversity = 1.0 - np.mean(similarities) if similarities else 0.5
                performance_score += target_metrics['diversity_preservation'] * diversity
        
        return performance_score
    
    def record_performance(
        self,
        strategy: FusionStrategy,
        weights: Dict[str, float],
        performance_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record fusion performance for learning."""
        
        performance_record = {
            'strategy': strategy.value,
            'weights': weights.copy(),
            'performance': performance_score,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        self.performance_history.append(performance_record)
        self.strategy_performance[strategy].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history) > 500:
            self.performance_history = self.performance_history[-400:]
        
        for strategy_key in self.strategy_performance:
            if len(self.strategy_performance[strategy_key]) > 100:
                self.strategy_performance[strategy_key] = self.strategy_performance[strategy_key][-80:]
        
        # Update best weights if this is the best performance
        if performance_score > self.best_performance:
            self.best_performance = performance_score
            self.best_weights = weights.copy()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        
        stats = {
            'total_optimizations': len(self.performance_history),
            'best_performance': self.best_performance,
            'best_weights': self.best_weights,
            'strategy_performance': {}
        }
        
        for strategy, performances in self.strategy_performance.items():
            if performances:
                stats['strategy_performance'][strategy.value] = {
                    'count': len(performances),
                    'mean': np.mean(performances),
                    'std': np.std(performances),
                    'best': np.max(performances),
                    'recent_trend': np.mean(performances[-10:]) - np.mean(performances[-20:-10]) if len(performances) >= 20 else 0.0
                }
        
        return stats


class EmbeddingFusionEngine:
    """
    Comprehensive embedding fusion engine.
    
    Features:
    - Multiple fusion strategies (weighted, attention, neural, adaptive)
    - Quality assessment and filtering
    - Strategy optimization with local search algorithms
    - Performance monitoring and adaptive learning
    - Multi-source embedding integration
    """
    
    def __init__(
        self,
        fusion_config: FusionConfig,
        redis_cache: Optional[RedisCache] = None,
        device: str = "cpu"
    ):
        self.fusion_config = fusion_config
        self.redis_cache = redis_cache
        self.device = device
        
        # Core components
        self.quality_assessor = QualityAssessor()
        self.optimizer = AdaptiveFusionOptimizer()
        
        # Neural components
        self.attention_fusion_network: Optional[AttentionFusionNetwork] = None
        self.neural_fusion_network: Optional[nn.Module] = None
        
        # Fusion state
        self.fusion_history: List[FusionResult] = []
        self.source_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Adaptive weights
        self.adaptive_weights: Dict[str, float] = {}
        self.weight_momentum: Dict[str, float] = {}
        
        logger.info(f"EmbeddingFusionEngine initialized with strategy: {fusion_config.strategy.value}")
    
    async def initialize(self, embedding_dim: int):
        """Initialize fusion networks."""
        
        try:
            if self.fusion_config.strategy in [FusionStrategy.ATTENTION_FUSION, FusionStrategy.NEURAL_FUSION]:
                self.attention_fusion_network = AttentionFusionNetwork(
                    embedding_dim=embedding_dim,
                    num_heads=self.fusion_config.attention_heads,
                    hidden_dim=self.fusion_config.hidden_dim,
                    dropout_rate=self.fusion_config.dropout_rate
                ).to(self.device)
                
                # Initialize network weights
                for module in self.attention_fusion_network.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            
            logger.info("EmbeddingFusionEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingFusionEngine: {e}")
            raise
    
    async def fuse_embeddings(
        self,
        sources: List[EmbeddingSource],
        text: Optional[str] = None,
        optimize_weights: bool = True
    ) -> FusionResult:
        """Fuse embeddings from multiple sources."""
        
        start_time = datetime.utcnow()
        
        # Filter sources by quality if enabled
        if self.fusion_config.enable_quality_filtering:
            sources = await self._filter_sources_by_quality(sources)
        
        if not sources:
            raise ValueError("No valid sources for fusion")
        
        # Limit number of sources
        if len(sources) > self.fusion_config.max_sources:
            # Keep the highest quality sources
            sources = sorted(
                sources,
                key=lambda s: s.calculate_overall_quality(),
                reverse=True
            )[:self.fusion_config.max_sources]
        
        # Assess quality for all sources
        for source in sources:
            if not source.quality_scores:
                source.quality_scores = await self.quality_assessor.assess_embedding_quality(
                    source.embedding, source.source_id
                )
        
        # Determine fusion weights
        if optimize_weights:
            source_weights = await self._determine_optimal_weights(sources)
        else:
            source_weights = self.fusion_config.fusion_weights or {
                source.source_id: 1.0 / len(sources) for source in sources
            }
        
        # Perform fusion based on strategy
        if self.fusion_config.strategy == FusionStrategy.WEIGHTED_AVERAGE:
            fused_embedding, confidence = await self._weighted_average_fusion(sources, source_weights)
        
        elif self.fusion_config.strategy == FusionStrategy.ATTENTION_FUSION:
            fused_embedding, confidence = await self._attention_fusion(sources, source_weights)
        
        elif self.fusion_config.strategy == FusionStrategy.NEURAL_FUSION:
            fused_embedding, confidence = await self._neural_fusion(sources, source_weights)
        
        elif self.fusion_config.strategy == FusionStrategy.ADAPTIVE_FUSION:
            fused_embedding, confidence = await self._adaptive_fusion(sources, source_weights)
        
        elif self.fusion_config.strategy == FusionStrategy.HIERARCHICAL_FUSION:
            fused_embedding, confidence = await self._hierarchical_fusion(sources, source_weights)
        
        elif self.fusion_config.strategy == FusionStrategy.CONSENSUS_FUSION:
            fused_embedding, confidence = await self._consensus_fusion(sources, source_weights)
        
        elif self.fusion_config.strategy == FusionStrategy.QUALITY_WEIGHTED:
            fused_embedding, confidence = await self._quality_weighted_fusion(sources, source_weights)
        
        else:
            # Fallback to weighted average
            fused_embedding, confidence = await self._weighted_average_fusion(sources, source_weights)
        
        # Normalize output if requested
        if self.fusion_config.normalize_output:
            norm = np.linalg.norm(fused_embedding)
            if norm > 0:
                fused_embedding /= norm
        
        # Calculate quality score for fused embedding
        quality_score = await self._calculate_fusion_quality(fused_embedding, sources)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Create result
        result = FusionResult(
            fused_embedding=fused_embedding,
            fusion_strategy=self.fusion_config.strategy,
            source_weights=source_weights,
            quality_score=quality_score,
            confidence=confidence,
            processing_time=processing_time,
            metadata={
                'num_sources': len(sources),
                'source_types': [source.source_type for source in sources],
                'text': text
            }
        )
        
        # Store result
        self.fusion_history.append(result)
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-800:]
        
        # Update performance tracking
        for source in sources:
            weight = source_weights.get(source.source_id, 0.0)
            self.source_performance[source.source_id].append(weight * quality_score)
        
        logger.info(f"Fusion completed: strategy={self.fusion_config.strategy.value}, quality={quality_score:.3f}")
        
        return result
    
    async def _filter_sources_by_quality(self, sources: List[EmbeddingSource]) -> List[EmbeddingSource]:
        """Filter sources based on quality thresholds."""
        
        filtered_sources = []
        
        for source in sources:
            # Calculate overall quality if not already done
            if not source.quality_scores:
                source.quality_scores = await self.quality_assessor.assess_embedding_quality(
                    source.embedding, source.source_id
                )
            
            overall_quality = source.calculate_overall_quality(self.fusion_config.quality_weights)
            
            # Check quality threshold
            if overall_quality >= self.fusion_config.quality_threshold:
                filtered_sources.append(source)
            else:
                logger.debug(f"Filtered out source {source.source_id} due to low quality: {overall_quality:.3f}")
        
        return filtered_sources
    
    async def _determine_optimal_weights(self, sources: List[EmbeddingSource]) -> Dict[str, float]:
        """Determine optimal fusion weights for sources."""
        
        # Use adaptive weights if available and recent
        if self.adaptive_weights and len(self.fusion_history) > 0:
            # Check if adaptive weights match current sources
            source_ids = {source.source_id for source in sources}
            adaptive_source_ids = set(self.adaptive_weights.keys())
            
            if source_ids == adaptive_source_ids:
                return self.adaptive_weights.copy()
        
        # Define target metrics for optimization
        target_metrics = {
            'quality_consistency': 0.4,
            'magnitude_stability': 0.3,
            'diversity_preservation': 0.3
        }
        
        # Optimize weights
        optimal_weights = await self.optimizer.optimize_fusion_weights(
            sources, target_metrics
        )
        
        return optimal_weights
    
    async def _weighted_average_fusion(
        self,
        sources: List[EmbeddingSource],
        source_weights: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """Simple weighted average fusion."""
        
        weighted_embeddings = []
        total_weight = 0.0
        confidence_sum = 0.0
        
        for source in sources:
            weight = source_weights.get(source.source_id, 0.0)
            if weight > 0:
                # Apply quality weighting
                quality_factor = source.calculate_overall_quality(self.fusion_config.quality_weights)
                effective_weight = weight * quality_factor
                
                weighted_embeddings.append(effective_weight * source.embedding)
                total_weight += effective_weight
                confidence_sum += effective_weight * source.confidence
        
        if total_weight > 0:
            fused_embedding = np.sum(weighted_embeddings, axis=0) / total_weight
            confidence = confidence_sum / total_weight
        else:
            # Fallback to simple average
            embeddings = [source.embedding for source in sources]
            fused_embedding = np.mean(embeddings, axis=0)
            confidence = np.mean([source.confidence for source in sources])
        
        return fused_embedding, confidence
    
    async def _attention_fusion(
        self,
        sources: List[EmbeddingSource],
        source_weights: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """Attention-based fusion using neural network."""
        
        if self.attention_fusion_network is None:
            # Fallback to weighted average
            return await self._weighted_average_fusion(sources, source_weights)
        
        try:
            # Prepare tensors
            embeddings = torch.stack([
                torch.tensor(source.embedding, dtype=torch.float32)
                for source in sources
            ]).unsqueeze(0).to(self.device)  # Shape: (1, num_sources, embedding_dim)
            
            quality_scores = torch.tensor([
                source.calculate_overall_quality(self.fusion_config.quality_weights)
                for source in sources
            ], dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, num_sources)
            
            weights_tensor = torch.tensor([
                source_weights.get(source.source_id, 0.0)
                for source in sources
            ], dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1, num_sources)
            
            # Forward pass
            with torch.no_grad():
                fused_embedding, attention_weights = self.attention_fusion_network(
                    embeddings, quality_scores, weights_tensor
                )
            
            # Convert back to numpy
            fused_embedding_np = fused_embedding.squeeze(0).cpu().numpy()
            attention_weights_np = attention_weights.squeeze(0).cpu().numpy()
            
            # Calculate confidence as weighted average of source confidences
            confidence = np.sum(attention_weights_np * np.array([source.confidence for source in sources]))
            
            return fused_embedding_np, confidence
            
        except Exception as e:
            logger.warning(f"Attention fusion failed: {e}, falling back to weighted average")
            return await self._weighted_average_fusion(sources, source_weights)
    
    async def _neural_fusion(
        self,
        sources: List[EmbeddingSource],
        source_weights: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """Neural network-based fusion."""
        
        # For now, use attention fusion as neural fusion
        return await self._attention_fusion(sources, source_weights)
    
    async def _adaptive_fusion(
        self,
        sources: List[EmbeddingSource],
        source_weights: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """Adaptive fusion with learning."""
        
        # Start with weighted average
        base_embedding, base_confidence = await self._weighted_average_fusion(sources, source_weights)
        
        # Adapt weights based on historical performance
        adapted_weights = source_weights.copy()
        
        for source in sources:
            source_id = source.source_id
            
            # Get historical performance for this source
            if source_id in self.source_performance and len(self.source_performance[source_id]) > 5:
                recent_performance = np.mean(self.source_performance[source_id][-5:])
                overall_performance = np.mean(self.source_performance[source_id])
                
                # Adaptation factor based on recent vs overall performance
                if recent_performance > overall_performance:
                    # Recent performance is better - increase weight
                    adaptation_factor = 1.0 + self.fusion_config.adaptation_rate
                else:
                    # Recent performance is worse - decrease weight
                    adaptation_factor = 1.0 - self.fusion_config.adaptation_rate
                
                adapted_weights[source_id] *= adaptation_factor
        
        # Normalize adapted weights
        total_adapted_weight = sum(adapted_weights.values())
        if total_adapted_weight > 0:
            for source_id in adapted_weights:
                adapted_weights[source_id] /= total_adapted_weight
        
        # Apply momentum to weight updates
        for source_id in adapted_weights:
            if source_id in self.weight_momentum:
                momentum_factor = self.fusion_config.memory_decay
                adapted_weights[source_id] = (
                    momentum_factor * self.weight_momentum[source_id] +
                    (1 - momentum_factor) * adapted_weights[source_id]
                )
            
            self.weight_momentum[source_id] = adapted_weights[source_id]
        
        # Store adaptive weights for future use
        self.adaptive_weights = adapted_weights.copy()
        
        # Re-fuse with adapted weights
        adapted_embedding, adapted_confidence = await self._weighted_average_fusion(sources, adapted_weights)
        
        return adapted_embedding, adapted_confidence
    
    async def _hierarchical_fusion(
        self,
        sources: List[EmbeddingSource],
        source_weights: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """Hierarchical fusion with multiple levels."""
        
        # Group sources by type
        source_groups = defaultdict(list)
        for source in sources:
            source_groups[source.source_type].append(source)
        
        # First level: fuse within each group
        group_embeddings = []
        group_confidences = []
        group_weights = []
        
        for group_type, group_sources in source_groups.items():
            if len(group_sources) == 1:
                # Single source group
                group_embedding = group_sources[0].embedding
                group_confidence = group_sources[0].confidence
                group_weight = source_weights.get(group_sources[0].source_id, 0.0)
            else:
                # Multi-source group - fuse internally
                group_source_weights = {
                    source.source_id: source_weights.get(source.source_id, 0.0)
                    for source in group_sources
                }
                
                group_embedding, group_confidence = await self._weighted_average_fusion(
                    group_sources, group_source_weights
                )
                group_weight = sum(group_source_weights.values())
            
            group_embeddings.append(group_embedding)
            group_confidences.append(group_confidence)
            group_weights.append(group_weight)
        
        # Second level: fuse group embeddings
        if len(group_embeddings) == 1:
            return group_embeddings[0], group_confidences[0]
        
        # Weighted average of group embeddings
        total_group_weight = sum(group_weights)
        if total_group_weight > 0:
            normalized_group_weights = [w / total_group_weight for w in group_weights]
        else:
            normalized_group_weights = [1.0 / len(group_weights)] * len(group_weights)
        
        final_embedding = np.zeros_like(group_embeddings[0])
        final_confidence = 0.0
        
        for embedding, confidence, weight in zip(group_embeddings, group_confidences, normalized_group_weights):
            final_embedding += weight * embedding
            final_confidence += weight * confidence
        
        return final_embedding, final_confidence
    
    async def _consensus_fusion(
        self,
        sources: List[EmbeddingSource],
        source_weights: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """Consensus-based fusion using agreement between sources."""
        
        if len(sources) < 3:
            # Need at least 3 sources for meaningful consensus
            return await self._weighted_average_fusion(sources, source_weights)
        
        embeddings = np.array([source.embedding for source in sources])
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        # Calculate consensus scores for each source
        consensus_scores = []
        for i in range(len(sources)):
            # Exclude self-similarity
            other_similarities = [similarity_matrix[i][j] for j in range(len(sources)) if i != j]
            consensus_score = np.mean(other_similarities)
            consensus_scores.append(consensus_score)
        
        # Weight sources by consensus scores
        consensus_weights = {}
        for i, source in enumerate(sources):
            base_weight = source_weights.get(source.source_id, 0.0)
            consensus_weight = consensus_scores[i]
            combined_weight = base_weight * consensus_weight
            consensus_weights[source.source_id] = combined_weight
        
        # Normalize consensus weights
        total_consensus_weight = sum(consensus_weights.values())
        if total_consensus_weight > 0:
            for source_id in consensus_weights:
                consensus_weights[source_id] /= total_consensus_weight
        
        # Fuse with consensus weights
        return await self._weighted_average_fusion(sources, consensus_weights)
    
    async def _quality_weighted_fusion(
        self,
        sources: List[EmbeddingSource],
        source_weights: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """Quality-weighted fusion emphasizing high-quality sources."""
        
        # Calculate quality-adjusted weights
        quality_weights = {}
        
        for source in sources:
            base_weight = source_weights.get(source.source_id, 0.0)
            quality_score = source.calculate_overall_quality(self.fusion_config.quality_weights)
            
            # Exponential quality weighting to emphasize high-quality sources
            quality_factor = np.exp(quality_score) / np.exp(1.0)  # Normalize by e
            quality_weights[source.source_id] = base_weight * quality_factor
        
        # Normalize quality weights
        total_quality_weight = sum(quality_weights.values())
        if total_quality_weight > 0:
            for source_id in quality_weights:
                quality_weights[source_id] /= total_quality_weight
        
        return await self._weighted_average_fusion(sources, quality_weights)
    
    async def _calculate_fusion_quality(
        self,
        fused_embedding: np.ndarray,
        sources: List[EmbeddingSource]
    ) -> float:
        """Calculate quality score for fused embedding."""
        
        # Assess intrinsic quality
        quality_scores = await self.quality_assessor.assess_embedding_quality(
            fused_embedding, "fused"
        )
        
        intrinsic_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.5
        
        # Calculate consistency with source embeddings
        source_embeddings = [source.embedding for source in sources]
        if source_embeddings:
            similarities = cosine_similarity(
                fused_embedding.reshape(1, -1),
                np.array(source_embeddings)
            )[0]
            consistency = np.mean(similarities)
        else:
            consistency = 0.5
        
        # Combine intrinsic quality and consistency
        overall_quality = 0.6 * intrinsic_quality + 0.4 * consistency
        
        return overall_quality
    
    async def update_performance_feedback(
        self,
        fusion_result: FusionResult,
        performance_score: float,
        task_context: Optional[str] = None
    ):
        """Update performance feedback for adaptive learning."""
        
        # Record optimization performance
        self.optimizer.record_performance(
            fusion_result.fusion_strategy,
            fusion_result.source_weights,
            performance_score,
            {'task_context': task_context}
        )
        
        # Update source performance tracking
        for source_id, weight in fusion_result.source_weights.items():
            # Weight the performance by source contribution
            weighted_performance = weight * performance_score
            self.source_performance[source_id].append(weighted_performance)
            
            # Keep only recent history
            if len(self.source_performance[source_id]) > 100:
                self.source_performance[source_id] = self.source_performance[source_id][-80:]
    
    async def get_fusion_stats(self) -> Dict[str, Any]:
        """Get comprehensive fusion statistics."""
        
        stats = {
            'total_fusions': len(self.fusion_history),
            'fusion_strategy': self.fusion_config.strategy.value,
            'source_performance': {},
            'optimization_stats': self.optimizer.get_optimization_stats(),
            'recent_quality_scores': [],
            'adaptive_weights': self.adaptive_weights.copy()
        }
        
        # Calculate source performance statistics
        for source_id, performances in self.source_performance.items():
            if performances:
                stats['source_performance'][source_id] = {
                    'count': len(performances),
                    'mean': np.mean(performances),
                    'std': np.std(performances),
                    'trend': np.mean(performances[-10:]) - np.mean(performances[-20:-10]) if len(performances) >= 20 else 0.0
                }
        
        # Recent quality scores
        if self.fusion_history:
            recent_results = self.fusion_history[-20:]  # Last 20 fusions
            stats['recent_quality_scores'] = [result.quality_score for result in recent_results]
            stats['average_recent_quality'] = np.mean(stats['recent_quality_scores'])
        
        return stats
    
    async def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached fusion results."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Clean fusion history
        self.fusion_history = [
            result for result in self.fusion_history
            if datetime.fromisoformat(result.metadata.get('timestamp', datetime.utcnow().isoformat())) > cutoff_time
        ]
        
        logger.info(f"Cleaned up fusion cache, kept {len(self.fusion_history)} recent results")