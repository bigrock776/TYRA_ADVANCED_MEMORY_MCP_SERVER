"""
Memory Lifecycle Optimization for Predictive Memory Management.

This module provides comprehensive memory lifecycle optimization using local aging algorithms,
transition predictions with local ML models, and optimization recommendations using heuristics.
All processing is performed locally with zero external API calls.
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

# ML and analytics imports - all local
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import scipy.stats as stats
from scipy.optimize import minimize
import networkx as nx

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from .usage_analyzer import UsageAnalyzer, UsagePattern
from .auto_archiver import AutoArchiver, ImportanceScore
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class LifecycleStage(str, Enum):
    """Stages in memory lifecycle."""
    CREATION = "creation"
    ACTIVE_USE = "active_use"
    DECLINING_USE = "declining_use"
    OCCASIONAL_USE = "occasional_use"
    DORMANT = "dormant"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class TransitionType(str, Enum):
    """Types of lifecycle transitions."""
    NATURAL_AGING = "natural_aging"
    USAGE_DRIVEN = "usage_driven"
    IMPORTANCE_CHANGE = "importance_change"
    POLICY_DRIVEN = "policy_driven"
    USER_ACTION = "user_action"
    SYSTEM_TRIGGERED = "system_triggered"


class OptimizationType(str, Enum):
    """Types of optimization recommendations."""
    PRELOAD = "preload"
    CACHE = "cache"
    ARCHIVE = "archive"
    DELETE = "delete"
    COMPRESS = "compress"
    MIGRATE = "migrate"
    REPLICATE = "replicate"
    NO_ACTION = "no_action"


class AgingFunction(str, Enum):
    """Different aging function types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    CUSTOM = "custom"


@dataclass
class LifecycleMetrics:
    """Metrics for memory lifecycle analysis."""
    memory_id: str
    current_stage: LifecycleStage
    age_days: float
    access_frequency: float
    last_access_days: float
    importance_score: float
    usage_trend: float  # Positive = increasing, negative = decreasing
    transition_probability: Dict[LifecycleStage, float] = field(default_factory=dict)
    stage_duration: float = 0.0  # Days in current stage
    predicted_next_stage: Optional[LifecycleStage] = None
    confidence: float = 0.0


@dataclass
class TransitionPrediction:
    """Prediction for lifecycle stage transition."""
    memory_id: str
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    transition_type: TransitionType
    probability: float
    predicted_time: datetime
    factors: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Recommendation for memory optimization."""
    memory_id: str
    recommendation_type: OptimizationType
    current_stage: LifecycleStage
    priority: float
    potential_benefit: float  # 0-1 score
    resource_cost: float  # 0-1 score
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    """Performance metrics for lifecycle optimization."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    total_memories_analyzed: int = Field(description="Total memories analyzed")
    stage_distribution: Dict[str, int] = Field(description="Distribution across lifecycle stages")
    transition_accuracy: float = Field(description="Accuracy of transition predictions")
    optimization_success_rate: float = Field(description="Success rate of optimizations")
    resource_efficiency: float = Field(description="Resource utilization efficiency")
    average_memory_lifespan: float = Field(description="Average memory lifespan in days")
    cost_savings: Dict[str, float] = Field(description="Cost savings by optimization type")


class AgingAlgorithm:
    """
    Advanced aging algorithm for memory lifecycle management.
    
    Features:
    - Multiple aging functions (linear, exponential, sigmoid)
    - Access pattern-based aging adjustment
    - Importance-weighted aging
    - Environmental factor consideration
    """
    
    def __init__(
        self,
        aging_function: AgingFunction = AgingFunction.EXPONENTIAL,
        base_decay_rate: float = 0.1,
        access_boost_factor: float = 0.5
    ):
        """Initialize aging algorithm."""
        self.aging_function = aging_function
        self.base_decay_rate = base_decay_rate
        self.access_boost_factor = access_boost_factor
        
        # Aging parameters
        self.aging_params = {
            'alpha': 0.1,  # Decay rate parameter
            'beta': 0.5,   # Access boost parameter
            'gamma': 0.2,  # Importance weight parameter
            'delta': 0.1   # Environmental factor parameter
        }
    
    def calculate_aging_score(
        self,
        memory: Memory,
        usage_pattern: Optional[UsagePattern] = None,
        importance_score: Optional[ImportanceScore] = None
    ) -> float:
        """Calculate aging score for a memory (0 = new, 1 = very old)."""
        
        # Basic age calculation
        if hasattr(memory, 'created_at') and memory.created_at:
            age_days = (datetime.utcnow() - memory.created_at).days
        else:
            age_days = 0
        
        # Base aging score
        base_score = self._apply_aging_function(age_days)
        
        # Adjust for access patterns
        if usage_pattern:
            access_adjustment = self._calculate_access_adjustment(usage_pattern)
            base_score *= (1 - access_adjustment * self.access_boost_factor)
        
        # Adjust for importance
        if importance_score:
            importance_adjustment = importance_score.final_score
            base_score *= (1 - importance_adjustment * self.aging_params['gamma'])
        
        # Apply environmental factors
        env_adjustment = self._calculate_environmental_adjustment(memory)
        base_score *= (1 + env_adjustment * self.aging_params['delta'])
        
        return max(0.0, min(1.0, base_score))
    
    def _apply_aging_function(self, age_days: float) -> float:
        """Apply the selected aging function."""
        if self.aging_function == AgingFunction.LINEAR:
            return min(1.0, age_days * self.aging_params['alpha'] / 365)  # Linear over 1 year
        
        elif self.aging_function == AgingFunction.EXPONENTIAL:
            return 1.0 - math.exp(-self.aging_params['alpha'] * age_days / 30)  # 30-day scale
        
        elif self.aging_function == AgingFunction.LOGARITHMIC:
            return min(1.0, math.log(1 + age_days * self.aging_params['alpha']) / math.log(365))
        
        elif self.aging_function == AgingFunction.SIGMOID:
            # Sigmoid function centered around 180 days
            x = (age_days - 180) / 60  # Scale factor
            return 1.0 / (1.0 + math.exp(-x))
        
        else:  # Custom or fallback
            return min(1.0, age_days / 365.0)
    
    def _calculate_access_adjustment(self, usage_pattern: UsagePattern) -> float:
        """Calculate adjustment based on access patterns."""
        # Higher access frequency reduces aging
        frequency_map = {
            'very_high': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3,
            'very_low': 0.1
        }
        
        frequency_adjustment = frequency_map.get(usage_pattern.access_frequency.value, 0.5)
        
        # Recent access provides additional boost
        recency_adjustment = usage_pattern.metrics.get('recency_score', 0.5)
        
        return (frequency_adjustment * 0.7 + recency_adjustment * 0.3)
    
    def _calculate_environmental_adjustment(self, memory: Memory) -> float:
        """Calculate environmental factors affecting aging."""
        adjustment = 0.0
        
        if hasattr(memory, 'metadata') and memory.metadata:
            # Content type affects aging rate
            content_type = memory.metadata.get('type', 'text')
            type_adjustments = {
                'documentation': -0.2,  # Documentation ages slower
                'code': -0.1,          # Code ages slower
                'note': 0.1,           # Notes age faster
                'temporary': 0.3       # Temporary content ages much faster
            }
            adjustment += type_adjustments.get(content_type, 0.0)
            
            # Domain-specific adjustments
            domain = memory.metadata.get('domain', '')
            if 'critical' in domain.lower():
                adjustment -= 0.2
            elif 'temporary' in domain.lower():
                adjustment += 0.2
        
        return adjustment


class MemoryLifecycleOptimizer:
    """
    Comprehensive memory lifecycle optimizer with predictive capabilities.
    
    Features:
    - Local aging algorithms based on access patterns
    - ML-based transition predictions using local models
    - Optimization recommendations using heuristics
    - Performance metrics and analytics
    - Resource optimization and cost analysis
    - Adaptive parameter tuning
    """
    
    def __init__(
        self,
        usage_analyzer: Optional[UsageAnalyzer] = None,
        auto_archiver: Optional[AutoArchiver] = None,
        redis_cache: Optional[RedisCache] = None,
        optimization_interval_hours: int = 24
    ):
        """
        Initialize the memory lifecycle optimizer.
        
        Args:
            usage_analyzer: Optional usage analyzer for pattern insights
            auto_archiver: Optional auto-archiver for importance scoring
            redis_cache: Optional Redis cache for persistence
            optimization_interval_hours: Hours between optimization runs
        """
        self.usage_analyzer = usage_analyzer
        self.auto_archiver = auto_archiver
        self.redis_cache = redis_cache
        self.optimization_interval_hours = optimization_interval_hours
        
        # Core components
        self.aging_algorithm = AgingAlgorithm()
        
        # Memory lifecycle tracking
        self.lifecycle_metrics: Dict[str, LifecycleMetrics] = {}
        self.stage_transitions: List[Dict[str, Any]] = []
        self.optimization_history: List[OptimizationRecommendation] = []
        
        # ML models
        self.transition_classifier: Optional[RandomForestClassifier] = None
        self.lifespan_predictor: Optional[GradientBoostingRegressor] = None
        self.feature_scaler = StandardScaler()
        self.stage_encoder = LabelEncoder()
        self.models_trained = False
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            total_memories_analyzed=0,
            stage_distribution={},
            transition_accuracy=0.0,
            optimization_success_rate=0.0,
            resource_efficiency=0.0,
            average_memory_lifespan=0.0,
            cost_savings={}
        )
        
        # Optimization parameters
        self.stage_thresholds = {
            LifecycleStage.CREATION: 0.0,
            LifecycleStage.ACTIVE_USE: 0.1,
            LifecycleStage.DECLINING_USE: 0.3,
            LifecycleStage.OCCASIONAL_USE: 0.5,
            LifecycleStage.DORMANT: 0.7,
            LifecycleStage.ARCHIVED: 0.85,
            LifecycleStage.DEPRECATED: 0.95
        }
        
        # Optimization weights
        self.optimization_weights = {
            'performance_impact': 0.4,
            'resource_efficiency': 0.3,
            'cost_benefit': 0.2,
            'risk_factor': 0.1
        }
        
        logger.info(
            "Memory lifecycle optimizer initialized",
            optimization_interval_hours=optimization_interval_hours
        )
    
    async def analyze_memory_lifecycle(
        self,
        memories: List[Memory],
        force_reanalysis: bool = False
    ) -> Dict[str, LifecycleMetrics]:
        """
        Analyze lifecycle metrics for memories.
        
        Args:
            memories: List of memories to analyze
            force_reanalysis: Force reanalysis even if recently done
            
        Returns:
            Dictionary of memory ID to lifecycle metrics
        """
        for memory in memories:
            # Skip if recently analyzed and not forcing
            if (not force_reanalysis and 
                memory.id in self.lifecycle_metrics):
                continue
            
            # Get usage pattern and importance score
            usage_pattern = None
            importance_score = None
            
            if self.usage_analyzer:
                patterns = await self.usage_analyzer.analyze_patterns([memory])
                usage_pattern = patterns.get(memory.id)
            
            if self.auto_archiver:
                importance_score = await self.auto_archiver.analyze_memory_importance(
                    memory, usage_pattern
                )
            
            # Calculate lifecycle metrics
            metrics = await self._calculate_lifecycle_metrics(
                memory, usage_pattern, importance_score
            )
            
            self.lifecycle_metrics[memory.id] = metrics
            self.performance_metrics.total_memories_analyzed += 1
        
        # Update stage distribution
        self._update_stage_distribution()
        
        logger.info(f"Analyzed lifecycle for {len(memories)} memories")
        
        return self.lifecycle_metrics
    
    async def _calculate_lifecycle_metrics(
        self,
        memory: Memory,
        usage_pattern: Optional[UsagePattern],
        importance_score: Optional[ImportanceScore]
    ) -> LifecycleMetrics:
        """Calculate comprehensive lifecycle metrics for a memory."""
        
        # Calculate aging score
        aging_score = self.aging_algorithm.calculate_aging_score(
            memory, usage_pattern, importance_score
        )
        
        # Determine current stage
        current_stage = self._determine_lifecycle_stage(aging_score, usage_pattern)
        
        # Calculate basic metrics
        age_days = 0.0
        last_access_days = 0.0
        
        if hasattr(memory, 'created_at') and memory.created_at:
            age_days = (datetime.utcnow() - memory.created_at).days
        
        # Get access metrics from usage pattern
        access_frequency = 0.0
        usage_trend = 0.0
        
        if usage_pattern:
            frequency_map = {
                'very_high': 1.0, 'high': 0.8, 'medium': 0.6, 
                'low': 0.4, 'very_low': 0.2
            }
            access_frequency = frequency_map.get(usage_pattern.access_frequency.value, 0.0)
            
            # Calculate usage trend from metrics
            if 'trend_strength' in usage_pattern.metrics:
                usage_trend = usage_pattern.metrics['trend_strength']
            
            # Last access calculation would need access to actual access logs
            last_access_days = age_days  # Simplified
        
        # Calculate importance score
        final_importance = importance_score.final_score if importance_score else 0.5
        
        # Predict transition probabilities
        transition_probs = await self._predict_transition_probabilities(
            memory, current_stage, aging_score, usage_pattern, importance_score
        )
        
        # Determine predicted next stage
        predicted_next_stage = None
        confidence = 0.0
        if transition_probs:
            best_transition = max(transition_probs.items(), key=lambda x: x[1])
            predicted_next_stage = best_transition[0]
            confidence = best_transition[1]
        
        return LifecycleMetrics(
            memory_id=memory.id,
            current_stage=current_stage,
            age_days=age_days,
            access_frequency=access_frequency,
            last_access_days=last_access_days,
            importance_score=final_importance,
            usage_trend=usage_trend,
            transition_probability=transition_probs,
            predicted_next_stage=predicted_next_stage,
            confidence=confidence
        )
    
    def _determine_lifecycle_stage(
        self,
        aging_score: float,
        usage_pattern: Optional[UsagePattern]
    ) -> LifecycleStage:
        """Determine current lifecycle stage based on aging score and usage."""
        
        # Adjust thresholds based on usage pattern
        adjusted_thresholds = self.stage_thresholds.copy()
        
        if usage_pattern:
            # High usage memories stay in active stages longer
            if usage_pattern.access_frequency.value in ['very_high', 'high']:
                for stage in adjusted_thresholds:
                    if stage in [LifecycleStage.DECLINING_USE, LifecycleStage.OCCASIONAL_USE]:
                        adjusted_thresholds[stage] += 0.1
            
            # Low usage memories move to dormant faster
            elif usage_pattern.access_frequency.value in ['very_low', 'low']:
                for stage in adjusted_thresholds:
                    if stage in [LifecycleStage.DECLINING_USE, LifecycleStage.OCCASIONAL_USE]:
                        adjusted_thresholds[stage] -= 0.1
        
        # Find appropriate stage
        for stage in reversed(list(LifecycleStage)):
            if stage in adjusted_thresholds and aging_score >= adjusted_thresholds[stage]:
                return stage
        
        return LifecycleStage.CREATION
    
    async def _predict_transition_probabilities(
        self,
        memory: Memory,
        current_stage: LifecycleStage,
        aging_score: float,
        usage_pattern: Optional[UsagePattern],
        importance_score: Optional[ImportanceScore]
    ) -> Dict[LifecycleStage, float]:
        """Predict probabilities of transitioning to each stage."""
        
        if self.models_trained and self.transition_classifier:
            # Use ML model for predictions
            return await self._ml_predict_transitions(
                memory, current_stage, aging_score, usage_pattern, importance_score
            )
        else:
            # Use heuristic-based predictions
            return await self._heuristic_predict_transitions(
                current_stage, aging_score, usage_pattern, importance_score
            )
    
    async def _heuristic_predict_transitions(
        self,
        current_stage: LifecycleStage,
        aging_score: float,
        usage_pattern: Optional[UsagePattern],
        importance_score: Optional[ImportanceScore]
    ) -> Dict[LifecycleStage, float]:
        """Use heuristic rules to predict stage transitions."""
        
        probabilities = {}
        
        # Define possible next stages for each current stage
        stage_transitions = {
            LifecycleStage.CREATION: [LifecycleStage.ACTIVE_USE],
            LifecycleStage.ACTIVE_USE: [LifecycleStage.ACTIVE_USE, LifecycleStage.DECLINING_USE],
            LifecycleStage.DECLINING_USE: [LifecycleStage.OCCASIONAL_USE, LifecycleStage.ACTIVE_USE],
            LifecycleStage.OCCASIONAL_USE: [LifecycleStage.DORMANT, LifecycleStage.DECLINING_USE],
            LifecycleStage.DORMANT: [LifecycleStage.ARCHIVED, LifecycleStage.OCCASIONAL_USE],
            LifecycleStage.ARCHIVED: [LifecycleStage.DEPRECATED, LifecycleStage.DORMANT],
            LifecycleStage.DEPRECATED: [LifecycleStage.DELETED]
        }
        
        possible_stages = stage_transitions.get(current_stage, [])
        
        for next_stage in possible_stages:
            prob = 0.5  # Base probability
            
            # Adjust based on aging score
            stage_threshold = self.stage_thresholds.get(next_stage, 0.5)
            if aging_score > stage_threshold:
                prob += 0.3
            else:
                prob -= 0.2
            
            # Adjust based on usage pattern
            if usage_pattern:
                freq_value = usage_pattern.access_frequency.value
                if next_stage in [LifecycleStage.ACTIVE_USE, LifecycleStage.DECLINING_USE]:
                    if freq_value in ['very_high', 'high']:
                        prob += 0.2
                    elif freq_value in ['very_low', 'low']:
                        prob -= 0.2
            
            # Adjust based on importance
            if importance_score:
                if next_stage in [LifecycleStage.ARCHIVED, LifecycleStage.DEPRECATED, LifecycleStage.DELETED]:
                    if importance_score.final_score > 0.7:
                        prob -= 0.3
                    elif importance_score.final_score < 0.3:
                        prob += 0.2
            
            probabilities[next_stage] = max(0.0, min(1.0, prob))
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            for stage in probabilities:
                probabilities[stage] /= total_prob
        
        return probabilities
    
    async def _ml_predict_transitions(
        self,
        memory: Memory,
        current_stage: LifecycleStage,
        aging_score: float,
        usage_pattern: Optional[UsagePattern],
        importance_score: Optional[ImportanceScore]
    ) -> Dict[LifecycleStage, float]:
        """Use ML model to predict stage transitions."""
        
        try:
            # Extract features
            features = self._extract_transition_features(
                memory, current_stage, aging_score, usage_pattern, importance_score
            )
            
            # Scale features
            features_scaled = self.feature_scaler.transform([features])
            
            # Get predictions
            probabilities = self.transition_classifier.predict_proba(features_scaled)[0]
            
            # Map to stage names
            stage_names = self.stage_encoder.classes_
            result = {}
            
            for i, prob in enumerate(probabilities):
                if i < len(stage_names):
                    stage = LifecycleStage(stage_names[i])
                    result[stage] = float(prob)
            
            return result
            
        except Exception as e:
            logger.warning(f"ML transition prediction failed: {e}")
            # Fallback to heuristic
            return await self._heuristic_predict_transitions(
                current_stage, aging_score, usage_pattern, importance_score
            )
    
    def _extract_transition_features(
        self,
        memory: Memory,
        current_stage: LifecycleStage,
        aging_score: float,
        usage_pattern: Optional[UsagePattern],
        importance_score: Optional[ImportanceScore]
    ) -> List[float]:
        """Extract features for ML transition prediction."""
        
        features = [
            aging_score,
            self.stage_thresholds.get(current_stage, 0.5),
            len(memory.content) / 1000.0,  # Content length in KB
        ]
        
        # Usage pattern features
        if usage_pattern:
            freq_map = {'very_high': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4, 'very_low': 0.2}
            features.extend([
                freq_map.get(usage_pattern.access_frequency.value, 0.0),
                usage_pattern.confidence,
                usage_pattern.metrics.get('recency_score', 0.5),
                usage_pattern.metrics.get('regularity_score', 0.5)
            ])
        else:
            features.extend([0.0, 0.0, 0.5, 0.5])
        
        # Importance features
        if importance_score:
            features.extend([
                importance_score.final_score,
                importance_score.usage_score,
                importance_score.recency_score,
                importance_score.content_score
            ])
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])
        
        return features
    
    async def generate_optimization_recommendations(
        self,
        memories: List[Memory],
        target_resource_usage: float = 0.8
    ) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations for memories.
        
        Args:
            memories: List of memories to optimize
            target_resource_usage: Target resource utilization (0-1)
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Ensure lifecycle analysis is up to date
        await self.analyze_memory_lifecycle(memories)
        
        for memory in memories:
            metrics = self.lifecycle_metrics.get(memory.id)
            if not metrics:
                continue
            
            # Generate recommendations based on lifecycle stage and metrics
            memory_recommendations = await self._generate_memory_recommendations(
                memory, metrics, target_resource_usage
            )
            
            recommendations.extend(memory_recommendations)
        
        # Sort recommendations by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        # Store in history
        self.optimization_history.extend(recommendations)
        
        logger.info(f"Generated {len(recommendations)} optimization recommendations")
        
        return recommendations
    
    async def _generate_memory_recommendations(
        self,
        memory: Memory,
        metrics: LifecycleMetrics,
        target_resource_usage: float
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations for a single memory."""
        
        recommendations = []
        
        # Stage-based recommendations
        if metrics.current_stage == LifecycleStage.ACTIVE_USE:
            if metrics.access_frequency > 0.8:
                # High-usage memory - recommend caching/preloading
                rec = OptimizationRecommendation(
                    memory_id=memory.id,
                    recommendation_type=OptimizationType.PRELOAD,
                    current_stage=metrics.current_stage,
                    priority=0.8,
                    potential_benefit=0.7,
                    resource_cost=0.3,
                    confidence=0.8,
                    reasoning=["High access frequency", "Active usage pattern"],
                    metadata={'access_frequency': metrics.access_frequency}
                )
                recommendations.append(rec)
        
        elif metrics.current_stage == LifecycleStage.DORMANT:
            # Dormant memory - recommend archiving
            rec = OptimizationRecommendation(
                memory_id=memory.id,
                recommendation_type=OptimizationType.ARCHIVE,
                current_stage=metrics.current_stage,
                priority=0.6,
                potential_benefit=0.5,
                resource_cost=0.2,
                confidence=0.7,
                reasoning=["Dormant state", "Low access frequency"],
                metadata={'last_access_days': metrics.last_access_days}
            )
            recommendations.append(rec)
        
        elif metrics.current_stage == LifecycleStage.DEPRECATED:
            # Deprecated memory - recommend deletion
            if metrics.importance_score < 0.3:
                rec = OptimizationRecommendation(
                    memory_id=memory.id,
                    recommendation_type=OptimizationType.DELETE,
                    current_stage=metrics.current_stage,
                    priority=0.9,
                    potential_benefit=0.8,
                    resource_cost=0.1,
                    confidence=0.9,
                    reasoning=["Deprecated state", "Low importance", "Resource recovery"],
                    metadata={'importance_score': metrics.importance_score}
                )
                recommendations.append(rec)
        
        # Transition-based recommendations
        if metrics.predicted_next_stage and metrics.confidence > 0.7:
            next_stage = metrics.predicted_next_stage
            
            if next_stage == LifecycleStage.ARCHIVED:
                # Proactively archive before it becomes necessary
                rec = OptimizationRecommendation(
                    memory_id=memory.id,
                    recommendation_type=OptimizationType.ARCHIVE,
                    current_stage=metrics.current_stage,
                    priority=0.5,
                    potential_benefit=0.4,
                    resource_cost=0.2,
                    confidence=metrics.confidence,
                    reasoning=["Predicted transition to archived", "Proactive optimization"],
                    metadata={'predicted_stage': next_stage.value}
                )
                recommendations.append(rec)
        
        # Resource-based recommendations
        content_size_kb = len(memory.content) / 1024
        if content_size_kb > 100 and metrics.access_frequency < 0.3:
            # Large, infrequently accessed memory - recommend compression
            rec = OptimizationRecommendation(
                memory_id=memory.id,
                recommendation_type=OptimizationType.COMPRESS,
                current_stage=metrics.current_stage,
                priority=0.4,
                potential_benefit=0.6,
                resource_cost=0.1,
                confidence=0.6,
                reasoning=["Large content size", "Low access frequency", "Compression savings"],
                metadata={'content_size_kb': content_size_kb}
            )
            recommendations.append(rec)
        
        return recommendations
    
    async def train_models(self, historical_data: List[Dict[str, Any]]):
        """Train ML models on historical lifecycle data."""
        
        if len(historical_data) < 200:
            logger.warning("Insufficient data for ML model training")
            return
        
        try:
            # Prepare training data
            X = []
            y_transitions = []
            y_lifespan = []
            
            for data in historical_data:
                features = self._extract_transition_features(
                    data.get('memory'),
                    LifecycleStage(data.get('current_stage', 'creation')),
                    data.get('aging_score', 0.5),
                    data.get('usage_pattern'),
                    data.get('importance_score')
                )
                
                X.append(features)
                y_transitions.append(data.get('actual_next_stage', 'no_change'))
                y_lifespan.append(data.get('actual_lifespan_days', 365))
            
            X = np.array(X)
            y_transitions = np.array(y_transitions)
            y_lifespan = np.array(y_lifespan)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train transition classifier
            y_transitions_encoded = self.stage_encoder.fit_transform(y_transitions)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_transitions_encoded, test_size=0.2, random_state=42
            )
            
            self.transition_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.transition_classifier.fit(X_train, y_train)
            
            # Evaluate transition classifier
            y_pred = self.transition_classifier.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            self.performance_metrics.transition_accuracy = accuracy
            
            # Train lifespan predictor
            self.lifespan_predictor = GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
            self.lifespan_predictor.fit(X_scaled, y_lifespan)
            
            # Evaluate lifespan predictor
            lifespan_pred = self.lifespan_predictor.predict(X_scaled)
            r2 = r2_score(y_lifespan, lifespan_pred)
            
            self.models_trained = True
            
            logger.info(
                "ML models trained successfully",
                transition_accuracy=accuracy,
                lifespan_r2=r2
            )
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
    
    def _update_stage_distribution(self):
        """Update stage distribution statistics."""
        if not self.lifecycle_metrics:
            return
        
        stage_counts = defaultdict(int)
        for metrics in self.lifecycle_metrics.values():
            stage_counts[metrics.current_stage.value] += 1
        
        self.performance_metrics.stage_distribution = dict(stage_counts)
    
    async def optimize_parameters(self):
        """Optimize algorithm parameters based on performance metrics."""
        
        if len(self.optimization_history) < 50:
            return  # Need sufficient history
        
        # Analyze optimization success rates
        success_rates = defaultdict(list)
        
        for rec in self.optimization_history[-100:]:  # Last 100 recommendations
            # In practice, would track actual success/failure
            # For now, simulate based on confidence and priority
            simulated_success = rec.confidence * rec.priority > 0.5
            success_rates[rec.recommendation_type.value].append(simulated_success)
        
        # Update optimization weights based on success rates
        for opt_type, successes in success_rates.items():
            if len(successes) >= 10:  # Minimum samples
                success_rate = np.mean(successes)
                if success_rate > 0.7:
                    # Increase weight for successful optimization types
                    self.optimization_weights['performance_impact'] *= 1.1
                elif success_rate < 0.3:
                    # Decrease weight for unsuccessful types
                    self.optimization_weights['performance_impact'] *= 0.9
        
        # Normalize weights
        total_weight = sum(self.optimization_weights.values())
        for key in self.optimization_weights:
            self.optimization_weights[key] /= total_weight
        
        logger.info("Optimization parameters updated", new_weights=self.optimization_weights)
    
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics."""
        
        if not self.lifecycle_metrics:
            return {}
        
        # Calculate average lifespan
        lifespans = []
        for metrics in self.lifecycle_metrics.values():
            if metrics.current_stage not in [LifecycleStage.CREATION, LifecycleStage.ACTIVE_USE]:
                lifespans.append(metrics.age_days)
        
        avg_lifespan = np.mean(lifespans) if lifespans else 0.0
        self.performance_metrics.average_memory_lifespan = avg_lifespan
        
        # Calculate optimization success rate
        recent_recs = self.optimization_history[-50:] if self.optimization_history else []
        if recent_recs:
            # Simulate success rate based on confidence scores
            simulated_successes = [rec.confidence > 0.6 for rec in recent_recs]
            success_rate = np.mean(simulated_successes)
            self.performance_metrics.optimization_success_rate = success_rate
        
        return {
            'performance_metrics': self.performance_metrics.model_dump(),
            'total_tracked_memories': len(self.lifecycle_metrics),
            'stage_transitions_recorded': len(self.stage_transitions),
            'optimization_recommendations': len(self.optimization_history),
            'models_trained': self.models_trained,
            'aging_algorithm_params': self.aging_algorithm.aging_params,
            'optimization_weights': self.optimization_weights,
            'stage_thresholds': {k.value: v for k, v in self.stage_thresholds.items()}
        }
    
    async def batch_lifecycle_analysis(
        self,
        memory_ids: List[str],
        batch_size: int = 100
    ) -> Dict[str, LifecycleAnalysisResult]:
        """Perform lifecycle analysis on a batch of memories efficiently."""
        try:
            results = {}
            
            # Process in batches for efficiency
            for i in range(0, len(memory_ids), batch_size):
                batch = memory_ids[i:i + batch_size]
                batch_tasks = []
                
                for memory_id in batch:
                    task = asyncio.create_task(self.analyze_memory_lifecycle(memory_id))
                    batch_tasks.append((memory_id, task))
                
                # Execute batch concurrently
                for memory_id, task in batch_tasks:
                    try:
                        result = await task
                        results[memory_id] = result
                    except Exception as e:
                        logger.error(
                            "Batch lifecycle analysis failed for memory",
                            memory_id=memory_id,
                            error=str(e)
                        )
                        # Continue with other memories
                        continue
                
                # Small delay between batches to prevent overwhelming
                if i + batch_size < len(memory_ids):
                    await asyncio.sleep(0.1)
            
            logger.info(
                "Batch lifecycle analysis completed",
                total_memories=len(memory_ids),
                successful_analyses=len(results),
                success_rate=len(results) / len(memory_ids)
            )
            
            return results
            
        except Exception as e:
            logger.error("Batch lifecycle analysis failed", error=str(e))
            return {}
    
    async def export_lifecycle_report(
        self,
        user_id: str,
        format: str = "json",
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """Export comprehensive lifecycle report for a user."""
        try:
            # Get all user memories for analysis
            user_memories = await self._get_user_memories(user_id)
            
            if not user_memories:
                return {
                    "user_id": user_id,
                    "error": "No memories found for user",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Perform batch analysis
            memory_ids = [m["id"] for m in user_memories]
            lifecycle_results = await self.batch_lifecycle_analysis(memory_ids)
            
            # Aggregate statistics
            stage_distribution = defaultdict(int)
            avg_stage_durations = defaultdict(list)
            transition_patterns = defaultdict(int)
            
            for result in lifecycle_results.values():
                stage_distribution[result.current_stage.value] += 1
                
                # Collect stage durations for averages
                for stage, duration in result.stage_durations.items():
                    avg_stage_durations[stage.value].append(duration)
                
                # Track transition patterns
                for transition in result.transition_history:
                    pattern = f"{transition.from_stage.value}->{transition.to_stage.value}"
                    transition_patterns[pattern] += 1
            
            # Calculate averages
            avg_durations = {}
            for stage, durations in avg_stage_durations.items():
                if durations:
                    avg_durations[stage] = {
                        "mean_hours": np.mean(durations),
                        "median_hours": np.median(durations),
                        "std_dev": np.std(durations),
                        "min_hours": min(durations),
                        "max_hours": max(durations)
                    }
            
            # Generate predictions if requested
            predictions = {}
            if include_predictions:
                for memory_id, result in lifecycle_results.items():
                    try:
                        prediction = await self.predict_next_transition(memory_id)
                        predictions[memory_id] = {
                            "next_stage": prediction.predicted_stage.value,
                            "confidence": prediction.confidence,
                            "estimated_days": prediction.time_to_transition_days,
                            "factors": prediction.key_factors
                        }
                    except Exception as e:
                        logger.warning(f"Failed to generate prediction for {memory_id}: {e}")
            
            # Create comprehensive report
            report = {
                "user_id": user_id,
                "generated_at": datetime.utcnow().isoformat(),
                "format": format,
                "summary": {
                    "total_memories": len(memory_ids),
                    "analyzed_memories": len(lifecycle_results),
                    "analysis_success_rate": len(lifecycle_results) / len(memory_ids),
                    "stage_distribution": dict(stage_distribution),
                    "most_common_transitions": dict(
                        sorted(transition_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
                    )
                },
                "stage_analytics": {
                    "average_durations": avg_durations,
                    "transition_matrix": self._build_transition_matrix(lifecycle_results.values()),
                    "retention_rates": self._calculate_retention_rates(lifecycle_results.values())
                },
                "individual_analyses": lifecycle_results,
                "predictions": predictions if include_predictions else {},
                "recommendations": await self._generate_user_recommendations(user_id, lifecycle_results)
            }
            
            # Format output based on requested format
            if format.lower() == "csv":
                return self._convert_report_to_csv(report)
            elif format.lower() == "summary":
                return self._extract_report_summary(report)
            else:
                return report
                
        except Exception as e:
            logger.error("Export lifecycle report failed", user_id=user_id, error=str(e))
            return {
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _build_transition_matrix(self, results: List[LifecycleAnalysisResult]) -> Dict[str, Dict[str, float]]:
        """Build transition probability matrix from lifecycle results."""
        transitions = defaultdict(lambda: defaultdict(int))
        stage_counts = defaultdict(int)
        
        for result in results:
            for transition in result.transition_history:
                from_stage = transition.from_stage.value
                to_stage = transition.to_stage.value
                transitions[from_stage][to_stage] += 1
                stage_counts[from_stage] += 1
        
        # Convert to probabilities
        probability_matrix = {}
        for from_stage, to_stages in transitions.items():
            probability_matrix[from_stage] = {}
            total = stage_counts[from_stage]
            for to_stage, count in to_stages.items():
                probability_matrix[from_stage][to_stage] = count / total if total > 0 else 0.0
        
        return probability_matrix
    
    def _calculate_retention_rates(self, results: List[LifecycleAnalysisResult]) -> Dict[str, float]:
        """Calculate retention rates for each lifecycle stage."""
        stage_retention = defaultdict(list)
        
        for result in results:
            current_stage = result.current_stage.value
            time_in_stage = result.stage_durations.get(result.current_stage, 0)
            
            # Calculate retention score based on time in stage vs expected duration
            expected_duration = self.stage_thresholds.get(result.current_stage, 168)  # 1 week default
            retention_score = min(1.0, time_in_stage / expected_duration)
            stage_retention[current_stage].append(retention_score)
        
        # Calculate average retention rate per stage
        retention_rates = {}
        for stage, scores in stage_retention.items():
            if scores:
                retention_rates[stage] = np.mean(scores)
            else:
                retention_rates[stage] = 0.0
        
        return retention_rates
    
    async def _generate_user_recommendations(
        self,
        user_id: str,
        lifecycle_results: Dict[str, LifecycleAnalysisResult]
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on lifecycle analysis."""
        try:
            recommendations = []
            
            # Analyze patterns
            stale_memories = []
            underutilized_memories = []
            high_value_memories = []
            
            for memory_id, result in lifecycle_results.items():
                # Check for stale memories
                if result.current_stage in [MemoryStage.ARCHIVED, MemoryStage.DORMANT]:
                    if result.stage_durations.get(result.current_stage, 0) > 720:  # 30 days
                        stale_memories.append(memory_id)
                
                # Check for underutilized memories
                if (result.access_frequency < 0.1 and 
                    result.current_stage == MemoryStage.ACTIVE):
                    underutilized_memories.append(memory_id)
                
                # Identify high-value memories
                if (result.access_frequency > 0.5 and 
                    result.current_stage in [MemoryStage.ARCHIVED, MemoryStage.DORMANT]):
                    high_value_memories.append(memory_id)
            
            # Generate recommendations
            if stale_memories:
                recommendations.append({
                    "type": "cleanup",
                    "priority": "medium",
                    "title": "Clean up stale memories",
                    "description": f"Consider reviewing {len(stale_memories)} memories that have been inactive for over 30 days",
                    "memory_ids": stale_memories[:10],  # Limit to top 10
                    "action": "review_for_deletion"
                })
            
            if underutilized_memories:
                recommendations.append({
                    "type": "optimization",
                    "priority": "low",
                    "title": "Optimize underutilized memories",
                    "description": f"Archive {len(underutilized_memories)} rarely accessed memories to improve performance",
                    "memory_ids": underutilized_memories[:10],
                    "action": "auto_archive"
                })
            
            if high_value_memories:
                recommendations.append({
                    "type": "recovery",
                    "priority": "high",
                    "title": "Restore valuable archived memories",
                    "description": f"Consider restoring {len(high_value_memories)} high-value memories that were archived",
                    "memory_ids": high_value_memories[:10],
                    "action": "restore_to_active"
                })
            
            # Pattern-based recommendations
            patterns = await self._analyze_usage_patterns(user_id, lifecycle_results)
            if patterns.get("irregular_access"):
                recommendations.append({
                    "type": "pattern",
                    "priority": "medium",
                    "title": "Irregular access pattern detected",
                    "description": "Your memory access patterns suggest optimizing for burst usage",
                    "action": "adjust_preloading_strategy"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to generate user recommendations", user_id=user_id, error=str(e))
            return []
    
    async def _analyze_usage_patterns(
        self,
        user_id: str,
        lifecycle_results: Dict[str, LifecycleAnalysisResult]
    ) -> Dict[str, Any]:
        """Analyze user-specific usage patterns."""
        try:
            access_frequencies = [r.access_frequency for r in lifecycle_results.values()]
            access_times = []
            
            for result in lifecycle_results.values():
                for transition in result.transition_history:
                    access_times.append(transition.timestamp.hour)
            
            patterns = {
                "avg_access_frequency": np.mean(access_frequencies) if access_frequencies else 0,
                "access_frequency_std": np.std(access_frequencies) if access_frequencies else 0,
                "irregular_access": np.std(access_frequencies) > 0.3 if access_frequencies else False,
                "peak_access_hours": [],
                "access_consistency": 0.0
            }
            
            # Analyze peak access hours
            if access_times:
                hour_counts = defaultdict(int)
                for hour in access_times:
                    hour_counts[hour] += 1
                
                # Find peak hours (top 3)
                sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
                patterns["peak_access_hours"] = [hour for hour, count in sorted_hours[:3]]
                
                # Calculate access consistency
                total_accesses = sum(hour_counts.values())
                if total_accesses > 0:
                    hour_variance = np.var(list(hour_counts.values()))
                    patterns["access_consistency"] = 1.0 / (1.0 + hour_variance / total_accesses)
            
            return patterns
            
        except Exception as e:
            logger.error("Usage pattern analysis failed", user_id=user_id, error=str(e))
            return {}
    
    def _convert_report_to_csv(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Convert lifecycle report to CSV format."""
        # This would implement CSV conversion logic
        # For now, return a placeholder
        return {
            "format": "csv",
            "note": "CSV conversion not yet implemented",
            "summary": str(report["summary"])
        }
    
    def _extract_report_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key summary information from full report."""
        return {
            "format": "summary",
            "user_id": report["user_id"],
            "generated_at": report["generated_at"],
            "summary": report["summary"],
            "key_recommendations": report["recommendations"][:3],  # Top 3 recommendations
            "stage_distribution": report["summary"]["stage_distribution"],
            "success_rate": report["summary"]["analysis_success_rate"]
        }
    
    async def _get_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a user (placeholder for integration)."""
        # This would integrate with the memory manager
        # For now, return empty list
        return []


# Export main components
__all__ = [
    "LifecycleOptimizer",
    "MemoryStage",
    "LifecycleTransition", 
    "LifecycleAnalysisResult",
    "TransitionPrediction",
    "AgingAlgorithm",
    "OptimizationRecommendation"
]