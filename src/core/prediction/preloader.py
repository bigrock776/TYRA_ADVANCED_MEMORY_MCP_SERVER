"""
Predictive Preloading System for Enhanced Memory Performance.

This module provides intelligent preloading capabilities using local Markov chains,
n-gram models for query anticipation, priority queuing, and success tracking.
All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable, Deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import json
import math
import heapq
import hashlib

# ML and analytics imports - all local
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from .usage_analyzer import UsageAnalyzer, UsagePattern
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class PreloadingStrategy(str, Enum):
    """Types of preloading strategies."""
    MARKOV_CHAIN = "markov_chain"
    NGRAM_MODEL = "ngram_model"
    USAGE_PATTERN = "usage_pattern"
    TIME_BASED = "time_based"
    COLLABORATIVE = "collaborative"
    HYBRID = "hybrid"


class PredictionType(str, Enum):
    """Types of predictions for preloading."""
    NEXT_ACCESS = "next_access"
    SEQUENCE_CONTINUATION = "sequence_continuation"
    TIME_WINDOW_ACCESS = "time_window_access"
    USER_SESSION_PATTERN = "user_session_pattern"
    RELATED_MEMORIES = "related_memories"


class CacheLevel(str, Enum):
    """Cache levels for preloaded memories."""
    L1_HOT = "l1_hot"        # Most likely to be accessed
    L2_WARM = "l2_warm"      # Likely to be accessed
    L3_COLD = "l3_cold"      # Might be accessed
    PREFETCH = "prefetch"    # Speculative preload


@dataclass
class CachePrediction:
    """Represents a cache prediction for preloading."""
    memory_id: str
    user_id: Optional[str]
    prediction_type: PredictionType
    strategy: PreloadingStrategy
    confidence: float
    priority: float
    predicted_access_time: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __lt__(self, other):
        """Comparison for priority queue (higher priority = lower value)."""
        return self.priority > other.priority


@dataclass
class MarkovState:
    """State in Markov chain for memory access prediction."""
    state_id: str
    memory_ids: Tuple[str, ...]  # Sequence of memory IDs
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.state_id, self.memory_ids))
    
    def __eq__(self, other):
        if not isinstance(other, MarkovState):
            return False
        return self.state_id == other.state_id and self.memory_ids == other.memory_ids


@dataclass
class NGramModel:
    """N-gram model for query/access pattern prediction."""
    n: int
    frequencies: Dict[Tuple[str, ...], Counter] = field(default_factory=dict)
    total_sequences: int = 0
    
    def add_sequence(self, sequence: List[str]):
        """Add a sequence to the n-gram model."""
        if len(sequence) < self.n:
            return
        
        for i in range(len(sequence) - self.n + 1):
            ngram = tuple(sequence[i:i + self.n - 1])
            next_item = sequence[i + self.n - 1]
            
            if ngram not in self.frequencies:
                self.frequencies[ngram] = Counter()
            
            self.frequencies[ngram][next_item] += 1
            self.total_sequences += 1
    
    def predict_next(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next items given context."""
        if len(context) < self.n - 1:
            return []
        
        ngram = tuple(context[-(self.n - 1):])
        
        if ngram not in self.frequencies:
            return []
        
        # Get predictions with probabilities
        total_count = sum(self.frequencies[ngram].values())
        predictions = []
        
        for item, count in self.frequencies[ngram].most_common(top_k):
            probability = count / total_count
            predictions.append((item, probability))
        
        return predictions


class MarkovChainPredictor:
    """
    Markov chain-based predictor for memory access patterns.
    
    Features:
    - Multi-order Markov chains for different prediction horizons
    - Context-aware state transitions
    - User-specific and global models
    - Adaptive model training and updating
    """
    
    def __init__(self, order: int = 2, max_states: int = 10000):
        """
        Initialize Markov chain predictor.
        
        Args:
            order: Order of Markov chain (number of previous states to consider)
            max_states: Maximum number of states to maintain
        """
        self.order = order
        self.max_states = max_states
        
        # State transition matrices
        self.transitions: Dict[MarkovState, Dict[str, float]] = {}
        self.state_counts: Dict[MarkovState, int] = defaultdict(int)
        
        # User-specific models
        self.user_transitions: Dict[str, Dict[MarkovState, Dict[str, float]]] = defaultdict(dict)
        
        # Recent sequences for model updates
        self.recent_sequences: Deque[List[str]] = deque(maxlen=1000)
        
    def add_sequence(self, sequence: List[str], user_id: Optional[str] = None):
        """Add an access sequence to train the model."""
        if len(sequence) <= self.order:
            return
        
        self.recent_sequences.append(sequence)
        
        # Create states and transitions
        for i in range(len(sequence) - self.order):
            # Create state from previous items
            state_items = tuple(sequence[i:i + self.order])
            state = MarkovState(
                state_id=hashlib.md5(str(state_items).encode()).hexdigest()[:8],
                memory_ids=state_items
            )
            
            next_item = sequence[i + self.order]
            
            # Update global transitions
            self._update_transitions(self.transitions, state, next_item)
            
            # Update user-specific transitions
            if user_id:
                user_trans = self.user_transitions[user_id]
                self._update_transitions(user_trans, state, next_item)
    
    def _update_transitions(
        self, 
        transitions: Dict[MarkovState, Dict[str, float]], 
        state: MarkovState, 
        next_item: str
    ):
        """Update transition probabilities."""
        if state not in transitions:
            transitions[state] = defaultdict(float)
        
        # Increment count
        transitions[state][next_item] += 1
        self.state_counts[state] += 1
        
        # Normalize probabilities for this state
        total_count = sum(transitions[state].values())
        for item in transitions[state]:
            transitions[state][item] = transitions[state][item] / total_count
    
    def predict(
        self, 
        context: List[str], 
        user_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict next memories based on context."""
        if len(context) < self.order:
            return []
        
        # Create state from context
        state_items = tuple(context[-self.order:])
        state = MarkovState(
            state_id=hashlib.md5(str(state_items).encode()).hexdigest()[:8],
            memory_ids=state_items
        )
        
        # Try user-specific model first
        if user_id and user_id in self.user_transitions:
            user_trans = self.user_transitions[user_id]
            if state in user_trans:
                predictions = sorted(
                    user_trans[state].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                return predictions
        
        # Fallback to global model
        if state in self.transitions:
            predictions = sorted(
                self.transitions[state].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            return predictions
        
        return []
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'total_states': len(self.transitions),
            'user_models': len(self.user_transitions),
            'average_transitions_per_state': (
                np.mean([len(trans) for trans in self.transitions.values()])
                if self.transitions else 0
            ),
            'recent_sequences': len(self.recent_sequences),
            'model_order': self.order
        }


class QueryAnticipator:
    """
    Query anticipation using n-gram models and pattern recognition.
    
    Features:
    - Multiple n-gram models for different prediction horizons
    - Query pattern learning and recognition
    - Context-aware query suggestions
    - Adaptive model updating
    """
    
    def __init__(self, max_ngram_order: int = 4):
        """Initialize query anticipator."""
        self.max_ngram_order = max_ngram_order
        
        # N-gram models for different orders
        self.ngram_models: Dict[int, NGramModel] = {}
        for n in range(2, max_ngram_order + 1):
            self.ngram_models[n] = NGramModel(n)
        
        # Query patterns and frequencies
        self.query_patterns: Counter = Counter()
        self.query_sequences: List[List[str]] = []
        
        # Temporal patterns
        self.temporal_patterns: Dict[int, Counter] = defaultdict(Counter)  # hour -> queries
        
    def add_query_sequence(self, queries: List[str], timestamp: Optional[datetime] = None):
        """Add a sequence of queries for pattern learning."""
        if not queries:
            return
        
        self.query_sequences.append(queries)
        
        # Add to n-gram models
        for n, model in self.ngram_models.items():
            model.add_sequence(queries)
        
        # Track query patterns
        for query in queries:
            self.query_patterns[query] += 1
        
        # Track temporal patterns
        if timestamp:
            hour = timestamp.hour
            for query in queries:
                self.temporal_patterns[hour][query] += 1
    
    def anticipate_queries(
        self,
        context: List[str],
        current_time: Optional[datetime] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Anticipate likely next queries based on context."""
        predictions = {}
        
        # Get predictions from different n-gram orders
        for n, model in self.ngram_models.items():
            if len(context) >= n - 1:
                ngram_predictions = model.predict_next(context, top_k * 2)
                
                # Weight predictions by model order (higher order = higher weight)
                weight = n / self.max_ngram_order
                
                for query, prob in ngram_predictions:
                    if query not in predictions:
                        predictions[query] = 0
                    predictions[query] += prob * weight
        
        # Add temporal bias
        if current_time:
            hour = current_time.hour
            temporal_queries = self.temporal_patterns.get(hour, Counter())
            
            if temporal_queries:
                total_temporal = sum(temporal_queries.values())
                
                for query, count in temporal_queries.items():
                    temporal_prob = count / total_temporal
                    
                    if query not in predictions:
                        predictions[query] = 0
                    predictions[query] += temporal_prob * 0.3  # Temporal weight
        
        # Sort and return top predictions
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return sorted_predictions
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query anticipation statistics."""
        return {
            'total_queries_learned': sum(self.query_patterns.values()),
            'unique_queries': len(self.query_patterns),
            'query_sequences': len(self.query_sequences),
            'ngram_models': {
                n: {
                    'total_ngrams': len(model.frequencies),
                    'total_sequences': model.total_sequences
                }
                for n, model in self.ngram_models.items()
            },
            'temporal_patterns': len(self.temporal_patterns)
        }


class PriorityQueue:
    """
    Priority queue for managing preloading tasks.
    
    Features:
    - Multiple priority levels
    - Dynamic priority adjustment
    - Resource-aware scheduling
    - Performance tracking
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize priority queue."""
        self.max_size = max_size
        self.queue: List[CachePrediction] = []
        self.in_queue: Set[str] = set()  # Track memory IDs in queue
        
        # Priority adjustment factors
        self.priority_factors = {
            'confidence': 0.4,
            'recency': 0.3,
            'popularity': 0.2,
            'user_preference': 0.1
        }
    
    def add_prediction(self, prediction: CachePrediction) -> bool:
        """Add a prediction to the queue."""
        if prediction.memory_id in self.in_queue:
            return False  # Already in queue
        
        if len(self.queue) >= self.max_size:
            # Remove lowest priority item
            if self.queue and prediction.priority > self.queue[0].priority:
                removed = heapq.heappop(self.queue)
                self.in_queue.discard(removed.memory_id)
            else:
                return False  # Queue full and new item has low priority
        
        heapq.heappush(self.queue, prediction)
        self.in_queue.add(prediction.memory_id)
        return True
    
    def get_next_prediction(self) -> Optional[CachePrediction]:
        """Get the highest priority prediction."""
        if not self.queue:
            return None
        
        prediction = heapq.heappop(self.queue)
        self.in_queue.discard(prediction.memory_id)
        return prediction
    
    def get_predictions_by_priority(self, count: int) -> List[CachePrediction]:
        """Get multiple predictions ordered by priority."""
        predictions = []
        temp_removed = []
        
        for _ in range(min(count, len(self.queue))):
            if self.queue:
                prediction = heapq.heappop(self.queue)
                self.in_queue.discard(prediction.memory_id)
                predictions.append(prediction)
                temp_removed.append(prediction)
        
        # Put back remaining predictions
        for pred in temp_removed[len(predictions):]:
            heapq.heappush(self.queue, pred)
            self.in_queue.add(pred.memory_id)
        
        return predictions
    
    def update_priorities(self, factor_weights: Dict[str, float]):
        """Update priority calculation factors."""
        self.priority_factors.update(factor_weights)
        
        # Recalculate priorities for existing predictions
        temp_predictions = []
        while self.queue:
            prediction = heapq.heappop(self.queue)
            self.in_queue.discard(prediction.memory_id)
            temp_predictions.append(prediction)
        
        # Recalculate and re-add
        for prediction in temp_predictions:
            # Recalculate priority based on new factors
            prediction.priority = self._calculate_dynamic_priority(prediction)
            heapq.heappush(self.queue, prediction)
            self.in_queue.add(prediction.memory_id)
    
    def _calculate_dynamic_priority(self, prediction: CachePrediction) -> float:
        """Calculate dynamic priority for a prediction."""
        priority = prediction.confidence * self.priority_factors.get('confidence', 0.4)
        
        # Add time-based urgency
        time_until_access = (prediction.predicted_access_time - datetime.utcnow()).total_seconds() / 3600
        urgency = max(0, 1.0 - time_until_access / 24.0)  # More urgent if access predicted soon
        priority += urgency * 0.3
        
        # Add context-specific factors
        if 'popularity' in prediction.context:
            priority += prediction.context['popularity'] * self.priority_factors.get('popularity', 0.2)
        
        if 'user_preference' in prediction.context:
            priority += prediction.context['user_preference'] * self.priority_factors.get('user_preference', 0.1)
        
        return priority
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        if not self.queue:
            return {
                'size': 0,
                'max_size': self.max_size,
                'utilization': 0.0
            }
        
        priorities = [pred.priority for pred in self.queue]
        strategies = [pred.strategy.value for pred in self.queue]
        
        return {
            'size': len(self.queue),
            'max_size': self.max_size,
            'utilization': len(self.queue) / self.max_size,
            'avg_priority': np.mean(priorities),
            'min_priority': min(priorities),
            'max_priority': max(priorities),
            'strategy_distribution': dict(Counter(strategies))
        }


class SuccessTracker:
    """
    Tracks preloading success rates and performance metrics.
    
    Features:
    - Hit/miss rate tracking
    - Strategy performance comparison
    - Adaptive threshold adjustment
    - Performance analytics
    """
    
    def __init__(self, window_size: int = 1000):
        """Initialize success tracker."""
        self.window_size = window_size
        
        # Tracking data
        self.predictions: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.hits: Deque[bool] = deque(maxlen=window_size)
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'hits': 0, 'total': 0, 'avg_confidence': 0.0}
        )
        
        # Performance metrics
        self.total_predictions = 0
        self.total_hits = 0
        self.total_misses = 0
        
    def record_prediction(
        self,
        prediction: CachePrediction,
        was_hit: bool,
        actual_access_time: Optional[datetime] = None
    ):
        """Record a prediction result."""
        prediction_data = {
            'memory_id': prediction.memory_id,
            'strategy': prediction.strategy.value,
            'confidence': prediction.confidence,
            'predicted_time': prediction.predicted_access_time,
            'actual_time': actual_access_time,
            'was_hit': was_hit,
            'recorded_at': datetime.utcnow()
        }
        
        self.predictions.append(prediction_data)
        self.hits.append(was_hit)
        
        # Update strategy performance
        strategy = prediction.strategy.value
        self.strategy_performance[strategy]['total'] += 1
        if was_hit:
            self.strategy_performance[strategy]['hits'] += 1
        
        # Update running confidence average
        current_avg = self.strategy_performance[strategy]['avg_confidence']
        total = self.strategy_performance[strategy]['total']
        new_avg = ((current_avg * (total - 1)) + prediction.confidence) / total
        self.strategy_performance[strategy]['avg_confidence'] = new_avg
        
        # Update totals
        self.total_predictions += 1
        if was_hit:
            self.total_hits += 1
        else:
            self.total_misses += 1
    
    def get_hit_rate(self, strategy: Optional[str] = None) -> float:
        """Get hit rate for overall or specific strategy."""
        if strategy:
            perf = self.strategy_performance.get(strategy, {'hits': 0, 'total': 0})
            return perf['hits'] / perf['total'] if perf['total'] > 0 else 0.0
        
        return len([h for h in self.hits if h]) / len(self.hits) if self.hits else 0.0
    
    def get_strategy_rankings(self) -> List[Tuple[str, float]]:
        """Get strategies ranked by performance."""
        rankings = []
        
        for strategy, perf in self.strategy_performance.items():
            if perf['total'] >= 10:  # Minimum samples for reliable ranking
                hit_rate = perf['hits'] / perf['total']
                rankings.append((strategy, hit_rate))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'overall_hit_rate': self.get_hit_rate(),
            'total_predictions': self.total_predictions,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'strategy_performance': dict(self.strategy_performance),
            'strategy_rankings': self.get_strategy_rankings(),
            'recent_hit_rate': (
                len([h for h in list(self.hits)[-100:] if h]) / 
                min(100, len(self.hits))
            ) if self.hits else 0.0
        }


class PredictivePreloader:
    """
    Main predictive preloading system for enhanced memory performance.
    
    Features:
    - Multiple prediction strategies (Markov chains, n-grams, usage patterns)
    - Intelligent cache management with priority queuing
    - Success tracking and adaptive optimization
    - Local ML training for pattern recognition
    - Performance monitoring and analytics
    """
    
    def __init__(
        self,
        usage_analyzer: Optional[UsageAnalyzer] = None,
        redis_cache: Optional[RedisCache] = None,
        cache_size_mb: int = 100,
        max_preload_items: int = 500
    ):
        """
        Initialize the predictive preloader.
        
        Args:
            usage_analyzer: Optional usage analyzer for pattern insights
            redis_cache: Optional Redis cache for persistence
            cache_size_mb: Cache size limit in MB
            max_preload_items: Maximum items to preload
        """
        self.usage_analyzer = usage_analyzer
        self.redis_cache = redis_cache
        self.cache_size_mb = cache_size_mb
        self.max_preload_items = max_preload_items
        
        # Core components
        self.markov_predictor = MarkovChainPredictor()
        self.query_anticipator = QueryAnticipator()
        self.priority_queue = PriorityQueue(max_preload_items)
        self.success_tracker = SuccessTracker()
        
        # Cache management
        self.preloaded_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_usage: Dict[str, datetime] = {}
        
        # Strategy weights (can be adjusted based on performance)
        self.strategy_weights = {
            PreloadingStrategy.MARKOV_CHAIN: 0.3,
            PreloadingStrategy.NGRAM_MODEL: 0.2,
            PreloadingStrategy.USAGE_PATTERN: 0.3,
            PreloadingStrategy.TIME_BASED: 0.1,
            PreloadingStrategy.COLLABORATIVE: 0.1
        }
        
        # Performance tracking
        self.stats = {
            'predictions_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'preloads_executed': 0,
            'cache_evictions': 0,
            'total_queries_processed': 0
        }
        
        # ML components
        self.ml_model: Optional[RandomForestRegressor] = None
        self.feature_scaler = StandardScaler()
        self.is_ml_trained = False
        
        logger.info(
            "Predictive preloader initialized",
            cache_size_mb=cache_size_mb,
            max_preload_items=max_preload_items
        )
    
    async def process_access_sequence(
        self,
        memory_ids: List[str],
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """Process a sequence of memory accesses for pattern learning."""
        if not memory_ids:
            return
        
        timestamp = timestamp or datetime.utcnow()
        
        # Train Markov chain predictor
        self.markov_predictor.add_sequence(memory_ids, user_id)
        
        # Update query anticipator if this represents query sequence
        self.query_anticipator.add_query_sequence(memory_ids, timestamp)
        
        # Generate predictions based on this sequence
        await self._generate_predictions_from_sequence(memory_ids, user_id, timestamp)
        
        self.stats['total_queries_processed'] += len(memory_ids)
    
    async def _generate_predictions_from_sequence(
        self,
        memory_ids: List[str],
        user_id: Optional[str],
        timestamp: datetime
    ):
        """Generate preloading predictions from access sequence."""
        predictions = []
        
        # Markov chain predictions
        markov_predictions = self.markov_predictor.predict(memory_ids, user_id, top_k=5)
        for memory_id, confidence in markov_predictions:
            prediction = CachePrediction(
                memory_id=memory_id,
                user_id=user_id,
                prediction_type=PredictionType.NEXT_ACCESS,
                strategy=PreloadingStrategy.MARKOV_CHAIN,
                confidence=confidence * self.strategy_weights[PreloadingStrategy.MARKOV_CHAIN],
                priority=confidence,
                predicted_access_time=timestamp + timedelta(minutes=30),  # Predict access in 30 min
                context={'sequence_context': memory_ids[-3:]}
            )
            predictions.append(prediction)
        
        # N-gram model predictions
        ngram_predictions = self.query_anticipator.anticipate_queries(
            memory_ids, timestamp, top_k=3
        )
        for memory_id, confidence in ngram_predictions:
            prediction = CachePrediction(
                memory_id=memory_id,
                user_id=user_id,
                prediction_type=PredictionType.SEQUENCE_CONTINUATION,
                strategy=PreloadingStrategy.NGRAM_MODEL,
                confidence=confidence * self.strategy_weights[PreloadingStrategy.NGRAM_MODEL],
                priority=confidence,
                predicted_access_time=timestamp + timedelta(minutes=15),
                context={'ngram_context': memory_ids[-2:]}
            )
            predictions.append(prediction)
        
        # Usage pattern predictions
        if self.usage_analyzer:
            usage_predictions = await self._generate_usage_pattern_predictions(
                memory_ids, user_id, timestamp
            )
            predictions.extend(usage_predictions)
        
        # Add predictions to priority queue
        for prediction in predictions:
            self.priority_queue.add_prediction(prediction)
            self.stats['predictions_generated'] += 1
    
    async def _generate_usage_pattern_predictions(
        self,
        memory_ids: List[str],
        user_id: Optional[str],
        timestamp: datetime
    ) -> List[CachePrediction]:
        """Generate predictions based on usage patterns."""
        predictions = []
        
        try:
            # Get usage patterns for recent memories
            patterns = await self.usage_analyzer.analyze_patterns()
            
            for memory_id in memory_ids[-3:]:  # Focus on recent accesses
                pattern = patterns.get(memory_id)
                if not pattern:
                    continue
                
                # Predict based on pattern type and metrics
                confidence = pattern.confidence * 0.8  # Slight discount
                
                if 'next_access_time' in pattern.predictions:
                    predicted_time_str = pattern.predictions['next_access_time']
                    predicted_time = datetime.fromisoformat(predicted_time_str)
                else:
                    # Estimate based on pattern metrics
                    mean_interval = pattern.metrics.get('mean_interval', 24)  # hours
                    predicted_time = timestamp + timedelta(hours=mean_interval)
                
                prediction = CachePrediction(
                    memory_id=memory_id,
                    user_id=user_id,
                    prediction_type=PredictionType.TIME_WINDOW_ACCESS,
                    strategy=PreloadingStrategy.USAGE_PATTERN,
                    confidence=confidence * self.strategy_weights[PreloadingStrategy.USAGE_PATTERN],
                    priority=confidence,
                    predicted_access_time=predicted_time,
                    context={
                        'pattern_type': pattern.pattern_type.value,
                        'access_frequency': pattern.access_frequency.value
                    }
                )
                predictions.append(prediction)
                
        except Exception as e:
            logger.warning(f"Usage pattern prediction failed: {e}")
        
        return predictions
    
    async def execute_preloading(self, batch_size: int = 10) -> int:
        """Execute preloading based on priority queue."""
        preloaded_count = 0
        
        # Get high-priority predictions
        predictions = self.priority_queue.get_predictions_by_priority(batch_size)
        
        for prediction in predictions:
            if await self._should_preload(prediction):
                success = await self._preload_memory(prediction)
                if success:
                    preloaded_count += 1
                    self.stats['preloads_executed'] += 1
        
        # Cleanup old cache entries
        await self._cleanup_cache()
        
        return preloaded_count
    
    async def _should_preload(self, prediction: CachePrediction) -> bool:
        """Determine if memory should be preloaded."""
        # Check if already in cache
        if prediction.memory_id in self.preloaded_cache:
            return False
        
        # Check confidence threshold
        if prediction.confidence < 0.3:
            return False
        
        # Check cache space
        if len(self.preloaded_cache) >= self.max_preload_items:
            # Only preload if higher priority than lowest in cache
            if self.preloaded_cache:
                lowest_priority = min(
                    item.get('priority', 0) 
                    for item in self.preloaded_cache.values()
                )
                if prediction.priority <= lowest_priority:
                    return False
        
        # Check if prediction time is reasonable
        time_until_predicted = (prediction.predicted_access_time - datetime.utcnow()).total_seconds()
        if time_until_predicted < 0 or time_until_predicted > 86400:  # 24 hours
            return False
        
        return True
    
    async def _preload_memory(self, prediction: CachePrediction) -> bool:
        """Preload memory into cache."""
        try:
            # In a real implementation, this would load memory from storage
            # For now, we'll simulate by storing prediction metadata
            
            cache_entry = {
                'memory_id': prediction.memory_id,
                'user_id': prediction.user_id,
                'loaded_at': datetime.utcnow(),
                'predicted_access_time': prediction.predicted_access_time,
                'strategy': prediction.strategy.value,
                'confidence': prediction.confidence,
                'priority': prediction.priority,
                'prediction_context': prediction.context
            }
            
            # Make room if necessary
            if len(self.preloaded_cache) >= self.max_preload_items:
                await self._evict_lowest_priority()
            
            # Add to cache
            self.preloaded_cache[prediction.memory_id] = cache_entry
            self.cache_usage[prediction.memory_id] = datetime.utcnow()
            
            logger.debug(f"Preloaded memory {prediction.memory_id} with strategy {prediction.strategy.value}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to preload memory {prediction.memory_id}: {e}")
            return False
    
    async def _evict_lowest_priority(self):
        """Evict the lowest priority item from cache."""
        if not self.preloaded_cache:
            return
        
        # Find item with lowest priority
        lowest_memory_id = min(
            self.preloaded_cache.keys(),
            key=lambda mid: self.preloaded_cache[mid].get('priority', 0)
        )
        
        # Remove from cache
        del self.preloaded_cache[lowest_memory_id]
        self.cache_usage.pop(lowest_memory_id, None)
        
        self.stats['cache_evictions'] += 1
        
        logger.debug(f"Evicted memory {lowest_memory_id} from preload cache")
    
    async def _cleanup_cache(self):
        """Clean up old cache entries."""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for memory_id, loaded_time in self.cache_usage.items():
            # Remove entries older than 2 hours
            if (current_time - loaded_time).total_seconds() > 7200:
                expired_keys.append(memory_id)
        
        for key in expired_keys:
            self.preloaded_cache.pop(key, None)
            self.cache_usage.pop(key, None)
            self.stats['cache_evictions'] += 1
    
    async def check_cache_hit(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """Check if memory access was a cache hit and record result."""
        is_hit = memory_id in self.preloaded_cache
        
        if is_hit:
            self.stats['cache_hits'] += 1
            
            # Update cache usage
            self.cache_usage[memory_id] = datetime.utcnow()
            
            # Record success for the prediction that led to this preload
            cache_entry = self.preloaded_cache[memory_id]
            
            # Create a prediction object for success tracking
            prediction = CachePrediction(
                memory_id=memory_id,
                user_id=cache_entry.get('user_id'),
                prediction_type=PredictionType.NEXT_ACCESS,
                strategy=PreloadingStrategy(cache_entry['strategy']),
                confidence=cache_entry['confidence'],
                priority=cache_entry['priority'],
                predicted_access_time=cache_entry['predicted_access_time']
            )
            
            self.success_tracker.record_prediction(prediction, True, datetime.utcnow())
            
            logger.debug(f"Cache hit for memory {memory_id}")
            
        else:
            self.stats['cache_misses'] += 1
        
        return is_hit
    
    async def optimize_strategies(self):
        """Optimize strategy weights based on performance."""
        strategy_rankings = self.success_tracker.get_strategy_rankings()
        
        if len(strategy_rankings) < 2:
            return  # Need multiple strategies to optimize
        
        # Adjust weights based on performance
        total_performance = sum(performance for _, performance in strategy_rankings)
        
        if total_performance > 0:
            for strategy_name, performance in strategy_rankings:
                try:
                    strategy = PreloadingStrategy(strategy_name)
                    # Increase weight for better performing strategies
                    new_weight = (performance / total_performance) * 0.8 + 0.1  # Min 10% weight
                    self.strategy_weights[strategy] = new_weight
                except ValueError:
                    continue  # Skip unknown strategies
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        for strategy in self.strategy_weights:
            self.strategy_weights[strategy] /= total_weight
        
        logger.info("Strategy weights optimized", new_weights=self.strategy_weights)
    
    async def train_ml_model(self, training_data: List[Dict[str, Any]]):
        """Train ML model for enhanced prediction accuracy."""
        if len(training_data) < 100:
            logger.warning("Insufficient training data for ML model")
            return
        
        try:
            # Extract features and targets
            X = []
            y = []
            
            for data in training_data:
                features = [
                    data.get('confidence', 0.5),
                    data.get('time_until_access', 3600),  # seconds
                    data.get('sequence_length', 1),
                    data.get('user_activity_level', 0.5),
                    data.get('memory_popularity', 0.5),
                    data.get('pattern_strength', 0.5)
                ]
                
                # Target: actual access probability
                target = 1.0 if data.get('was_accessed', False) else 0.0
                
                X.append(features)
                y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.ml_model.fit(X_scaled, y)
            
            # Evaluate model
            predictions = self.ml_model.predict(X_scaled)
            r2 = r2_score(y, predictions)
            
            self.is_ml_trained = True
            
            logger.info(f"ML model trained with R² score: {r2:.3f}")
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
    
    def get_preloader_stats(self) -> Dict[str, Any]:
        """Get comprehensive preloader statistics."""
        cache_hit_rate = (
            self.stats['cache_hits'] / 
            (self.stats['cache_hits'] + self.stats['cache_misses'])
        ) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'current_cache_size': len(self.preloaded_cache),
            'cache_utilization': len(self.preloaded_cache) / self.max_preload_items,
            'strategy_weights': self.strategy_weights,
            'markov_model_stats': self.markov_predictor.get_model_stats(),
            'query_anticipator_stats': self.query_anticipator.get_query_stats(),
            'priority_queue_stats': self.priority_queue.get_queue_stats(),
            'success_tracker_stats': self.success_tracker.get_performance_stats(),
            'ml_model_trained': self.is_ml_trained
        }