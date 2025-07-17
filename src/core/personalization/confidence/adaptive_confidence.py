"""
Adaptive Confidence Threshold System for Personalized Memory Retrieval.

This module provides comprehensive adaptive confidence threshold capabilities with reinforcement learning,
context-aware adjustment using local models, performance monitoring with local metrics,
and recommendation system. All processing is performed locally with zero external dependencies.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import pickle
import hashlib
import threading
from pathlib import Path

# ML and optimization imports - all local
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
from scipy.optimize import minimize

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ....core.cache.redis_cache import RedisCache
from ....core.utils.config import settings

logger = structlog.get_logger(__name__)


class ConfidenceContext(str, Enum):
    """Types of contexts for confidence adaptation."""
    WORK = "work"                           # Work-related queries
    PERSONAL = "personal"                   # Personal information
    RESEARCH = "research"                   # Research and learning
    CREATIVE = "creative"                   # Creative projects
    TECHNICAL = "technical"                 # Technical documentation
    MEETING = "meeting"                     # Meeting notes and discussions
    PROJECT = "project"                     # Project-specific information
    GENERAL = "general"                     # General queries


class AdaptationStrategy(str, Enum):
    """Strategies for threshold adaptation."""
    CONSERVATIVE = "conservative"           # Err on side of high confidence
    BALANCED = "balanced"                   # Balance precision and recall
    AGGRESSIVE = "aggressive"               # Lower thresholds for more results
    CONTEXTUAL = "contextual"              # Adapt based on context
    REINFORCEMENT = "reinforcement"        # Learn from user feedback
    BAYESIAN = "bayesian"                  # Bayesian optimization


class FeedbackType(str, Enum):
    """Types of user feedback."""
    EXPLICIT_ACCEPT = "explicit_accept"     # User explicitly accepts result
    EXPLICIT_REJECT = "explicit_reject"     # User explicitly rejects result
    IMPLICIT_ACCEPT = "implicit_accept"     # User uses/interacts with result
    IMPLICIT_REJECT = "implicit_reject"     # User ignores/skips result
    CORRECTION = "correction"               # User provides correction
    REFINEMENT = "refinement"              # User refines query


@dataclass
class ConfidenceEvent:
    """Represents a confidence-related event for learning."""
    user_id: str
    query: str
    context: ConfidenceContext
    predicted_confidence: float
    actual_usefulness: float  # 0-1 based on user feedback
    threshold_used: float
    result_shown: bool
    feedback_type: FeedbackType
    timestamp: datetime
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "query": self.query,
            "context": self.context.value,
            "predicted_confidence": self.predicted_confidence,
            "actual_usefulness": self.actual_usefulness,
            "threshold_used": self.threshold_used,
            "result_shown": self.result_shown,
            "feedback_type": self.feedback_type.value,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "metadata": self.metadata
        }


@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholds."""
    user_id: str
    context: ConfidenceContext
    current_threshold: float
    min_threshold: float = 0.1
    max_threshold: float = 0.95
    adaptation_rate: float = 0.1
    confidence_history: List[float] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    strategy: AdaptationStrategy = AdaptationStrategy.BALANCED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "context": self.context.value,
            "current_threshold": self.current_threshold,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "adaptation_rate": self.adaptation_rate,
            "confidence_history": self.confidence_history,
            "performance_history": self.performance_history,
            "last_updated": self.last_updated.isoformat(),
            "strategy": self.strategy.value
        }


class ReinforcementLearner:
    """Implements reinforcement learning for threshold adaptation."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[str, Dict[str, float]] = {}  # state -> action -> value
        self.epsilon = 0.1  # exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Action space: threshold adjustments
        self.actions = [-0.1, -0.05, 0.0, 0.05, 0.1]  # threshold changes
        
    def get_state_key(self, context: ConfidenceContext, recent_performance: float,
                     query_complexity: float, user_experience: float) -> str:
        """Create state key from features."""
        # Discretize continuous values
        perf_bucket = int(recent_performance * 10)  # 0-10
        complexity_bucket = int(query_complexity * 5)  # 0-5
        experience_bucket = int(user_experience * 5)  # 0-5
        
        return f"{context.value}:{perf_bucket}:{complexity_bucket}:{experience_bucket}"
    
    def choose_action(self, state_key: str) -> float:
        """Choose action using epsilon-greedy policy."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(action): 0.0 for action in self.actions}
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            # Choose best action
            q_values = self.q_table[state_key]
            best_action = max(q_values.keys(), key=lambda a: q_values[a])
            return float(best_action)
    
    def update_q_value(self, state_key: str, action: float, reward: float, 
                      next_state_key: str):
        """Update Q-value using temporal difference learning."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(a): 0.0 for a in self.actions}
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {str(a): 0.0 for a in self.actions}
        
        # Current Q-value
        current_q = self.q_table[state_key][str(action)]
        
        # Max Q-value for next state
        max_next_q = max(self.q_table[next_state_key].values())
        
        # Temporal difference update
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        
        self.q_table[state_key][str(action)] += self.learning_rate * td_error
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def calculate_reward(self, predicted_confidence: float, actual_usefulness: float,
                        threshold_used: float, result_shown: bool) -> float:
        """Calculate reward for reinforcement learning."""
        # Base reward: how well confidence predicted usefulness
        confidence_accuracy = 1.0 - abs(predicted_confidence - actual_usefulness)
        
        # Threshold appropriateness
        if result_shown and actual_usefulness > 0.7:
            # Good result was shown
            threshold_reward = 1.0
        elif result_shown and actual_usefulness < 0.3:
            # Poor result was shown (threshold too low)
            threshold_reward = -0.5
        elif not result_shown and actual_usefulness > 0.7:
            # Good result was hidden (threshold too high)
            threshold_reward = -0.8
        else:
            # Appropriate filtering
            threshold_reward = 0.5
        
        # Combine rewards
        total_reward = 0.6 * confidence_accuracy + 0.4 * threshold_reward
        
        return total_reward


class BayesianOptimizer:
    """Implements Bayesian optimization for threshold tuning."""
    
    def __init__(self, bounds: Tuple[float, float] = (0.1, 0.95)):
        self.bounds = bounds
        self.observations_x = []
        self.observations_y = []
        self.gp_model = None
        
    def gaussian_process_predict(self, x_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple GP prediction (using sklearn approximation)."""
        if len(self.observations_x) < 3:
            # Not enough data for GP, return uniform
            mean = np.full(len(x_new), 0.5)
            std = np.full(len(x_new), 0.3)
            return mean, std
        
        # Use polynomial features as approximation to RBF kernel
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import BayesianRidge
        
        poly_features = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly_features.fit_transform(np.array(self.observations_x).reshape(-1, 1))
        
        # Fit Bayesian ridge regression
        model = BayesianRidge()
        model.fit(X_poly, self.observations_y)
        
        # Predict
        X_new_poly = poly_features.transform(x_new.reshape(-1, 1))
        mean, std = model.predict(X_new_poly, return_std=True)
        
        return mean, std
    
    def acquisition_function(self, x: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected improvement acquisition function."""
        mean, std = self.gaussian_process_predict(x)
        
        if len(self.observations_y) == 0:
            return mean
        
        best_y = max(self.observations_y)
        
        # Expected improvement
        z = (mean - best_y - xi) / (std + 1e-9)
        ei = (mean - best_y - xi) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
        
        return ei
    
    def suggest_threshold(self) -> float:
        """Suggest next threshold to try."""
        if len(self.observations_x) < 2:
            # Random exploration
            return np.random.uniform(self.bounds[0], self.bounds[1])
        
        # Optimize acquisition function
        x_candidates = np.linspace(self.bounds[0], self.bounds[1], 100)
        ei_values = self.acquisition_function(x_candidates)
        
        best_idx = np.argmax(ei_values)
        return x_candidates[best_idx]
    
    def add_observation(self, threshold: float, performance: float):
        """Add observation to the optimizer."""
        self.observations_x.append(threshold)
        self.observations_y.append(performance)
        
        # Keep only recent observations to adapt to changing preferences
        if len(self.observations_x) > 50:
            self.observations_x = self.observations_x[-50:]
            self.observations_y = self.observations_y[-50:]


class ContextualPredictor:
    """Predicts optimal confidence thresholds based on context."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        
    def extract_features(self, event: ConfidenceEvent) -> Dict[str, float]:
        """Extract features from confidence event."""
        features = {
            # Query characteristics
            'query_length': len(event.query.split()),
            'query_complexity': self._estimate_query_complexity(event.query),
            'has_keywords': 1.0 if any(kw in event.query.lower() 
                                     for kw in ['how', 'what', 'when', 'where', 'why']) else 0.0,
            
            # Temporal features
            'hour_of_day': event.timestamp.hour / 24.0,
            'day_of_week': event.timestamp.weekday() / 7.0,
            'is_weekend': 1.0 if event.timestamp.weekday() >= 5 else 0.0,
            
            # Context features
            'context_work': 1.0 if event.context == ConfidenceContext.WORK else 0.0,
            'context_personal': 1.0 if event.context == ConfidenceContext.PERSONAL else 0.0,
            'context_research': 1.0 if event.context == ConfidenceContext.RESEARCH else 0.0,
            'context_technical': 1.0 if event.context == ConfidenceContext.TECHNICAL else 0.0,
            
            # Historical features
            'predicted_confidence': event.predicted_confidence,
            'threshold_used': event.threshold_used,
        }
        
        # Add custom features from event
        features.update(event.features)
        
        return features
    
    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity based on linguistic features."""
        words = query.split()
        
        # Basic complexity indicators
        complexity = 0.0
        
        # Length factor
        complexity += min(len(words) / 20.0, 1.0) * 0.3
        
        # Technical terms
        technical_words = ['algorithm', 'function', 'implementation', 'optimization', 
                          'configuration', 'architecture', 'framework', 'protocol']
        tech_count = sum(1 for word in words if word.lower() in technical_words)
        complexity += min(tech_count / len(words), 0.5) * 0.3
        
        # Question complexity
        question_words = ['how', 'why', 'explain', 'compare', 'analyze', 'evaluate']
        if any(qw in query.lower() for qw in question_words):
            complexity += 0.4
        
        return min(complexity, 1.0)
    
    def train_model(self, events: List[ConfidenceEvent], context: ConfidenceContext) -> bool:
        """Train prediction model for a specific context."""
        try:
            # Filter events for this context
            context_events = [e for e in events if e.context == context]
            
            if len(context_events) < 10:
                logger.warning("Not enough events to train model", context=context.value)
                return False
            
            # Extract features and targets
            X = []
            y = []
            
            for event in context_events:
                features = self.extract_features(event)
                X.append(list(features.values()))
                y.append(event.actual_usefulness)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            context_key = context.value
            if context_key not in self.scalers:
                self.scalers[context_key] = StandardScaler()
            
            X_scaled = self.scalers[context_key].fit_transform(X)
            
            # Train ensemble model
            models = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'ridge': Ridge(alpha=1.0)
            }
            
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
            
            # Train best model on full data
            best_model.fit(X_scaled, y)
            self.models[context_key] = best_model
            
            # Calculate feature importance
            if hasattr(best_model, 'feature_importances_'):
                feature_names = list(self.extract_features(context_events[0]).keys())
                importance = dict(zip(feature_names, best_model.feature_importances_))
                self.feature_importance[context_key] = importance
            
            logger.info("Model trained successfully", context=context.value, 
                       score=best_score, events=len(context_events))
            return True
            
        except Exception as e:
            logger.error("Failed to train model", context=context.value, error=str(e))
            return False
    
    def predict_optimal_threshold(self, query: str, context: ConfidenceContext,
                                user_id: str, timestamp: datetime = None) -> float:
        """Predict optimal confidence threshold for given context."""
        context_key = context.value
        
        if context_key not in self.models:
            # Return default threshold
            return 0.7
        
        try:
            timestamp = timestamp or datetime.utcnow()
            
            # Create dummy event for feature extraction
            dummy_event = ConfidenceEvent(
                user_id=user_id,
                query=query,
                context=context,
                predicted_confidence=0.5,
                actual_usefulness=0.5,
                threshold_used=0.5,
                result_shown=True,
                feedback_type=FeedbackType.IMPLICIT_ACCEPT,
                timestamp=timestamp
            )
            
            features = self.extract_features(dummy_event)
            X = np.array(list(features.values())).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scalers[context_key].transform(X)
            
            # Predict usefulness
            predicted_usefulness = self.models[context_key].predict(X_scaled)[0]
            
            # Convert predicted usefulness to optimal threshold
            # Higher predicted usefulness -> lower threshold needed
            optimal_threshold = max(0.1, min(0.95, 1.0 - predicted_usefulness))
            
            return optimal_threshold
            
        except Exception as e:
            logger.error("Failed to predict threshold", context=context.value, error=str(e))
            return 0.7


class AdaptiveConfidenceEngine:
    """
    Complete Adaptive Confidence Threshold System.
    
    Provides comprehensive confidence threshold adaptation with reinforcement learning,
    Bayesian optimization, contextual prediction, and performance monitoring.
    """
    
    def __init__(self, cache: Optional[RedisCache] = None, storage_path: str = "./adaptive_confidence"):
        self.cache = cache
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.rl_learner = ReinforcementLearner()
        self.bayesian_optimizer = BayesianOptimizer()
        self.contextual_predictor = ContextualPredictor()
        
        # Data storage
        self.threshold_configs: Dict[str, Dict[str, ThresholdConfig]] = {}  # user_id -> context -> config
        self.confidence_events: List[ConfidenceEvent] = []
        self.performance_history: Dict[str, deque] = {}  # user_id -> performance scores
        
        # Threading for async operations
        self.update_lock = threading.Lock()
        
        # Default configurations
        self.default_thresholds = {
            ConfidenceContext.WORK: 0.75,
            ConfidenceContext.PERSONAL: 0.65,
            ConfidenceContext.RESEARCH: 0.7,
            ConfidenceContext.CREATIVE: 0.6,
            ConfidenceContext.TECHNICAL: 0.8,
            ConfidenceContext.MEETING: 0.7,
            ConfidenceContext.PROJECT: 0.75,
            ConfidenceContext.GENERAL: 0.7
        }
        
        logger.info("Adaptive confidence engine initialized")
    
    async def get_confidence_threshold(self, user_id: str, query: str, 
                                     context: ConfidenceContext,
                                     predicted_confidence: float = None) -> float:
        """Get adaptive confidence threshold for a user query."""
        try:
            # Initialize user configs if needed
            if user_id not in self.threshold_configs:
                await self._initialize_user_configs(user_id)
            
            # Get user's config for this context
            config = self.threshold_configs[user_id].get(context)
            if not config:
                config = await self._create_context_config(user_id, context)
            
            # Determine adaptation strategy
            if config.strategy == AdaptationStrategy.CONSERVATIVE:
                return min(0.9, config.current_threshold + 0.1)
            
            elif config.strategy == AdaptationStrategy.AGGRESSIVE:
                return max(0.3, config.current_threshold - 0.1)
            
            elif config.strategy == AdaptationStrategy.CONTEXTUAL:
                # Use contextual predictor
                predicted_threshold = self.contextual_predictor.predict_optimal_threshold(
                    query, context, user_id
                )
                return predicted_threshold
            
            elif config.strategy == AdaptationStrategy.BAYESIAN:
                # Use Bayesian optimization
                suggested_threshold = self.bayesian_optimizer.suggest_threshold()
                return suggested_threshold
            
            elif config.strategy == AdaptationStrategy.REINFORCEMENT:
                # Use reinforcement learning
                state_key = self._get_rl_state(user_id, context, query)
                threshold_adjustment = self.rl_learner.choose_action(state_key)
                new_threshold = config.current_threshold + threshold_adjustment
                return max(config.min_threshold, min(config.max_threshold, new_threshold))
            
            else:  # BALANCED
                return config.current_threshold
            
        except Exception as e:
            logger.error("Failed to get confidence threshold", user_id=user_id, error=str(e))
            return self.default_thresholds.get(context, 0.7)
    
    async def record_confidence_feedback(self, user_id: str, query: str,
                                       context: ConfidenceContext,
                                       predicted_confidence: float,
                                       threshold_used: float,
                                       result_shown: bool,
                                       feedback_type: FeedbackType,
                                       actual_usefulness: float = None) -> bool:
        """Record user feedback for confidence threshold adaptation."""
        try:
            # Infer usefulness from feedback if not provided
            if actual_usefulness is None:
                actual_usefulness = self._infer_usefulness_from_feedback(feedback_type)
            
            # Create confidence event
            event = ConfidenceEvent(
                user_id=user_id,
                query=query,
                context=context,
                predicted_confidence=predicted_confidence,
                actual_usefulness=actual_usefulness,
                threshold_used=threshold_used,
                result_shown=result_shown,
                feedback_type=feedback_type,
                timestamp=datetime.utcnow()
            )
            
            self.confidence_events.append(event)
            
            # Update performance history
            if user_id not in self.performance_history:
                self.performance_history[user_id] = deque(maxlen=100)
            
            self.performance_history[user_id].append(actual_usefulness)
            
            # Cache event if available
            if self.cache:
                await self.cache.lpush(
                    f"confidence_events:{user_id}",
                    json.dumps(event.to_dict()),
                    max_length=1000
                )
            
            # Trigger adaptation
            await self._adapt_threshold(user_id, context, event)
            
            logger.info("Confidence feedback recorded", user_id=user_id, 
                       context=context.value, usefulness=actual_usefulness)
            return True
            
        except Exception as e:
            logger.error("Failed to record confidence feedback", user_id=user_id, error=str(e))
            return False
    
    def _infer_usefulness_from_feedback(self, feedback_type: FeedbackType) -> float:
        """Infer usefulness score from feedback type."""
        feedback_scores = {
            FeedbackType.EXPLICIT_ACCEPT: 0.9,
            FeedbackType.EXPLICIT_REJECT: 0.1,
            FeedbackType.IMPLICIT_ACCEPT: 0.7,
            FeedbackType.IMPLICIT_REJECT: 0.3,
            FeedbackType.CORRECTION: 0.4,
            FeedbackType.REFINEMENT: 0.5
        }
        
        return feedback_scores.get(feedback_type, 0.5)
    
    async def _adapt_threshold(self, user_id: str, context: ConfidenceContext,
                             event: ConfidenceEvent):
        """Adapt confidence threshold based on feedback."""
        with self.update_lock:
            try:
                config = self.threshold_configs[user_id][context]
                
                # Update history
                config.confidence_history.append(event.predicted_confidence)
                config.performance_history.append(event.actual_usefulness)
                
                # Keep recent history only
                if len(config.confidence_history) > 100:
                    config.confidence_history = config.confidence_history[-100:]
                    config.performance_history = config.performance_history[-100:]
                
                # Adapt based on strategy
                if config.strategy == AdaptationStrategy.REINFORCEMENT:
                    await self._adapt_with_reinforcement_learning(user_id, context, event)
                
                elif config.strategy == AdaptationStrategy.BAYESIAN:
                    await self._adapt_with_bayesian_optimization(user_id, context, event)
                
                else:
                    await self._adapt_with_simple_rule(user_id, context, event)
                
                config.last_updated = datetime.utcnow()
                
                # Save updated config
                await self._save_threshold_config(config)
                
            except Exception as e:
                logger.error("Failed to adapt threshold", user_id=user_id, error=str(e))
    
    async def _adapt_with_reinforcement_learning(self, user_id: str, 
                                               context: ConfidenceContext,
                                               event: ConfidenceEvent):
        """Adapt threshold using reinforcement learning."""
        # Get current state
        state_key = self._get_rl_state(user_id, context, event.query)
        
        # Calculate reward
        reward = self.rl_learner.calculate_reward(
            event.predicted_confidence,
            event.actual_usefulness,
            event.threshold_used,
            event.result_shown
        )
        
        # Get next state (after adaptation)
        next_state_key = self._get_rl_state(user_id, context, event.query, is_next=True)
        
        # Calculate threshold adjustment action
        config = self.threshold_configs[user_id][context]
        threshold_diff = event.threshold_used - config.current_threshold
        action = min(self.rl_learner.actions, key=lambda x: abs(x - threshold_diff))
        
        # Update Q-value
        self.rl_learner.update_q_value(state_key, action, reward, next_state_key)
        
        # Apply best action to current threshold
        best_action = self.rl_learner.choose_action(next_state_key)
        new_threshold = config.current_threshold + best_action
        config.current_threshold = max(config.min_threshold, 
                                     min(config.max_threshold, new_threshold))
    
    async def _adapt_with_bayesian_optimization(self, user_id: str,
                                              context: ConfidenceContext,
                                              event: ConfidenceEvent):
        """Adapt threshold using Bayesian optimization."""
        # Add observation to Bayesian optimizer
        self.bayesian_optimizer.add_observation(
            event.threshold_used,
            event.actual_usefulness
        )
        
        # Get new suggested threshold
        suggested_threshold = self.bayesian_optimizer.suggest_threshold()
        
        config = self.threshold_configs[user_id][context]
        config.current_threshold = suggested_threshold
    
    async def _adapt_with_simple_rule(self, user_id: str, context: ConfidenceContext,
                                    event: ConfidenceEvent):
        """Adapt threshold using simple rules."""
        config = self.threshold_configs[user_id][context]
        
        # Simple adaptation based on feedback
        if event.actual_usefulness > 0.8 and not event.result_shown:
            # Good result was filtered out - lower threshold
            adjustment = -config.adaptation_rate
        elif event.actual_usefulness < 0.3 and event.result_shown:
            # Poor result was shown - raise threshold
            adjustment = config.adaptation_rate
        elif event.actual_usefulness > 0.7 and event.result_shown:
            # Good result was shown - small reward (lower threshold slightly)
            adjustment = -config.adaptation_rate * 0.3
        else:
            # No significant adjustment needed
            adjustment = 0
        
        new_threshold = config.current_threshold + adjustment
        config.current_threshold = max(config.min_threshold,
                                     min(config.max_threshold, new_threshold))
    
    def _get_rl_state(self, user_id: str, context: ConfidenceContext, 
                     query: str, is_next: bool = False) -> str:
        """Get reinforcement learning state representation."""
        # Recent performance
        recent_performance = 0.5
        if user_id in self.performance_history and self.performance_history[user_id]:
            recent_performance = np.mean(list(self.performance_history[user_id])[-10:])
        
        # Query complexity
        query_complexity = self.contextual_predictor._estimate_query_complexity(query)
        
        # User experience (number of interactions)
        user_experience = min(1.0, len(self.performance_history.get(user_id, [])) / 100.0)
        
        return self.rl_learner.get_state_key(context, recent_performance, 
                                           query_complexity, user_experience)
    
    async def _initialize_user_configs(self, user_id: str):
        """Initialize threshold configurations for a new user."""
        self.threshold_configs[user_id] = {}
        
        for context in ConfidenceContext:
            config = await self._create_context_config(user_id, context)
            self.threshold_configs[user_id][context] = config
    
    async def _create_context_config(self, user_id: str, 
                                   context: ConfidenceContext) -> ThresholdConfig:
        """Create threshold configuration for a user-context pair."""
        config = ThresholdConfig(
            user_id=user_id,
            context=context,
            current_threshold=self.default_thresholds[context],
            strategy=AdaptationStrategy.BALANCED
        )
        
        if user_id not in self.threshold_configs:
            self.threshold_configs[user_id] = {}
        
        self.threshold_configs[user_id][context] = config
        await self._save_threshold_config(config)
        
        return config
    
    async def set_user_strategy(self, user_id: str, context: ConfidenceContext,
                              strategy: AdaptationStrategy) -> bool:
        """Set adaptation strategy for a user-context pair."""
        try:
            if user_id not in self.threshold_configs:
                await self._initialize_user_configs(user_id)
            
            config = self.threshold_configs[user_id][context]
            config.strategy = strategy
            config.last_updated = datetime.utcnow()
            
            await self._save_threshold_config(config)
            
            logger.info("User strategy updated", user_id=user_id, 
                       context=context.value, strategy=strategy.value)
            return True
            
        except Exception as e:
            logger.error("Failed to set user strategy", user_id=user_id, error=str(e))
            return False
    
    async def retrain_models(self) -> bool:
        """Retrain contextual prediction models with latest data."""
        try:
            # Group events by context
            context_events = defaultdict(list)
            for event in self.confidence_events:
                context_events[event.context].append(event)
            
            success_count = 0
            
            # Train model for each context
            for context, events in context_events.items():
                if self.contextual_predictor.train_model(events, context):
                    success_count += 1
            
            logger.info("Models retrained", contexts_trained=success_count,
                       total_events=len(self.confidence_events))
            return success_count > 0
            
        except Exception as e:
            logger.error("Failed to retrain models", error=str(e))
            return False
    
    async def _save_threshold_config(self, config: ThresholdConfig) -> bool:
        """Save threshold configuration to disk."""
        try:
            config_file = self.storage_path / f"config_{config.user_id}_{config.context.value}.json"
            
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error("Failed to save threshold config", 
                        user_id=config.user_id, context=config.context.value, error=str(e))
            return False
    
    async def load_threshold_config(self, user_id: str, 
                                  context: ConfidenceContext) -> Optional[ThresholdConfig]:
        """Load threshold configuration from disk."""
        try:
            config_file = self.storage_path / f"config_{user_id}_{context.value}.json"
            
            if not config_file.exists():
                return None
            
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            config = ThresholdConfig(
                user_id=data['user_id'],
                context=ConfidenceContext(data['context']),
                current_threshold=data['current_threshold'],
                min_threshold=data.get('min_threshold', 0.1),
                max_threshold=data.get('max_threshold', 0.95),
                adaptation_rate=data.get('adaptation_rate', 0.1),
                confidence_history=data.get('confidence_history', []),
                performance_history=data.get('performance_history', []),
                last_updated=datetime.fromisoformat(data['last_updated']),
                strategy=AdaptationStrategy(data.get('strategy', 'balanced'))
            )
            
            return config
            
        except Exception as e:
            logger.error("Failed to load threshold config", 
                        user_id=user_id, context=context.value, error=str(e))
            return None
    
    def get_user_threshold_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's threshold configurations."""
        if user_id not in self.threshold_configs:
            return {}
        
        summary = {}
        
        for context, config in self.threshold_configs[user_id].items():
            recent_performance = 0.0
            if config.performance_history:
                recent_performance = np.mean(config.performance_history[-10:])
            
            summary[context.value] = {
                'current_threshold': config.current_threshold,
                'strategy': config.strategy.value,
                'recent_performance': recent_performance,
                'adaptation_count': len(config.performance_history),
                'last_updated': config.last_updated.isoformat()
            }
        
        return summary
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide confidence adaptation metrics."""
        total_users = len(self.threshold_configs)
        total_events = len(self.confidence_events)
        
        # Calculate average performance by context
        context_performance = defaultdict(list)
        for event in self.confidence_events:
            context_performance[event.context.value].append(event.actual_usefulness)
        
        avg_performance_by_context = {}
        for context, performances in context_performance.items():
            avg_performance_by_context[context] = np.mean(performances) if performances else 0.0
        
        # Strategy distribution
        strategy_counts = defaultdict(int)
        for user_configs in self.threshold_configs.values():
            for config in user_configs.values():
                strategy_counts[config.strategy.value] += 1
        
        return {
            "total_users": total_users,
            "total_events": total_events,
            "avg_performance_by_context": avg_performance_by_context,
            "strategy_distribution": dict(strategy_counts),
            "rl_epsilon": self.rl_learner.epsilon,
            "bayesian_observations": len(self.bayesian_optimizer.observations_x)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform confidence engine health check."""
        try:
            # Test threshold adaptation with dummy data
            test_user = "test_user_confidence"
            test_context = ConfidenceContext.GENERAL
            
            # Get initial threshold
            initial_threshold = await self.get_confidence_threshold(
                test_user, "test query", test_context
            )
            
            # Record feedback
            feedback_success = await self.record_confidence_feedback(
                test_user, "test query", test_context,
                predicted_confidence=0.8,
                threshold_used=initial_threshold,
                result_shown=True,
                feedback_type=FeedbackType.IMPLICIT_ACCEPT,
                actual_usefulness=0.9
            )
            
            # Get adapted threshold
            adapted_threshold = await self.get_confidence_threshold(
                test_user, "test query", test_context
            )
            
            return {
                "status": "healthy" if feedback_success else "unhealthy",
                "test_threshold_obtained": initial_threshold is not None,
                "test_feedback_recorded": feedback_success,
                "threshold_adaptation_working": abs(adapted_threshold - initial_threshold) >= 0,
                "system_metrics": self.get_system_metrics()
            }
            
        except Exception as e:
            logger.error("Confidence engine health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "system_metrics": self.get_system_metrics()
            }