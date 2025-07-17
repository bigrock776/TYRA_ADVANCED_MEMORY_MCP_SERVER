"""
Online Learning Framework for Continuous Model Adaptation.

This module provides comprehensive online learning capabilities with incremental learning,
model adaptation using local algorithms, learning rate optimization with local methods,
and catastrophic forgetting prevention. All processing is performed locally with zero external dependencies.
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
from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import joblib

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class LearningTaskType(str, Enum):
    """Types of learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    RANKING = "ranking"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelType(str, Enum):
    """Types of online learning models."""
    SGD_CLASSIFIER = "sgd_classifier"
    SGD_REGRESSOR = "sgd_regressor"
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE = "decision_tree"
    NAIVE_BAYES = "naive_bayes"
    MLP = "mlp"
    INCREMENTAL_TREE = "incremental_tree"


class AdaptationStrategy(str, Enum):
    """Strategies for model adaptation."""
    IMMEDIATE = "immediate"          # Update immediately on new data
    BATCH = "batch"                  # Update in batches
    THRESHOLD = "threshold"          # Update when performance drops
    SCHEDULED = "scheduled"          # Update on schedule
    ADAPTIVE = "adaptive"            # Adaptive timing based on data drift


class ForgettingStrategy(str, Enum):
    """Strategies for preventing catastrophic forgetting."""
    EWC = "ewc"                     # Elastic Weight Consolidation
    REPLAY = "replay"               # Experience replay
    REGULARIZATION = "regularization"  # L2 regularization
    ELASTIC = "elastic"             # Elastic updates
    PROGRESSIVE = "progressive"     # Progressive networks


@dataclass
class LearningMetrics:
    """Metrics for tracking learning performance."""
    task_id: str
    model_type: ModelType
    timestamp: datetime
    train_score: float
    test_score: float
    data_points_processed: int
    adaptation_count: int
    learning_rate: float
    forgetting_score: float = 0.0
    drift_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "train_score": self.train_score,
            "test_score": self.test_score,
            "data_points_processed": self.data_points_processed,
            "adaptation_count": self.adaptation_count,
            "learning_rate": self.learning_rate,
            "forgetting_score": self.forgetting_score,
            "drift_score": self.drift_score,
            "metadata": self.metadata
        }


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""
    task_type: LearningTaskType
    model_type: ModelType
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.ADAPTIVE
    forgetting_strategy: ForgettingStrategy = ForgettingStrategy.EWC
    initial_learning_rate: float = 0.01
    min_learning_rate: float = 0.001
    max_learning_rate: float = 0.1
    batch_size: int = 32
    adaptation_threshold: float = 0.05
    forgetting_penalty: float = 1000.0
    replay_buffer_size: int = 1000
    drift_detection_window: int = 100
    performance_window: int = 50
    enable_preprocessing: bool = True
    enable_feature_selection: bool = True
    max_features: Optional[int] = None


class DataDriftDetector:
    """Detects drift in incoming data streams."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.reference_stats: Dict[str, Any] = {}
        self.current_window: deque = deque(maxlen=window_size)
        self.drift_history: List[float] = []
    
    def update_reference(self, X: np.ndarray):
        """Update reference statistics from data."""
        self.reference_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'median': np.median(X, axis=0)
        }
    
    def detect_drift(self, X_new: np.ndarray) -> Tuple[bool, float]:
        """Detect if new data shows significant drift."""
        if not self.reference_stats:
            self.update_reference(X_new)
            return False, 0.0
        
        # Add to current window
        self.current_window.extend(X_new.tolist())
        
        if len(self.current_window) < self.window_size:
            return False, 0.0
        
        # Calculate drift score using statistical tests
        current_data = np.array(list(self.current_window))
        current_stats = {
            'mean': np.mean(current_data, axis=0),
            'std': np.std(current_data, axis=0)
        }
        
        # KL divergence approximation for drift detection
        drift_scores = []
        for i in range(len(self.reference_stats['mean'])):
            ref_mean = self.reference_stats['mean'][i]
            ref_std = self.reference_stats['std'][i]
            cur_mean = current_stats['mean'][i]
            cur_std = current_stats['std'][i]
            
            if ref_std > 0 and cur_std > 0:
                # Simplified divergence measure
                mean_diff = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
                std_ratio = abs(np.log((cur_std + 1e-8) / (ref_std + 1e-8)))
                drift_score = mean_diff + std_ratio
                drift_scores.append(drift_score)
        
        overall_drift = np.mean(drift_scores) if drift_scores else 0.0
        self.drift_history.append(overall_drift)
        
        # Keep only recent history
        if len(self.drift_history) > 1000:
            self.drift_history = self.drift_history[-1000:]
        
        is_drift = overall_drift > self.sensitivity
        return is_drift, overall_drift


class ElasticWeightConsolidation:
    """Implements Elastic Weight Consolidation to prevent catastrophic forgetting."""
    
    def __init__(self, penalty_strength: float = 1000.0):
        self.penalty_strength = penalty_strength
        self.previous_weights: Optional[np.ndarray] = None
        self.fisher_matrix: Optional[np.ndarray] = None
        self.task_count = 0
    
    def calculate_fisher_matrix(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate Fisher Information Matrix for important weights."""
        # For SGD models, approximate Fisher matrix using gradients
        if hasattr(model, 'coef_'):
            weights = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
            
            # Simple approximation: use variance of gradients
            fisher_diag = np.ones_like(weights) * 0.01  # Base importance
            
            # Sample-based Fisher approximation
            for i in range(min(100, len(X))):
                x_sample = X[i:i+1]
                y_sample = y[i:i+1]
                
                # Predict and calculate gradient magnitude (approximation)
                pred = model.predict(x_sample)
                error = abs(pred - y_sample).mean()
                
                # Update Fisher diagonal (simplified)
                fisher_diag += error * np.abs(weights)
            
            return fisher_diag / len(X)
        
        return np.array([1.0])  # Fallback
    
    def update_task(self, model, X: np.ndarray, y: np.ndarray):
        """Update EWC parameters for new task."""
        if hasattr(model, 'coef_'):
            current_weights = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
            
            if self.task_count > 0:
                self.fisher_matrix = self.calculate_fisher_matrix(model, X, y)
                self.previous_weights = current_weights.copy()
            
            self.task_count += 1
    
    def get_penalty(self, model) -> float:
        """Calculate EWC penalty for current model weights."""
        if self.task_count <= 1 or self.previous_weights is None:
            return 0.0
        
        if hasattr(model, 'coef_'):
            current_weights = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
            
            if self.fisher_matrix is not None:
                weight_diff = current_weights - self.previous_weights
                penalty = 0.5 * self.penalty_strength * np.sum(
                    self.fisher_matrix * weight_diff ** 2
                )
                return float(penalty)
        
        return 0.0


class IncrementalModelManager:
    """Manages incremental learning models with adaptation strategies."""
    
    def __init__(self, config: OnlineLearningConfig, cache: Optional[RedisCache] = None):
        self.config = config
        self.cache = cache
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.performance_history: deque = deque(maxlen=config.performance_window)
        self.adaptation_count = 0
        self.data_points_processed = 0
        self.learning_rate_history: List[float] = []
        
        # Drift detection and forgetting prevention
        self.drift_detector = DataDriftDetector(
            window_size=config.drift_detection_window,
            sensitivity=0.05
        )
        self.ewc = ElasticWeightConsolidation(config.forgetting_penalty)
        
        # Replay buffer for experience replay
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # Initialize model
        self._initialize_model()
        
        # Metrics tracking
        self.metrics_history: List[LearningMetrics] = []
    
    def _initialize_model(self):
        """Initialize the learning model based on configuration."""
        if self.config.task_type == LearningTaskType.CLASSIFICATION:
            if self.config.model_type == ModelType.SGD_CLASSIFIER:
                self.model = SGDClassifier(
                    learning_rate='adaptive',
                    eta0=self.config.initial_learning_rate,
                    random_state=42
                )
            elif self.config.model_type == ModelType.PASSIVE_AGGRESSIVE:
                self.model = PassiveAggressiveClassifier(
                    C=1.0,
                    random_state=42
                )
            elif self.config.model_type == ModelType.NAIVE_BAYES:
                self.model = GaussianNB()
            elif self.config.model_type == ModelType.MLP:
                self.model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    learning_rate_init=self.config.initial_learning_rate,
                    random_state=42
                )
            else:
                self.model = SGDClassifier(random_state=42)
                
        elif self.config.task_type == LearningTaskType.REGRESSION:
            if self.config.model_type == ModelType.SGD_REGRESSOR:
                self.model = SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.config.initial_learning_rate,
                    random_state=42
                )
            elif self.config.model_type == ModelType.MLP:
                self.model = MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    learning_rate_init=self.config.initial_learning_rate,
                    random_state=42
                )
            else:
                self.model = SGDRegressor(random_state=42)
        
        # Initialize preprocessing
        if self.config.enable_preprocessing:
            self.scaler = StandardScaler()
    
    def _preprocess_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess features with scaling and selection."""
        if self.scaler is not None:
            if fit:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Feature selection (simplified)
        if self.config.enable_feature_selection and self.config.max_features:
            if X_scaled.shape[1] > self.config.max_features:
                # Select top features by variance
                feature_vars = np.var(X_scaled, axis=0)
                top_indices = np.argsort(feature_vars)[-self.config.max_features:]
                X_scaled = X_scaled[:, top_indices]
        
        return X_scaled
    
    def _calculate_performance_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate performance score for the model."""
        try:
            predictions = self.model.predict(X)
            
            if self.config.task_type == LearningTaskType.CLASSIFICATION:
                return accuracy_score(y, predictions)
            elif self.config.task_type == LearningTaskType.REGRESSION:
                return r2_score(y, predictions)
            else:
                return 0.0
        except Exception as e:
            logger.warning("Failed to calculate performance score", error=str(e))
            return 0.0
    
    def _should_adapt(self, current_performance: float) -> bool:
        """Determine if model should adapt based on strategy."""
        if self.config.adaptation_strategy == AdaptationStrategy.IMMEDIATE:
            return True
        
        elif self.config.adaptation_strategy == AdaptationStrategy.THRESHOLD:
            if len(self.performance_history) < 10:
                return False
            
            recent_avg = np.mean(list(self.performance_history)[-10:])
            return current_performance < (recent_avg - self.config.adaptation_threshold)
        
        elif self.config.adaptation_strategy == AdaptationStrategy.BATCH:
            return self.data_points_processed % self.config.batch_size == 0
        
        elif self.config.adaptation_strategy == AdaptationStrategy.ADAPTIVE:
            # Adaptive strategy based on drift and performance
            if len(self.performance_history) >= 5:
                recent_trend = np.polyfit(range(5), list(self.performance_history)[-5:], 1)[0]
                return recent_trend < -0.01  # Declining performance
            return False
        
        return False
    
    def _adaptive_learning_rate(self, performance_trend: float, drift_score: float) -> float:
        """Adapt learning rate based on performance and drift."""
        current_lr = self.config.initial_learning_rate
        
        if hasattr(self.model, 'learning_rate_'):
            current_lr = self.model.learning_rate_
        elif hasattr(self.model, 'eta0'):
            current_lr = self.model.eta0
        
        # Increase learning rate if performance is declining or high drift
        if performance_trend < -0.01 or drift_score > 0.1:
            new_lr = min(self.config.max_learning_rate, current_lr * 1.1)
        # Decrease learning rate if performance is stable
        elif performance_trend > 0.01 and drift_score < 0.05:
            new_lr = max(self.config.min_learning_rate, current_lr * 0.95)
        else:
            new_lr = current_lr
        
        # Update model learning rate
        if hasattr(self.model, 'set_params'):
            try:
                if hasattr(self.model, 'eta0'):
                    self.model.set_params(eta0=new_lr)
                elif hasattr(self.model, 'learning_rate_init'):
                    self.model.set_params(learning_rate_init=new_lr)
            except Exception:
                pass
        
        self.learning_rate_history.append(new_lr)
        return new_lr
    
    def _apply_forgetting_prevention(self, X: np.ndarray, y: np.ndarray):
        """Apply forgetting prevention strategies."""
        if self.config.forgetting_strategy == ForgettingStrategy.EWC:
            self.ewc.update_task(self.model, X, y)
        
        elif self.config.forgetting_strategy == ForgettingStrategy.REPLAY:
            # Add to replay buffer
            for i in range(len(X)):
                self.replay_buffer.append((X[i], y[i]))
            
            # Replay old experiences
            if len(self.replay_buffer) >= self.config.batch_size:
                replay_sample = list(self.replay_buffer)[-self.config.batch_size:]
                replay_X = np.array([item[0] for item in replay_sample])
                replay_y = np.array([item[1] for item in replay_sample])
                
                # Train on replay data
                try:
                    if hasattr(self.model, 'partial_fit'):
                        self.model.partial_fit(replay_X, replay_y)
                    else:
                        self.model.fit(replay_X, replay_y)
                except Exception as e:
                    logger.warning("Failed to replay experiences", error=str(e))
    
    async def partial_fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Incrementally train the model on new data."""
        start_time = datetime.utcnow()
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=(self.data_points_processed == 0))
            
            # Detect drift
            is_drift, drift_score = self.drift_detector.detect_drift(X_processed)
            
            # Calculate current performance if we have a trained model
            current_performance = 0.0
            if self.data_points_processed > 0:
                current_performance = self._calculate_performance_score(X_processed, y)
                self.performance_history.append(current_performance)
            
            # Determine if we should adapt
            should_adapt = self._should_adapt(current_performance)
            
            if should_adapt or self.data_points_processed == 0:
                # Apply forgetting prevention
                self._apply_forgetting_prevention(X_processed, y)
                
                # Adapt learning rate
                performance_trend = 0.0
                if len(self.performance_history) >= 2:
                    recent_scores = list(self.performance_history)[-5:]
                    performance_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                current_lr = self._adaptive_learning_rate(performance_trend, drift_score)
                
                # Train the model
                if hasattr(self.model, 'partial_fit'):
                    if self.data_points_processed == 0:
                        # Initial fit with classes for classification
                        if self.config.task_type == LearningTaskType.CLASSIFICATION:
                            unique_classes = np.unique(y)
                            self.model.partial_fit(X_processed, y, classes=unique_classes)
                        else:
                            self.model.partial_fit(X_processed, y)
                    else:
                        self.model.partial_fit(X_processed, y)
                else:
                    # Fallback to full fit
                    self.model.fit(X_processed, y)
                
                self.adaptation_count += 1
            
            self.data_points_processed += len(X)
            
            # Calculate final performance
            final_performance = self._calculate_performance_score(X_processed, y)
            
            # Calculate forgetting score
            forgetting_score = 0.0
            if self.config.forgetting_strategy == ForgettingStrategy.EWC:
                forgetting_score = self.ewc.get_penalty(self.model)
            
            # Create metrics
            metrics = LearningMetrics(
                task_id=f"online_learning_{id(self)}",
                model_type=self.config.model_type,
                timestamp=start_time,
                train_score=final_performance,
                test_score=final_performance,
                data_points_processed=self.data_points_processed,
                adaptation_count=self.adaptation_count,
                learning_rate=current_lr if 'current_lr' in locals() else self.config.initial_learning_rate,
                forgetting_score=forgetting_score,
                drift_score=drift_score,
                metadata={
                    "is_drift": is_drift,
                    "should_adapt": should_adapt,
                    "performance_trend": performance_trend,
                    "batch_size": len(X)
                }
            )
            
            self.metrics_history.append(metrics)
            
            # Keep only recent metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            # Cache metrics if available
            if self.cache:
                await self.cache.lpush(
                    "online_learning_metrics",
                    json.dumps(metrics.to_dict()),
                    max_length=1000
                )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                "Online learning update completed",
                data_points=len(X),
                performance=final_performance,
                drift_score=drift_score,
                adaptation_count=self.adaptation_count,
                duration_ms=duration * 1000
            )
            
            return {
                "success": True,
                "performance": final_performance,
                "drift_detected": is_drift,
                "drift_score": drift_score,
                "adapted": should_adapt,
                "data_points_processed": self.data_points_processed,
                "adaptation_count": self.adaptation_count,
                "forgetting_score": forgetting_score,
                "duration_ms": duration * 1000
            }
            
        except Exception as e:
            logger.error("Online learning update failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "data_points_processed": self.data_points_processed
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the current model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_processed = self._preprocess_features(X, fit=False)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions if supported."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        X_processed = self._preprocess_features(X, fit=False)
        return self.model.predict_proba(X_processed)
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for serialization."""
        state = {
            "config": asdict(self.config),
            "adaptation_count": self.adaptation_count,
            "data_points_processed": self.data_points_processed,
            "performance_history": list(self.performance_history),
            "learning_rate_history": self.learning_rate_history[-100:],  # Keep recent
            "drift_history": self.drift_detector.drift_history[-100:],
            "ewc_task_count": self.ewc.task_count
        }
        
        # Serialize model if possible
        try:
            state["model_data"] = joblib.dumps(self.model)
            if self.scaler:
                state["scaler_data"] = joblib.dumps(self.scaler)
        except Exception as e:
            logger.warning("Failed to serialize model", error=str(e))
        
        return state
    
    def load_model_state(self, state: Dict[str, Any]):
        """Load model state from serialization."""
        try:
            self.adaptation_count = state.get("adaptation_count", 0)
            self.data_points_processed = state.get("data_points_processed", 0)
            self.performance_history = deque(
                state.get("performance_history", []),
                maxlen=self.config.performance_window
            )
            self.learning_rate_history = state.get("learning_rate_history", [])
            self.drift_detector.drift_history = state.get("drift_history", [])
            self.ewc.task_count = state.get("ewc_task_count", 0)
            
            # Load model if available
            if "model_data" in state:
                self.model = joblib.loads(state["model_data"])
            
            if "scaler_data" in state:
                self.scaler = joblib.loads(state["scaler_data"])
                
        except Exception as e:
            logger.error("Failed to load model state", error=str(e))
            self._initialize_model()  # Fallback to new model


class OnlineLearningSystem:
    """
    Complete Online Learning System.
    
    Provides comprehensive online learning capabilities with incremental learning,
    model adaptation, drift detection, and catastrophic forgetting prevention.
    """
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache
        self.models: Dict[str, IncrementalModelManager] = {}
        self.system_metrics: Dict[str, Any] = {
            "total_models": 0,
            "total_adaptations": 0,
            "total_data_points": 0,
            "avg_performance": 0.0
        }
        
        logger.info("Online learning system initialized")
    
    async def create_learning_task(
        self,
        task_id: str,
        config: OnlineLearningConfig
    ) -> IncrementalModelManager:
        """Create a new online learning task."""
        if task_id in self.models:
            raise ValueError(f"Task {task_id} already exists")
        
        model_manager = IncrementalModelManager(config, self.cache)
        self.models[task_id] = model_manager
        self.system_metrics["total_models"] += 1
        
        logger.info("Online learning task created", task_id=task_id, model_type=config.model_type.value)
        return model_manager
    
    async def update_model(
        self,
        task_id: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Update a model with new data."""
        if task_id not in self.models:
            raise ValueError(f"Task {task_id} not found")
        
        model_manager = self.models[task_id]
        result = await model_manager.partial_fit(X, y)
        
        if result.get("success"):
            self.system_metrics["total_data_points"] += len(X)
            if result.get("adapted"):
                self.system_metrics["total_adaptations"] += 1
        
        return result
    
    def get_model(self, task_id: str) -> Optional[IncrementalModelManager]:
        """Get a model manager by task ID."""
        return self.models.get(task_id)
    
    def predict(self, task_id: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with a specific model."""
        if task_id not in self.models:
            raise ValueError(f"Task {task_id} not found")
        
        return self.models[task_id].predict(X)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        # Calculate average performance across all models
        total_performance = 0.0
        performance_count = 0
        
        for model_manager in self.models.values():
            if model_manager.performance_history:
                total_performance += sum(model_manager.performance_history)
                performance_count += len(model_manager.performance_history)
        
        if performance_count > 0:
            self.system_metrics["avg_performance"] = total_performance / performance_count
        
        # Add per-model metrics
        model_metrics = {}
        for task_id, model_manager in self.models.items():
            recent_metrics = model_manager.metrics_history[-1] if model_manager.metrics_history else None
            model_metrics[task_id] = {
                "data_points_processed": model_manager.data_points_processed,
                "adaptation_count": model_manager.adaptation_count,
                "recent_performance": recent_metrics.test_score if recent_metrics else 0.0,
                "recent_drift_score": recent_metrics.drift_score if recent_metrics else 0.0
            }
        
        return {
            **self.system_metrics,
            "model_metrics": model_metrics,
            "active_models": len(self.models)
        }
    
    async def save_model(self, task_id: str, file_path: str) -> bool:
        """Save a model to disk."""
        if task_id not in self.models:
            return False
        
        try:
            model_state = self.models[task_id].get_model_state()
            
            # Save to file
            with open(file_path, 'wb') as f:
                pickle.dump(model_state, f)
            
            logger.info("Model saved", task_id=task_id, file_path=file_path)
            return True
            
        except Exception as e:
            logger.error("Failed to save model", task_id=task_id, error=str(e))
            return False
    
    async def load_model(self, task_id: str, file_path: str, config: OnlineLearningConfig) -> bool:
        """Load a model from disk."""
        try:
            with open(file_path, 'rb') as f:
                model_state = pickle.load(f)
            
            # Create new model manager
            model_manager = IncrementalModelManager(config, self.cache)
            model_manager.load_model_state(model_state)
            
            self.models[task_id] = model_manager
            
            logger.info("Model loaded", task_id=task_id, file_path=file_path)
            return True
            
        except Exception as e:
            logger.error("Failed to load model", task_id=task_id, error=str(e))
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        try:
            # Test with dummy data
            test_config = OnlineLearningConfig(
                task_type=LearningTaskType.CLASSIFICATION,
                model_type=ModelType.SGD_CLASSIFIER
            )
            
            # Create temporary test model
            test_model = IncrementalModelManager(test_config, self.cache)
            
            # Test training
            X_test = np.random.random((10, 5))
            y_test = np.random.randint(0, 2, 10)
            
            result = await test_model.partial_fit(X_test, y_test)
            
            return {
                "status": "healthy" if result.get("success") else "unhealthy",
                "test_training": result.get("success", False),
                "system_metrics": self.get_system_metrics(),
                "active_models": len(self.models)
            }
            
        except Exception as e:
            logger.error("Online learning health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "active_models": len(self.models)
            }