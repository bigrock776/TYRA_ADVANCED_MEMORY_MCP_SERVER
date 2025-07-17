"""
Continuous Model Improvement System for Automated Performance Enhancement.

This module provides comprehensive continuous improvement capabilities with performance monitoring,
automated retraining triggers using local algorithms, model versioning with local storage,
and improvement assessment with local validation. All processing is performed locally with zero external dependencies.
"""

import asyncio
import os
import shutil
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import hashlib
import threading
from pathlib import Path

# ML and evaluation imports - all local
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.inspection import permutation_importance
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from .online_learning import OnlineLearningSystem, OnlineLearningConfig, LearningTaskType, ModelType
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ImprovementTrigger(str, Enum):
    """Types of improvement triggers."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    THRESHOLD_BASED = "threshold_based"
    TIME_BASED = "time_based"
    VOLUME_BASED = "volume_based"


class ModelStatus(str, Enum):
    """Status of models in the system."""
    ACTIVE = "active"
    TRAINING = "training"
    EVALUATING = "evaluating"
    STAGED = "staged"
    RETIRED = "retired"
    FAILED = "failed"


class ImprovementStrategy(str, Enum):
    """Strategies for model improvement."""
    RETRAIN_FROM_SCRATCH = "retrain_from_scratch"
    INCREMENTAL_UPDATE = "incremental_update"
    ENSEMBLE_UPGRADE = "ensemble_upgrade"
    ARCHITECTURE_CHANGE = "architecture_change"
    HYPERPARAMETER_TUNE = "hyperparameter_tune"
    FEATURE_ENGINEERING = "feature_engineering"


@dataclass
class ModelVersion:
    """Represents a version of a model."""
    version_id: str
    model_id: str
    version_number: int
    created_at: datetime
    status: ModelStatus
    performance_metrics: Dict[str, float]
    model_config: Dict[str, Any]
    file_path: Optional[str] = None
    parent_version: Optional[str] = None
    improvement_notes: str = ""
    deployment_date: Optional[datetime] = None
    retirement_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "version_number": self.version_number,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "performance_metrics": self.performance_metrics,
            "model_config": self.model_config,
            "file_path": self.file_path,
            "parent_version": self.parent_version,
            "improvement_notes": self.improvement_notes,
            "deployment_date": self.deployment_date.isoformat() if self.deployment_date else None,
            "retirement_date": self.retirement_date.isoformat() if self.retirement_date else None
        }


@dataclass
class ImprovementPlan:
    """Plan for improving a model."""
    plan_id: str
    model_id: str
    trigger: ImprovementTrigger
    strategy: ImprovementStrategy
    expected_improvement: float
    confidence: float
    estimated_duration: timedelta
    resource_requirements: Dict[str, Any]
    success_criteria: Dict[str, float]
    rollback_criteria: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "model_id": self.model_id,
            "trigger": self.trigger.value,
            "strategy": self.strategy.value,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "estimated_duration": self.estimated_duration.total_seconds(),
            "resource_requirements": self.resource_requirements,
            "success_criteria": self.success_criteria,
            "rollback_criteria": self.rollback_criteria,
            "created_at": self.created_at.isoformat()
        }


class PerformanceMonitor:
    """Monitors model performance over time."""
    
    def __init__(self, window_size: int = 100, alert_threshold: float = 0.05):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_performance: Dict[str, float] = {}
        self.alerts: List[Dict[str, Any]] = []
    
    def update_performance(self, model_id: str, metric_name: str, value: float):
        """Update performance metrics for a model."""
        key = f"{model_id}_{metric_name}"
        self.performance_history[key].append({
            "value": value,
            "timestamp": datetime.utcnow()
        })
        
        # Set baseline if not exists
        if key not in self.baseline_performance:
            self.baseline_performance[key] = value
        
        # Check for performance degradation
        self._check_degradation(model_id, metric_name, value)
    
    def _check_degradation(self, model_id: str, metric_name: str, current_value: float):
        """Check if performance has degraded significantly."""
        key = f"{model_id}_{metric_name}"
        
        if key not in self.baseline_performance:
            return
        
        baseline = self.baseline_performance[key]
        
        # Calculate relative change
        if baseline != 0:
            relative_change = (current_value - baseline) / abs(baseline)
        else:
            relative_change = current_value - baseline
        
        # Check if degradation exceeds threshold
        if abs(relative_change) > self.alert_threshold:
            alert = {
                "model_id": model_id,
                "metric_name": metric_name,
                "current_value": current_value,
                "baseline_value": baseline,
                "relative_change": relative_change,
                "timestamp": datetime.utcnow(),
                "severity": "high" if abs(relative_change) > self.alert_threshold * 2 else "medium"
            }
            
            self.alerts.append(alert)
            
            # Keep only recent alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
    
    def get_performance_trend(self, model_id: str, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """Get performance trend analysis."""
        key = f"{model_id}_{metric_name}"
        
        if key not in self.performance_history:
            return {"trend": "no_data", "slope": 0, "confidence": 0}
        
        # Filter recent data
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_data = [
            point for point in self.performance_history[key]
            if point["timestamp"] > cutoff_time
        ]
        
        if len(recent_data) < 5:
            return {"trend": "insufficient_data", "slope": 0, "confidence": 0}
        
        # Calculate trend
        values = [point["value"] for point in recent_data]
        x = range(len(values))
        
        # Linear regression to find trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "improving"
        else:
            trend = "declining"
        
        return {
            "trend": trend,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "confidence": 1 - p_value if p_value < 1 else 0,
            "recent_values": values[-10:],  # Last 10 values
            "baseline": self.baseline_performance.get(key, 0)
        }
    
    def get_alerts(self, model_id: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts for a model or all models."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alerts
            if alert["timestamp"] > cutoff_time
        ]
        
        if model_id:
            recent_alerts = [alert for alert in recent_alerts if alert["model_id"] == model_id]
        
        return recent_alerts


class ModelVersionManager:
    """Manages model versions and lifecycle."""
    
    def __init__(self, storage_path: str = "./model_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        self.active_versions: Dict[str, str] = {}  # model_id -> version_id
        self.lock = threading.Lock()
    
    def create_version(
        self,
        model_id: str,
        model_config: Dict[str, Any],
        performance_metrics: Dict[str, float],
        improvement_notes: str = ""
    ) -> ModelVersion:
        """Create a new model version."""
        with self.lock:
            # Determine version number
            existing_versions = self.versions[model_id]
            version_number = len(existing_versions) + 1
            
            # Create version
            version = ModelVersion(
                version_id=f"{model_id}_v{version_number}",
                model_id=model_id,
                version_number=version_number,
                created_at=datetime.utcnow(),
                status=ModelStatus.TRAINING,
                performance_metrics=performance_metrics,
                model_config=model_config,
                improvement_notes=improvement_notes
            )
            
            # Set parent version
            if existing_versions:
                latest_version = max(existing_versions, key=lambda v: v.version_number)
                version.parent_version = latest_version.version_id
            
            self.versions[model_id].append(version)
            
            logger.info("Model version created", version_id=version.version_id, model_id=model_id)
            return version
    
    def update_version_status(
        self,
        version_id: str,
        status: ModelStatus,
        performance_metrics: Optional[Dict[str, float]] = None
    ):
        """Update version status and metrics."""
        with self.lock:
            for model_versions in self.versions.values():
                for version in model_versions:
                    if version.version_id == version_id:
                        version.status = status
                        
                        if performance_metrics:
                            version.performance_metrics.update(performance_metrics)
                        
                        if status == ModelStatus.ACTIVE:
                            version.deployment_date = datetime.utcnow()
                            self.active_versions[version.model_id] = version_id
                        elif status == ModelStatus.RETIRED:
                            version.retirement_date = datetime.utcnow()
                        
                        return
    
    def get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the active version of a model."""
        active_version_id = self.active_versions.get(model_id)
        if not active_version_id:
            return None
        
        for version in self.versions[model_id]:
            if version.version_id == active_version_id:
                return version
        
        return None
    
    def get_version_history(self, model_id: str) -> List[ModelVersion]:
        """Get version history for a model."""
        return sorted(self.versions[model_id], key=lambda v: v.version_number, reverse=True)
    
    def save_model_to_version(self, version_id: str, model_data: Any) -> bool:
        """Save model data to version storage."""
        try:
            version_dir = self.storage_path / version_id
            version_dir.mkdir(exist_ok=True)
            
            file_path = version_dir / "model.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Update version with file path
            for model_versions in self.versions.values():
                for version in model_versions:
                    if version.version_id == version_id:
                        version.file_path = str(file_path)
                        break
            
            return True
            
        except Exception as e:
            logger.error("Failed to save model version", version_id=version_id, error=str(e))
            return False
    
    def load_model_from_version(self, version_id: str) -> Optional[Any]:
        """Load model data from version storage."""
        try:
            for model_versions in self.versions.values():
                for version in model_versions:
                    if version.version_id == version_id and version.file_path:
                        with open(version.file_path, 'rb') as f:
                            return pickle.load(f)
            
            return None
            
        except Exception as e:
            logger.error("Failed to load model version", version_id=version_id, error=str(e))
            return None
    
    def cleanup_old_versions(self, model_id: str, keep_count: int = 10):
        """Clean up old model versions, keeping only the most recent."""
        with self.lock:
            model_versions = self.versions[model_id]
            
            if len(model_versions) <= keep_count:
                return
            
            # Sort by version number and keep most recent
            sorted_versions = sorted(model_versions, key=lambda v: v.version_number, reverse=True)
            versions_to_remove = sorted_versions[keep_count:]
            
            for version in versions_to_remove:
                # Remove file
                if version.file_path and os.path.exists(version.file_path):
                    try:
                        shutil.rmtree(os.path.dirname(version.file_path))
                    except Exception as e:
                        logger.warning("Failed to remove version files", version_id=version.version_id, error=str(e))
                
                # Remove from memory
                model_versions.remove(version)
            
            logger.info("Cleaned up old model versions", model_id=model_id, removed_count=len(versions_to_remove))


class ImprovementPlanner:
    """Plans model improvements based on performance data."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.improvement_history: List[ImprovementPlan] = []
    
    def analyze_improvement_opportunity(
        self,
        model_id: str,
        current_performance: Dict[str, float],
        trigger: ImprovementTrigger
    ) -> Optional[ImprovementPlan]:
        """Analyze if and how a model should be improved."""
        
        # Get performance trends
        trends = {}
        for metric_name in current_performance.keys():
            trends[metric_name] = self.performance_monitor.get_performance_trend(model_id, metric_name)
        
        # Determine if improvement is needed
        improvement_needed = False
        declining_metrics = []
        
        for metric_name, trend_data in trends.items():
            if trend_data["trend"] == "declining" and trend_data["confidence"] > 0.7:
                improvement_needed = True
                declining_metrics.append(metric_name)
        
        if not improvement_needed and trigger != ImprovementTrigger.SCHEDULED:
            return None
        
        # Select improvement strategy
        strategy = self._select_improvement_strategy(trends, declining_metrics)
        
        # Estimate improvement potential
        expected_improvement = self._estimate_improvement_potential(trends, strategy)
        
        # Calculate confidence
        confidence = self._calculate_confidence(trends, strategy)
        
        # Create improvement plan
        plan = ImprovementPlan(
            plan_id=f"plan_{model_id}_{int(datetime.utcnow().timestamp())}",
            model_id=model_id,
            trigger=trigger,
            strategy=strategy,
            expected_improvement=expected_improvement,
            confidence=confidence,
            estimated_duration=self._estimate_duration(strategy),
            resource_requirements=self._estimate_resources(strategy),
            success_criteria=self._define_success_criteria(current_performance, expected_improvement),
            rollback_criteria=self._define_rollback_criteria(current_performance)
        )
        
        self.improvement_history.append(plan)
        
        logger.info("Improvement plan created", plan_id=plan.plan_id, model_id=model_id, strategy=strategy.value)
        return plan
    
    def _select_improvement_strategy(
        self,
        trends: Dict[str, Dict[str, Any]],
        declining_metrics: List[str]
    ) -> ImprovementStrategy:
        """Select the best improvement strategy based on trends."""
        
        # Count declining trends
        declining_count = len(declining_metrics)
        
        # Analyze trend severity
        severe_decline = any(
            trends[metric]["confidence"] > 0.8 and abs(trends[metric]["slope"]) > 0.1
            for metric in declining_metrics
        )
        
        # Select strategy
        if severe_decline or declining_count > 2:
            return ImprovementStrategy.RETRAIN_FROM_SCRATCH
        elif declining_count > 0:
            return ImprovementStrategy.HYPERPARAMETER_TUNE
        else:
            return ImprovementStrategy.INCREMENTAL_UPDATE
    
    def _estimate_improvement_potential(
        self,
        trends: Dict[str, Dict[str, Any]],
        strategy: ImprovementStrategy
    ) -> float:
        """Estimate potential improvement from strategy."""
        
        base_improvements = {
            ImprovementStrategy.RETRAIN_FROM_SCRATCH: 0.15,
            ImprovementStrategy.HYPERPARAMETER_TUNE: 0.08,
            ImprovementStrategy.INCREMENTAL_UPDATE: 0.05,
            ImprovementStrategy.ENSEMBLE_UPGRADE: 0.12,
            ImprovementStrategy.ARCHITECTURE_CHANGE: 0.20,
            ImprovementStrategy.FEATURE_ENGINEERING: 0.10
        }
        
        base_improvement = base_improvements.get(strategy, 0.05)
        
        # Adjust based on current decline severity
        decline_severity = 0.0
        for trend_data in trends.values():
            if trend_data["trend"] == "declining":
                decline_severity += abs(trend_data.get("slope", 0))
        
        # More decline = more improvement potential
        adjustment = min(decline_severity * 0.5, 0.2)
        
        return base_improvement + adjustment
    
    def _calculate_confidence(
        self,
        trends: Dict[str, Dict[str, Any]],
        strategy: ImprovementStrategy
    ) -> float:
        """Calculate confidence in the improvement plan."""
        
        # Base confidence for different strategies
        base_confidence = {
            ImprovementStrategy.RETRAIN_FROM_SCRATCH: 0.8,
            ImprovementStrategy.HYPERPARAMETER_TUNE: 0.7,
            ImprovementStrategy.INCREMENTAL_UPDATE: 0.6,
            ImprovementStrategy.ENSEMBLE_UPGRADE: 0.75,
            ImprovementStrategy.ARCHITECTURE_CHANGE: 0.5,
            ImprovementStrategy.FEATURE_ENGINEERING: 0.65
        }
        
        confidence = base_confidence.get(strategy, 0.6)
        
        # Adjust based on trend reliability
        avg_trend_confidence = np.mean([
            trend_data.get("confidence", 0) for trend_data in trends.values()
        ])
        
        # Weight by trend confidence
        confidence = confidence * 0.7 + avg_trend_confidence * 0.3
        
        return min(confidence, 0.95)
    
    def _estimate_duration(self, strategy: ImprovementStrategy) -> timedelta:
        """Estimate duration for improvement strategy."""
        durations = {
            ImprovementStrategy.RETRAIN_FROM_SCRATCH: timedelta(hours=4),
            ImprovementStrategy.HYPERPARAMETER_TUNE: timedelta(hours=2),
            ImprovementStrategy.INCREMENTAL_UPDATE: timedelta(minutes=30),
            ImprovementStrategy.ENSEMBLE_UPGRADE: timedelta(hours=3),
            ImprovementStrategy.ARCHITECTURE_CHANGE: timedelta(hours=8),
            ImprovementStrategy.FEATURE_ENGINEERING: timedelta(hours=6)
        }
        
        return durations.get(strategy, timedelta(hours=2))
    
    def _estimate_resources(self, strategy: ImprovementStrategy) -> Dict[str, Any]:
        """Estimate resource requirements for strategy."""
        return {
            "cpu_cores": 2 if strategy == ImprovementStrategy.RETRAIN_FROM_SCRATCH else 1,
            "memory_gb": 4 if strategy == ImprovementStrategy.RETRAIN_FROM_SCRATCH else 2,
            "disk_gb": 1,
            "gpu_required": False
        }
    
    def _define_success_criteria(
        self,
        current_performance: Dict[str, float],
        expected_improvement: float
    ) -> Dict[str, float]:
        """Define success criteria for improvement."""
        success_criteria = {}
        
        for metric_name, current_value in current_performance.items():
            # Set improvement threshold
            if "accuracy" in metric_name.lower() or "precision" in metric_name.lower():
                # For metrics that should increase
                success_criteria[metric_name] = current_value * (1 + expected_improvement)
            elif "error" in metric_name.lower() or "loss" in metric_name.lower():
                # For metrics that should decrease
                success_criteria[metric_name] = current_value * (1 - expected_improvement)
            else:
                # Default: increase
                success_criteria[metric_name] = current_value * (1 + expected_improvement)
        
        return success_criteria
    
    def _define_rollback_criteria(self, current_performance: Dict[str, float]) -> Dict[str, float]:
        """Define rollback criteria (performance must not fall below these)."""
        rollback_criteria = {}
        
        for metric_name, current_value in current_performance.items():
            # Set rollback threshold (10% worse than current)
            if "accuracy" in metric_name.lower() or "precision" in metric_name.lower():
                rollback_criteria[metric_name] = current_value * 0.9
            elif "error" in metric_name.lower() or "loss" in metric_name.lower():
                rollback_criteria[metric_name] = current_value * 1.1
            else:
                rollback_criteria[metric_name] = current_value * 0.9
        
        return rollback_criteria


class ContinuousImprovementSystem:
    """
    Complete Continuous Model Improvement System.
    
    Provides comprehensive continuous improvement capabilities with performance monitoring,
    automated retraining, model versioning, and improvement assessment.
    """
    
    def __init__(
        self,
        online_learning_system: OnlineLearningSystem,
        cache: Optional[RedisCache] = None,
        storage_path: str = "./model_versions"
    ):
        self.online_learning_system = online_learning_system
        self.cache = cache
        
        # Core components
        self.performance_monitor = PerformanceMonitor()
        self.version_manager = ModelVersionManager(storage_path)
        self.improvement_planner = ImprovementPlanner(self.performance_monitor)
        
        # System state
        self.active_improvements: Dict[str, Dict[str, Any]] = {}
        self.system_metrics = {
            "total_improvements": 0,
            "successful_improvements": 0,
            "failed_improvements": 0,
            "avg_improvement_time": 0.0
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._improvement_task: Optional[asyncio.Task] = None
        
        logger.info("Continuous improvement system initialized")
    
    async def start_monitoring(self, check_interval: int = 300):  # 5 minutes
        """Start background monitoring and improvement tasks."""
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(check_interval)
        )
        self._improvement_task = asyncio.create_task(
            self._improvement_loop()
        )
        
        logger.info("Continuous improvement monitoring started", check_interval=check_interval)
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._improvement_task:
            self._improvement_task.cancel()
            try:
                await self._improvement_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Continuous improvement monitoring stopped")
    
    async def _monitoring_loop(self, check_interval: int):
        """Background loop for monitoring model performance."""
        while True:
            try:
                await asyncio.sleep(check_interval)
                
                # Check performance for all active models
                for model_id in self.online_learning_system.models.keys():
                    await self._check_model_performance(model_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
    
    async def _improvement_loop(self):
        """Background loop for processing improvement plans."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Process pending improvement plans
                await self._process_improvement_queue()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in improvement loop", error=str(e))
    
    async def _check_model_performance(self, model_id: str):
        """Check performance of a specific model."""
        try:
            model_manager = self.online_learning_system.get_model(model_id)
            if not model_manager or not model_manager.metrics_history:
                return
            
            # Get latest metrics
            latest_metrics = model_manager.metrics_history[-1]
            
            # Update performance monitor
            self.performance_monitor.update_performance(
                model_id, "train_score", latest_metrics.train_score
            )
            self.performance_monitor.update_performance(
                model_id, "test_score", latest_metrics.test_score
            )
            self.performance_monitor.update_performance(
                model_id, "drift_score", latest_metrics.drift_score
            )
            
            # Check for improvement opportunities
            current_performance = {
                "train_score": latest_metrics.train_score,
                "test_score": latest_metrics.test_score,
                "drift_score": latest_metrics.drift_score
            }
            
            # Check for alerts
            alerts = self.performance_monitor.get_alerts(model_id, hours=1)
            if alerts:
                # High-priority alerts trigger immediate improvement planning
                for alert in alerts:
                    if alert["severity"] == "high":
                        await self._trigger_improvement(
                            model_id, 
                            ImprovementTrigger.PERFORMANCE_DEGRADATION,
                            current_performance
                        )
                        break
            
        except Exception as e:
            logger.error("Failed to check model performance", model_id=model_id, error=str(e))
    
    async def _trigger_improvement(
        self,
        model_id: str,
        trigger: ImprovementTrigger,
        current_performance: Dict[str, float]
    ):
        """Trigger improvement planning for a model."""
        
        # Skip if already improving
        if model_id in self.active_improvements:
            return
        
        # Create improvement plan
        plan = self.improvement_planner.analyze_improvement_opportunity(
            model_id, current_performance, trigger
        )
        
        if plan is None:
            return
        
        # Queue improvement
        self.active_improvements[model_id] = {
            "plan": plan,
            "status": "queued",
            "started_at": None,
            "current_step": None
        }
        
        logger.info("Improvement triggered", model_id=model_id, trigger=trigger.value, plan_id=plan.plan_id)
    
    async def _process_improvement_queue(self):
        """Process queued improvement plans."""
        for model_id, improvement_data in list(self.active_improvements.items()):
            if improvement_data["status"] == "queued":
                await self._execute_improvement_plan(model_id, improvement_data["plan"])
    
    async def _execute_improvement_plan(self, model_id: str, plan: ImprovementPlan):
        """Execute an improvement plan for a model."""
        try:
            self.active_improvements[model_id]["status"] = "executing"
            self.active_improvements[model_id]["started_at"] = datetime.utcnow()
            
            start_time = datetime.utcnow()
            
            # Get current model
            model_manager = self.online_learning_system.get_model(model_id)
            if not model_manager:
                raise ValueError(f"Model {model_id} not found")
            
            # Create new model version
            current_performance = {
                "train_score": model_manager.metrics_history[-1].train_score if model_manager.metrics_history else 0,
                "test_score": model_manager.metrics_history[-1].test_score if model_manager.metrics_history else 0
            }
            
            new_version = self.version_manager.create_version(
                model_id=model_id,
                model_config=model_manager.config.__dict__,
                performance_metrics=current_performance,
                improvement_notes=f"Improvement via {plan.strategy.value}"
            )
            
            # Execute improvement strategy
            success = await self._apply_improvement_strategy(model_manager, plan, new_version)
            
            if success:
                # Evaluate new model
                evaluation_result = await self._evaluate_improved_model(model_manager, plan, new_version)
                
                if evaluation_result["meets_success_criteria"]:
                    # Deploy new version
                    self.version_manager.update_version_status(new_version.version_id, ModelStatus.ACTIVE)
                    
                    # Save model
                    model_state = model_manager.get_model_state()
                    self.version_manager.save_model_to_version(new_version.version_id, model_state)
                    
                    self.system_metrics["successful_improvements"] += 1
                    
                    logger.info("Model improvement successful", 
                               model_id=model_id, 
                               version_id=new_version.version_id,
                               improvement=evaluation_result["improvement_achieved"])
                    
                else:
                    # Rollback - revert to previous version
                    self.version_manager.update_version_status(new_version.version_id, ModelStatus.FAILED)
                    
                    # Load previous version if exists
                    previous_versions = self.version_manager.get_version_history(model_id)
                    if len(previous_versions) > 1:
                        previous_version = previous_versions[1]  # Second most recent
                        if previous_version.status == ModelStatus.ACTIVE:
                            previous_state = self.version_manager.load_model_from_version(previous_version.version_id)
                            if previous_state:
                                model_manager.load_model_state(previous_state)
                    
                    self.system_metrics["failed_improvements"] += 1
                    
                    logger.warning("Model improvement failed criteria, rolled back", 
                                 model_id=model_id, 
                                 version_id=new_version.version_id)
            else:
                self.version_manager.update_version_status(new_version.version_id, ModelStatus.FAILED)
                self.system_metrics["failed_improvements"] += 1
                
                logger.error("Model improvement execution failed", model_id=model_id)
            
            # Update timing metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            total_improvements = self.system_metrics["successful_improvements"] + self.system_metrics["failed_improvements"]
            if total_improvements > 0:
                self.system_metrics["avg_improvement_time"] = (
                    (self.system_metrics["avg_improvement_time"] * (total_improvements - 1) + duration) / 
                    total_improvements
                )
            
            self.system_metrics["total_improvements"] += 1
            
        except Exception as e:
            logger.error("Failed to execute improvement plan", model_id=model_id, plan_id=plan.plan_id, error=str(e))
            self.system_metrics["failed_improvements"] += 1
        
        finally:
            # Clean up
            if model_id in self.active_improvements:
                del self.active_improvements[model_id]
    
    async def _apply_improvement_strategy(
        self,
        model_manager,
        plan: ImprovementPlan,
        version: ModelVersion
    ) -> bool:
        """Apply the improvement strategy to the model."""
        
        try:
            if plan.strategy == ImprovementStrategy.RETRAIN_FROM_SCRATCH:
                # Reinitialize model with same config
                model_manager._initialize_model()
                return True
                
            elif plan.strategy == ImprovementStrategy.HYPERPARAMETER_TUNE:
                # Adjust learning rate and other hyperparameters
                if hasattr(model_manager.model, 'set_params'):
                    # Increase learning rate slightly
                    current_lr = getattr(model_manager.model, 'eta0', model_manager.config.initial_learning_rate)
                    new_lr = min(current_lr * 1.1, model_manager.config.max_learning_rate)
                    
                    try:
                        model_manager.model.set_params(eta0=new_lr)
                        return True
                    except Exception:
                        pass
                return True
                
            elif plan.strategy == ImprovementStrategy.INCREMENTAL_UPDATE:
                # Already handled by online learning system
                return True
                
            else:
                # Other strategies not implemented yet
                logger.warning("Improvement strategy not implemented", strategy=plan.strategy.value)
                return False
                
        except Exception as e:
            logger.error("Failed to apply improvement strategy", strategy=plan.strategy.value, error=str(e))
            return False
    
    async def _evaluate_improved_model(
        self,
        model_manager,
        plan: ImprovementPlan,
        version: ModelVersion
    ) -> Dict[str, Any]:
        """Evaluate the improved model against success criteria."""
        
        # Get current performance (would normally use validation set)
        if model_manager.metrics_history:
            latest_metrics = model_manager.metrics_history[-1]
            current_performance = {
                "train_score": latest_metrics.train_score,
                "test_score": latest_metrics.test_score
            }
        else:
            current_performance = {"train_score": 0, "test_score": 0}
        
        # Check success criteria
        meets_success_criteria = True
        improvement_achieved = {}
        
        for metric_name, target_value in plan.success_criteria.items():
            current_value = current_performance.get(metric_name, 0)
            
            if "accuracy" in metric_name.lower() or "score" in metric_name.lower():
                # Higher is better
                meets_criterion = current_value >= target_value
                improvement = (current_value - target_value) / target_value if target_value > 0 else 0
            else:
                # Lower is better (errors)
                meets_criterion = current_value <= target_value
                improvement = (target_value - current_value) / target_value if target_value > 0 else 0
            
            improvement_achieved[metric_name] = improvement
            
            if not meets_criterion:
                meets_success_criteria = False
        
        # Check rollback criteria
        violates_rollback = False
        for metric_name, min_value in plan.rollback_criteria.items():
            current_value = current_performance.get(metric_name, 0)
            
            if "accuracy" in metric_name.lower() or "score" in metric_name.lower():
                if current_value < min_value:
                    violates_rollback = True
                    break
            else:
                if current_value > min_value:
                    violates_rollback = True
                    break
        
        # Update version performance
        version.performance_metrics.update(current_performance)
        
        result = {
            "meets_success_criteria": meets_success_criteria and not violates_rollback,
            "violates_rollback": violates_rollback,
            "improvement_achieved": improvement_achieved,
            "current_performance": current_performance,
            "evaluation_timestamp": datetime.utcnow()
        }
        
        return result
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a model."""
        # Get active version
        active_version = self.version_manager.get_active_version(model_id)
        
        # Get performance trends
        model_manager = self.online_learning_system.get_model(model_id)
        current_performance = {}
        if model_manager and model_manager.metrics_history:
            latest_metrics = model_manager.metrics_history[-1]
            current_performance = {
                "train_score": latest_metrics.train_score,
                "test_score": latest_metrics.test_score,
                "drift_score": latest_metrics.drift_score
            }
        
        trends = {}
        for metric_name in current_performance.keys():
            trends[metric_name] = self.performance_monitor.get_performance_trend(model_id, metric_name)
        
        # Get alerts
        recent_alerts = self.performance_monitor.get_alerts(model_id, hours=24)
        
        # Get improvement status
        improvement_status = self.active_improvements.get(model_id)
        
        return {
            "model_id": model_id,
            "active_version": active_version.to_dict() if active_version else None,
            "current_performance": current_performance,
            "performance_trends": trends,
            "recent_alerts": recent_alerts,
            "improvement_in_progress": improvement_status is not None,
            "improvement_status": improvement_status,
            "version_history": [v.to_dict() for v in self.version_manager.get_version_history(model_id)[:5]]
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide improvement metrics."""
        return {
            **self.system_metrics,
            "active_improvements": len(self.active_improvements),
            "total_models_managed": len(self.version_manager.versions),
            "success_rate": (
                self.system_metrics["successful_improvements"] / 
                max(1, self.system_metrics["total_improvements"])
            )
        }
    
    async def manual_trigger_improvement(
        self,
        model_id: str,
        strategy: Optional[ImprovementStrategy] = None
    ) -> bool:
        """Manually trigger improvement for a model."""
        
        model_manager = self.online_learning_system.get_model(model_id)
        if not model_manager:
            return False
        
        # Get current performance
        current_performance = {}
        if model_manager.metrics_history:
            latest_metrics = model_manager.metrics_history[-1]
            current_performance = {
                "train_score": latest_metrics.train_score,
                "test_score": latest_metrics.test_score,
                "drift_score": latest_metrics.drift_score
            }
        
        # Create manual improvement plan
        if strategy:
            # Override strategy selection
            plan = ImprovementPlan(
                plan_id=f"manual_{model_id}_{int(datetime.utcnow().timestamp())}",
                model_id=model_id,
                trigger=ImprovementTrigger.MANUAL,
                strategy=strategy,
                expected_improvement=0.1,  # Conservative estimate
                confidence=0.6,
                estimated_duration=timedelta(hours=2),
                resource_requirements={"cpu_cores": 1, "memory_gb": 2},
                success_criteria=self.improvement_planner._define_success_criteria(current_performance, 0.1),
                rollback_criteria=self.improvement_planner._define_rollback_criteria(current_performance)
            )
        else:
            # Use planner
            plan = self.improvement_planner.analyze_improvement_opportunity(
                model_id, current_performance, ImprovementTrigger.MANUAL
            )
        
        if plan:
            self.active_improvements[model_id] = {
                "plan": plan,
                "status": "queued",
                "started_at": None,
                "current_step": None
            }
            return True
        
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        try:
            # Test performance monitoring
            test_model_id = "health_check_model"
            self.performance_monitor.update_performance(test_model_id, "test_metric", 0.8)
            trend = self.performance_monitor.get_performance_trend(test_model_id, "test_metric")
            
            # Test version management
            test_version = self.version_manager.create_version(
                test_model_id,
                {"type": "test"},
                {"accuracy": 0.8},
                "Health check version"
            )
            
            # Clean up test data
            if test_model_id in self.version_manager.versions:
                del self.version_manager.versions[test_model_id]
            
            return {
                "status": "healthy",
                "performance_monitoring": trend is not None,
                "version_management": test_version is not None,
                "system_metrics": self.get_system_metrics(),
                "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done()
            }
            
        except Exception as e:
            logger.error("Continuous improvement health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "system_metrics": self.get_system_metrics()
            }