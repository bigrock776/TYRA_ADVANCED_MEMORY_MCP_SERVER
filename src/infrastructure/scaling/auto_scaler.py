"""
Auto-Scaling Engine for Dynamic Resource Management.

This module provides comprehensive auto-scaling capabilities using local system metrics,
including CPU, memory, and queue length monitoring, scaling decisions with local algorithms,
and safety controls with rate limiting. All processing is performed locally with zero external dependencies.
"""

import asyncio
import psutil
import time
import numpy as np
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import threading
import math

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ScalingAction(str, Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    EMERGENCY_SCALE = "emergency_scale"


class MetricType(str, Enum):
    """Types of system metrics to monitor."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    ACTIVE_CONNECTIONS = "active_connections"


class ScalingStrategy(str, Enum):
    """Strategies for scaling decisions."""
    REACTIVE = "reactive"           # React to current metrics
    PREDICTIVE = "predictive"       # Predict future load
    HYBRID = "hybrid"              # Combine reactive and predictive
    CONSERVATIVE = "conservative"   # Scale slowly and carefully
    AGGRESSIVE = "aggressive"      # Scale quickly for performance


@dataclass
class SystemMetric:
    """Represents a system metric measurement."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_above_threshold(self, threshold: float) -> bool:
        """Check if metric is above threshold."""
        return self.value > threshold
    
    def is_below_threshold(self, threshold: float) -> bool:
        """Check if metric is below threshold."""
        return self.value < threshold


@dataclass
class ScalingDecision:
    """Represents a scaling decision."""
    action: ScalingAction
    reason: str
    confidence: float
    target_instances: int
    current_instances: int
    metrics_snapshot: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    safety_checks_passed: bool = True
    estimated_impact: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingRule:
    """Defines a scaling rule."""
    name: str
    metric_type: MetricType
    scale_up_threshold: float
    scale_down_threshold: float
    min_duration_seconds: int = 60  # Minimum time above/below threshold
    cooldown_seconds: int = 300     # Cooldown between actions
    weight: float = 1.0             # Weight in decision making
    enabled: bool = True


class MetricsCollector:
    """
    Local System Metrics Collection Engine.
    
    Collects CPU, memory, disk, network, and application-specific metrics.
    """
    
    def __init__(self, collection_interval: int = 5):
        self.collection_interval = collection_interval
        self.metrics_history: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Application-specific metrics
        self.queue_lengths: Dict[str, int] = {}
        self.response_times: deque = deque(maxlen=100)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.active_connections = 0
    
    async def start_collection(self):
        """Start metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        if not self.running:
            return
        
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Collect all system metrics."""
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_metric = SystemMetric(
            metric_type=MetricType.CPU_USAGE,
            value=cpu_percent,
            timestamp=timestamp,
            unit="percent"
        )
        self.metrics_history[MetricType.CPU_USAGE].append(cpu_metric)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_metric = SystemMetric(
            metric_type=MetricType.MEMORY_USAGE,
            value=memory.percent,
            timestamp=timestamp,
            unit="percent",
            metadata={
                'total': memory.total,
                'available': memory.available,
                'used': memory.used
            }
        )
        self.metrics_history[MetricType.MEMORY_USAGE].append(memory_metric)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_metric = SystemMetric(
            metric_type=MetricType.DISK_USAGE,
            value=(disk.used / disk.total) * 100,
            timestamp=timestamp,
            unit="percent",
            metadata={
                'total': disk.total,
                'used': disk.used,
                'free': disk.free
            }
        )
        self.metrics_history[MetricType.DISK_USAGE].append(disk_metric)
        
        # Network metrics
        network = psutil.net_io_counters()
        network_metric = SystemMetric(
            metric_type=MetricType.NETWORK_IO,
            value=network.bytes_sent + network.bytes_recv,
            timestamp=timestamp,
            unit="bytes",
            metadata={
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        )
        self.metrics_history[MetricType.NETWORK_IO].append(network_metric)
        
        # Application metrics
        await self._collect_application_metrics(timestamp)
    
    async def _collect_application_metrics(self, timestamp: datetime):
        """Collect application-specific metrics."""
        
        # Queue length metrics
        total_queue_length = sum(self.queue_lengths.values())
        queue_metric = SystemMetric(
            metric_type=MetricType.QUEUE_LENGTH,
            value=total_queue_length,
            timestamp=timestamp,
            unit="count",
            metadata={'queues': self.queue_lengths.copy()}
        )
        self.metrics_history[MetricType.QUEUE_LENGTH].append(queue_metric)
        
        # Response time metrics
        if self.response_times:
            avg_response_time = np.mean(list(self.response_times))
            response_metric = SystemMetric(
                metric_type=MetricType.RESPONSE_TIME,
                value=avg_response_time,
                timestamp=timestamp,
                unit="milliseconds",
                metadata={
                    'p95': np.percentile(list(self.response_times), 95) if len(self.response_times) > 1 else avg_response_time,
                    'p99': np.percentile(list(self.response_times), 99) if len(self.response_times) > 1 else avg_response_time
                }
            )
            self.metrics_history[MetricType.RESPONSE_TIME].append(response_metric)
        
        # Error rate metrics
        total_errors = sum(self.error_counts.values())
        error_metric = SystemMetric(
            metric_type=MetricType.ERROR_RATE,
            value=total_errors,
            timestamp=timestamp,
            unit="count",
            metadata={'errors_by_type': self.error_counts.copy()}
        )
        self.metrics_history[MetricType.ERROR_RATE].append(error_metric)
        
        # Active connections
        connection_metric = SystemMetric(
            metric_type=MetricType.ACTIVE_CONNECTIONS,
            value=self.active_connections,
            timestamp=timestamp,
            unit="count"
        )
        self.metrics_history[MetricType.ACTIVE_CONNECTIONS].append(connection_metric)
    
    def record_response_time(self, response_time_ms: float):
        """Record a response time measurement."""
        self.response_times.append(response_time_ms)
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
    
    def update_queue_length(self, queue_name: str, length: int):
        """Update queue length for a specific queue."""
        self.queue_lengths[queue_name] = length
    
    def update_active_connections(self, count: int):
        """Update active connections count."""
        self.active_connections = count
    
    def get_recent_metrics(self, metric_type: MetricType, duration_minutes: int = 5) -> List[SystemMetric]:
        """Get recent metrics for a specific type."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        
        if metric_type not in self.metrics_history:
            return []
        
        return [
            metric for metric in self.metrics_history[metric_type]
            if metric.timestamp >= cutoff_time
        ]
    
    def get_average_metric(self, metric_type: MetricType, duration_minutes: int = 5) -> Optional[float]:
        """Get average value for a metric over duration."""
        recent_metrics = self.get_recent_metrics(metric_type, duration_minutes)
        
        if not recent_metrics:
            return None
        
        return np.mean([metric.value for metric in recent_metrics])


class LoadPredictor:
    """
    Predictive Load Analysis Engine.
    
    Predicts future load based on historical patterns and trends.
    """
    
    def __init__(self, history_window_hours: int = 24):
        self.history_window_hours = history_window_hours
        self.load_patterns: Dict[str, List[float]] = {}
        self.trend_coefficients: Dict[MetricType, float] = {}
    
    async def analyze_patterns(self, metrics_collector: MetricsCollector):
        """Analyze historical patterns to identify trends."""
        
        for metric_type in MetricType:
            # Get longer history for pattern analysis
            recent_metrics = metrics_collector.get_recent_metrics(
                metric_type, duration_minutes=self.history_window_hours * 60
            )
            
            if len(recent_metrics) < 10:
                continue
            
            # Extract time series
            timestamps = [metric.timestamp.timestamp() for metric in recent_metrics]
            values = [metric.value for metric in recent_metrics]
            
            # Calculate trend using linear regression
            trend_coeff = self._calculate_trend(timestamps, values)
            self.trend_coefficients[metric_type] = trend_coeff
            
            # Identify daily/hourly patterns
            await self._identify_patterns(metric_type, recent_metrics)
    
    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> float:
        """Calculate trend coefficient using linear regression."""
        
        if len(timestamps) < 2:
            return 0.0
        
        # Normalize timestamps
        min_time = min(timestamps)
        x = np.array([(t - min_time) / 3600 for t in timestamps])  # Hours
        y = np.array(values)
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    async def _identify_patterns(self, metric_type: MetricType, metrics: List[SystemMetric]):
        """Identify recurring patterns in metrics."""
        
        # Group by hour of day
        hourly_patterns = defaultdict(list)
        
        for metric in metrics:
            hour = metric.timestamp.hour
            hourly_patterns[hour].append(metric.value)
        
        # Calculate average for each hour
        hourly_averages = {}
        for hour, values in hourly_patterns.items():
            hourly_averages[hour] = np.mean(values)
        
        # Store pattern
        pattern_key = f"{metric_type.value}_hourly"
        self.load_patterns[pattern_key] = [
            hourly_averages.get(hour, 0.0) for hour in range(24)
        ]
    
    async def predict_load(
        self,
        metric_type: MetricType,
        prediction_horizon_minutes: int = 30
    ) -> Optional[float]:
        """Predict future load for a specific metric."""
        
        # Get trend coefficient
        trend = self.trend_coefficients.get(metric_type, 0.0)
        
        # Get hourly pattern
        pattern_key = f"{metric_type.value}_hourly"
        if pattern_key not in self.load_patterns:
            return None
        
        hourly_pattern = self.load_patterns[pattern_key]
        
        # Predict based on time of day and trend
        future_time = datetime.utcnow() + timedelta(minutes=prediction_horizon_minutes)
        future_hour = future_time.hour
        
        # Base prediction from pattern
        base_prediction = hourly_pattern[future_hour]
        
        # Apply trend
        trend_adjustment = trend * (prediction_horizon_minutes / 60)
        
        predicted_value = base_prediction + trend_adjustment
        
        # Ensure non-negative
        return max(0.0, predicted_value)


class SafetyController:
    """
    Safety Controls for Auto-Scaling.
    
    Implements rate limiting, bounds checking, and safety validations.
    """
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        max_scale_per_hour: int = 5,
        min_cooldown_seconds: int = 300
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.max_scale_per_hour = max_scale_per_hour
        self.min_cooldown_seconds = min_cooldown_seconds
        
        # Track scaling history
        self.scaling_history: deque = deque(maxlen=100)
        self.last_scaling_time: Optional[datetime] = None
    
    def validate_scaling_decision(
        self,
        decision: ScalingDecision,
        current_instances: int
    ) -> Tuple[bool, str]:
        """Validate a scaling decision against safety rules."""
        
        # Check instance bounds
        if decision.target_instances < self.min_instances:
            return False, f"Target instances ({decision.target_instances}) below minimum ({self.min_instances})"
        
        if decision.target_instances > self.max_instances:
            return False, f"Target instances ({decision.target_instances}) above maximum ({self.max_instances})"
        
        # Check cooldown period
        if self.last_scaling_time:
            time_since_last = (datetime.utcnow() - self.last_scaling_time).total_seconds()
            if time_since_last < self.min_cooldown_seconds:
                return False, f"Cooldown period not elapsed ({time_since_last:.0f}s < {self.min_cooldown_seconds}s)"
        
        # Check rate limiting
        if not self._check_rate_limit():
            return False, "Rate limit exceeded for scaling operations"
        
        # Check for oscillation
        if self._detect_oscillation(decision):
            return False, "Oscillation detected in scaling decisions"
        
        return True, "Safety checks passed"
    
    def _check_rate_limit(self) -> bool:
        """Check if scaling rate limit is exceeded."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_scalings = [
            event for event in self.scaling_history
            if event['timestamp'] >= cutoff_time
        ]
        
        return len(recent_scalings) < self.max_scale_per_hour
    
    def _detect_oscillation(self, decision: ScalingDecision) -> bool:
        """Detect if scaling decisions are oscillating."""
        
        if len(self.scaling_history) < 4:
            return False
        
        # Check last 4 decisions for up-down-up-down pattern
        recent_actions = [event['action'] for event in list(self.scaling_history)[-4:]]
        
        oscillation_patterns = [
            [ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN, ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN],
            [ScalingAction.SCALE_DOWN, ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN, ScalingAction.SCALE_UP]
        ]
        
        return recent_actions in oscillation_patterns
    
    def record_scaling_event(self, decision: ScalingDecision):
        """Record a scaling event."""
        
        event = {
            'timestamp': decision.timestamp,
            'action': decision.action,
            'target_instances': decision.target_instances,
            'reason': decision.reason
        }
        
        self.scaling_history.append(event)
        self.last_scaling_time = decision.timestamp


class AutoScalingEngine:
    """
    Main Auto-Scaling Engine.
    
    Coordinates metrics collection, decision making, and scaling actions.
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        strategy: ScalingStrategy = ScalingStrategy.HYBRID
    ):
        self.redis_cache = redis_cache
        self.strategy = strategy
        
        # Core components
        self.metrics_collector = MetricsCollector()
        self.load_predictor = LoadPredictor()
        self.safety_controller = SafetyController()
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = self._create_default_rules()
        
        # State
        self.current_instances = 1
        self.target_instances = 1
        self.running = False
        self.decision_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'scale_up_decisions': 0,
            'scale_down_decisions': 0,
            'safety_blocks': 0,
            'prediction_accuracy': 0.0
        }
        
        logger.info(f"AutoScalingEngine initialized with {strategy.value} strategy")
    
    def _create_default_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        
        return [
            ScalingRule(
                name="cpu_scaling",
                metric_type=MetricType.CPU_USAGE,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                min_duration_seconds=120,
                cooldown_seconds=300,
                weight=0.4
            ),
            ScalingRule(
                name="memory_scaling",
                metric_type=MetricType.MEMORY_USAGE,
                scale_up_threshold=85.0,
                scale_down_threshold=40.0,
                min_duration_seconds=60,
                cooldown_seconds=300,
                weight=0.3
            ),
            ScalingRule(
                name="queue_scaling",
                metric_type=MetricType.QUEUE_LENGTH,
                scale_up_threshold=100.0,
                scale_down_threshold=10.0,
                min_duration_seconds=30,
                cooldown_seconds=180,
                weight=0.2
            ),
            ScalingRule(
                name="response_time_scaling",
                metric_type=MetricType.RESPONSE_TIME,
                scale_up_threshold=500.0,  # 500ms
                scale_down_threshold=100.0,  # 100ms
                min_duration_seconds=60,
                cooldown_seconds=240,
                weight=0.1
            )
        ]
    
    async def start(self):
        """Start the auto-scaling engine."""
        
        if self.running:
            return
        
        self.running = True
        
        # Start metrics collection
        await self.metrics_collector.start_collection()
        
        # Start decision making loop
        self.decision_task = asyncio.create_task(self._decision_loop())
        
        logger.info("Auto-scaling engine started")
    
    async def stop(self):
        """Stop the auto-scaling engine."""
        
        if not self.running:
            return
        
        self.running = False
        
        # Stop metrics collection
        await self.metrics_collector.stop_collection()
        
        # Stop decision making
        if self.decision_task:
            self.decision_task.cancel()
            try:
                await self.decision_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaling engine stopped")
    
    async def _decision_loop(self):
        """Main decision making loop."""
        
        decision_interval = 30  # seconds
        
        while self.running:
            try:
                # Analyze patterns periodically
                if self.stats['total_decisions'] % 10 == 0:
                    await self.load_predictor.analyze_patterns(self.metrics_collector)
                
                # Make scaling decision
                decision = await self._make_scaling_decision()
                
                if decision.action != ScalingAction.NO_ACTION:
                    await self._execute_scaling_decision(decision)
                
                await asyncio.sleep(decision_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Decision loop error: {e}")
                await asyncio.sleep(decision_interval)
    
    async def _make_scaling_decision(self) -> ScalingDecision:
        """Make a scaling decision based on current metrics and strategy."""
        
        # Collect current metrics snapshot
        metrics_snapshot = {}
        decision_factors = []
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Get recent average for this metric
            avg_value = self.metrics_collector.get_average_metric(
                rule.metric_type, duration_minutes=rule.min_duration_seconds // 60
            )
            
            if avg_value is None:
                continue
            
            metrics_snapshot[rule.metric_type.value] = avg_value
            
            # Evaluate rule
            if avg_value > rule.scale_up_threshold:
                decision_factors.append({
                    'rule': rule.name,
                    'action': ScalingAction.SCALE_UP,
                    'confidence': min(1.0, (avg_value - rule.scale_up_threshold) / rule.scale_up_threshold),
                    'weight': rule.weight
                })
            elif avg_value < rule.scale_down_threshold:
                decision_factors.append({
                    'rule': rule.name,
                    'action': ScalingAction.SCALE_DOWN,
                    'confidence': min(1.0, (rule.scale_down_threshold - avg_value) / rule.scale_down_threshold),
                    'weight': rule.weight
                })
        
        # Apply strategy-specific logic
        decision = await self._apply_scaling_strategy(decision_factors, metrics_snapshot)
        
        # Update statistics
        self.stats['total_decisions'] += 1
        
        return decision
    
    async def _apply_scaling_strategy(
        self,
        decision_factors: List[Dict[str, Any]],
        metrics_snapshot: Dict[str, float]
    ) -> ScalingDecision:
        """Apply scaling strategy to make final decision."""
        
        if not decision_factors:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="No scaling rules triggered",
                confidence=1.0,
                target_instances=self.current_instances,
                current_instances=self.current_instances,
                metrics_snapshot=metrics_snapshot
            )
        
        if self.strategy == ScalingStrategy.REACTIVE:
            return await self._reactive_strategy(decision_factors, metrics_snapshot)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            return await self._predictive_strategy(decision_factors, metrics_snapshot)
        elif self.strategy == ScalingStrategy.HYBRID:
            return await self._hybrid_strategy(decision_factors, metrics_snapshot)
        elif self.strategy == ScalingStrategy.CONSERVATIVE:
            return await self._conservative_strategy(decision_factors, metrics_snapshot)
        else:  # AGGRESSIVE
            return await self._aggressive_strategy(decision_factors, metrics_snapshot)
    
    async def _reactive_strategy(
        self,
        decision_factors: List[Dict[str, Any]],
        metrics_snapshot: Dict[str, float]
    ) -> ScalingDecision:
        """Reactive scaling based on current metrics only."""
        
        # Calculate weighted vote
        scale_up_score = 0.0
        scale_down_score = 0.0
        
        for factor in decision_factors:
            weight = factor['weight'] * factor['confidence']
            
            if factor['action'] == ScalingAction.SCALE_UP:
                scale_up_score += weight
            else:
                scale_down_score += weight
        
        # Make decision
        if scale_up_score > scale_down_score and scale_up_score > 0.3:
            action = ScalingAction.SCALE_UP
            target = self.current_instances + 1
            confidence = scale_up_score
            reason = f"Reactive scale up (score: {scale_up_score:.2f})"
        elif scale_down_score > scale_up_score and scale_down_score > 0.3:
            action = ScalingAction.SCALE_DOWN
            target = max(1, self.current_instances - 1)
            confidence = scale_down_score
            reason = f"Reactive scale down (score: {scale_down_score:.2f})"
        else:
            action = ScalingAction.NO_ACTION
            target = self.current_instances
            confidence = 1.0
            reason = "No clear scaling signal"
        
        return ScalingDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            target_instances=target,
            current_instances=self.current_instances,
            metrics_snapshot=metrics_snapshot
        )
    
    async def _predictive_strategy(
        self,
        decision_factors: List[Dict[str, Any]],
        metrics_snapshot: Dict[str, float]
    ) -> ScalingDecision:
        """Predictive scaling based on forecasted load."""
        
        # Get predictions for key metrics
        predictions = {}
        for metric_type in [MetricType.CPU_USAGE, MetricType.MEMORY_USAGE, MetricType.QUEUE_LENGTH]:
            prediction = await self.load_predictor.predict_load(metric_type, prediction_horizon_minutes=15)
            if prediction is not None:
                predictions[metric_type] = prediction
        
        # Evaluate predictions against thresholds
        predicted_scale_up = 0
        predicted_scale_down = 0
        
        for rule in self.scaling_rules:
            if rule.metric_type in predictions:
                predicted_value = predictions[rule.metric_type]
                
                if predicted_value > rule.scale_up_threshold:
                    predicted_scale_up += 1
                elif predicted_value < rule.scale_down_threshold:
                    predicted_scale_down += 1
        
        # Make decision
        if predicted_scale_up > predicted_scale_down:
            action = ScalingAction.SCALE_UP
            target = self.current_instances + 1
            confidence = 0.7  # Lower confidence for predictions
            reason = f"Predictive scale up based on forecasts"
        elif predicted_scale_down > predicted_scale_up:
            action = ScalingAction.SCALE_DOWN
            target = max(1, self.current_instances - 1)
            confidence = 0.7
            reason = f"Predictive scale down based on forecasts"
        else:
            action = ScalingAction.NO_ACTION
            target = self.current_instances
            confidence = 1.0
            reason = "No predictive scaling signal"
        
        return ScalingDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            target_instances=target,
            current_instances=self.current_instances,
            metrics_snapshot=metrics_snapshot,
            estimated_impact={'predictions': predictions}
        )
    
    async def _hybrid_strategy(
        self,
        decision_factors: List[Dict[str, Any]],
        metrics_snapshot: Dict[str, float]
    ) -> ScalingDecision:
        """Hybrid strategy combining reactive and predictive approaches."""
        
        # Get both reactive and predictive decisions
        reactive_decision = await self._reactive_strategy(decision_factors, metrics_snapshot)
        predictive_decision = await self._predictive_strategy(decision_factors, metrics_snapshot)
        
        # Combine decisions with weighting
        reactive_weight = 0.7
        predictive_weight = 0.3
        
        # If both agree, high confidence
        if reactive_decision.action == predictive_decision.action:
            confidence = reactive_weight * reactive_decision.confidence + predictive_weight * predictive_decision.confidence
            
            return ScalingDecision(
                action=reactive_decision.action,
                reason=f"Hybrid: {reactive_decision.reason} + {predictive_decision.reason}",
                confidence=confidence,
                target_instances=reactive_decision.target_instances,
                current_instances=self.current_instances,
                metrics_snapshot=metrics_snapshot
            )
        
        # If they disagree, prefer reactive with lower confidence
        confidence = reactive_decision.confidence * 0.6
        
        return ScalingDecision(
            action=reactive_decision.action,
            reason=f"Hybrid (reactive preferred): {reactive_decision.reason}",
            confidence=confidence,
            target_instances=reactive_decision.target_instances,
            current_instances=self.current_instances,
            metrics_snapshot=metrics_snapshot
        )
    
    async def _conservative_strategy(
        self,
        decision_factors: List[Dict[str, Any]],
        metrics_snapshot: Dict[str, float]
    ) -> ScalingDecision:
        """Conservative scaling strategy."""
        
        # Use reactive strategy but with higher thresholds
        reactive_decision = await self._reactive_strategy(decision_factors, metrics_snapshot)
        
        # Only scale if confidence is very high
        if reactive_decision.confidence < 0.8:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason="Conservative: confidence too low for scaling",
                confidence=1.0,
                target_instances=self.current_instances,
                current_instances=self.current_instances,
                metrics_snapshot=metrics_snapshot
            )
        
        # Scale by smaller increments
        if reactive_decision.action == ScalingAction.SCALE_UP:
            # Only add one instance at a time
            target = self.current_instances + 1
        elif reactive_decision.action == ScalingAction.SCALE_DOWN:
            # Only remove one instance at a time
            target = max(1, self.current_instances - 1)
        else:
            target = self.current_instances
        
        return ScalingDecision(
            action=reactive_decision.action,
            reason=f"Conservative: {reactive_decision.reason}",
            confidence=reactive_decision.confidence,
            target_instances=target,
            current_instances=self.current_instances,
            metrics_snapshot=metrics_snapshot
        )
    
    async def _aggressive_strategy(
        self,
        decision_factors: List[Dict[str, Any]],
        metrics_snapshot: Dict[str, float]
    ) -> ScalingDecision:
        """Aggressive scaling strategy."""
        
        # Use reactive strategy but scale more aggressively
        reactive_decision = await self._reactive_strategy(decision_factors, metrics_snapshot)
        
        if reactive_decision.action == ScalingAction.SCALE_UP:
            # Scale up by more instances based on severity
            scale_factor = min(3, max(1, int(reactive_decision.confidence * 3)))
            target = self.current_instances + scale_factor
        elif reactive_decision.action == ScalingAction.SCALE_DOWN:
            # Scale down more gradually to avoid over-scaling
            target = max(1, self.current_instances - 1)
        else:
            target = self.current_instances
        
        return ScalingDecision(
            action=reactive_decision.action,
            reason=f"Aggressive: {reactive_decision.reason}",
            confidence=reactive_decision.confidence,
            target_instances=target,
            current_instances=self.current_instances,
            metrics_snapshot=metrics_snapshot
        )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        
        # Validate with safety controller
        is_safe, safety_reason = self.safety_controller.validate_scaling_decision(
            decision, self.current_instances
        )
        
        if not is_safe:
            decision.safety_checks_passed = False
            self.stats['safety_blocks'] += 1
            logger.warning(f"Scaling decision blocked: {safety_reason}")
            return
        
        # Execute the scaling action
        try:
            if decision.action == ScalingAction.SCALE_UP:
                await self._scale_up(decision.target_instances)
                self.stats['scale_up_decisions'] += 1
            elif decision.action == ScalingAction.SCALE_DOWN:
                await self._scale_down(decision.target_instances)
                self.stats['scale_down_decisions'] += 1
            
            # Record the scaling event
            self.safety_controller.record_scaling_event(decision)
            
            logger.info(
                f"Scaling executed: {decision.action.value} from {self.current_instances} to {decision.target_instances}",
                reason=decision.reason,
                confidence=decision.confidence
            )
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
    
    async def _scale_up(self, target_instances: int):
        """Scale up to target number of instances."""
        
        # In a real implementation, this would:
        # - Start new application instances
        # - Update load balancer configuration
        # - Update service discovery
        
        logger.info(f"Scaling up from {self.current_instances} to {target_instances} instances")
        
        # Simulate scaling delay
        await asyncio.sleep(1)
        
        self.current_instances = target_instances
        self.target_instances = target_instances
    
    async def _scale_down(self, target_instances: int):
        """Scale down to target number of instances."""
        
        # In a real implementation, this would:
        # - Gracefully shutdown excess instances
        # - Update load balancer configuration
        # - Update service discovery
        
        logger.info(f"Scaling down from {self.current_instances} to {target_instances} instances")
        
        # Simulate scaling delay
        await asyncio.sleep(1)
        
        self.current_instances = target_instances
        self.target_instances = target_instances
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule by name."""
        self.scaling_rules = [rule for rule in self.scaling_rules if rule.name != rule_name]
        logger.info(f"Removed scaling rule: {rule_name}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        
        return {
            'running': self.running,
            'current_instances': self.current_instances,
            'target_instances': self.target_instances,
            'strategy': self.strategy.value,
            'active_rules': len([rule for rule in self.scaling_rules if rule.enabled]),
            'statistics': self.stats.copy(),
            'recent_metrics': {
                metric_type.value: self.metrics_collector.get_average_metric(metric_type)
                for metric_type in MetricType
            }
        }
    
    async def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        
        return {
            'auto_scaling_stats': self.stats.copy(),
            'safety_controller_stats': {
                'min_instances': self.safety_controller.min_instances,
                'max_instances': self.safety_controller.max_instances,
                'scaling_events': len(self.safety_controller.scaling_history)
            },
            'metrics_collection': {
                'collection_interval': self.metrics_collector.collection_interval,
                'metrics_history_size': {
                    metric_type.value: len(self.metrics_collector.metrics_history[metric_type])
                    for metric_type in MetricType
                }
            },
            'scaling_rules': [
                {
                    'name': rule.name,
                    'metric_type': rule.metric_type.value,
                    'scale_up_threshold': rule.scale_up_threshold,
                    'scale_down_threshold': rule.scale_down_threshold,
                    'enabled': rule.enabled,
                    'weight': rule.weight
                }
                for rule in self.scaling_rules
            ]
        }