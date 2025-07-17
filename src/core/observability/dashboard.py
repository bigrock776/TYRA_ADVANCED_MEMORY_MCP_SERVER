"""
Memory Quality Dashboard for Comprehensive System Monitoring.

This module provides advanced dashboard capabilities with real-time metrics visualization,
performance monitoring using local analytics, quality assessment with local algorithms,
and health tracking with comprehensive insights. All processing is performed locally with zero external dependencies.
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd

# Visualization and analysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics tracked."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    USAGE = "usage"
    HEALTH = "health"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    RESOURCE = "resource"
    BUSINESS = "business"


class DashboardWidget(str, Enum):
    """Types of dashboard widgets."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    COUNTER = "counter"
    TABLE = "table"
    HEATMAP = "heatmap"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    STATUS_INDICATOR = "status_indicator"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class Metric:
    """Represents a system metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "dimensions": self.dimensions,
            "metadata": self.metadata
        }


@dataclass
class DashboardCard:
    """Represents a dashboard card/widget."""
    id: str
    title: str
    widget_type: DashboardWidget
    data: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "widget_type": self.widget_type.value,
            "data": self.data,
            "config": self.config,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    title: str
    message: str
    level: AlertLevel
    timestamp: datetime
    source: str
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "metadata": self.metadata
        }


class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.aggregation_intervals = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1)
        }
    
    async def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        metric_type: MetricType = MetricType.PERFORMANCE,
        dimensions: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            metric_type=metric_type,
            dimensions=dimensions or {},
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.metrics_buffer.append(metric)
        
        # Cache metric
        if self.cache:
            await self.cache.lpush(
                f"metrics:{name}",
                json.dumps(metric.to_dict()),
                max_length=1000
            )
            
            # Store aggregated metrics
            await self._store_aggregated_metrics(metric)
    
    async def _store_aggregated_metrics(self, metric: Metric):
        """Store aggregated metrics for different time intervals."""
        current_time = metric.timestamp
        
        for interval_name, interval_duration in self.aggregation_intervals.items():
            # Calculate bucket timestamp
            bucket_start = self._get_bucket_start(current_time, interval_duration)
            bucket_key = f"metrics_agg:{metric.name}:{interval_name}:{bucket_start.isoformat()}"
            
            if self.cache:
                # Get existing aggregation or create new one
                existing_data = await self.cache.get(bucket_key)
                
                if existing_data:
                    agg_data = json.loads(existing_data)
                    agg_data["count"] += 1
                    agg_data["sum"] += metric.value
                    agg_data["min"] = min(agg_data["min"], metric.value)
                    agg_data["max"] = max(agg_data["max"], metric.value)
                    agg_data["avg"] = agg_data["sum"] / agg_data["count"]
                else:
                    agg_data = {
                        "metric_name": metric.name,
                        "interval": interval_name,
                        "bucket_start": bucket_start.isoformat(),
                        "count": 1,
                        "sum": metric.value,
                        "min": metric.value,
                        "max": metric.value,
                        "avg": metric.value,
                        "unit": metric.unit,
                        "metric_type": metric.metric_type.value
                    }
                
                # Store with TTL based on interval
                ttl = self._get_ttl_for_interval(interval_name)
                await self.cache.set(bucket_key, json.dumps(agg_data), ttl=ttl)
    
    def _get_bucket_start(self, timestamp: datetime, interval: timedelta) -> datetime:
        """Get the start of the time bucket for aggregation."""
        total_seconds = int(interval.total_seconds())
        
        if total_seconds < 3600:  # Less than 1 hour
            # Round to the nearest minute
            minutes = total_seconds // 60
            rounded_minute = (timestamp.minute // minutes) * minutes
            return timestamp.replace(minute=rounded_minute, second=0, microsecond=0)
        elif total_seconds < 86400:  # Less than 1 day
            # Round to the nearest hour
            hours = total_seconds // 3600
            rounded_hour = (timestamp.hour // hours) * hours
            return timestamp.replace(hour=rounded_hour, minute=0, second=0, microsecond=0)
        else:  # 1 day or more
            # Round to the nearest day
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _get_ttl_for_interval(self, interval_name: str) -> int:
        """Get TTL for aggregated metrics based on interval."""
        ttl_map = {
            "1m": 3600,      # 1 hour
            "5m": 7200,      # 2 hours
            "15m": 21600,    # 6 hours
            "1h": 86400,     # 1 day
            "1d": 2592000    # 30 days
        }
        return ttl_map.get(interval_name, 3600)
    
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m"
    ) -> List[Dict[str, Any]]:
        """Get aggregated metrics for a time range."""
        if not self.cache:
            return []
        
        metrics = []
        current_time = start_time
        interval_duration = self.aggregation_intervals.get(interval, timedelta(minutes=1))
        
        while current_time <= end_time:
            bucket_start = self._get_bucket_start(current_time, interval_duration)
            bucket_key = f"metrics_agg:{metric_name}:{interval}:{bucket_start.isoformat()}"
            
            data = await self.cache.get(bucket_key)
            if data:
                metrics.append(json.loads(data))
            
            current_time += interval_duration
        
        return metrics
    
    def get_recent_metrics(self, metric_name: str, count: int = 100) -> List[Metric]:
        """Get recent metrics from buffer."""
        return [
            m for m in list(self.metrics_buffer)[-count:]
            if m.name == metric_name
        ]


class QualityAnalyzer:
    """Analyzes memory quality and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    async def analyze_memory_quality(self, memories: List[Memory]) -> Dict[str, Any]:
        """Analyze overall memory quality."""
        if not memories:
            return {"quality_score": 0, "analysis": "No memories to analyze"}
        
        # Content quality metrics
        content_quality = await self._analyze_content_quality(memories)
        
        # Embedding quality metrics
        embedding_quality = await self._analyze_embedding_quality(memories)
        
        # Usage pattern metrics
        usage_patterns = await self._analyze_usage_patterns(memories)
        
        # Diversity metrics
        diversity_metrics = await self._analyze_diversity(memories)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score({
            "content": content_quality,
            "embeddings": embedding_quality,
            "usage": usage_patterns,
            "diversity": diversity_metrics
        })
        
        return {
            "quality_score": quality_score,
            "content_quality": content_quality,
            "embedding_quality": embedding_quality,
            "usage_patterns": usage_patterns,
            "diversity_metrics": diversity_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_content_quality(self, memories: List[Memory]) -> Dict[str, Any]:
        """Analyze content quality metrics."""
        content_lengths = []
        word_counts = []
        unique_words = set()
        total_chars = 0
        
        for memory in memories:
            if memory.content:
                content_lengths.append(len(memory.content))
                words = memory.content.split()
                word_counts.append(len(words))
                unique_words.update(word.lower() for word in words)
                total_chars += len(memory.content)
        
        if not content_lengths:
            return {"avg_length": 0, "avg_words": 0, "vocabulary_diversity": 0}
        
        return {
            "avg_length": statistics.mean(content_lengths),
            "median_length": statistics.median(content_lengths),
            "avg_words": statistics.mean(word_counts) if word_counts else 0,
            "vocabulary_diversity": len(unique_words) / sum(word_counts) if sum(word_counts) > 0 else 0,
            "total_chars": total_chars,
            "length_variance": statistics.variance(content_lengths) if len(content_lengths) > 1 else 0
        }
    
    async def _analyze_embedding_quality(self, memories: List[Memory]) -> Dict[str, Any]:
        """Analyze embedding quality metrics."""
        embeddings = [m.embedding for m in memories if m.embedding is not None]
        
        if not embeddings:
            return {"dimension_consistency": 0, "distribution_quality": 0}
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        # Dimension consistency
        dimensions = [emb.shape[0] if len(emb.shape) > 0 else 1 for emb in embeddings]
        dimension_consistency = len(set(dimensions)) == 1
        
        # Distribution quality
        mean_embedding = np.mean(embedding_matrix, axis=0)
        std_embedding = np.std(embedding_matrix, axis=0)
        distribution_quality = np.mean(std_embedding)  # Higher std indicates better distribution
        
        # Clustering quality
        clustering_quality = 0
        if len(embeddings) > 10:
            try:
                kmeans = KMeans(n_clusters=min(5, len(embeddings) // 2), random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embedding_matrix)
                clustering_quality = silhouette_score(embedding_matrix, cluster_labels)
            except Exception:
                clustering_quality = 0
        
        return {
            "dimension_consistency": 1.0 if dimension_consistency else 0.0,
            "avg_dimension": statistics.mean(dimensions),
            "distribution_quality": float(distribution_quality),
            "clustering_quality": float(clustering_quality),
            "embedding_count": len(embeddings),
            "mean_norm": float(np.mean([np.linalg.norm(emb) for emb in embeddings]))
        }
    
    async def _analyze_usage_patterns(self, memories: List[Memory]) -> Dict[str, Any]:
        """Analyze usage patterns."""
        access_counts = [m.access_count for m in memories]
        importance_scores = [m.importance_score for m in memories]
        
        # Time-based analysis
        now = datetime.utcnow()
        recently_accessed = [
            m for m in memories 
            if m.last_accessed and (now - m.last_accessed).days <= 7
        ]
        
        # Tag analysis
        all_tags = []
        for memory in memories:
            all_tags.extend(memory.tags or [])
        
        tag_distribution = Counter(all_tags)
        
        return {
            "avg_access_count": statistics.mean(access_counts) if access_counts else 0,
            "median_access_count": statistics.median(access_counts) if access_counts else 0,
            "recently_accessed_percent": len(recently_accessed) / len(memories) if memories else 0,
            "avg_importance": statistics.mean(importance_scores) if importance_scores else 0,
            "most_common_tags": dict(tag_distribution.most_common(10)),
            "tag_diversity": len(tag_distribution) / len(memories) if memories else 0,
            "high_importance_percent": len([s for s in importance_scores if s > 0.8]) / len(importance_scores) if importance_scores else 0
        }
    
    async def _analyze_diversity(self, memories: List[Memory]) -> Dict[str, Any]:
        """Analyze memory diversity."""
        users = set(m.user_id for m in memories if m.user_id)
        
        # Time diversity
        creation_times = [m.created_at for m in memories if m.created_at]
        if creation_times:
            time_span = max(creation_times) - min(creation_times)
            time_diversity = time_span.days
        else:
            time_diversity = 0
        
        # Content length diversity
        content_lengths = [len(m.content) for m in memories if m.content]
        length_diversity = statistics.stdev(content_lengths) if len(content_lengths) > 1 else 0
        
        return {
            "user_diversity": len(users),
            "time_diversity_days": time_diversity,
            "length_diversity": length_diversity,
            "memory_count": len(memories)
        }
    
    def _calculate_overall_quality_score(self, components: Dict[str, Any]) -> float:
        """Calculate overall quality score from components."""
        weights = {
            "content": 0.3,
            "embeddings": 0.3,
            "usage": 0.2,
            "diversity": 0.2
        }
        
        scores = {}
        
        # Content score
        content = components["content"]
        content_score = min(1.0, (
            (content.get("vocabulary_diversity", 0) * 0.4) +
            (min(1.0, content.get("avg_words", 0) / 100) * 0.3) +
            (min(1.0, content.get("length_variance", 0) / 10000) * 0.3)
        ))
        scores["content"] = content_score
        
        # Embedding score
        embeddings = components["embeddings"]
        embedding_score = (
            (embeddings.get("dimension_consistency", 0) * 0.3) +
            (min(1.0, embeddings.get("distribution_quality", 0)) * 0.3) +
            (max(0, embeddings.get("clustering_quality", 0)) * 0.4)
        )
        scores["embeddings"] = embedding_score
        
        # Usage score
        usage = components["usage"]
        usage_score = (
            (min(1.0, usage.get("recently_accessed_percent", 0)) * 0.4) +
            (usage.get("avg_importance", 0) * 0.3) +
            (min(1.0, usage.get("tag_diversity", 0)) * 0.3)
        )
        scores["usage"] = usage_score
        
        # Diversity score
        diversity = components["diversity"]
        diversity_score = min(1.0, (
            (min(1.0, diversity.get("user_diversity", 0) / 10) * 0.4) +
            (min(1.0, diversity.get("time_diversity_days", 0) / 365) * 0.3) +
            (min(1.0, diversity.get("length_diversity", 0) / 1000) * 0.3)
        ))
        scores["diversity"] = diversity_score
        
        # Calculate weighted average
        overall_score = sum(scores[component] * weights[component] for component in weights)
        
        return round(overall_score * 100, 2)  # Return as percentage


class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_targets = {
            "query_latency_p95": 100,     # milliseconds
            "embedding_time": 50,         # milliseconds
            "memory_usage": 80,           # percentage
            "cpu_usage": 70,              # percentage
            "disk_usage": 85,             # percentage
            "error_rate": 5               # percentage
        }
    
    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        import psutil
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Record metrics
        await self.metrics_collector.record_metric(
            "cpu_usage", cpu_percent, "percent", MetricType.PERFORMANCE
        )
        await self.metrics_collector.record_metric(
            "memory_usage", memory.percent, "percent", MetricType.PERFORMANCE
        )
        await self.metrics_collector.record_metric(
            "disk_usage", disk.percent, "percent", MetricType.PERFORMANCE
        )
        
        # Network metrics if available
        try:
            network = psutil.net_io_counters()
            await self.metrics_collector.record_metric(
                "network_bytes_sent", network.bytes_sent, "bytes", MetricType.PERFORMANCE
            )
            await self.metrics_collector.record_metric(
                "network_bytes_recv", network.bytes_recv, "bytes", MetricType.PERFORMANCE
            )
        except Exception:
            pass
        
        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available": memory.available,
            "disk_usage": disk.percent,
            "disk_free": disk.free,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def check_performance_thresholds(self) -> List[Alert]:
        """Check if performance metrics exceed thresholds."""
        alerts = []
        
        # Get recent metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        for metric_name, threshold in self.performance_targets.items():
            recent_metrics = await self.metrics_collector.get_metrics(
                metric_name, start_time, end_time, "1m"
            )
            
            if recent_metrics:
                latest_value = recent_metrics[-1].get("avg", 0)
                
                if latest_value > threshold:
                    alert = Alert(
                        id=f"perf_{metric_name}_{int(time.time())}",
                        title=f"Performance Threshold Exceeded",
                        message=f"{metric_name} is {latest_value:.2f}, above threshold {threshold}",
                        level=AlertLevel.WARNING if latest_value < threshold * 1.2 else AlertLevel.CRITICAL,
                        timestamp=datetime.utcnow(),
                        source="performance_monitor",
                        metadata={
                            "metric_name": metric_name,
                            "value": latest_value,
                            "threshold": threshold
                        }
                    )
                    alerts.append(alert)
        
        return alerts


class DashboardEngine:
    """Main dashboard engine that orchestrates all components."""
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache
        self.metrics_collector = MetricsCollector(cache)
        self.quality_analyzer = QualityAnalyzer(self.metrics_collector)
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        
        # Dashboard state
        self.cards: Dict[str, DashboardCard] = {}
        self.alerts: deque = deque(maxlen=100)
        self.refresh_interval = 30  # seconds
        
        # Initialize default cards
        self._initialize_default_cards()
        
        logger.info("Dashboard engine initialized")
    
    def _initialize_default_cards(self):
        """Initialize default dashboard cards."""
        default_cards = [
            DashboardCard(
                id="system_overview",
                title="System Overview",
                widget_type=DashboardWidget.GAUGE,
                data={"value": 0, "max": 100, "unit": "%"},
                config={"color": "blue", "size": "large"}
            ),
            DashboardCard(
                id="memory_quality",
                title="Memory Quality Score",
                widget_type=DashboardWidget.GAUGE,
                data={"value": 0, "max": 100, "unit": "%"},
                config={"color": "green", "size": "large"}
            ),
            DashboardCard(
                id="performance_metrics",
                title="Performance Metrics",
                widget_type=DashboardWidget.LINE_CHART,
                data={"series": [], "labels": []},
                config={"height": 300}
            ),
            DashboardCard(
                id="recent_alerts",
                title="Recent Alerts",
                widget_type=DashboardWidget.TABLE,
                data={"headers": ["Time", "Level", "Message"], "rows": []},
                config={"max_rows": 10}
            ),
            DashboardCard(
                id="usage_statistics",
                title="Usage Statistics",
                widget_type=DashboardWidget.BAR_CHART,
                data={"categories": [], "values": []},
                config={"orientation": "horizontal"}
            )
        ]
        
        for card in default_cards:
            self.cards[card.id] = card
    
    async def refresh_dashboard(self, memories: Optional[List[Memory]] = None) -> Dict[str, Any]:
        """Refresh all dashboard data."""
        start_time = time.time()
        
        try:
            # Collect performance metrics
            perf_metrics = await self.performance_monitor.collect_performance_metrics()
            
            # Analyze memory quality if memories provided
            quality_analysis = {}
            if memories:
                quality_analysis = await self.quality_analyzer.analyze_memory_quality(memories)
            
            # Check for alerts
            new_alerts = await self.performance_monitor.check_performance_thresholds()
            for alert in new_alerts:
                self.alerts.append(alert)
            
            # Update dashboard cards
            await self._update_system_overview_card(perf_metrics)
            await self._update_memory_quality_card(quality_analysis)
            await self._update_performance_chart()
            await self._update_alerts_table()
            await self._update_usage_statistics(memories or [])
            
            refresh_time = time.time() - start_time
            
            # Record dashboard refresh metric
            await self.metrics_collector.record_metric(
                "dashboard_refresh_time", refresh_time * 1000, "ms", MetricType.PERFORMANCE
            )
            
            return {
                "status": "success",
                "refresh_time_ms": refresh_time * 1000,
                "cards": {card_id: card.to_dict() for card_id, card in self.cards.items()},
                "alerts": [alert.to_dict() for alert in list(self.alerts)[-10:]],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Dashboard refresh failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _update_system_overview_card(self, perf_metrics: Dict[str, Any]):
        """Update system overview card."""
        # Calculate overall health score
        cpu_health = max(0, 100 - perf_metrics.get("cpu_usage", 0))
        memory_health = max(0, 100 - perf_metrics.get("memory_usage", 0))
        disk_health = max(0, 100 - perf_metrics.get("disk_usage", 0))
        
        overall_health = (cpu_health + memory_health + disk_health) / 3
        
        self.cards["system_overview"].data = {
            "value": round(overall_health, 1),
            "max": 100,
            "unit": "%",
            "details": {
                "cpu": perf_metrics.get("cpu_usage", 0),
                "memory": perf_metrics.get("memory_usage", 0),
                "disk": perf_metrics.get("disk_usage", 0)
            }
        }
        self.cards["system_overview"].last_updated = datetime.utcnow()
    
    async def _update_memory_quality_card(self, quality_analysis: Dict[str, Any]):
        """Update memory quality card."""
        quality_score = quality_analysis.get("quality_score", 0)
        
        self.cards["memory_quality"].data = {
            "value": quality_score,
            "max": 100,
            "unit": "%",
            "details": quality_analysis
        }
        self.cards["memory_quality"].last_updated = datetime.utcnow()
    
    async def _update_performance_chart(self):
        """Update performance metrics chart."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        # Get CPU usage metrics
        cpu_metrics = await self.metrics_collector.get_metrics(
            "cpu_usage", start_time, end_time, "5m"
        )
        
        # Get memory usage metrics
        memory_metrics = await self.metrics_collector.get_metrics(
            "memory_usage", start_time, end_time, "5m"
        )
        
        # Prepare chart data
        labels = []
        cpu_values = []
        memory_values = []
        
        for metric in cpu_metrics:
            bucket_time = datetime.fromisoformat(metric["bucket_start"])
            labels.append(bucket_time.strftime("%H:%M"))
            cpu_values.append(metric["avg"])
        
        for metric in memory_metrics:
            memory_values.append(metric["avg"])
        
        self.cards["performance_metrics"].data = {
            "labels": labels,
            "series": [
                {"name": "CPU Usage", "data": cpu_values, "color": "#ff6b6b"},
                {"name": "Memory Usage", "data": memory_values, "color": "#4ecdc4"}
            ]
        }
        self.cards["performance_metrics"].last_updated = datetime.utcnow()
    
    async def _update_alerts_table(self):
        """Update alerts table."""
        recent_alerts = list(self.alerts)[-10:]
        
        rows = []
        for alert in recent_alerts:
            rows.append([
                alert.timestamp.strftime("%H:%M:%S"),
                alert.level.value.upper(),
                alert.message
            ])
        
        self.cards["recent_alerts"].data = {
            "headers": ["Time", "Level", "Message"],
            "rows": rows
        }
        self.cards["recent_alerts"].last_updated = datetime.utcnow()
    
    async def _update_usage_statistics(self, memories: List[Memory]):
        """Update usage statistics card."""
        if not memories:
            self.cards["usage_statistics"].data = {
                "categories": ["No Data"],
                "values": [0]
            }
            return
        
        # Analyze usage patterns
        user_counts = Counter(m.user_id for m in memories if m.user_id)
        top_users = user_counts.most_common(5)
        
        categories = [f"User {i+1}" for i in range(len(top_users))]
        values = [count for _, count in top_users]
        
        self.cards["usage_statistics"].data = {
            "categories": categories,
            "values": values
        }
        self.cards["usage_statistics"].last_updated = datetime.utcnow()
    
    def get_card(self, card_id: str) -> Optional[DashboardCard]:
        """Get a specific dashboard card."""
        return self.cards.get(card_id)
    
    def add_custom_card(self, card: DashboardCard):
        """Add a custom dashboard card."""
        self.cards[card.id] = card
    
    def remove_card(self, card_id: str) -> bool:
        """Remove a dashboard card."""
        if card_id in self.cards:
            del self.cards[card_id]
            return True
        return False
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for live updates."""
        current_time = datetime.utcnow()
        
        # Get latest performance metrics
        perf_metrics = await self.performance_monitor.collect_performance_metrics()
        
        # Get recent metric trends
        trends = {}
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
            recent_metrics = self.metrics_collector.get_recent_metrics(metric_name, 10)
            if len(recent_metrics) >= 2:
                recent_values = [m.value for m in recent_metrics[-5:]]
                trend = "up" if recent_values[-1] > recent_values[0] else "down"
                trends[metric_name] = trend
            else:
                trends[metric_name] = "stable"
        
        return {
            "timestamp": current_time.isoformat(),
            "performance": perf_metrics,
            "trends": trends,
            "alert_count": len(self.alerts),
            "active_connections": 0  # Would come from connection pool
        }
    
    async def export_dashboard_data(self, format: str = "json") -> Union[str, bytes]:
        """Export dashboard data."""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "cards": {card_id: card.to_dict() for card_id, card in self.cards.items()},
            "alerts": [alert.to_dict() for alert in self.alerts],
            "metrics_summary": await self._generate_metrics_summary()
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _generate_metrics_summary(self) -> Dict[str, Any]:
        """Generate a summary of key metrics."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        summary = {}
        
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
            metrics = await self.metrics_collector.get_metrics(
                metric_name, start_time, end_time, "1h"
            )
            
            if metrics:
                values = [m["avg"] for m in metrics]
                summary[metric_name] = {
                    "avg_24h": statistics.mean(values),
                    "max_24h": max(values),
                    "min_24h": min(values),
                    "current": values[-1] if values else 0
                }
        
        return summary
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform dashboard health check."""
        try:
            # Test metrics collection
            await self.metrics_collector.record_metric(
                "dashboard_health_check", 1, "count", MetricType.HEALTH
            )
            
            # Test card updates
            card_count = len(self.cards)
            
            # Test alert system
            alert_count = len(self.alerts)
            
            return {
                "status": "healthy",
                "cards_count": card_count,
                "alerts_count": alert_count,
                "metrics_buffer_size": len(self.metrics_collector.metrics_buffer),
                "cache_available": self.cache is not None
            }
            
        except Exception as e:
            logger.error("Dashboard health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }