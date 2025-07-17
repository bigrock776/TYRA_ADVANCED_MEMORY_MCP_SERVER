"""
Local Usage Pattern Analysis Dashboard.

This module provides comprehensive usage analytics and pattern analysis
using local analytics with pandas, plotly, trend analysis, custom dashboard
creation with templates, and local export capabilities for sharing.
"""

import asyncio
import json
import math
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ...core.memory.manager import MemoryManager
from ...core.graph.neo4j_client import Neo4jClient
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class AnalysisTimeframe(str, Enum):
    """Time frame options for usage analysis."""
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    ALL_TIME = "all_time"
    CUSTOM = "custom"


class AnalysisType(str, Enum):
    """Types of usage analysis."""
    TEMPORAL_PATTERNS = "temporal_patterns"      # When users are most active
    ACCESS_PATTERNS = "access_patterns"          # What content is accessed most
    SEARCH_PATTERNS = "search_patterns"          # Search query analysis
    SESSION_PATTERNS = "session_patterns"        # Session duration and behavior
    CONTENT_PATTERNS = "content_patterns"        # Content creation and modification
    NETWORK_PATTERNS = "network_patterns"        # Graph connectivity patterns
    PERFORMANCE_PATTERNS = "performance_patterns" # System performance over time
    USER_JOURNEY = "user_journey"                # User flow analysis


class MetricType(str, Enum):
    """Types of metrics to track."""
    COUNT = "count"                 # Simple counts
    FREQUENCY = "frequency"         # Frequency over time
    DURATION = "duration"           # Time-based metrics
    EFFICIENCY = "efficiency"       # Performance ratios
    ENGAGEMENT = "engagement"       # User engagement scores
    GROWTH = "growth"              # Growth rates
    RETENTION = "retention"         # User retention metrics
    SATISFACTION = "satisfaction"   # Quality indicators


class VisualizationStyle(str, Enum):
    """Visualization style options."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    SANKEY_DIAGRAM = "sankey_diagram"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"


@dataclass
class AnalysisConfig:
    """Configuration for usage pattern analysis."""
    timeframe: AnalysisTimeframe = AnalysisTimeframe.LAST_WEEK
    analysis_types: List[AnalysisType] = field(default_factory=lambda: [AnalysisType.TEMPORAL_PATTERNS])
    metric_types: List[MetricType] = field(default_factory=lambda: [MetricType.COUNT])
    user_filters: Optional[List[str]] = None
    content_filters: Optional[List[str]] = None
    tag_filters: Optional[List[str]] = None
    min_activity_threshold: int = 1
    include_anonymous: bool = True
    granularity: str = "hour"  # hour, day, week, month
    custom_start_date: Optional[datetime] = None
    custom_end_date: Optional[datetime] = None


class UsageMetric(BaseModel):
    """Individual usage metric."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metric_id: str = Field(..., description="Unique metric identifier")
    metric_name: str = Field(..., description="Human-readable metric name")
    metric_type: MetricType = Field(..., description="Type of metric")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(..., description="When metric was recorded")
    
    # Context
    user_id: Optional[str] = Field(default=None, description="Associated user")
    session_id: Optional[str] = Field(default=None, description="Associated session")
    memory_id: Optional[str] = Field(default=None, description="Associated memory")
    tags: List[str] = Field(default_factory=list, description="Metric tags")
    
    # Additional metadata
    dimensions: Dict[str, Any] = Field(default_factory=dict, description="Additional dimensions")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in metric")


class AnalysisInsight(BaseModel):
    """Insight derived from usage analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    insight_id: str = Field(..., description="Unique insight identifier")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    insight_type: str = Field(..., description="Type of insight")
    
    # Significance
    significance_score: float = Field(..., ge=0.0, le=1.0, description="How significant this insight is")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Statistical confidence")
    impact_level: str = Field(..., description="Expected impact level")
    
    # Supporting data
    supporting_metrics: List[str] = Field(..., description="Metrics that support this insight")
    visualizations: List[str] = Field(default_factory=list, description="Associated visualizations")
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")
    
    # Temporal context
    timeframe: str = Field(..., description="Time period for this insight")
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When insight was discovered")


class DashboardTemplate(BaseModel):
    """Template for custom dashboard creation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: str = Field(..., description="Template category")
    
    # Layout configuration
    layout_config: Dict[str, Any] = Field(..., description="Dashboard layout configuration")
    widget_configs: List[Dict[str, Any]] = Field(..., description="Widget configurations")
    default_timeframe: AnalysisTimeframe = Field(default=AnalysisTimeframe.LAST_WEEK)
    
    # Sharing and permissions
    is_public: bool = Field(default=False, description="Whether template is publicly available")
    created_by: str = Field(..., description="Creator user ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation time")
    tags: List[str] = Field(default_factory=list, description="Template tags")


class LocalUsageAnalyzer:
    """Local Usage Pattern Analysis Dashboard System."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        graph_client: Neo4jClient,
        cache: Optional[RedisCache] = None
    ):
        self.memory_manager = memory_manager
        self.graph_client = graph_client
        self.cache = cache
        
        # Data storage
        self.usage_metrics: List[UsageMetric] = []
        self.analysis_cache = {}
        self.insights_cache = {}
        self.templates: Dict[str, DashboardTemplate] = {}
        
        # Analysis state
        self.current_analysis_config = AnalysisConfig()
        self.last_analysis_time = datetime.now(timezone.utc)
        
        # Dash app
        self.app = None
        self.is_running = False
        
        logger.info("LocalUsageAnalyzer initialized")
    
    async def collect_usage_metrics(
        self,
        user_id: str,
        config: Optional[AnalysisConfig] = None
    ) -> List[UsageMetric]:
        """Collect usage metrics for analysis."""
        try:
            if config is None:
                config = self.current_analysis_config
            
            # Define time range
            end_time = datetime.now(timezone.utc)
            start_time = self._calculate_start_time(config.timeframe, end_time)
            
            metrics = []
            
            # Collect temporal patterns
            if AnalysisType.TEMPORAL_PATTERNS in config.analysis_types:
                temporal_metrics = await self._collect_temporal_metrics(user_id, start_time, end_time, config)
                metrics.extend(temporal_metrics)
            
            # Collect access patterns
            if AnalysisType.ACCESS_PATTERNS in config.analysis_types:
                access_metrics = await self._collect_access_metrics(user_id, start_time, end_time, config)
                metrics.extend(access_metrics)
            
            # Collect search patterns
            if AnalysisType.SEARCH_PATTERNS in config.analysis_types:
                search_metrics = await self._collect_search_metrics(user_id, start_time, end_time, config)
                metrics.extend(search_metrics)
            
            # Collect session patterns
            if AnalysisType.SESSION_PATTERNS in config.analysis_types:
                session_metrics = await self._collect_session_metrics(user_id, start_time, end_time, config)
                metrics.extend(session_metrics)
            
            # Collect content patterns
            if AnalysisType.CONTENT_PATTERNS in config.analysis_types:
                content_metrics = await self._collect_content_metrics(user_id, start_time, end_time, config)
                metrics.extend(content_metrics)
            
            # Collect network patterns
            if AnalysisType.NETWORK_PATTERNS in config.analysis_types:
                network_metrics = await self._collect_network_metrics(user_id, start_time, end_time, config)
                metrics.extend(network_metrics)
            
            # Collect performance patterns
            if AnalysisType.PERFORMANCE_PATTERNS in config.analysis_types:
                performance_metrics = await self._collect_performance_metrics(user_id, start_time, end_time, config)
                metrics.extend(performance_metrics)
            
            # Cache metrics
            if self.cache:
                cache_key = f"usage_metrics:{user_id}:{hash(str(config))}"
                await self.cache.set(
                    cache_key,
                    json.dumps([m.model_dump() for m in metrics], default=str),
                    expire_minutes=30
                )
            
            logger.info(
                "Usage metrics collected",
                user_id=user_id,
                metric_count=len(metrics),
                timeframe=config.timeframe
            )
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to collect usage metrics", error=str(e), user_id=user_id)
            raise
    
    async def _collect_temporal_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        config: AnalysisConfig
    ) -> List[UsageMetric]:
        """Collect temporal usage patterns."""
        metrics = []
        
        try:
            # Get memories within timeframe
            memories = await self.memory_manager.get_memories_by_timeframe(
                user_id, start_time, end_time
            )
            
            # Group by time periods
            time_groups = self._group_by_time_period(memories, config.granularity)
            
            for period, period_memories in time_groups.items():
                # Activity count metric
                metrics.append(UsageMetric(
                    metric_id=f"temporal_activity_{period}",
                    metric_name=f"Activity Count - {period}",
                    metric_type=MetricType.COUNT,
                    value=len(period_memories),
                    unit="actions",
                    timestamp=period,
                    user_id=user_id,
                    dimensions={"period": period.isoformat(), "granularity": config.granularity}
                ))
                
                # Content creation rate
                creation_count = sum(1 for m in period_memories if m.created_at >= start_time)
                metrics.append(UsageMetric(
                    metric_id=f"temporal_creation_{period}",
                    metric_name=f"Content Creation - {period}",
                    metric_type=MetricType.FREQUENCY,
                    value=creation_count,
                    unit="memories_created",
                    timestamp=period,
                    user_id=user_id,
                    dimensions={"period": period.isoformat(), "granularity": config.granularity}
                ))
            
            # Peak activity hours
            hourly_activity = defaultdict(int)
            for memory in memories:
                hour = memory.created_at.hour
                hourly_activity[hour] += 1
            
            if hourly_activity:
                peak_hour = max(hourly_activity.items(), key=lambda x: x[1])
                metrics.append(UsageMetric(
                    metric_id=f"peak_activity_hour_{user_id}",
                    metric_name="Peak Activity Hour",
                    metric_type=MetricType.COUNT,
                    value=float(peak_hour[0]),
                    unit="hour_of_day",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={"activity_count": peak_hour[1]}
                ))
            
        except Exception as e:
            logger.error("Failed to collect temporal metrics", error=str(e))
        
        return metrics
    
    async def _collect_access_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        config: AnalysisConfig
    ) -> List[UsageMetric]:
        """Collect access pattern metrics."""
        metrics = []
        
        try:
            # Get memories with access data
            memories = await self.memory_manager.get_all_memories(user_id)
            
            # Most accessed memories
            access_counts = [(m.id, getattr(m, 'access_count', 0)) for m in memories]
            access_counts.sort(key=lambda x: x[1], reverse=True)
            
            if access_counts:
                top_memory = access_counts[0]
                metrics.append(UsageMetric(
                    metric_id=f"most_accessed_memory_{user_id}",
                    metric_name="Most Accessed Memory",
                    metric_type=MetricType.COUNT,
                    value=float(top_memory[1]),
                    unit="access_count",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    memory_id=top_memory[0],
                    dimensions={"memory_rank": 1}
                ))
            
            # Access frequency distribution
            access_values = [count for _, count in access_counts if count > 0]
            if access_values:
                metrics.append(UsageMetric(
                    metric_id=f"avg_access_frequency_{user_id}",
                    metric_name="Average Access Frequency",
                    metric_type=MetricType.FREQUENCY,
                    value=np.mean(access_values),
                    unit="accesses_per_memory",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={
                        "std_dev": np.std(access_values),
                        "median": np.median(access_values),
                        "total_memories": len(memories)
                    }
                ))
            
            # Content type access patterns
            content_types = defaultdict(int)
            for memory in memories:
                tags = getattr(memory, 'tags', [])
                if tags:
                    content_types[tags[0]] += getattr(memory, 'access_count', 0)
                else:
                    content_types['untagged'] += getattr(memory, 'access_count', 0)
            
            for content_type, total_access in content_types.items():
                if total_access > 0:
                    metrics.append(UsageMetric(
                        metric_id=f"content_type_access_{content_type}_{user_id}",
                        metric_name=f"Access Count - {content_type}",
                        metric_type=MetricType.COUNT,
                        value=float(total_access),
                        unit="total_accesses",
                        timestamp=datetime.now(timezone.utc),
                        user_id=user_id,
                        tags=[content_type],
                        dimensions={"content_type": content_type}
                    ))
            
        except Exception as e:
            logger.error("Failed to collect access metrics", error=str(e))
        
        return metrics
    
    async def _collect_search_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        config: AnalysisConfig
    ) -> List[UsageMetric]:
        """Collect search pattern metrics."""
        metrics = []
        
        try:
            # In a real implementation, you would have search logs
            # For now, we'll simulate based on memory content analysis
            
            memories = await self.memory_manager.get_all_memories(user_id)
            
            # Simulate search query analysis based on memory content
            search_terms = []
            for memory in memories:
                words = memory.content.lower().split()
                # Extract potential search terms (words > 3 chars, not too common)
                common_words = {'the', 'and', 'that', 'this', 'with', 'for', 'are', 'was', 'but', 'not'}
                terms = [w for w in words if len(w) > 3 and w not in common_words]
                search_terms.extend(terms[:3])  # Take first 3 terms per memory
            
            # Most common search terms
            term_counts = Counter(search_terms)
            top_terms = term_counts.most_common(10)
            
            for i, (term, count) in enumerate(top_terms):
                metrics.append(UsageMetric(
                    metric_id=f"search_term_{term}_{user_id}",
                    metric_name=f"Search Term Frequency - {term}",
                    metric_type=MetricType.FREQUENCY,
                    value=float(count),
                    unit="occurrences",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={"search_term": term, "rank": i + 1}
                ))
            
            # Search diversity metric
            unique_terms = len(set(search_terms))
            total_terms = len(search_terms)
            diversity = unique_terms / max(total_terms, 1)
            
            metrics.append(UsageMetric(
                metric_id=f"search_diversity_{user_id}",
                metric_name="Search Diversity",
                metric_type=MetricType.EFFICIENCY,
                value=diversity,
                unit="diversity_ratio",
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                dimensions={
                    "unique_terms": unique_terms,
                    "total_terms": total_terms
                }
            ))
            
        except Exception as e:
            logger.error("Failed to collect search metrics", error=str(e))
        
        return metrics
    
    async def _collect_session_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        config: AnalysisConfig
    ) -> List[UsageMetric]:
        """Collect session pattern metrics."""
        metrics = []
        
        try:
            # Simulate session analysis based on memory creation patterns
            memories = await self.memory_manager.get_memories_by_timeframe(
                user_id, start_time, end_time
            )
            
            if not memories:
                return metrics
            
            # Group memories into sessions (memories within 30 minutes of each other)
            sessions = []
            current_session = [memories[0]]
            
            for i in range(1, len(memories)):
                time_diff = (memories[i].created_at - memories[i-1].created_at).total_seconds()
                if time_diff <= 1800:  # 30 minutes
                    current_session.append(memories[i])
                else:
                    sessions.append(current_session)
                    current_session = [memories[i]]
            
            if current_session:
                sessions.append(current_session)
            
            # Session metrics
            session_durations = []
            session_activities = []
            
            for session in sessions:
                if len(session) > 1:
                    duration = (session[-1].created_at - session[0].created_at).total_seconds() / 60
                    session_durations.append(duration)
                    session_activities.append(len(session))
            
            if session_durations:
                # Average session duration
                metrics.append(UsageMetric(
                    metric_id=f"avg_session_duration_{user_id}",
                    metric_name="Average Session Duration",
                    metric_type=MetricType.DURATION,
                    value=np.mean(session_durations),
                    unit="minutes",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={
                        "total_sessions": len(sessions),
                        "std_dev": np.std(session_durations),
                        "max_duration": np.max(session_durations)
                    }
                ))
                
                # Session productivity (activities per minute)
                productivities = [activities / max(duration, 1) for activities, duration in zip(session_activities, session_durations)]
                metrics.append(UsageMetric(
                    metric_id=f"session_productivity_{user_id}",
                    metric_name="Session Productivity",
                    metric_type=MetricType.EFFICIENCY,
                    value=np.mean(productivities),
                    unit="activities_per_minute",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={
                        "avg_activities_per_session": np.mean(session_activities)
                    }
                ))
            
        except Exception as e:
            logger.error("Failed to collect session metrics", error=str(e))
        
        return metrics
    
    async def _collect_content_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        config: AnalysisConfig
    ) -> List[UsageMetric]:
        """Collect content pattern metrics."""
        metrics = []
        
        try:
            memories = await self.memory_manager.get_memories_by_timeframe(
                user_id, start_time, end_time
            )
            
            if not memories:
                return metrics
            
            # Content length analysis
            content_lengths = [len(memory.content) for memory in memories]
            
            metrics.append(UsageMetric(
                metric_id=f"avg_content_length_{user_id}",
                metric_name="Average Content Length",
                metric_type=MetricType.COUNT,
                value=np.mean(content_lengths),
                unit="characters",
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                dimensions={
                    "std_dev": np.std(content_lengths),
                    "median": np.median(content_lengths),
                    "total_memories": len(memories)
                }
            ))
            
            # Content complexity (unique words per memory)
            complexity_scores = []
            for memory in memories:
                words = set(memory.content.lower().split())
                complexity = len(words) / max(len(memory.content.split()), 1)
                complexity_scores.append(complexity)
            
            if complexity_scores:
                metrics.append(UsageMetric(
                    metric_id=f"content_complexity_{user_id}",
                    metric_name="Content Complexity",
                    metric_type=MetricType.EFFICIENCY,
                    value=np.mean(complexity_scores),
                    unit="unique_word_ratio",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={
                        "avg_unique_words": np.mean([len(set(m.content.lower().split())) for m in memories])
                    }
                ))
            
            # Tag usage patterns
            all_tags = []
            for memory in memories:
                tags = getattr(memory, 'tags', [])
                all_tags.extend(tags)
            
            if all_tags:
                tag_counts = Counter(all_tags)
                tag_diversity = len(set(all_tags)) / len(all_tags)
                
                metrics.append(UsageMetric(
                    metric_id=f"tag_diversity_{user_id}",
                    metric_name="Tag Diversity",
                    metric_type=MetricType.EFFICIENCY,
                    value=tag_diversity,
                    unit="diversity_ratio",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={
                        "unique_tags": len(set(all_tags)),
                        "total_tag_uses": len(all_tags),
                        "most_common_tag": tag_counts.most_common(1)[0][0] if tag_counts else None
                    }
                ))
            
        except Exception as e:
            logger.error("Failed to collect content metrics", error=str(e))
        
        return metrics
    
    async def _collect_network_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        config: AnalysisConfig
    ) -> List[UsageMetric]:
        """Collect network pattern metrics."""
        metrics = []
        
        try:
            # Get relationships from graph
            relationships = await self.graph_client.get_relationships_by_timeframe(
                user_id, start_time, end_time
            )
            
            if not relationships:
                return metrics
            
            # Build network graph
            G = nx.Graph()
            for rel in relationships:
                G.add_edge(rel.source_id, rel.target_id, weight=rel.strength)
            
            if len(G.nodes()) > 0:
                # Network density
                density = nx.density(G)
                metrics.append(UsageMetric(
                    metric_id=f"network_density_{user_id}",
                    metric_name="Network Density",
                    metric_type=MetricType.EFFICIENCY,
                    value=density,
                    unit="density_ratio",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={
                        "nodes": len(G.nodes()),
                        "edges": len(G.edges())
                    }
                ))
                
                # Average clustering coefficient
                try:
                    avg_clustering = nx.average_clustering(G)
                    metrics.append(UsageMetric(
                        metric_id=f"network_clustering_{user_id}",
                        metric_name="Network Clustering",
                        metric_type=MetricType.EFFICIENCY,
                        value=avg_clustering,
                        unit="clustering_coefficient",
                        timestamp=datetime.now(timezone.utc),
                        user_id=user_id
                    ))
                except:
                    pass
                
                # Network growth rate
                time_span_days = (end_time - start_time).days
                if time_span_days > 0:
                    growth_rate = len(relationships) / time_span_days
                    metrics.append(UsageMetric(
                        metric_id=f"network_growth_{user_id}",
                        metric_name="Network Growth Rate",
                        metric_type=MetricType.GROWTH,
                        value=growth_rate,
                        unit="connections_per_day",
                        timestamp=datetime.now(timezone.utc),
                        user_id=user_id,
                        dimensions={"time_span_days": time_span_days}
                    ))
            
        except Exception as e:
            logger.error("Failed to collect network metrics", error=str(e))
        
        return metrics
    
    async def _collect_performance_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        config: AnalysisConfig
    ) -> List[UsageMetric]:
        """Collect performance pattern metrics."""
        metrics = []
        
        try:
            # Simulate performance metrics based on content and system usage
            memories = await self.memory_manager.get_memories_by_timeframe(
                user_id, start_time, end_time
            )
            
            if not memories:
                return metrics
            
            # Simulated response time based on content complexity
            response_times = []
            for memory in memories:
                # Simulate response time based on content length
                base_time = 50  # Base 50ms
                content_factor = len(memory.content) / 1000  # 1ms per 1000 chars
                simulated_time = base_time + content_factor * 10
                response_times.append(simulated_time)
            
            if response_times:
                metrics.append(UsageMetric(
                    metric_id=f"avg_response_time_{user_id}",
                    metric_name="Average Response Time",
                    metric_type=MetricType.DURATION,
                    value=np.mean(response_times),
                    unit="milliseconds",
                    timestamp=datetime.now(timezone.utc),
                    user_id=user_id,
                    dimensions={
                        "p95_response_time": np.percentile(response_times, 95),
                        "p99_response_time": np.percentile(response_times, 99),
                        "min_response_time": np.min(response_times),
                        "max_response_time": np.max(response_times)
                    }
                ))
            
            # Memory efficiency (how well memories are being used)
            total_memories = len(memories)
            accessed_memories = len([m for m in memories if getattr(m, 'access_count', 0) > 0])
            efficiency = accessed_memories / max(total_memories, 1)
            
            metrics.append(UsageMetric(
                metric_id=f"memory_efficiency_{user_id}",
                metric_name="Memory Efficiency",
                metric_type=MetricType.EFFICIENCY,
                value=efficiency,
                unit="utilization_ratio",
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                dimensions={
                    "total_memories": total_memories,
                    "accessed_memories": accessed_memories,
                    "unused_memories": total_memories - accessed_memories
                }
            ))
            
        except Exception as e:
            logger.error("Failed to collect performance metrics", error=str(e))
        
        return metrics
    
    def _calculate_start_time(self, timeframe: AnalysisTimeframe, end_time: datetime) -> datetime:
        """Calculate start time based on timeframe."""
        if timeframe == AnalysisTimeframe.LAST_HOUR:
            return end_time - timedelta(hours=1)
        elif timeframe == AnalysisTimeframe.LAST_DAY:
            return end_time - timedelta(days=1)
        elif timeframe == AnalysisTimeframe.LAST_WEEK:
            return end_time - timedelta(weeks=1)
        elif timeframe == AnalysisTimeframe.LAST_MONTH:
            return end_time - timedelta(days=30)
        elif timeframe == AnalysisTimeframe.LAST_QUARTER:
            return end_time - timedelta(days=90)
        elif timeframe == AnalysisTimeframe.LAST_YEAR:
            return end_time - timedelta(days=365)
        else:  # ALL_TIME
            return datetime(2020, 1, 1, tzinfo=timezone.utc)
    
    def _group_by_time_period(self, memories: List[Any], granularity: str) -> Dict[datetime, List[Any]]:
        """Group memories by time period."""
        groups = defaultdict(list)
        
        for memory in memories:
            if granularity == "hour":
                period = memory.created_at.replace(minute=0, second=0, microsecond=0)
            elif granularity == "day":
                period = memory.created_at.replace(hour=0, minute=0, second=0, microsecond=0)
            elif granularity == "week":
                days_since_monday = memory.created_at.weekday()
                period = memory.created_at.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
            elif granularity == "month":
                period = memory.created_at.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                period = memory.created_at.replace(hour=0, minute=0, second=0, microsecond=0)
            
            groups[period].append(memory)
        
        return dict(groups)
    
    async def generate_insights(
        self,
        metrics: List[UsageMetric],
        config: Optional[AnalysisConfig] = None
    ) -> List[AnalysisInsight]:
        """Generate insights from usage metrics."""
        try:
            if config is None:
                config = self.current_analysis_config
            
            insights = []
            
            # Peak activity time insight
            temporal_metrics = [m for m in metrics if "temporal" in m.metric_id]
            if temporal_metrics:
                activity_by_hour = defaultdict(float)
                for metric in temporal_metrics:
                    if "hour" in metric.dimensions:
                        hour = int(metric.dimensions["hour"])
                        activity_by_hour[hour] += metric.value
                
                if activity_by_hour:
                    peak_hour = max(activity_by_hour.items(), key=lambda x: x[1])
                    insights.append(AnalysisInsight(
                        insight_id=f"peak_activity_{datetime.now(timezone.utc).isoformat()}",
                        title=f"Peak Activity at {peak_hour[0]}:00",
                        description=f"User is most active at {peak_hour[0]}:00 with {peak_hour[1]:.1f} average activities.",
                        insight_type="temporal_pattern",
                        significance_score=0.8,
                        confidence_level=0.9,
                        impact_level="medium",
                        supporting_metrics=[m.metric_id for m in temporal_metrics[:3]],
                        recommendations=[
                            f"Schedule important tasks around {peak_hour[0]}:00",
                            "Consider automated suggestions during peak hours"
                        ],
                        timeframe=config.timeframe.value
                    ))
            
            # Content efficiency insight
            complexity_metrics = [m for m in metrics if "complexity" in m.metric_id]
            length_metrics = [m for m in metrics if "length" in m.metric_id]
            
            if complexity_metrics and length_metrics:
                avg_complexity = complexity_metrics[0].value
                avg_length = length_metrics[0].value
                
                if avg_complexity < 0.3:  # Low complexity
                    insights.append(AnalysisInsight(
                        insight_id=f"content_complexity_{datetime.now(timezone.utc).isoformat()}",
                        title="Content Could Be More Detailed",
                        description=f"Average content complexity is {avg_complexity:.2f}, suggesting opportunities for richer detail.",
                        insight_type="content_quality",
                        significance_score=0.6,
                        confidence_level=0.7,
                        impact_level="low",
                        supporting_metrics=[m.metric_id for m in complexity_metrics + length_metrics],
                        recommendations=[
                            "Add more context and details to memories",
                            "Use more descriptive language",
                            "Include examples and explanations"
                        ],
                        timeframe=config.timeframe.value
                    ))
            
            # Network connectivity insight
            network_metrics = [m for m in metrics if "network" in m.metric_id]
            if network_metrics:
                density_metric = next((m for m in network_metrics if "density" in m.metric_id), None)
                if density_metric and density_metric.value < 0.1:
                    insights.append(AnalysisInsight(
                        insight_id=f"network_connectivity_{datetime.now(timezone.utc).isoformat()}",
                        title="Memories Are Not Well Connected",
                        description=f"Network density is {density_metric.value:.3f}, indicating isolated memories.",
                        insight_type="connectivity",
                        significance_score=0.7,
                        confidence_level=0.8,
                        impact_level="medium",
                        supporting_metrics=[m.metric_id for m in network_metrics],
                        recommendations=[
                            "Create more connections between related memories",
                            "Use consistent tags and topics",
                            "Review and link related content"
                        ],
                        timeframe=config.timeframe.value
                    ))
            
            # Efficiency insight
            efficiency_metrics = [m for m in metrics if m.metric_type == MetricType.EFFICIENCY]
            if efficiency_metrics:
                avg_efficiency = np.mean([m.value for m in efficiency_metrics])
                if avg_efficiency > 0.8:
                    insights.append(AnalysisInsight(
                        insight_id=f"high_efficiency_{datetime.now(timezone.utc).isoformat()}",
                        title="Excellent Memory Utilization",
                        description=f"Average efficiency is {avg_efficiency:.2f}, showing excellent memory usage patterns.",
                        insight_type="performance",
                        significance_score=0.9,
                        confidence_level=0.9,
                        impact_level="high",
                        supporting_metrics=[m.metric_id for m in efficiency_metrics],
                        recommendations=[
                            "Continue current usage patterns",
                            "Share best practices with other users",
                            "Consider expanding memory collection"
                        ],
                        timeframe=config.timeframe.value
                    ))
            
            logger.info(
                "Insights generated",
                insight_count=len(insights),
                metric_count=len(metrics)
            )
            
            return insights
            
        except Exception as e:
            logger.error("Failed to generate insights", error=str(e))
            return []
    
    def create_dashboard_template(
        self,
        name: str,
        description: str,
        category: str,
        creator_id: str,
        widget_configs: List[Dict[str, Any]]
    ) -> DashboardTemplate:
        """Create a custom dashboard template."""
        template = DashboardTemplate(
            template_id=f"template_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            name=name,
            description=description,
            category=category,
            layout_config={
                "columns": 12,
                "rows": "auto",
                "spacing": 20,
                "responsive": True
            },
            widget_configs=widget_configs,
            created_by=creator_id
        )
        
        self.templates[template.template_id] = template
        
        logger.info(
            "Dashboard template created",
            template_id=template.template_id,
            name=name,
            widget_count=len(widget_configs)
        )
        
        return template
    
    def create_dash_app(self) -> dash.Dash:
        """Create Dash application for usage analytics."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("Usage Pattern Analysis Dashboard", className="text-center mb-4"),
                        
                        # Configuration controls
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("User ID:"),
                                        dcc.Input(
                                            id="user-id-input",
                                            type="text",
                                            placeholder="Enter user ID",
                                            value="default_user"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Timeframe:"),
                                        dcc.Dropdown(
                                            id="timeframe-dropdown",
                                            options=[
                                                {"label": "Last Hour", "value": "last_hour"},
                                                {"label": "Last Day", "value": "last_day"},
                                                {"label": "Last Week", "value": "last_week"},
                                                {"label": "Last Month", "value": "last_month"}
                                            ],
                                            value="last_week"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Analysis Types:"),
                                        dcc.Dropdown(
                                            id="analysis-types-dropdown",
                                            options=[
                                                {"label": "Temporal Patterns", "value": "temporal_patterns"},
                                                {"label": "Access Patterns", "value": "access_patterns"},
                                                {"label": "Content Patterns", "value": "content_patterns"},
                                                {"label": "Network Patterns", "value": "network_patterns"}
                                            ],
                                            value=["temporal_patterns", "access_patterns"],
                                            multi=True
                                        )
                                    ], width=4),
                                    
                                    dbc.Col([
                                        dbc.Button(
                                            "Analyze",
                                            id="analyze-btn",
                                            color="primary",
                                            className="mt-4"
                                        )
                                    ], width=2)
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Main content tabs
                        dbc.Tabs([
                            dbc.Tab(label="Overview", tab_id="overview-tab"),
                            dbc.Tab(label="Temporal Analysis", tab_id="temporal-tab"),
                            dbc.Tab(label="Content Analysis", tab_id="content-tab"),
                            dbc.Tab(label="Network Analysis", tab_id="network-tab"),
                            dbc.Tab(label="Insights", tab_id="insights-tab")
                        ], id="main-tabs", active_tab="overview-tab"),
                        
                        # Tab content
                        html.Div(id="tab-content", className="mt-4")
                        
                    ], width=12)
                ])
            ], fluid=True),
            
            # Data stores
            dcc.Store(id="metrics-store"),
            dcc.Store(id="insights-store"),
            
            # Auto-refresh interval
            dcc.Interval(
                id="refresh-interval",
                interval=60000,  # 1 minute
                n_intervals=0,
                disabled=True
            )
        ])
        
        # Setup callbacks
        self._setup_analytics_callbacks(app)
        
        return app
    
    def _setup_analytics_callbacks(self, app: dash.Dash) -> None:
        """Setup Dash application callbacks for analytics."""
        
        @app.callback(
            [Output("metrics-store", "data"),
             Output("insights-store", "data")],
            [Input("analyze-btn", "n_clicks"),
             Input("refresh-interval", "n_intervals")],
            [State("user-id-input", "value"),
             State("timeframe-dropdown", "value"),
             State("analysis-types-dropdown", "value")]
        )
        def update_analysis(n_clicks, n_intervals, user_id, timeframe, analysis_types):
            if not user_id or not analysis_types:
                raise PreventUpdate
            
            try:
                # Create configuration
                config = AnalysisConfig(
                    timeframe=AnalysisTimeframe(timeframe),
                    analysis_types=[AnalysisType(t) for t in analysis_types]
                )
                
                # This would be async in real implementation
                # metrics = await self.collect_usage_metrics(user_id, config)
                # insights = await self.generate_insights(metrics, config)
                
                # For demo, create sample data
                metrics_data = self._create_sample_metrics(user_id, config)
                insights_data = self._create_sample_insights(config)
                
                return metrics_data, insights_data
                
            except Exception as e:
                logger.error("Failed to update analysis", error=str(e))
                return {}, {}
        
        @app.callback(
            Output("tab-content", "children"),
            [Input("main-tabs", "active_tab"),
             Input("metrics-store", "data"),
             Input("insights-store", "data")]
        )
        def update_tab_content(active_tab, metrics_data, insights_data):
            if active_tab == "overview-tab":
                return self._create_overview_content(metrics_data, insights_data)
            elif active_tab == "temporal-tab":
                return self._create_temporal_content(metrics_data)
            elif active_tab == "content-tab":
                return self._create_content_content(metrics_data)
            elif active_tab == "network-tab":
                return self._create_network_content(metrics_data)
            elif active_tab == "insights-tab":
                return self._create_insights_content(insights_data)
            else:
                return html.Div("Select a tab to view content")
    
    def _create_sample_metrics(self, user_id: str, config: AnalysisConfig) -> Dict[str, Any]:
        """Create sample metrics for demo."""
        return {
            "temporal_activity": [
                {"hour": i, "activity": np.random.poisson(5)} 
                for i in range(24)
            ],
            "content_stats": {
                "avg_length": np.random.normal(200, 50),
                "complexity": np.random.uniform(0.3, 0.8),
                "total_memories": np.random.randint(50, 200)
            },
            "network_stats": {
                "density": np.random.uniform(0.1, 0.5),
                "clustering": np.random.uniform(0.2, 0.7),
                "connections": np.random.randint(100, 500)
            }
        }
    
    def _create_sample_insights(self, config: AnalysisConfig) -> List[Dict[str, Any]]:
        """Create sample insights for demo."""
        return [
            {
                "title": "Peak Activity at 14:00",
                "description": "User is most active in the afternoon",
                "significance": 0.8,
                "type": "temporal_pattern"
            },
            {
                "title": "Good Memory Utilization",
                "description": "85% of memories are being accessed regularly",
                "significance": 0.7,
                "type": "efficiency"
            }
        ]
    
    def _create_overview_content(self, metrics_data: Dict[str, Any], insights_data: List[Dict[str, Any]]) -> html.Div:
        """Create overview tab content."""
        if not metrics_data:
            return html.Div("No data available. Click 'Analyze' to generate metrics.")
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Key Metrics", className="card-title"),
                            html.P(f"Total Memories: {metrics_data.get('content_stats', {}).get('total_memories', 0)}"),
                            html.P(f"Network Density: {metrics_data.get('network_stats', {}).get('density', 0):.3f}"),
                            html.P(f"Avg Content Length: {metrics_data.get('content_stats', {}).get('avg_length', 0):.0f} chars")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Recent Insights", className="card-title"),
                            html.Div([
                                html.P(f"• {insight['title']}")
                                for insight in insights_data[:3]
                            ] if insights_data else [html.P("No insights available")])
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_temporal_content(self, metrics_data: Dict[str, Any]) -> html.Div:
        """Create temporal analysis tab content."""
        if not metrics_data or 'temporal_activity' not in metrics_data:
            return html.Div("No temporal data available.")
        
        # Create activity chart
        activity_data = metrics_data['temporal_activity']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[d['hour'] for d in activity_data],
            y=[d['activity'] for d in activity_data],
            mode='lines+markers',
            name='Activity'
        ))
        fig.update_layout(
            title="Activity by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Activity Count"
        )
        
        return html.Div([
            dcc.Graph(figure=fig)
        ])
    
    def _create_content_content(self, metrics_data: Dict[str, Any]) -> html.Div:
        """Create content analysis tab content."""
        if not metrics_data:
            return html.Div("No content data available.")
        
        content_stats = metrics_data.get('content_stats', {})
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Content Statistics"),
                    html.P(f"Average Length: {content_stats.get('avg_length', 0):.0f} characters"),
                    html.P(f"Complexity Score: {content_stats.get('complexity', 0):.2f}"),
                    html.P(f"Total Memories: {content_stats.get('total_memories', 0)}")
                ])
            ])
        ])
    
    def _create_network_content(self, metrics_data: Dict[str, Any]) -> html.Div:
        """Create network analysis tab content."""
        if not metrics_data:
            return html.Div("No network data available.")
        
        network_stats = metrics_data.get('network_stats', {})
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Network Statistics"),
                    html.P(f"Network Density: {network_stats.get('density', 0):.3f}"),
                    html.P(f"Clustering Coefficient: {network_stats.get('clustering', 0):.3f}"),
                    html.P(f"Total Connections: {network_stats.get('connections', 0)}")
                ])
            ])
        ])
    
    def _create_insights_content(self, insights_data: List[Dict[str, Any]]) -> html.Div:
        """Create insights tab content."""
        if not insights_data:
            return html.Div("No insights available.")
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(insight['title']),
                            html.P(insight['description']),
                            dbc.Badge(f"Significance: {insight['significance']:.2f}", color="info")
                        ])
                    ], className="mb-3")
                ], width=12)
            ])
            for insight in insights_data
        ])
    
    async def export_dashboard(
        self,
        template_id: str,
        export_format: str = "html",
        include_data: bool = True
    ) -> str:
        """Export dashboard to various formats."""
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template {template_id} not found")
            
            template = self.templates[template_id]
            
            if export_format == "html":
                # Export as static HTML
                export_path = f"/tmp/dashboard_{template_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
                
                # Create static HTML content
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>{template.name} - Usage Analytics Dashboard</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .header {{ background: #f8f9fa; padding: 20px; margin-bottom: 20px; }}
                        .metric {{ background: #e9ecef; padding: 15px; margin: 10px; border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>{template.name}</h1>
                        <p>{template.description}</p>
                        <p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
                    </div>
                    <div class="content">
                        <h2>Dashboard Template</h2>
                        <p>Category: {template.category}</p>
                        <p>Widgets: {len(template.widget_configs)}</p>
                        <p>Created by: {template.created_by}</p>
                    </div>
                </body>
                </html>
                """
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logger.info(
                    "Dashboard exported",
                    template_id=template_id,
                    format=export_format,
                    path=export_path
                )
                
                return export_path
            
            else:
                raise ValueError(f"Export format {export_format} not supported")
                
        except Exception as e:
            logger.error("Failed to export dashboard", error=str(e))
            raise
    
    async def start_analytics_server(self, host: str = "127.0.0.1", port: int = 8051) -> None:
        """Start the analytics dashboard server."""
        if self.is_running:
            logger.warning("Analytics server already running")
            return
        
        try:
            self.app = self.create_dash_app()
            self.is_running = True
            
            logger.info(
                "Starting analytics server",
                host=host,
                port=port
            )
            
            # Run in separate thread
            import threading
            server_thread = threading.Thread(
                target=lambda: self.app.run_server(
                    host=host,
                    port=port,
                    debug=False,
                    use_reloader=False
                )
            )
            server_thread.daemon = True
            server_thread.start()
            
        except Exception as e:
            logger.error("Failed to start analytics server", error=str(e))
            self.is_running = False
            raise


# Module exports
__all__ = [
    "LocalUsageAnalyzer",
    "UsageMetric",
    "AnalysisInsight", 
    "DashboardTemplate",
    "AnalysisConfig",
    "AnalysisTimeframe",
    "AnalysisType",
    "MetricType",
    "VisualizationStyle"
]