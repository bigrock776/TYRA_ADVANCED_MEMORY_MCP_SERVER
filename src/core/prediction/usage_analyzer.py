"""
Usage Pattern Analysis for Predictive Memory Management.

This module provides comprehensive usage pattern analysis using local ML clustering,
time series analysis, user behavior modeling, and PageRank-based popularity scoring.
All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import json
import math
import heapq

# ML and analytics imports - all local
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class PatternType(str, Enum):
    """Types of usage patterns."""
    SEQUENTIAL = "sequential"        # Sequential access patterns
    CLUSTERED = "clustered"         # Clustered access patterns
    RANDOM = "random"               # Random access patterns
    PERIODIC = "periodic"           # Periodic access patterns
    BURSTY = "bursty"              # Bursty access patterns
    TRENDING = "trending"           # Trending access patterns
    SEASONAL = "seasonal"           # Seasonal access patterns


class AccessFrequency(str, Enum):
    """Frequency categories for access patterns."""
    VERY_HIGH = "very_high"     # Multiple times per hour
    HIGH = "high"               # Multiple times per day
    MEDIUM = "medium"           # Daily access
    LOW = "low"                 # Weekly access
    VERY_LOW = "very_low"       # Monthly or less


@dataclass
class AccessPattern:
    """Represents an access pattern for a memory."""
    memory_id: str
    access_count: int
    last_access: datetime
    first_access: datetime
    access_frequency: AccessFrequency
    pattern_type: PatternType
    time_between_accesses: List[float]  # Hours between accesses
    access_times: List[datetime]
    user_ids: Set[str]
    confidence: float = 0.0
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate various metrics for the access pattern."""
        if not self.time_between_accesses:
            return {}
        
        intervals = np.array(self.time_between_accesses)
        return {
            'mean_interval': float(np.mean(intervals)),
            'std_interval': float(np.std(intervals)),
            'median_interval': float(np.median(intervals)),
            'cv_interval': float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0,
            'regularity_score': 1.0 / (1.0 + np.std(intervals)) if len(intervals) > 1 else 1.0,
            'recency_score': max(0, 1.0 - (datetime.utcnow() - self.last_access).days / 30.0)
        }


@dataclass
class UserBehaviorModel:
    """Models user behavior patterns."""
    user_id: str
    access_patterns: Dict[str, AccessPattern]
    preferred_times: List[int]  # Hours of day (0-23)
    session_duration: float  # Average session duration in minutes
    memory_types: Counter  # Counter of memory types accessed
    domains: Counter  # Counter of domains accessed
    activity_level: str  # "high", "medium", "low"
    behavior_vector: Optional[np.ndarray] = None
    
    def calculate_similarity(self, other: 'UserBehaviorModel') -> float:
        """Calculate similarity with another user behavior model."""
        if self.behavior_vector is None or other.behavior_vector is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(self.behavior_vector, other.behavior_vector)
        norm_product = np.linalg.norm(self.behavior_vector) * np.linalg.norm(other.behavior_vector)
        
        return float(dot_product / norm_product) if norm_product > 0 else 0.0


@dataclass
class PopularityScore:
    """PageRank-based popularity scoring."""
    memory_id: str
    pagerank_score: float
    local_popularity: float  # Within user's network
    global_popularity: float  # Across all users
    trend_score: float  # Recent trend in popularity
    final_score: float


class UsagePattern(BaseModel):
    """Pydantic model for usage pattern analysis results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    memory_id: str = Field(description="Memory ID")
    pattern_type: PatternType = Field(description="Type of usage pattern")
    access_frequency: AccessFrequency = Field(description="Access frequency category")
    confidence: float = Field(ge=0.0, le=1.0, description="Pattern confidence")
    metrics: Dict[str, float] = Field(description="Pattern metrics")
    predictions: Dict[str, Any] = Field(description="Future access predictions")


class UsageAnalyzer:
    """
    Advanced usage pattern analyzer for predictive memory management.
    
    Features:
    - Scikit-learn clustering for access pattern discovery
    - Time series analysis with statsmodels for trend detection
    - User behavior modeling with collaborative filtering
    - PageRank-based popularity scoring using NetworkX
    - Seasonal pattern detection and forecasting
    - Local ML training for pattern recognition
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        analysis_window_days: int = 30,
        min_accesses_for_pattern: int = 5,
        clustering_algorithm: str = "kmeans"
    ):
        """
        Initialize the usage analyzer.
        
        Args:
            redis_cache: Optional Redis cache for analytics persistence
            analysis_window_days: Days of data to analyze
            min_accesses_for_pattern: Minimum accesses needed for pattern detection
            clustering_algorithm: Clustering algorithm ("kmeans", "dbscan", "hierarchical")
        """
        self.redis_cache = redis_cache
        self.analysis_window_days = analysis_window_days
        self.min_accesses_for_pattern = min_accesses_for_pattern
        self.clustering_algorithm = clustering_algorithm
        
        # Data storage
        self.access_logs: List[Dict[str, Any]] = []
        self.memory_patterns: Dict[str, AccessPattern] = {}
        self.user_models: Dict[str, UserBehaviorModel] = {}
        self.popularity_scores: Dict[str, PopularityScore] = {}
        
        # ML models
        self.scaler = StandardScaler()
        self.clusterer = None
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        
        # Graph for PageRank
        self.access_graph = nx.Graph()
        
        # Analysis cache
        self.pattern_cache: Dict[str, UsagePattern] = {}
        self.last_analysis: Optional[datetime] = None
        
        # Performance tracking
        self.stats = {
            'total_accesses_analyzed': 0,
            'patterns_discovered': 0,
            'users_modeled': 0,
            'prediction_accuracy': 0.0,
            'analysis_runs': 0
        }
        
        logger.info(
            "Usage analyzer initialized",
            analysis_window_days=analysis_window_days,
            clustering_algorithm=clustering_algorithm
        )
    
    async def log_access(
        self,
        memory_id: str,
        user_id: str,
        access_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a memory access for pattern analysis.
        
        Args:
            memory_id: ID of accessed memory
            user_id: ID of user accessing memory
            access_time: Time of access (defaults to now)
            session_id: Optional session ID
            metadata: Additional metadata
        """
        access_time = access_time or datetime.utcnow()
        
        access_log = {
            'memory_id': memory_id,
            'user_id': user_id,
            'access_time': access_time,
            'session_id': session_id,
            'metadata': metadata or {}
        }
        
        self.access_logs.append(access_log)
        self.stats['total_accesses_analyzed'] += 1
        
        # Add to graph for PageRank
        self.access_graph.add_edge(user_id, memory_id)
        
        # Trigger analysis if we have enough new data
        if len(self.access_logs) % 100 == 0:  # Every 100 accesses
            await self._incremental_analysis()
    
    async def analyze_patterns(
        self,
        memories: Optional[List[Memory]] = None,
        force_full_analysis: bool = False
    ) -> Dict[str, UsagePattern]:
        """
        Analyze usage patterns for memories.
        
        Args:
            memories: Optional list of specific memories to analyze
            force_full_analysis: Force full reanalysis
            
        Returns:
            Dictionary of memory ID to usage patterns
        """
        start_time = datetime.utcnow()
        
        # Check if we need to run analysis
        if (not force_full_analysis and 
            self.last_analysis and 
            (start_time - self.last_analysis).hours < 1):
            return self.pattern_cache
        
        logger.info("Starting usage pattern analysis")
        
        # Prepare data
        await self._prepare_analysis_data()
        
        # Perform clustering analysis
        patterns = await self._cluster_access_patterns()
        
        # Perform time series analysis
        await self._analyze_temporal_patterns(patterns)
        
        # Build user behavior models
        await self._build_user_models()
        
        # Calculate popularity scores
        await self._calculate_popularity_scores()
        
        # Generate predictions
        await self._generate_predictions(patterns)
        
        # Update cache
        self.pattern_cache = patterns
        self.last_analysis = start_time
        self.stats['analysis_runs'] += 1
        self.stats['patterns_discovered'] = len(patterns)
        
        analysis_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "Usage pattern analysis completed",
            patterns_found=len(patterns),
            analysis_time=analysis_time
        )
        
        return patterns
    
    async def _prepare_analysis_data(self):
        """Prepare access log data for analysis."""
        # Filter to analysis window
        cutoff_date = datetime.utcnow() - timedelta(days=self.analysis_window_days)
        recent_logs = [
            log for log in self.access_logs 
            if log['access_time'] >= cutoff_date
        ]
        
        # Group by memory ID
        memory_accesses = defaultdict(list)
        for log in recent_logs:
            memory_accesses[log['memory_id']].append(log)
        
        # Create access patterns
        self.memory_patterns = {}
        for memory_id, accesses in memory_accesses.items():
            if len(accesses) >= self.min_accesses_for_pattern:
                pattern = await self._create_access_pattern(memory_id, accesses)
                self.memory_patterns[memory_id] = pattern
    
    async def _create_access_pattern(
        self, 
        memory_id: str, 
        accesses: List[Dict[str, Any]]
    ) -> AccessPattern:
        """Create an access pattern from access logs."""
        # Sort by access time
        accesses.sort(key=lambda x: x['access_time'])
        
        access_times = [acc['access_time'] for acc in accesses]
        user_ids = set(acc['user_id'] for acc in accesses)
        
        # Calculate time intervals between accesses
        intervals = []
        for i in range(1, len(access_times)):
            interval = (access_times[i] - access_times[i-1]).total_seconds() / 3600.0
            intervals.append(interval)
        
        # Determine access frequency
        if not intervals:
            frequency = AccessFrequency.VERY_LOW
        else:
            mean_interval = np.mean(intervals)
            if mean_interval < 1:  # Less than 1 hour
                frequency = AccessFrequency.VERY_HIGH
            elif mean_interval < 24:  # Less than 1 day
                frequency = AccessFrequency.HIGH
            elif mean_interval < 168:  # Less than 1 week
                frequency = AccessFrequency.MEDIUM
            elif mean_interval < 720:  # Less than 1 month
                frequency = AccessFrequency.LOW
            else:
                frequency = AccessFrequency.VERY_LOW
        
        # Determine pattern type (will be refined by clustering)
        pattern_type = await self._detect_pattern_type(intervals)
        
        return AccessPattern(
            memory_id=memory_id,
            access_count=len(accesses),
            last_access=access_times[-1],
            first_access=access_times[0],
            access_frequency=frequency,
            pattern_type=pattern_type,
            time_between_accesses=intervals,
            access_times=access_times,
            user_ids=user_ids
        )
    
    async def _detect_pattern_type(self, intervals: List[float]) -> PatternType:
        """Detect pattern type from access intervals."""
        if not intervals or len(intervals) < 2:
            return PatternType.RANDOM
        
        intervals_array = np.array(intervals)
        
        # Calculate statistics
        mean_interval = np.mean(intervals_array)
        std_interval = np.std(intervals_array)
        cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
        
        # Check for periodicity using autocorrelation
        if len(intervals) >= 10:
            try:
                autocorr = np.correlate(intervals_array, intervals_array, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # Look for peaks in autocorrelation (indicating periodicity)
                if len(autocorr) > 3 and max(autocorr[2:min(10, len(autocorr))]) > 0.5:
                    return PatternType.PERIODIC
            except:
                pass
        
        # Determine pattern based on coefficient of variation
        if cv < 0.3:  # Low variation - regular pattern
            return PatternType.SEQUENTIAL
        elif cv > 2.0:  # High variation - bursty pattern
            return PatternType.BURSTY
        else:
            return PatternType.CLUSTERED
    
    async def _cluster_access_patterns(self) -> Dict[str, UsagePattern]:
        """Perform clustering analysis on access patterns."""
        if not self.memory_patterns:
            return {}
        
        # Extract features for clustering
        features = []
        memory_ids = []
        
        for memory_id, pattern in self.memory_patterns.items():
            metrics = pattern.calculate_metrics()
            if metrics:
                feature_vector = [
                    pattern.access_count,
                    metrics.get('mean_interval', 0),
                    metrics.get('std_interval', 0),
                    metrics.get('cv_interval', 0),
                    metrics.get('regularity_score', 0),
                    metrics.get('recency_score', 0),
                    len(pattern.user_ids),
                    (pattern.last_access - pattern.first_access).days
                ]
                features.append(feature_vector)
                memory_ids.append(memory_id)
        
        if len(features) < 2:
            return {}
        
        # Normalize features
        features_array = np.array(features)
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Apply PCA for dimensionality reduction
        if features_normalized.shape[1] > 2:
            features_pca = self.pca.fit_transform(features_normalized)
        else:
            features_pca = features_normalized
        
        # Perform clustering
        patterns = {}
        try:
            if self.clustering_algorithm == "kmeans":
                # Determine optimal number of clusters
                max_clusters = min(8, len(features) // 2)
                if max_clusters >= 2:
                    silhouette_scores = []
                    for k in range(2, max_clusters + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(features_pca)
                        score = silhouette_score(features_pca, labels)
                        silhouette_scores.append((k, score))
                    
                    # Choose k with best silhouette score
                    best_k = max(silhouette_scores, key=lambda x: x[1])[0]
                    self.clusterer = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                    
            elif self.clustering_algorithm == "dbscan":
                self.clusterer = DBSCAN(eps=0.5, min_samples=2)
                
            elif self.clustering_algorithm == "hierarchical":
                n_clusters = min(5, len(features) // 2)
                self.clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            
            # Fit clustering model
            if self.clusterer:
                cluster_labels = self.clusterer.fit_predict(features_pca)
                
                # Create usage patterns
                for i, memory_id in enumerate(memory_ids):
                    pattern = self.memory_patterns[memory_id]
                    metrics = pattern.calculate_metrics()
                    
                    # Refine pattern type based on cluster
                    refined_pattern_type = await self._refine_pattern_type(
                        pattern.pattern_type, 
                        cluster_labels[i],
                        features_pca[i]
                    )
                    
                    # Calculate confidence based on cluster cohesion
                    confidence = await self._calculate_pattern_confidence(
                        features_pca[i], 
                        cluster_labels[i], 
                        features_pca, 
                        cluster_labels
                    )
                    
                    usage_pattern = UsagePattern(
                        memory_id=memory_id,
                        pattern_type=refined_pattern_type,
                        access_frequency=pattern.access_frequency,
                        confidence=confidence,
                        metrics=metrics,
                        predictions={}
                    )
                    
                    patterns[memory_id] = usage_pattern
                    
        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
            
            # Fallback: create patterns without clustering
            for memory_id, pattern in self.memory_patterns.items():
                metrics = pattern.calculate_metrics()
                usage_pattern = UsagePattern(
                    memory_id=memory_id,
                    pattern_type=pattern.pattern_type,
                    access_frequency=pattern.access_frequency,
                    confidence=0.5,  # Default confidence
                    metrics=metrics,
                    predictions={}
                )
                patterns[memory_id] = usage_pattern
        
        return patterns
    
    async def _refine_pattern_type(
        self, 
        original_type: PatternType, 
        cluster_label: int, 
        feature_vector: np.ndarray
    ) -> PatternType:
        """Refine pattern type based on clustering results."""
        # Use cluster characteristics to refine pattern type
        # This is a simplified heuristic - could be enhanced with more sophisticated analysis
        
        if cluster_label == -1:  # DBSCAN noise
            return PatternType.RANDOM
        
        # Analyze feature characteristics
        if len(feature_vector) >= 4:
            regularity = feature_vector[4] if len(feature_vector) > 4 else 0
            cv = feature_vector[3] if len(feature_vector) > 3 else 0
            
            if regularity > 0.8:
                return PatternType.PERIODIC
            elif cv > 1.5:
                return PatternType.BURSTY
            elif cv < 0.5:
                return PatternType.SEQUENTIAL
        
        return original_type
    
    async def _calculate_pattern_confidence(
        self,
        point: np.ndarray,
        cluster_label: int,
        all_points: np.ndarray,
        all_labels: np.ndarray
    ) -> float:
        """Calculate confidence in pattern classification."""
        if cluster_label == -1:  # Noise cluster
            return 0.1
        
        # Find other points in the same cluster
        cluster_points = all_points[all_labels == cluster_label]
        
        if len(cluster_points) <= 1:
            return 0.5
        
        # Calculate distance to cluster centroid
        centroid = np.mean(cluster_points, axis=0)
        distance_to_centroid = np.linalg.norm(point - centroid)
        
        # Calculate average distance within cluster
        distances = [np.linalg.norm(p - centroid) for p in cluster_points]
        avg_distance = np.mean(distances)
        
        # Confidence is inversely related to relative distance from centroid
        if avg_distance == 0:
            confidence = 1.0
        else:
            confidence = max(0.1, 1.0 - (distance_to_centroid / avg_distance))
        
        return min(1.0, confidence)
    
    async def _analyze_temporal_patterns(self, patterns: Dict[str, UsagePattern]):
        """Analyze temporal patterns using time series analysis."""
        for memory_id, usage_pattern in patterns.items():
            access_pattern = self.memory_patterns.get(memory_id)
            if not access_pattern or len(access_pattern.access_times) < 10:
                continue
            
            try:
                # Create time series
                access_series = await self._create_time_series(access_pattern.access_times)
                
                # Seasonal decomposition
                if len(access_series) >= 24:  # Need enough data points
                    decomposition = seasonal_decompose(
                        access_series, 
                        model='additive', 
                        period=min(24, len(access_series) // 3)
                    )
                    
                    # Extract trend and seasonal components
                    trend_strength = float(np.var(decomposition.trend.dropna()) / np.var(access_series))
                    seasonal_strength = float(np.var(decomposition.seasonal) / np.var(access_series))
                    
                    # Update pattern type if strong seasonal component
                    if seasonal_strength > 0.3:
                        usage_pattern.pattern_type = PatternType.SEASONAL
                    elif trend_strength > 0.5:
                        usage_pattern.pattern_type = PatternType.TRENDING
                    
                    # Add temporal metrics
                    usage_pattern.metrics.update({
                        'trend_strength': trend_strength,
                        'seasonal_strength': seasonal_strength,
                        'residual_variance': float(np.var(decomposition.resid.dropna()))
                    })
                
                # ARIMA forecasting for predictions
                if len(access_series) >= 20:
                    try:
                        model = ARIMA(access_series, order=(1, 1, 1))
                        fitted_model = model.fit()
                        
                        # Forecast next 7 days
                        forecast = fitted_model.forecast(steps=7)
                        
                        usage_pattern.predictions['next_7_days_forecast'] = forecast.tolist()
                        usage_pattern.predictions['forecast_confidence'] = 0.8  # Simplified
                        
                    except Exception as e:
                        logger.debug(f"ARIMA forecasting failed for {memory_id}: {e}")
                
            except Exception as e:
                logger.warning(f"Temporal analysis failed for {memory_id}: {e}")
    
    async def _create_time_series(self, access_times: List[datetime]) -> pd.Series:
        """Create time series from access times."""
        # Create hourly bins
        start_time = min(access_times)
        end_time = max(access_times)
        
        # Create hourly index
        hourly_index = pd.date_range(
            start=start_time.replace(minute=0, second=0, microsecond=0),
            end=end_time.replace(minute=59, second=59, microsecond=999999),
            freq='H'
        )
        
        # Count accesses per hour
        access_counts = pd.Series(0, index=hourly_index)
        
        for access_time in access_times:
            hour_key = access_time.replace(minute=0, second=0, microsecond=0)
            if hour_key in access_counts.index:
                access_counts[hour_key] += 1
        
        return access_counts
    
    async def _build_user_models(self):
        """Build user behavior models."""
        user_accesses = defaultdict(list)
        
        # Group accesses by user
        for log in self.access_logs:
            if log['access_time'] >= datetime.utcnow() - timedelta(days=self.analysis_window_days):
                user_accesses[log['user_id']].append(log)
        
        self.user_models = {}
        
        for user_id, accesses in user_accesses.items():
            if len(accesses) < 5:  # Need minimum accesses
                continue
            
            # Extract user behavior features
            access_hours = [acc['access_time'].hour for acc in accesses]
            memory_types = Counter()
            domains = Counter()
            
            # Calculate session durations (simplified)
            sessions = await self._group_into_sessions(accesses)
            avg_session_duration = np.mean([s['duration'] for s in sessions])
            
            # Determine activity level
            accesses_per_day = len(accesses) / self.analysis_window_days
            if accesses_per_day > 10:
                activity_level = "high"
            elif accesses_per_day > 3:
                activity_level = "medium"
            else:
                activity_level = "low"
            
            # Extract memory patterns for this user
            user_patterns = {}
            for memory_id, pattern in self.memory_patterns.items():
                if user_id in pattern.user_ids:
                    user_patterns[memory_id] = pattern
            
            # Create behavior vector
            behavior_vector = await self._create_behavior_vector(
                access_hours, memory_types, domains, activity_level
            )
            
            user_model = UserBehaviorModel(
                user_id=user_id,
                access_patterns=user_patterns,
                preferred_times=self._find_preferred_times(access_hours),
                session_duration=avg_session_duration,
                memory_types=memory_types,
                domains=domains,
                activity_level=activity_level,
                behavior_vector=behavior_vector
            )
            
            self.user_models[user_id] = user_model
        
        self.stats['users_modeled'] = len(self.user_models)
    
    async def _group_into_sessions(self, accesses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group accesses into sessions based on time gaps."""
        if not accesses:
            return []
        
        # Sort by time
        sorted_accesses = sorted(accesses, key=lambda x: x['access_time'])
        
        sessions = []
        current_session = [sorted_accesses[0]]
        
        for i in range(1, len(sorted_accesses)):
            time_gap = (sorted_accesses[i]['access_time'] - 
                       sorted_accesses[i-1]['access_time']).total_seconds() / 60  # minutes
            
            if time_gap <= 30:  # 30 minutes session timeout
                current_session.append(sorted_accesses[i])
            else:
                # End current session and start new one
                if len(current_session) > 1:
                    session_duration = (current_session[-1]['access_time'] - 
                                      current_session[0]['access_time']).total_seconds() / 60
                    sessions.append({
                        'accesses': current_session,
                        'duration': session_duration
                    })
                current_session = [sorted_accesses[i]]
        
        # Add final session
        if len(current_session) > 1:
            session_duration = (current_session[-1]['access_time'] - 
                              current_session[0]['access_time']).total_seconds() / 60
            sessions.append({
                'accesses': current_session,
                'duration': session_duration
            })
        
        return sessions
    
    def _find_preferred_times(self, access_hours: List[int]) -> List[int]:
        """Find preferred hours of access."""
        if not access_hours:
            return []
        
        hour_counts = Counter(access_hours)
        mean_count = np.mean(list(hour_counts.values()))
        
        # Hours with above-average access
        preferred = [hour for hour, count in hour_counts.items() if count > mean_count]
        return sorted(preferred)
    
    async def _create_behavior_vector(
        self,
        access_hours: List[int],
        memory_types: Counter,
        domains: Counter,
        activity_level: str
    ) -> np.ndarray:
        """Create numerical behavior vector for user."""
        vector = []
        
        # Time-based features (24 hours)
        hour_distribution = np.zeros(24)
        for hour in access_hours:
            hour_distribution[hour] += 1
        if len(access_hours) > 0:
            hour_distribution /= len(access_hours)  # Normalize
        vector.extend(hour_distribution)
        
        # Activity level encoding
        activity_encoding = {"low": 0.33, "medium": 0.66, "high": 1.0}
        vector.append(activity_encoding.get(activity_level, 0.5))
        
        # Pattern diversity (entropy of access distribution)
        if access_hours:
            unique_hours = len(set(access_hours))
            diversity = unique_hours / 24.0
            vector.append(diversity)
        else:
            vector.append(0.0)
        
        return np.array(vector)
    
    async def _calculate_popularity_scores(self):
        """Calculate PageRank-based popularity scores."""
        if not self.access_graph.nodes():
            return
        
        try:
            # Calculate PageRank scores
            pagerank_scores = nx.pagerank(self.access_graph, weight='weight')
            
            # Separate memory nodes from user nodes
            memory_pagerank = {}
            for node, score in pagerank_scores.items():
                # Check if node is a memory ID (assuming memory IDs don't contain '@')
                if '@' not in str(node):  # Simple heuristic
                    memory_pagerank[node] = score
            
            # Calculate additional popularity metrics
            self.popularity_scores = {}
            
            for memory_id in memory_pagerank:
                # Get neighbors (users who accessed this memory)
                neighbors = list(self.access_graph.neighbors(memory_id))
                user_neighbors = [n for n in neighbors if '@' in str(n)]  # Assume user IDs contain '@'
                
                # Local popularity (within user network)
                local_pop = len(user_neighbors) / max(1, len(self.user_models))
                
                # Global popularity (PageRank score)
                global_pop = memory_pagerank[memory_id]
                
                # Trend score (recent vs historical access)
                trend_score = await self._calculate_trend_score(memory_id)
                
                # Final combined score
                final_score = (0.4 * global_pop + 0.3 * local_pop + 0.3 * trend_score)
                
                popularity = PopularityScore(
                    memory_id=memory_id,
                    pagerank_score=global_pop,
                    local_popularity=local_pop,
                    global_popularity=global_pop,
                    trend_score=trend_score,
                    final_score=final_score
                )
                
                self.popularity_scores[memory_id] = popularity
                
        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")
    
    async def _calculate_trend_score(self, memory_id: str) -> float:
        """Calculate trend score for memory popularity."""
        pattern = self.memory_patterns.get(memory_id)
        if not pattern or len(pattern.access_times) < 4:
            return 0.5  # Neutral trend
        
        # Compare recent vs older accesses
        total_accesses = len(pattern.access_times)
        recent_count = total_accesses // 2
        
        recent_accesses = pattern.access_times[-recent_count:]
        older_accesses = pattern.access_times[:-recent_count]
        
        # Calculate access rates
        if not older_accesses:
            return 1.0  # All accesses are recent
        
        recent_duration = (recent_accesses[-1] - recent_accesses[0]).total_seconds() / 3600
        older_duration = (older_accesses[-1] - older_accesses[0]).total_seconds() / 3600
        
        if recent_duration <= 0 or older_duration <= 0:
            return 0.5
        
        recent_rate = len(recent_accesses) / recent_duration
        older_rate = len(older_accesses) / older_duration
        
        # Trend score based on rate change
        if older_rate == 0:
            return 1.0
        
        trend_ratio = recent_rate / older_rate
        trend_score = min(1.0, max(0.0, (trend_ratio - 0.5) / 1.5 + 0.5))
        
        return trend_score
    
    async def _generate_predictions(self, patterns: Dict[str, UsagePattern]):
        """Generate usage predictions for patterns."""
        for memory_id, pattern in patterns.items():
            access_pattern = self.memory_patterns.get(memory_id)
            if not access_pattern:
                continue
            
            # Predict next access time
            next_access_prediction = await self._predict_next_access(access_pattern)
            if next_access_prediction:
                pattern.predictions['next_access_time'] = next_access_prediction.isoformat()
                pattern.predictions['next_access_confidence'] = min(pattern.confidence, 0.8)
            
            # Predict access probability in next 24 hours
            prob_24h = await self._predict_access_probability(access_pattern, hours=24)
            pattern.predictions['access_probability_24h'] = prob_24h
            
            # Predict likely users for next access
            likely_users = await self._predict_likely_users(access_pattern)
            pattern.predictions['likely_next_users'] = likely_users[:5]  # Top 5
    
    async def _predict_next_access(self, pattern: AccessPattern) -> Optional[datetime]:
        """Predict next access time for a memory."""
        if len(pattern.time_between_accesses) < 2:
            return None
        
        # Use exponential smoothing for prediction
        intervals = np.array(pattern.time_between_accesses)
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        smoothed = [intervals[0]]
        
        for i in range(1, len(intervals)):
            smoothed_value = alpha * intervals[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)
        
        # Predict next interval
        predicted_interval = smoothed[-1]
        
        # Add to last access time
        next_access = pattern.last_access + timedelta(hours=predicted_interval)
        
        return next_access
    
    async def _predict_access_probability(self, pattern: AccessPattern, hours: int) -> float:
        """Predict probability of access within specified hours."""
        if not pattern.time_between_accesses:
            return 0.1  # Low default probability
        
        mean_interval = np.mean(pattern.time_between_accesses)
        
        # Use exponential distribution for modeling inter-arrival times
        lambda_param = 1.0 / mean_interval if mean_interval > 0 else 0.01
        
        # Probability of access within the specified hours
        probability = 1 - math.exp(-lambda_param * hours)
        
        return min(1.0, probability)
    
    async def _predict_likely_users(self, pattern: AccessPattern) -> List[str]:
        """Predict users likely to access this memory next."""
        user_access_counts = defaultdict(int)
        
        # Count recent accesses by user from logs
        cutoff_time = datetime.utcnow() - timedelta(days=7)  # Last week
        
        for log in self.access_logs:
            if (log['memory_id'] == pattern.memory_id and 
                log['access_time'] >= cutoff_time):
                user_access_counts[log['user_id']] += 1
        
        # Sort by access frequency
        sorted_users = sorted(
            user_access_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [user_id for user_id, _ in sorted_users]
    
    async def _incremental_analysis(self):
        """Perform incremental analysis on new data."""
        # Simple incremental update - could be enhanced
        if len(self.access_logs) % 1000 == 0:  # Every 1000 accesses
            await self.analyze_patterns(force_full_analysis=False)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage analysis statistics."""
        return {
            **self.stats,
            'cached_patterns': len(self.pattern_cache),
            'active_memory_patterns': len(self.memory_patterns),
            'user_models': len(self.user_models),
            'popularity_scores': len(self.popularity_scores),
            'graph_nodes': self.access_graph.number_of_nodes(),
            'graph_edges': self.access_graph.number_of_edges(),
            'last_analysis': self.last_analysis.isoformat() if self.last_analysis else None
        }
    
    async def get_memory_insights(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive insights for a specific memory."""
        usage_pattern = self.pattern_cache.get(memory_id)
        access_pattern = self.memory_patterns.get(memory_id)
        popularity = self.popularity_scores.get(memory_id)
        
        if not usage_pattern:
            return None
        
        insights = {
            'memory_id': memory_id,
            'pattern_type': usage_pattern.pattern_type.value,
            'access_frequency': usage_pattern.access_frequency.value,
            'confidence': usage_pattern.confidence,
            'metrics': usage_pattern.metrics,
            'predictions': usage_pattern.predictions
        }
        
        if access_pattern:
            insights.update({
                'total_accesses': access_pattern.access_count,
                'unique_users': len(access_pattern.user_ids),
                'first_access': access_pattern.first_access.isoformat(),
                'last_access': access_pattern.last_access.isoformat()
            })
        
        if popularity:
            insights.update({
                'popularity_score': popularity.final_score,
                'pagerank_score': popularity.pagerank_score,
                'trend_score': popularity.trend_score
            })
        
        return insights