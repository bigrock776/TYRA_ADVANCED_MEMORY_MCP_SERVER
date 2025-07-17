"""
Query Optimization Engine for Enhanced Performance.

This module provides comprehensive query optimization capabilities using local algorithms,
including SQL parsing, index recommendations, query rewriting, and performance tracking.
All processing is performed locally with zero external API calls.
"""

import asyncio
import time
import re
import hashlib
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import json

# SQL parsing and optimization imports - all local
import sqlparse
from sqlparse import sql, tokens
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import psutil

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class QueryType(str, Enum):
    """Types of SQL queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    UNKNOWN = "unknown"


class OptimizationStrategy(str, Enum):
    """Query optimization strategies."""
    INDEX_RECOMMENDATION = "index_recommendation"
    QUERY_REWRITING = "query_rewriting"
    JOIN_OPTIMIZATION = "join_optimization"
    WHERE_CLAUSE_OPTIMIZATION = "where_clause_optimization"
    SUBQUERY_OPTIMIZATION = "subquery_optimization"
    LIMIT_PUSHDOWN = "limit_pushdown"
    PROJECTION_PUSHDOWN = "projection_pushdown"
    PREDICATE_PUSHDOWN = "predicate_pushdown"
    UNION_OPTIMIZATION = "union_optimization"
    AGGREGATION_OPTIMIZATION = "aggregation_optimization"


class PerformanceMetric(str, Enum):
    """Performance metrics for query optimization."""
    EXECUTION_TIME = "execution_time"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    IO_OPERATIONS = "io_operations"
    ROWS_EXAMINED = "rows_examined"
    ROWS_RETURNED = "rows_returned"
    INDEX_USAGE = "index_usage"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class QueryPattern:
    """Represents a query pattern for optimization."""
    pattern_id: str
    query_template: str
    frequency: int = 0
    avg_execution_time: float = 0.0
    tables_involved: List[str] = field(default_factory=list)
    columns_accessed: List[str] = field(default_factory=list)
    join_types: List[str] = field(default_factory=list)
    where_conditions: List[str] = field(default_factory=list)
    optimization_potential: float = 0.0


@dataclass
class IndexRecommendation:
    """Index recommendation for optimization."""
    table_name: str
    columns: List[str]
    index_type: str = "btree"
    estimated_benefit: float = 0.0
    creation_cost: float = 0.0
    maintenance_cost: float = 0.0
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class QueryOptimization:
    """Query optimization result."""
    original_query: str
    optimized_query: str
    optimization_strategies: List[OptimizationStrategy]
    estimated_improvement: float
    index_recommendations: List[IndexRecommendation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profile for a query."""
    query_hash: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    cpu_usage_stats: Dict[str, float] = field(default_factory=dict)
    memory_usage_stats: Dict[str, float] = field(default_factory=dict)
    io_stats: Dict[str, float] = field(default_factory=dict)
    last_execution: Optional[datetime] = None
    performance_trend: str = "stable"  # improving, degrading, stable


class QueryOptimizer:
    """Advanced Query Optimization Engine."""

    def __init__(
        self,
        cache: Optional[RedisCache] = None,
        optimization_threshold: float = 0.1,
        max_cached_queries: int = 10000
    ):
        self.cache = cache
        self.optimization_threshold = optimization_threshold
        self.max_cached_queries = max_cached_queries
        
        # Query analysis and caching
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.optimization_history: deque = deque(maxlen=1000)
        
        # Optimization rules and strategies
        self.optimization_rules = self._initialize_optimization_rules()
        self.index_recommendations_cache: Dict[str, List[IndexRecommendation]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, deque] = {
            metric.value: deque(maxlen=1000) for metric in PerformanceMetric
        }
        
        # ML model for performance prediction
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.model_trained = False
        
        logger.info("Query Optimizer initialized", max_cached_queries=max_cached_queries)

    def _initialize_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize query optimization rules."""
        return {
            "avoid_select_star": {
                "pattern": r"SELECT\s+\*\s+FROM",
                "replacement": lambda match, query: self._suggest_explicit_columns(query),
                "description": "Replace SELECT * with explicit column list",
                "priority": 0.8
            },
            "add_where_conditions": {
                "pattern": r"SELECT.*FROM\s+(\w+)(?!\s+WHERE)",
                "replacement": lambda match, query: self._suggest_where_conditions(query),
                "description": "Add WHERE conditions to reduce result set",
                "priority": 0.9
            },
            "optimize_joins": {
                "pattern": r"JOIN\s+(\w+)\s+ON",
                "replacement": lambda match, query: self._optimize_join_order(query),
                "description": "Optimize JOIN order and conditions",
                "priority": 0.7
            },
            "add_limits": {
                "pattern": r"SELECT.*FROM.*(?!LIMIT)",
                "replacement": lambda match, query: self._suggest_limit_clause(query),
                "description": "Add LIMIT clause to prevent large result sets",
                "priority": 0.6
            },
            "optimize_subqueries": {
                "pattern": r"\(\s*SELECT.*\)",
                "replacement": lambda match, query: self._optimize_subqueries(query),
                "description": "Convert subqueries to JOINs where possible",
                "priority": 0.8
            },
            "optimize_union": {
                "pattern": r"UNION\s+SELECT",
                "replacement": lambda match, query: self._optimize_union(query),
                "description": "Optimize UNION operations",
                "priority": 0.5
            },
            "optimize_group_by": {
                "pattern": r"GROUP\s+BY\s+(\w+)",
                "replacement": lambda match, query: self._optimize_group_by(query),
                "description": "Optimize GROUP BY operations",
                "priority": 0.6
            },
            "optimize_order_by": {
                "pattern": r"ORDER\s+BY\s+(\w+)",
                "replacement": lambda match, query: self._optimize_order_by(query),
                "description": "Optimize ORDER BY operations",
                "priority": 0.5
            },
            "eliminate_redundant_conditions": {
                "pattern": r"WHERE.*AND.*AND",
                "replacement": lambda match, query: self._eliminate_redundant_conditions(query),
                "description": "Remove redundant WHERE conditions",
                "priority": 0.4
            },
            "use_exists_instead_of_in": {
                "pattern": r"IN\s*\(\s*SELECT",
                "replacement": lambda match, query: self._convert_in_to_exists(query),
                "description": "Convert IN subqueries to EXISTS",
                "priority": 0.7
            },
            "optimize_like_operations": {
                "pattern": r"LIKE\s*'%.*%'",
                "replacement": lambda match, query: self._optimize_like_operations(query),
                "description": "Optimize LIKE operations with wildcards",
                "priority": 0.6
            },
            "add_covering_indexes": {
                "pattern": r"SELECT\s+(.*?)\s+FROM\s+(\w+).*WHERE\s+(.*)",
                "replacement": lambda match, query: self._suggest_covering_indexes(query),
                "description": "Suggest covering indexes for queries",
                "priority": 0.8
            },
            "optimize_case_statements": {
                "pattern": r"CASE\s+WHEN",
                "replacement": lambda match, query: self._optimize_case_statements(query),
                "description": "Optimize CASE statements",
                "priority": 0.5
            },
            "partition_pruning": {
                "pattern": r"FROM\s+(\w+)\s+WHERE.*date",
                "replacement": lambda match, query: self._suggest_partition_pruning(query),
                "description": "Suggest partition pruning optimizations",
                "priority": 0.7
            },
            "materialized_view_suggestions": {
                "pattern": r"SELECT.*FROM.*GROUP\s+BY",
                "replacement": lambda match, query: self._suggest_materialized_views(query),
                "description": "Suggest materialized views for complex aggregations",
                "priority": 0.6
            }
        }

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query and extract structural information."""
        try:
            # Parse the query
            parsed = sqlparse.parse(query)[0]
            
            # Extract query components
            analysis = {
                "query_type": self._extract_query_type(parsed),
                "tables": self._extract_tables(parsed),
                "columns": self._extract_columns(parsed),
                "joins": self._extract_joins(parsed),
                "where_conditions": self._extract_where_conditions(parsed),
                "aggregations": self._extract_aggregations(parsed),
                "subqueries": self._extract_subqueries(parsed),
                "complexity_score": self._calculate_complexity_score(parsed),
                "estimated_cost": self._estimate_query_cost(parsed)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}", query=query[:100])
            return {"error": str(e)}

    async def optimize_query(self, query: str, optimization_level: str = "standard") -> QueryOptimization:
        """Optimize a SQL query using various strategies."""
        try:
            start_time = time.time()
            
            # Analyze the original query
            analysis = await self.analyze_query(query)
            
            # Apply optimization rules
            optimized_query = query
            applied_strategies = []
            total_improvement = 0.0
            warnings = []
            
            # Apply optimization rules based on level
            rules_to_apply = self._get_rules_for_level(optimization_level)
            
            for rule_name, rule_config in rules_to_apply.items():
                try:
                    pattern = rule_config["pattern"]
                    if re.search(pattern, query, re.IGNORECASE):
                        optimization_result = rule_config["replacement"](None, optimized_query)
                        
                        if optimization_result and optimization_result != optimized_query:
                            optimized_query = optimization_result
                            applied_strategies.append(OptimizationStrategy(rule_name))
                            total_improvement += rule_config["priority"]
                            
                except Exception as e:
                    warnings.append(f"Failed to apply {rule_name}: {str(e)}")
                    continue
            
            # Generate index recommendations
            index_recommendations = await self._generate_index_recommendations(analysis)
            
            # Calculate estimated improvement
            estimated_improvement = min(total_improvement * 0.1, 0.9)  # Cap at 90% improvement
            
            optimization = QueryOptimization(
                original_query=query,
                optimized_query=optimized_query,
                optimization_strategies=applied_strategies,
                estimated_improvement=estimated_improvement,
                index_recommendations=index_recommendations,
                warnings=warnings,
                metadata={
                    "analysis": analysis,
                    "optimization_time_ms": (time.time() - start_time) * 1000,
                    "optimization_level": optimization_level
                }
            )
            
            # Cache the optimization
            if self.cache:
                query_hash = hashlib.md5(query.encode()).hexdigest()
                await self.cache.set(
                    f"query_optimization:{query_hash}",
                    optimization.__dict__,
                    expire=3600
                )
            
            return optimization
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}", query=query[:100])
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_strategies=[],
                estimated_improvement=0.0,
                warnings=[f"Optimization failed: {str(e)}"]
            )

    async def track_query_performance(
        self,
        query: str,
        execution_time: float,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None
    ):
        """Track performance metrics for a query."""
        try:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            # Update performance profile
            if query_hash not in self.performance_profiles:
                self.performance_profiles[query_hash] = PerformanceProfile(query_hash=query_hash)
            
            profile = self.performance_profiles[query_hash]
            profile.execution_count += 1
            profile.total_execution_time += execution_time
            profile.avg_execution_time = profile.total_execution_time / profile.execution_count
            profile.min_execution_time = min(profile.min_execution_time, execution_time)
            profile.max_execution_time = max(profile.max_execution_time, execution_time)
            profile.last_execution = datetime.utcnow()
            
            # Update CPU usage stats
            if cpu_usage is not None:
                if not profile.cpu_usage_stats:
                    profile.cpu_usage_stats = {"min": cpu_usage, "max": cpu_usage, "avg": cpu_usage}
                else:
                    profile.cpu_usage_stats["min"] = min(profile.cpu_usage_stats["min"], cpu_usage)
                    profile.cpu_usage_stats["max"] = max(profile.cpu_usage_stats["max"], cpu_usage)
                    profile.cpu_usage_stats["avg"] = (profile.cpu_usage_stats["avg"] + cpu_usage) / 2
            
            # Update memory usage stats
            if memory_usage is not None:
                if not profile.memory_usage_stats:
                    profile.memory_usage_stats = {"min": memory_usage, "max": memory_usage, "avg": memory_usage}
                else:
                    profile.memory_usage_stats["min"] = min(profile.memory_usage_stats["min"], memory_usage)
                    profile.memory_usage_stats["max"] = max(profile.memory_usage_stats["max"], memory_usage)
                    profile.memory_usage_stats["avg"] = (profile.memory_usage_stats["avg"] + memory_usage) / 2
            
            # Determine performance trend
            if profile.execution_count >= 10:
                recent_times = [execution_time] + [profile.avg_execution_time] * 9
                trend_slope = np.polyfit(range(len(recent_times)), recent_times, 1)[0]
                
                if trend_slope < -0.1:
                    profile.performance_trend = "improving"
                elif trend_slope > 0.1:
                    profile.performance_trend = "degrading"
                else:
                    profile.performance_trend = "stable"
            
            # Track global metrics
            self.performance_metrics[PerformanceMetric.EXECUTION_TIME.value].append(execution_time)
            if cpu_usage is not None:
                self.performance_metrics[PerformanceMetric.CPU_USAGE.value].append(cpu_usage)
            if memory_usage is not None:
                self.performance_metrics[PerformanceMetric.MEMORY_USAGE.value].append(memory_usage)
            
            # Clean up old profiles to prevent memory bloat
            if len(self.performance_profiles) > self.max_cached_queries:
                # Remove least recently used profiles
                sorted_profiles = sorted(
                    self.performance_profiles.items(),
                    key=lambda x: x[1].last_execution or datetime.min
                )
                profiles_to_remove = len(self.performance_profiles) - self.max_cached_queries
                for query_hash, _ in sorted_profiles[:profiles_to_remove]:
                    del self.performance_profiles[query_hash]
            
        except Exception as e:
            logger.error(f"Failed to track query performance: {e}")

    async def get_performance_insights(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Filter recent profiles
            recent_profiles = {
                query_hash: profile
                for query_hash, profile in self.performance_profiles.items()
                if profile.last_execution and profile.last_execution >= cutoff_time
            }
            
            # Calculate insights
            insights = {
                "summary": {
                    "total_queries_analyzed": len(recent_profiles),
                    "avg_execution_time": np.mean([p.avg_execution_time for p in recent_profiles.values()]) if recent_profiles else 0,
                    "queries_with_degrading_performance": len([p for p in recent_profiles.values() if p.performance_trend == "degrading"]),
                    "queries_with_improving_performance": len([p for p in recent_profiles.values() if p.performance_trend == "improving"])
                },
                "slow_queries": [],
                "optimization_opportunities": [],
                "trending_patterns": {},
                "resource_usage": {
                    "avg_cpu_usage": 0.0,
                    "avg_memory_usage": 0.0,
                    "peak_cpu_usage": 0.0,
                    "peak_memory_usage": 0.0
                }
            }
            
            # Identify slow queries
            if recent_profiles:
                execution_times = [p.avg_execution_time for p in recent_profiles.values()]
                slow_threshold = np.percentile(execution_times, 90)
                
                slow_queries = [
                    {
                        "query_hash": query_hash,
                        "avg_execution_time": profile.avg_execution_time,
                        "execution_count": profile.execution_count,
                        "performance_trend": profile.performance_trend
                    }
                    for query_hash, profile in recent_profiles.items()
                    if profile.avg_execution_time >= slow_threshold
                ]
                
                insights["slow_queries"] = sorted(slow_queries, key=lambda x: x["avg_execution_time"], reverse=True)[:10]
            
            # Generate optimization opportunities
            optimization_opportunities = []
            for query_hash, profile in recent_profiles.items():
                if profile.avg_execution_time > 1.0:  # Queries taking more than 1 second
                    optimization_opportunities.append({
                        "query_hash": query_hash,
                        "issue": "High execution time",
                        "current_avg_time": profile.avg_execution_time,
                        "recommendation": "Consider adding indexes or optimizing query structure",
                        "priority": "high" if profile.avg_execution_time > 5.0 else "medium"
                    })
                
                if profile.performance_trend == "degrading":
                    optimization_opportunities.append({
                        "query_hash": query_hash,
                        "issue": "Degrading performance",
                        "current_avg_time": profile.avg_execution_time,
                        "recommendation": "Investigate recent changes or data growth",
                        "priority": "medium"
                    })
            
            insights["optimization_opportunities"] = optimization_opportunities[:20]
            
            # Calculate resource usage
            if recent_profiles:
                cpu_usages = [
                    profile.cpu_usage_stats.get("avg", 0)
                    for profile in recent_profiles.values()
                    if profile.cpu_usage_stats
                ]
                memory_usages = [
                    profile.memory_usage_stats.get("avg", 0)
                    for profile in recent_profiles.values()
                    if profile.memory_usage_stats
                ]
                
                if cpu_usages:
                    insights["resource_usage"]["avg_cpu_usage"] = np.mean(cpu_usages)
                    insights["resource_usage"]["peak_cpu_usage"] = np.max(cpu_usages)
                
                if memory_usages:
                    insights["resource_usage"]["avg_memory_usage"] = np.mean(memory_usages)
                    insights["resource_usage"]["peak_memory_usage"] = np.max(memory_usages)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate performance insights: {e}")
            return {"error": str(e)}

    async def generate_system_recommendations(self) -> List[Dict[str, Any]]:
        """Generate system-wide optimization recommendations."""
        try:
            recommendations = []
            
            # Analyze query patterns
            if self.performance_profiles:
                # Identify frequently executed slow queries
                frequent_slow_queries = [
                    (query_hash, profile)
                    for query_hash, profile in self.performance_profiles.items()
                    if profile.execution_count > 100 and profile.avg_execution_time > 0.5
                ]
                
                if frequent_slow_queries:
                    recommendations.append({
                        "type": "query_optimization",
                        "priority": "high",
                        "title": "Optimize Frequently Executed Slow Queries",
                        "description": f"Found {len(frequent_slow_queries)} queries that are executed frequently but perform slowly",
                        "action": "Review and optimize these queries or add appropriate indexes",
                        "affected_queries": len(frequent_slow_queries),
                        "estimated_impact": "high"
                    })
                
                # Check for queries without indexes
                queries_needing_indexes = []
                for query_hash, profile in self.performance_profiles.items():
                    if profile.avg_execution_time > 1.0 and query_hash in self.index_recommendations_cache:
                        if len(self.index_recommendations_cache[query_hash]) > 0:
                            queries_needing_indexes.append(query_hash)
                
                if queries_needing_indexes:
                    recommendations.append({
                        "type": "index_optimization",
                        "priority": "medium",
                        "title": "Add Missing Indexes",
                        "description": f"Found {len(queries_needing_indexes)} queries that could benefit from additional indexes",
                        "action": "Review index recommendations and create appropriate indexes",
                        "affected_queries": len(queries_needing_indexes),
                        "estimated_impact": "medium"
                    })
                
                # Check for resource usage patterns
                high_cpu_queries = [
                    profile for profile in self.performance_profiles.values()
                    if profile.cpu_usage_stats and profile.cpu_usage_stats.get("avg", 0) > 80
                ]
                
                if high_cpu_queries:
                    recommendations.append({
                        "type": "resource_optimization",
                        "priority": "medium",
                        "title": "Optimize High CPU Usage Queries",
                        "description": f"Found {len(high_cpu_queries)} queries with high CPU usage",
                        "action": "Review query complexity and consider query rewriting or caching",
                        "affected_queries": len(high_cpu_queries),
                        "estimated_impact": "medium"
                    })
                
                # Check performance trends
                degrading_queries = [
                    profile for profile in self.performance_profiles.values()
                    if profile.performance_trend == "degrading"
                ]
                
                if degrading_queries:
                    recommendations.append({
                        "type": "performance_monitoring",
                        "priority": "medium",
                        "title": "Monitor Degrading Query Performance",
                        "description": f"Found {len(degrading_queries)} queries with degrading performance",
                        "action": "Investigate data growth, schema changes, or system resource constraints",
                        "affected_queries": len(degrading_queries),
                        "estimated_impact": "medium"
                    })
            
            # System-level recommendations
            if len(self.performance_profiles) > self.max_cached_queries * 0.8:
                recommendations.append({
                    "type": "system_configuration",
                    "priority": "low",
                    "title": "Increase Query Cache Size",
                    "description": "Query cache is approaching capacity",
                    "action": "Consider increasing max_cached_queries parameter",
                    "estimated_impact": "low"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate system recommendations: {e}")
            return []

    # Helper methods for optimization rules
    def _suggest_explicit_columns(self, query: str) -> str:
        """Suggest explicit columns instead of SELECT *."""
        # This is a simplified implementation
        # In practice, you'd need to analyze the table schema
        return re.sub(r'SELECT\s+\*', 'SELECT id, name, created_at', query, flags=re.IGNORECASE)

    def _suggest_where_conditions(self, query: str) -> str:
        """Suggest WHERE conditions to reduce result set."""
        # Add a comment suggesting WHERE clause
        if "WHERE" not in query.upper():
            return query + " -- Consider adding WHERE clause to filter results"
        return query

    def _optimize_join_order(self, query: str) -> str:
        """Optimize JOIN order and conditions."""
        # Simplified optimization: suggest reviewing join order
        return query + " -- Consider optimizing JOIN order based on table sizes"

    def _suggest_limit_clause(self, query: str) -> str:
        """Suggest LIMIT clause for large result sets."""
        if "LIMIT" not in query.upper() and "SELECT" in query.upper():
            return query + " LIMIT 1000"
        return query

    def _optimize_subqueries(self, query: str) -> str:
        """Optimize subqueries by converting to JOINs where possible."""
        # Simplified: add comment about subquery optimization
        return query + " -- Consider converting subqueries to JOINs for better performance"

    def _optimize_union(self, query: str) -> str:
        """Optimize UNION operations."""
        # Convert UNION to UNION ALL where appropriate
        return re.sub(r'UNION\s+SELECT', 'UNION ALL SELECT', query, flags=re.IGNORECASE)

    def _optimize_group_by(self, query: str) -> str:
        """Optimize GROUP BY operations."""
        return query + " -- Consider adding indexes on GROUP BY columns"

    def _optimize_order_by(self, query: str) -> str:
        """Optimize ORDER BY operations."""
        return query + " -- Consider adding indexes on ORDER BY columns"

    def _eliminate_redundant_conditions(self, query: str) -> str:
        """Remove redundant WHERE conditions."""
        # Simplified implementation
        return query

    def _convert_in_to_exists(self, query: str) -> str:
        """Convert IN subqueries to EXISTS."""
        # Simplified: add comment about EXISTS optimization
        return query + " -- Consider using EXISTS instead of IN for better performance"

    def _optimize_like_operations(self, query: str) -> str:
        """Optimize LIKE operations with wildcards."""
        return query + " -- Consider full-text search for LIKE operations with leading wildcards"

    def _suggest_covering_indexes(self, query: str) -> str:
        """Suggest covering indexes for queries."""
        return query + " -- Consider creating covering indexes for better performance"

    def _optimize_case_statements(self, query: str) -> str:
        """Optimize CASE statements."""
        return query + " -- Consider optimizing CASE statements or using lookup tables"

    def _suggest_partition_pruning(self, query: str) -> str:
        """Suggest partition pruning optimizations."""
        return query + " -- Consider partition pruning for date-based queries"

    def _suggest_materialized_views(self, query: str) -> str:
        """Suggest materialized views for complex aggregations."""
        return query + " -- Consider creating materialized views for complex aggregations"

    def _get_rules_for_level(self, level: str) -> Dict[str, Dict[str, Any]]:
        """Get optimization rules based on optimization level."""
        if level == "basic":
            return {k: v for k, v in self.optimization_rules.items() if v["priority"] >= 0.7}
        elif level == "aggressive":
            return self.optimization_rules
        else:  # standard
            return {k: v for k, v in self.optimization_rules.items() if v["priority"] >= 0.5}

    # Query analysis helper methods
    def _extract_query_type(self, parsed) -> QueryType:
        """Extract the type of SQL query."""
        try:
            first_token = str(parsed.tokens[0]).upper().strip()
            if first_token.startswith('SELECT'):
                return QueryType.SELECT
            elif first_token.startswith('INSERT'):
                return QueryType.INSERT
            elif first_token.startswith('UPDATE'):
                return QueryType.UPDATE
            elif first_token.startswith('DELETE'):
                return QueryType.DELETE
            elif first_token.startswith('CREATE'):
                return QueryType.CREATE
            elif first_token.startswith('DROP'):
                return QueryType.DROP
            elif first_token.startswith('ALTER'):
                return QueryType.ALTER
            else:
                return QueryType.UNKNOWN
        except:
            return QueryType.UNKNOWN

    def _extract_tables(self, parsed) -> List[str]:
        """Extract table names from parsed query."""
        tables = []
        try:
            # This is a simplified implementation
            # A full implementation would need to handle various SQL constructs
            query_str = str(parsed).upper()
            
            # Look for FROM clauses
            from_matches = re.findall(r'FROM\s+(\w+)', query_str)
            tables.extend(from_matches)
            
            # Look for JOIN clauses
            join_matches = re.findall(r'JOIN\s+(\w+)', query_str)
            tables.extend(join_matches)
            
            # Look for UPDATE clauses
            update_matches = re.findall(r'UPDATE\s+(\w+)', query_str)
            tables.extend(update_matches)
            
            # Look for INSERT INTO clauses
            insert_matches = re.findall(r'INSERT\s+INTO\s+(\w+)', query_str)
            tables.extend(insert_matches)
            
        except Exception as e:
            logger.warning(f"Failed to extract tables: {e}")
        
        return list(set(tables))  # Remove duplicates

    def _extract_columns(self, parsed) -> List[str]:
        """Extract column names from parsed query."""
        columns = []
        try:
            query_str = str(parsed)
            
            # This is a simplified implementation
            # Look for column names in SELECT clauses
            select_matches = re.findall(r'SELECT\s+(.*?)\s+FROM', query_str, re.IGNORECASE | re.DOTALL)
            for match in select_matches:
                if '*' not in match:
                    # Split by comma and clean up
                    cols = [col.strip() for col in match.split(',')]
                    columns.extend(cols)
            
        except Exception as e:
            logger.warning(f"Failed to extract columns: {e}")
        
        return columns

    def _extract_joins(self, parsed) -> List[str]:
        """Extract JOIN information from parsed query."""
        joins = []
        try:
            query_str = str(parsed).upper()
            
            join_patterns = [
                r'INNER\s+JOIN',
                r'LEFT\s+JOIN',
                r'RIGHT\s+JOIN',
                r'FULL\s+JOIN',
                r'CROSS\s+JOIN',
                r'JOIN'
            ]
            
            for pattern in join_patterns:
                matches = re.findall(pattern, query_str)
                joins.extend(matches)
                
        except Exception as e:
            logger.warning(f"Failed to extract joins: {e}")
        
        return joins

    def _extract_where_conditions(self, parsed) -> List[str]:
        """Extract WHERE conditions from parsed query."""
        conditions = []
        try:
            query_str = str(parsed)
            
            # Look for WHERE clauses
            where_matches = re.findall(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', 
                                    query_str, re.IGNORECASE | re.DOTALL)
            
            for match in where_matches:
                # Split by AND/OR and clean up
                parts = re.split(r'\s+(?:AND|OR)\s+', match, flags=re.IGNORECASE)
                conditions.extend([part.strip() for part in parts])
                
        except Exception as e:
            logger.warning(f"Failed to extract WHERE conditions: {e}")
        
        return conditions

    def _extract_aggregations(self, parsed) -> List[str]:
        """Extract aggregation functions from parsed query."""
        aggregations = []
        try:
            query_str = str(parsed).upper()
            
            agg_patterns = [
                r'COUNT\s*\([^)]*\)',
                r'SUM\s*\([^)]*\)',
                r'AVG\s*\([^)]*\)',
                r'MIN\s*\([^)]*\)',
                r'MAX\s*\([^)]*\)',
                r'GROUP_CONCAT\s*\([^)]*\)'
            ]
            
            for pattern in agg_patterns:
                matches = re.findall(pattern, query_str)
                aggregations.extend(matches)
                
        except Exception as e:
            logger.warning(f"Failed to extract aggregations: {e}")
        
        return aggregations

    def _extract_subqueries(self, parsed) -> List[str]:
        """Extract subqueries from parsed query."""
        subqueries = []
        try:
            query_str = str(parsed)
            
            # Look for subqueries in parentheses
            subquery_matches = re.findall(r'\(\s*(SELECT.*?)\)', query_str, re.IGNORECASE | re.DOTALL)
            subqueries.extend(subquery_matches)
            
        except Exception as e:
            logger.warning(f"Failed to extract subqueries: {e}")
        
        return subqueries

    def _calculate_complexity_score(self, parsed) -> float:
        """Calculate a complexity score for the query."""
        try:
            query_str = str(parsed).upper()
            score = 0.0
            
            # Base score for query type
            if 'SELECT' in query_str:
                score += 1.0
            if 'INSERT' in query_str or 'UPDATE' in query_str or 'DELETE' in query_str:
                score += 2.0
            
            # Add score for joins
            join_count = len(re.findall(r'JOIN', query_str))
            score += join_count * 0.5
            
            # Add score for subqueries
            subquery_count = len(re.findall(r'\(\s*SELECT', query_str))
            score += subquery_count * 1.0
            
            # Add score for aggregations
            agg_count = len(re.findall(r'COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT', query_str))
            score += agg_count * 0.3
            
            # Add score for UNION operations
            union_count = len(re.findall(r'UNION', query_str))
            score += union_count * 0.7
            
            return min(score, 10.0)  # Cap at 10.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate complexity score: {e}")
            return 1.0

    def _estimate_query_cost(self, parsed) -> float:
        """Estimate the computational cost of the query."""
        try:
            complexity = self._calculate_complexity_score(parsed)
            
            # Simple cost estimation based on complexity
            # In a real implementation, this would consider:
            # - Table sizes
            # - Index availability
            # - Join selectivity
            # - Hardware resources
            
            base_cost = complexity * 10.0
            
            # Adjust based on specific operations
            query_str = str(parsed).upper()
            
            if 'ORDER BY' in query_str:
                base_cost *= 1.2  # Sorting adds cost
            
            if 'GROUP BY' in query_str:
                base_cost *= 1.3  # Grouping adds cost
            
            if 'DISTINCT' in query_str:
                base_cost *= 1.1  # Deduplication adds cost
            
            return base_cost
            
        except Exception as e:
            logger.warning(f"Failed to estimate query cost: {e}")
            return 10.0

    async def _generate_index_recommendations(self, analysis: Dict[str, Any]) -> List[IndexRecommendation]:
        """Generate index recommendations based on query analysis."""
        recommendations = []
        
        try:
            tables = analysis.get("tables", [])
            where_conditions = analysis.get("where_conditions", [])
            joins = analysis.get("joins", [])
            
            # Recommend indexes for WHERE clause columns
            for table in tables:
                for condition in where_conditions:
                    # Extract column names from conditions (simplified)
                    column_matches = re.findall(r'(\w+)\s*[=<>!]', condition)
                    for column in column_matches:
                        recommendations.append(IndexRecommendation(
                            table_name=table,
                            columns=[column],
                            index_type="btree",
                            estimated_benefit=0.7,
                            creation_cost=0.3,
                            maintenance_cost=0.1,
                            reasoning=f"WHERE clause condition on {column}",
                            confidence=0.8
                        ))
            
            # Recommend indexes for JOIN columns
            if len(tables) > 1 and joins:
                for table in tables:
                    recommendations.append(IndexRecommendation(
                        table_name=table,
                        columns=["id"],  # Simplified assumption
                        index_type="btree",
                        estimated_benefit=0.8,
                        creation_cost=0.2,
                        maintenance_cost=0.05,
                        reasoning="JOIN operation optimization",
                        confidence=0.9
                    ))
            
            # Remove duplicate recommendations
            unique_recommendations = []
            seen = set()
            for rec in recommendations:
                key = f"{rec.table_name}:{':'.join(rec.columns)}"
                if key not in seen:
                    seen.add(key)
                    unique_recommendations.append(rec)
            
            return unique_recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate index recommendations: {e}")
            return []

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            current_time = datetime.utcnow()
            
            metrics = {
                "timestamp": current_time.isoformat(),
                "query_optimizer": {
                    "total_queries_analyzed": len(self.performance_profiles),
                    "avg_optimization_time_ms": 0.0,
                    "optimization_success_rate": 0.0,
                    "cache_hit_rate": 0.0
                },
                "performance_tracking": {
                    "avg_query_execution_time": 0.0,
                    "slow_query_count": 0,
                    "queries_with_degrading_performance": 0,
                    "optimization_opportunities": 0
                },
                "resource_usage": {
                    "avg_cpu_usage": 0.0,
                    "avg_memory_usage": 0.0,
                    "peak_cpu_usage": 0.0,
                    "peak_memory_usage": 0.0
                },
                "index_recommendations": {
                    "total_recommendations": len(self.index_recommendations_cache),
                    "high_impact_recommendations": 0,
                    "estimated_total_benefit": 0.0
                }
            }
            
            if self.performance_profiles:
                # Calculate performance metrics
                execution_times = [p.avg_execution_time for p in self.performance_profiles.values()]
                metrics["performance_tracking"]["avg_query_execution_time"] = np.mean(execution_times)
                metrics["performance_tracking"]["slow_query_count"] = len([t for t in execution_times if t > 1.0])
                metrics["performance_tracking"]["queries_with_degrading_performance"] = len([
                    p for p in self.performance_profiles.values() if p.performance_trend == "degrading"
                ])
                
                # Calculate resource usage
                cpu_usages = [
                    p.cpu_usage_stats.get("avg", 0) for p in self.performance_profiles.values()
                    if p.cpu_usage_stats
                ]
                memory_usages = [
                    p.memory_usage_stats.get("avg", 0) for p in self.performance_profiles.values()
                    if p.memory_usage_stats
                ]
                
                if cpu_usages:
                    metrics["resource_usage"]["avg_cpu_usage"] = np.mean(cpu_usages)
                    metrics["resource_usage"]["peak_cpu_usage"] = np.max(cpu_usages)
                
                if memory_usages:
                    metrics["resource_usage"]["avg_memory_usage"] = np.mean(memory_usages)
                    metrics["resource_usage"]["peak_memory_usage"] = np.max(memory_usages)
            
            # Calculate index recommendation metrics
            all_recommendations = []
            for recs in self.index_recommendations_cache.values():
                all_recommendations.extend(recs)
            
            if all_recommendations:
                metrics["index_recommendations"]["high_impact_recommendations"] = len([
                    r for r in all_recommendations if r.estimated_benefit > 0.7
                ])
                metrics["index_recommendations"]["estimated_total_benefit"] = sum(
                    r.estimated_benefit for r in all_recommendations
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}


# Export main class
__all__ = ["QueryOptimizer", "QueryOptimization", "IndexRecommendation", "PerformanceProfile"]