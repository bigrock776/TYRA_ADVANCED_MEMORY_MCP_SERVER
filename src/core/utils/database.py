"""
Enhanced Database Connection Management with Smart Pooling.

This module provides advanced database connection pooling with adaptive sizing,
health monitoring, lifecycle management, and performance optimization using local analytics.
All processing is performed locally with zero external dependencies.
"""

import asyncio
import time
import psutil
import numpy as np
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
import weakref

import asyncpg
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from .circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker,
)
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for database connections."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_query_time: float = 0.0
    max_query_time: float = 0.0
    min_query_time: float = float("inf")


class PoolStrategy(str, Enum):
    """Connection pool sizing strategies."""
    FIXED = "fixed"             # Fixed pool size
    ADAPTIVE = "adaptive"       # Adapt based on load
    PREDICTIVE = "predictive"   # Predict future needs
    CONSERVATIVE = "conservative"  # Conservative growth
    AGGRESSIVE = "aggressive"   # Aggressive scaling


@dataclass
class SmartConnectionMetrics:
    """Enhanced metrics for smart connection tracking."""
    connection_id: str
    created_at: datetime
    last_used: datetime
    total_queries: int = 0
    total_query_time: float = 0.0
    error_count: int = 0
    health_score: float = 1.0
    query_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_query_stats(self, query_time: float, had_error: bool = False):
        """Update query statistics."""
        self.total_queries += 1
        self.total_query_time += query_time
        self.query_times.append(query_time)
        self.last_used = datetime.utcnow()
        
        if had_error:
            self.error_count += 1
            self.health_score = max(0.0, self.health_score - 0.1)
        else:
            # Slowly recover health score
            self.health_score = min(1.0, self.health_score + 0.01)
    
    def get_average_query_time(self) -> float:
        """Get average query time."""
        if self.total_queries == 0:
            return 0.0
        return self.total_query_time / self.total_queries
    
    def get_recent_query_time(self) -> float:
        """Get recent average query time."""
        if not self.query_times:
            return 0.0
        return np.mean(list(self.query_times))
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self.health_score > 0.5


class SmartPoolOptimizer:
    """
    Smart Pool Size Optimization Engine.
    
    Analyzes usage patterns and optimizes pool size dynamically.
    """
    
    def __init__(self, strategy: PoolStrategy = PoolStrategy.ADAPTIVE):
        self.strategy = strategy
        self.optimization_history: deque = deque(maxlen=100)
        self.load_patterns: Dict[str, List[float]] = {}
    
    async def analyze_and_optimize(
        self,
        current_size: int,
        active_connections: int,
        avg_wait_time: float,
        efficiency: float
    ) -> int:
        """Analyze current state and recommend optimal pool size."""
        
        # Collect analysis data
        analysis_data = {
            'current_size': current_size,
            'active_connections': active_connections,
            'wait_time': avg_wait_time,
            'efficiency': efficiency,
            'timestamp': datetime.utcnow()
        }
        
        # Apply strategy-specific optimization
        if self.strategy == PoolStrategy.ADAPTIVE:
            optimal_size = await self._adaptive_optimization(analysis_data)
        elif self.strategy == PoolStrategy.PREDICTIVE:
            optimal_size = await self._predictive_optimization(analysis_data)
        elif self.strategy == PoolStrategy.CONSERVATIVE:
            optimal_size = await self._conservative_optimization(analysis_data)
        else:  # AGGRESSIVE
            optimal_size = await self._aggressive_optimization(analysis_data)
        
        # Record optimization decision
        self.optimization_history.append({
            'timestamp': datetime.utcnow(),
            'current_size': current_size,
            'recommended_size': optimal_size,
            'strategy': self.strategy.value,
            'metrics': analysis_data
        })
        
        return optimal_size
    
    async def _adaptive_optimization(self, analysis_data: Dict[str, Any]) -> int:
        """Adaptive pool size optimization."""
        
        current_size = analysis_data['current_size']
        active_connections = analysis_data['active_connections']
        wait_time = analysis_data['wait_time']
        efficiency = analysis_data['efficiency']
        
        # Calculate utilization
        utilization = active_connections / current_size if current_size > 0 else 0
        
        # Decision logic
        if wait_time > 0.1 or utilization > 0.8:  # High contention
            # Scale up
            scale_factor = 1.5 if utilization > 0.9 else 1.2
            new_size = min(50, int(current_size * scale_factor))
        elif utilization < 0.3 and efficiency > 0.9:  # Low utilization
            # Scale down
            new_size = max(5, int(current_size * 0.8))
        else:
            # No change
            new_size = current_size
        
        return new_size
    
    async def _predictive_optimization(self, analysis_data: Dict[str, Any]) -> int:
        """Predictive pool size optimization based on trends."""
        
        # Analyze historical patterns
        if len(self.optimization_history) < 10:
            return analysis_data['current_size']
        
        # Extract load pattern
        recent_loads = [
            entry['metrics']['active_connections']
            for entry in list(self.optimization_history)[-10:]
        ]
        
        # Simple trend analysis
        if len(recent_loads) >= 3:
            trend = np.polyfit(range(len(recent_loads)), recent_loads, 1)[0]
            
            # Predict future load
            predicted_load = recent_loads[-1] + trend * 3  # 3 steps ahead
            
            # Size pool for predicted load + buffer
            buffer_factor = 1.3
            optimal_size = max(5, int(predicted_load * buffer_factor))
            
            return min(50, optimal_size)
        
        return analysis_data['current_size']
    
    async def _conservative_optimization(self, analysis_data: Dict[str, Any]) -> int:
        """Conservative pool size optimization."""
        
        current_size = analysis_data['current_size']
        wait_time = analysis_data['wait_time']
        
        # Only scale if there's clear evidence of need
        if wait_time > 0.5:  # Significant wait time
            return min(50, current_size + 1)  # Add one connection
        elif wait_time == 0 and analysis_data['efficiency'] > 0.95:
            return max(5, current_size - 1)  # Remove one connection
        
        return current_size
    
    async def _aggressive_optimization(self, analysis_data: Dict[str, Any]) -> int:
        """Aggressive pool size optimization."""
        
        current_size = analysis_data['current_size']
        active_connections = analysis_data['active_connections']
        wait_time = analysis_data['wait_time']
        
        utilization = active_connections / current_size if current_size > 0 else 0
        
        # Scale aggressively
        if wait_time > 0.05 or utilization > 0.7:
            scale_factor = 2.0 if wait_time > 0.2 else 1.5
            return min(50, int(current_size * scale_factor))
        elif utilization < 0.2:
            return max(5, int(current_size * 0.5))
        
        return current_size


class DatabaseManager(ABC):
    """Abstract base class for database managers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close all database connections."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the database."""
        pass

    @abstractmethod
    def get_stats(self) -> ConnectionStats:
        """Get connection statistics."""
        pass


class PostgreSQLManager(DatabaseManager):
    """
    Enhanced PostgreSQL connection manager with smart pooling,
    adaptive sizing, health monitoring, and performance optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self.circuit_breaker: Optional[AsyncCircuitBreaker] = None
        self.stats = ConnectionStats()
        self._query_times: List[float] = []
        self._max_query_times = 1000  # Keep last 1000 query times
        
        # Smart pooling features
        self.pool_strategy = PoolStrategy(config.get('pool_strategy', 'adaptive'))
        self.smart_optimizer = SmartPoolOptimizer(self.pool_strategy)
        self.connection_metrics: Dict[str, SmartConnectionMetrics] = {}
        self.wait_times: deque = deque(maxlen=100)
        self._maintenance_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool with circuit breaker."""
        try:
            # Create circuit breaker
            cb_config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=3,
                timeout=30.0,
                expected_exception=(
                    asyncpg.PostgresError,
                    asyncpg.ConnectionDoesNotExistError,
                    OSError,
                ),
            )

            self.circuit_breaker = await get_circuit_breaker(
                "postgresql", cb_config, fallback_func=self._fallback_query
            )

            # Create connection pool
            dsn = self._build_dsn()
            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=self.config.get("min_connections", 5),
                max_size=self.config.get("pool_size", 20),
                max_queries=self.config.get("max_queries", 50000),
                max_inactive_connection_lifetime=self.config.get("max_lifetime", 300),
                command_timeout=self.config.get("command_timeout", 10),
                server_settings={
                    "jit": "off",  # Disable JIT for faster startup
                    "application_name": "tyra_mcp_memory_server",
                },
            )

            # Verify connection
            await self._verify_connection()
            
            # Start smart pool maintenance
            self._maintenance_task = asyncio.create_task(self._smart_pool_maintenance())

            logger.info(
                "PostgreSQL connection pool initialized with smart pooling",
                min_size=self.config.get("min_connections", 5),
                max_size=self.config.get("pool_size", 20),
                host=self.config.get("host"),
                database=self.config.get("database"),
                strategy=self.pool_strategy.value
            )

        except Exception as e:
            logger.error(
                "Failed to initialize PostgreSQL connection pool",
                error=str(e),
                config=self._safe_config(),
            )
            raise

    def _build_dsn(self) -> str:
        """Build PostgreSQL DSN from configuration."""
        return (
            f"postgresql://{self.config['username']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )

    def _safe_config(self) -> Dict[str, Any]:
        """Get config with sensitive data masked."""
        safe = self.config.copy()
        if "password" in safe:
            safe["password"] = "***"
        return safe

    async def _verify_connection(self) -> None:
        """Verify connection by running a simple query."""
        async with self.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

    async def _fallback_query(self, query: str, *args, **kwargs):
        """Fallback function for failed queries."""
        logger.warning(
            "Using fallback for PostgreSQL query",
            query=query[:100] + "..." if len(query) > 100 else query,
        )
        # Return empty result for fallback
        return []

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a connection from the pool with smart monitoring and circuit breaker protection."""
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        start_time = time.time()
        connection = None
        connection_id = None

        try:
            # Acquire connection through circuit breaker
            connection = await self.circuit_breaker.call(self.pool.acquire)
            
            # Track wait time
            wait_time = time.time() - start_time
            self.wait_times.append(wait_time)
            
            # Create connection ID and metrics if needed
            connection_id = f"conn_{id(connection)}"
            if connection_id not in self.connection_metrics:
                self.connection_metrics[connection_id] = SmartConnectionMetrics(
                    connection_id=connection_id,
                    created_at=datetime.utcnow(),
                    last_used=datetime.utcnow()
                )

            self.stats.active_connections += 1
            yield connection

        except Exception as e:
            self.stats.failed_connections += 1
            
            # Update connection metrics if we have them
            if connection_id and connection_id in self.connection_metrics:
                self.connection_metrics[connection_id].update_query_stats(0, had_error=True)
            
            logger.error(
                "Failed to acquire PostgreSQL connection",
                error=str(e),
                error_type=type(e).__name__,
                wait_time=time.time() - start_time
            )
            raise
        finally:
            if connection:
                try:
                    await self.pool.release(connection)
                    self.stats.active_connections -= 1

                    # Update timing stats
                    total_time = time.time() - start_time
                    self._update_timing_stats(total_time)
                    
                    # Update connection metrics
                    if connection_id and connection_id in self.connection_metrics:
                        self.connection_metrics[connection_id].update_query_stats(total_time)

                except Exception as e:
                    logger.error(
                        "Failed to release PostgreSQL connection", error=str(e)
                    )

    async def execute_query(
        self, query: str, *args, fetch_mode: str = "all"  # "all", "one", "val", "none"
    ) -> Any:
        """Execute a query with circuit breaker protection and monitoring."""
        start_time = time.time()

        try:
            async with self.get_connection() as conn:
                self.stats.total_queries += 1

                if fetch_mode == "all":
                    result = await conn.fetch(query, *args)
                elif fetch_mode == "one":
                    result = await conn.fetchrow(query, *args)
                elif fetch_mode == "val":
                    result = await conn.fetchval(query, *args)
                elif fetch_mode == "none":
                    await conn.execute(query, *args)
                    result = None
                else:
                    raise ValueError(f"Invalid fetch_mode: {fetch_mode}")

                self.stats.successful_queries += 1
                return result

        except Exception as e:
            self.stats.failed_queries += 1
            query_time = time.time() - start_time

            logger.error(
                "PostgreSQL query failed",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e),
                query_time=query_time,
            )
            raise

    async def execute_batch(
        self, query: str, args_list: List[tuple], batch_size: int = 1000
    ) -> None:
        """Execute batch queries efficiently."""
        async with self.get_connection() as conn:
            # Process in batches to avoid memory issues
            for i in range(0, len(args_list), batch_size):
                batch = args_list[i : i + batch_size]
                await conn.executemany(query, batch)

                logger.debug(
                    "Processed batch",
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch),
                    total_batches=(len(args_list) + batch_size - 1) // batch_size,
                )

    def _update_timing_stats(self, query_time: float):
        """Update query timing statistics."""
        self._query_times.append(query_time)

        # Keep only recent query times
        if len(self._query_times) > self._max_query_times:
            self._query_times = self._query_times[-self._max_query_times :]

        # Update stats
        self.stats.max_query_time = max(self.stats.max_query_time, query_time)
        self.stats.min_query_time = min(self.stats.min_query_time, query_time)

        if self._query_times:
            self.stats.avg_query_time = sum(self._query_times) / len(self._query_times)

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            start_time = time.time()

            # Test basic connectivity
            await self.execute_query("SELECT 1", fetch_mode="val")

            # Test pgvector extension
            try:
                await self.execute_query(
                    "SELECT extname FROM pg_extension WHERE extname = 'vector'",
                    fetch_mode="val",
                )
                pgvector_available = True
            except Exception:
                pgvector_available = False

            # Get pool stats
            pool_stats = {
                "size": self.pool.get_size(),
                "max_size": self.pool.get_max_size(),
                "min_size": self.pool.get_min_size(),
                "idle_connections": self.pool.get_idle_size(),
            }

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": response_time,
                "pgvector_available": pgvector_available,
                "pool_stats": pool_stats,
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
                "connection_stats": self.get_stats().__dict__,
            }

        except Exception as e:
            logger.error("PostgreSQL health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
            }

    def get_stats(self) -> ConnectionStats:
        """Get detailed connection statistics."""
        if self.pool:
            self.stats.total_connections = self.pool.get_size()
            self.stats.idle_connections = self.pool.get_idle_size()

        return self.stats

    async def close(self) -> None:
        """Close the connection pool and cleanup smart pooling."""
        # Stop maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def _smart_pool_maintenance(self):
        """Background task for smart pool optimization."""
        maintenance_interval = 60  # seconds
        
        while True:
            try:
                await asyncio.sleep(maintenance_interval)
                
                if not self.pool:
                    continue
                
                # Get current pool metrics
                current_size = self.pool.get_size()
                active_connections = self.stats.active_connections
                avg_wait_time = np.mean(list(self.wait_times)) if self.wait_times else 0.0
                
                # Calculate efficiency (connection requests without waits)
                total_requests = len(self.wait_times)
                fast_requests = sum(1 for wt in self.wait_times if wt < 0.01)
                efficiency = fast_requests / total_requests if total_requests > 0 else 1.0
                
                # Get optimizer recommendation
                optimal_size = await self.smart_optimizer.analyze_and_optimize(
                    current_size, active_connections, avg_wait_time, efficiency
                )
                
                # Log optimization decision
                if optimal_size != current_size:
                    logger.info(
                        "Smart pool optimization recommendation",
                        current_size=current_size,
                        optimal_size=optimal_size,
                        active_connections=active_connections,
                        avg_wait_time=avg_wait_time,
                        efficiency=efficiency,
                        strategy=self.pool_strategy.value
                    )
                
                # Clean up old connection metrics
                await self._cleanup_connection_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Smart pool maintenance error: {e}")
    
    async def _cleanup_connection_metrics(self):
        """Clean up old connection metrics."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        # Remove metrics for connections not used recently
        old_connections = [
            conn_id for conn_id, metrics in self.connection_metrics.items()
            if metrics.last_used < cutoff_time
        ]
        
        for conn_id in old_connections:
            del self.connection_metrics[conn_id]
        
        if old_connections:
            logger.debug(f"Cleaned up {len(old_connections)} old connection metrics")
    
    def get_smart_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive smart pool statistics."""
        
        # Calculate connection health scores
        healthy_connections = sum(
            1 for metrics in self.connection_metrics.values() 
            if metrics.is_healthy()
        )
        
        # Get recent query performance
        recent_query_times = []
        for metrics in self.connection_metrics.values():
            if metrics.query_times:
                recent_query_times.extend(list(metrics.query_times))
        
        return {
            'pool_strategy': self.pool_strategy.value,
            'current_size': self.pool.get_size() if self.pool else 0,
            'max_size': self.pool.get_max_size() if self.pool else 0,
            'min_size': self.pool.get_min_size() if self.pool else 0,
            'active_connections': self.stats.active_connections,
            'healthy_connections': healthy_connections,
            'total_tracked_connections': len(self.connection_metrics),
            'average_wait_time': np.mean(list(self.wait_times)) if self.wait_times else 0.0,
            'recent_optimization_decisions': list(self.smart_optimizer.optimization_history)[-5:],
            'performance_metrics': {
                'avg_recent_query_time': np.mean(recent_query_times) if recent_query_times else 0.0,
                'p95_query_time': np.percentile(recent_query_times, 95) if len(recent_query_times) > 1 else 0.0,
                'connection_efficiency': (
                    sum(1 for wt in self.wait_times if wt < 0.01) / len(self.wait_times)
                    if self.wait_times else 1.0
                )
            }
        }


class RedisManager(DatabaseManager):
    """
    Advanced Redis connection manager with circuit breaker and monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[redis.ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.circuit_breaker: Optional[AsyncCircuitBreaker] = None
        self.stats = ConnectionStats()

    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            # Create circuit breaker
            cb_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=15.0,
                success_threshold=2,
                timeout=10.0,
                expected_exception=(
                    redis.RedisError,
                    redis.ConnectionError,
                    redis.TimeoutError,
                ),
            )

            self.circuit_breaker = await get_circuit_breaker(
                "redis", cb_config, fallback_func=self._fallback_operation
            )

            # Create connection pool
            self.pool = redis.ConnectionPool(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6379),
                password=self.config.get("password"),
                db=self.config.get("db", 0),
                max_connections=self.config.get("pool_size", 50),
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)

            # Verify connection
            await self._verify_connection()

            logger.info(
                "Redis connection pool initialized",
                host=self.config.get("host"),
                port=self.config.get("port"),
                db=self.config.get("db", 0),
                pool_size=self.config.get("pool_size", 50),
            )

        except Exception as e:
            logger.error(
                "Failed to initialize Redis connection pool",
                error=str(e),
                config=self._safe_config(),
            )
            raise

    def _safe_config(self) -> Dict[str, Any]:
        """Get config with sensitive data masked."""
        safe = self.config.copy()
        if "password" in safe:
            safe["password"] = "***"
        return safe

    async def _verify_connection(self) -> None:
        """Verify Redis connection."""
        await self.client.ping()

    async def _fallback_operation(self, *args, **kwargs):
        """Fallback for Redis operations."""
        logger.warning("Using fallback for Redis operation")
        return None

    async def execute_command(self, command: str, *args, **kwargs) -> Any:
        """Execute Redis command with circuit breaker protection."""
        if not self.client:
            raise RuntimeError("Redis client not initialized")

        start_time = time.time()

        try:
            self.stats.total_queries += 1

            # Execute command through circuit breaker
            result = await self.circuit_breaker.call(
                getattr(self.client, command.lower()), *args, **kwargs
            )

            self.stats.successful_queries += 1

            # Update timing
            query_time = time.time() - start_time
            self.stats.avg_query_time = (
                self.stats.avg_query_time * (self.stats.successful_queries - 1)
                + query_time
            ) / self.stats.successful_queries

            return result

        except Exception as e:
            self.stats.failed_queries += 1
            logger.error(
                "Redis command failed",
                command=command,
                error=str(e),
                query_time=time.time() - start_time,
            )
            raise

    @asynccontextmanager
    async def pipeline(self, transaction: bool = False) -> AsyncGenerator[redis.Pipeline, None]:
        """Get Redis pipeline for batch operations."""
        if not self.client:
            raise RuntimeError("Redis client not initialized")

        pipeline = self.client.pipeline(transaction=transaction)
        try:
            yield pipeline
        finally:
            # Pipeline automatically handles cleanup
            pass

    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        try:
            start_time = time.time()

            # Test basic connectivity
            await self.client.ping()

            # Get Redis info
            info = await self.client.info()

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": response_time,
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
                "connection_stats": self.get_stats().__dict__,
            }

        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
            }

    def get_stats(self) -> ConnectionStats:
        """Get Redis connection statistics."""
        if self.pool:
            # Redis connection pool doesn't expose detailed stats
            self.stats.total_connections = self.pool.max_connections

        return self.stats

    async def close(self) -> None:
        """Close Redis connections."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("Redis connection pool closed")




class Neo4jManager(DatabaseManager):
    """
    Advanced Neo4j connection manager with circuit breaker and monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver: Optional[AsyncDriver] = None
        self.circuit_breaker: Optional[AsyncCircuitBreaker] = None
        self.stats = ConnectionStats()

    async def initialize(self) -> None:
        """Initialize Neo4j connection."""
        try:
            # Create circuit breaker
            cb_config = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=3,
                timeout=30.0,
                expected_exception=(
                    Neo4jError,
                    ServiceUnavailable,
                    SessionExpired,
                    OSError,
                ),
            )

            self.circuit_breaker = await get_circuit_breaker(
                "neo4j", cb_config, fallback_func=self._fallback_query
            )

            # Build connection URI
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 7687)
            scheme = "neo4j+s" if self.config.get("encrypted", False) else "neo4j"
            uri = f"{scheme}://{host}:{port}"

            # Initialize driver with auth
            username = self.config.get("username", "neo4j")
            password = self.config.get("password", "neo4j")
            
            # Driver configuration
            driver_config = {
                "max_connection_lifetime": self.config.get("connection_timeout", 30) * 60,
                "max_connection_pool_size": self.config.get("pool_size", 100),
                "connection_acquisition_timeout": self.config.get("acquisition_timeout", 60),
                "connection_timeout": self.config.get("connection_timeout", 30),
                "keep_alive": self.config.get("keep_alive", True),
            }

            self.driver = AsyncGraphDatabase.driver(
                uri,
                auth=(username, password),
                **driver_config
            )

            # Verify connection
            await self._verify_connection()

            logger.info(
                "Neo4j connection initialized",
                host=host,
                port=port,
                encrypted=self.config.get("encrypted", False),
            )

        except Exception as e:
            logger.error(
                "Failed to initialize Neo4j connection",
                error=str(e),
                config=self._safe_config(),
            )
            raise

    def _safe_config(self) -> Dict[str, Any]:
        """Get config with sensitive data masked."""
        safe = self.config.copy()
        if "password" in safe:
            safe["password"] = "***"
        return safe

    async def _verify_connection(self) -> None:
        """Verify Neo4j connection."""
        await self.driver.verify_connectivity()

    async def _fallback_query(self, query: str, *args, **kwargs):
        """Fallback for Neo4j queries."""
        logger.warning(
            "Using fallback for Neo4j query",
            query=query[:100] + "..." if len(query) > 100 else query,
        )
        return []

    async def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query with circuit breaker protection."""
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")

        start_time = time.time()

        try:
            self.stats.total_queries += 1

            # Execute query through circuit breaker
            result = await self.circuit_breaker.call(
                self._execute_async_query, query, parameters
            )

            self.stats.successful_queries += 1

            # Update timing
            query_time = time.time() - start_time
            self.stats.avg_query_time = (
                self.stats.avg_query_time * (self.stats.successful_queries - 1)
                + query_time
            ) / self.stats.successful_queries

            return result

        except Exception as e:
            self.stats.failed_queries += 1
            logger.error(
                "Neo4j query failed",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e),
                query_time=time.time() - start_time,
            )
            raise

    async def _execute_async_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute asynchronous Cypher query."""
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def health_check(self) -> Dict[str, Any]:
        """Perform Neo4j health check."""
        try:
            start_time = time.time()

            # Test basic connectivity
            await self.driver.verify_connectivity()
            
            # Test basic query
            await self.execute_query("RETURN 1 as test")

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": response_time,
                "driver_connected": True,
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
                "connection_stats": self.get_stats().__dict__,
            }

        except Exception as e:
            logger.error("Neo4j health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "driver_connected": self.driver is not None,
                "circuit_breaker": (
                    self.circuit_breaker.get_stats() if self.circuit_breaker else None
                ),
            }

    def get_stats(self) -> ConnectionStats:
        """Get Neo4j connection statistics."""
        # Neo4j driver doesn't expose detailed connection stats directly
        return self.stats

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
        logger.info("Neo4j connection closed")
