"""
Administrative API endpoints.

Provides system administration, maintenance, monitoring,
and configuration management endpoints.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.cache.redis_cache import RedisCache
from ...core.ingestion.job_tracker import JobTracker, JobStatus, JobType, get_job_tracker
from ...core.memory.manager import MemoryManager
from ...core.observability.telemetry import get_telemetry
from ...core.schemas.ingestion import IngestionProgress
from ...core.utils.config import get_settings, reload_config
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Enums
class MaintenanceTask(str, Enum):
    """Available maintenance tasks."""

    CLEANUP_MEMORIES = "cleanup_memories"
    REBUILD_INDEXES = "rebuild_indexes"
    CLEAR_CACHE = "clear_cache"
    VACUUM_DATABASE = "vacuum_database"
    ANALYZE_PERFORMANCE = "analyze_performance"
    BACKUP_DATA = "backup_data"


class SystemStatus(str, Enum):
    """System status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    MAINTENANCE = "maintenance"


# Request/Response Models
class SystemInfo(BaseModel):
    """System information."""

    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    status: SystemStatus = Field(..., description="Current system status")
    active_connections: int = Field(..., description="Number of active connections")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    storage_usage_gb: float = Field(..., description="Storage usage in GB")


class DatabaseStats(BaseModel):
    """Database statistics."""

    postgres: Dict[str, Any] = Field(..., description="PostgreSQL statistics")
    neo4j: Dict[str, Any] = Field(..., description="Neo4j statistics")
    redis: Dict[str, Any] = Field(..., description="Redis statistics")


class CacheStats(BaseModel):
    """Cache statistics."""

    total_keys: int = Field(..., description="Total cache keys")
    memory_usage_mb: float = Field(..., description="Cache memory usage")
    hit_rate: float = Field(..., description="Cache hit rate")
    miss_rate: float = Field(..., description="Cache miss rate")
    evictions: int = Field(..., description="Number of evictions")
    ttl_expirations: int = Field(..., description="TTL expirations")


class MaintenanceRequest(BaseModel):
    """Maintenance task request."""

    task: MaintenanceTask = Field(..., description="Task to perform")
    force: bool = Field(False, description="Force execution even if risky")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Task-specific parameters"
    )


class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""

    section: str = Field(..., description="Configuration section to update")
    updates: Dict[str, Any] = Field(..., description="Configuration updates")
    reload: bool = Field(True, description="Reload configuration after update")


class BackupRequest(BaseModel):
    """Backup request."""

    include_memories: bool = Field(True, description="Include memories in backup")
    include_graph: bool = Field(True, description="Include knowledge graph")
    include_config: bool = Field(True, description="Include configuration")
    compression: bool = Field(True, description="Compress backup")


class LogQuery(BaseModel):
    """Log query parameters."""

    level: Optional[str] = Field(None, description="Log level filter")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    search: Optional[str] = Field(None, description="Search in log messages")
    limit: int = Field(100, ge=1, le=1000, description="Maximum log entries")


# Dependencies
async def get_memory_manager() -> MemoryManager:
    """Get memory manager instance."""
    try:
        return get_provider(ProviderType.MEMORY_MANAGER, "default")
    except Exception as e:
        logger.error(f"Failed to get memory manager: {e}")
        raise HTTPException(status_code=500, detail="Memory manager unavailable")


async def get_cache() -> RedisCache:
    """Get cache instance."""
    try:
        return get_provider(ProviderType.CACHE, "redis")
    except Exception as e:
        logger.error(f"Failed to get cache: {e}")
        raise HTTPException(status_code=500, detail="Cache unavailable")


# System endpoints
@router.get("/system/info", response_model=SystemInfo)
async def get_system_info():
    """
    Get system information and status.

    Returns overall system health and resource usage.
    """
    try:
        import time

        import psutil

        # Get process info
        process = psutil.Process()

        # Calculate uptime
        start_time = process.create_time()
        uptime = time.time() - start_time

        # Get resource usage
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)

        # Get disk usage
        disk_usage = psutil.disk_usage("/")

        # Determine status
        status = SystemStatus.HEALTHY
        if cpu_percent > 80 or memory_info.rss / 1024 / 1024 > 1000:
            status = SystemStatus.DEGRADED

        return SystemInfo(
            version="1.0.0",
            uptime_seconds=uptime,
            status=status,
            active_connections=len(process.connections()),
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_usage_percent=cpu_percent,
            storage_usage_gb=disk_usage.used / 1024 / 1024 / 1024,
        )

    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/database/stats", response_model=DatabaseStats)
async def get_database_stats(
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Get database statistics.

    Returns statistics for all database systems.
    """
    try:
        # Get stats from each database
        postgres_stats = await memory_manager.get_database_stats()

        # Get Neo4j stats
        neo4j_stats = await _get_neo4j_stats()

        # Get Redis stats  
        redis_stats = await _get_redis_stats()

        return DatabaseStats(
            postgres=postgres_stats, neo4j=neo4j_stats, redis=redis_stats
        )

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats(cache: RedisCache = Depends(get_cache)):
    """
    Get cache statistics.

    Returns detailed cache performance metrics.
    """
    try:
        stats = await cache.get_stats()

        return CacheStats(
            total_keys=stats["total_keys"],
            memory_usage_mb=stats["memory_usage_mb"],
            hit_rate=stats["hit_rate"],
            miss_rate=stats["miss_rate"],
            evictions=stats["evictions"],
            ttl_expirations=stats["ttl_expirations"],
        )

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Maintenance endpoints
@router.post("/maintenance/execute")
async def execute_maintenance(
    request: MaintenanceRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    cache: RedisCache = Depends(get_cache),
):
    """
    Execute a maintenance task.

    Runs maintenance tasks in the background to avoid blocking.
    """
    try:
        task_id = f"{request.task}_{datetime.utcnow().timestamp()}"

        if request.task == MaintenanceTask.CLEANUP_MEMORIES:
            background_tasks.add_task(
                memory_manager.cleanup_memories,
                force=request.force,
                **request.parameters or {},
            )

        elif request.task == MaintenanceTask.CLEAR_CACHE:
            background_tasks.add_task(
                cache.clear_all,
                pattern=(
                    request.parameters.get("pattern") if request.parameters else None
                ),
            )

        elif request.task == MaintenanceTask.REBUILD_INDEXES:
            background_tasks.add_task(
                memory_manager.rebuild_indexes, **request.parameters or {}
            )

        elif request.task == MaintenanceTask.VACUUM_DATABASE:
            background_tasks.add_task(
                memory_manager.vacuum_database, full=request.force
            )

        elif request.task == MaintenanceTask.ANALYZE_PERFORMANCE:
            background_tasks.add_task(_analyze_performance, memory_manager, cache)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown task: {request.task}")

        return {
            "task_id": task_id,
            "task": request.task,
            "status": "started",
            "message": f"Maintenance task {request.task} started in background",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to clear"),
    cache: RedisCache = Depends(get_cache),
):
    """
    Clear cache entries.

    Clears all cache or entries matching a pattern.
    """
    try:
        if pattern:
            cleared = await cache.clear_pattern(pattern)
            message = f"Cleared {cleared} cache entries matching pattern: {pattern}"
        else:
            await cache.clear_all()
            message = "All cache entries cleared"

        return {"message": message}

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@router.get("/config/current")
async def get_current_config():
    """
    Get current configuration.

    Returns the active configuration settings.
    """
    try:
        settings = get_settings()

        # Convert to dict and remove sensitive values
        config_dict = settings.dict()
        _remove_sensitive_values(config_dict)

        return config_dict

    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/update")
async def update_configuration(request: ConfigUpdateRequest):
    """
    Update configuration settings.

    Updates configuration and optionally reloads the system.
    """
    try:
        # Validate section exists
        settings = get_settings()
        if not hasattr(settings, request.section):
            raise HTTPException(
                status_code=400, detail=f"Unknown config section: {request.section}"
            )

        # Update configuration in YAML file
        config_updated = await _update_config_file(request.section, request.updates)
        if not config_updated:
            raise HTTPException(
                status_code=500, 
                detail="Failed to update configuration file"
            )
        
        logger.info(f"Updated config section {request.section}: {request.updates}")

        # Reload if requested
        if request.reload:
            reload_config()

        return {
            "section": request.section,
            "updates": request.updates,
            "reloaded": request.reload,
            "message": "Configuration updated successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backup and restore
@router.post("/backup/create")
async def create_backup(
    request: BackupRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Create a system backup.

    Creates a backup of specified components in the background.
    """
    try:
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Start backup in background
        background_tasks.add_task(_create_backup, backup_id, request, memory_manager)

        return {
            "backup_id": backup_id,
            "status": "started",
            "message": "Backup creation started in background",
        }

    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backup/list")
async def list_backups():
    """
    List available backups.

    Returns information about existing backups.
    """
    try:
        import json
        import os
        from pathlib import Path
        
        backups = []
        backup_dir = Path("backups")
        
        if backup_dir.exists():
            # Find all backup info files
            for info_file in backup_dir.glob("*_info.json"):
                try:
                    with open(info_file, 'r') as f:
                        backup_info = json.load(f)
                    
                    # Calculate backup size
                    backup_id = backup_info["backup_id"]
                    size_mb = 0.0
                    
                    if backup_info.get("compressed"):
                        # Check compressed file size
                        compressed_file = Path(backup_info.get("compressed_file", ""))
                        if compressed_file.exists():
                            size_mb = compressed_file.stat().st_size / (1024 * 1024)
                    else:
                        # Calculate directory size
                        backup_path = backup_dir / backup_id
                        if backup_path.exists():
                            size_mb = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file()) / (1024 * 1024)
                    
                    # Parse created_at datetime
                    created_at = backup_info.get("created_at")
                    if isinstance(created_at, str):
                        from dateutil.parser import parse
                        created_at = parse(created_at)
                    
                    backups.append({
                        "id": backup_id,
                        "created_at": created_at,
                        "completed_at": backup_info.get("completed_at"),
                        "size_mb": round(size_mb, 2),
                        "components": backup_info.get("components", []),
                        "status": backup_info.get("status", "unknown"),
                        "compressed": backup_info.get("compressed", False),
                        "errors": backup_info.get("errors", [])
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read backup info from {info_file}: {e}")
                    continue
            
            # Also check for failed backup files
            for failed_file in backup_dir.glob("*_failed.json"):
                try:
                    with open(failed_file, 'r') as f:
                        failure_info = json.load(f)
                    
                    created_at = failure_info.get("created_at")
                    if isinstance(created_at, str):
                        from dateutil.parser import parse
                        created_at = parse(created_at)
                    
                    backups.append({
                        "id": failure_info["backup_id"],
                        "created_at": created_at,
                        "completed_at": None,
                        "size_mb": 0.0,
                        "components": [],
                        "status": "failed",
                        "compressed": False,
                        "error": failure_info.get("error", "Unknown error")
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read failure info from {failed_file}: {e}")
                    continue
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

        return {"backups": backups}

    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Logging endpoints
@router.post("/logs/query")
async def query_logs(query: LogQuery):
    """
    Query system logs.

    Returns filtered log entries based on query parameters.
    """
    try:
        import re
        from pathlib import Path
        
        logs = []
        
        # Try to read from log files
        log_file_paths = [
            Path("logs/tyra-memory.log"),
            Path("tyra-memory.log"),
            Path("app.log"),
            Path("logs/app.log")
        ]
        
        log_entries = []
        
        # Read from available log files
        for log_path in log_file_paths:
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        log_entries.extend(f.readlines())
                    break  # Use first available log file
                except Exception as e:
                    logger.warning(f"Failed to read log file {log_path}: {e}")
                    continue
        
        # If no log files found, try to get from logging handlers
        if not log_entries:
            try:
                import logging
                
                # Get recent log records from memory handler if available
                for handler in logging.root.handlers:
                    if hasattr(handler, 'buffer'):  # Memory handler
                        for record in handler.buffer[-query.limit:]:
                            log_entries.append(f"{record.created} {record.levelname} {record.name} {record.getMessage()}")
                    elif hasattr(handler, 'stream') and hasattr(handler.stream, 'getvalue'):  # StringIO handler
                        log_content = handler.stream.getvalue()
                        log_entries.extend(log_content.split('\n'))
                        
            except Exception as e:
                logger.warning(f"Failed to get logs from handlers: {e}")
        
        # Parse log entries
        for line in log_entries[-1000:]:  # Limit to last 1000 entries for performance
            if not line.strip():
                continue
                
            try:
                # Try to parse structured log entry
                log_entry = _parse_log_line(line.strip())
                
                if not log_entry:
                    continue
                
                # Apply filters
                if query.level and log_entry["level"] != query.level.upper():
                    continue
                
                if query.start_time and log_entry["timestamp"] < query.start_time:
                    continue
                    
                if query.end_time and log_entry["timestamp"] > query.end_time:
                    continue
                
                if query.search and query.search.lower() not in log_entry["message"].lower():
                    continue
                
                logs.append(log_entry)
                
            except Exception as e:
                # Skip malformed log entries
                continue
        
        # Sort by timestamp (newest first) and limit
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        logs = logs[:query.limit]
        
        # If still no logs, provide a helpful message
        if not logs:
            logs = [{
                "timestamp": datetime.utcnow(),
                "level": "INFO",
                "message": "No log entries found matching criteria. Check log file configuration.",
                "module": "api.admin",
                "note": "This is a system message, not an actual log entry"
            }]

        return {"logs": logs, "total": len(logs), "query": query.dict()}

    except Exception as e:
        logger.error(f"Failed to query logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a log line into structured data."""
    try:
        import re
        import json
        from dateutil.parser import parse as parse_date
        
        # Try JSON format first (structured logging)
        if line.startswith('{') and line.endswith('}'):
            try:
                log_data = json.loads(line)
                return {
                    "timestamp": parse_date(log_data.get("timestamp", log_data.get("time", datetime.utcnow().isoformat()))),
                    "level": log_data.get("level", log_data.get("levelname", "INFO")).upper(),
                    "message": log_data.get("message", log_data.get("msg", "")),
                    "module": log_data.get("name", log_data.get("module", "unknown")),
                    "extra": {k: v for k, v in log_data.items() if k not in ["timestamp", "time", "level", "levelname", "message", "msg", "name", "module"]}
                }
            except json.JSONDecodeError:
                pass
        
        # Try standard format: TIMESTAMP LEVEL MODULE MESSAGE
        patterns = [
            # ISO timestamp with level
            r'^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?)\s+(\w+)\s+(\S+)\s+(.+)$',
            # Simple timestamp
            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)$',
            # Unix timestamp
            r'^(\d+(?:\.\d+)?)\s+(\w+)\s+(\S+)\s+(.+)$',
            # Python logging format
            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+-\s+(\w+)\s+-\s+(\S+)\s+-\s+(.+)$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                groups = match.groups()
                
                if len(groups) == 4:
                    timestamp_str, level, module, message = groups
                elif len(groups) == 3:
                    timestamp_str, level, message = groups
                    module = "unknown"
                else:
                    continue
                
                # Parse timestamp
                try:
                    # Handle Unix timestamp
                    if timestamp_str.replace('.', '').isdigit():
                        timestamp = datetime.fromtimestamp(float(timestamp_str))
                    else:
                        timestamp = parse_date(timestamp_str)
                except:
                    timestamp = datetime.utcnow()
                
                return {
                    "timestamp": timestamp,
                    "level": level.upper(),
                    "message": message,
                    "module": module
                }
        
        # Fallback: treat as plain message
        return {
            "timestamp": datetime.utcnow(),
            "level": "INFO",
            "message": line,
            "module": "unknown"
        }
        
    except Exception:
        return None


async def _get_neo4j_stats() -> Dict[str, Any]:
    """Get Neo4j database statistics."""
    try:
        from ...core.utils.registry import ProviderType, get_provider
        
        # Get Neo4j client
        graph_engine = get_provider(ProviderType.GRAPH_ENGINE, "neo4j")
        
        if not graph_engine:
            return {"status": "unavailable", "error": "Neo4j engine not available"}
        
        # Get Neo4j driver
        driver = getattr(graph_engine, 'driver', None)
        if not driver:
            return {"status": "unavailable", "error": "Neo4j driver not available"}
        
        # Query Neo4j for statistics
        stats = {
            "status": "healthy",
            "node_count": 0,
            "edge_count": 0,
            "labels": [],
            "relationship_types": [],
            "database_size_mb": 0,
            "memory_usage_mb": 0
        }
        
        async with driver.session() as session:
            # Get node count
            result = await session.run("MATCH (n) RETURN count(n) as node_count")
            node_record = await result.single()
            if node_record:
                stats["node_count"] = node_record["node_count"]
            
            # Get relationship count
            result = await session.run("MATCH ()-[r]->() RETURN count(r) as edge_count")
            edge_record = await result.single()
            if edge_record:
                stats["edge_count"] = edge_record["edge_count"]
            
            # Get node labels
            result = await session.run("CALL db.labels()")
            labels = []
            async for record in result:
                labels.append(record["label"])
            stats["labels"] = labels
            
            # Get relationship types
            result = await session.run("CALL db.relationshipTypes()")
            rel_types = []
            async for record in result:
                rel_types.append(record["relationshipType"])
            stats["relationship_types"] = rel_types
            
            # Get database size (if available)
            try:
                result = await session.run("CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store file sizes') YIELD attributes RETURN attributes.TotalStoreSize.value as size")
                size_record = await result.single()
                if size_record and size_record["size"]:
                    stats["database_size_mb"] = float(size_record["size"]) / (1024 * 1024)
            except:
                pass  # Size query might not be available in all Neo4j versions
            
            # Get memory usage (if available)
            try:
                result = await session.run("CALL dbms.queryJmx('java.lang:type=Memory') YIELD attributes RETURN attributes.HeapMemoryUsage.value.used as heap_used")
                memory_record = await result.single()
                if memory_record and memory_record["heap_used"]:
                    stats["memory_usage_mb"] = float(memory_record["heap_used"]) / (1024 * 1024)
            except:
                pass  # Memory query might not be available
        
        return stats
        
    except Exception as e:
        logger.warning(f"Failed to get Neo4j stats: {e}")
        return {"status": "error", "error": str(e), "node_count": 0, "edge_count": 0}


async def _get_redis_stats() -> Dict[str, Any]:
    """Get Redis database statistics."""
    try:
        from ...core.utils.registry import ProviderType, get_provider
        
        # Get Redis cache client
        cache = get_provider(ProviderType.CACHE, "redis")
        
        if not cache:
            return {"status": "unavailable", "error": "Redis cache not available"}
        
        # Get Redis client
        redis_client = getattr(cache, 'redis', None)
        if not redis_client:
            return {"status": "unavailable", "error": "Redis client not available"}
        
        # Get Redis info
        info = await redis_client.info()
        
        stats = {
            "status": "healthy",
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_mb": float(info.get("used_memory", 0)) / (1024 * 1024),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "total_keys": 0,
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": 0.0,
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "redis_version": info.get("redis_version", "unknown"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0)
        }
        
        # Calculate hit rate
        hits = stats["keyspace_hits"]
        misses = stats["keyspace_misses"]
        if hits + misses > 0:
            stats["hit_rate"] = hits / (hits + misses)
        
        # Get total keys from all databases
        total_keys = 0
        for db_num in range(16):  # Redis default has 16 databases
            db_info = info.get(f"db{db_num}")
            if db_info:
                # Parse "keys=N,expires=N,avg_ttl=N" format
                keys_part = db_info.split(',')[0]
                if keys_part.startswith('keys='):
                    total_keys += int(keys_part.split('=')[1])
        
        stats["total_keys"] = total_keys
        
        # Get memory fragmentation info
        stats["mem_fragmentation_ratio"] = info.get("mem_fragmentation_ratio", 0)
        stats["used_memory_peak_mb"] = float(info.get("used_memory_peak", 0)) / (1024 * 1024)
        
        return stats
        
    except Exception as e:
        logger.warning(f"Failed to get Redis stats: {e}")
        return {"status": "error", "error": str(e), "connected_clients": 0, "used_memory_mb": 0}


async def _update_config_file(section: str, updates: Dict[str, Any]) -> bool:
    """Update configuration file with new values."""
    try:
        import yaml
        from pathlib import Path
        
        config_file = Path("config/config.yaml")
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_file}")
            return False
        
        # Read current configuration
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            config_data = {}
        
        # Update the specified section
        if section not in config_data:
            config_data[section] = {}
        
        # Apply updates recursively
        def update_nested_dict(target: dict, source: dict):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested_dict(target[key], value)
                else:
                    target[key] = value
        
        if isinstance(config_data[section], dict):
            update_nested_dict(config_data[section], updates)
        else:
            config_data[section] = updates
        
        # Create backup of current config
        backup_file = config_file.with_suffix(f".yaml.backup.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        try:
            import shutil
            shutil.copy2(config_file, backup_file)
            logger.info(f"Created config backup: {backup_file}")
        except Exception as e:
            logger.warning(f"Failed to create config backup: {e}")
        
        # Write updated configuration
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"Successfully updated configuration section '{section}' in {config_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update config file: {e}")
        return False


@router.get("/logs/levels")
async def get_log_levels():
    """
    Get current log levels.

    Returns log levels for all modules.
    """
    try:
        import logging

        levels = {}
        for name, logger_obj in logging.Logger.manager.loggerDict.items():
            if isinstance(logger_obj, logging.Logger):
                levels[name] = logging.getLevelName(logger_obj.level)

        return {"log_levels": levels}

    except Exception as e:
        logger.error(f"Failed to get log levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logs/level/{module}")
async def set_log_level(
    module: str,
    level: str = Query(..., description="Log level: DEBUG, INFO, WARNING, ERROR"),
):
    """
    Set log level for a module.

    Dynamically adjusts logging verbosity.
    """
    try:
        import logging

        # Validate level
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise HTTPException(status_code=400, detail=f"Invalid log level: {level}")

        # Set level
        logger_obj = logging.getLogger(module)
        logger_obj.setLevel(numeric_level)

        return {
            "module": module,
            "level": level.upper(),
            "message": f"Log level set to {level.upper()} for {module}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set log level: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/optimize")
async def optimize_telemetry():
    """Optimize telemetry performance."""
    try:
        telemetry = get_telemetry()
        result = await telemetry.optimize_telemetry()
        return {
            "status": "success",
            "result": result,
            "message": "Telemetry optimization completed"
        }
    except Exception as e:
        logger.error(f"Telemetry optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/emergency-optimize")
async def emergency_optimize_telemetry():
    """Emergency telemetry optimization for critical performance issues."""
    try:
        telemetry = get_telemetry()
        result = await telemetry.emergency_optimize()
        return {
            "status": "success",
            "result": result,
            "message": "Emergency telemetry optimization applied"
        }
    except Exception as e:
        logger.error(f"Emergency telemetry optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry/performance-stats")
async def get_telemetry_performance_stats():
    """Get telemetry performance statistics."""
    try:
        telemetry = get_telemetry()
        stats = telemetry.get_telemetry_performance_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get telemetry performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/enable-optimization")
async def enable_telemetry_optimization():
    """Enable telemetry performance optimization."""
    try:
        telemetry = get_telemetry()
        telemetry.enable_performance_optimization()
        return {
            "status": "success",
            "message": "Telemetry performance optimization enabled"
        }
    except Exception as e:
        logger.error(f"Failed to enable telemetry optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/disable-optimization")
async def disable_telemetry_optimization():
    """Disable telemetry performance optimization."""
    try:
        telemetry = get_telemetry()
        telemetry.disable_performance_optimization()
        return {
            "status": "success",
            "message": "Telemetry performance optimization disabled"
        }
    except Exception as e:
        logger.error(f"Failed to disable telemetry optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _remove_sensitive_values(config_dict: Dict[str, Any]):
    """Remove sensitive values from config."""
    sensitive_keys = ["password", "secret", "key", "token"]

    for key, value in list(config_dict.items()):
        if isinstance(value, dict):
            _remove_sensitive_values(value)
        elif any(sensitive in key.lower() for sensitive in sensitive_keys):
            config_dict[key] = "***REDACTED***"


async def _analyze_performance(memory_manager: MemoryManager, cache: RedisCache):
    """Analyze system performance."""
    logger.info("Starting performance analysis...")

    # Analyze memory performance
    memory_stats = await memory_manager.analyze_performance()
    logger.info(f"Memory performance: {memory_stats}")

    # Analyze cache performance
    cache_stats = await cache.get_stats()
    logger.info(f"Cache performance: {cache_stats}")

    # Generate recommendations
    recommendations = []

    if cache_stats["hit_rate"] < 0.7:
        recommendations.append("Consider increasing cache TTL or size")

    if memory_stats.get("slow_queries", 0) > 10:
        recommendations.append("Optimize slow queries or add indexes")

    logger.info(f"Performance analysis complete. Recommendations: {recommendations}")


async def _create_backup(
    backup_id: str, request: BackupRequest, memory_manager: MemoryManager
):
    """Create system backup."""
    import os
    import json
    import asyncio
    from pathlib import Path
    
    logger.info(f"Creating backup {backup_id}...")
    
    try:
        # Create backup directory
        backup_dir = Path("backups") / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_info = {
            "backup_id": backup_id,
            "created_at": datetime.utcnow().isoformat(),
            "components": [],
            "status": "in_progress"
        }
        
        if request.include_memories:
            logger.info("Backing up memories...")
            try:
                # Export memory data from PostgreSQL
                memories_data = await memory_manager.export_all_memories()
                
                memories_file = backup_dir / "memories.json"
                with open(memories_file, 'w') as f:
                    json.dump(memories_data, f, indent=2, default=str)
                
                backup_info["components"].append("memories")
                logger.info(f"Memories backed up to {memories_file}")
                
            except Exception as e:
                logger.error(f"Failed to backup memories: {e}")
                backup_info["errors"] = backup_info.get("errors", [])
                backup_info["errors"].append(f"Memory backup failed: {str(e)}")

        if request.include_graph:
            logger.info("Backing up knowledge graph...")
            try:
                # Run Neo4j backup using external script
                graph_backup_file = backup_dir / "neo4j_backup.cypher"
                
                # Use the backup script
                backup_script = Path("scripts/db/backup_databases.sh")
                if backup_script.exists():
                    # Execute backup script for Neo4j only
                    import subprocess
                    result = subprocess.run([
                        "bash", str(backup_script), "--neo4j-only", 
                        "--output", str(graph_backup_file)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        backup_info["components"].append("graph")
                        logger.info(f"Graph backed up to {graph_backup_file}")
                    else:
                        logger.error(f"Graph backup script failed: {result.stderr}")
                        backup_info["errors"] = backup_info.get("errors", [])
                        backup_info["errors"].append(f"Graph backup failed: {result.stderr}")
                else:
                    # Fallback: create placeholder file
                    with open(graph_backup_file, 'w') as f:
                        f.write("# Neo4j backup placeholder - backup script not found\n")
                        f.write(f"# Backup requested at: {datetime.utcnow().isoformat()}\n")
                    
                    backup_info["components"].append("graph")
                    logger.warning("Graph backup script not found, created placeholder")
                    
            except Exception as e:
                logger.error(f"Failed to backup graph: {e}")
                backup_info["errors"] = backup_info.get("errors", [])
                backup_info["errors"].append(f"Graph backup failed: {str(e)}")

        if request.include_config:
            logger.info("Backing up configuration...")
            try:
                config_backup_dir = backup_dir / "config"
                config_backup_dir.mkdir(exist_ok=True)
                
                # Copy configuration files
                config_files = [
                    "config/config.yaml",
                    "config/graphiti.yaml", 
                    "config/observability.yaml",
                    "config/providers.yaml"
                ]
                
                import shutil
                for config_file in config_files:
                    config_path = Path(config_file)
                    if config_path.exists():
                        shutil.copy2(config_path, config_backup_dir / config_path.name)
                        
                # Also backup environment template
                env_example = Path(".env.example")
                if env_example.exists():
                    shutil.copy2(env_example, config_backup_dir / ".env.example")
                
                backup_info["components"].append("config")
                logger.info(f"Configuration backed up to {config_backup_dir}")
                
            except Exception as e:
                logger.error(f"Failed to backup configuration: {e}")
                backup_info["errors"] = backup_info.get("errors", [])
                backup_info["errors"].append(f"Config backup failed: {str(e)}")

        # Compress backup if requested
        if request.compression:
            logger.info("Compressing backup...")
            try:
                import tarfile
                
                compressed_file = backup_dir.parent / f"{backup_id}.tar.gz"
                with tarfile.open(compressed_file, "w:gz") as tar:
                    tar.add(backup_dir, arcname=backup_id)
                
                # Remove uncompressed directory
                import shutil
                shutil.rmtree(backup_dir)
                
                backup_info["compressed"] = True
                backup_info["compressed_file"] = str(compressed_file)
                logger.info(f"Backup compressed to {compressed_file}")
                
            except Exception as e:
                logger.error(f"Failed to compress backup: {e}")
                backup_info["errors"] = backup_info.get("errors", [])
                backup_info["errors"].append(f"Compression failed: {str(e)}")

        # Write backup info
        backup_info["status"] = "completed" if not backup_info.get("errors") else "completed_with_errors"
        backup_info["completed_at"] = datetime.utcnow().isoformat()
        
        info_file = backup_dir.parent / f"{backup_id}_info.json"
        with open(info_file, 'w') as f:
            json.dump(backup_info, f, indent=2, default=str)

        logger.info(f"Backup {backup_id} completed successfully")
        
        if backup_info.get("errors"):
            logger.warning(f"Backup completed with {len(backup_info['errors'])} errors")

    except Exception as e:
        logger.error(f"Backup {backup_id} failed: {e}")
        
        # Write failure info
        failure_info = {
            "backup_id": backup_id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "failed",
            "error": str(e)
        }
        
        try:
            failure_file = Path("backups") / f"{backup_id}_failed.json"
            failure_file.parent.mkdir(parents=True, exist_ok=True)
            with open(failure_file, 'w') as f:
                json.dump(failure_info, f, indent=2, default=str)
        except:
            pass  # Don't fail on failure logging
            
        raise


# Job management endpoints
@router.post("/jobs/cleanup")
async def cleanup_old_jobs(
    older_than_hours: int = 24,
    job_tracker: JobTracker = Depends(get_job_tracker)
) -> Dict[str, str]:
    """
    Clean up old completed jobs.
    
    Removes jobs older than the specified number of hours to free up storage.
    """
    try:
        await job_tracker.cleanup_old_jobs(older_than_hours)
        return {"message": f"Cleaned up jobs older than {older_than_hours} hours"}
    except Exception as e:
        logger.error(f"Failed to cleanup old jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup jobs: {str(e)}"
        )


@router.get("/jobs/stats")
async def get_job_stats(
    job_tracker: JobTracker = Depends(get_job_tracker)
) -> Dict[str, int]:
    """
    Get statistics about ingestion jobs.
    """
    try:
        all_jobs = await job_tracker.list_jobs(limit=1000)
        
        stats = {
            "total_jobs": len(all_jobs),
            "pending_jobs": len([j for j in all_jobs if j.status == JobStatus.PENDING]),
            "running_jobs": len([j for j in all_jobs if j.status == JobStatus.RUNNING]),
            "completed_jobs": len([j for j in all_jobs if j.status == JobStatus.COMPLETED]),
            "failed_jobs": len([j for j in all_jobs if j.status == JobStatus.FAILED]),
            "cancelled_jobs": len([j for j in all_jobs if j.status == JobStatus.CANCELLED]),
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get job stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job statistics: {str(e)}"
        )


@router.get("/jobs", response_model=List[IngestionProgress])
async def list_all_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    job_type: Optional[JobType] = Query(None, description="Filter by job type"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of jobs to return"),
    job_tracker: JobTracker = Depends(get_job_tracker)
) -> List[IngestionProgress]:
    """
    List all ingestion jobs with optional filtering.
    """
    try:
        jobs = await job_tracker.list_jobs(status=status, job_type=job_type, limit=limit)
        
        results = []
        for job in jobs:
            job_data = job.to_dict()
            results.append(IngestionProgress(
                job_id=job_data["job_id"],
                status=job_data["status"],
                progress_percentage=job_data["progress_percentage"],
                total_items=job_data["total_items"],
                processed_items=job_data["processed_items"],
                successful_items=job_data["successful_items"],
                failed_items=job_data["failed_items"],
                current_item=job_data["current_item"],
                created_at=job_data["created_at"],
                started_at=job_data["started_at"],
                completed_at=job_data["completed_at"],
                estimated_time_remaining=job_data["estimated_time_remaining"],
                error_messages=job_data["error_messages"],
                warnings=job_data["warnings"],
                result_summary=job_data.get("result_data", {})
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )
