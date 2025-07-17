"""
API routes for file watcher service management.

Provides endpoints for monitoring and controlling the file watcher service
that automatically processes files dropped in the tyra-ingest folder.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.services.file_watcher_service import get_file_watcher_manager
from ...core.utils.simple_logger import get_logger
from ..middleware.auth import verify_api_key
from ..middleware.rate_limit import rate_limit

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/file-watcher", tags=["file-watcher"])


class FileWatcherStatus(BaseModel):
    """File watcher service status."""
    enabled: bool = Field(..., description="Whether the service is enabled")
    running: bool = Field(..., description="Whether the service is currently running")
    monitored_path: str = Field(..., description="Path being monitored")
    supported_extensions: list = Field(..., description="Supported file extensions")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: datetime = Field(..., description="Status timestamp")


class FileWatcherStats(BaseModel):
    """File watcher processing statistics."""
    total_files_processed: int = Field(..., description="Total files processed")
    successful_ingestions: int = Field(..., description="Successful ingestions")
    failed_ingestions: int = Field(..., description="Failed ingestions")
    duplicate_files: int = Field(..., description="Duplicate files detected")
    unsupported_files: int = Field(..., description="Unsupported file types")
    success_rate: float = Field(..., description="Success rate (0.0-1.0)")
    last_processed: Optional[datetime] = Field(None, description="Last file processed timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class FileWatcherHealth(BaseModel):
    """File watcher service health status."""
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    is_running: bool = Field(..., description="Whether the service is running")
    observer_alive: bool = Field(..., description="Whether the file observer is alive")
    paths_accessible: Dict[str, bool] = Field(..., description="Path accessibility status")
    warnings: Optional[list] = Field(None, description="Health warnings")
    error: Optional[str] = Field(None, description="Error message if unhealthy")
    timestamp: datetime = Field(..., description="Health check timestamp")


@router.get("/status", response_model=FileWatcherStatus)
@rate_limit(calls=30, period=60)
async def get_file_watcher_status(
    api_key: str = Depends(verify_api_key)
) -> FileWatcherStatus:
    """
    Get the current status of the file watcher service.
    
    Returns information about whether the service is running,
    what path it's monitoring, and basic uptime information.
    """
    try:
        manager = get_file_watcher_manager()
        
        # Get basic status
        is_running = manager.is_running()
        
        # Get detailed stats to extract info
        stats = await manager.get_stats()
        
        return FileWatcherStatus(
            enabled=True,  # If we can access it, it's enabled
            running=is_running,
            monitored_path=stats.get("monitored_path", ""),
            supported_extensions=stats.get("supported_extensions", []),
            uptime_seconds=stats.get("uptime_seconds", 0),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get file watcher status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=FileWatcherStats)
@rate_limit(calls=30, period=60)
async def get_file_watcher_stats(
    api_key: str = Depends(verify_api_key)
) -> FileWatcherStats:
    """
    Get processing statistics for the file watcher service.
    
    Returns detailed statistics about file processing including
    success rates, error counts, and performance metrics.
    """
    try:
        manager = get_file_watcher_manager()
        stats = await manager.get_stats()
        
        return FileWatcherStats(
            total_files_processed=stats.get("total_files_processed", 0),
            successful_ingestions=stats.get("successful_ingestions", 0),
            failed_ingestions=stats.get("failed_ingestions", 0),
            duplicate_files=stats.get("duplicate_files", 0),
            unsupported_files=stats.get("unsupported_files", 0),
            success_rate=stats.get("success_rate", 0.0),
            last_processed=stats.get("last_processed"),
            uptime_seconds=stats.get("uptime_seconds", 0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get file watcher stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=FileWatcherHealth)
@rate_limit(calls=30, period=60)
async def get_file_watcher_health(
    api_key: str = Depends(verify_api_key)
) -> FileWatcherHealth:
    """
    Get health status of the file watcher service.
    
    Returns comprehensive health information including
    service status, path accessibility, and any warnings.
    """
    try:
        manager = get_file_watcher_manager()
        health = await manager.get_health_status()
        
        return FileWatcherHealth(
            status=health.get("status", "unknown"),
            is_running=health.get("is_running", False),
            observer_alive=health.get("observer_alive", False),
            paths_accessible=health.get("paths_accessible", {}),
            warnings=health.get("warnings"),
            error=health.get("error"),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get file watcher health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
@rate_limit(calls=5, period=60)
async def start_file_watcher(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Start the file watcher service.
    
    Starts monitoring the tyra-ingest folder for new files.
    Returns success status and current service state.
    """
    try:
        manager = get_file_watcher_manager()
        
        if manager.is_running():
            return {
                "success": True,
                "message": "File watcher service is already running",
                "status": "running",
                "timestamp": datetime.utcnow()
            }
        
        await manager.start()
        
        return {
            "success": True,
            "message": "File watcher service started successfully",
            "status": "running",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to start file watcher service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
@rate_limit(calls=5, period=60)
async def stop_file_watcher(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Stop the file watcher service.
    
    Stops monitoring the tyra-ingest folder.
    Returns success status and current service state.
    """
    try:
        manager = get_file_watcher_manager()
        
        if not manager.is_running():
            return {
                "success": True,
                "message": "File watcher service is already stopped",
                "status": "stopped",
                "timestamp": datetime.utcnow()
            }
        
        await manager.stop()
        
        return {
            "success": True,
            "message": "File watcher service stopped successfully",
            "status": "stopped",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop file watcher service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart")
@rate_limit(calls=3, period=60)
async def restart_file_watcher(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Restart the file watcher service.
    
    Stops and then starts the file watcher service.
    Useful for applying configuration changes.
    """
    try:
        manager = get_file_watcher_manager()
        
        await manager.restart()
        
        return {
            "success": True,
            "message": "File watcher service restarted successfully",
            "status": "running",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to restart file watcher service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
@rate_limit(calls=10, period=60)
async def get_file_watcher_config(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get the current file watcher configuration.
    
    Returns the configuration settings for the file watcher service
    including paths, supported file types, and processing options.
    """
    try:
        from ...core.utils.simple_config import get_settings
        
        settings = get_settings()
        file_watcher_config = settings.get("file_watcher", {})
        
        return {
            "success": True,
            "config": file_watcher_config,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get file watcher config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
@rate_limit(calls=10, period=60)
async def get_file_watcher_logs(
    api_key: str = Depends(verify_api_key),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of log entries"),
    level: str = Query("INFO", description="Log level filter")
) -> Dict[str, Any]:
    """
    Get recent file watcher service logs.
    
    Returns recent log entries related to file processing activities.
    Useful for monitoring and debugging file ingestion issues.
    """
    try:
        # This is a placeholder implementation
        # In a real implementation, you would retrieve logs from your logging system
        
        return {
            "success": True,
            "logs": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "INFO",
                    "message": "File watcher service logs endpoint accessed",
                    "module": "file_watcher"
                }
            ],
            "total_entries": 1,
            "limit": limit,
            "level_filter": level,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get file watcher logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))