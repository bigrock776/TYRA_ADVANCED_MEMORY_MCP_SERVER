"""
Job tracking system for document ingestion.

Provides persistent job tracking for long-running ingestion tasks
with progress monitoring, status updates, and result storage.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from ..cache.redis_cache import RedisCache
from ..utils.logger import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL_SUCCESS = "partial_success"


class JobType(str, Enum):
    """Job type enumeration."""
    
    SINGLE_DOCUMENT = "single_document"
    BATCH_INGESTION = "batch_ingestion"
    FOLDER_SCAN = "folder_scan"
    URL_CRAWL = "url_crawl"


class IngestionJob:
    """Represents an ingestion job with tracking information."""
    
    def __init__(
        self,
        job_id: str,
        job_type: JobType,
        total_items: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.status = JobStatus.PENDING
        self.total_items = total_items
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.current_item = ""
        self.error_messages: List[str] = []
        self.warnings: List[str] = []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.last_updated = datetime.utcnow()
        self.result_data: Optional[Dict[str, Any]] = None
        
    def start(self):
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        
    def update_progress(
        self,
        processed: Optional[int] = None,
        current_item: Optional[str] = None,
        increment_success: bool = False,
        increment_failed: bool = False,
    ):
        """Update job progress."""
        if processed is not None:
            self.processed_items = processed
        if current_item is not None:
            self.current_item = current_item
        if increment_success:
            self.successful_items += 1
        if increment_failed:
            self.failed_items += 1
        self.last_updated = datetime.utcnow()
        
    def add_error(self, error: str):
        """Add error message to job."""
        self.error_messages.append(f"[{datetime.utcnow().isoformat()}] {error}")
        self.last_updated = datetime.utcnow()
        
    def add_warning(self, warning: str):
        """Add warning message to job."""
        self.warnings.append(f"[{datetime.utcnow().isoformat()}] {warning}")
        self.last_updated = datetime.utcnow()
        
    def complete(self, result_data: Optional[Dict[str, Any]] = None):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        if self.failed_items > 0 and self.successful_items > 0:
            self.status = JobStatus.PARTIAL_SUCCESS
        elif self.failed_items > 0 and self.successful_items == 0:
            self.status = JobStatus.FAILED
        
        self.completed_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        self.result_data = result_data
        
    def fail(self, error: str):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.add_error(error)
        self.completed_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        
    def cancel(self):
        """Mark job as cancelled."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 100.0 if self.status == JobStatus.COMPLETED else 0.0
        return (self.processed_items / self.total_items) * 100
        
    @property
    def estimated_time_remaining(self) -> Optional[timedelta]:
        """Estimate time remaining based on current progress."""
        if not self.started_at or self.processed_items == 0:
            return None
            
        elapsed = datetime.utcnow() - self.started_at
        if elapsed.total_seconds() == 0:
            return None
            
        rate = self.processed_items / elapsed.total_seconds()
        if rate == 0:
            return None
            
        remaining_items = self.total_items - self.processed_items
        remaining_seconds = remaining_items / rate
        
        return timedelta(seconds=remaining_seconds)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        estimated_remaining = self.estimated_time_remaining
        
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "progress_percentage": round(self.progress_percentage, 2),
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "current_item": self.current_item,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_updated": self.last_updated.isoformat(),
            "estimated_time_remaining": (
                estimated_remaining.total_seconds() if estimated_remaining else None
            ),
            "error_messages": self.error_messages[-10:],  # Last 10 errors
            "warnings": self.warnings[-10:],  # Last 10 warnings
            "metadata": self.metadata,
            "result_data": self.result_data,
        }


class JobTracker:
    """Manages ingestion job tracking with Redis persistence."""
    
    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.redis_cache = redis_cache
        self._local_jobs: Dict[str, IngestionJob] = {}
        self._job_ttl = 86400  # 24 hours
        
    async def create_job(
        self,
        job_type: JobType,
        total_items: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionJob:
        """Create a new job and store it."""
        job_id = str(uuid.uuid4())
        job = IngestionJob(job_id, job_type, total_items, metadata)
        
        # Store locally
        self._local_jobs[job_id] = job
        
        # Store in Redis if available
        if self.redis_cache:
            await self._store_job_in_cache(job)
            
        logger.info(
            "Created ingestion job",
            job_id=job_id,
            job_type=job_type,
            total_items=total_items,
        )
        
        return job
        
    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Retrieve a job by ID."""
        # Check local cache first
        if job_id in self._local_jobs:
            return self._local_jobs[job_id]
            
        # Try Redis if available
        if self.redis_cache:
            job_data = await self._get_job_from_cache(job_id)
            if job_data:
                job = self._deserialize_job(job_data)
                self._local_jobs[job_id] = job
                return job
                
        return None
        
    async def update_job(self, job: IngestionJob):
        """Update job in storage."""
        # Update local cache
        self._local_jobs[job.job_id] = job
        
        # Update Redis if available
        if self.redis_cache:
            await self._store_job_in_cache(job)
            
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 100,
    ) -> List[IngestionJob]:
        """List jobs with optional filtering."""
        jobs = []
        
        # Get from Redis if available
        if self.redis_cache:
            job_ids = await self._list_job_ids_from_cache()
            for job_id in job_ids[:limit]:
                job = await self.get_job(job_id)
                if job:
                    if status and job.status != status:
                        continue
                    if job_type and job.job_type != job_type:
                        continue
                    jobs.append(job)
        else:
            # Use local cache
            for job in self._local_jobs.values():
                if status and job.status != status:
                    continue
                if job_type and job.job_type != job_type:
                    continue
                jobs.append(job)
                if len(jobs) >= limit:
                    break
                    
        # Sort by last updated
        jobs.sort(key=lambda j: j.last_updated, reverse=True)
        
        return jobs[:limit]
        
    async def cleanup_old_jobs(self, older_than_hours: int = 24):
        """Remove jobs older than specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        jobs_to_remove = []
        for job_id, job in self._local_jobs.items():
            if job.completed_at and job.completed_at < cutoff_time:
                jobs_to_remove.append(job_id)
                
        for job_id in jobs_to_remove:
            del self._local_jobs[job_id]
            if self.redis_cache:
                await self._remove_job_from_cache(job_id)
                
        logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
        
    # Redis storage methods
    async def _store_job_in_cache(self, job: IngestionJob):
        """Store job in Redis cache."""
        if not self.redis_cache:
            return
            
        key = f"ingestion_job:{job.job_id}"
        value = job.to_dict()
        
        try:
            await self.redis_cache.set(key, value, ttl=self._job_ttl)
            
            # Also store in job list
            list_key = "ingestion_jobs:list"
            job_list = await self.redis_cache.get(list_key) or []
            if job.job_id not in job_list:
                job_list.append(job.job_id)
                # Keep only last 1000 jobs
                job_list = job_list[-1000:]
                await self.redis_cache.set(list_key, job_list, ttl=self._job_ttl * 7)
                
        except Exception as e:
            logger.error(f"Failed to store job in cache: {e}")
            
    async def _get_job_from_cache(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job from Redis cache."""
        if not self.redis_cache:
            return None
            
        key = f"ingestion_job:{job_id}"
        
        try:
            return await self.redis_cache.get(key)
        except Exception as e:
            logger.error(f"Failed to get job from cache: {e}")
            return None
            
    async def _list_job_ids_from_cache(self) -> List[str]:
        """Get list of job IDs from cache."""
        if not self.redis_cache:
            return []
            
        list_key = "ingestion_jobs:list"
        
        try:
            return await self.redis_cache.get(list_key) or []
        except Exception as e:
            logger.error(f"Failed to list jobs from cache: {e}")
            return []
            
    async def _remove_job_from_cache(self, job_id: str):
        """Remove job from Redis cache."""
        if not self.redis_cache:
            return
            
        key = f"ingestion_job:{job_id}"
        
        try:
            await self.redis_cache.delete(key)
            
            # Also remove from list
            list_key = "ingestion_jobs:list"
            job_list = await self.redis_cache.get(list_key) or []
            if job_id in job_list:
                job_list.remove(job_id)
                await self.redis_cache.set(list_key, job_list, ttl=self._job_ttl * 7)
                
        except Exception as e:
            logger.error(f"Failed to remove job from cache: {e}")
            
    def _deserialize_job(self, data: Dict[str, Any]) -> IngestionJob:
        """Deserialize job from dictionary."""
        job = IngestionJob(
            job_id=data["job_id"],
            job_type=data["job_type"],
            total_items=data["total_items"],
            metadata=data.get("metadata", {}),
        )
        
        # Restore state
        job.status = data["status"]
        job.processed_items = data["processed_items"]
        job.successful_items = data["successful_items"]
        job.failed_items = data["failed_items"]
        job.current_item = data["current_item"]
        job.error_messages = data.get("error_messages", [])
        job.warnings = data.get("warnings", [])
        job.result_data = data.get("result_data")
        
        # Restore timestamps
        job.created_at = datetime.fromisoformat(data["created_at"])
        job.last_updated = datetime.fromisoformat(data["last_updated"])
        
        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])
            
        return job


# Global job tracker instance
_job_tracker: Optional[JobTracker] = None


async def get_job_tracker() -> JobTracker:
    """Get or create the global job tracker instance."""
    global _job_tracker
    
    if _job_tracker is None:
        # Try to get Redis cache
        try:
            from ..cache.redis_cache import get_redis_cache
            redis_cache = await get_redis_cache()
        except Exception as e:
            logger.warning(f"Redis not available for job tracking: {e}")
            redis_cache = None
            
        _job_tracker = JobTracker(redis_cache)
        
    return _job_tracker