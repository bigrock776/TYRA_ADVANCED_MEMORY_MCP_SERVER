"""
File Watcher Service Manager.

Manages the lifecycle of the file watcher service as part of the
main application startup and shutdown process.
"""

import asyncio
from typing import Optional

from ..ingestion.file_watcher import FileWatcherService, get_file_watcher_service
from ..utils.simple_logger import get_logger

logger = get_logger(__name__)


class FileWatcherServiceManager:
    """
    Manager for the file watcher service lifecycle.
    
    Handles starting, stopping, and monitoring the file watcher service
    as part of the main application lifecycle.
    """
    
    def __init__(self):
        self.service: Optional[FileWatcherService] = None
        self.is_enabled = True  # Can be configured
        
    async def start(self):
        """Start the file watcher service."""
        if not self.is_enabled:
            logger.info("File watcher service is disabled")
            return
            
        try:
            logger.info("Starting file watcher service...")
            self.service = await get_file_watcher_service()
            await self.service.start()
            logger.info("File watcher service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start file watcher service: {e}")
            raise
    
    async def stop(self):
        """Stop the file watcher service."""
        if self.service:
            try:
                logger.info("Stopping file watcher service...")
                await self.service.stop()
                logger.info("File watcher service stopped successfully")
                
            except Exception as e:
                logger.error(f"Error stopping file watcher service: {e}")
    
    async def restart(self):
        """Restart the file watcher service."""
        await self.stop()
        await self.start()
    
    def is_running(self) -> bool:
        """Check if the file watcher service is running."""
        return self.service is not None and self.service.is_running
    
    async def get_health_status(self):
        """Get health status of the file watcher service."""
        if not self.service:
            return {
                'status': 'not_running',
                'enabled': self.is_enabled,
                'message': 'Service not initialized'
            }
        
        return await self.service.health_check()
    
    async def get_stats(self):
        """Get processing statistics."""
        if not self.service:
            return {
                'error': 'Service not initialized',
                'enabled': self.is_enabled
            }
        
        return self.service.get_stats()


# Global service manager instance
_service_manager: Optional[FileWatcherServiceManager] = None


def get_file_watcher_manager() -> FileWatcherServiceManager:
    """Get or create the global file watcher service manager."""
    global _service_manager
    
    if _service_manager is None:
        _service_manager = FileWatcherServiceManager()
        
    return _service_manager