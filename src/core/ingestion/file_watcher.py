"""
File watcher service for automatic document ingestion.

Monitors the tyra-ingest folder for new files and automatically processes them
through the document ingestion pipeline with comprehensive error handling
and file organization.
"""

import asyncio
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..utils.simple_logger import get_logger
from ..utils.simple_config import get_settings
from .document_processor import DocumentProcessor
from .file_loaders import get_file_loader

logger = get_logger(__name__)


class FileIngestHandler(FileSystemEventHandler):
    """
    File system event handler for automatic document ingestion.
    
    Monitors file creation events and processes supported file types
    through the existing document ingestion pipeline.
    """
    
    def __init__(self, watcher_service: 'FileWatcherService'):
        self.watcher_service = watcher_service
        self._processing_files: Set[str] = set()
        self._file_checksums: Dict[str, str] = {}
        
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        file_path = event.src_path
        logger.info(f"File created: {file_path}")
        
        # Add to processing queue
        asyncio.create_task(self._process_file_async(file_path))
    
    def on_moved(self, event):
        """Handle file move events (drag & drop)."""
        if event.is_directory:
            return
            
        file_path = event.dest_path
        logger.info(f"File moved to: {file_path}")
        
        # Add to processing queue
        asyncio.create_task(self._process_file_async(file_path))
    
    async def _process_file_async(self, file_path: str):
        """Process a file asynchronously."""
        try:
            # Wait a bit to ensure file is fully written
            await asyncio.sleep(0.5)
            
            # Check if file still exists and is not being processed
            if not os.path.exists(file_path):
                logger.warning(f"File no longer exists: {file_path}")
                return
                
            if file_path in self._processing_files:
                logger.info(f"File already being processed: {file_path}")
                return
                
            # Check for file stability (ensure it's not still being written)
            if not await self._is_file_stable(file_path):
                logger.info(f"File not stable yet, skipping: {file_path}")
                return
                
            # Mark as processing
            self._processing_files.add(file_path)
            
            try:
                await self.watcher_service.process_file(file_path)
            finally:
                # Remove from processing set
                self._processing_files.discard(file_path)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self._processing_files.discard(file_path)
    
    async def _is_file_stable(self, file_path: str, check_interval: float = 0.1) -> bool:
        """Check if file is stable (not being written to)."""
        try:
            # Get initial size
            initial_size = os.path.getsize(file_path)
            
            # Wait briefly
            await asyncio.sleep(check_interval)
            
            # Check if size changed
            current_size = os.path.getsize(file_path)
            
            # If size is the same, file is likely stable
            return initial_size == current_size
            
        except OSError:
            # File might be locked or not accessible
            return False


class FileWatcherService:
    """
    Service for monitoring and processing files in the tyra-ingest folder.
    
    Features:
    - Automatic file detection and processing
    - Support for all document types handled by the ingestion pipeline
    - File organization (processed/failed)
    - Duplicate detection
    - Error handling and retry logic
    - Processing statistics and monitoring
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd() / "tyra-ingest"
        self.inbox_path = self.base_path / "inbox"
        self.processed_path = self.base_path / "processed"
        self.failed_path = self.base_path / "failed"
        
        # Create directories if they don't exist
        self.inbox_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.failed_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.observer: Optional[Observer] = None
        self.is_running = False
        
        # Supported file extensions
        self.supported_extensions = {
            '.pdf', '.docx', '.txt', '.md', '.html', '.json', '.csv'
        }
        
        # Processing statistics
        self.stats = {
            'total_files_processed': 0,
            'successful_ingestions': 0,
            'failed_ingestions': 0,
            'duplicate_files': 0,
            'unsupported_files': 0,
            'start_time': datetime.utcnow(),
            'last_processed': None
        }
        
        # Settings
        self.settings = get_settings()
        
        logger.info(f"FileWatcherService initialized with base path: {self.base_path}")
    
    async def start(self):
        """Start the file watcher service."""
        if self.is_running:
            logger.warning("FileWatcherService is already running")
            return
            
        try:
            # Initialize document processor
            await self.document_processor.initialize()
            
            # Set up file system watcher
            event_handler = FileIngestHandler(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, str(self.inbox_path), recursive=False)
            
            # Start observer
            self.observer.start()
            self.is_running = True
            
            logger.info(f"FileWatcherService started, monitoring: {self.inbox_path}")
            
            # Process any existing files in inbox
            await self._process_existing_files()
            
        except Exception as e:
            logger.error(f"Failed to start FileWatcherService: {e}")
            raise
    
    async def stop(self):
        """Stop the file watcher service."""
        if not self.is_running:
            return
            
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
                
            self.is_running = False
            logger.info("FileWatcherService stopped")
            
        except Exception as e:
            logger.error(f"Error stopping FileWatcherService: {e}")
    
    async def process_file(self, file_path: str):
        """Process a single file through the ingestion pipeline."""
        file_path = Path(file_path)
        
        try:
            logger.info(f"Processing file: {file_path}")
            self.stats['total_files_processed'] += 1
            
            # Check if file is supported
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                self.stats['unsupported_files'] += 1
                await self._move_to_failed(file_path, "Unsupported file type")
                return
            
            # Check for duplicates
            if await self._is_duplicate(file_path):
                logger.info(f"Duplicate file detected: {file_path}")
                self.stats['duplicate_files'] += 1
                await self._move_to_failed(file_path, "Duplicate file")
                return
            
            # Read file content
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                self.stats['failed_ingestions'] += 1
                await self._move_to_failed(file_path, f"Failed to read file: {e}")
                return
            
            # Process through document processor
            try:
                # Determine file type
                file_type = file_path.suffix.lower().lstrip('.')
                
                # Create ingestion request
                result = await self.document_processor.process_document(
                    content=file_content,
                    file_name=file_path.name,
                    file_type=file_type,
                    agent_id="tyra",  # Default agent for file watcher
                    metadata={
                        "source": "file_watcher",
                        "original_path": str(file_path),
                        "processed_at": datetime.utcnow().isoformat(),
                        "file_size": len(file_content)
                    }
                )
                
                if result.success:
                    logger.info(f"Successfully processed: {file_path}")
                    self.stats['successful_ingestions'] += 1
                    await self._move_to_processed(file_path, result)
                else:
                    logger.error(f"Failed to process {file_path}: {result.error}")
                    self.stats['failed_ingestions'] += 1
                    await self._move_to_failed(file_path, result.error or "Unknown error")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.stats['failed_ingestions'] += 1
                await self._move_to_failed(file_path, f"Processing error: {e}")
            
            # Update stats
            self.stats['last_processed'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            self.stats['failed_ingestions'] += 1
            try:
                await self._move_to_failed(file_path, f"Unexpected error: {e}")
            except:
                pass  # Don't fail if we can't move the file
    
    async def _process_existing_files(self):
        """Process any files that already exist in the inbox."""
        if not self.inbox_path.exists():
            return
            
        existing_files = list(self.inbox_path.glob("*"))
        if not existing_files:
            return
            
        logger.info(f"Processing {len(existing_files)} existing files in inbox")
        
        for file_path in existing_files:
            if file_path.is_file():
                await self.process_file(str(file_path))
    
    async def _is_duplicate(self, file_path: Path) -> bool:
        """Check if file is a duplicate based on content hash."""
        try:
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Check against processed files
            hash_file = self.processed_path / f"{file_hash}.json"
            return hash_file.exists()
            
        except Exception as e:
            logger.error(f"Error checking duplicate for {file_path}: {e}")
            return False
    
    async def _move_to_processed(self, file_path: Path, result: Any):
        """Move file to processed folder with metadata."""
        try:
            # Create timestamp folder
            timestamp = datetime.utcnow().strftime("%Y%m%d")
            processed_dir = self.processed_path / timestamp
            processed_dir.mkdir(exist_ok=True)
            
            # Move file
            new_path = processed_dir / file_path.name
            shutil.move(str(file_path), str(new_path))
            
            # Create metadata file
            metadata = {
                "original_path": str(file_path),
                "processed_path": str(new_path),
                "processed_at": datetime.utcnow().isoformat(),
                "file_size": new_path.stat().st_size,
                "file_hash": hashlib.md5(new_path.read_bytes()).hexdigest(),
                "processing_result": {
                    "success": result.success,
                    "memory_id": result.memory_id,
                    "chunks_created": len(result.chunks),
                    "processing_time": result.processing_time
                }
            }
            
            metadata_path = new_path.with_suffix(new_path.suffix + '.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create hash reference
            hash_file = self.processed_path / f"{metadata['file_hash']}.json"
            with open(hash_file, 'w') as f:
                json.dump({"file_path": str(new_path), "metadata": metadata}, f)
            
            logger.info(f"Moved to processed: {new_path}")
            
        except Exception as e:
            logger.error(f"Error moving file to processed: {e}")
    
    async def _move_to_failed(self, file_path: Path, error_message: str):
        """Move file to failed folder with error information."""
        try:
            # Create timestamp folder
            timestamp = datetime.utcnow().strftime("%Y%m%d")
            failed_dir = self.failed_path / timestamp
            failed_dir.mkdir(exist_ok=True)
            
            # Move file
            new_path = failed_dir / file_path.name
            shutil.move(str(file_path), str(new_path))
            
            # Create error metadata
            error_metadata = {
                "original_path": str(file_path),
                "failed_path": str(new_path),
                "failed_at": datetime.utcnow().isoformat(),
                "error_message": error_message,
                "file_size": new_path.stat().st_size,
                "file_extension": file_path.suffix.lower()
            }
            
            error_path = new_path.with_suffix(new_path.suffix + '.error.json')
            with open(error_path, 'w') as f:
                json.dump(error_metadata, f, indent=2)
            
            logger.info(f"Moved to failed: {new_path}")
            
        except Exception as e:
            logger.error(f"Error moving file to failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        current_time = datetime.utcnow()
        uptime = (current_time - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'is_running': self.is_running,
            'monitored_path': str(self.inbox_path),
            'processed_path': str(self.processed_path),
            'failed_path': str(self.failed_path),
            'supported_extensions': list(self.supported_extensions),
            'success_rate': (
                self.stats['successful_ingestions'] / 
                max(self.stats['total_files_processed'], 1)
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the file watcher service."""
        try:
            health_status = {
                'status': 'healthy',
                'is_running': self.is_running,
                'observer_alive': self.observer.is_alive() if self.observer else False,
                'paths_accessible': {
                    'inbox': self.inbox_path.exists() and os.access(self.inbox_path, os.R_OK | os.W_OK),
                    'processed': self.processed_path.exists() and os.access(self.processed_path, os.R_OK | os.W_OK),
                    'failed': self.failed_path.exists() and os.access(self.failed_path, os.R_OK | os.W_OK)
                },
                'stats': self.get_stats(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Check if any paths are inaccessible
            if not all(health_status['paths_accessible'].values()):
                health_status['status'] = 'degraded'
                health_status['warnings'] = ['Some paths are not accessible']
            
            # Check if observer is not running when it should be
            if self.is_running and not health_status['observer_alive']:
                health_status['status'] = 'unhealthy'
                health_status['error'] = 'Observer is not running'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


# Global file watcher instance
_file_watcher_service: Optional[FileWatcherService] = None


async def get_file_watcher_service() -> FileWatcherService:
    """Get or create the global file watcher service."""
    global _file_watcher_service
    
    if _file_watcher_service is None:
        _file_watcher_service = FileWatcherService()
        
    return _file_watcher_service


async def start_file_watcher():
    """Start the file watcher service."""
    service = await get_file_watcher_service()
    await service.start()


async def stop_file_watcher():
    """Stop the file watcher service."""
    if _file_watcher_service:
        await _file_watcher_service.stop()