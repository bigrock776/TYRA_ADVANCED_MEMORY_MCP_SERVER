"""
Service management module.

Provides service lifecycle management for background services
like file watcher, monitoring, and other automated processes.
"""

from .file_watcher_service import FileWatcherServiceManager, get_file_watcher_manager

__all__ = [
    "FileWatcherServiceManager",
    "get_file_watcher_manager",
]