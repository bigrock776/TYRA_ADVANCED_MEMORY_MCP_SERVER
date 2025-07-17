"""
WebSocket Infrastructure for Real-time Memory Streams.

This module provides comprehensive real-time capabilities for the Tyra memory system including:
- WebSocket connection management
- Live memory updates and notifications
- Real-time search streaming
- Event-driven triggers and automation

All processing is performed locally with zero external dependencies.
"""

from .server import (
    WebSocketServer,
    ConnectionManager,
    WebSocketConnection,
    ConnectionState,
    MessageType
)

from .memory_stream import (
    MemoryStreamManager,
    MemoryEvent,
    MemoryEventType,
    MemorySubscription,
    StreamFilter
)

from .search_stream import (
    SearchStreamManager,
    SearchEvent,
    SearchEventType,
    QuerySuggestion,
    SearchState
)

__all__ = [
    # WebSocket infrastructure
    "WebSocketServer",
    "ConnectionManager", 
    "WebSocketConnection",
    "ConnectionState",
    "MessageType",
    
    # Memory streaming
    "MemoryStreamManager",
    "MemoryEvent",
    "MemoryEventType", 
    "MemorySubscription",
    "StreamFilter",
    
    # Search streaming
    "SearchStreamManager",
    "SearchEvent",
    "SearchEventType",
    "QuerySuggestion",
    "SearchState"
]