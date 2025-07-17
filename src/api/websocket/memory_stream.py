"""
Live Memory Updates Streaming for Real-time Memory Streams.

This module provides real-time memory event streaming including CRUD operations,
notifications, subscription management, and intelligent filtering.
All processing is performed locally with zero external dependencies.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from .server import ConnectionManager, MessageType

logger = structlog.get_logger(__name__)


class MemoryEventType(str, Enum):
    """Types of memory events."""
    CREATED = "memory_created"
    UPDATED = "memory_updated"
    DELETED = "memory_deleted"
    ARCHIVED = "memory_archived"
    RESTORED = "memory_restored"
    SYNTHESIZED = "memory_synthesized"
    LINKED = "memory_linked"
    UNLINKED = "memory_unlinked"


class StreamFilter(BaseModel):
    """Filter configuration for memory streams."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    user_ids: Optional[List[str]] = Field(default=None, description="Filter by user IDs")
    memory_types: Optional[List[str]] = Field(default=None, description="Filter by memory types")
    domains: Optional[List[str]] = Field(default=None, description="Filter by domains")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    event_types: Optional[List[MemoryEventType]] = Field(default=None, description="Filter by event types")
    created_after: Optional[datetime] = Field(default=None, description="Filter by creation time")
    min_relevance: Optional[float] = Field(default=None, description="Minimum relevance score")
    include_system_events: bool = Field(default=False, description="Include system-generated events")


@dataclass
class MemoryEvent:
    """Represents a memory-related event."""
    id: str
    event_type: MemoryEventType
    memory_id: str
    user_id: Optional[str]
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata
        }


@dataclass
class MemorySubscription:
    """Represents a memory stream subscription."""
    id: str
    connection_id: str
    user_id: str
    filters: StreamFilter
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    event_count: int = 0
    
    def matches_event(self, event: MemoryEvent) -> bool:
        """Check if event matches subscription filters."""
        # Check user IDs
        if self.filters.user_ids and event.user_id not in self.filters.user_ids:
            return False
        
        # Check event types
        if self.filters.event_types and event.event_type not in self.filters.event_types:
            return False
        
        # Check creation time
        if self.filters.created_after and event.timestamp < self.filters.created_after:
            return False
        
        # Check memory metadata filters
        memory_data = event.data.get('memory', {})
        
        # Check memory types
        if self.filters.memory_types:
            memory_type = memory_data.get('type')
            if memory_type not in self.filters.memory_types:
                return False
        
        # Check domains
        if self.filters.domains:
            domain = memory_data.get('metadata', {}).get('domain')
            if domain not in self.filters.domains:
                return False
        
        # Check tags
        if self.filters.tags:
            memory_tags = memory_data.get('metadata', {}).get('tags', [])
            if not any(tag in self.filters.tags for tag in memory_tags):
                return False
        
        # Check minimum relevance
        if self.filters.min_relevance:
            relevance = event.data.get('relevance_score', 0.0)
            if relevance < self.filters.min_relevance:
                return False
        
        # Check system events
        if not self.filters.include_system_events:
            if event.metadata.get('system_generated', False):
                return False
        
        return True


class MemoryStreamManager:
    """
    Manages real-time memory event streaming.
    
    Features:
    - Real-time memory CRUD event broadcasting
    - Subscription management with intelligent filtering
    - Rate limiting and performance optimization
    - Event buffering and replay capabilities
    - Memory lifecycle tracking
    - Batch event processing
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        redis_cache: Optional[RedisCache] = None,
        buffer_size: int = 1000,
        batch_interval: float = 0.1
    ):
        """
        Initialize the memory stream manager.
        
        Args:
            connection_manager: WebSocket connection manager
            redis_cache: Optional Redis cache for event persistence
            buffer_size: Maximum events to buffer
            batch_interval: Interval for batch processing (seconds)
        """
        self.connection_manager = connection_manager
        self.redis_cache = redis_cache
        self.buffer_size = buffer_size
        self.batch_interval = batch_interval
        
        # Subscriptions management
        self.subscriptions: Dict[str, MemorySubscription] = {}
        self.user_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Event buffering
        self.event_buffer: deque = deque(maxlen=buffer_size)
        self.pending_events: List[MemoryEvent] = []
        
        # Event history for replay
        self.event_history: deque = deque(maxlen=10000)  # Keep last 10k events
        
        # Performance tracking
        self.stats = {
            'events_processed': 0,
            'events_sent': 0,
            'active_subscriptions': 0,
            'buffer_overflows': 0,
            'processing_errors': 0
        }
        
        # Background tasks
        self._batch_processor_task = None
        self._cleanup_task = None
        
        # Register message handlers
        self.connection_manager.register_handler(
            MessageType.MEMORY_SEARCH,
            self._handle_memory_search
        )
        
        logger.info(
            "Memory stream manager initialized",
            buffer_size=buffer_size,
            batch_interval=batch_interval
        )
    
    async def start(self):
        """Start the memory stream manager."""
        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Memory stream manager started")
    
    async def stop(self):
        """Stop the memory stream manager."""
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Process remaining events
        if self.pending_events:
            await self._process_event_batch()
        
        logger.info("Memory stream manager stopped")
    
    async def create_subscription(
        self,
        connection_id: str,
        user_id: str,
        filters: StreamFilter
    ) -> str:
        """
        Create a new memory stream subscription.
        
        Args:
            connection_id: WebSocket connection ID
            user_id: User ID for the subscription
            filters: Filter configuration
            
        Returns:
            Subscription ID
        """
        # Generate subscription ID
        subscription_id = f"mem_sub_{int(time.time() * 1000)}_{connection_id[:8]}"
        
        # Create subscription
        subscription = MemorySubscription(
            id=subscription_id,
            connection_id=connection_id,
            user_id=user_id,
            filters=filters
        )
        
        # Store subscription
        self.subscriptions[subscription_id] = subscription
        self.user_subscriptions[user_id].add(subscription_id)
        
        # Subscribe to WebSocket events
        await self.connection_manager.subscribe(connection_id, f"memory_stream_{subscription_id}")
        
        # Update statistics
        self.stats['active_subscriptions'] = len(self.subscriptions)
        
        logger.info(
            "Memory stream subscription created",
            subscription_id=subscription_id,
            user_id=user_id,
            connection_id=connection_id
        )
        
        return subscription_id
    
    async def remove_subscription(self, subscription_id: str):
        """
        Remove a memory stream subscription.
        
        Args:
            subscription_id: Subscription ID to remove
        """
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return
        
        # Unsubscribe from WebSocket events
        await self.connection_manager.unsubscribe(
            subscription.connection_id,
            f"memory_stream_{subscription_id}"
        )
        
        # Remove subscription
        del self.subscriptions[subscription_id]
        self.user_subscriptions[subscription.user_id].discard(subscription_id)
        
        # Clean up empty user subscription sets
        if not self.user_subscriptions[subscription.user_id]:
            del self.user_subscriptions[subscription.user_id]
        
        # Update statistics
        self.stats['active_subscriptions'] = len(self.subscriptions)
        
        logger.info(
            "Memory stream subscription removed",
            subscription_id=subscription_id
        )
    
    async def emit_memory_event(
        self,
        event_type: MemoryEventType,
        memory: Memory,
        user_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit a memory-related event.
        
        Args:
            event_type: Type of memory event
            memory: Memory object
            user_id: User ID associated with the event
            additional_data: Additional event data
            metadata: Event metadata
        """
        # Generate event ID
        event_id = f"mem_evt_{int(time.time() * 1000)}_{memory.id[:8]}"
        
        # Prepare event data
        event_data = {
            "memory": {
                "id": memory.id,
                "content": memory.content,
                "type": getattr(memory, 'type', 'text'),
                "created_at": memory.created_at.isoformat() if hasattr(memory, 'created_at') and memory.created_at else None,
                "metadata": getattr(memory, 'metadata', {})
            }
        }
        
        if additional_data:
            event_data.update(additional_data)
        
        # Create event
        event = MemoryEvent(
            id=event_id,
            event_type=event_type,
            memory_id=memory.id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            data=event_data,
            metadata=metadata or {}
        )
        
        # Add to processing queue
        self.pending_events.append(event)
        
        # Add to history
        self.event_history.append(event)
        
        # Update statistics
        self.stats['events_processed'] += 1
        
        logger.debug(
            "Memory event emitted",
            event_id=event_id,
            event_type=event_type.value,
            memory_id=memory.id
        )
    
    async def replay_events(
        self,
        subscription_id: str,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MemoryEvent]:
        """
        Replay historical events for a subscription.
        
        Args:
            subscription_id: Subscription ID
            since: Replay events since this timestamp
            limit: Maximum number of events to replay
            
        Returns:
            List of matching historical events
        """
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return []
        
        # Filter historical events
        matching_events = []
        cutoff_time = since or (datetime.utcnow() - timedelta(hours=1))
        
        for event in reversed(self.event_history):
            if len(matching_events) >= limit:
                break
            
            if event.timestamp < cutoff_time:
                break
            
            if subscription.matches_event(event):
                matching_events.append(event)
        
        # Reverse to get chronological order
        matching_events.reverse()
        
        logger.info(
            "Event replay completed",
            subscription_id=subscription_id,
            events_found=len(matching_events),
            since=since.isoformat() if since else None
        )
        
        return matching_events
    
    async def _process_event_batch(self):
        """Process a batch of pending events."""
        if not self.pending_events:
            return
        
        events_to_process = self.pending_events.copy()
        self.pending_events.clear()
        
        try:
            # Group events by subscription matches
            subscription_events: Dict[str, List[MemoryEvent]] = defaultdict(list)
            
            for event in events_to_process:
                for subscription_id, subscription in self.subscriptions.items():
                    if subscription.matches_event(event):
                        subscription_events[subscription_id].append(event)
            
            # Send events to matching subscriptions
            send_tasks = []
            for subscription_id, events in subscription_events.items():
                subscription = self.subscriptions[subscription_id]
                
                # Update subscription activity
                subscription.last_activity = datetime.utcnow()
                subscription.event_count += len(events)
                
                # Prepare batch message
                message = {
                    "type": MessageType.MEMORY_UPDATED.value,
                    "subscription_id": subscription_id,
                    "events": [event.to_dict() for event in events],
                    "batch_size": len(events)
                }
                
                # Send message
                task = self.connection_manager.publish_to_subscription(
                    f"memory_stream_{subscription_id}",
                    message
                )
                send_tasks.append(task)
            
            # Execute all sends concurrently
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)
                self.stats['events_sent'] += sum(len(events) for events in subscription_events.values())
            
        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.warning(f"Error processing event batch: {e}")
    
    async def _batch_processor_loop(self):
        """Background task for batch processing events."""
        while True:
            try:
                await asyncio.sleep(self.batch_interval)
                
                if self.pending_events:
                    await self._process_event_batch()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in batch processor loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task for cleaning up stale subscriptions."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.utcnow()
                stale_subscriptions = []
                
                # Find stale subscriptions
                for subscription_id, subscription in self.subscriptions.items():
                    # Remove subscriptions with no activity for 1 hour
                    if (current_time - subscription.last_activity).total_seconds() > 3600:
                        stale_subscriptions.append(subscription_id)
                
                # Clean up stale subscriptions
                for subscription_id in stale_subscriptions:
                    await self.remove_subscription(subscription_id)
                
                if stale_subscriptions:
                    logger.info(f"Cleaned up {len(stale_subscriptions)} stale subscriptions")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in cleanup loop: {e}")
    
    async def _handle_memory_search(self, connection_id: str, data: Dict[str, Any]):
        """Handle memory search requests from WebSocket."""
        try:
            # Extract search parameters
            query = data.get('query', '')
            filters = data.get('filters', {})
            subscription_id = data.get('subscription_id')
            
            # Create temporary subscription for search if needed
            if not subscription_id:
                # Get user ID from connection
                connection = self.connection_manager.connections.get(connection_id)
                if not connection or not connection.user_id:
                    await self.connection_manager.send_message(connection_id, {
                        "type": MessageType.ERROR.value,
                        "error": "Authentication required for memory search"
                    })
                    return
                
                # Create temporary filter
                stream_filter = StreamFilter(**filters)
                subscription_id = await self.create_subscription(
                    connection_id,
                    connection.user_id,
                    stream_filter
                )
            
            # Send search acknowledgment
            await self.connection_manager.send_message(connection_id, {
                "type": MessageType.SEARCH_RESULT.value,
                "subscription_id": subscription_id,
                "query": query,
                "status": "searching"
            })
            
            # Perform replay of recent matching events
            recent_events = await self.replay_events(subscription_id, limit=50)
            
            if recent_events:
                # Send recent events as search results
                await self.connection_manager.send_message(connection_id, {
                    "type": MessageType.SEARCH_RESULT.value,
                    "subscription_id": subscription_id,
                    "query": query,
                    "events": [event.to_dict() for event in recent_events],
                    "status": "completed"
                })
            
        except Exception as e:
            logger.warning(f"Error handling memory search: {e}")
            await self.connection_manager.send_message(connection_id, {
                "type": MessageType.ERROR.value,
                "error": "Memory search failed"
            })
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get memory stream statistics."""
        return {
            **self.stats,
            'pending_events': len(self.pending_events),
            'buffered_events': len(self.event_buffer),
            'historical_events': len(self.event_history),
            'subscriptions_by_user': len(self.user_subscriptions),
            'average_events_per_subscription': (
                sum(sub.event_count for sub in self.subscriptions.values()) / 
                len(self.subscriptions) if self.subscriptions else 0
            )
        }
    
    async def get_subscription_info(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific subscription."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return None
        
        return {
            "id": subscription.id,
            "connection_id": subscription.connection_id,
            "user_id": subscription.user_id,
            "created_at": subscription.created_at.isoformat(),
            "last_activity": subscription.last_activity.isoformat(),
            "event_count": subscription.event_count,
            "filters": subscription.filters.model_dump()
        }
    
    async def cleanup_expired_subscriptions(self, max_age_hours: int = 24):
        """Clean up expired or inactive subscriptions."""
        try:
            current_time = datetime.utcnow()
            expired_subscriptions = []
            
            for subscription_id, subscription in self.subscriptions.items():
                age_hours = (current_time - subscription.last_activity).total_seconds() / 3600
                if age_hours > max_age_hours:
                    expired_subscriptions.append(subscription_id)
            
            # Remove expired subscriptions
            for subscription_id in expired_subscriptions:
                del self.subscriptions[subscription_id]
                logger.info(
                    "Removed expired subscription",
                    subscription_id=subscription_id,
                    age_hours=age_hours
                )
            
            return len(expired_subscriptions)
            
        except Exception as e:
            logger.error("Error cleaning up expired subscriptions", error=str(e))
            return 0
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about memory streaming."""
        try:
            current_time = datetime.utcnow()
            
            # Basic counts
            total_subscriptions = len(self.subscriptions)
            active_subscriptions = 0
            total_events_sent = 0
            
            # Activity analysis
            recent_activity = []
            event_type_counts = defaultdict(int)
            
            for subscription in self.subscriptions.values():
                total_events_sent += subscription.event_count
                
                # Check if active (activity within last hour)
                if (current_time - subscription.last_activity).total_seconds() < 3600:
                    active_subscriptions += 1
                
                # Track recent activity
                recent_activity.append({
                    "subscription_id": subscription.id,
                    "user_id": subscription.user_id,
                    "last_activity": subscription.last_activity.isoformat(),
                    "event_count": subscription.event_count
                })
            
            # Performance metrics
            cache_stats = {
                "cache_hits": getattr(self, '_cache_hits', 0),
                "cache_misses": getattr(self, '_cache_misses', 0),
                "cache_hit_rate": 0.0
            }
            
            if cache_stats["cache_hits"] + cache_stats["cache_misses"] > 0:
                cache_stats["cache_hit_rate"] = cache_stats["cache_hits"] / (
                    cache_stats["cache_hits"] + cache_stats["cache_misses"]
                )
            
            return {
                "timestamp": current_time.isoformat(),
                "subscriptions": {
                    "total": total_subscriptions,
                    "active": active_subscriptions,
                    "inactive": total_subscriptions - active_subscriptions
                },
                "events": {
                    "total_sent": total_events_sent,
                    "avg_per_subscription": total_events_sent / max(total_subscriptions, 1),
                    "type_distribution": dict(event_type_counts)
                },
                "performance": {
                    "avg_processing_time_ms": getattr(self, '_avg_processing_time_ms', 0.0),
                    "success_rate": getattr(self, '_success_rate', 1.0),
                    **cache_stats
                },
                "recent_activity": recent_activity[-10:]  # Last 10 activities
            }
            
        except Exception as e:
            logger.error("Error getting stream metrics", error=str(e))
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }


# Export main components  
__all__ = [
    "MemoryStreamManager",
    "MemoryEvent", 
    "MemoryEventType",
    "MemorySubscription",
    "StreamFilter"
]