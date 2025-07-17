"""
WebSocket Infrastructure Server for Real-time Memory Streams.

This module provides the core WebSocket infrastructure for real-time communication
including connection management, authentication, message routing, and rate limiting.
All processing is performed locally with zero external dependencies.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Set, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
import jwt
import hashlib
import hmac

# FastAPI WebSocket imports
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.security import HTTPBearer
import redis.asyncio as redis

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(str, Enum):
    """Types of WebSocket messages."""
    # Authentication
    AUTH_REQUEST = "auth_request"
    AUTH_RESPONSE = "auth_response"
    
    # Memory operations
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_SEARCH = "memory_search"
    
    # Search operations
    SEARCH_QUERY = "search_query"
    SEARCH_RESULT = "search_result"
    SEARCH_SUGGESTION = "search_suggestion"
    
    # System messages
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SYSTEM_STATUS = "system_status"
    SERVER_SHUTDOWN = "server_shutdown"
    
    # Subscription management
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"


class RateLimitBucket:
    """Token bucket for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection with metadata."""
    id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    state: ConnectionState = ConnectionState.CONNECTING
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    rate_limiter: RateLimitBucket = field(default_factory=lambda: RateLimitBucket(100, 10))


class ConnectionManager:
    """
    Manages WebSocket connections with authentication and rate limiting.
    
    Features:
    - Connection lifecycle management
    - JWT-based authentication
    - Rate limiting with token bucket algorithm
    - Message routing and broadcasting
    - Subscription management
    - Connection health monitoring
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        jwt_secret: str = "tyra-memory-server-secret",
        jwt_algorithm: str = "HS256",
        max_connections: int = 1000,
        heartbeat_interval: int = 30
    ):
        """
        Initialize the connection manager.
        
        Args:
            redis_cache: Optional Redis cache for distributed state
            jwt_secret: Secret key for JWT authentication
            jwt_algorithm: JWT algorithm to use
            max_connections: Maximum number of concurrent connections
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        self.redis_cache = redis_cache
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Connection statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'authentication_attempts': 0,
            'authentication_failures': 0
        }
        
        # Start background tasks
        self._heartbeat_task = None
        self._cleanup_task = None
        
        logger.info(
            "Initialized connection manager",
            max_connections=max_connections,
            heartbeat_interval=heartbeat_interval
        )
    
    async def start(self):
        """Start background tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Connection manager started")
    
    async def stop(self):
        """Stop background tasks and disconnect all connections."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Disconnect all connections
        for connection in list(self.connections.values()):
            await self.disconnect(connection.id, "Server shutdown")
        
        logger.info("Connection manager stopped")
    
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            
        Returns:
            Connection ID
            
        Raises:
            HTTPException: If connection limit exceeded
        """
        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Server overloaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Maximum connections exceeded"
            )
        
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Accept the connection
        await websocket.accept()
        
        # Create connection object
        connection = WebSocketConnection(
            id=connection_id,
            websocket=websocket,
            state=ConnectionState.CONNECTED
        )
        
        # Store connection
        self.connections[connection_id] = connection
        
        # Update statistics
        self.stats['total_connections'] += 1
        self.stats['active_connections'] = len(self.connections)
        
        logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            active_connections=self.stats['active_connections']
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str, reason: str = "Normal closure"):
        """
        Disconnect a WebSocket connection.
        
        Args:
            connection_id: Connection ID to disconnect
            reason: Reason for disconnection
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Update connection state
        connection.state = ConnectionState.DISCONNECTING
        
        try:
            # Remove from subscriptions
            for subscription in connection.subscriptions.copy():
                await self.unsubscribe(connection_id, subscription)
            
            # Remove from user connections
            if connection.user_id:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
            
            # Close WebSocket
            await connection.websocket.close(code=1000, reason=reason)
            
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            # Remove from connections
            if connection_id in self.connections:
                del self.connections[connection_id]
            
            # Update statistics
            self.stats['active_connections'] = len(self.connections)
            
            logger.info(
                "WebSocket connection closed",
                connection_id=connection_id,
                reason=reason,
                active_connections=self.stats['active_connections']
            )
    
    async def authenticate(self, connection_id: str, token: str) -> bool:
        """
        Authenticate a WebSocket connection using JWT.
        
        Args:
            connection_id: Connection ID to authenticate
            token: JWT token
            
        Returns:
            True if authentication successful
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        self.stats['authentication_attempts'] += 1
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Extract user information
            user_id = payload.get('user_id')
            session_id = payload.get('session_id')
            
            if not user_id:
                raise ValueError("Missing user_id in token")
            
            # Update connection
            connection.user_id = user_id
            connection.session_id = session_id
            connection.state = ConnectionState.AUTHENTICATED
            connection.last_activity = datetime.utcnow()
            
            # Add to user connections
            self.user_connections[user_id].add(connection_id)
            
            # Send authentication response
            await self.send_message(connection_id, {
                "type": MessageType.AUTH_RESPONSE.value,
                "status": "success",
                "user_id": user_id,
                "session_id": session_id
            })
            
            logger.info(
                "WebSocket authentication successful",
                connection_id=connection_id,
                user_id=user_id
            )
            
            return True
            
        except Exception as e:
            self.stats['authentication_failures'] += 1
            
            # Send error response
            await self.send_message(connection_id, {
                "type": MessageType.AUTH_RESPONSE.value,
                "status": "error",
                "error": str(e)
            })
            
            logger.warning(
                "WebSocket authentication failed",
                connection_id=connection_id,
                error=str(e)
            )
            
            return False
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: Target connection ID
            message: Message to send
            
        Returns:
            True if message sent successfully
        """
        connection = self.connections.get(connection_id)
        if not connection or connection.state == ConnectionState.DISCONNECTED:
            return False
        
        try:
            # Add timestamp to message
            message['timestamp'] = datetime.utcnow().isoformat()
            
            # Convert to JSON
            json_message = json.dumps(message)
            
            # Send message
            await connection.websocket.send_text(json_message)
            
            # Update statistics and activity
            self.stats['messages_sent'] += 1
            connection.last_activity = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.warning(
                "Failed to send WebSocket message",
                connection_id=connection_id,
                error=str(e)
            )
            
            # Mark connection for cleanup
            connection.state = ConnectionState.ERROR
            return False
    
    async def broadcast_message(self, message: Dict[str, Any], user_filter: Optional[str] = None):
        """
        Broadcast a message to multiple connections.
        
        Args:
            message: Message to broadcast
            user_filter: Optional user ID to filter connections
        """
        target_connections = []
        
        if user_filter:
            # Send to specific user's connections
            connection_ids = self.user_connections.get(user_filter, set())
            target_connections = [
                self.connections[cid] for cid in connection_ids 
                if cid in self.connections
            ]
        else:
            # Send to all authenticated connections
            target_connections = [
                conn for conn in self.connections.values()
                if conn.state == ConnectionState.AUTHENTICATED
            ]
        
        # Send messages concurrently
        tasks = [
            self.send_message(conn.id, message.copy())
            for conn in target_connections
        ]
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for result in results if result is True)
            
            logger.info(
                "Broadcast message sent",
                message_type=message.get('type'),
                target_connections=len(target_connections),
                successful_sends=successful
            )
    
    async def handle_message(self, connection_id: str, message: str):
        """
        Handle incoming WebSocket message.
        
        Args:
            connection_id: Source connection ID
            message: Raw message string
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Check rate limiting
        if not connection.rate_limiter.consume():
            await self.send_message(connection_id, {
                "type": MessageType.ERROR.value,
                "error": "Rate limit exceeded"
            })
            return
        
        try:
            # Parse JSON message
            data = json.loads(message)
            message_type = MessageType(data.get('type'))
            
            # Update statistics and activity
            self.stats['messages_received'] += 1
            connection.last_activity = datetime.utcnow()
            
            # Handle authentication messages without authentication requirement
            if message_type == MessageType.AUTH_REQUEST:
                token = data.get('token')
                if token:
                    await self.authenticate(connection_id, token)
                return
            
            # Require authentication for other messages
            if connection.state != ConnectionState.AUTHENTICATED:
                await self.send_message(connection_id, {
                    "type": MessageType.ERROR.value,
                    "error": "Authentication required"
                })
                return
            
            # Handle message based on type
            if message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(connection_id, data)
            elif message_type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(connection_id, data)
            elif message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(connection_id, data)
            else:
                # Delegate to registered handlers
                handler = self.message_handlers.get(message_type)
                if handler:
                    await handler(connection_id, data)
                else:
                    await self.send_message(connection_id, {
                        "type": MessageType.ERROR.value,
                        "error": f"Unknown message type: {message_type.value}"
                    })
        
        except Exception as e:
            logger.warning(
                "Error handling WebSocket message",
                connection_id=connection_id,
                error=str(e)
            )
            
            await self.send_message(connection_id, {
                "type": MessageType.ERROR.value,
                "error": "Invalid message format"
            })
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type.value}")
    
    async def subscribe(self, connection_id: str, subscription: str) -> bool:
        """
        Subscribe a connection to a topic.
        
        Args:
            connection_id: Connection to subscribe
            subscription: Subscription topic
            
        Returns:
            True if subscription successful
        """
        connection = self.connections.get(connection_id)
        if not connection or connection.state != ConnectionState.AUTHENTICATED:
            return False
        
        # Add subscription
        connection.subscriptions.add(subscription)
        self.subscriptions[subscription].add(connection_id)
        
        # Send confirmation
        await self.send_message(connection_id, {
            "type": MessageType.SUBSCRIPTION_CONFIRMED.value,
            "subscription": subscription,
            "status": "subscribed"
        })
        
        logger.info(
            "Connection subscribed",
            connection_id=connection_id,
            subscription=subscription
        )
        
        return True
    
    async def unsubscribe(self, connection_id: str, subscription: str) -> bool:
        """
        Unsubscribe a connection from a topic.
        
        Args:
            connection_id: Connection to unsubscribe
            subscription: Subscription topic
            
        Returns:
            True if unsubscription successful
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        # Remove subscription
        connection.subscriptions.discard(subscription)
        self.subscriptions[subscription].discard(connection_id)
        
        # Clean up empty subscription sets
        if not self.subscriptions[subscription]:
            del self.subscriptions[subscription]
        
        # Send confirmation
        await self.send_message(connection_id, {
            "type": MessageType.SUBSCRIPTION_CONFIRMED.value,
            "subscription": subscription,
            "status": "unsubscribed"
        })
        
        logger.info(
            "Connection unsubscribed",
            connection_id=connection_id,
            subscription=subscription
        )
        
        return True
    
    async def publish_to_subscription(self, subscription: str, message: Dict[str, Any]):
        """
        Publish a message to all subscribers of a topic.
        
        Args:
            subscription: Subscription topic
            message: Message to publish
        """
        connection_ids = self.subscriptions.get(subscription, set())
        
        if connection_ids:
            tasks = [
                self.send_message(cid, message.copy())
                for cid in connection_ids
                if cid in self.connections
            ]
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for result in results if result is True)
                
                logger.info(
                    "Message published to subscription",
                    subscription=subscription,
                    subscribers=len(connection_ids),
                    successful_sends=successful
                )
    
    async def _handle_heartbeat(self, connection_id: str, data: Dict[str, Any]):
        """Handle heartbeat message."""
        await self.send_message(connection_id, {
            "type": MessageType.HEARTBEAT.value,
            "status": "ok",
            "server_time": datetime.utcnow().isoformat()
        })
    
    async def _handle_subscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription request."""
        subscription = data.get('subscription')
        if subscription:
            await self.subscribe(connection_id, subscription)
        else:
            await self.send_message(connection_id, {
                "type": MessageType.ERROR.value,
                "error": "Missing subscription topic"
            })
    
    async def _handle_unsubscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscription request."""
        subscription = data.get('subscription')
        if subscription:
            await self.unsubscribe(connection_id, subscription)
        else:
            await self.send_message(connection_id, {
                "type": MessageType.ERROR.value,
                "error": "Missing subscription topic"
            })
    
    async def _heartbeat_loop(self):
        """Background task for sending heartbeat messages."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat to all connections
                heartbeat_message = {
                    "type": MessageType.HEARTBEAT.value,
                    "server_time": datetime.utcnow().isoformat()
                }
                
                await self.broadcast_message(heartbeat_message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task for cleaning up stale connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                # Find stale connections
                for connection in self.connections.values():
                    if connection.state == ConnectionState.ERROR:
                        stale_connections.append(connection.id)
                    elif (current_time - connection.last_activity).total_seconds() > 300:  # 5 minutes
                        stale_connections.append(connection.id)
                
                # Clean up stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id, "Stale connection cleanup")
                
                if stale_connections:
                    logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in cleanup loop: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self.stats,
            'connections_by_state': {
                state.value: len([
                    c for c in self.connections.values() 
                    if c.state == state
                ])
                for state in ConnectionState
            },
            'subscriptions_count': len(self.subscriptions),
            'unique_users': len(self.user_connections)
        }


class WebSocketServer:
    """
    Main WebSocket server for real-time memory streams.
    
    Features:
    - FastAPI WebSocket integration
    - Connection management with authentication
    - Message routing and broadcasting
    - Rate limiting and security
    - Health monitoring and statistics
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        jwt_secret: Optional[str] = None
    ):
        """
        Initialize the WebSocket server.
        
        Args:
            redis_cache: Optional Redis cache for distributed state
            jwt_secret: JWT secret key (defaults to config)
        """
        self.redis_cache = redis_cache
        self.jwt_secret = jwt_secret or settings.security.jwt_secret
        
        # Server state tracking
        self.start_time = datetime.utcnow()
        self.is_running = False
        
        # Initialize connection manager
        self.connection_manager = ConnectionManager(
            redis_cache=redis_cache,
            jwt_secret=self.jwt_secret
        )
        
        logger.info("WebSocket server initialized")
    
    async def start(self):
        """Start the WebSocket server."""
        self.is_running = True
        self.start_time = datetime.utcnow()
        await self.connection_manager.start()
        logger.info("WebSocket server started", start_time=self.start_time.isoformat())
    
    async def stop(self):
        """Stop the WebSocket server."""
        await self.connection_manager.stop()
        logger.info("WebSocket server stopped")
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """
        Main WebSocket endpoint handler.
        
        Args:
            websocket: FastAPI WebSocket instance
        """
        connection_id = None
        
        try:
            # Accept connection
            connection_id = await self.connection_manager.connect(websocket)
            
            # Handle messages
            while True:
                try:
                    # Receive message
                    message = await websocket.receive_text()
                    
                    # Handle message
                    await self.connection_manager.handle_message(connection_id, message)
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.warning(
                        "Error handling WebSocket message",
                        connection_id=connection_id,
                        error=str(e)
                    )
                    break
        
        except Exception as e:
            logger.error(
                "WebSocket connection error",
                connection_id=connection_id,
                error=str(e)
            )
        
        finally:
            # Ensure cleanup
            if connection_id:
                await self.connection_manager.disconnect(
                    connection_id, 
                    "Connection ended"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self.connection_manager.get_connection_stats()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on WebSocket server."""
        try:
            # Check connection manager health
            connection_stats = self.connection_manager.get_connection_stats()
            
            # Check Redis connection if available
            redis_healthy = True
            if self.connection_manager.cache:
                try:
                    await self.connection_manager.cache.ping()
                except Exception:
                    redis_healthy = False
            
            # Check authentication service
            auth_healthy = True
            try:
                # Test token generation
                test_payload = {"user_id": "health_check", "exp": time.time() + 60}
                self.connection_manager.auth_handler._generate_token(test_payload)
            except Exception:
                auth_healthy = False
            
            # Check rate limiter
            rate_limiter_healthy = True
            try:
                test_bucket = self.connection_manager.rate_limiter._get_bucket("health_check")
                rate_limiter_healthy = test_bucket is not None
            except Exception:
                rate_limiter_healthy = False
            
            health_status = {
                "status": "healthy" if all([redis_healthy, auth_healthy, rate_limiter_healthy]) else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "connection_manager": {
                        "status": "healthy",
                        "active_connections": connection_stats["active_connections"],
                        "total_connections": connection_stats["total_connections"]
                    },
                    "redis_cache": {
                        "status": "healthy" if redis_healthy else "unhealthy",
                        "available": redis_healthy
                    },
                    "authentication": {
                        "status": "healthy" if auth_healthy else "unhealthy",
                        "available": auth_healthy
                    },
                    "rate_limiter": {
                        "status": "healthy" if rate_limiter_healthy else "unhealthy",
                        "available": rate_limiter_healthy
                    }
                },
                "metrics": {
                    "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                    "total_messages": connection_stats.get("total_messages", 0),
                    "error_rate": connection_stats.get("error_rate", 0.0),
                    "avg_response_time_ms": connection_stats.get("avg_response_time_ms", 0.0)
                }
            }
            
            return health_status
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def shutdown(self, graceful_timeout: int = 30):
        """Gracefully shutdown the WebSocket server."""
        logger.info("Starting WebSocket server shutdown", timeout=graceful_timeout)
        
        try:
            # Stop accepting new connections
            self.is_running = False
            
            # Notify all connected clients about shutdown
            shutdown_message = {
                "type": MessageType.SERVER_SHUTDOWN.value,
                "data": {
                    "message": "Server is shutting down",
                    "graceful_timeout": graceful_timeout,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            active_connections = list(self.connection_manager.connections.values())
            if active_connections:
                logger.info(f"Notifying {len(active_connections)} active connections about shutdown")
                
                # Send shutdown notification to all connections
                notification_tasks = []
                for connection in active_connections:
                    if connection.state == ConnectionState.AUTHENTICATED:
                        task = asyncio.create_task(
                            connection.websocket.send_text(json.dumps(shutdown_message))
                        )
                        notification_tasks.append(task)
                
                # Wait for notifications to be sent (with timeout)
                if notification_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*notification_tasks, return_exceptions=True),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Shutdown notification timeout")
            
            # Give clients time to disconnect gracefully
            await asyncio.sleep(2)
            
            # Force disconnect any remaining connections
            remaining_connections = list(self.connection_manager.connections.values())
            if remaining_connections:
                logger.info(f"Force disconnecting {len(remaining_connections)} remaining connections")
                
                disconnect_tasks = []
                for connection in remaining_connections:
                    task = asyncio.create_task(
                        self.connection_manager.disconnect(
                            connection.connection_id,
                            "Server shutdown"
                        )
                    )
                    disconnect_tasks.append(task)
                
                # Wait for disconnections (with timeout)
                if disconnect_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*disconnect_tasks, return_exceptions=True),
                            timeout=min(graceful_timeout, 10)
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Force disconnect timeout")
            
            # Cleanup resources
            if hasattr(self.connection_manager, 'cleanup'):
                await self.connection_manager.cleanup()
            
            logger.info("WebSocket server shutdown completed")
            
        except Exception as e:
            logger.error("Error during WebSocket server shutdown", error=str(e))
            raise


# Export main components
__all__ = [
    "WebSocketServer",
    "ConnectionManager", 
    "WebSocketConnection",
    "ConnectionState",
    "MessageType",
    "AuthenticationHandler",
    "RateLimiter",
    "MessageRouter"
]