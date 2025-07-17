"""
Neo4j Client wrapper for the Tyra MCP Memory Server.

Provides a high-level interface to Neo4j graph operations through the
graph engine provider system.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..interfaces.graph_engine import Entity, Relationship, GraphEngine
from ..providers.graph_engines.neo4j import Neo4jEngine
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """
    High-level Neo4j client that wraps the Neo4j graph engine.
    
    Provides simplified access to graph operations for API endpoints
    and other high-level components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine: Optional[Neo4jEngine] = None
        self._initialized = False
        
        # Performance tracking
        self._operation_count = 0
        self._total_time = 0.0
        self._error_count = 0
    
    async def initialize(self) -> None:
        """Initialize the Neo4j client."""
        if self._initialized:
            return
        
        try:
            self.engine = Neo4jEngine()
            await self.engine.initialize(self.config)
            self._initialized = True
            
            logger.info(
                "Neo4j client initialized",
                host=self.config.get("host"),
                port=self.config.get("port"),
                database=self.config.get("database")
            )
            
        except Exception as e:
            logger.error("Failed to initialize Neo4j client", error=str(e))
            raise
    
    async def _ensure_initialized(self) -> None:
        """Ensure client is initialized before operations."""
        if not self._initialized:
            await self.initialize()
    
    async def _track_operation(self, operation_name: str, func, *args, **kwargs):
        """Track operation performance and errors."""
        start_time = time.time()
        self._operation_count += 1
        
        try:
            result = await func(*args, **kwargs)
            
            # Update timing stats
            operation_time = time.time() - start_time
            self._total_time += operation_time
            
            logger.debug(
                "Neo4j operation completed",
                operation=operation_name,
                time=operation_time,
                count=self._operation_count
            )
            
            return result
            
        except Exception as e:
            self._error_count += 1
            logger.error(
                "Neo4j operation failed",
                operation=operation_name,
                error=str(e),
                error_count=self._error_count
            )
            raise
    
    async def create_entity(self, entity_data: Dict[str, Any]) -> str:
        """Create a new entity in the graph."""
        await self._ensure_initialized()
        
        entity = Entity(
            id=entity_data.get("id"),
            name=entity_data.get("name"),
            entity_type=entity_data.get("entity_type", "unknown"),
            properties=entity_data.get("properties", {}),
            confidence=entity_data.get("confidence", 1.0)
        )
        
        return await self._track_operation(
            "create_entity",
            self.engine.create_entity,
            entity
        )
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity by ID."""
        await self._ensure_initialized()
        
        entity = await self._track_operation(
            "get_entity",
            self.engine.get_entity,
            entity_id
        )
        
        if entity:
            return {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "properties": entity.properties,
                "created_at": entity.created_at,
                "updated_at": entity.updated_at,
                "confidence": entity.confidence
            }
        
        return None
    
    async def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing entity."""
        await self._ensure_initialized()
        
        # Get existing entity
        existing_entity = await self.engine.get_entity(entity_id)
        if not existing_entity:
            return False
        
        # Apply updates
        existing_entity.name = updates.get("name", existing_entity.name)
        existing_entity.entity_type = updates.get("entity_type", existing_entity.entity_type)
        existing_entity.confidence = updates.get("confidence", existing_entity.confidence)
        
        # Update properties
        if "properties" in updates:
            existing_entity.properties.update(updates["properties"])
        
        existing_entity.updated_at = datetime.utcnow()
        
        return await self._track_operation(
            "update_entity",
            self.engine.update_entity,
            existing_entity
        )
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        await self._ensure_initialized()
        
        return await self._track_operation(
            "delete_entity",
            self.engine.delete_entity,
            entity_id
        )
    
    async def create_relationship(self, relationship_data: Dict[str, Any]) -> str:
        """Create a new relationship between entities."""
        await self._ensure_initialized()
        
        relationship = Relationship(
            id=relationship_data.get("id"),
            source_entity_id=relationship_data["source_entity_id"],
            target_entity_id=relationship_data["target_entity_id"],
            relationship_type=relationship_data.get("relationship_type", "RELATED_TO"),
            properties=relationship_data.get("properties", {}),
            confidence=relationship_data.get("confidence", 1.0),
            valid_from=relationship_data.get("valid_from"),
            valid_to=relationship_data.get("valid_to")
        )
        
        return await self._track_operation(
            "create_relationship",
            self.engine.create_relationship,
            relationship
        )
    
    async def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a relationship by ID."""
        await self._ensure_initialized()
        
        relationship = await self._track_operation(
            "get_relationship",
            self.engine.get_relationship,
            relationship_id
        )
        
        if relationship:
            return {
                "id": relationship.id,
                "source_entity_id": relationship.source_entity_id,
                "target_entity_id": relationship.target_entity_id,
                "relationship_type": relationship.relationship_type,
                "properties": relationship.properties,
                "created_at": relationship.created_at,
                "updated_at": relationship.updated_at,
                "confidence": relationship.confidence,
                "valid_from": relationship.valid_from,
                "valid_to": relationship.valid_to
            }
        
        return None
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        await self._ensure_initialized()
        
        return await self._track_operation(
            "delete_relationship",
            self.engine.delete_relationship,
            relationship_id
        )
    
    async def find_entities(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find entities matching criteria."""
        await self._ensure_initialized()
        
        entities = await self._track_operation(
            "find_entities",
            self.engine.find_entities,
            entity_type,
            properties,
            limit
        )
        
        return [
            {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "properties": entity.properties,
                "created_at": entity.created_at,
                "updated_at": entity.updated_at,
                "confidence": entity.confidence
            }
            for entity in entities
        ]
    
    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        await self._ensure_initialized()
        
        relationships = await self._track_operation(
            "get_entity_relationships",
            self.engine.get_entity_relationships,
            entity_id,
            relationship_type,
            direction
        )
        
        return [
            {
                "id": rel.id,
                "source_entity_id": rel.source_entity_id,
                "target_entity_id": rel.target_entity_id,
                "relationship_type": rel.relationship_type,
                "properties": rel.properties,
                "created_at": rel.created_at,
                "updated_at": rel.updated_at,
                "confidence": rel.confidence,
                "valid_from": rel.valid_from,
                "valid_to": rel.valid_to
            }
            for rel in relationships
        ]
    
    async def get_connected_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get entities connected to the given entity."""
        await self._ensure_initialized()
        
        entities = await self._track_operation(
            "get_connected_entities",
            self.engine.get_connected_entities,
            entity_id,
            relationship_type,
            max_depth
        )
        
        return [
            {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "properties": entity.properties,
                "created_at": entity.created_at,
                "updated_at": entity.updated_at,
                "confidence": entity.confidence
            }
            for entity in entities
        ]
    
    async def find_path(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_depth: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities."""
        await self._ensure_initialized()
        
        relationships = await self._track_operation(
            "find_path",
            self.engine.find_path,
            source_entity_id,
            target_entity_id,
            max_depth
        )
        
        if relationships:
            return [
                {
                    "id": rel.id,
                    "source_entity_id": rel.source_entity_id,
                    "target_entity_id": rel.target_entity_id,
                    "relationship_type": rel.relationship_type,
                    "properties": rel.properties,
                    "created_at": rel.created_at,
                    "updated_at": rel.updated_at,
                    "confidence": rel.confidence,
                    "valid_from": rel.valid_from,
                    "valid_to": rel.valid_to
                }
                for rel in relationships
            ]
        
        return None
    
    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a raw Cypher query."""
        await self._ensure_initialized()
        
        return await self._track_operation(
            "execute_cypher",
            self.engine.execute_cypher,
            query,
            parameters
        )
    
    async def get_entity_timeline(
        self,
        entity_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get temporal relationships for an entity."""
        await self._ensure_initialized()
        
        relationships = await self._track_operation(
            "get_entity_timeline",
            self.engine.get_entity_timeline,
            entity_id,
            start_time,
            end_time
        )
        
        return [
            {
                "id": rel.id,
                "source_entity_id": rel.source_entity_id,
                "target_entity_id": rel.target_entity_id,
                "relationship_type": rel.relationship_type,
                "properties": rel.properties,
                "created_at": rel.created_at,
                "updated_at": rel.updated_at,
                "confidence": rel.confidence,
                "valid_from": rel.valid_from,
                "valid_to": rel.valid_to
            }
            for rel in relationships
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the Neo4j client."""
        try:
            await self._ensure_initialized()
            
            engine_health = await self.engine.health_check()
            
            return {
                "status": engine_health.get("status", "unknown"),
                "client_stats": {
                    "operation_count": self._operation_count,
                    "total_time": self._total_time,
                    "avg_time": self._total_time / max(self._operation_count, 1),
                    "error_count": self._error_count,
                    "error_rate": self._error_count / max(self._operation_count, 1)
                },
                "engine_health": engine_health
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "client_stats": {
                    "operation_count": self._operation_count,
                    "error_count": self._error_count
                }
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        try:
            await self._ensure_initialized()
            
            engine_stats = await self.engine.get_stats()
            
            return {
                "client_stats": {
                    "operation_count": self._operation_count,
                    "total_time": self._total_time,
                    "avg_time": self._total_time / max(self._operation_count, 1),
                    "error_count": self._error_count,
                    "error_rate": self._error_count / max(self._operation_count, 1)
                },
                "engine_stats": engine_stats
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "client_stats": {
                    "operation_count": self._operation_count,
                    "error_count": self._error_count
                }
            }
    
    async def close(self) -> None:
        """Close the Neo4j client."""
        try:
            if self.engine:
                await self.engine.close()
            
            logger.info(
                "Neo4j client closed",
                operation_count=self._operation_count,
                error_count=self._error_count
            )
            
        except Exception as e:
            logger.error("Failed to close Neo4j client", error=str(e))