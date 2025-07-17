"""
Neo4j graph engine implementation with Graphiti integration.

High-performance graph database provider with temporal knowledge graphs,
entity relationship tracking, and advanced Cypher query optimization.
"""

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from ...interfaces.graph_engine import (
    Entity,
    GraphEngine,
    GraphEngineError,
    GraphEngineInitializationError,
    GraphEngineOperationError,
    GraphSearchResult,
)
from ...interfaces.graph_engine import Relationship as RelationshipInterface
from ...interfaces.graph_engine import (
    RelationshipType,
)
from ...utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jEngine(GraphEngine):
    """
    Advanced Neo4j graph engine with temporal knowledge graphs.

    Features:
    - Temporal relationship tracking with validity periods
    - High-performance Cypher query execution
    - Entity relationship optimization
    - Batch operations for large datasets
    - Advanced graph analytics and traversal
    - Integration with Graphiti framework
    - Comprehensive monitoring and statistics
    """

    def __init__(self):
        self.driver: Optional[AsyncDriver] = None
        self.config: Dict[str, Any] = {}

        # Performance tracking
        self._total_queries: int = 0
        self._total_entities: int = 0
        self._total_relationships: int = 0
        self._avg_query_time: float = 0.0
        self._error_count: int = 0

        # Schema tracking
        self._entity_types: set = set()
        self._relationship_types: set = set()

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Neo4j graph engine."""
        try:
            self.config = config

            # Build connection URI
            host = config.get("host", "localhost")
            port = config.get("port", 7687)
            scheme = "neo4j+s" if config.get("encrypted", False) else "neo4j"
            uri = f"{scheme}://{host}:{port}"

            # Initialize driver with auth
            username = config.get("username", "neo4j")
            password = config.get("password", "neo4j")
            
            # Driver configuration
            driver_config = {
                "max_connection_lifetime": config.get("connection_timeout", 30) * 60,
                "max_connection_pool_size": config.get("pool_size", 100),
                "connection_acquisition_timeout": config.get("acquisition_timeout", 60),
                "connection_timeout": config.get("connection_timeout", 30),
                "keep_alive": config.get("keep_alive", True),
            }

            self.driver = AsyncGraphDatabase.driver(
                uri,
                auth=(username, password),
                **driver_config
            )

            # Verify connectivity
            await self.driver.verify_connectivity()

            # Initialize schema and indexes
            await self._initialize_schema()
            await self._create_indexes()

            logger.info(
                "Neo4j engine initialized",
                host=host,
                port=port,
                encrypted=config.get("encrypted", False),
            )

        except Exception as e:
            logger.error("Failed to initialize Neo4j engine", error=str(e))
            raise GraphEngineInitializationError(f"Neo4j initialization failed: {e}")

    async def _initialize_schema(self) -> None:
        """Initialize graph schema and constraints."""
        schema_queries = [
            # Create constraints for entity uniqueness
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            # Create indexes for performance
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_created_index IF NOT EXISTS FOR (e:Entity) ON (e.created_at)",
            "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
        ]

        for query in schema_queries:
            try:
                await self._execute_query(query)
            except Exception as e:
                # Some constraints/indexes might already exist
                logger.debug(f"Schema query warning: {e}")

    async def _create_indexes(self) -> None:
        """Create optimized indexes for graph operations."""
        index_queries = [
            # Text indexes for search
            "CREATE TEXT INDEX entity_name_text IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            # Composite indexes for complex queries
            "CREATE INDEX entity_type_created IF NOT EXISTS FOR (n:Entity) ON (n.entity_type, n.created_at)",
            # Range indexes for temporal queries
            "CREATE INDEX relationship_valid_from IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_from)",
            "CREATE INDEX relationship_valid_to IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_to)",
            "CREATE INDEX relationship_type IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.relationship_type)",
            "CREATE INDEX relationship_confidence IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.confidence)",
        ]

        for query in index_queries:
            try:
                await self._execute_query(query)
            except Exception as e:
                logger.debug(f"Index creation warning: {e}")

    async def _execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query with performance tracking."""
        start_time = time.time()

        try:
            self._total_queries += 1

            async with self.driver.session() as session:
                result = await session.run(query, parameters or {})
                records = await result.data()

            # Update performance tracking
            query_time = time.time() - start_time
            self._avg_query_time = (
                self._avg_query_time * (self._total_queries - 1) + query_time
            ) / self._total_queries

            logger.debug(
                "Cypher query executed",
                query=query[:100] + "..." if len(query) > 100 else query,
                time=query_time,
                results=len(records),
            )

            return records

        except ServiceUnavailable as e:
            self._error_count += 1
            logger.error("Neo4j service unavailable", error=str(e))
            raise GraphEngineOperationError(f"Neo4j service unavailable: {e}")
        except SessionExpired as e:
            self._error_count += 1
            logger.error("Neo4j session expired", error=str(e))
            raise GraphEngineOperationError(f"Neo4j session expired: {e}")
        except Neo4jError as e:
            self._error_count += 1
            logger.error(
                "Neo4j query failed",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Query execution failed: {e}")
        except Exception as e:
            self._error_count += 1
            logger.error(
                "Unexpected error in query execution",
                query=query[:100] + "..." if len(query) > 100 else query,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Query execution failed: {e}")

    async def create_entity(self, entity: Entity) -> str:
        """Create a new entity in the graph."""
        try:
            # Prepare entity properties
            properties = entity.properties.copy() if entity.properties else {}
            properties.update(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "created_at": (entity.created_at or datetime.utcnow()).isoformat(),
                    "updated_at": (entity.updated_at or datetime.utcnow()).isoformat(),
                    "confidence": entity.confidence or 1.0,
                }
            )

            # Serialize complex properties to JSON
            for key, value in properties.items():
                if isinstance(value, (dict, list)):
                    properties[key] = json.dumps(value)

            query = """
            MERGE (e:Entity {id: $id})
            SET e = $properties
            RETURN e.id as id
            """

            result = await self._execute_query(query, {"id": entity.id, "properties": properties})

            # Track entity type
            self._entity_types.add(entity.entity_type)
            self._total_entities += 1

            logger.debug(
                "Entity created",
                entity_id=entity.id,
                entity_type=entity.entity_type,
                name=entity.name,
            )

            return entity.id

        except Exception as e:
            logger.error("Failed to create entity", entity_id=entity.id, error=str(e))
            raise GraphEngineOperationError(f"Entity creation failed: {e}")

    async def create_entities(self, entities: List[Entity]) -> List[str]:
        """Create multiple entities efficiently using batch operations."""
        if not entities:
            return []

        try:
            # Prepare batch data
            entity_data = []
            for entity in entities:
                properties = entity.properties.copy() if entity.properties else {}
                properties.update(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "created_at": (
                            entity.created_at or datetime.utcnow()
                        ).isoformat(),
                        "updated_at": (
                            entity.updated_at or datetime.utcnow()
                        ).isoformat(),
                        "confidence": entity.confidence or 1.0,
                    }
                )
                
                # Serialize complex properties
                for key, value in properties.items():
                    if isinstance(value, (dict, list)):
                        properties[key] = json.dumps(value)
                
                entity_data.append(properties)
                self._entity_types.add(entity.entity_type)

            # Batch create query
            query = """
            UNWIND $entities as entity_data
            MERGE (e:Entity {id: entity_data.id})
            SET e = entity_data
            RETURN e.id as id
            """

            result = await self._execute_query(query, {"entities": entity_data})

            self._total_entities += len(entities)

            logger.info(
                "Batch entities created",
                count=len(entities),
                unique_types=len(set(e.entity_type for e in entities)),
            )

            return [entity.id for entity in entities]

        except Exception as e:
            logger.error(
                "Failed to create entities in batch", count=len(entities), error=str(e)
            )
            raise GraphEngineOperationError(f"Batch entity creation failed: {e}")

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})
            RETURN e
            """

            result = await self._execute_query(query, {"entity_id": entity_id})

            if result:
                node = result[0]["e"]
                
                # Deserialize complex properties
                properties = {}
                for key, value in node.items():
                    if key not in ["id", "name", "entity_type", "created_at", "updated_at", "confidence"]:
                        try:
                            properties[key] = json.loads(value) if isinstance(value, str) and value.startswith(("{", "[")) else value
                        except:
                            properties[key] = value
                
                return Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=node["entity_type"],
                    properties=properties,
                    created_at=(
                        datetime.fromisoformat(node["created_at"])
                        if node.get("created_at")
                        else None
                    ),
                    updated_at=(
                        datetime.fromisoformat(node["updated_at"])
                        if node.get("updated_at")
                        else None
                    ),
                    confidence=node.get("confidence"),
                )

            return None

        except Exception as e:
            logger.error("Failed to get entity", entity_id=entity_id, error=str(e))
            raise GraphEngineOperationError(f"Get entity failed: {e}")

    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        try:
            properties = entity.properties.copy() if entity.properties else {}
            properties.update(
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "updated_at": datetime.utcnow().isoformat(),
                    "confidence": entity.confidence or 1.0,
                }
            )

            # Serialize complex properties
            for key, value in properties.items():
                if isinstance(value, (dict, list)):
                    properties[key] = json.dumps(value)

            query = """
            MATCH (e:Entity {id: $id})
            SET e += $properties
            RETURN e.id as id
            """

            result = await self._execute_query(
                query, {"id": entity.id, "properties": properties}
            )

            return len(result) > 0

        except Exception as e:
            logger.error("Failed to update entity", entity_id=entity.id, error=str(e))
            raise GraphEngineOperationError(f"Entity update failed: {e}")

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})
            DETACH DELETE e
            RETURN count(e) as deleted_count
            """

            result = await self._execute_query(query, {"entity_id": entity_id})

            deleted = result[0]["deleted_count"] > 0 if result else False

            if deleted:
                self._total_entities = max(0, self._total_entities - 1)

            return deleted

        except Exception as e:
            logger.error("Failed to delete entity", entity_id=entity_id, error=str(e))
            raise GraphEngineOperationError(f"Entity deletion failed: {e}")

    async def create_relationship(self, relationship: RelationshipInterface) -> str:
        """Create a new relationship between entities."""
        try:
            # Prepare relationship properties
            properties = (
                relationship.properties.copy() if relationship.properties else {}
            )
            properties.update(
                {
                    "id": relationship.id,
                    "relationship_type": relationship.relationship_type,
                    "created_at": (
                        relationship.created_at or datetime.utcnow()
                    ).isoformat(),
                    "updated_at": (
                        relationship.updated_at or datetime.utcnow()
                    ).isoformat(),
                    "confidence": relationship.confidence or 1.0,
                    "valid_from": (
                        relationship.valid_from.isoformat()
                        if relationship.valid_from
                        else None
                    ),
                    "valid_to": (
                        relationship.valid_to.isoformat()
                        if relationship.valid_to
                        else None
                    ),
                }
            )

            # Remove None values and serialize complex properties
            clean_properties = {}
            for k, v in properties.items():
                if v is not None:
                    if isinstance(v, (dict, list)):
                        clean_properties[k] = json.dumps(v)
                    else:
                        clean_properties[k] = v

            query = """
            MATCH (source:Entity {id: $source_id})
            MATCH (target:Entity {id: $target_id})
            CREATE (source)-[r:RELATIONSHIP $properties]->(target)
            RETURN r.id as id
            """

            parameters = {
                "source_id": relationship.source_entity_id,
                "target_id": relationship.target_entity_id,
                "properties": clean_properties,
            }

            result = await self._execute_query(query, parameters)

            # Track relationship type
            self._relationship_types.add(relationship.relationship_type)
            self._total_relationships += 1

            logger.debug(
                "Relationship created",
                relationship_id=relationship.id,
                type=relationship.relationship_type,
                source=relationship.source_entity_id,
                target=relationship.target_entity_id,
            )

            return relationship.id

        except Exception as e:
            logger.error(
                "Failed to create relationship",
                relationship_id=relationship.id,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Relationship creation failed: {e}")

    async def create_relationships(
        self, relationships: List[RelationshipInterface]
    ) -> List[str]:
        """Create multiple relationships efficiently."""
        if not relationships:
            return []

        try:
            # Prepare batch data
            rel_data = []
            for rel in relationships:
                properties = rel.properties.copy() if rel.properties else {}
                properties.update(
                    {
                        "id": rel.id,
                        "relationship_type": rel.relationship_type,
                        "created_at": (rel.created_at or datetime.utcnow()).isoformat(),
                        "updated_at": (rel.updated_at or datetime.utcnow()).isoformat(),
                        "confidence": rel.confidence or 1.0,
                        "valid_from": (
                            rel.valid_from.isoformat() if rel.valid_from else None
                        ),
                        "valid_to": rel.valid_to.isoformat() if rel.valid_to else None,
                    }
                )
                
                # Clean and serialize properties
                clean_properties = {}
                for k, v in properties.items():
                    if v is not None:
                        if isinstance(v, (dict, list)):
                            clean_properties[k] = json.dumps(v)
                        else:
                            clean_properties[k] = v
                
                rel_data.append({
                    "source_id": rel.source_entity_id,
                    "target_id": rel.target_entity_id,
                    "properties": clean_properties
                })
                self._relationship_types.add(rel.relationship_type)

            query = """
            UNWIND $relationships as rel_data
            MATCH (source:Entity {id: rel_data.source_id})
            MATCH (target:Entity {id: rel_data.target_id})
            CREATE (source)-[r:RELATIONSHIP]->(target)
            SET r = rel_data.properties
            RETURN r.id as id
            """

            result = await self._execute_query(query, {"relationships": rel_data})

            self._total_relationships += len(relationships)

            logger.info(
                "Batch relationships created",
                count=len(relationships),
                unique_types=len(set(r.relationship_type for r in relationships)),
            )

            return [rel.id for rel in relationships]

        except Exception as e:
            logger.error(
                "Failed to create relationships in batch",
                count=len(relationships),
                error=str(e),
            )
            raise GraphEngineOperationError(f"Batch relationship creation failed: {e}")

    async def get_relationship(
        self, relationship_id: str
    ) -> Optional[RelationshipInterface]:
        """Retrieve a relationship by ID."""
        try:
            query = """
            MATCH (source)-[r:RELATIONSHIP {id: $relationship_id}]->(target)
            RETURN r, source.id as source_id, target.id as target_id
            """

            result = await self._execute_query(
                query, {"relationship_id": relationship_id}
            )

            if result:
                row = result[0]
                rel = row["r"]
                
                # Deserialize complex properties
                properties = {}
                for key, value in rel.items():
                    if key not in ["id", "relationship_type", "created_at", "updated_at", "confidence", "valid_from", "valid_to"]:
                        try:
                            properties[key] = json.loads(value) if isinstance(value, str) and value.startswith(("{", "[")) else value
                        except:
                            properties[key] = value
                
                return RelationshipInterface(
                    id=rel["id"],
                    source_entity_id=row["source_id"],
                    target_entity_id=row["target_id"],
                    relationship_type=rel["relationship_type"],
                    properties=properties,
                    created_at=(
                        datetime.fromisoformat(rel["created_at"])
                        if rel.get("created_at")
                        else None
                    ),
                    updated_at=(
                        datetime.fromisoformat(rel["updated_at"])
                        if rel.get("updated_at")
                        else None
                    ),
                    confidence=rel.get("confidence"),
                    valid_from=(
                        datetime.fromisoformat(rel["valid_from"])
                        if rel.get("valid_from")
                        else None
                    ),
                    valid_to=(
                        datetime.fromisoformat(rel["valid_to"])
                        if rel.get("valid_to")
                        else None
                    ),
                )

            return None

        except Exception as e:
            logger.error(
                "Failed to get relationship",
                relationship_id=relationship_id,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Get relationship failed: {e}")

    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        try:
            query = """
            MATCH ()-[r:RELATIONSHIP {id: $relationship_id}]->()
            DELETE r
            RETURN count(r) as deleted_count
            """

            result = await self._execute_query(
                query, {"relationship_id": relationship_id}
            )

            deleted = result[0]["deleted_count"] > 0 if result else False

            if deleted:
                self._total_relationships = max(0, self._total_relationships - 1)

            return deleted

        except Exception as e:
            logger.error(
                "Failed to delete relationship",
                relationship_id=relationship_id,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Relationship deletion failed: {e}")

    async def find_entities(
        self,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """Find entities matching criteria."""
        try:
            where_clauses = []
            parameters = {"limit": limit}

            if entity_type:
                where_clauses.append("e.entity_type = $entity_type")
                parameters["entity_type"] = entity_type

            if properties:
                for key, value in properties.items():
                    param_name = f"prop_{key}"
                    where_clauses.append(f"e.{key} = ${param_name}")
                    parameters[param_name] = value

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            query = f"""
            MATCH (e:Entity)
            WHERE {where_clause}
            RETURN e
            ORDER BY e.created_at DESC
            LIMIT $limit
            """

            result = await self._execute_query(query, parameters)

            entities = []
            for row in result:
                node = row["e"]
                
                # Deserialize complex properties
                properties = {}
                for key, value in node.items():
                    if key not in ["id", "name", "entity_type", "created_at", "updated_at", "confidence"]:
                        try:
                            properties[key] = json.loads(value) if isinstance(value, str) and value.startswith(("{", "[")) else value
                        except:
                            properties[key] = value
                
                entities.append(
                    Entity(
                        id=node["id"],
                        name=node["name"],
                        entity_type=node["entity_type"],
                        properties=properties,
                        created_at=(
                            datetime.fromisoformat(node["created_at"])
                            if node.get("created_at")
                            else None
                        ),
                        updated_at=(
                            datetime.fromisoformat(node["updated_at"])
                            if node.get("updated_at")
                            else None
                        ),
                        confidence=node.get("confidence"),
                    )
                )

            return entities

        except Exception as e:
            logger.error(
                "Failed to find entities", entity_type=entity_type, error=str(e)
            )
            raise GraphEngineOperationError(f"Entity search failed: {e}")

    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[RelationshipInterface]:
        """Get all relationships for an entity."""
        try:
            # Build direction pattern
            if direction == "outgoing":
                pattern = "(e:Entity {id: $entity_id})-[r:RELATIONSHIP]->(target)"
                source_id = "e.id"
                target_id = "target.id"
            elif direction == "incoming":
                pattern = "(source)-[r:RELATIONSHIP]->(e:Entity {id: $entity_id})"
                source_id = "source.id"
                target_id = "e.id"
            else:  # both
                pattern = "(source)-[r:RELATIONSHIP]-(target)"
                where_clause = "(source.id = $entity_id OR target.id = $entity_id)"
                source_id = "source.id"
                target_id = "target.id"

            parameters = {"entity_id": entity_id}

            if relationship_type:
                rel_filter = " AND r.relationship_type = $relationship_type"
                parameters["relationship_type"] = relationship_type
            else:
                rel_filter = ""

            if direction == "both":
                query = f"""
                MATCH {pattern}
                WHERE {where_clause}{rel_filter}
                RETURN r, {source_id} as source_id, {target_id} as target_id
                ORDER BY r.created_at DESC
                """
            else:
                query = f"""
                MATCH {pattern}
                WHERE 1=1{rel_filter}
                RETURN r, {source_id} as source_id, {target_id} as target_id
                ORDER BY r.created_at DESC
                """

            result = await self._execute_query(query, parameters)

            relationships = []
            for row in result:
                rel = row["r"]
                
                # Deserialize complex properties
                properties = {}
                for key, value in rel.items():
                    if key not in ["id", "relationship_type", "created_at", "updated_at", "confidence", "valid_from", "valid_to"]:
                        try:
                            properties[key] = json.loads(value) if isinstance(value, str) and value.startswith(("{", "[")) else value
                        except:
                            properties[key] = value
                
                relationships.append(
                    RelationshipInterface(
                        id=rel["id"],
                        source_entity_id=row["source_id"],
                        target_entity_id=row["target_id"],
                        relationship_type=rel["relationship_type"],
                        properties=properties,
                        created_at=(
                            datetime.fromisoformat(rel["created_at"])
                            if rel.get("created_at")
                            else None
                        ),
                        updated_at=(
                            datetime.fromisoformat(rel["updated_at"])
                            if rel.get("updated_at")
                            else None
                        ),
                        confidence=rel.get("confidence"),
                        valid_from=(
                            datetime.fromisoformat(rel["valid_from"])
                            if rel.get("valid_from")
                            else None
                        ),
                        valid_to=(
                            datetime.fromisoformat(rel["valid_to"])
                            if rel.get("valid_to")
                            else None
                        ),
                    )
                )

            return relationships

        except Exception as e:
            logger.error(
                "Failed to get entity relationships", entity_id=entity_id, error=str(e)
            )
            raise GraphEngineOperationError(f"Get relationships failed: {e}")

    async def get_connected_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1,
    ) -> List[Entity]:
        """Get entities connected to the given entity."""
        try:
            parameters = {"entity_id": entity_id}

            if relationship_type:
                rel_filter = "[r:RELATIONSHIP {relationship_type: $relationship_type}]"
                parameters["relationship_type"] = relationship_type
            else:
                rel_filter = "[r:RELATIONSHIP]"

            query = f"""
            MATCH (start:Entity {{id: $entity_id}})-{rel_filter}*1..{max_depth}-(connected:Entity)
            WHERE connected.id <> $entity_id
            RETURN DISTINCT connected
            ORDER BY connected.name
            """

            result = await self._execute_query(query, parameters)

            entities = []
            for row in result:
                node = row["connected"]
                
                # Deserialize complex properties
                properties = {}
                for key, value in node.items():
                    if key not in ["id", "name", "entity_type", "created_at", "updated_at", "confidence"]:
                        try:
                            properties[key] = json.loads(value) if isinstance(value, str) and value.startswith(("{", "[")) else value
                        except:
                            properties[key] = value
                
                entities.append(
                    Entity(
                        id=node["id"],
                        name=node["name"],
                        entity_type=node["entity_type"],
                        properties=properties,
                        created_at=(
                            datetime.fromisoformat(node["created_at"])
                            if node.get("created_at")
                            else None
                        ),
                        updated_at=(
                            datetime.fromisoformat(node["updated_at"])
                            if node.get("updated_at")
                            else None
                        ),
                        confidence=node.get("confidence"),
                    )
                )

            return entities

        except Exception as e:
            logger.error(
                "Failed to get connected entities", entity_id=entity_id, error=str(e)
            )
            raise GraphEngineOperationError(f"Get connected entities failed: {e}")

    async def find_path(
        self, source_entity_id: str, target_entity_id: str, max_depth: int = 3
    ) -> Optional[List[RelationshipInterface]]:
        """Find shortest path between two entities."""
        try:
            query = f"""
            MATCH path = shortestPath((source:Entity {{id: $source_id}})-[*1..{max_depth}]-(target:Entity {{id: $target_id}}))
            WITH relationships(path) as rels, nodes(path) as nodes
            UNWIND range(0, size(rels)-1) as idx
            WITH rels[idx] as r, nodes[idx] as source, nodes[idx+1] as target
            RETURN r, source.id as source_id, target.id as target_id
            """

            parameters = {"source_id": source_entity_id, "target_id": target_entity_id}

            result = await self._execute_query(query, parameters)

            if result:
                relationships = []
                for row in result:
                    rel = row["r"]
                    
                    # Deserialize complex properties
                    properties = {}
                    for key, value in rel.items():
                        if key not in ["id", "relationship_type", "created_at", "updated_at", "confidence", "valid_from", "valid_to"]:
                            try:
                                properties[key] = json.loads(value) if isinstance(value, str) and value.startswith(("{", "[")) else value
                            except:
                                properties[key] = value
                    
                    relationships.append(
                        RelationshipInterface(
                            id=rel.get("id", str(uuid.uuid4())),
                            source_entity_id=row["source_id"],
                            target_entity_id=row["target_id"],
                            relationship_type=rel.get("relationship_type", "RELATED_TO"),
                            properties=properties,
                            created_at=(
                                datetime.fromisoformat(rel["created_at"])
                                if rel.get("created_at")
                                else None
                            ),
                            updated_at=(
                                datetime.fromisoformat(rel["updated_at"])
                                if rel.get("updated_at")
                                else None
                            ),
                            confidence=rel.get("confidence"),
                            valid_from=(
                                datetime.fromisoformat(rel["valid_from"])
                                if rel.get("valid_from")
                                else None
                            ),
                            valid_to=(
                                datetime.fromisoformat(rel["valid_to"])
                                if rel.get("valid_to")
                                else None
                            ),
                        )
                    )
                return relationships

            return None

        except Exception as e:
            logger.error(
                "Failed to find path",
                source=source_entity_id,
                target=target_entity_id,
                error=str(e),
            )
            raise GraphEngineOperationError(f"Path finding failed: {e}")

    async def execute_cypher(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a raw Cypher query."""
        return await self._execute_query(query, parameters)

    async def get_entity_timeline(
        self,
        entity_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[RelationshipInterface]:
        """Get temporal relationships for an entity."""
        try:
            where_clauses = ["(source.id = $entity_id OR target.id = $entity_id)"]
            parameters = {"entity_id": entity_id}

            if start_time:
                where_clauses.append(
                    "(r.valid_from IS NULL OR r.valid_from >= $start_time)"
                )
                parameters["start_time"] = start_time.isoformat()

            if end_time:
                where_clauses.append("(r.valid_to IS NULL OR r.valid_to <= $end_time)")
                parameters["end_time"] = end_time.isoformat()

            where_clause = " AND ".join(where_clauses)

            query = f"""
            MATCH (source)-[r:RELATIONSHIP]-(target)
            WHERE {where_clause}
            RETURN r, source.id as source_id, target.id as target_id
            ORDER BY COALESCE(r.valid_from, r.created_at) ASC
            """

            result = await self._execute_query(query, parameters)

            relationships = []
            for row in result:
                rel = row["r"]
                
                # Deserialize complex properties
                properties = {}
                for key, value in rel.items():
                    if key not in ["id", "relationship_type", "created_at", "updated_at", "confidence", "valid_from", "valid_to"]:
                        try:
                            properties[key] = json.loads(value) if isinstance(value, str) and value.startswith(("{", "[")) else value
                        except:
                            properties[key] = value
                
                relationships.append(
                    RelationshipInterface(
                        id=rel["id"],
                        source_entity_id=row["source_id"],
                        target_entity_id=row["target_id"],
                        relationship_type=rel["relationship_type"],
                        properties=properties,
                        created_at=(
                            datetime.fromisoformat(rel["created_at"])
                            if rel.get("created_at")
                            else None
                        ),
                        updated_at=(
                            datetime.fromisoformat(rel["updated_at"])
                            if rel.get("updated_at")
                            else None
                        ),
                        confidence=rel.get("confidence"),
                        valid_from=(
                            datetime.fromisoformat(rel["valid_from"])
                            if rel.get("valid_from")
                            else None
                        ),
                        valid_to=(
                            datetime.fromisoformat(rel["valid_to"])
                            if rel.get("valid_to")
                            else None
                        ),
                    )
                )

            return relationships

        except Exception as e:
            logger.error(
                "Failed to get entity timeline", entity_id=entity_id, error=str(e)
            )
            raise GraphEngineOperationError(f"Timeline query failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            # Check database connection
            test_start = time.time()
            await self.driver.verify_connectivity()
            connectivity_time = time.time() - test_start

            # Get graph statistics
            stats_query = """
            MATCH (e:Entity)
            WITH count(DISTINCT e) as entity_count
            OPTIONAL MATCH ()-[r:RELATIONSHIP]->()
            WITH entity_count, count(DISTINCT r) as relationship_count
            MATCH (e:Entity)
            WITH entity_count, relationship_count, collect(DISTINCT e.entity_type) as entity_types
            OPTIONAL MATCH ()-[r:RELATIONSHIP]->()
            RETURN
                entity_count,
                relationship_count,
                size(entity_types) as entity_type_count,
                size(collect(DISTINCT r.relationship_type)) as relationship_type_count
            """

            stats = await self._execute_query(stats_query)
            graph_stats = stats[0] if stats else {
                "entity_count": 0,
                "relationship_count": 0,
                "entity_type_count": 0,
                "relationship_type_count": 0
            }

            # Test basic operations
            test_start = time.time()
            await self._execute_query("RETURN 1 as test")
            response_time = time.time() - test_start

            return {
                "status": "healthy",
                "database": {
                    "status": "healthy",
                    "connectivity_time": connectivity_time,
                    "driver_connected": True,
                },
                "response_time": response_time,
                "graph_stats": graph_stats,
                "performance": {
                    "total_queries": self._total_queries,
                    "avg_query_time": self._avg_query_time,
                    "error_count": self._error_count,
                    "error_rate": self._error_count / max(self._total_queries, 1),
                },
                "schema": {
                    "entity_types": list(self._entity_types),
                    "relationship_types": list(self._relationship_types),
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_count": self._error_count,
                "database": {
                    "status": "unhealthy",
                    "driver_connected": self.driver is not None,
                }
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph database statistics."""
        try:
            # Detailed statistics query
            stats_query = """
            MATCH (e:Entity)
            WITH count(DISTINCT e) as total_entities, avg(e.confidence) as avg_entity_confidence
            OPTIONAL MATCH ()-[r:RELATIONSHIP]->()
            WITH total_entities, avg_entity_confidence, 
                 count(DISTINCT r) as total_relationships, 
                 avg(r.confidence) as avg_relationship_confidence
            MATCH (e:Entity)
            WITH total_entities, avg_entity_confidence, total_relationships, avg_relationship_confidence,
                 collect(DISTINCT e.entity_type) as entity_types
            OPTIONAL MATCH ()-[r:RELATIONSHIP]->()
            RETURN
                total_entities,
                total_relationships,
                size(entity_types) as unique_entity_types,
                size(collect(DISTINCT r.relationship_type)) as unique_relationship_types,
                avg_entity_confidence,
                avg_relationship_confidence
            """

            stats = await self._execute_query(stats_query)
            graph_stats = stats[0] if stats else {}

            return {
                "graph_statistics": graph_stats,
                "performance": {
                    "total_queries": self._total_queries,
                    "avg_query_time": self._avg_query_time,
                    "queries_per_second": self._total_queries
                    / max(self._avg_query_time * self._total_queries, 1),
                    "error_count": self._error_count,
                    "error_rate": self._error_count / max(self._total_queries, 1),
                },
                "schema": {
                    "tracked_entity_types": list(self._entity_types),
                    "tracked_relationship_types": list(self._relationship_types),
                },
                "configuration": {
                    "host": self.config.get("host"),
                    "port": self.config.get("port"),
                    "encrypted": self.config.get("encrypted", False),
                },
            }

        except Exception as e:
            logger.error("Failed to get graph stats", error=str(e))
            return {"error": str(e)}

    async def close(self) -> None:
        """Close the graph engine connections."""
        if self.driver:
            await self.driver.close()

        logger.info(
            "Neo4j engine closed",
            total_queries=self._total_queries,
            total_entities=self._total_entities,
            total_relationships=self._total_relationships,
        )