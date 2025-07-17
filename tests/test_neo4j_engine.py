"""
Comprehensive tests for Neo4j graph engine implementation.

Tests all aspects of the Neo4j engine including connection management,
CRUD operations, temporal features, and error handling.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from src.core.providers.graph_engines.neo4j import Neo4jEngine
from src.core.interfaces.graph_engine import (
    Entity,
    Relationship,
    GraphEngineError,
    GraphEngineInitializationError,
    GraphEngineOperationError,
)


class TestNeo4jEngine:
    """Test suite for Neo4j graph engine."""
    
    @pytest.fixture
    async def mock_driver(self):
        """Mock Neo4j driver for testing."""
        driver = AsyncMock()
        driver.verify_connectivity = AsyncMock()
        
        # Mock session
        session = AsyncMock()
        session.run = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        
        driver.session.return_value = session
        
        return driver
    
    @pytest.fixture
    async def neo4j_engine(self, mock_driver):
        """Create Neo4j engine with mocked driver."""
        engine = Neo4jEngine()
        
        with patch('src.core.providers.graph_engines.neo4j.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver_factory.return_value = mock_driver
            
            config = {
                "host": "localhost",
                "port": 7687,
                "username": "neo4j",
                "password": "test",
                "database": "neo4j",
                "encrypted": False
            }
            
            await engine.initialize(config)
            
        return engine, mock_driver
    
    @pytest.fixture
    def sample_entity(self):
        """Sample entity for testing."""
        return Entity(
            id="test-entity-123",
            name="Test Entity",
            entity_type="test",
            properties={"key": "value", "number": 42},
            confidence=0.95
        )
    
    @pytest.fixture
    def sample_relationship(self):
        """Sample relationship for testing."""
        return Relationship(
            id="test-rel-123",
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="RELATED_TO",
            properties={"strength": 0.8},
            confidence=0.9,
            valid_from=datetime.now(),
            valid_to=datetime.now() + timedelta(days=30)
        )


class TestInitialization:
    """Test Neo4j engine initialization."""
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self):
        """Test successful engine initialization."""
        engine = Neo4jEngine()
        
        with patch('src.core.providers.graph_engines.neo4j.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            config = {
                "host": "localhost",
                "port": 7687,
                "username": "neo4j",
                "password": "test"
            }
            
            await engine.initialize(config)
            
            assert engine.driver == mock_driver
            assert engine.config == config
            mock_driver.verify_connectivity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_with_encryption(self):
        """Test initialization with SSL encryption."""
        engine = Neo4jEngine()
        
        with patch('src.core.providers.graph_engines.neo4j.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver_factory.return_value = mock_driver
            
            config = {
                "host": "localhost",
                "port": 7687,
                "username": "neo4j",
                "password": "test",
                "encrypted": True
            }
            
            await engine.initialize(config)
            
            # Verify SSL URI was used
            call_args = mock_driver_factory.call_args
            assert call_args[0][0] == "neo4j+s://localhost:7687"
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test initialization failure handling."""
        engine = Neo4jEngine()
        
        with patch('src.core.providers.graph_engines.neo4j.AsyncGraphDatabase.driver') as mock_driver_factory:
            mock_driver_factory.side_effect = Exception("Connection failed")
            
            config = {
                "host": "localhost",
                "port": 7687,
                "username": "neo4j",
                "password": "test"
            }
            
            with pytest.raises(GraphEngineInitializationError):
                await engine.initialize(config)


class TestEntityOperations:
    """Test entity CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_create_entity(self, neo4j_engine, sample_entity):
        """Test entity creation."""
        engine, mock_driver = neo4j_engine
        
        # Mock successful query execution
        mock_result = [{"id": sample_entity.id}]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.create_entity(sample_entity)
        
        assert result == sample_entity.id
        assert engine._total_entities == 1
        assert sample_entity.entity_type in engine._entity_types
    
    @pytest.mark.asyncio
    async def test_create_entities_batch(self, neo4j_engine):
        """Test batch entity creation."""
        engine, mock_driver = neo4j_engine
        
        entities = [
            Entity(id=f"entity-{i}", name=f"Entity {i}", entity_type="test", properties={})
            for i in range(5)
        ]
        
        # Mock successful batch query
        mock_result = [{"id": entity.id} for entity in entities]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.create_entities(entities)
        
        assert len(result) == 5
        assert engine._total_entities == 5
        assert all(entity_id in result for entity_id in [e.id for e in entities])
    
    @pytest.mark.asyncio
    async def test_get_entity(self, neo4j_engine, sample_entity):
        """Test entity retrieval."""
        engine, mock_driver = neo4j_engine
        
        # Mock entity data from Neo4j
        mock_result = [{
            "e": {
                "id": sample_entity.id,
                "name": sample_entity.name,
                "entity_type": sample_entity.entity_type,
                "confidence": sample_entity.confidence,
                "created_at": datetime.now().isoformat(),
                "key": "value",
                "number": 42
            }
        }]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.get_entity(sample_entity.id)
        
        assert result is not None
        assert result.id == sample_entity.id
        assert result.name == sample_entity.name
        assert result.entity_type == sample_entity.entity_type
        assert result.properties["key"] == "value"
        assert result.properties["number"] == 42
    
    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, neo4j_engine):
        """Test entity retrieval when entity doesn't exist."""
        engine, mock_driver = neo4j_engine
        
        # Mock empty result
        engine._execute_query = AsyncMock(return_value=[])
        
        result = await engine.get_entity("non-existent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_entity(self, neo4j_engine, sample_entity):
        """Test entity update."""
        engine, mock_driver = neo4j_engine
        
        # Mock successful update
        mock_result = [{"id": sample_entity.id}]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        # Update entity properties
        sample_entity.name = "Updated Name"
        sample_entity.properties["new_key"] = "new_value"
        
        result = await engine.update_entity(sample_entity)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_entity(self, neo4j_engine, sample_entity):
        """Test entity deletion."""
        engine, mock_driver = neo4j_engine
        
        # Mock successful deletion
        mock_result = [{"deleted_count": 1}]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.delete_entity(sample_entity.id)
        
        assert result is True


class TestRelationshipOperations:
    """Test relationship CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_create_relationship(self, neo4j_engine, sample_relationship):
        """Test relationship creation."""
        engine, mock_driver = neo4j_engine
        
        # Mock successful query execution
        mock_result = [{"id": sample_relationship.id}]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.create_relationship(sample_relationship)
        
        assert result == sample_relationship.id
        assert engine._total_relationships == 1
        assert sample_relationship.relationship_type in engine._relationship_types
    
    @pytest.mark.asyncio
    async def test_create_relationships_batch(self, neo4j_engine):
        """Test batch relationship creation."""
        engine, mock_driver = neo4j_engine
        
        relationships = [
            Relationship(
                id=f"rel-{i}",
                source_entity_id="entity-1",
                target_entity_id="entity-2",
                relationship_type="RELATED_TO",
                properties={}
            )
            for i in range(3)
        ]
        
        # Mock successful batch query
        mock_result = [{"id": rel.id} for rel in relationships]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.create_relationships(relationships)
        
        assert len(result) == 3
        assert engine._total_relationships == 3
    
    @pytest.mark.asyncio
    async def test_get_relationship(self, neo4j_engine, sample_relationship):
        """Test relationship retrieval."""
        engine, mock_driver = neo4j_engine
        
        # Mock relationship data from Neo4j
        mock_result = [{
            "r": {
                "id": sample_relationship.id,
                "relationship_type": sample_relationship.relationship_type,
                "confidence": sample_relationship.confidence,
                "created_at": datetime.now().isoformat(),
                "valid_from": sample_relationship.valid_from.isoformat(),
                "valid_to": sample_relationship.valid_to.isoformat(),
                "strength": 0.8
            },
            "source_id": sample_relationship.source_entity_id,
            "target_id": sample_relationship.target_entity_id
        }]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.get_relationship(sample_relationship.id)
        
        assert result is not None
        assert result.id == sample_relationship.id
        assert result.relationship_type == sample_relationship.relationship_type
        assert result.source_entity_id == sample_relationship.source_entity_id
        assert result.target_entity_id == sample_relationship.target_entity_id
    
    @pytest.mark.asyncio
    async def test_delete_relationship(self, neo4j_engine, sample_relationship):
        """Test relationship deletion."""
        engine, mock_driver = neo4j_engine
        
        # Mock successful deletion
        mock_result = [{"deleted_count": 1}]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.delete_relationship(sample_relationship.id)
        
        assert result is True


class TestQueryOperations:
    """Test graph query operations."""
    
    @pytest.mark.asyncio
    async def test_find_entities(self, neo4j_engine):
        """Test entity search."""
        engine, mock_driver = neo4j_engine
        
        # Mock search results
        mock_result = [
            {
                "e": {
                    "id": "entity-1",
                    "name": "Entity 1",
                    "entity_type": "test",
                    "confidence": 0.9,
                    "created_at": datetime.now().isoformat()
                }
            },
            {
                "e": {
                    "id": "entity-2",
                    "name": "Entity 2",
                    "entity_type": "test",
                    "confidence": 0.8,
                    "created_at": datetime.now().isoformat()
                }
            }
        ]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.find_entities(entity_type="test", limit=10)
        
        assert len(result) == 2
        assert all(entity.entity_type == "test" for entity in result)
    
    @pytest.mark.asyncio
    async def test_get_entity_relationships(self, neo4j_engine):
        """Test getting entity relationships."""
        engine, mock_driver = neo4j_engine
        
        # Mock relationship results
        mock_result = [
            {
                "r": {
                    "id": "rel-1",
                    "relationship_type": "RELATED_TO",
                    "confidence": 0.9,
                    "created_at": datetime.now().isoformat()
                },
                "source_id": "entity-1",
                "target_id": "entity-2"
            }
        ]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.get_entity_relationships("entity-1")
        
        assert len(result) == 1
        assert result[0].relationship_type == "RELATED_TO"
    
    @pytest.mark.asyncio
    async def test_get_connected_entities(self, neo4j_engine):
        """Test getting connected entities."""
        engine, mock_driver = neo4j_engine
        
        # Mock connected entities
        mock_result = [
            {
                "connected": {
                    "id": "connected-1",
                    "name": "Connected Entity",
                    "entity_type": "test",
                    "confidence": 0.8,
                    "created_at": datetime.now().isoformat()
                }
            }
        ]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.get_connected_entities("entity-1", max_depth=2)
        
        assert len(result) == 1
        assert result[0].id == "connected-1"
    
    @pytest.mark.asyncio
    async def test_find_path(self, neo4j_engine):
        """Test path finding between entities."""
        engine, mock_driver = neo4j_engine
        
        # Mock path results
        mock_result = [
            {
                "r": {
                    "id": "rel-1",
                    "relationship_type": "RELATED_TO",
                    "confidence": 0.9,
                    "created_at": datetime.now().isoformat()
                },
                "source_id": "entity-1",
                "target_id": "entity-2"
            }
        ]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.find_path("entity-1", "entity-2", max_depth=3)
        
        assert result is not None
        assert len(result) == 1
        assert result[0].relationship_type == "RELATED_TO"
    
    @pytest.mark.asyncio
    async def test_execute_cypher(self, neo4j_engine):
        """Test raw Cypher query execution."""
        engine, mock_driver = neo4j_engine
        
        # Mock query result
        mock_result = [{"count": 42}]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.execute_cypher("MATCH (n) RETURN count(n) as count")
        
        assert result == mock_result


class TestTemporalOperations:
    """Test temporal graph operations."""
    
    @pytest.mark.asyncio
    async def test_get_entity_timeline(self, neo4j_engine):
        """Test entity timeline retrieval."""
        engine, mock_driver = neo4j_engine
        
        # Mock timeline results
        mock_result = [
            {
                "r": {
                    "id": "rel-1",
                    "relationship_type": "HAPPENED_AT",
                    "valid_from": "2023-01-01T00:00:00",
                    "valid_to": "2023-12-31T23:59:59",
                    "created_at": datetime.now().isoformat()
                },
                "source_id": "entity-1",
                "target_id": "event-1"
            }
        ]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 12, 31)
        
        result = await engine.get_entity_timeline("entity-1", start_time, end_time)
        
        assert len(result) == 1
        assert result[0].relationship_type == "HAPPENED_AT"


class TestErrorHandling:
    """Test error handling and resilience."""
    
    @pytest.mark.asyncio
    async def test_service_unavailable_error(self, neo4j_engine):
        """Test handling of Neo4j service unavailable errors."""
        engine, mock_driver = neo4j_engine
        
        # Mock service unavailable error
        engine._execute_query = AsyncMock(side_effect=ServiceUnavailable("Service unavailable"))
        
        with pytest.raises(GraphEngineOperationError):
            await engine.create_entity(Entity(
                id="test",
                name="test",
                entity_type="test",
                properties={}
            ))
        
        assert engine._error_count == 1
    
    @pytest.mark.asyncio
    async def test_general_neo4j_error(self, neo4j_engine):
        """Test handling of general Neo4j errors."""
        engine, mock_driver = neo4j_engine
        
        # Mock Neo4j error
        engine._execute_query = AsyncMock(side_effect=Neo4jError("Neo4j error"))
        
        with pytest.raises(GraphEngineOperationError):
            await engine.get_entity("test-id")
        
        assert engine._error_count == 1
    
    @pytest.mark.asyncio
    async def test_unexpected_error(self, neo4j_engine):
        """Test handling of unexpected errors."""
        engine, mock_driver = neo4j_engine
        
        # Mock unexpected error
        engine._execute_query = AsyncMock(side_effect=Exception("Unexpected error"))
        
        with pytest.raises(GraphEngineOperationError):
            await engine.find_entities()
        
        assert engine._error_count == 1


class TestHealthAndStats:
    """Test health checks and statistics."""
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, neo4j_engine):
        """Test successful health check."""
        engine, mock_driver = neo4j_engine
        
        # Mock successful health check queries
        mock_driver.verify_connectivity = AsyncMock()
        engine._execute_query = AsyncMock(side_effect=[
            [{"entity_count": 100, "relationship_count": 200, "entity_type_count": 5, "relationship_type_count": 3}],
            [{"test": 1}]
        ])
        
        result = await engine.health_check()
        
        assert result["status"] == "healthy"
        assert "response_time" in result
        assert "graph_stats" in result
        assert result["graph_stats"]["entity_count"] == 100
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, neo4j_engine):
        """Test health check when service is unhealthy."""
        engine, mock_driver = neo4j_engine
        
        # Mock connection failure
        mock_driver.verify_connectivity = AsyncMock(side_effect=Exception("Connection failed"))
        
        result = await engine.health_check()
        
        assert result["status"] == "unhealthy"
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_get_stats(self, neo4j_engine):
        """Test statistics retrieval."""
        engine, mock_driver = neo4j_engine
        
        # Set some test stats
        engine._total_queries = 100
        engine._avg_query_time = 0.05
        engine._error_count = 2
        
        # Mock stats query
        engine._execute_query = AsyncMock(return_value=[{
            "total_entities": 1000,
            "total_relationships": 2000,
            "unique_entity_types": 10,
            "unique_relationship_types": 5,
            "avg_entity_confidence": 0.85,
            "avg_relationship_confidence": 0.90
        }])
        
        result = await engine.get_stats()
        
        assert "graph_statistics" in result
        assert "performance" in result
        assert result["performance"]["total_queries"] == 100
        assert result["performance"]["error_count"] == 2


class TestPropertySerialization:
    """Test property serialization and deserialization."""
    
    @pytest.mark.asyncio
    async def test_complex_property_serialization(self, neo4j_engine):
        """Test serialization of complex properties."""
        engine, mock_driver = neo4j_engine
        
        # Entity with complex properties
        entity = Entity(
            id="test-complex",
            name="Complex Entity",
            entity_type="test",
            properties={
                "simple_string": "value",
                "number": 42,
                "nested_dict": {"key": "value", "number": 123},
                "list_prop": ["item1", "item2", "item3"],
                "mixed_list": [1, "string", {"nested": "object"}]
            }
        )
        
        # Mock successful creation
        engine._execute_query = AsyncMock(return_value=[{"id": entity.id}])
        
        result = await engine.create_entity(entity)
        
        assert result == entity.id
        
        # Verify that _execute_query was called with serialized properties
        call_args = engine._execute_query.call_args
        properties = call_args[1]["properties"]
        
        # Complex properties should be JSON strings
        assert isinstance(properties["nested_dict"], str)
        assert json.loads(properties["nested_dict"]) == {"key": "value", "number": 123}
        assert isinstance(properties["list_prop"], str)
        assert json.loads(properties["list_prop"]) == ["item1", "item2", "item3"]
    
    @pytest.mark.asyncio
    async def test_property_deserialization(self, neo4j_engine):
        """Test deserialization of complex properties."""
        engine, mock_driver = neo4j_engine
        
        # Mock entity with serialized properties
        mock_result = [{
            "e": {
                "id": "test-complex",
                "name": "Complex Entity",
                "entity_type": "test",
                "confidence": 0.9,
                "created_at": datetime.now().isoformat(),
                "simple_string": "value",
                "number": 42,
                "nested_dict": json.dumps({"key": "value", "number": 123}),
                "list_prop": json.dumps(["item1", "item2", "item3"])
            }
        }]
        engine._execute_query = AsyncMock(return_value=mock_result)
        
        result = await engine.get_entity("test-complex")
        
        assert result is not None
        assert result.properties["simple_string"] == "value"
        assert result.properties["number"] == 42
        assert result.properties["nested_dict"] == {"key": "value", "number": 123}
        assert result.properties["list_prop"] == ["item1", "item2", "item3"]


class TestPerformanceTracking:
    """Test performance tracking and metrics."""
    
    @pytest.mark.asyncio
    async def test_query_performance_tracking(self, neo4j_engine):
        """Test that query performance is tracked correctly."""
        engine, mock_driver = neo4j_engine
        
        # Mock successful queries with different timing
        engine._execute_query = AsyncMock(return_value=[{"id": "test"}])
        
        # Execute multiple operations
        entity = Entity(id="test", name="test", entity_type="test", properties={})
        
        await engine.create_entity(entity)
        await engine.get_entity("test")
        await engine.find_entities()
        
        # Verify performance tracking
        assert engine._total_queries >= 3
        assert engine._avg_query_time > 0
    
    @pytest.mark.asyncio
    async def test_error_rate_tracking(self, neo4j_engine):
        """Test that error rates are tracked correctly."""
        engine, mock_driver = neo4j_engine
        
        # Mock some successful and some failed operations
        engine._execute_query = AsyncMock(side_effect=[
            [{"id": "success1"}],  # Success
            Exception("Error"),    # Failure
            [{"id": "success2"}],  # Success
            Exception("Error"),    # Failure
        ])
        
        entity = Entity(id="test", name="test", entity_type="test", properties={})
        
        # Execute operations with mixed success/failure
        try:
            await engine.create_entity(entity)  # Success
        except:
            pass
        
        try:
            await engine.get_entity("test")  # Failure
        except:
            pass
        
        try:
            await engine.create_entity(entity)  # Success
        except:
            pass
        
        try:
            await engine.get_entity("test")  # Failure
        except:
            pass
        
        # Verify error tracking
        assert engine._error_count == 2
        assert engine._total_queries == 4


@pytest.mark.asyncio
async def test_close_cleanup(neo4j_engine):
    """Test proper cleanup on close."""
    engine, mock_driver = neo4j_engine
    
    await engine.close()
    
    mock_driver.close.assert_called_once()