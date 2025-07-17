// Neo4j Schema Initialization for Tyra Memory Server
// This file contains Cypher queries to set up the complete graph schema

// ============================================================================
// CONSTRAINTS - Ensure data integrity
// ============================================================================

// Entity constraints
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE;

// ============================================================================
// INDEXES - Optimize query performance
// ============================================================================

// Entity indexes
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type);
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_created_index IF NOT EXISTS FOR (e:Entity) ON (e.created_at);
CREATE INDEX entity_updated_index IF NOT EXISTS FOR (e:Entity) ON (e.updated_at);
CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence);

// Memory indexes
CREATE INDEX memory_created_index IF NOT EXISTS FOR (m:Memory) ON (m.created_at);
CREATE INDEX memory_updated_index IF NOT EXISTS FOR (m:Memory) ON (m.updated_at);
CREATE INDEX memory_user_index IF NOT EXISTS FOR (m:Memory) ON (m.user_id);
CREATE INDEX memory_confidence_index IF NOT EXISTS FOR (m:Memory) ON (m.confidence);
CREATE INDEX memory_importance_index IF NOT EXISTS FOR (m:Memory) ON (m.importance);

// Relationship indexes
CREATE INDEX relationship_type IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.relationship_type);
CREATE INDEX relationship_created IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.created_at);
CREATE INDEX relationship_valid_from IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_from);
CREATE INDEX relationship_valid_to IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_to);
CREATE INDEX relationship_confidence IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.confidence);

// ============================================================================
// TEXT INDEXES - Enable full-text search
// ============================================================================

// Entity text search
CREATE TEXT INDEX entity_name_text IF NOT EXISTS FOR (n:Entity) ON (n.name);
CREATE TEXT INDEX entity_properties_text IF NOT EXISTS FOR (n:Entity) ON (n.properties);

// Memory text search
CREATE TEXT INDEX memory_content_text IF NOT EXISTS FOR (m:Memory) ON (m.content);
CREATE TEXT INDEX memory_metadata_text IF NOT EXISTS FOR (m:Memory) ON (m.metadata);

// ============================================================================
// COMPOSITE INDEXES - Optimize complex queries
// ============================================================================

// Entity type and time-based queries
CREATE INDEX entity_type_created IF NOT EXISTS FOR (n:Entity) ON (n.entity_type, n.created_at);
CREATE INDEX entity_type_confidence IF NOT EXISTS FOR (n:Entity) ON (n.entity_type, n.confidence);

// Memory user and time-based queries
CREATE INDEX memory_user_created IF NOT EXISTS FOR (m:Memory) ON (m.user_id, m.created_at);
CREATE INDEX memory_user_confidence IF NOT EXISTS FOR (m:Memory) ON (m.user_id, m.confidence);

// Temporal relationship queries
CREATE INDEX relationship_temporal IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_from, r.valid_to);

// ============================================================================
// SAMPLE DATA SETUP (Optional - for testing)
// ============================================================================

// Create sample entity types for validation
MERGE (et1:EntityType {
    name: 'person',
    description: 'Individual human beings',
    created_at: datetime()
});

MERGE (et2:EntityType {
    name: 'organization',
    description: 'Companies, institutions, and groups',
    created_at: datetime()
});

MERGE (et3:EntityType {
    name: 'location',
    description: 'Physical or virtual places',
    created_at: datetime()
});

MERGE (et4:EntityType {
    name: 'concept',
    description: 'Ideas, topics, and abstract concepts',
    created_at: datetime()
});

// Create sample relationship types
MERGE (rt1:RelationshipType {
    name: 'WORKS_FOR',
    description: 'Employment or work relationship',
    created_at: datetime()
});

MERGE (rt2:RelationshipType {
    name: 'LOCATED_IN',
    description: 'Physical or logical location relationship',
    created_at: datetime()
});

MERGE (rt3:RelationshipType {
    name: 'RELATED_TO',
    description: 'General relationship between entities',
    created_at: datetime()
});

MERGE (rt4:RelationshipType {
    name: 'MENTIONED_IN',
    description: 'Entity mentioned in memory or document',
    created_at: datetime()
});

// ============================================================================
// VALIDATION QUERIES
// ============================================================================

// These queries can be used to validate the schema setup

// Count constraints
// SHOW CONSTRAINTS;

// Count indexes
// SHOW INDEXES;

// Verify node labels
// CALL db.labels() YIELD label RETURN label ORDER BY label;

// Verify relationship types
// CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType;

// ============================================================================
// PERFORMANCE TUNING HINTS
// ============================================================================

// For production deployment, consider these additional optimizations:

// 1. Configure Neo4j memory settings in neo4j.conf:
//    dbms.memory.heap.initial_size=1G
//    dbms.memory.heap.max_size=4G
//    dbms.memory.pagecache.size=2G

// 2. Enable query logging for performance monitoring:
//    dbms.logs.query.enabled=true
//    dbms.logs.query.threshold=1000ms

// 3. For high-volume scenarios, consider partitioning:
//    - Time-based partitioning for memories
//    - Entity type-based partitioning for large entity sets

// 4. Regular maintenance procedures:
//    - Monitor index usage with SHOW INDEXES YIELD *
//    - Optimize queries based on EXPLAIN and PROFILE results
//    - Consider periodic CALL apoc.schema.assert() for schema validation

// ============================================================================
// COMPLETION MESSAGE
// ============================================================================

// Return success message
RETURN "Neo4j schema initialization completed successfully" AS status,
       datetime() AS completed_at;