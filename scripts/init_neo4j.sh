#!/bin/bash
# Neo4j Database Initialization Script
# Initializes Neo4j with proper schema, indexes, and constraints for Tyra Memory Server

set -euo pipefail

# Configuration
NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_PORT="${NEO4J_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j}"
NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Neo4j is running
check_neo4j_connection() {
    log_info "Checking Neo4j connection..."
    
    if command -v cypher-shell >/dev/null 2>&1; then
        if cypher-shell -a "neo4j://${NEO4J_HOST}:${NEO4J_PORT}" -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" -d "${NEO4J_DATABASE}" "RETURN 1;" >/dev/null 2>&1; then
            log_success "Neo4j connection successful"
            return 0
        else
            log_error "Failed to connect to Neo4j"
            return 1
        fi
    else
        log_warning "cypher-shell not found, attempting Python connection..."
        python3 -c "
from neo4j import GraphDatabase
import sys
try:
    driver = GraphDatabase.driver('neo4j://${NEO4J_HOST}:${NEO4J_PORT}', auth=('${NEO4J_USER}', '${NEO4J_PASSWORD}'))
    driver.verify_connectivity()
    driver.close()
    print('Neo4j connection successful')
    sys.exit(0)
except Exception as e:
    print(f'Failed to connect to Neo4j: {e}')
    sys.exit(1)
"
        return $?
    fi
}

# Execute Cypher commands
execute_cypher() {
    local cypher_command="$1"
    local description="$2"
    
    log_info "Executing: $description"
    
    if command -v cypher-shell >/dev/null 2>&1; then
        if cypher-shell -a "neo4j://${NEO4J_HOST}:${NEO4J_PORT}" -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" -d "${NEO4J_DATABASE}" "$cypher_command"; then
            log_success "$description completed"
            return 0
        else
            log_warning "$description failed, continuing..."
            return 1
        fi
    else
        # Fallback to Python execution
        python3 -c "
from neo4j import GraphDatabase
import sys
try:
    driver = GraphDatabase.driver('neo4j://${NEO4J_HOST}:${NEO4J_PORT}', auth=('${NEO4J_USER}', '${NEO4J_PASSWORD}'))
    with driver.session(database='${NEO4J_DATABASE}') as session:
        session.run('$cypher_command')
    driver.close()
    print('$description completed')
    sys.exit(0)
except Exception as e:
    print(f'$description failed: {e}')
    sys.exit(1)
"
        if [ $? -eq 0 ]; then
            log_success "$description completed"
            return 0
        else
            log_warning "$description failed, continuing..."
            return 1
        fi
    fi
}

# Initialize Neo4j schema
initialize_schema() {
    log_info "Initializing Neo4j schema for Tyra Memory Server..."
    
    # Create constraints
    execute_cypher "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE" \
        "Creating entity ID constraint"
    
    execute_cypher "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE" \
        "Creating memory ID constraint"
    
    # Create indexes for performance
    execute_cypher "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)" \
        "Creating entity type index"
    
    execute_cypher "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)" \
        "Creating entity name index"
    
    execute_cypher "CREATE INDEX entity_created_index IF NOT EXISTS FOR (e:Entity) ON (e.created_at)" \
        "Creating entity created_at index"
    
    execute_cypher "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)" \
        "Creating entity confidence index"
    
    # Text indexes for search
    execute_cypher "CREATE TEXT INDEX entity_name_text IF NOT EXISTS FOR (n:Entity) ON (n.name)" \
        "Creating entity name text index"
    
    execute_cypher "CREATE TEXT INDEX memory_content_text IF NOT EXISTS FOR (m:Memory) ON (m.content)" \
        "Creating memory content text index"
    
    # Relationship indexes
    execute_cypher "CREATE INDEX relationship_type IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.relationship_type)" \
        "Creating relationship type index"
    
    execute_cypher "CREATE INDEX relationship_valid_from IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_from)" \
        "Creating relationship valid_from index"
    
    execute_cypher "CREATE INDEX relationship_valid_to IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.valid_to)" \
        "Creating relationship valid_to index"
    
    execute_cypher "CREATE INDEX relationship_confidence IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.confidence)" \
        "Creating relationship confidence index"
    
    # Memory indexes
    execute_cypher "CREATE INDEX memory_created_index IF NOT EXISTS FOR (m:Memory) ON (m.created_at)" \
        "Creating memory created_at index"
    
    execute_cypher "CREATE INDEX memory_user_index IF NOT EXISTS FOR (m:Memory) ON (m.user_id)" \
        "Creating memory user_id index"
    
    execute_cypher "CREATE INDEX memory_confidence_index IF NOT EXISTS FOR (m:Memory) ON (m.confidence)" \
        "Creating memory confidence index"
}

# Create sample data for testing
create_sample_data() {
    log_info "Creating sample data for testing..."
    
    execute_cypher "
    MERGE (e1:Entity {id: 'test-entity-1', name: 'Test Entity 1', entity_type: 'person', confidence: 0.95, created_at: datetime()})
    MERGE (e2:Entity {id: 'test-entity-2', name: 'Test Entity 2', entity_type: 'organization', confidence: 0.90, created_at: datetime()})
    MERGE (e1)-[r:RELATIONSHIP {id: 'test-rel-1', relationship_type: 'WORKS_FOR', confidence: 0.85, created_at: datetime()}]->(e2)
    " \
    "Creating sample entities and relationships"
    
    execute_cypher "
    MERGE (m:Memory {
        id: 'test-memory-1',
        content: 'This is a test memory for the Tyra system',
        user_id: 'test-user',
        confidence: 0.95,
        created_at: datetime(),
        metadata: '{\"source\": \"test\", \"type\": \"sample\"}'
    })
    " \
    "Creating sample memory"
}

# Validate schema
validate_schema() {
    log_info "Validating Neo4j schema..."
    
    # Check constraints
    execute_cypher "SHOW CONSTRAINTS" "Listing constraints"
    
    # Check indexes
    execute_cypher "SHOW INDEXES" "Listing indexes"
    
    # Test basic operations
    execute_cypher "MATCH (n) RETURN count(n) as node_count" "Counting nodes"
    
    execute_cypher "MATCH ()-[r]->() RETURN count(r) as relationship_count" "Counting relationships"
}

# Main execution
main() {
    log_info "Starting Neo4j initialization for Tyra Memory Server"
    log_info "Neo4j connection: neo4j://${NEO4J_HOST}:${NEO4J_PORT}"
    log_info "Database: ${NEO4J_DATABASE}"
    
    # Check connection
    if ! check_neo4j_connection; then
        log_error "Cannot connect to Neo4j. Please ensure Neo4j is running and accessible."
        exit 1
    fi
    
    # Initialize schema
    initialize_schema
    
    # Create sample data if requested
    if [[ "${1:-}" == "--with-samples" ]]; then
        create_sample_data
    fi
    
    # Validate setup
    validate_schema
    
    log_success "Neo4j initialization completed successfully!"
    log_info "You can now start the Tyra Memory Server"
}

# Script help
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Neo4j Initialization Script for Tyra Memory Server"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --with-samples    Create sample data for testing"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NEO4J_HOST        Neo4j host (default: localhost)"
    echo "  NEO4J_PORT        Neo4j port (default: 7687)"
    echo "  NEO4J_USER        Neo4j username (default: neo4j)"
    echo "  NEO4J_PASSWORD    Neo4j password (default: neo4j)"
    echo "  NEO4J_DATABASE    Neo4j database (default: neo4j)"
    exit 0
fi

# Run main function
main "$@"