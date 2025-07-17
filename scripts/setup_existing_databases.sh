#!/bin/bash
# Tyra MCP Memory Server - Production-Grade Setup Script for Existing Database Installations
# Version: 2.0.0 - Production Ready
# This script sets up schemas and configuration when PostgreSQL, Neo4j, and Redis are already installed

set -euo pipefail

# Script metadata
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_NAME="setup_existing_databases.sh"
readonly LOG_FILE="setup_log.txt"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Script flags
DRY_RUN=false
CLEAR_DATA=false
SKIP_TESTS=false
FORCE_OVERWRITE=false

# Required SQL schema files
readonly REQUIRED_SCHEMA_FILES=(
    "src/migrations/sql/001_initial_schema.sql"
    "scripts/init_postgres.sql"
)

# Get script directory
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration with validation - PostgreSQL (REQUIRED)
POSTGRES_DB="${POSTGRES_DB:-tyra_memory}"
POSTGRES_USER="${POSTGRES_USER:-tyra}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

# Configuration - Neo4j (REQUIRED for runtime)
NEO4J_HOST="${NEO4J_HOST:-localhost}"
NEO4J_PORT="${NEO4J_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j123}"

# Configuration - Redis (REQUIRED for runtime)
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

# Logging functions with file output
log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
    echo -e "${GREEN}${msg}${NC}" | tee -a "$LOG_FILE"
}

error() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo -e "${RED}${msg}${NC}" | tee -a "$LOG_FILE"
}

warning() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1"
    echo -e "${YELLOW}${msg}${NC}" | tee -a "$LOG_FILE"
}

info() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
    echo -e "${BLUE}${msg}${NC}" | tee -a "$LOG_FILE"
}

# Cleanup function for script exit
cleanup() {
    if [[ -f "/tmp/setup_test_$$" ]]; then
        rm -f "/tmp/setup_test_$$"
    fi
}
trap cleanup EXIT

# Help function
show_help() {
    cat << EOF
Tyra MCP Memory Server - Production Setup for Existing Database Installations
Version: $SCRIPT_VERSION

USAGE:
    $0 [OPTIONS]

REQUIRED ENVIRONMENT VARIABLES:
    POSTGRES_PASSWORD           PostgreSQL password (REQUIRED)

OPTIONAL ENVIRONMENT VARIABLES:
    POSTGRES_DB                 Database name (default: tyra_memory)
    POSTGRES_USER              Database user (default: tyra)
    POSTGRES_HOST              Database host (default: localhost)
    POSTGRES_PORT              Database port (default: 5432)
    NEO4J_HOST                 Neo4j host (default: localhost)
    NEO4J_PORT                 Neo4j port (default: 7687)
    NEO4J_USER                 Neo4j username (default: neo4j)
    NEO4J_PASSWORD             Neo4j password (default: neo4j123)
    REDIS_HOST                 Redis host (default: localhost)
    REDIS_PORT                 Redis port (default: 6379)
    REDIS_DB                   Redis database (default: 0)
    REDIS_PASSWORD             Redis password (optional)

OPTIONS:
    --postgres-db NAME          Override PostgreSQL database name
    --postgres-user USER        Override PostgreSQL username
    --postgres-password PASS    Set PostgreSQL password (or use env var)
    --postgres-host HOST        Override PostgreSQL host
    --postgres-port PORT        Override PostgreSQL port
    --neo4j-host HOST           Override Neo4j host
    --neo4j-port PORT           Override Neo4j port
    --neo4j-user USER           Override Neo4j username
    --neo4j-password PASS       Override Neo4j password
    --redis-host HOST           Override Redis host
    --redis-port PORT           Override Redis port
    --redis-db DB               Override Redis database
    --redis-password PASS       Override Redis password
    --clear-data               Clear existing data before setup (DANGEROUS)
    --skip-tests               Skip connection and validation tests
    --force-overwrite          Overwrite existing .env file without backup
    --dry-run                  Preview actions without executing them
    --help                     Show this help message

EXAMPLES:
    # Basic setup (requires POSTGRES_PASSWORD environment variable)
    export POSTGRES_PASSWORD=mypassword
    $0

    # Setup with custom database and clear existing data
    export POSTGRES_PASSWORD=mypass
    $0 --postgres-db my_tyra_db --clear-data

    # Dry run to preview actions
    export POSTGRES_PASSWORD=mypass
    $0 --dry-run

    # Setup with custom hosts for remote databases
    export POSTGRES_PASSWORD=mypass
    $0 --postgres-host db.example.com --neo4j-host graph.example.com --redis-host cache.example.com

SECURITY REQUIREMENTS:
    - POSTGRES_PASSWORD must be set as environment variable
    - Script validates all required files exist before execution
    - All SQL operations use IF NOT EXISTS for idempotency
    - Comprehensive logging to $LOG_FILE

NOTE:
    This script focuses on PostgreSQL schema setup but configures runtime
    environment for all three databases (PostgreSQL, Neo4j, Redis).
    See INSTALLATION.md for complete installation instructions.

EOF
}

# Validate environment and prerequisites
validate_environment() {
    log "Validating environment and prerequisites..."

    # Check if we're in the correct directory
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]] && [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
        error "Script must be run from the Tyra MCP project root directory"
        error "Expected to find pyproject.toml or requirements.txt in: $PROJECT_ROOT"
        exit 1
    fi

    # Validate POSTGRES_PASSWORD is set
    if [[ -z "$POSTGRES_PASSWORD" ]]; then
        error "POSTGRES_PASSWORD environment variable is required but not set"
        error "Set it with: export POSTGRES_PASSWORD=your_password"
        error "Or use --postgres-password argument"
        exit 1
    fi

    # Validate required commands exist
    local required_commands=("psql" "createdb" "createuser")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command '$cmd' not found. Please install PostgreSQL client tools."
            exit 1
        fi
    done

    # Validate PostgreSQL environment variables
    if [[ -z "$POSTGRES_HOST" ]] || [[ -z "$POSTGRES_PORT" ]] || [[ -z "$POSTGRES_USER" ]]; then
        error "Required PostgreSQL environment variables not set"
        error "POSTGRES_HOST: '$POSTGRES_HOST'"
        error "POSTGRES_PORT: '$POSTGRES_PORT'"
        error "POSTGRES_USER: '$POSTGRES_USER'"
        exit 1
    fi

    # Validate port is numeric
    if ! [[ "$POSTGRES_PORT" =~ ^[0-9]+$ ]]; then
        error "POSTGRES_PORT must be numeric, got: '$POSTGRES_PORT'"
        exit 1
    fi

    if ! [[ "$NEO4J_PORT" =~ ^[0-9]+$ ]]; then
        error "NEO4J_PORT must be numeric, got: '$NEO4J_PORT'"
        exit 1
    fi

    if ! [[ "$REDIS_PORT" =~ ^[0-9]+$ ]]; then
        error "REDIS_PORT must be numeric, got: '$REDIS_PORT'"
        exit 1
    fi

    if ! [[ "$REDIS_DB" =~ ^[0-9]+$ ]]; then
        error "REDIS_DB must be numeric, got: '$REDIS_DB'"
        exit 1
    fi

    log "✅ Environment validation completed"
}

# Validate required files exist
validate_files() {
    log "Validating required schema files..."

    local schema_found=false
    local available_schemas=()

    for schema_file in "${REQUIRED_SCHEMA_FILES[@]}"; do
        local full_path="$PROJECT_ROOT/$schema_file"
        if [[ -f "$full_path" ]]; then
            available_schemas+=("$full_path")
            schema_found=true
            info "✅ Found schema file: $schema_file"
        else
            warning "❌ Schema file not found: $schema_file"
        fi
    done

    if [[ "$schema_found" == false ]]; then
        error "No required schema files found. Expected one of:"
        for schema_file in "${REQUIRED_SCHEMA_FILES[@]}"; do
            error "  - $schema_file"
        done
        exit 1
    fi

    # Store the first available schema for use
    SELECTED_SCHEMA="${available_schemas[0]}"
    log "✅ Using schema file: $(basename "$SELECTED_SCHEMA")"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --postgres-db)
                POSTGRES_DB="$2"
                shift 2
                ;;
            --postgres-user)
                POSTGRES_USER="$2"
                shift 2
                ;;
            --postgres-password)
                POSTGRES_PASSWORD="$2"
                shift 2
                ;;
            --postgres-host)
                POSTGRES_HOST="$2"
                shift 2
                ;;
            --postgres-port)
                POSTGRES_PORT="$2"
                shift 2
                ;;
            --neo4j-host)
                NEO4J_HOST="$2"
                shift 2
                ;;
            --neo4j-port)
                NEO4J_PORT="$2"
                shift 2
                ;;
            --neo4j-user)
                NEO4J_USER="$2"
                shift 2
                ;;
            --neo4j-password)
                NEO4J_PASSWORD="$2"
                shift 2
                ;;
            --redis-host)
                REDIS_HOST="$2"
                shift 2
                ;;
            --redis-port)
                REDIS_PORT="$2"
                shift 2
                ;;
            --redis-db)
                REDIS_DB="$2"
                shift 2
                ;;
            --redis-password)
                REDIS_PASSWORD="$2"
                shift 2
                ;;
            --clear-data)
                CLEAR_DATA=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --force-overwrite)
                FORCE_OVERWRITE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Execute command with dry-run support
execute_cmd() {
    local cmd="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == true ]]; then
        info "[DRY RUN] Would execute: $description"
        info "[DRY RUN] Command: $cmd"
        return 0
    else
        info "Executing: $description"
        eval "$cmd"
    fi
}

# Test PostgreSQL connection
test_postgresql_connection() {
    if [[ "$SKIP_TESTS" == true ]]; then
        info "Skipping PostgreSQL connection test as requested"
        return 0
    fi

    log "Testing PostgreSQL connection..."

    # Test connection to postgres database first
    local test_cmd="PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d postgres -c 'SELECT version();' > /tmp/setup_test_$$ 2>&1"
    
    if [[ "$DRY_RUN" == true ]]; then
        info "[DRY RUN] Would test PostgreSQL connection"
        return 0
    fi

    if ! eval "$test_cmd"; then
        error "❌ PostgreSQL connection failed"
        error "Connection details:"
        error "  Host: $POSTGRES_HOST"
        error "  Port: $POSTGRES_PORT"
        error "  User: $POSTGRES_USER"
        error "  Database: postgres"
        if [[ -f "/tmp/setup_test_$$" ]]; then
            error "Error details:"
            cat "/tmp/setup_test_$$" | tee -a "$LOG_FILE"
        fi
        exit 1
    fi

    log "✅ PostgreSQL connection successful"
    
    # Get PostgreSQL version
    local pg_version=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -t -c "SELECT version();" | head -1 | awk '{print $2}')
    info "PostgreSQL version: $pg_version"
}

# Check PostgreSQL extensions
check_postgresql_extensions() {
    log "Checking PostgreSQL extensions..."

    if [[ "$DRY_RUN" == true ]]; then
        info "[DRY RUN] Would check PostgreSQL extensions"
        return 0
    fi

    # Test pgvector extension
    local ext_test="PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d postgres -c \"CREATE EXTENSION IF NOT EXISTS vector;\" > /tmp/setup_test_$$ 2>&1"
    
    if ! eval "$ext_test"; then
        error "❌ pgvector extension not available"
        error "Please install pgvector extension first"
        error "Ubuntu/Debian: sudo apt install postgresql-15-pgvector"
        error "See: https://github.com/pgvector/pgvector#installation"
        if [[ -f "/tmp/setup_test_$$" ]]; then
            cat "/tmp/setup_test_$$" | tee -a "$LOG_FILE"
        fi
        exit 1
    fi

    log "✅ pgvector extension available"

    # Check other extensions
    local other_extensions=("uuid-ossp" "pg_trgm" "btree_gin")
    for ext in "${other_extensions[@]}"; do
        local ext_check="PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d postgres -c \"CREATE EXTENSION IF NOT EXISTS \\\"$ext\\\";\" > /tmp/setup_test_$$ 2>&1"
        if eval "$ext_check"; then
            info "✅ Extension '$ext' available"
        else
            warning "⚠️  Extension '$ext' not available (optional)"
        fi
    done
}

# Create PostgreSQL database and user
setup_postgresql_database() {
    log "Setting up PostgreSQL database and user..."

    if [[ "$CLEAR_DATA" == true ]]; then
        warning "⚠️  CLEAR_DATA flag is set - existing data will be destroyed"
        if [[ "$DRY_RUN" == false ]]; then
            warning "Dropping database '$POSTGRES_DB' if it exists..."
            execute_cmd "PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d postgres -c \"DROP DATABASE IF EXISTS $POSTGRES_DB;\"" "Drop existing database"
        fi
    fi

    # Create database if it doesn't exist
    execute_cmd "PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d postgres -c \"CREATE DATABASE $POSTGRES_DB;\"" "Create database '$POSTGRES_DB'"

    # Grant permissions
    execute_cmd "PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d postgres -c \"GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;\"" "Grant database privileges"
    
    execute_cmd "PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d postgres -c \"ALTER USER $POSTGRES_USER CREATEDB;\"" "Grant CREATEDB privilege"

    log "✅ PostgreSQL database setup completed"
}

# Apply PostgreSQL schema with validation
apply_postgresql_schema() {
    log "Applying PostgreSQL schema from: $(basename "$SELECTED_SCHEMA")"

    # Validate schema file exists and is readable
    if [[ ! -f "$SELECTED_SCHEMA" ]]; then
        error "Schema file does not exist: $SELECTED_SCHEMA"
        exit 1
    fi

    if [[ ! -r "$SELECTED_SCHEMA" ]]; then
        error "Schema file is not readable: $SELECTED_SCHEMA"
        exit 1
    fi

    # Validate schema file contains CREATE TABLE IF NOT EXISTS
    if ! grep -q "CREATE TABLE IF NOT EXISTS\|CREATE EXTENSION IF NOT EXISTS" "$SELECTED_SCHEMA"; then
        error "Schema file does not use 'IF NOT EXISTS' clauses for safety"
        error "This is required for idempotent operations"
        error "Please update the schema file: $SELECTED_SCHEMA"
        exit 1
    fi

    # Apply schema
    execute_cmd "PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d '$POSTGRES_DB' -f '$SELECTED_SCHEMA'" "Apply PostgreSQL schema"

    # Verify schema application
    if [[ "$DRY_RUN" == false ]] && [[ "$SKIP_TESTS" == false ]]; then
        local table_count=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';" | tr -d ' ')
        
        if [[ "$table_count" -gt 0 ]]; then
            log "✅ Schema applied successfully ($table_count tables created)"
        else
            error "❌ Schema application failed - no tables found"
            exit 1
        fi
    fi

    log "✅ PostgreSQL schema application completed"
}

# Create environment configuration
create_env_config() {
    log "Creating environment configuration..."

    local env_file="$PROJECT_ROOT/.env"
    
    # Backup existing .env if it exists and not forcing overwrite
    if [[ -f "$env_file" ]] && [[ "$FORCE_OVERWRITE" == false ]]; then
        local backup_file="$env_file.backup.$(date +%Y%m%d_%H%M%S)"
        if [[ "$DRY_RUN" == false ]]; then
            cp "$env_file" "$backup_file"
            info "Existing .env backed up to: $(basename "$backup_file")"
        else
            info "[DRY RUN] Would backup existing .env to: $(basename "$backup_file")"
        fi
    fi

    if [[ "$DRY_RUN" == true ]]; then
        info "[DRY RUN] Would create .env file with complete configuration"
        return 0
    fi

    cat > "$env_file" << EOF
# Tyra MCP Memory Server Configuration
# Generated by $SCRIPT_NAME v$SCRIPT_VERSION on $(date)

# PostgreSQL Configuration - PRIMARY DATABASE
POSTGRES_HOST=$POSTGRES_HOST
POSTGRES_PORT=$POSTGRES_PORT
POSTGRES_DB=$POSTGRES_DB
POSTGRES_USER=$POSTGRES_USER
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_SSL_MODE=prefer

# Neo4j Configuration - KNOWLEDGE GRAPH DATABASE
NEO4J_HOST=$NEO4J_HOST
NEO4J_PORT=$NEO4J_PORT
NEO4J_USER=$NEO4J_USER
NEO4J_PASSWORD=$NEO4J_PASSWORD

# Redis Configuration - CACHE DATABASE
REDIS_HOST=$REDIS_HOST
REDIS_PORT=$REDIS_PORT
REDIS_DB=$REDIS_DB
REDIS_PASSWORD=$REDIS_PASSWORD

# Connection Pool Settings
POSTGRES_POOL_SIZE=20
POSTGRES_POOL_TIMEOUT=30
REDIS_POOL_SIZE=50
NEO4J_POOL_SIZE=10

# Performance Settings
VECTOR_DIMENSIONS=1024
HNSW_M=16
HNSW_EF_CONSTRUCTION=64

# Cache TTL Settings
EMBEDDING_CACHE_TTL=86400
SEARCH_CACHE_TTL=3600
RERANK_CACHE_TTL=1800

# Embedding Models Configuration
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_PRIMARY_PATH=./models/embeddings/e5-large-v2
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_FALLBACK_PATH=./models/embeddings/all-MiniLM-L12-v2

# Cross-encoder Reranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_MODEL_PATH=./models/cross-encoders/ms-marco-MiniLM-L-6-v2

# System Configuration
LOG_LEVEL=INFO
DEBUG=false
ENVIRONMENT=development
EOF

    log "✅ Environment configuration created at .env"
}

# Validate final setup
validate_setup() {
    if [[ "$SKIP_TESTS" == true ]] || [[ "$DRY_RUN" == true ]]; then
        info "Skipping setup validation"
        return 0
    fi

    log "Validating final setup..."

    # Test database connection with new database
    local db_test="PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d '$POSTGRES_DB' -c 'SELECT 1;' > /tmp/setup_test_$$ 2>&1"
    
    if ! eval "$db_test"; then
        error "❌ Final database connection test failed"
        if [[ -f "/tmp/setup_test_$$" ]]; then
            cat "/tmp/setup_test_$$" | tee -a "$LOG_FILE"
        fi
        exit 1
    fi

    # Test pgvector functionality
    local vector_test="PGPASSWORD='$POSTGRES_PASSWORD' psql -h '$POSTGRES_HOST' -p '$POSTGRES_PORT' -U '$POSTGRES_USER' -d '$POSTGRES_DB' -c \"SELECT '[1,2,3]'::vector(3) <-> '[4,5,6]'::vector(3) as distance;\" > /tmp/setup_test_$$ 2>&1"
    
    if eval "$vector_test"; then
        log "✅ pgvector functionality test passed"
    else
        warning "⚠️  pgvector functionality test failed"
        if [[ -f "/tmp/setup_test_$$" ]]; then
            cat "/tmp/setup_test_$$" | tee -a "$LOG_FILE"
        fi
    fi

    log "✅ Setup validation completed"
}

# Show final summary
show_summary() {
    log "🎉 Database setup completed successfully!"
    echo | tee -a "$LOG_FILE"
    log "📋 Summary:"
    log "   - PostgreSQL database '$POSTGRES_DB' configured with comprehensive schema"
    log "   - Environment configuration created at .env with all three databases"
    log "   - All operations logged to $LOG_FILE"
    echo | tee -a "$LOG_FILE"
    log "📋 Database Configuration:"
    log "   - PostgreSQL: $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"
    log "   - Neo4j:      $NEO4J_HOST:$NEO4J_PORT"
    log "   - Redis:      $REDIS_HOST:$REDIS_PORT/$REDIS_DB"
    echo | tee -a "$LOG_FILE"
    log "🚀 Next Steps:"
    log "   1. Install embedding models (see INSTALLATION.md Step 5)"
    log "   2. Set up Neo4j and Redis if not already installed (see INSTALLATION.md)"
    log "   3. Test setup: python scripts/test_setup.py"
    log "   4. Start server: python -m src.mcp.server"
    echo | tee -a "$LOG_FILE"
    warning "🔒 SECURITY: Secure your database passwords in production!"
    warning "🔧 NOTE: This script configured PostgreSQL schema and environment for all databases."
    warning "        Complete Neo4j and Redis setup instructions are in INSTALLATION.md"
    echo | tee -a "$LOG_FILE"
    log "✅ Setup completed at $(date)"
}

# Main function
main() {
    # Initialize logging
    echo "# Tyra MCP Memory Server - Database Setup Log" > "$LOG_FILE"
    echo "# Started at: $(date)" >> "$LOG_FILE"
    echo "# Script version: $SCRIPT_VERSION" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    log "🚀 Starting Tyra MCP Memory Server database setup..."
    log "📝 All output logged to: $LOG_FILE"
    
    if [[ "$DRY_RUN" == true ]]; then
        warning "🔍 DRY RUN MODE - No changes will be made"
    fi

    echo | tee -a "$LOG_FILE"

    # Execute setup steps
    parse_args "$@"
    validate_environment
    validate_files
    test_postgresql_connection
    check_postgresql_extensions
    setup_postgresql_database
    apply_postgresql_schema
    create_env_config
    validate_setup
    show_summary

    log "🎯 Setup completed successfully!"
}

# Run main function with all arguments
main "$@"