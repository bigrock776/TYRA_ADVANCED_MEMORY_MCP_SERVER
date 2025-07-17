#!/bin/bash
# =============================================================================
# Tyra MCP Memory Server - Backup and Restore Script
# =============================================================================
# Comprehensive backup solution for all system components

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/tyra/backups}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="tyra-backup-$TIMESTAMP"

# Database configuration
DATABASE_URL="${DATABASE_URL:-postgresql://tyra:tyra123@localhost:5432/tyra_memory}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
NEO4J_URL="${NEO4J_URL:-bolt://localhost:7687}"

# Service configuration
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
USE_DOCKER="${USE_DOCKER:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# =============================================================================
# Utility Functions
# =============================================================================
create_backup_directory() {
    local backup_path="$BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$backup_path"
    echo "$backup_path"
}

cleanup_old_backups() {
    info "Cleaning up backups older than $RETENTION_DAYS days..."

    if [ -d "$BACKUP_DIR" ]; then
        find "$BACKUP_DIR" -name "tyra-backup-*" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
        success "Old backups cleaned up"
    fi
}

get_db_connection_params() {
    # Extract connection parameters from DATABASE_URL
    # Format: postgresql://user:password@host:port/database

    PGUSER=$(echo "$DATABASE_URL" | sed -n 's|postgresql://\([^:]*\):.*|\1|p')
    PGPASSWORD=$(echo "$DATABASE_URL" | sed -n 's|postgresql://[^:]*:\([^@]*\)@.*|\1|p')
    PGHOST=$(echo "$DATABASE_URL" | sed -n 's|postgresql://[^@]*@\([^:]*\):.*|\1|p')
    PGPORT=$(echo "$DATABASE_URL" | sed -n 's|postgresql://[^@]*@[^:]*:\([0-9]*\)/.*|\1|p')
    PGDATABASE=$(echo "$DATABASE_URL" | sed -n 's|postgresql://[^/]*/\(.*\)|\1|p')

    export PGUSER PGPASSWORD PGHOST PGPORT PGDATABASE
}

get_redis_connection_params() {
    # Extract Redis connection parameters
    REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's|redis://\([^:]*\).*|\1|p')
    REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's|redis://[^:]*:\([0-9]*\).*|\1|p')
    REDIS_PORT=${REDIS_PORT:-6379}
    REDIS_DB=$(echo "$REDIS_URL" | sed -n 's|redis://[^/]*/\([0-9]*\).*|\1|p')
    REDIS_DB=${REDIS_DB:-0}
}

get_neo4j_connection_params() {
    # Extract Neo4j connection parameters
    NEO4J_HOST=$(echo "$NEO4J_URL" | sed -n 's|bolt://\([^:]*\).*|\1|p')
    NEO4J_PORT=$(echo "$NEO4J_URL" | sed -n 's|bolt://[^:]*:\([0-9]*\).*|\1|p')
    NEO4J_PORT=${NEO4J_PORT:-7687}
}

# =============================================================================
# Backup Functions
# =============================================================================
backup_postgresql() {
    local backup_path="$1"

    info "Backing up PostgreSQL database..."

    get_db_connection_params

    local pg_backup_file="$backup_path/postgresql_dump.sql"
    local pg_globals_file="$backup_path/postgresql_globals.sql"

    if [ "$USE_DOCKER" = "true" ]; then
        # Use docker exec to run pg_dump
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_dump -U "$PGUSER" "$PGDATABASE" > "$pg_backup_file"
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_dumpall -U "$PGUSER" --globals-only > "$pg_globals_file"
    else
        # Use local pg_dump
        pg_dump "$DATABASE_URL" > "$pg_backup_file"
        pg_dumpall --globals-only --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" > "$pg_globals_file"
    fi

    # Compress the backup
    gzip "$pg_backup_file"
    gzip "$pg_globals_file"

    success "PostgreSQL backup completed"
}

backup_redis() {
    local backup_path="$1"

    info "Backing up Redis database..."

    get_redis_connection_params

    local redis_backup_file="$backup_path/redis_dump.rdb"

    if [ "$USE_DOCKER" = "true" ]; then
        # Copy RDB file from container
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli --rdb - > "$redis_backup_file"
    else
        # Use local redis-cli
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --rdb "$redis_backup_file"
    fi

    # Also save Redis configuration
    local redis_config_file="$backup_path/redis_config.txt"
    if [ "$USE_DOCKER" = "true" ]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli CONFIG GET "*" > "$redis_config_file"
    else
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG GET "*" > "$redis_config_file"
    fi

    success "Redis backup completed"
}

backup_neo4j() {
    local backup_path="$1"

    info "Backing up Neo4j database..."

    get_neo4j_connection_params

    local neo4j_backup_file="$backup_path/neo4j_export.cypher"

    if [ "$USE_DOCKER" = "true" ]; then
        # Export all data from Neo4j
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T neo4j cypher-shell -u neo4j -p neo4j123 "MATCH (n) RETURN n LIMIT 0; CALL apoc.export.cypher.all('export.cypher', {});" > "$neo4j_backup_file"
    else
        # Use local cypher-shell if available
        if command -v cypher-shell &> /dev/null; then
            echo "CALL apoc.export.cypher.all('export.cypher', {});" | cypher-shell -u neo4j -p neo4j123 --address "bolt://$NEO4J_HOST:$NEO4J_PORT" > "$neo4j_backup_file"
        else
            warning "cypher-shell not available, skipping Neo4j backup"
            return
        fi
    fi

    # Compress the export
    gzip "$neo4j_backup_file"

    success "Neo4j backup completed"
}

backup_application_data() {
    local backup_path="$1"

    info "Backing up application data..."

    # Backup configuration files
    local config_backup_dir="$backup_path/config"
    mkdir -p "$config_backup_dir"

    if [ -d "$PROJECT_ROOT/config" ]; then
        cp -r "$PROJECT_ROOT/config"/* "$config_backup_dir/"
    fi

    # Backup environment file (excluding secrets)
    if [ -f "$PROJECT_ROOT/.env" ]; then
        grep -v -E "(PASSWORD|SECRET|KEY)" "$PROJECT_ROOT/.env" > "$backup_path/env_template" || true
    fi

    # Backup logs (last 7 days)
    local logs_backup_dir="$backup_path/logs"
    mkdir -p "$logs_backup_dir"

    if [ -d "$PROJECT_ROOT/logs" ]; then
        find "$PROJECT_ROOT/logs" -name "*.log" -mtime -7 -exec cp {} "$logs_backup_dir/" \; 2>/dev/null || true
    fi

    # Backup Docker volumes if using Docker
    if [ "$USE_DOCKER" = "true" ]; then
        local volumes_backup_dir="$backup_path/volumes"
        mkdir -p "$volumes_backup_dir"

        # Export volume data
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T memory-server tar czf - /app/data 2>/dev/null | cat > "$volumes_backup_dir/memory_data.tar.gz" || true
        docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T memory-server tar czf - /app/cache 2>/dev/null | cat > "$volumes_backup_dir/memory_cache.tar.gz" || true
    fi

    success "Application data backup completed"
}

create_backup_manifest() {
    local backup_path="$1"

    local manifest_file="$backup_path/backup_manifest.json"

    cat > "$manifest_file" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "timestamp": "$TIMESTAMP",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "1.0.0",
    "components": {
        "postgresql": $([ -f "$backup_path/postgresql_dump.sql.gz" ] && echo "true" || echo "false"),
        "redis": $([ -f "$backup_path/redis_dump.rdb" ] && echo "true" || echo "false"),
        "neo4j": $([ -f "$backup_path/neo4j_export.cypher.gz" ] && echo "true" || echo "false"),
        "application_data": $([ -d "$backup_path/config" ] && echo "true" || echo "false")
    },
    "backup_size_bytes": $(du -sb "$backup_path" | cut -f1),
    "files": [
EOF

    # List all files in backup
    find "$backup_path" -type f -printf '        "%P",\n' | sed '$ s/,$//' >> "$manifest_file"

    cat >> "$manifest_file" << EOF
    ]
}
EOF

    success "Backup manifest created"
}

# =============================================================================
# Restore Functions
# =============================================================================
restore_postgresql() {
    local backup_path="$1"

    info "Restoring PostgreSQL database..."

    local pg_backup_file="$backup_path/postgresql_dump.sql.gz"
    local pg_globals_file="$backup_path/postgresql_globals.sql.gz"

    if [ ! -f "$pg_backup_file" ]; then
        error "PostgreSQL backup file not found: $pg_backup_file"
        return 1
    fi

    get_db_connection_params

    # Restore globals first
    if [ -f "$pg_globals_file" ]; then
        info "Restoring PostgreSQL globals..."
        if [ "$USE_DOCKER" = "true" ]; then
            zcat "$pg_globals_file" | docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U "$PGUSER" -d postgres
        else
            zcat "$pg_globals_file" | psql --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" -d postgres
        fi
    fi

    # Restore database
    info "Restoring PostgreSQL database..."
    if [ "$USE_DOCKER" = "true" ]; then
        zcat "$pg_backup_file" | docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres psql -U "$PGUSER" "$PGDATABASE"
    else
        zcat "$pg_backup_file" | psql "$DATABASE_URL"
    fi

    success "PostgreSQL restore completed"
}

restore_redis() {
    local backup_path="$1"

    info "Restoring Redis database..."

    local redis_backup_file="$backup_path/redis_dump.rdb"

    if [ ! -f "$redis_backup_file" ]; then
        error "Redis backup file not found: $redis_backup_file"
        return 1
    fi

    if [ "$USE_DOCKER" = "true" ]; then
        # Stop Redis, copy RDB file, start Redis
        docker-compose -f "$DOCKER_COMPOSE_FILE" stop redis
        docker cp "$redis_backup_file" "$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps -q redis)":/data/dump.rdb
        docker-compose -f "$DOCKER_COMPOSE_FILE" start redis
    else
        warning "Manual Redis restore required - copy $redis_backup_file to Redis data directory"
    fi

    success "Redis restore completed"
}

restore_neo4j() {
    local backup_path="$1"

    info "Restoring Neo4j database..."

    local neo4j_backup_file="$backup_path/neo4j_export.cypher.gz"

    if [ ! -f "$neo4j_backup_file" ]; then
        error "Neo4j backup file not found: $neo4j_backup_file"
        return 1
    fi

    get_neo4j_connection_params

    # Clear existing data first
    if [ "$USE_DOCKER" = "true" ]; then
        echo "MATCH (n) DETACH DELETE n;" | docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T neo4j cypher-shell -u neo4j -p neo4j123

        # Restore data
        zcat "$neo4j_backup_file" | docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T neo4j cypher-shell -u neo4j -p neo4j123
    else
        if command -v cypher-shell &> /dev/null; then
            echo "MATCH (n) DETACH DELETE n;" | cypher-shell -u neo4j -p neo4j123 --address "bolt://$NEO4J_HOST:$NEO4J_PORT"
            zcat "$neo4j_backup_file" | cypher-shell -u neo4j -p neo4j123 --address "bolt://$NEO4J_HOST:$NEO4J_PORT"
        else
            warning "cypher-shell not available, skipping Neo4j restore"
        fi
    fi

    success "Neo4j restore completed"
}

restore_application_data() {
    local backup_path="$1"

    info "Restoring application data..."

    # Restore configuration files
    if [ -d "$backup_path/config" ]; then
        cp -r "$backup_path/config"/* "$PROJECT_ROOT/config/"
        success "Configuration files restored"
    fi

    # Restore Docker volumes if using Docker
    if [ "$USE_DOCKER" = "true" ] && [ -d "$backup_path/volumes" ]; then
        if [ -f "$backup_path/volumes/memory_data.tar.gz" ]; then
            docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T memory-server tar xzf - -C / < "$backup_path/volumes/memory_data.tar.gz"
        fi

        if [ -f "$backup_path/volumes/memory_cache.tar.gz" ]; then
            docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T memory-server tar xzf - -C / < "$backup_path/volumes/memory_cache.tar.gz"
        fi

        success "Volume data restored"
    fi
}

# =============================================================================
# Main Functions
# =============================================================================
create_backup() {
    echo "============================================================================="
    info "Creating Tyra MCP Memory Server Backup"
    echo "============================================================================="

    local backup_path=$(create_backup_directory)
    info "Backup location: $backup_path"

    # Create individual backups
    backup_postgresql "$backup_path"
    backup_redis "$backup_path"
    backup_neo4j "$backup_path"
    backup_application_data "$backup_path"

    # Create manifest
    create_backup_manifest "$backup_path"

    # Create compressed archive
    local archive_name="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    tar -czf "$archive_name" -C "$BACKUP_DIR" "$BACKUP_NAME"

    # Remove uncompressed backup directory
    rm -rf "$backup_path"

    # Cleanup old backups
    cleanup_old_backups

    success "Backup completed successfully"
    info "Backup archive: $archive_name"
    info "Backup size: $(du -h "$archive_name" | cut -f1)"
}

restore_backup() {
    local backup_archive="$1"

    echo "============================================================================="
    info "Restoring Tyra MCP Memory Server Backup"
    echo "============================================================================="

    if [ ! -f "$backup_archive" ]; then
        error "Backup archive not found: $backup_archive"
        exit 1
    fi

    # Extract backup
    local temp_dir="/tmp/tyra-restore-$$"
    mkdir -p "$temp_dir"

    info "Extracting backup archive..."
    tar -xzf "$backup_archive" -C "$temp_dir"

    # Find backup directory
    local backup_path=$(find "$temp_dir" -name "tyra-backup-*" -type d | head -1)

    if [ -z "$backup_path" ]; then
        error "Invalid backup archive structure"
        rm -rf "$temp_dir"
        exit 1
    fi

    # Verify backup manifest
    if [ -f "$backup_path/backup_manifest.json" ]; then
        info "Backup manifest found:"
        cat "$backup_path/backup_manifest.json" | jq '.' 2>/dev/null || cat "$backup_path/backup_manifest.json"
    fi

    # Stop services before restore
    if [ "$USE_DOCKER" = "true" ]; then
        info "Stopping services..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
    fi

    # Restore components
    restore_postgresql "$backup_path"
    restore_redis "$backup_path"
    restore_neo4j "$backup_path"
    restore_application_data "$backup_path"

    # Start services after restore
    if [ "$USE_DOCKER" = "true" ]; then
        info "Starting services..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    fi

    # Cleanup
    rm -rf "$temp_dir"

    success "Restore completed successfully"
}

list_backups() {
    echo "============================================================================="
    info "Available Backups"
    echo "============================================================================="

    if [ ! -d "$BACKUP_DIR" ]; then
        info "No backup directory found: $BACKUP_DIR"
        return
    fi

    local backups=$(find "$BACKUP_DIR" -name "tyra-backup-*.tar.gz" -type f | sort -r)

    if [ -z "$backups" ]; then
        info "No backups found"
        return
    fi

    printf "%-30s %-15s %-20s\n" "Backup Name" "Size" "Created"
    printf "%-30s %-15s %-20s\n" "----------" "----" "-------"

    for backup in $backups; do
        local name=$(basename "$backup" .tar.gz)
        local size=$(du -h "$backup" | cut -f1)
        local date=$(date -r "$backup" '+%Y-%m-%d %H:%M:%S')
        printf "%-30s %-15s %-20s\n" "$name" "$size" "$date"
    done
}

show_help() {
    cat << EOF
Tyra MCP Memory Server - Backup and Restore Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    backup              Create a new backup (default)
    restore ARCHIVE     Restore from backup archive
    list                List available backups
    help                Show this help message

Options:
    --backup-dir DIR    Set backup directory [default: /opt/tyra/backups]
    --retention-days N  Set backup retention in days [default: 7]
    --no-docker         Don't use Docker commands

Environment Variables:
    DATABASE_URL        PostgreSQL connection string
    REDIS_URL          Redis connection string
    NEO4J_URL          Neo4j connection string
    BACKUP_DIR         Backup directory path
    BACKUP_RETENTION_DAYS  Backup retention period

Examples:
    $0                                    # Create backup
    $0 backup --retention-days 14        # Create backup with 14-day retention
    $0 restore /opt/tyra/backups/tyra-backup-20240115-120000.tar.gz
    $0 list                              # List all backups

EOF
}

# =============================================================================
# Command Line Interface
# =============================================================================
COMMAND="${1:-backup}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        backup)
            COMMAND="backup"
            shift
            ;;
        restore)
            COMMAND="restore"
            RESTORE_ARCHIVE="$2"
            shift 2
            ;;
        list)
            COMMAND="list"
            shift
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --retention-days)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --no-docker)
            USE_DOCKER="false"
            shift
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            if [ "$COMMAND" = "backup" ]; then
                shift
            else
                error "Unknown argument: $1"
                show_help
                exit 1
            fi
            ;;
    esac
done

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Execute command
case $COMMAND in
    backup)
        create_backup
        ;;
    restore)
        if [ -z "${RESTORE_ARCHIVE:-}" ]; then
            error "Backup archive not specified"
            show_help
            exit 1
        fi
        restore_backup "$RESTORE_ARCHIVE"
        ;;
    list)
        list_backups
        ;;
    *)
        error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
