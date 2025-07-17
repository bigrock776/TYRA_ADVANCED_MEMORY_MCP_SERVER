#!/bin/bash
# Neo4j Installation Script for Tyra Memory Server
# Installs Neo4j Community Edition with APOC plugin support

set -euo pipefail

# Configuration
NEO4J_VERSION="${NEO4J_VERSION:-5.15.0}"
NEO4J_HOME="${NEO4J_HOME:-/opt/neo4j}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j}"
INSTALL_APOC="${INSTALL_APOC:-true}"

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

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Java
    if command -v java >/dev/null 2>&1; then
        JAVA_VERSION=$(java -version 2>&1 | head -n1 | cut -d'"' -f2 | cut -d'.' -f1)
        if [ "$JAVA_VERSION" -ge 11 ]; then
            log_success "Java $JAVA_VERSION found"
        else
            log_error "Java 11 or higher required, found Java $JAVA_VERSION"
            exit 1
        fi
    else
        log_error "Java not found. Please install Java 11 or higher"
        exit 1
    fi
    
    # Check curl
    if ! command -v curl >/dev/null 2>&1; then
        log_error "curl not found. Please install curl"
        exit 1
    fi
    
    # Check tar
    if ! command -v tar >/dev/null 2>&1; then
        log_error "tar not found. Please install tar"
        exit 1
    fi
}

# Install Neo4j on Debian/Ubuntu
install_debian() {
    log_info "Installing Neo4j on Debian/Ubuntu..."
    
    # Add Neo4j repository
    curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg
    echo "deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list
    
    # Update package list
    sudo apt-get update
    
    # Install Neo4j
    sudo apt-get install -y neo4j="${NEO4J_VERSION}"
    
    log_success "Neo4j installed via APT"
}

# Install Neo4j on RedHat/CentOS/Fedora
install_redhat() {
    log_info "Installing Neo4j on RedHat/CentOS/Fedora..."
    
    # Add Neo4j repository
    sudo tee /etc/yum.repos.d/neo4j.repo > /dev/null <<EOF
[neo4j]
name=Neo4j RPM Repository
baseurl=https://yum.neo4j.com/stable/5
enabled=1
gpgcheck=1
gpgkey=https://debian.neo4j.com/neotechnology.gpg.key
EOF
    
    # Install Neo4j
    sudo yum install -y neo4j-${NEO4J_VERSION}
    
    log_success "Neo4j installed via YUM"
}

# Install Neo4j on macOS
install_macos() {
    log_info "Installing Neo4j on macOS..."
    
    if command -v brew >/dev/null 2>&1; then
        # Use Homebrew
        brew install neo4j
        log_success "Neo4j installed via Homebrew"
    else
        # Manual installation
        install_manual
    fi
}

# Manual installation (universal)
install_manual() {
    log_info "Performing manual Neo4j installation..."
    
    # Create installation directory
    sudo mkdir -p "$NEO4J_HOME"
    
    # Download Neo4j
    NEO4J_DOWNLOAD_URL="https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz"
    log_info "Downloading Neo4j ${NEO4J_VERSION}..."
    
    TEMP_DIR=$(mktemp -d)
    curl -L "$NEO4J_DOWNLOAD_URL" -o "$TEMP_DIR/neo4j.tar.gz"
    
    # Extract Neo4j
    log_info "Extracting Neo4j..."
    tar -xzf "$TEMP_DIR/neo4j.tar.gz" -C "$TEMP_DIR"
    
    # Move to installation directory
    sudo mv "$TEMP_DIR/neo4j-community-${NEO4J_VERSION}"/* "$NEO4J_HOME/"
    
    # Clean up
    rm -rf "$TEMP_DIR"
    
    # Set ownership
    sudo chown -R $(whoami):$(whoami) "$NEO4J_HOME"
    
    # Add to PATH
    if ! grep -q "$NEO4J_HOME/bin" ~/.bashrc; then
        echo "export PATH=\$PATH:$NEO4J_HOME/bin" >> ~/.bashrc
        log_info "Added Neo4j to PATH in ~/.bashrc"
    fi
    
    log_success "Neo4j installed manually to $NEO4J_HOME"
}

# Install APOC plugin
install_apoc() {
    if [ "$INSTALL_APOC" != "true" ]; then
        return
    fi
    
    log_info "Installing APOC plugin..."
    
    # Determine plugins directory
    if [ -d "/var/lib/neo4j/plugins" ]; then
        PLUGINS_DIR="/var/lib/neo4j/plugins"
    elif [ -d "$NEO4J_HOME/plugins" ]; then
        PLUGINS_DIR="$NEO4J_HOME/plugins"
    else
        log_warning "Could not find Neo4j plugins directory"
        return
    fi
    
    # Download APOC
    APOC_VERSION="5.15.0"
    APOC_URL="https://github.com/neo4j/apoc/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-core.jar"
    
    log_info "Downloading APOC plugin..."
    sudo curl -L "$APOC_URL" -o "$PLUGINS_DIR/apoc-${APOC_VERSION}-core.jar"
    
    log_success "APOC plugin installed"
}

# Configure Neo4j
configure_neo4j() {
    log_info "Configuring Neo4j..."
    
    # Determine config file location
    if [ -f "/etc/neo4j/neo4j.conf" ]; then
        CONFIG_FILE="/etc/neo4j/neo4j.conf"
    elif [ -f "$NEO4J_HOME/conf/neo4j.conf" ]; then
        CONFIG_FILE="$NEO4J_HOME/conf/neo4j.conf"
    else
        log_warning "Could not find Neo4j configuration file"
        return
    fi
    
    log_info "Configuring $CONFIG_FILE"
    
    # Backup original config
    sudo cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
    
    # Apply Tyra-specific configurations
    sudo tee -a "$CONFIG_FILE" > /dev/null <<EOF

# Tyra Memory Server Configurations
# Added by install_neo4j.sh

# Memory settings for optimal performance
dbms.memory.heap.initial_size=512M
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G

# Security settings
dbms.security.auth_enabled=true
dbms.default_listen_address=0.0.0.0
dbms.connector.bolt.listen_address=:7687
dbms.connector.http.listen_address=:7474

# APOC settings
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*
apoc.export.file.enabled=true
apoc.import.file.enabled=true
apoc.import.file.use_neo4j_config=true

# Query logging for monitoring
dbms.logs.query.enabled=true
dbms.logs.query.threshold=1000ms
dbms.logs.query.parameter_logging_enabled=true

# Transaction settings
dbms.transaction.timeout=60s
dbms.transaction.concurrent.maximum=1000

# Cypher settings
cypher.lenient_create_relationship=true
cypher.forbid_exhaustive_shortestpath=false

# Network settings
dbms.connector.bolt.thread_pool_min_size=5
dbms.connector.bolt.thread_pool_max_size=400
EOF
    
    log_success "Neo4j configuration updated"
}

# Set initial password
set_initial_password() {
    log_info "Setting initial password..."
    
    if command -v neo4j-admin >/dev/null 2>&1; then
        echo "$NEO4J_PASSWORD" | sudo neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD" || true
        log_success "Initial password set"
    else
        log_warning "neo4j-admin not found, password will need to be set manually"
    fi
}

# Start Neo4j service
start_neo4j() {
    log_info "Starting Neo4j service..."
    
    OS=$(detect_os)
    
    case $OS in
        "debian"|"redhat")
            sudo systemctl enable neo4j
            sudo systemctl start neo4j
            sleep 10
            if sudo systemctl is-active --quiet neo4j; then
                log_success "Neo4j service started successfully"
            else
                log_error "Failed to start Neo4j service"
                exit 1
            fi
            ;;
        *)
            if command -v neo4j >/dev/null 2>&1; then
                neo4j start
                log_success "Neo4j started manually"
            else
                log_warning "Please start Neo4j manually"
            fi
            ;;
    esac
}

# Verify installation
verify_installation() {
    log_info "Verifying Neo4j installation..."
    
    # Wait for Neo4j to be ready
    local retries=30
    local count=0
    
    while [ $count -lt $retries ]; do
        if command -v cypher-shell >/dev/null 2>&1; then
            if cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
                log_success "Neo4j is running and accessible"
                return 0
            fi
        else
            # Try with Python
            if python3 -c "
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('neo4j://localhost:7687', auth=('$NEO4J_USER', '$NEO4J_PASSWORD'))
    driver.verify_connectivity()
    driver.close()
    exit(0)
except:
    exit(1)
" >/dev/null 2>&1; then
                log_success "Neo4j is running and accessible (verified via Python)"
                return 0
            fi
        fi
        
        log_info "Waiting for Neo4j to be ready... ($((count + 1))/$retries)"
        sleep 5
        ((count++))
    done
    
    log_error "Neo4j verification failed after $retries attempts"
    return 1
}

# Display connection information
show_connection_info() {
    log_success "Neo4j installation completed!"
    echo ""
    echo "Connection Information:"
    echo "  Bolt URL: neo4j://localhost:7687"
    echo "  HTTP URL: http://localhost:7474"
    echo "  Username: $NEO4J_USER"
    echo "  Password: $NEO4J_PASSWORD"
    echo ""
    echo "Next steps:"
    echo "  1. Run: scripts/init_neo4j.sh"
    echo "  2. Start Tyra Memory Server"
    echo ""
}

# Main installation function
main() {
    log_info "Starting Neo4j installation for Tyra Memory Server"
    log_info "Neo4j Version: $NEO4J_VERSION"
    log_info "Install Location: $NEO4J_HOME"
    
    # Check if already installed
    if command -v neo4j >/dev/null 2>&1; then
        log_warning "Neo4j appears to be already installed"
        if [[ "${1:-}" != "--force" ]]; then
            log_info "Use --force to reinstall"
            exit 0
        fi
    fi
    
    check_prerequisites
    
    # Detect OS and install accordingly
    OS=$(detect_os)
    log_info "Detected OS: $OS"
    
    case $OS in
        "debian")
            install_debian
            ;;
        "redhat")
            install_redhat
            ;;
        "macos")
            install_macos
            ;;
        *)
            log_warning "Unsupported OS, using manual installation"
            install_manual
            ;;
    esac
    
    install_apoc
    configure_neo4j
    set_initial_password
    start_neo4j
    
    if verify_installation; then
        show_connection_info
    else
        log_error "Installation verification failed"
        exit 1
    fi
}

# Script help
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Neo4j Installation Script for Tyra Memory Server"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --force           Force reinstallation even if Neo4j is detected"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NEO4J_VERSION     Neo4j version to install (default: 5.15.0)"
    echo "  NEO4J_HOME        Installation directory (default: /opt/neo4j)"
    echo "  NEO4J_USER        Neo4j username (default: neo4j)"
    echo "  NEO4J_PASSWORD    Neo4j password (default: neo4j)"
    echo "  INSTALL_APOC      Install APOC plugin (default: true)"
    exit 0
fi

# Run main function
main "$@"