# 📚 Tyra MCP Memory Server - Complete User Guide

*The ultimate guide to using all features and capabilities of the Tyra Advanced Memory System*

## 🎯 Table of Contents

1. [Quick Start](#-quick-start)
2. [Environment Setup](#-environment-setup)
3. [MCP Tools Reference](#-mcp-tools-reference)
4. [File Ingestion System](#-file-ingestion-system)
5. [Dashboard Access & Navigation](#-dashboard-access--navigation)
6. [Configuration Guide](#-configuration-guide)
7. [Local LLM Integration](#-local-llm-integration)
8. [API Usage](#-api-usage)
9. [Advanced Features](#-advanced-features)
10. [Troubleshooting](#-troubleshooting)
11. [Performance Tuning](#-performance-tuning)

---

## 🚀 Quick Start

### Prerequisites Checklist
- ✅ Python 3.11+
- ✅ PostgreSQL with pgvector extension
- ✅ Redis server
- ✅ Neo4j database
- ✅ HuggingFace CLI (`pip install huggingface-hub`)
- ✅ Git LFS (`git lfs install`)

### 1-Minute Setup
```bash
# Clone and setup
git clone <repository-url>
cd tyra-mcp-memory-server

# Automated setup (recommended)
./setup.sh --env development

# Start the server
source venv/bin/activate
python main.py
```

### Verify Installation
```bash
# Check health
curl http://localhost:8000/health

# Test MCP connection (if using with Claude)
# Server should appear in Claude's tool list
```

---

## ⚙️ Environment Setup

### Complete .env Configuration

Create `.env` file in project root with these settings:

```bash
# =============================================================================
# CORE APPLICATION SETTINGS
# =============================================================================
TYRA_ENV=development                    # development/production/testing
TYRA_DEBUG=true                         # Enable debug mode
TYRA_LOG_LEVEL=INFO                     # DEBUG/INFO/WARNING/ERROR
TYRA_HOT_RELOAD=true                    # Development hot reload

# =============================================================================
# DATABASE CONFIGURATIONS
# =============================================================================

# PostgreSQL (Primary Database + Vector Store)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tyra_memory
POSTGRES_USER=tyra
POSTGRES_PASSWORD=tyra_secure_password
POSTGRES_SSL_MODE=prefer               # disable/prefer/require
POSTGRES_SSL_CERT=                     # Path to SSL cert (optional)
POSTGRES_SSL_KEY=                      # Path to SSL key (optional)
POSTGRES_SSL_CA=                       # Path to SSL CA (optional)

# Redis (Caching Layer)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=                        # Leave empty if no auth

# Neo4j (Knowledge Graph)
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=neo4j                   # Default database name
NEO4J_ENCRYPTED=false                  # Use encrypted connection

# =============================================================================
# AI/ML SERVICE CONFIGURATIONS
# =============================================================================

# Graphiti (Temporal Knowledge Graphs)
GRAPHITI_LLM_URL=http://localhost:8000/v1
GRAPHITI_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
GRAPHITI_EMBEDDING_MODEL=intfloat/e5-large-v2

# vLLM Integration (Optional - for LLM context enhancement)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_RERANK_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_API_KEY=                          # Optional API key

# External AI Services (Optional Fallbacks)
OPENAI_API_KEY=                        # Optional - for embedding fallback
COHERE_API_KEY=                        # Optional - for reranking fallback

# =============================================================================
# APPLICATION PERFORMANCE SETTINGS
# =============================================================================

# Memory Management
MEMORY_MAX_CHUNK_SIZE=2048             # Max chunk size for documents
MEMORY_CHUNK_OVERLAP=100               # Overlap between chunks
MEMORY_BATCH_SIZE=50                   # Batch size for processing

# Database Connection Pools
POSTGRES_POOL_SIZE=20                  # PostgreSQL connection pool size
REDIS_POOL_SIZE=50                     # Redis connection pool size
NEO4J_POOL_SIZE=10                     # Neo4j connection pool size

# Caching Configuration
CACHE_TTL_EMBEDDINGS=86400             # 24 hours
CACHE_TTL_SEARCH=3600                  # 1 hour
CACHE_TTL_RERANK=1800                  # 30 minutes
CACHE_MAX_SIZE=2GB                     # Maximum cache size

# =============================================================================
# API SERVER SETTINGS
# =============================================================================
API_HOST=0.0.0.0                       # API server host
API_PORT=8000                          # API server port
API_ENABLE_DOCS=true                   # Enable OpenAPI documentation
API_RATE_LIMIT=1000                    # Requests per minute
API_TIMEOUT=30                         # Request timeout in seconds

# Security Settings
API_KEY_ENABLED=false                  # Enable API key authentication
API_KEY=                               # Your secure API key
JWT_SECRET=                            # JWT secret for token auth

# CORS Configuration
CORS_ORIGINS=*                         # Allowed origins (* for all)
CORS_METHODS=GET,POST,PUT,DELETE       # Allowed HTTP methods
CORS_HEADERS=*                         # Allowed headers

# =============================================================================
# DOCUMENT INGESTION SETTINGS
# =============================================================================

# File Processing
INGESTION_MAX_FILE_SIZE=104857600      # 100MB max file size
INGESTION_MAX_BATCH_SIZE=100           # Max files per batch
INGESTION_CONCURRENT_LIMIT=20          # Concurrent processing limit
INGESTION_TIMEOUT=300                  # Processing timeout (seconds)

# LLM Enhancement for Ingestion
INGESTION_LLM_ENHANCEMENT=true         # Enable LLM context enhancement
INGESTION_LLM_MODE=rule_based          # rule_based/vllm/disabled
VLLM_ENDPOINT=http://localhost:8000/v1 # vLLM server endpoint
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_TIMEOUT=30                        # vLLM request timeout
VLLM_MAX_TOKENS=150                    # Max tokens for context generation
VLLM_TEMPERATURE=0.3                   # vLLM temperature setting

# Processing Settings
INGESTION_AUTO_EMBED=true              # Auto-generate embeddings
INGESTION_AUTO_GRAPH=true              # Auto-add to knowledge graph
INGESTION_EXTRACT_ENTITIES=true        # Extract entities from content
INGESTION_CREATE_RELATIONSHIPS=true    # Create entity relationships

# Error Handling
INGESTION_RETRY_ATTEMPTS=3             # Number of retry attempts
INGESTION_RETRY_DELAY=1.0              # Delay between retries (seconds)
INGESTION_HALLUCINATION_THRESHOLD=0.8  # Hallucination detection threshold

# Performance Optimization
INGESTION_STREAMING_THRESHOLD=10485760 # 10MB - use streaming for larger files
INGESTION_PARALLEL_CHUNKS=5            # Parallel chunk processing
INGESTION_CACHE_CONTENT=true           # Cache parsed content
INGESTION_CACHE_TTL=3600               # Cache TTL (1 hour)

# =============================================================================
# OBSERVABILITY & MONITORING
# =============================================================================

# OpenTelemetry Configuration
OTEL_ENABLED=true                      # Enable OpenTelemetry
OTEL_SERVICE_NAME=tyra-mcp-memory-server
OTEL_SERVICE_VERSION=1.0.0
OTEL_ENVIRONMENT=development

# Tracing
OTEL_TRACES_ENABLED=true               # Enable distributed tracing
OTEL_TRACES_EXPORTER=console           # console/jaeger/otlp
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=1.0            # Sample rate (1.0 = 100%)
OTEL_TRACES_MAX_SPANS=1000             # Max spans per trace

# Metrics
OTEL_METRICS_ENABLED=true              # Enable metrics collection
OTEL_METRICS_EXPORTER=console          # console/prometheus/otlp
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4318/v1/metrics
OTEL_METRIC_EXPORT_INTERVAL=60000      # Export interval (60 seconds)
OTEL_METRIC_EXPORT_TIMEOUT=30000       # Export timeout (30 seconds)

# Logging
OTEL_LOGS_ENABLED=true                 # Enable structured logging
OTEL_LOGS_EXPORTER=console             # console/file/otlp
LOG_FORMAT=json                        # json/text
LOG_ROTATION_ENABLED=true              # Enable log rotation
LOG_ROTATION_MAX_SIZE=100MB            # Max log file size
LOG_ROTATION_MAX_FILES=10              # Max log files to keep
LOG_ROTATION_MAX_AGE=30                # Max log age (days)

# =============================================================================
# RATE LIMITING & SECURITY
# =============================================================================

# Rate Limiting
RATE_LIMIT_REQUESTS=1000               # Requests per minute
RATE_LIMIT_WINDOW=60                   # Time window (seconds)
RATE_LIMIT_BURST=50                    # Burst limit (requests per second)

# Circuit Breaker Settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5    # Failures before opening circuit
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60    # Recovery timeout (seconds)
CIRCUIT_BREAKER_EXPECTED_EXCEPTION=Exception

# =============================================================================
# ADVANCED AI SETTINGS
# =============================================================================

# Embedding Models
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=auto                 # auto/cpu/cuda
MODEL_CACHE_DIR=./models/cache         # Model cache directory

# Confidence Scoring
CONFIDENCE_ROCK_SOLID=95               # Rock solid confidence threshold
CONFIDENCE_HIGH=80                     # High confidence threshold
CONFIDENCE_FUZZY=60                    # Fuzzy confidence threshold
CONFIDENCE_LOW=0                       # Low confidence threshold

# Hallucination Detection
HALLUCINATION_DETECTION_ENABLED=true  # Enable hallucination detection
HALLUCINATION_THRESHOLD=75             # Detection threshold (0-100)
HALLUCINATION_REQUIRE_EVIDENCE=true    # Require evidence for claims

# Trading Safety (Critical for Financial Operations)
TRADING_SAFETY_ENABLED=true           # Enable trading safety features
TRADING_CONFIDENCE_REQUIRED=95        # Required confidence for trading (unbypassable)
TRADING_AUDIT_ENABLED=true            # Enable audit logging for trading operations

# =============================================================================
# DEVELOPMENT & DEBUGGING
# =============================================================================

# Development Features
DEV_RELOAD_CONFIGS=true               # Reload configs without restart
DEV_VERBOSE_LOGGING=true              # Extra verbose logging
DEV_MOCK_EXTERNAL_SERVICES=false     # Mock external services
DEV_PROFILE_PERFORMANCE=false        # Enable performance profiling

# Testing Configuration
TEST_DATABASE_URL=postgresql://tyra:test@localhost:5432/tyra_test
TEST_REDIS_URL=redis://localhost:6379/1
TEST_NEO4J_URL=neo4j://localhost:7687/test
TEST_SKIP_MODEL_DOWNLOAD=false       # Skip model downloads in tests
```

### Environment-Specific Configurations

#### Development (.env.development)
```bash
TYRA_ENV=development
TYRA_DEBUG=true
TYRA_LOG_LEVEL=DEBUG
API_ENABLE_DOCS=true
OTEL_TRACES_SAMPLER_ARG=1.0
```

#### Production (.env.production)
```bash
TYRA_ENV=production
TYRA_DEBUG=false
TYRA_LOG_LEVEL=INFO
API_ENABLE_DOCS=false
OTEL_TRACES_SAMPLER_ARG=0.1
POSTGRES_SSL_MODE=require
```

#### Testing (.env.testing)
```bash
TYRA_ENV=testing
TYRA_DEBUG=true
TYRA_LOG_LEVEL=DEBUG
TEST_SKIP_MODEL_DOWNLOAD=true
```

---

## 🛠️ MCP Tools Reference

The Tyra MCP Memory Server provides **17 powerful tools** for memory management, analysis, and optimization.

### 📝 Core Memory Operations

#### 1. `store_memory`
**Purpose**: Store content with automatic entity extraction and metadata enrichment.

**Parameters**:
- `content` (required): The content to store
- `agent_id` (default: "tyra"): Agent identifier for memory isolation
- `session_id` (optional): Session identifier for context grouping
- `metadata` (optional): Additional metadata object
- `extract_entities` (default: true): Automatically extract entities
- `chunk_content` (default: false): Split large content into chunks

**Example**:
```json
{
  "tool": "store_memory",
  "content": "User prefers morning trading sessions and uses technical analysis for swing trades",
  "agent_id": "tyra",
  "session_id": "trading_session_001",
  "extract_entities": true,
  "metadata": {
    "category": "trading_preferences",
    "confidence": 95,
    "source": "user_conversation"
  }
}
```

**Returns**: Memory ID, confidence score, extracted entities, and storage confirmation.

#### 2. `search_memory`
**Purpose**: Advanced hybrid search with confidence scoring and hallucination detection.

**Parameters**:
- `query` (required): Search query
- `agent_id` (optional): Filter by agent
- `session_id` (optional): Filter by session
- `top_k` (default: 10): Number of results to return
- `min_confidence` (default: 0.0): Minimum confidence threshold
- `search_type` (enum): "vector", "graph", or "hybrid" (default: "hybrid")
- `include_analysis` (default: true): Include hallucination analysis

**Example**:
```json
{
  "tool": "search_memory",
  "query": "What are the user's trading preferences and risk tolerance?",
  "agent_id": "tyra",
  "search_type": "hybrid",
  "top_k": 10,
  "min_confidence": 0.7,
  "include_analysis": true
}
```

**Returns**: Ranked memories with confidence scores, grounding analysis, and evidence collection.

#### 3. `delete_memory`
**Purpose**: Remove specific memories by ID.

**Parameters**:
- `memory_id` (required): The ID of the memory to delete

**Example**:
```json
{
  "tool": "delete_memory",
  "memory_id": "mem_12345"
}
```

### 🛡️ Analysis & Validation

#### 4. `analyze_response`
**Purpose**: Analyze any response for hallucinations, confidence scoring, and grounding.

**Parameters**:
- `response` (required): The response to analyze
- `query` (optional): Original query for context
- `retrieved_memories` (optional): Array of source memories

**Example**:
```json
{
  "tool": "analyze_response",
  "response": "Based on your trading history, you prefer swing trading with technical analysis",
  "query": "What's my trading style?",
  "retrieved_memories": [...]
}
```

**Returns**: Confidence levels (💪 Rock Solid 95%+, 🧠 High 80%+, 🤔 Fuzzy 60%+, ⚠️ Low <60%), hallucination detection, grounding score, and evidence collection.

### 📊 Advanced Analytics

#### 5. `deduplicate_memories`
**Purpose**: Find and merge duplicate or highly similar memories.

**Parameters**:
- `agent_id` (optional): Target agent
- `similarity_threshold` (0.7-1.0, default: 0.85): Similarity threshold for duplicates
- `auto_merge` (default: false): Automatically merge duplicates

**Example**:
```json
{
  "tool": "deduplicate_memories",
  "agent_id": "tyra",
  "similarity_threshold": 0.9,
  "auto_merge": false
}
```

#### 6. `summarize_memories`
**Purpose**: AI-powered summarization with anti-hallucination validation.

**Parameters**:
- `memory_ids` (required): Array of memory IDs to summarize
- `summary_type` (enum): "extractive", "abstractive", "hybrid", or "progressive" (default: "hybrid")
- `max_length` (50-500, default: 200): Maximum summary length

**Example**:
```json
{
  "tool": "summarize_memories",
  "memory_ids": ["mem_001", "mem_002", "mem_003"],
  "summary_type": "hybrid",
  "max_length": 150
}
```

#### 7. `detect_patterns`
**Purpose**: Pattern recognition and knowledge gap detection across memories.

**Parameters**:
- `agent_id` (optional): Target agent
- `min_cluster_size` (default: 3): Minimum memories per pattern
- `include_recommendations` (default: true): Include improvement recommendations

**Example**:
```json
{
  "tool": "detect_patterns",
  "agent_id": "tyra",
  "min_cluster_size": 5,
  "include_recommendations": true
}
```

#### 8. `analyze_temporal_evolution`
**Purpose**: Track how concepts and memories evolve over time.

**Parameters**:
- `concept` (optional): Specific concept to analyze
- `agent_id` (optional): Target agent
- `time_window_days` (default: 30): Analysis time window

**Example**:
```json
{
  "tool": "analyze_temporal_evolution",
  "concept": "trading_strategy",
  "agent_id": "tyra",
  "time_window_days": 90
}
```

### 🌐 Web Integration

#### 9. `crawl_website`
**Purpose**: Natural language web crawling with AI-powered content extraction.

**Parameters**:
- `command` (required): Natural language crawling instruction
- `max_pages` (1-100, default: 10): Maximum pages to crawl
- `strategy` (enum): "single_page", "site_map", "recursive", or "rss_feed"
- `store_in_memory` (default: true): Store extracted content in memory
- `agent_id` (default: "tyra"): Agent for memory storage

**Example**:
```json
{
  "tool": "crawl_website",
  "command": "Get the latest market analysis from tradingview.com focusing on S&P 500 trends",
  "max_pages": 5,
  "strategy": "recursive",
  "store_in_memory": true,
  "agent_id": "tyra"
}
```

### 📊 System Monitoring

#### 10. `get_memory_stats`
**Purpose**: Comprehensive system statistics and health metrics.

**Parameters**:
- `agent_id` (optional): Filter by agent
- `include_performance` (default: true): Include performance metrics
- `include_recommendations` (default: true): Include optimization recommendations

**Example**:
```json
{
  "tool": "get_memory_stats",
  "agent_id": "tyra",
  "include_performance": true,
  "include_recommendations": true
}
```

#### 11. `health_check`
**Purpose**: Complete system health assessment.

**Parameters**:
- `detailed` (default: false): Include detailed component status

**Example**:
```json
{
  "tool": "health_check",
  "detailed": true
}
```

### 🎯 Adaptive Learning

#### 12. `get_learning_insights`
**Purpose**: Access adaptive learning insights and optimization data.

**Parameters**:
- `category` (optional): Specific insight category
- `days` (1-90, default: 7): Analysis time window

**Example**:
```json
{
  "tool": "get_learning_insights",
  "category": "performance_optimization",
  "days": 14
}
```

### 🧠 Intelligent Suggestions

#### 13. `suggest_related_memories`
**Purpose**: ML-powered suggestions for related memories based on content or memory ID.

**Parameters**:
- `memory_id` OR `content` (one required): Reference for suggestions
- `agent_id` (default: "tyra"): Target agent
- `limit` (1-50, default: 10): Number of suggestions
- `min_relevance` (0.0-1.0, default: 0.3): Minimum relevance threshold

**Example**:
```json
{
  "tool": "suggest_related_memories",
  "content": "risk management strategies",
  "agent_id": "tyra",
  "limit": 15,
  "min_relevance": 0.5
}
```

#### 14. `detect_memory_connections`
**Purpose**: Automatically detect connections between memories.

**Parameters**:
- `agent_id` (default: "tyra"): Target agent
- `connection_types` (default: ["semantic", "temporal", "entity"]): Types of connections to detect
- `min_confidence` (0.0-1.0, default: 0.5): Minimum confidence for connections

**Example**:
```json
{
  "tool": "detect_memory_connections",
  "agent_id": "tyra",
  "connection_types": ["semantic", "temporal"],
  "min_confidence": 0.7
}
```

#### 15. `recommend_memory_organization`
**Purpose**: Get AI recommendations for improving memory organization.

**Parameters**:
- `agent_id` (default: "tyra"): Target agent
- `analysis_type` (enum): "clustering", "hierarchy", "topics", or "all" (default: "all")

**Example**:
```json
{
  "tool": "recommend_memory_organization",
  "agent_id": "tyra",
  "analysis_type": "clustering"
}
```

#### 16. `detect_knowledge_gaps`
**Purpose**: Identify gaps in the knowledge base.

**Parameters**:
- `agent_id` (default: "tyra"): Target agent
- `domains` (optional): Array of specific domains to analyze
- `gap_types` (default: ["topic", "temporal", "detail"]): Types of gaps to detect

**Example**:
```json
{
  "tool": "detect_knowledge_gaps",
  "agent_id": "tyra",
  "domains": ["trading", "risk_management"],
  "gap_types": ["topic", "detail"]
}
```

---

## 📁 File Ingestion System

### Automatic Folder Ingestion

The Tyra system includes an automatic file ingestion system that monitors a folder for new files and processes them automatically.

#### Folder Structure
```
tyra-ingest/
├── inbox/          # 📥 Drop files here for automatic processing
├── processed/      # ✅ Successfully processed files
└── failed/         # ❌ Files that failed processing (with error logs)
```

#### Supported File Types
- **📄 PDF** (`.pdf`) - PDF documents with text extraction
- **📝 Word** (`.docx`) - Microsoft Word documents
- **📄 Text** (`.txt`) - Plain text files
- **📖 Markdown** (`.md`) - Markdown formatted files
- **🌐 HTML** (`.html`) - HTML documents
- **🔧 JSON** (`.json`) - JSON data files
- **📊 CSV** (`.csv`) - Comma-separated value files

#### How It Works

1. **📂 File Detection**: File watcher monitors `inbox/` folder for new files
2. **⏳ Stability Check**: Ensures files are completely written (not being modified)
3. **🔍 Duplicate Detection**: MD5 hashing prevents processing duplicates
4. **⚙️ Processing**: Files processed through document ingestion pipeline
5. **🧠 Memory Storage**: Content embedded and stored with entity extraction
6. **📁 Organization**: Files moved to `processed/` or `failed/` based on outcome

#### Usage Example

```bash
# 1. Drop a PDF file into the inbox
cp "trading_strategy_2024.pdf" tyra-ingest/inbox/

# 2. File watcher automatically detects and processes the file
# 3. Check processing status
curl http://localhost:8000/v1/file-watcher/status

# 4. Verify the file was processed
ls tyra-ingest/processed/

# 5. Search for the content
# Use search_memory tool to find content from the PDF
```

#### File Watcher API

Monitor and control the file ingestion service:

```bash
# Check service status
curl http://localhost:8000/v1/file-watcher/status

# Get processing statistics
curl http://localhost:8000/v1/file-watcher/stats

# Start/stop the service
curl -X POST http://localhost:8000/v1/file-watcher/start
curl -X POST http://localhost:8000/v1/file-watcher/stop

# Restart the service
curl -X POST http://localhost:8000/v1/file-watcher/restart

# Health check
curl http://localhost:8000/v1/file-watcher/health
```

#### Configuration

File watcher settings in `config/config.yaml`:

```yaml
file_watcher:
  enabled: true
  paths:
    base_path: "tyra-ingest"
    inbox: "tyra-ingest/inbox"
    processed: "tyra-ingest/processed"
    failed: "tyra-ingest/failed"
  processing:
    agent_id: "tyra"              # Default agent for processed files
    auto_start: true              # Start service automatically
    check_interval: 0.5           # Check for files every 0.5 seconds
    stability_delay: 2.0          # Wait 2 seconds to ensure file stability
    max_file_size: 104857600      # 100MB maximum file size
    batch_processing: true        # Process multiple files in parallel
    max_concurrent: 5             # Maximum concurrent file processing
```

#### Troubleshooting File Ingestion

**Files not being processed?**
1. Check service status: `curl http://localhost:8000/v1/file-watcher/status`
2. Verify file watcher is enabled in config
3. Check file permissions on `tyra-ingest/` folders
4. Review logs for error messages
5. Ensure file types are supported

**Files ending up in failed folder?**
1. Check error logs in the failed folder
2. Verify file is not corrupted
3. Check file size limits
4. Review memory storage quotas

---

## 🖥️ Dashboard Access & Navigation

### Accessing the Dashboard

The Tyra system provides multiple dashboard interfaces for monitoring and analytics.

#### Dashboard URLs
- **🏠 Main Dashboard**: `http://localhost:8050` (Dash application)
- **📚 API Documentation**: `http://localhost:8000/docs` (FastAPI auto-docs)
- **💚 Health Monitoring**: `http://localhost:8000/health` (System health)
- **📊 Metrics Endpoint**: `http://localhost:8000/metrics` (Prometheus metrics)

#### Starting the Dashboard

```bash
# Method 1: Integrated startup (recommended)
python main.py
# This starts both MCP server and dashboard

# Method 2: Separate dashboard startup
cd src/dashboard
python main.py
# Dashboard runs on port 8050

# Method 3: FastAPI with dashboard mounting
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# Includes dashboard routes at /dashboard/
```

### Dashboard Components

#### 1. 📊 Usage Analytics Dashboard
**URL**: `http://localhost:8050/usage`

**Features**:
- Memory usage patterns by agent
- Search query frequency and performance
- Document ingestion statistics
- Agent activity timelines
- Performance metrics over time

**What you'll see**:
- Interactive charts showing memory growth
- Heat maps of agent activity
- Search performance trends
- Memory utilization graphs

#### 2. 💰 ROI Analytics Dashboard
**URL**: `http://localhost:8050/roi`

**Features**:
- Return on investment metrics
- Memory efficiency analysis
- Cost-benefit analysis of different features
- Performance vs. resource usage

**What you'll see**:
- ROI calculations for memory operations
- Efficiency score trends
- Resource utilization vs. output quality
- Cost optimization recommendations

#### 3. 🔍 Knowledge Gaps Dashboard
**URL**: `http://localhost:8050/gaps`

**Features**:
- Identified knowledge gaps in the memory system
- Coverage analysis by topic/domain
- Recommendations for improving knowledge base
- Gap evolution over time

**What you'll see**:
- Knowledge gap visualizations
- Coverage heat maps
- Recommendation lists
- Gap closure tracking

#### 4. 🕸️ Network Visualization Dashboard
**URL**: `http://localhost:8050/network`

**Features**:
- Interactive knowledge graph visualization
- Entity relationship networks
- Memory connection patterns
- Graph analytics and centrality metrics

**What you'll see**:
- Interactive network graphs
- Entity clustering visualization
- Relationship strength indicators
- Network analysis metrics

### Dashboard Navigation

#### Main Navigation Menu
```
🏠 Home                     # Overview and system status
├── 📊 Usage Analytics      # Memory usage and performance metrics
├── 💰 ROI Analysis        # Return on investment calculations
├── 🔍 Knowledge Gaps      # Knowledge gap analysis and recommendations
├── 🕸️ Network Visualization # Interactive knowledge graph
└── ❤️ System Health       # Component health and diagnostics
```

#### Using the Dashboards

1. **📊 Monitor Performance**: Use Usage Analytics to track system performance and identify bottlenecks
2. **💰 Optimize ROI**: Use ROI Analysis to understand cost-effectiveness and optimize configurations
3. **🔍 Fill Knowledge Gaps**: Use Knowledge Gaps dashboard to identify and address missing information
4. **🕸️ Explore Connections**: Use Network Visualization to understand relationships and discover insights
5. **❤️ Check Health**: Use System Health to monitor component status and troubleshoot issues

#### Dashboard API Endpoints

You can also access dashboard data programmatically:

```bash
# Get usage analytics data
curl http://localhost:8000/v1/analytics/usage

# Get ROI metrics
curl http://localhost:8000/v1/analytics/roi

# Get knowledge gap analysis
curl http://localhost:8000/v1/analytics/gaps

# Get network visualization data
curl http://localhost:8000/v1/analytics/network

# Get system health status
curl http://localhost:8000/v1/health/detailed
```

---

## 🔧 Configuration Guide

### Configuration File Structure

The Tyra system uses a layered configuration approach with multiple YAML files:

```
config/
├── config.yaml              # 🏠 Main application settings
├── providers.yaml           # 🔌 Provider configurations
├── rag.yaml                # 🧠 RAG and search settings
├── ingestion.yaml          # 📄 Document processing settings
├── agents.yaml             # 🤖 Agent-specific configurations
├── models.yaml             # 🔬 AI model configurations
├── observability.yaml      # 📊 Monitoring and logging
├── self_learning.yaml      # 🎯 Adaptive learning settings
└── local/                  # 🔒 Local overrides (gitignored)
    ├── secrets.yaml        # 🔐 Security secrets
    └── overrides.yaml      # ⚙️ Environment-specific overrides
```

### Main Configuration (`config/config.yaml`)

```yaml
# =============================================================================
# CORE APPLICATION SETTINGS
# =============================================================================
app:
  name: "Tyra MCP Memory Server"
  version: "1.0.0"
  environment: ${TYRA_ENV:-development}
  debug: ${TYRA_DEBUG:-false}
  log_level: ${TYRA_LOG_LEVEL:-INFO}

# =============================================================================
# API SERVER CONFIGURATION
# =============================================================================
api:
  host: ${API_HOST:-0.0.0.0}
  port: ${API_PORT:-8000}
  enable_docs: ${API_ENABLE_DOCS:-true}
  enable_redoc: true
  cors_origins: ${CORS_ORIGINS:-["*"]}
  cors_methods: ${CORS_METHODS:-["GET", "POST", "PUT", "DELETE"]}
  cors_headers: ${CORS_HEADERS:-["*"]}
  
  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: ${API_RATE_LIMIT:-1000}
    requests_per_second: ${API_RATE_LIMIT_BURST:-50}
    
  # Request settings
  max_request_size: ${API_MAX_REQUEST_SIZE:-104857600}  # 100MB
  request_timeout: ${API_TIMEOUT:-30}

# =============================================================================
# MEMORY SYSTEM CONFIGURATION
# =============================================================================
memory:
  # Backend storage
  backend: "postgres"
  vector_dimensions: 1024
  
  # Chunking settings
  chunk_size: ${MEMORY_CHUNK_SIZE:-512}
  chunk_overlap: ${MEMORY_CHUNK_OVERLAP:-50}
  max_chunk_size: ${MEMORY_MAX_CHUNK_SIZE:-2048}
  
  # Agent limits
  max_memories_per_agent: ${MEMORY_MAX_PER_AGENT:-1000000}
  max_agents: ${MEMORY_MAX_AGENTS:-100}
  
  # Performance settings
  batch_size: ${MEMORY_BATCH_SIZE:-50}
  concurrent_operations: ${MEMORY_CONCURRENT_OPS:-10}

# =============================================================================
# DATABASE CONFIGURATIONS
# =============================================================================
databases:
  postgres:
    host: ${POSTGRES_HOST:-localhost}
    port: ${POSTGRES_PORT:-5432}
    database: ${POSTGRES_DB:-tyra_memory}
    username: ${POSTGRES_USER:-tyra}
    password: ${POSTGRES_PASSWORD:-}
    pool_size: ${POSTGRES_POOL_SIZE:-20}
    max_connections: ${POSTGRES_MAX_CONNECTIONS:-100}
    ssl_mode: ${POSTGRES_SSL_MODE:-prefer}
    
  redis:
    host: ${REDIS_HOST:-localhost}
    port: ${REDIS_PORT:-6379}
    database: ${REDIS_DB:-0}
    password: ${REDIS_PASSWORD:-}
    pool_size: ${REDIS_POOL_SIZE:-50}
    
    # Cache TTL settings (in seconds)
    cache_ttl:
      embeddings: ${CACHE_TTL_EMBEDDINGS:-86400}      # 24 hours
      search_results: ${CACHE_TTL_SEARCH:-3600}       # 1 hour
      rerank_cache: ${CACHE_TTL_RERANK:-1800}         # 30 minutes
      graph_queries: ${CACHE_TTL_GRAPH:-7200}         # 2 hours
      
  neo4j:
    host: ${NEO4J_HOST:-localhost}
    port: ${NEO4J_PORT:-7687}
    username: ${NEO4J_USER:-neo4j}
    password: ${NEO4J_PASSWORD:-neo4j}
    database: ${NEO4J_DATABASE:-neo4j}
    encrypted: ${NEO4J_ENCRYPTED:-false}
    pool_size: ${NEO4J_POOL_SIZE:-10}

# =============================================================================
# FILE WATCHER CONFIGURATION
# =============================================================================
file_watcher:
  enabled: ${FILE_WATCHER_ENABLED:-true}
  
  paths:
    base_path: ${FILE_WATCHER_BASE_PATH:-tyra-ingest}
    inbox: ${FILE_WATCHER_INBOX:-tyra-ingest/inbox}
    processed: ${FILE_WATCHER_PROCESSED:-tyra-ingest/processed}
    failed: ${FILE_WATCHER_FAILED:-tyra-ingest/failed}
    
  processing:
    agent_id: ${FILE_WATCHER_AGENT_ID:-tyra}
    auto_start: ${FILE_WATCHER_AUTO_START:-true}
    check_interval: ${FILE_WATCHER_CHECK_INTERVAL:-0.5}
    stability_delay: ${FILE_WATCHER_STABILITY_DELAY:-2.0}
    max_file_size: ${FILE_WATCHER_MAX_FILE_SIZE:-104857600}  # 100MB
    batch_processing: ${FILE_WATCHER_BATCH_PROCESSING:-true}
    max_concurrent: ${FILE_WATCHER_MAX_CONCURRENT:-5}
    
  # Supported file extensions
  supported_extensions:
    - ".pdf"
    - ".docx"
    - ".txt"
    - ".md"
    - ".html"
    - ".json"
    - ".csv"

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================
security:
  # API authentication
  api_key:
    enabled: ${API_KEY_ENABLED:-false}
    key: ${API_KEY:-}
    
  # JWT settings
  jwt:
    secret: ${JWT_SECRET:-}
    algorithm: "HS256"
    expire_minutes: ${JWT_EXPIRE_MINUTES:-1440}  # 24 hours
    
  # Rate limiting
  rate_limiting:
    enabled: true
    storage: "redis"  # redis, memory
    key_func: "ip"    # ip, user, api_key
    
  # CORS settings
  cors:
    allow_origins: ${CORS_ORIGINS:-["*"]}
    allow_methods: ${CORS_METHODS:-["GET", "POST", "PUT", "DELETE"]}
    allow_headers: ${CORS_HEADERS:-["*"]}
    allow_credentials: ${CORS_ALLOW_CREDENTIALS:-false}
```

### Provider Configuration (`config/providers.yaml`)

```yaml
# =============================================================================
# EMBEDDING PROVIDERS
# =============================================================================
embeddings:
  primary: "huggingface"
  fallback: "huggingface_light"
  
  providers:
    huggingface:
      model_name: "intfloat/e5-large-v2"
      model_path: "./models/embeddings/e5-large-v2"
      use_local_files: true
      device: ${EMBEDDINGS_DEVICE:-auto}
      batch_size: ${EMBEDDINGS_BATCH_SIZE:-32}
      max_length: 512
      normalize_embeddings: true
      
    huggingface_light:
      model_name: "sentence-transformers/all-MiniLM-L12-v2"
      model_path: "./models/embeddings/all-MiniLM-L12-v2"
      use_local_files: true
      device: ${EMBEDDINGS_DEVICE:-auto}
      batch_size: ${EMBEDDINGS_BATCH_SIZE:-16}
      max_length: 384
      normalize_embeddings: true
      
    # Optional external providers (fallback)
    openai:
      model: "text-embedding-3-large"
      api_key: ${OPENAI_API_KEY:-}
      dimensions: 1024
      
    cohere:
      model: "embed-english-v3.0"
      api_key: ${COHERE_API_KEY:-}

# =============================================================================
# RERANKING PROVIDERS
# =============================================================================
reranking:
  primary: "cross_encoder"
  fallback: "similarity"
  
  providers:
    cross_encoder:
      model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      model_path: "./models/cross-encoders/ms-marco-MiniLM-L-6-v2"
      use_local_files: true
      device: ${RERANKING_DEVICE:-auto}
      batch_size: ${RERANKING_BATCH_SIZE:-16}
      
    vllm:
      base_url: ${VLLM_BASE_URL:-http://localhost:8000/v1}
      model: ${VLLM_RERANK_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
      api_key: ${VLLM_API_KEY:-}
      temperature: 0.1
      max_tokens: 10
      
    similarity:
      method: "cosine"
      normalize: true

# =============================================================================
# GRAPH PROVIDERS
# =============================================================================
graph:
  primary: "neo4j"
  
  providers:
    neo4j:
      integration: "graphiti"
      temporal_support: true
      causal_inference: true
      
      graphiti:
        llm_url: ${GRAPHITI_LLM_URL:-http://localhost:8000/v1}
        llm_model: ${GRAPHITI_LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
        embedding_model: ${GRAPHITI_EMBEDDING_MODEL:-intfloat/e5-large-v2}

# =============================================================================
# CACHE PROVIDERS
# =============================================================================
cache:
  layers:
    l1:
      provider: "memory"
      max_size: ${CACHE_L1_MAX_SIZE:-100MB}
      ttl: ${CACHE_L1_TTL:-300}  # 5 minutes
      
    l2:
      provider: "redis"
      max_size: ${CACHE_L2_MAX_SIZE:-1GB}
      ttl: ${CACHE_L2_TTL:-3600}  # 1 hour
      
    l3:
      provider: "postgres"
      max_size: ${CACHE_L3_MAX_SIZE:-10GB}
      ttl: ${CACHE_L3_TTL:-86400}  # 24 hours
```

### RAG Configuration (`config/rag.yaml`)

```yaml
# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
retrieval:
  # Search strategy weights
  hybrid_weight: ${RAG_HYBRID_WEIGHT:-0.7}  # vector vs keyword (0.7 = 70% vector, 30% keyword)
  max_results: ${RAG_MAX_RESULTS:-20}
  diversity_penalty: ${RAG_DIVERSITY_PENALTY:-0.1}
  
  # Vector search settings
  vector:
    similarity_threshold: ${RAG_VECTOR_THRESHOLD:-0.5}
    algorithm: "hnsw"
    ef_search: 100
    
  # Keyword search settings
  keyword:
    min_term_frequency: 1
    boost_phrase_matches: true
    fuzzy_matching: true
    
  # Graph traversal settings
  graph:
    max_depth: ${RAG_GRAPH_MAX_DEPTH:-3}
    relationship_weight: ${RAG_GRAPH_REL_WEIGHT:-0.5}
    temporal_decay: ${RAG_GRAPH_TEMPORAL_DECAY:-0.1}

# =============================================================================
# RERANKING CONFIGURATION
# =============================================================================
reranking:
  enabled: ${RAG_RERANKING_ENABLED:-true}
  provider: ${RAG_RERANKING_PROVIDER:-cross_encoder}
  top_k: ${RAG_RERANKING_TOP_K:-10}
  
  # Cross-encoder settings
  cross_encoder:
    batch_size: 16
    max_length: 512
    
  # vLLM settings
  vllm:
    temperature: 0.1
    max_tokens: 10
    timeout: 30

# =============================================================================
# HALLUCINATION DETECTION
# =============================================================================
hallucination:
  enabled: ${RAG_HALLUCINATION_ENABLED:-true}
  threshold: ${RAG_HALLUCINATION_THRESHOLD:-75}
  require_evidence: ${RAG_REQUIRE_EVIDENCE:-true}
  
  # Detection methods
  methods:
    grounding_check: true
    consistency_analysis: true
    fact_verification: true
    confidence_scoring: true
    
  # Confidence levels
  confidence_levels:
    rock_solid: ${CONFIDENCE_ROCK_SOLID:-95}
    high: ${CONFIDENCE_HIGH:-80}
    fuzzy: ${CONFIDENCE_FUZZY:-60}
    low: ${CONFIDENCE_LOW:-0}

# =============================================================================
# CONTEXT ENHANCEMENT
# =============================================================================
context:
  max_context_length: ${RAG_MAX_CONTEXT_LENGTH:-4096}
  context_overlap: ${RAG_CONTEXT_OVERLAP:-100}
  
  # Graph context
  graph_context:
    enabled: true
    max_relationships: 50
    relationship_depth: 2
    
  # Temporal context
  temporal_context:
    enabled: true
    time_decay: 0.1
    max_time_span: "365d"

# =============================================================================
# TRADING SAFETY (CRITICAL)
# =============================================================================
trading_safety:
  enabled: ${TRADING_SAFETY_ENABLED:-true}
  
  # Unbypassable confidence requirements
  confidence_requirements:
    trading_decisions: ${TRADING_CONFIDENCE_REQUIRED:-95}  # Rock solid only
    financial_advice: 90
    risk_assessment: 85
    
  # Audit requirements
  audit:
    enabled: ${TRADING_AUDIT_ENABLED:-true}
    log_all_operations: true
    require_evidence: true
    confidence_tracking: true
```

### Agent Configuration (`config/agents.yaml`)

```yaml
# =============================================================================
# AGENT DEFINITIONS
# =============================================================================
agents:
  tyra:
    name: "Tyra Trading Agent"
    description: "Advanced trading and financial analysis agent"
    
    # Memory settings
    memory:
      max_memories: 1000000
      retention_days: 365
      auto_cleanup: true
      
    # Trading safety settings
    trading_safety:
      enabled: true
      confidence_required: 95  # Rock solid confidence for trading
      audit_all_operations: true
      
    # Specialized features
    features:
      technical_analysis: true
      risk_assessment: true
      market_analysis: true
      portfolio_management: true
      
  claude:
    name: "Claude Assistant"
    description: "General purpose AI assistant"
    
    memory:
      max_memories: 500000
      retention_days: 180
      auto_cleanup: true
      
    features:
      general_assistance: true
      research_support: true
      document_analysis: true
      
  archon:
    name: "Archon Multi-Agent System"
    description: "Multi-agent coordination and orchestration"
    
    memory:
      max_memories: 750000
      retention_days: 270
      auto_cleanup: true
      
    features:
      agent_coordination: true
      workflow_management: true
      system_monitoring: true

# =============================================================================
# DEFAULT AGENT SETTINGS
# =============================================================================
defaults:
  memory:
    chunk_size: 512
    chunk_overlap: 50
    extract_entities: true
    create_relationships: true
    
  search:
    default_top_k: 10
    min_confidence: 0.5
    include_analysis: true
    
  safety:
    hallucination_detection: true
    confidence_scoring: true
    evidence_requirement: true
```

### Model Configuration (`config/models.yaml`)

```yaml
# =============================================================================
# AI MODEL CONFIGURATIONS
# =============================================================================
models:
  # Embedding models
  embeddings:
    primary:
      name: "intfloat/e5-large-v2"
      path: "./models/embeddings/e5-large-v2"
      dimensions: 1024
      max_length: 512
      performance:
        inference_time_gpu: "15ms"
        inference_time_cpu: "150ms"
        memory_usage: "2.5GB"
        
    fallback:
      name: "sentence-transformers/all-MiniLM-L12-v2"
      path: "./models/embeddings/all-MiniLM-L12-v2"
      dimensions: 384
      max_length: 384
      performance:
        inference_time_gpu: "5ms"
        inference_time_cpu: "50ms"
        memory_usage: "400MB"
        
  # Cross-encoder models
  cross_encoders:
    primary:
      name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      path: "./models/cross-encoders/ms-marco-MiniLM-L-6-v2"
      max_length: 512
      performance:
        inference_time_gpu: "10ms"
        inference_time_cpu: "100ms"
        memory_usage: "400MB"

# =============================================================================
# MODEL DOWNLOAD CONFIGURATIONS
# =============================================================================
download:
  # Required models (must be downloaded manually)
  required_models:
    - name: "intfloat/e5-large-v2"
      size: "1.34GB"
      local_path: "./models/embeddings/e5-large-v2"
      command: |
        huggingface-cli download intfloat/e5-large-v2 \
          --local-dir ./models/embeddings/e5-large-v2 \
          --local-dir-use-symlinks False
          
    - name: "sentence-transformers/all-MiniLM-L12-v2"
      size: "120MB"
      local_path: "./models/embeddings/all-MiniLM-L12-v2"
      command: |
        huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 \
          --local-dir ./models/embeddings/all-MiniLM-L12-v2 \
          --local-dir-use-symlinks False
          
    - name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      size: "120MB"
      local_path: "./models/cross-encoders/ms-marco-MiniLM-L-6-v2"
      command: |
        huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
          --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2 \
          --local-dir-use-symlinks False

# =============================================================================
# DEVICE CONFIGURATIONS
# =============================================================================
devices:
  # Auto-detection settings
  auto_detect: true
  prefer_gpu: true
  
  # GPU settings
  gpu:
    enabled: ${GPU_ENABLED:-true}
    device_id: ${CUDA_DEVICE:-0}
    memory_fraction: ${GPU_MEMORY_FRACTION:-0.8}
    allow_growth: true
    
  # CPU settings
  cpu:
    threads: ${CPU_THREADS:-auto}
    optimization: "auto"  # auto, mkl, openblas
```

---

## 🤖 Local LLM Integration

### Setting Up Your Local LLM

The Tyra system can integrate with local LLM services for enhanced context generation during document ingestion and reranking.

#### Step 1: Choose Your LLM Setup

**Option A: vLLM (Recommended)**
```bash
# Install vLLM
pip install vllm

# Start vLLM server with your model
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0
```

**Option B: Ollama**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull and run a model
ollama pull llama3.1:8b
ollama serve
```

**Option C: Text Generation WebUI**
```bash
# Clone and setup
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui
pip install -r requirements.txt

# Start with OpenAI-compatible API
python server.py --api --api-port 8000
```

#### Step 2: Configure Tyra Integration

**Environment Variables**:
```bash
# Enable LLM enhancement
export INGESTION_LLM_ENHANCEMENT=true
export INGESTION_LLM_MODE=vllm

# vLLM configuration
export VLLM_ENDPOINT=http://localhost:8000/v1
export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export VLLM_TIMEOUT=30
export VLLM_MAX_TOKENS=150
export VLLM_TEMPERATURE=0.3

# Enable vLLM reranking (optional)
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_RERANK_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

**Configuration File** (`config/ingestion.yaml`):
```yaml
llm_enhancement:
  enabled: true
  default_mode: vllm  # Changes from rule_based to vllm
  
  vllm_integration:
    enabled: true
    endpoint: http://localhost:8000/v1
    model: meta-llama/Llama-3.1-8B-Instruct
    timeout: 30
    max_tokens: 150
    temperature: 0.3
    
    # Context generation prompts
    prompts:
      document_context: |
        Analyze this document and provide a concise context summary.
        Document: {content}
        
        Provide a 2-3 sentence summary focusing on:
        1. Main topic/purpose
        2. Key information
        3. Relevance for memory storage
        
        Context:
        
      chunk_context: |
        This is a chunk from a larger document. Provide context that will help with retrieval.
        
        Document title: {title}
        Chunk content: {chunk}
        
        Context (1-2 sentences):
```

#### Step 3: Verify LLM Integration

```bash
# Test LLM endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello, can you help me analyze documents?"}],
    "max_tokens": 50
  }'

# Test integration with file ingestion
# Drop a test file in tyra-ingest/inbox/ and check logs for LLM enhancement
```

#### Step 4: Monitor LLM Usage

```bash
# Check LLM enhancement status
curl http://localhost:8000/v1/ingestion/status

# Monitor processing logs
tail -f logs/tyra-memory.log | grep -i "llm\|vllm\|enhancement"

# Get performance statistics
curl http://localhost:8000/tools/get_memory_stats \
  -d '{"include_performance": true}'
```

### LLM Integration Features

#### Document Context Enhancement
When enabled, your local LLM will:
- **Analyze document content** and generate contextual summaries
- **Create better chunk descriptions** for improved retrieval
- **Extract key concepts** and themes from documents
- **Generate search-friendly metadata** for better discovery

#### Reranking Enhancement
Your LLM can also improve search result ranking by:
- **Semantic relevance scoring** based on query intent
- **Context-aware ranking** considering document relationships
- **Quality assessment** of search results
- **Confidence scoring** for result reliability

#### Configuration Options

**LLM Modes**:
- `rule_based`: Use predefined templates (fastest, no LLM required)
- `vllm`: Use local vLLM server (best quality)
- `disabled`: No context enhancement

**Performance Settings**:
```yaml
vllm_integration:
  timeout: 30              # Request timeout in seconds
  max_tokens: 150          # Maximum tokens for context generation
  temperature: 0.3         # Creativity vs consistency (0.0-1.0)
  batch_size: 5           # Process multiple chunks together
  retry_attempts: 3        # Retry failed requests
  fallback_to_rules: true  # Fall back to rule-based if LLM fails
```

#### Troubleshooting LLM Integration

**LLM server not responding?**
1. Check if LLM server is running: `curl http://localhost:8000/health`
2. Verify endpoint in configuration
3. Check firewall/port settings
4. Review LLM server logs

**Poor context quality?**
1. Adjust temperature (lower = more focused)
2. Increase max_tokens for longer contexts
3. Customize prompts in configuration
4. Try different models

**Performance issues?**
1. Reduce batch_size for faster processing
2. Decrease max_tokens to speed up generation
3. Enable fallback_to_rules for critical operations
4. Monitor GPU/CPU usage on LLM server

---

## 🌐 API Usage

### REST API Overview

The Tyra system provides a comprehensive REST API with 19 modules covering all functionality.

#### Base URL
```
http://localhost:8000
```

#### Authentication (Optional)
```bash
# If API key authentication is enabled
curl -H "X-API-Key: your-api-key" http://localhost:8000/v1/health

# If JWT authentication is enabled
curl -H "Authorization: Bearer your-jwt-token" http://localhost:8000/v1/health
```

### Core API Endpoints

#### Memory Operations (`/v1/memory/`)

**Store Memory**:
```bash
curl -X POST http://localhost:8000/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers morning trading with technical analysis",
    "agent_id": "tyra",
    "metadata": {"category": "trading", "confidence": 95},
    "extract_entities": true
  }'
```

**Search Memories**:
```bash
curl -X POST http://localhost:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "trading preferences",
    "agent_id": "tyra",
    "top_k": 10,
    "min_confidence": 0.7,
    "include_analysis": true
  }'
```

**Get All Memories**:
```bash
curl -X GET "http://localhost:8000/v1/memory/all?agent_id=tyra&limit=50&offset=0"
```

**Delete Memory**:
```bash
curl -X DELETE http://localhost:8000/v1/memory/mem_12345
```

#### Document Ingestion (`/v1/ingestion/`)

**Upload Document**:
```bash
curl -X POST http://localhost:8000/v1/ingestion/document \
  -F "file=@document.pdf" \
  -F "agent_id=tyra" \
  -F "chunking_strategy=semantic" \
  -F "extract_entities=true"
```

**Batch Upload**:
```bash
curl -X POST http://localhost:8000/v1/ingestion/batch \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"file_path": "/path/to/doc1.pdf", "agent_id": "tyra"},
      {"file_path": "/path/to/doc2.docx", "agent_id": "tyra"}
    ],
    "max_concurrent": 5
  }'
```

**Get Ingestion Status**:
```bash
curl -X GET http://localhost:8000/v1/ingestion/status/job_12345
```

#### Analysis & Validation (`/v1/rag/`)

**Analyze Response**:
```bash
curl -X POST http://localhost:8000/v1/rag/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "response": "Based on your history, you prefer swing trading",
    "query": "What is my trading style?",
    "retrieved_memories": [...],
    "detailed_analysis": true
  }'
```

**Rerank Results**:
```bash
curl -X POST http://localhost:8000/v1/rag/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "trading strategies",
    "results": [...],
    "reranker_type": "cross_encoder",
    "top_k": 5
  }'
```

#### Knowledge Graph (`/v1/graph/`)

**Execute Graph Query**:
```bash
curl -X POST http://localhost:8000/v1/graph/query \
  -H "Content-Type: application/json" \
  -d '{
    "cypher_query": "MATCH (p:PERSON)-[r:TRADES]->(s:STOCK) RETURN p, r, s",
    "agent_id": "tyra"
  }'
```

**Get Entity Relationships**:
```bash
curl -X GET "http://localhost:8000/v1/graph/entity/AAPL/relationships?max_depth=2"
```

**Get Entity Timeline**:
```bash
curl -X GET "http://localhost:8000/v1/graph/entity/entity_123/timeline?start_date=2024-01-01&end_date=2024-12-31"
```

#### Analytics (`/v1/analytics/`)

**Get Usage Analytics**:
```bash
curl -X GET "http://localhost:8000/v1/analytics/usage?agent_id=tyra&days=30"
```

**Get Performance Metrics**:
```bash
curl -X GET "http://localhost:8000/v1/analytics/performance?metric_types=latency,throughput&time_window=1h"
```

**Get ROI Analysis**:
```bash
curl -X GET "http://localhost:8000/v1/analytics/roi?agent_id=tyra&include_predictions=true"
```

#### System Health (`/v1/health/`)

**Health Check**:
```bash
curl -X GET http://localhost:8000/v1/health
```

**Detailed Health**:
```bash
curl -X GET http://localhost:8000/v1/health/detailed
```

**Component Status**:
```bash
curl -X GET http://localhost:8000/v1/health/components
```

### WebSocket API

#### Real-time Memory Updates
```javascript
// Connect to memory stream
const ws = new WebSocket('ws://localhost:8000/v1/ws/memory-stream');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Memory update:', update);
};

// Subscribe to specific agent
ws.send(JSON.stringify({
    action: 'subscribe',
    agent_id: 'tyra'
}));
```

#### Progressive Search Results
```javascript
// Connect to search stream
const ws = new WebSocket('ws://localhost:8000/v1/ws/search-stream');

ws.onmessage = function(event) {
    const result = JSON.parse(event.data);
    console.log('Search result:', result);
};

// Start progressive search
ws.send(JSON.stringify({
    action: 'search',
    query: 'trading strategies',
    agent_id: 'tyra'
}));
```

#### Live Analytics Stream
```javascript
// Connect to analytics stream
const ws = new WebSocket('ws://localhost:8000/v1/ws/analytics-stream');

ws.onmessage = function(event) {
    const metrics = JSON.parse(event.data);
    console.log('Live metrics:', metrics);
};
```

### API Response Formats

#### Standard Response Structure
```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_12345",
    "processing_time_ms": 150
  }
}
```

#### Error Response Structure
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameter: agent_id is required",
    "details": { ... }
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_12345"
  }
}
```

#### Memory Response Structure
```json
{
  "success": true,
  "data": {
    "memory_id": "mem_12345",
    "content": "User prefers morning trading...",
    "agent_id": "tyra",
    "confidence_score": 0.95,
    "grounding_score": 0.87,
    "entities": [...],
    "relationships": [...],
    "created_at": "2024-01-01T12:00:00Z"
  }
}
```

### API Client Libraries

#### Python Client Example
```python
import aiohttp
import asyncio

class TyraAPIClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    async def store_memory(self, content, agent_id="tyra", **kwargs):
        async with aiohttp.ClientSession() as session:
            data = {
                "content": content,
                "agent_id": agent_id,
                **kwargs
            }
            async with session.post(
                f"{self.base_url}/v1/memory/store",
                json=data,
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def search_memories(self, query, agent_id="tyra", **kwargs):
        async with aiohttp.ClientSession() as session:
            data = {
                "query": query,
                "agent_id": agent_id,
                **kwargs
            }
            async with session.post(
                f"{self.base_url}/v1/memory/search",
                json=data,
                headers=self.headers
            ) as response:
                return await response.json()

# Usage
async def main():
    client = TyraAPIClient()
    
    # Store a memory
    result = await client.store_memory(
        "User likes technical analysis for trading",
        agent_id="tyra",
        metadata={"category": "trading"}
    )
    print(f"Stored memory: {result['data']['memory_id']}")
    
    # Search memories
    results = await client.search_memories(
        "trading preferences",
        agent_id="tyra",
        top_k=5
    )
    print(f"Found {len(results['data']['memories'])} memories")

asyncio.run(main())
```

#### JavaScript Client Example
```javascript
class TyraAPIClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json',
            ...(apiKey && { 'X-API-Key': apiKey })
        };
    }
    
    async storeMemory(content, agentId = 'tyra', options = {}) {
        const response = await fetch(`${this.baseUrl}/v1/memory/store`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                content,
                agent_id: agentId,
                ...options
            })
        });
        return await response.json();
    }
    
    async searchMemories(query, agentId = 'tyra', options = {}) {
        const response = await fetch(`${this.baseUrl}/v1/memory/search`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                query,
                agent_id: agentId,
                ...options
            })
        });
        return await response.json();
    }
}

// Usage
const client = new TyraAPIClient();

async function example() {
    // Store a memory
    const result = await client.storeMemory(
        'User prefers swing trading strategies',
        'tyra',
        { metadata: { category: 'trading' } }
    );
    console.log('Stored memory:', result.data.memory_id);
    
    // Search memories
    const results = await client.searchMemories(
        'trading strategies',
        'tyra',
        { top_k: 5, min_confidence: 0.7 }
    );
    console.log('Found memories:', results.data.memories.length);
}
```

---

## 🔧 Advanced Features

### Confidence Scoring System

The Tyra system uses a sophisticated 4-level confidence scoring system to ensure reliable memory operations.

#### Confidence Levels

**💪 Rock Solid (95%+)**
- **Use case**: Critical financial decisions, trading operations
- **Characteristics**: High evidence support, multiple source validation
- **Safety**: Required for trading operations (unbypassable)

**🧠 High (80-94%)**
- **Use case**: General recommendations, reliable information
- **Characteristics**: Good evidence support, consistent with known facts

**🤔 Fuzzy (60-79%)**
- **Use case**: Exploratory insights, requires verification
- **Characteristics**: Some uncertainty, additional validation needed

**⚠️ Low (<60%)**
- **Use case**: Uncertain information, high risk
- **Characteristics**: Significant uncertainty, not recommended for decisions

#### Confidence Calculation Factors
1. **Source reliability** - Quality and trustworthiness of information sources
2. **Evidence support** - Amount and quality of supporting evidence
3. **Consistency** - Agreement with existing knowledge base
4. **Recency** - How current the information is
5. **Grounding** - Connection to verifiable facts

### Hallucination Detection

#### Multi-Layer Validation System

**Layer 1: Grounding Analysis**
- Checks if response claims are supported by retrieved memories
- Validates factual consistency
- Identifies unsupported assertions

**Layer 2: Evidence Collection**
- Gathers supporting evidence for each claim
- Scores evidence quality and relevance
- Identifies missing evidence

**Layer 3: Consistency Checking**
- Compares response against known facts
- Detects contradictions with existing knowledge
- Validates logical consistency

**Layer 4: Confidence Scoring**
- Assigns confidence levels based on validation results
- Provides detailed confidence breakdown
- Generates reliability assessment

#### Using Hallucination Detection

```bash
# Analyze any response for hallucinations
curl -X POST http://localhost:8000/tools/analyze_response \
  -d '{
    "response": "Based on market data, AAPL is expected to rise 50% next week",
    "query": "What is the outlook for AAPL stock?",
    "retrieved_memories": [...],
    "detailed_analysis": true
  }'
```

**Response**:
```json
{
  "confidence_level": "low",
  "confidence_score": 25,
  "hallucination_detected": true,
  "grounding_score": 0.2,
  "evidence_support": {
    "total_claims": 2,
    "supported_claims": 0,
    "unsupported_claims": 2
  },
  "analysis": {
    "issues": [
      "Specific price prediction (50% rise) lacks supporting evidence",
      "Timeframe (next week) is too specific without market data"
    ],
    "recommendations": [
      "Request current market analysis",
      "Provide more general outlook instead of specific predictions"
    ]
  }
}
```

### Trading Safety Features

#### Unbypassable 95% Confidence Requirement

For financial operations, the system enforces a mandatory 95% confidence threshold that cannot be bypassed.

**Configuration**:
```yaml
trading_safety:
  enabled: true
  confidence_requirements:
    trading_decisions: 95      # Rock solid only
    financial_advice: 90
    risk_assessment: 85
  
  audit:
    enabled: true
    log_all_operations: true
    require_evidence: true
```

**Example Trading Validation**:
```bash
curl -X POST http://localhost:8000/tools/validate_for_trading \
  -d '{
    "query": "Should I buy AAPL stock?",
    "response": "Based on your risk profile and recent analysis, AAPL shows strong fundamentals",
    "context_memories": [...],
    "require_rock_solid": true
  }'
```

### Memory Synthesis Features

#### Intelligent Deduplication

**Semantic Deduplication**:
- Identifies similar memories using embeddings
- Configurable similarity thresholds
- Multiple merge strategies

```bash
curl -X POST http://localhost:8000/tools/deduplicate_memories \
  -d '{
    "agent_id": "tyra",
    "similarity_threshold": 0.9,
    "auto_merge": false
  }'
```

**Merge Strategies**:
- `keep_newest`: Keep the most recent memory
- `keep_highest_confidence`: Keep the memory with highest confidence
- `merge_content`: Combine content from similar memories
- `create_summary`: Create a new summary memory

#### AI-Powered Summarization

**Summary Types**:
- `extractive`: Extract key sentences from original content
- `abstractive`: Generate new summary using AI
- `hybrid`: Combine extractive and abstractive approaches
- `progressive`: Build hierarchical summaries

```bash
curl -X POST http://localhost:8000/tools/summarize_memories \
  -d '{
    "memory_ids": ["mem_001", "mem_002", "mem_003"],
    "summary_type": "hybrid",
    "max_length": 200
  }'
```

#### Pattern Detection

**Pattern Types**:
- **Topic clusters**: Group memories by semantic similarity
- **Temporal patterns**: Identify time-based trends
- **Entity relationships**: Find connection patterns
- **Behavioral patterns**: Detect user preference patterns

```bash
curl -X POST http://localhost:8000/tools/detect_patterns \
  -d '{
    "agent_id": "tyra",
    "min_cluster_size": 5,
    "include_recommendations": true
  }'
```

### Real-Time Streaming Features

#### WebSocket Event Types

**Memory Events**:
- `memory_created`: New memory stored
- `memory_updated`: Memory modified
- `memory_deleted`: Memory removed
- `agent_activity`: Agent interaction events

**Search Events**:
- `search_started`: Search query initiated
- `search_progress`: Progressive search results
- `search_completed`: Final search results
- `reranking_completed`: Results reranked

**Analytics Events**:
- `performance_update`: Real-time performance metrics
- `threshold_alert`: Performance threshold exceeded
- `system_health`: Component health updates

#### Setting Up Real-Time Streams

```javascript
// Memory stream with filters
const memoryWs = new WebSocket('ws://localhost:8000/v1/ws/memory-stream');

memoryWs.onopen = function() {
    // Subscribe to specific agent events
    memoryWs.send(JSON.stringify({
        action: 'subscribe',
        filters: {
            agent_id: 'tyra',
            event_types: ['memory_created', 'memory_updated']
        }
    }));
};

memoryWs.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Memory event:', data);
    
    switch(data.event_type) {
        case 'memory_created':
            updateMemoryDisplay(data.memory);
            break;
        case 'memory_updated':
            refreshMemoryView(data.memory_id);
            break;
    }
};
```

### Adaptive Learning System

#### Online Learning Features

**Learning Components**:
- **Query pattern analysis**: Learn from search patterns
- **Response quality feedback**: Improve based on user interactions
- **Performance optimization**: Auto-tune system parameters
- **Preference learning**: Adapt to user preferences

#### A/B Testing Framework

**Experiment Types**:
- **Search algorithms**: Compare different search strategies
- **Reranking methods**: Test different reranking approaches
- **Confidence thresholds**: Optimize confidence scoring
- **UI/UX variations**: Test different interface approaches

```bash
curl -X POST http://localhost:8000/v1/learning/experiment \
  -d '{
    "experiment_name": "reranking_comparison",
    "variants": [
      {"name": "cross_encoder", "weight": 0.5},
      {"name": "vllm_rerank", "weight": 0.5}
    ],
    "success_metrics": ["relevance_score", "user_satisfaction"],
    "duration_days": 7
  }'
```

#### Self-Optimization

**Automatic Optimization**:
- **Parameter tuning**: Optimize search and processing parameters
- **Model selection**: Choose best performing models
- **Cache strategies**: Optimize cache hit rates
- **Resource allocation**: Balance performance vs. resources

```bash
curl -X POST http://localhost:8000/v1/learning/optimize \
  -d '{
    "components": ["search", "cache", "reranking"],
    "optimization_goal": "latency",
    "constraints": {"max_resource_usage": 80}
  }'
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Model Installation Issues

**Problem**: Models not found or incomplete downloads
```
ERROR: Model not found: ./models/embeddings/e5-large-v2
```

**Solution**:
```bash
# Verify model directory structure
ls -la ./models/embeddings/e5-large-v2/

# Re-download the model
huggingface-cli download intfloat/e5-large-v2 \
  --local-dir ./models/embeddings/e5-large-v2 \
  --local-dir-use-symlinks False

# Test model loading
python scripts/test_model_pipeline.py
```

#### 2. Database Connection Issues

**Problem**: PostgreSQL connection failed
```
ERROR: Could not connect to PostgreSQL: connection refused
```

**Solution**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Start PostgreSQL if not running
sudo systemctl start postgresql

# Test connection manually
psql -h localhost -p 5432 -U tyra -d tyra_memory

# Check environment variables
echo $POSTGRES_HOST $POSTGRES_PORT $POSTGRES_USER
```

**Problem**: pgvector extension not available
```
ERROR: extension "vector" is not available
```

**Solution**:
```bash
# Install pgvector extension
sudo apt install postgresql-14-pgvector

# Enable extension in database
psql -U tyra -d tyra_memory -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify installation
psql -U tyra -d tyra_memory -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

#### 3. Redis Connection Issues

**Problem**: Redis server not available
```
ERROR: Redis connection failed: Connection refused
```

**Solution**:
```bash
# Check Redis status
sudo systemctl status redis

# Start Redis if not running
sudo systemctl start redis

# Test connection
redis-cli ping

# Check Redis configuration
redis-cli CONFIG GET "*"
```

#### 4. Neo4j Connection Issues

**Problem**: Neo4j authentication failed
```
ERROR: Neo4j authentication failed
```

**Solution**:
```bash
# Reset Neo4j password
sudo neo4j-admin set-initial-password neo4j

# Or set new password
cypher-shell -u neo4j -p neo4j
:server change-password

# Update environment variables
export NEO4J_PASSWORD=your_new_password
```

#### 5. File Ingestion Issues

**Problem**: Files not being processed from inbox
```
WARNING: File watcher service is not processing files
```

**Solution**:
```bash
# Check file watcher status
curl http://localhost:8000/v1/file-watcher/status

# Check folder permissions
ls -la tyra-ingest/
chmod 755 tyra-ingest/inbox tyra-ingest/processed tyra-ingest/failed

# Restart file watcher
curl -X POST http://localhost:8000/v1/file-watcher/restart

# Check logs for errors
tail -f logs/tyra-memory.log | grep -i "file_watcher"
```

**Problem**: Large files failing to process
```
ERROR: File too large: exceeds maximum size limit
```

**Solution**:
```bash
# Increase file size limit in configuration
export INGESTION_MAX_FILE_SIZE=209715200  # 200MB

# Or modify config/ingestion.yaml
file_processing:
  max_file_size: 209715200

# Restart the service
```

#### 6. Performance Issues

**Problem**: Slow search responses
```
WARNING: Search latency exceeding 2000ms
```

**Solution**:
```bash
# Check system resources
htop
free -h
df -h

# Optimize database indexes
psql -U tyra -d tyra_memory << EOF
REINDEX INDEX CONCURRENTLY memory_embedding_idx;
ANALYZE memories;
EOF

# Clear and rebuild cache
redis-cli FLUSHALL

# Tune PostgreSQL settings
# Edit postgresql.conf:
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 256MB
```

**Problem**: High memory usage
```
WARNING: Memory usage exceeding 8GB
```

**Solution**:
```bash
# Reduce cache sizes
export CACHE_L1_MAX_SIZE=50MB
export CACHE_L2_MAX_SIZE=500MB

# Reduce batch sizes
export MEMORY_BATCH_SIZE=25
export EMBEDDINGS_BATCH_SIZE=16

# Enable memory cleanup
curl -X POST http://localhost:8000/tools/cleanup_memories \
  -d '{"agent_id": "tyra", "older_than_days": 90, "dry_run": false}'
```

#### 7. API Issues

**Problem**: API returning 500 errors
```
ERROR: Internal Server Error
```

**Solution**:
```bash
# Check API logs
tail -f logs/tyra-memory.log | grep -i "error\|exception"

# Test API health
curl http://localhost:8000/health

# Restart API server
pkill -f "uvicorn"
python -m uvicorn src.api.app:app --reload
```

**Problem**: Rate limiting issues
```
ERROR: Rate limit exceeded
```

**Solution**:
```bash
# Increase rate limits
export API_RATE_LIMIT=2000
export API_RATE_LIMIT_BURST=100

# Or disable rate limiting temporarily
export RATE_LIMIT_ENABLED=false

# Check current limits
curl http://localhost:8000/v1/health/rate-limits
```

#### 8. MCP Connection Issues

**Problem**: Claude can't connect to MCP server
```
ERROR: MCP server not responding
```

**Solution**:
```bash
# Check MCP server status
ps aux | grep "main.py"

# Test MCP server directly
python main.py --test-mode

# Check Claude MCP configuration
cat ~/.config/claude-desktop/mcp_settings.json

# Verify paths in MCP config
{
  "mcpServers": {
    "tyra-memory": {
      "command": "python",
      "args": ["/full/path/to/tyra-mcp-memory-server/main.py"],
      "env": {
        "TYRA_ENV": "production"
      }
    }
  }
}
```

#### 9. LLM Integration Issues

**Problem**: vLLM server not responding
```
ERROR: vLLM endpoint timeout
```

**Solution**:
```bash
# Check vLLM server status
curl http://localhost:8000/v1/models

# Restart vLLM server
pkill -f "vllm"
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000

# Increase timeout in configuration
export VLLM_TIMEOUT=60

# Test with different model
export VLLM_MODEL=microsoft/DialoGPT-medium
```

### Diagnostic Commands

#### Health Check Commands
```bash
# Comprehensive health check
curl http://localhost:8000/tools/health_check -d '{"detailed": true}'

# Component-specific checks
curl http://localhost:8000/v1/health/postgres
curl http://localhost:8000/v1/health/redis
curl http://localhost:8000/v1/health/neo4j
curl http://localhost:8000/v1/health/embeddings

# Performance metrics
curl http://localhost:8000/tools/get_memory_stats \
  -d '{"include_performance": true, "include_recommendations": true}'
```

#### Log Analysis
```bash
# View recent errors
tail -100 logs/tyra-memory.log | grep -i error

# Monitor real-time activity
tail -f logs/tyra-memory.log

# Search for specific issues
grep -i "timeout\|connection\|failed" logs/tyra-memory.log

# Performance monitoring
grep -i "latency\|slow\|performance" logs/tyra-memory.log
```

#### Database Diagnostics
```bash
# PostgreSQL connection test
psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1;"

# Check database size
psql -U tyra -d tyra_memory -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Check index usage
psql -U tyra -d tyra_memory -c "
SELECT 
    indexrelname as index_name,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;
"

# Redis memory usage
redis-cli INFO memory

# Neo4j status
cypher-shell -u neo4j -p $NEO4J_PASSWORD "SHOW DATABASES;"
```

### Performance Monitoring

#### Key Metrics to Monitor

**System Metrics**:
- CPU usage (<80% recommended)
- Memory usage (<90% recommended)
- Disk I/O (monitor for bottlenecks)
- Network latency

**Application Metrics**:
- API response time (<200ms p95)
- Search latency (<100ms p95)
- Memory storage time (<50ms p95)
- Cache hit rate (>85% target)

**Database Metrics**:
- PostgreSQL connection count
- Query execution time
- Redis memory usage
- Neo4j query performance

#### Monitoring Commands
```bash
# System resources
htop
iotop
nethogs

# Database performance
pg_stat_activity # PostgreSQL
redis-cli --stat # Redis
cypher-shell "CALL db.stats.retrieve('GRAPH COUNTS')" # Neo4j

# Application metrics
curl http://localhost:8000/metrics  # Prometheus metrics
```

### Recovery Procedures

#### Database Recovery
```bash
# PostgreSQL backup and restore
pg_dump -U tyra tyra_memory > backup.sql
psql -U tyra -d tyra_memory_new < backup.sql

# Redis backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb backup_dump.rdb

# Neo4j backup
neo4j-admin dump --database=neo4j --to=neo4j_backup.dump
```

#### Configuration Recovery
```bash
# Reset to default configuration
cp config/config.yaml.example config/config.yaml

# Backup current configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/

# Apply safe configuration
export TYRA_ENV=development
export TYRA_DEBUG=true
export CACHE_SIZES="small"
```

---

## ⚡ Performance Tuning

### System Optimization

#### Hardware Recommendations

**Minimum Requirements**:
- CPU: 4 cores (8 threads recommended)
- RAM: 8GB (16GB recommended)
- Storage: 100GB SSD
- Network: 1Gbps

**Optimal Configuration**:
- CPU: 8+ cores with AVX2 support
- RAM: 32GB+ (for large models and caching)
- GPU: NVIDIA GPU with 8GB+ VRAM (optional)
- Storage: NVMe SSD with 1000+ IOPS

#### Operating System Tuning

**Linux Optimization**:
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize network settings
echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.conf
sysctl -p

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Database Optimization

#### PostgreSQL Tuning

**Configuration** (`postgresql.conf`):
```ini
# Memory settings
shared_buffers = 4GB                    # 25% of RAM
effective_cache_size = 12GB             # 75% of RAM
work_mem = 256MB                        # For sorting/hashing
maintenance_work_mem = 1GB              # For VACUUM, CREATE INDEX

# Connection settings
max_connections = 200
max_prepared_transactions = 100

# WAL settings
wal_buffers = 64MB
checkpoint_completion_target = 0.9
wal_compression = on

# Query planning
random_page_cost = 1.1                  # For SSD storage
effective_io_concurrency = 200          # For SSD storage

# Parallel query settings
max_parallel_workers = 8
max_parallel_workers_per_gather = 4
max_parallel_maintenance_workers = 4

# Vector-specific settings
shared_preload_libraries = 'vector'
```

**Index Optimization**:
```sql
-- Create optimized vector index
CREATE INDEX CONCURRENTLY memory_embedding_hnsw_idx 
ON memories USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Optimize for hybrid search
CREATE INDEX CONCURRENTLY memory_content_gin_idx 
ON memories USING gin(to_tsvector('english', content));

-- Index for agent filtering
CREATE INDEX CONCURRENTLY memory_agent_btree_idx 
ON memories (agent_id, created_at DESC);

-- Composite index for common queries
CREATE INDEX CONCURRENTLY memory_composite_idx 
ON memories (agent_id, created_at DESC, confidence_score DESC);
```

**Maintenance Tasks**:
```sql
-- Regular maintenance (run weekly)
VACUUM ANALYZE memories;
REINDEX INDEX CONCURRENTLY memory_embedding_hnsw_idx;

-- Update statistics (run daily)
ANALYZE memories;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch 
FROM pg_stat_user_indexes 
WHERE schemaname = 'public' 
ORDER BY idx_tup_read DESC;
```

#### Redis Optimization

**Configuration** (`redis.conf`):
```ini
# Memory settings
maxmemory 4gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Persistence settings
save 900 1
save 300 10
save 60 10000

# AOF settings
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Network settings
tcp-keepalive 300
timeout 300

# Performance settings
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
```

**Cache Strategy Optimization**:
```bash
# Monitor cache performance
redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"

# Optimize cache keys
redis-cli --bigkeys

# Monitor memory usage
redis-cli INFO memory
```

#### Neo4j Optimization

**Configuration** (`neo4j.conf`):
```ini
# Memory settings
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=4g
dbms.memory.pagecache.size=2g

# Performance settings
dbms.tx_state.memory_allocation=ON_HEAP
dbms.query.cache_size=1000
dbms.query.cache_hit_threshold=5

# Security settings (for production)
dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=algo.*,apoc.*
```

### Application Optimization

#### Memory Management

**Configuration Tuning**:
```yaml
# config/config.yaml
performance:
  # Batch processing
  memory_batch_size: 50
  embedding_batch_size: 32
  concurrent_operations: 10
  
  # Cache optimization
  cache_sizes:
    l1_memory: "100MB"      # In-memory cache
    l2_redis: "1GB"         # Redis cache
    l3_postgres: "5GB"      # PostgreSQL cache
    
  # Connection pooling
  postgres_pool_size: 20
  redis_pool_size: 50
  neo4j_pool_size: 10
  
  # Request handling
  max_concurrent_requests: 100
  request_timeout: 30
  
  # GPU optimization (if available)
  gpu_enabled: true
  gpu_memory_fraction: 0.8
```

**Environment Variables**:
```bash
# Memory optimization
export MEMORY_BATCH_SIZE=50
export EMBEDDINGS_BATCH_SIZE=32
export CONCURRENT_OPERATIONS=10

# Cache optimization
export CACHE_L1_MAX_SIZE=100MB
export CACHE_L2_MAX_SIZE=1GB
export CACHE_TTL_EMBEDDINGS=86400

# Connection pools
export POSTGRES_POOL_SIZE=20
export REDIS_POOL_SIZE=50
export NEO4J_POOL_SIZE=10
```

#### Embedding Optimization

**Model Selection**:
```yaml
# config/providers.yaml
embeddings:
  primary: "huggingface"
  
  providers:
    huggingface:
      model_name: "intfloat/e5-large-v2"
      device: "cuda"  # Use GPU if available
      batch_size: 64  # Optimize for throughput
      max_length: 512
      precision: "float16"  # Reduce memory usage
```

**GPU Optimization**:
```bash
# Check GPU availability
nvidia-smi

# Set GPU device
export CUDA_VISIBLE_DEVICES=0
export EMBEDDINGS_DEVICE=cuda

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Performance Monitoring

#### Key Performance Indicators (KPIs)

**Latency Targets**:
- Memory storage: <100ms p95
- Vector search: <50ms p95
- Hybrid search: <150ms p95
- Document ingestion: <2s per page
- Graph queries: <30ms simple traversals
- Hallucination detection: <200ms

**Throughput Targets**:
- Concurrent users: 50-100
- Memory operations: 1000+/minute
- Document processing: 10-20/minute
- API requests: 1000+/minute

#### Monitoring Setup

**Prometheus Metrics**:
```yaml
# config/observability.yaml
metrics:
  enabled: true
  exporter: "prometheus"
  endpoint: "http://localhost:9090"
  
  custom_metrics:
    - name: "memory_operation_duration"
      type: "histogram"
      buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
      
    - name: "search_accuracy_score"
      type: "gauge"
      
    - name: "cache_hit_rate"
      type: "gauge"
```

**Grafana Dashboard**:
```json
{
  "dashboard": {
    "title": "Tyra Memory Server Performance",
    "panels": [
      {
        "title": "Request Latency (P95)",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(memory_operation_duration_bucket[5m]))"
        }]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [{
          "expr": "process_resident_memory_bytes / 1024 / 1024 / 1024"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [{
          "expr": "cache_hit_rate"
        }]
      }
    ]
  }
}
```

#### Performance Testing

**Load Testing Script**:
```python
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def performance_test():
    """Run performance test on Tyra memory server."""
    
    base_url = "http://localhost:8000"
    concurrent_requests = 50
    test_duration = 60  # seconds
    
    async def store_memory_test(session, test_id):
        data = {
            "content": f"Test memory content {test_id}",
            "agent_id": "performance_test",
            "metadata": {"test_id": test_id}
        }
        start_time = time.time()
        async with session.post(f"{base_url}/v1/memory/store", json=data) as response:
            await response.json()
            return time.time() - start_time
    
    async def search_memory_test(session, test_id):
        data = {
            "query": f"test content {test_id % 10}",
            "agent_id": "performance_test",
            "top_k": 5
        }
        start_time = time.time()
        async with session.post(f"{base_url}/v1/memory/search", json=data) as response:
            await response.json()
            return time.time() - start_time
    
    # Run concurrent tests
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Mix of store and search operations
        for i in range(concurrent_requests):
            if i % 2 == 0:
                tasks.append(store_memory_test(session, i))
            else:
                tasks.append(search_memory_test(session, i))
        
        start_time = time.time()
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        throughput = len(tasks) / total_time
        
        print(f"Performance Test Results:")
        print(f"Total requests: {len(tasks)}")
        print(f"Concurrent requests: {concurrent_requests}")
        print(f"Average latency: {avg_latency:.3f}s")
        print(f"P95 latency: {p95_latency:.3f}s")
        print(f"Throughput: {throughput:.2f} req/s")

if __name__ == "__main__":
    asyncio.run(performance_test())
```

**Benchmark Command**:
```bash
# Run performance tests
python scripts/performance_test.py

# Apache Bench testing
ab -n 1000 -c 10 -T application/json -p test_data.json \
   http://localhost:8000/v1/memory/search

# Custom load testing
wrk -t12 -c400 -d30s --script=load_test.lua http://localhost:8000/
```

### Optimization Strategies

#### Cache Optimization

**Multi-Layer Caching Strategy**:
```python
# L1: In-memory cache (fastest, smallest)
# L2: Redis cache (fast, medium size)
# L3: PostgreSQL materialized views (slower, largest)

# Optimize cache hit rates
cache_strategy = {
    "embeddings": {
        "l1_ttl": 300,    # 5 minutes
        "l2_ttl": 3600,   # 1 hour
        "l3_ttl": 86400   # 24 hours
    },
    "search_results": {
        "l1_ttl": 60,     # 1 minute
        "l2_ttl": 600,    # 10 minutes
        "l3_ttl": 3600    # 1 hour
    }
}
```

#### Query Optimization

**Vector Search Optimization**:
```sql
-- Optimize HNSW parameters for your data
CREATE INDEX memory_embedding_optimized_idx 
ON memories USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,                    -- Number of connections per element
    ef_construction = 128      -- Size of candidate list during construction
);

-- Query optimization
SET hnsw.ef_search = 200;    -- Increase for better recall
```

**Graph Query Optimization**:
```cypher
// Use PROFILE to analyze query performance
PROFILE MATCH (p:Person)-[:TRADES]->(s:Stock)
WHERE s.symbol = $symbol
RETURN p, s;

// Add indexes for frequently queried properties
CREATE INDEX ON :Stock(symbol);
CREATE INDEX ON :Person(agent_id);
```

#### Scaling Strategies

**Horizontal Scaling**:
```bash
# Load balancer configuration (NGINX)
upstream tyra_backend {
    least_conn;
    server 127.0.0.1:8001 weight=1;
    server 127.0.0.1:8002 weight=1;
    server 127.0.0.1:8003 weight=1;
}

# Database read replicas
# Configure PostgreSQL streaming replication
# Route read queries to replicas
```

**Vertical Scaling**:
```bash
# Scale up resources
# - Increase RAM for larger caches
# - Add GPU for faster embedding generation
# - Use faster storage (NVMe SSDs)
# - Increase CPU cores for parallel processing
```

This completes the comprehensive Tyra MCP Memory Server User Guide. The guide covers all aspects of setup, configuration, usage, and optimization for the system's advanced features and capabilities.