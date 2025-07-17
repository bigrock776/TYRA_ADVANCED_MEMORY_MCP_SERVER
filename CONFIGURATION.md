# ⚙️ Configuration Guide - Tyra MCP Memory Server

> **🚀 Enhanced Configuration**: This guide covers both current production configuration and upcoming enhancements from [ADVANCED_ENHANCEMENTS.md](ADVANCED_ENHANCEMENTS.md). All enhanced features maintain backward compatibility and include auto-migration.

## ⚠️ **PREREQUISITES - MANUAL MODEL INSTALLATION REQUIRED**

**CRITICAL**: Before configuring the system, you must manually download required models:

```bash
# Install HuggingFace CLI
pip install huggingface-hub
git lfs install

# Download models to local directories
mkdir -p ./models/embeddings ./models/cross-encoders

huggingface-cli download intfloat/e5-large-v2 \
  --local-dir ./models/embeddings/e5-large-v2 \
  --local-dir-use-symlinks False

huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 \
  --local-dir ./models/embeddings/all-MiniLM-L12-v2 \
  --local-dir-use-symlinks False

huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2 \
  --local-dir-use-symlinks False
```

## 📋 Overview

Tyra MCP Memory Server uses a sophisticated layered configuration system that provides flexibility, maintainability, and **hot-swappable component configuration**:

### **Current Configuration Layers**
1. **YAML Files**: Primary configuration in `config/` directory
2. **Environment Variables**: Runtime overrides via `.env` file
3. **Command Line Arguments**: Temporary overrides for testing
4. **Runtime Configuration**: Dynamic updates via API endpoints

### **Enhanced Configuration Features (Coming Soon)**
5. **🔄 Hot Reload**: Component configuration changes without restart
6. **🤖 Auto-Optimization**: ML-driven configuration tuning
7. **📊 Performance-Based Adaptation**: Automatic parameter adjustment
8. **🛡️ Validation Schemas**: Pydantic-based configuration validation
9. **🎯 Component-Specific Settings**: Per-provider optimization settings
10. **📈 A/B Testing Configuration**: Experimental feature toggles

## 🎯 Quick Configuration

> **Enhancement Preview**: Future versions will include automated configuration optimization and intelligent defaults based on your hardware and usage patterns.

### Essential Setup

1. **Copy Environment Template**
   ```bash
   cp .env.example .env
   ```

2. **Edit Essential Variables**
   ```bash
   nano .env
   ```

3. **Required Configuration**
   ```env
   # Database Connections
   DATABASE_URL=postgresql://tyra:password@localhost:5432/tyra_memory
   REDIS_URL=redis://localhost:6379/0
   NEO4J_URL=bolt://localhost:7687

   # Embedding Models - LOCAL PATHS REQUIRED
   EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
   EMBEDDINGS_PRIMARY_PATH=./models/embeddings/e5-large-v2
   EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
   EMBEDDINGS_FALLBACK_PATH=./models/embeddings/all-MiniLM-L12-v2
   EMBEDDINGS_USE_LOCAL_FILES=true
   EMBEDDINGS_DEVICE=auto

   # Application Settings
   ENVIRONMENT=development
   LOG_LEVEL=INFO
   
   # Enhanced Features (Phase 1 - Coming Soon)
   PYDANTIC_AI_ENABLED=true
   ANTI_HALLUCINATION_ENABLED=true
   MEMORY_SYNTHESIS_ENABLED=false  # Will be enabled in Phase 1
   REAL_TIME_STREAMS_ENABLED=false # Will be enabled in Phase 1
   
   # Self-Learning Features (Phase 2)
   SELF_OPTIMIZATION_ENABLED=false
   PREDICTIVE_PRELOADING_ENABLED=false
   AUTO_SCALING_ENABLED=false
   ```

## 🗂️ Configuration Files

### Main Configuration (`config/config.yaml`)

```yaml
# Application Settings
app:
  name: "Tyra MCP Memory Server"
  version: "1.0.0"
  environment: ${ENVIRONMENT:-development}
  debug: ${DEBUG:-false}
  log_level: ${LOG_LEVEL:-INFO}

# Server Configuration
server:
  mcp:
    host: ${MCP_HOST:-localhost}
    port: ${MCP_PORT:-3000}
    transport: ${MCP_TRANSPORT:-stdio}
    timeout: ${MCP_TIMEOUT:-30}
  
  fastapi:
    host: ${API_HOST:-0.0.0.0}
    port: ${API_PORT:-8000}
    workers: ${API_WORKERS:-4}
    reload: ${API_RELOAD:-false}
    enable_docs: ${API_ENABLE_DOCS:-true}
    cors_origins: ${API_CORS_ORIGINS:-["*"]}

# Database Configuration
database:
  postgresql:
    url: ${DATABASE_URL:-postgresql://tyra:tyra123@localhost:5432/tyra_memory}
    pool_size: ${POSTGRES_POOL_SIZE:-20}
    max_overflow: ${POSTGRES_MAX_OVERFLOW:-10}
    pool_timeout: ${POSTGRES_POOL_TIMEOUT:-30}
    pool_recycle: ${POSTGRES_POOL_RECYCLE:-3600}
    ssl_mode: ${POSTGRES_SSL_MODE:-prefer}
    
  redis:
    url: ${REDIS_URL:-redis://localhost:6379/0}
    max_connections: ${REDIS_MAX_CONNECTIONS:-50}
    connection_timeout: ${REDIS_CONNECTION_TIMEOUT:-5}
    socket_timeout: ${REDIS_SOCKET_TIMEOUT:-5}
    retry_on_timeout: true
    
  neo4j:
    url: ${NEO4J_URL:-bolt://localhost:7687}
    username: ${NEO4J_USERNAME:-}
    password: ${NEO4J_PASSWORD:-}
    connection_timeout: ${NEO4J_CONNECTION_TIMEOUT:-30}
    pool_size: ${NEO4J_POOL_SIZE:-10}

# Memory Configuration
memory:
  backend: postgres
  vector_dimensions: 1024
  similarity_threshold: 0.7
  max_results: 100
  chunk_size: 1000
  chunk_overlap: 200
  
# Security Settings
security:
  secret_key: ${SECRET_KEY:-your-secret-key-here}
  api_key: ${API_KEY:-your-api-key-here}
  jwt_secret: ${JWT_SECRET:-your-jwt-secret-here}
  cors_enabled: ${CORS_ENABLED:-true}
  rate_limiting:
    enabled: ${RATE_LIMIT_ENABLED:-true}
    requests_per_minute: ${RATE_LIMIT_RPM:-100}
    burst: ${RATE_LIMIT_BURST:-20}
```

### Provider Configuration (`config/providers.yaml`)

```yaml
# Embedding Providers
embeddings:
  primary:
    provider: huggingface
    model: ${EMBEDDINGS_PRIMARY_MODEL:-intfloat/e5-large-v2}
    device: ${EMBEDDINGS_DEVICE:-auto}
    batch_size: ${EMBEDDINGS_BATCH_SIZE:-32}
    max_length: ${EMBEDDINGS_MAX_LENGTH:-512}
    normalize: true
    
  fallback:
    provider: huggingface_fallback
    model: ${EMBEDDINGS_FALLBACK_MODEL:-sentence-transformers/all-MiniLM-L12-v2}
    device: cpu
    batch_size: ${EMBEDDINGS_FALLBACK_BATCH_SIZE:-16}
    max_length: ${EMBEDDINGS_FALLBACK_MAX_LENGTH:-256}
    normalize: true

# Vector Store Providers
vector_stores:
  default: pgvector
  pgvector:
    provider: pgvector
    connection_string: ${DATABASE_URL}
    table_name: ${VECTOR_TABLE_NAME:-memory_vectors}
    index_type: ${VECTOR_INDEX_TYPE:-hnsw}
    index_params:
      m: ${HNSW_M:-16}
      ef_construction: ${HNSW_EF_CONSTRUCTION:-64}

# Graph Engine Providers
graph_engines:
  default: neo4j
  neo4j:
    provider: neo4j
    connection_string: ${NEO4J_URL}
    timeout: ${NEO4J_TIMEOUT:-30}
    pool_size: ${NEO4J_POOL_SIZE:-10}

# Reranker Providers
rerankers:
  default: cross_encoder
  cross_encoder:
    provider: cross_encoder
    model: ${RERANKER_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}
    device: ${RERANKER_DEVICE:-auto}
    batch_size: ${RERANKER_BATCH_SIZE:-8}
    top_k: ${RERANKER_TOP_K:-10}
    
  vllm:
    provider: vllm
    model: ${VLLM_MODEL:-microsoft/DialoGPT-medium}
    endpoint: ${VLLM_ENDPOINT:-http://localhost:8001/v1}
    api_key: ${VLLM_API_KEY:-}
    timeout: ${VLLM_TIMEOUT:-30}
```

### Model Configuration (`config/models.yaml`)

```yaml
# Embedding Models
embedding_models:
  intfloat/e5-large-v2:
    dimensions: 1024
    max_length: 512
    device_preference: ["cuda", "cpu"]
    memory_usage: "high"
    accuracy: "high"
    
  sentence-transformers/all-MiniLM-L12-v2:
    dimensions: 384
    max_length: 256
    device_preference: ["cpu"]
    memory_usage: "low"
    accuracy: "medium"

# Reranking Models
reranking_models:
  cross-encoder/ms-marco-MiniLM-L-6-v2:
    type: "cross_encoder"
    device_preference: ["cuda", "cpu"]
    memory_usage: "medium"
    accuracy: "high"
    
  cross-encoder/ms-marco-TinyBERT-L-2-v2:
    type: "cross_encoder"
    device_preference: ["cpu"]
    memory_usage: "low"
    accuracy: "medium"

# Model Download Settings
model_cache:
  directory: ${MODEL_CACHE_DIR:-./data/models}
  max_size_gb: ${MODEL_CACHE_MAX_SIZE:-50}
  auto_download: ${MODEL_AUTO_DOWNLOAD:-true}
  timeout: ${MODEL_DOWNLOAD_TIMEOUT:-300}
```

### RAG Configuration (`config/rag.yaml`)

```yaml
# Retrieval Settings
retrieval:
  top_k: ${RAG_TOP_K:-20}
  min_confidence: ${RAG_MIN_CONFIDENCE:-0.0}
  hybrid_weight: ${RAG_HYBRID_WEIGHT:-0.7}
  diversity_penalty: ${RAG_DIVERSITY_PENALTY:-0.3}
  max_context_length: ${RAG_MAX_CONTEXT_LENGTH:-4000}

# Reranking Settings
reranking:
  enabled: ${RAG_RERANKING_ENABLED:-true}
  provider: ${RAG_RERANKING_PROVIDER:-cross_encoder}
  top_k: ${RAG_RERANKING_TOP_K:-10}
  score_threshold: ${RAG_RERANKING_THRESHOLD:-0.5}
  batch_size: ${RAG_RERANKING_BATCH_SIZE:-8}

# Hallucination Detection
hallucination:
  enabled: ${RAG_HALLUCINATION_ENABLED:-true}
  threshold: ${RAG_HALLUCINATION_THRESHOLD:-75}
  confidence_levels:
    rock_solid: 95
    high: 80
    fuzzy: 60
    low: 0
  grounding_check: ${RAG_GROUNDING_CHECK:-true}
  evidence_collection: ${RAG_EVIDENCE_COLLECTION:-true}

# Response Generation
response:
  max_tokens: ${RAG_MAX_TOKENS:-2000}
  temperature: ${RAG_TEMPERATURE:-0.1}
  include_sources: ${RAG_INCLUDE_SOURCES:-true}
  include_confidence: ${RAG_INCLUDE_CONFIDENCE:-true}
```

### Caching Configuration (`config/cache.yaml`)

```yaml
# Cache Settings
cache:
  enabled: ${CACHE_ENABLED:-true}
  compression: ${CACHE_COMPRESSION:-true}
  compression_threshold: ${CACHE_COMPRESSION_THRESHOLD:-1024}
  default_ttl: ${CACHE_DEFAULT_TTL:-3600}

# Cache Layers
layers:
  l1:
    type: "memory"
    max_size: ${CACHE_L1_MAX_SIZE:-1000}
    ttl: ${CACHE_L1_TTL:-300}
    
  l2:
    type: "redis"
    max_size: ${CACHE_L2_MAX_SIZE:-10000}
    ttl: ${CACHE_L2_TTL:-3600}

# Cache TTL Settings (seconds)
ttl:
  embeddings: ${CACHE_TTL_EMBEDDINGS:-86400}      # 24 hours
  search: ${CACHE_TTL_SEARCH:-3600}               # 1 hour
  rerank: ${CACHE_TTL_RERANK:-1800}               # 30 minutes
  hallucination: ${CACHE_TTL_HALLUCINATION:-900}  # 15 minutes
  graph: ${CACHE_TTL_GRAPH:-7200}                 # 2 hours
  health: ${CACHE_TTL_HEALTH:-60}                 # 1 minute

# Cache Warming
warming:
  enabled: ${CACHE_WARMING_ENABLED:-false}
  strategies: ["popular", "recent", "scheduled"]
  schedule: "0 2 * * *"  # Daily at 2 AM
```

### Document Ingestion Configuration (`config/ingestion.yaml`)

```yaml
# Document Ingestion System
ingestion:
  # File Processing Settings
  file_processing:
    max_file_size: ${INGESTION_MAX_FILE_SIZE:-104857600}  # 100MB
    max_batch_size: ${INGESTION_MAX_BATCH_SIZE:-100}
    concurrent_limit: ${INGESTION_CONCURRENT_LIMIT:-20}
    timeout_seconds: ${INGESTION_TIMEOUT:-300}
    temp_directory: ${INGESTION_TEMP_DIR:-/tmp/tyra_ingestion}
    
  # Supported File Types
  supported_formats:
    pdf:
      enabled: ${INGESTION_PDF_ENABLED:-true}
      max_size: "50MB"
      loader: "PyMuPDF"
      features: ["text_extraction", "metadata_extraction"]
      
    docx:
      enabled: ${INGESTION_DOCX_ENABLED:-true}
      max_size: "25MB"
      loader: "python-docx"
      features: ["paragraph_detection", "table_extraction"]
      
    pptx:
      enabled: ${INGESTION_PPTX_ENABLED:-true}
      max_size: "25MB"
      loader: "python-pptx"
      features: ["slide_extraction", "speaker_notes"]
      
    txt:
      enabled: ${INGESTION_TXT_ENABLED:-true}
      max_size: "10MB"
      encoding_detection: true
      
    markdown:
      enabled: ${INGESTION_MD_ENABLED:-true}
      max_size: "10MB"
      features: ["header_detection", "structure_preservation"]
      
    html:
      enabled: ${INGESTION_HTML_ENABLED:-true}
      max_size: "10MB"
      converter: "html2text"
      
    json:
      enabled: ${INGESTION_JSON_ENABLED:-true}
      max_size: "50MB"
      features: ["nested_object_handling", "array_processing"]
      
    csv:
      enabled: ${INGESTION_CSV_ENABLED:-true}
      max_size: "100MB"
      features: ["header_detection", "streaming_processing"]
      
    epub:
      enabled: ${INGESTION_EPUB_ENABLED:-true}
      max_size: "25MB"
      features: ["chapter_extraction", "metadata_extraction"]

  # Chunking Strategies
  chunking:
    default_strategy: ${INGESTION_DEFAULT_CHUNKING:-auto}
    default_chunk_size: ${INGESTION_DEFAULT_CHUNK_SIZE:-512}
    default_overlap: ${INGESTION_DEFAULT_OVERLAP:-50}
    
    strategies:
      auto:
        enabled: true
        file_type_mapping:
          pdf: "semantic"
          docx: "paragraph"
          pptx: "slide"
          txt: "paragraph"
          md: "paragraph"
          html: "paragraph"
          json: "object"
          csv: "row"
          epub: "semantic"
          
      paragraph:
        enabled: true
        min_chunk_size: 100
        max_chunk_size: 2000
        overlap_ratio: 0.1
        
      semantic:
        enabled: true
        similarity_threshold: 0.7
        min_chunk_size: 200
        max_chunk_size: 1500
        
      slide:
        enabled: true
        group_slides: true
        include_speaker_notes: true
        
      line:
        enabled: true
        lines_per_chunk: 10
        preserve_structure: true
        
      token:
        enabled: true
        tokens_per_chunk: 400
        tokenizer: "cl100k_base"

  # LLM Context Enhancement
  llm_enhancement:
    enabled: ${INGESTION_LLM_ENHANCEMENT:-true}
    default_mode: ${INGESTION_LLM_MODE:-rule_based}  # rule_based, vllm, disabled
    
    rule_based:
      enabled: true
      templates:
        default: "This content is from {file_name} ({file_type}): {description}"
        pdf: "From PDF document '{file_name}': {description}. Page context: {page_info}"
        docx: "From Word document '{file_name}': {description}. Section: {section_info}"
        
    vllm_integration:
      enabled: ${INGESTION_VLLM_ENABLED:-false}
      endpoint: ${VLLM_ENDPOINT:-http://localhost:8000/v1}
      model: ${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
      timeout: ${VLLM_TIMEOUT:-30}
      max_tokens: ${VLLM_MAX_TOKENS:-150}
      temperature: ${VLLM_TEMPERATURE:-0.3}
      
    confidence_scoring:
      enabled: true
      min_confidence: 0.5
      confidence_sources: ["content_match", "structure_analysis", "llm_assessment"]
      
    hallucination_detection:
      enabled: true
      threshold: ${INGESTION_HALLUCINATION_THRESHOLD:-0.8}
      validation_methods: ["grounding_check", "consistency_analysis"]

  # Storage Integration
  storage:
    auto_embed: ${INGESTION_AUTO_EMBED:-true}
    auto_graph: ${INGESTION_AUTO_GRAPH:-true}
    extract_entities: ${INGESTION_EXTRACT_ENTITIES:-true}
    create_relationships: ${INGESTION_CREATE_RELATIONSHIPS:-true}
    
  # Error Handling
  error_handling:
    retry_attempts: ${INGESTION_RETRY_ATTEMPTS:-3}
    retry_delay: ${INGESTION_RETRY_DELAY:-1.0}
    fallback_strategy: "graceful_degradation"  # strict, graceful_degradation, skip
    log_failures: true
    
  # Performance Optimization
  performance:
    streaming_threshold: ${INGESTION_STREAMING_THRESHOLD:-10485760}  # 10MB
    batch_processing: true
    parallel_chunks: ${INGESTION_PARALLEL_CHUNKS:-5}
    cache_parsed_content: ${INGESTION_CACHE_CONTENT:-true}
    cache_ttl: ${INGESTION_CACHE_TTL:-3600}  # 1 hour
```

### Observability Configuration (`config/observability.yaml`)

```yaml
# OpenTelemetry Configuration
otel:
  enabled: ${OTEL_ENABLED:-true}
  service_name: ${OTEL_SERVICE_NAME:-tyra-mcp-memory-server}
  service_version: ${OTEL_SERVICE_VERSION:-1.0.0}
  environment: ${OTEL_ENVIRONMENT:-development}

# Tracing Configuration
tracing:
  enabled: ${OTEL_TRACES_ENABLED:-true}
  exporter: ${OTEL_TRACES_EXPORTER:-console}
  endpoint: ${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4318/v1/traces}
  sampler: ${OTEL_TRACES_SAMPLER:-parentbased_traceidratio}
  sampler_arg: ${OTEL_TRACES_SAMPLER_ARG:-1.0}
  max_spans: ${OTEL_TRACES_MAX_SPANS:-1000}

# Metrics Configuration
metrics:
  enabled: ${OTEL_METRICS_ENABLED:-true}
  exporter: ${OTEL_METRICS_EXPORTER:-console}
  endpoint: ${OTEL_EXPORTER_OTLP_METRICS_ENDPOINT:-http://localhost:4318/v1/metrics}
  export_interval: ${OTEL_METRIC_EXPORT_INTERVAL:-60000}
  export_timeout: ${OTEL_METRIC_EXPORT_TIMEOUT:-30000}

# Logging Configuration
logging:
  enabled: ${OTEL_LOGS_ENABLED:-true}
  exporter: ${OTEL_LOGS_EXPORTER:-console}
  level: ${LOG_LEVEL:-INFO}
  format: ${LOG_FORMAT:-json}
  rotation:
    enabled: ${LOG_ROTATION_ENABLED:-true}
    max_size: ${LOG_ROTATION_MAX_SIZE:-100MB}
    max_files: ${LOG_ROTATION_MAX_FILES:-10}
    max_age: ${LOG_ROTATION_MAX_AGE:-30}
```

### Self-Learning Configuration (`config/self_learning.yaml`)

```yaml
# Self-Learning System
self_learning:
  enabled: ${SELF_LEARNING_ENABLED:-true}
  analysis_interval: ${SELF_LEARNING_ANALYSIS_INTERVAL:-3600}  # 1 hour
  improvement_interval: ${SELF_LEARNING_IMPROVEMENT_INTERVAL:-86400}  # 24 hours
  auto_optimize: ${SELF_LEARNING_AUTO_OPTIMIZE:-true}

# Performance Tracking
performance:
  tracking_enabled: ${PERFORMANCE_TRACKING_ENABLED:-true}
  sample_rate: ${PERFORMANCE_SAMPLE_RATE:-0.1}
  slow_query_threshold: ${PERFORMANCE_SLOW_QUERY_THRESHOLD:-1000}
  metrics_retention_days: ${PERFORMANCE_METRICS_RETENTION:-30}

# Memory Health Management
memory_health:
  enabled: ${MEMORY_HEALTH_ENABLED:-true}
  check_interval: ${MEMORY_HEALTH_CHECK_INTERVAL:-3600}  # 1 hour
  cleanup_interval: ${MEMORY_HEALTH_CLEANUP_INTERVAL:-86400}  # 24 hours
  stale_threshold_days: ${MEMORY_HEALTH_STALE_THRESHOLD:-30}
  redundancy_threshold: ${MEMORY_HEALTH_REDUNDANCY_THRESHOLD:-0.9}

# A/B Testing
ab_testing:
  enabled: ${AB_TESTING_ENABLED:-true}
  default_split: ${AB_TESTING_DEFAULT_SPLIT:-0.5}
  min_sample_size: ${AB_TESTING_MIN_SAMPLE_SIZE:-100}
  significance_level: ${AB_TESTING_SIGNIFICANCE_LEVEL:-0.05}
  max_experiments: ${AB_TESTING_MAX_EXPERIMENTS:-5}

# Adaptation Thresholds
thresholds:
  memory_quality:
    excellent: 0.9
    good: 0.8
    fair: 0.7
    poor: 0.6
  performance:
    response_time_ms: 500
    cache_hit_rate: 0.8
    error_rate: 0.01
  confidence:
    rock_solid: 0.95
    high: 0.8
    fuzzy: 0.6
    low: 0.4
```

## 🌍 Environment Variables

### Database Configuration

```env
# PostgreSQL
DATABASE_URL=postgresql://user:password@host:port/database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tyra_memory
POSTGRES_USER=tyra
POSTGRES_PASSWORD=secure_password
POSTGRES_SSL_MODE=prefer
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=10
POSTGRES_POOL_TIMEOUT=30

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_MAX_CONNECTIONS=50
REDIS_CONNECTION_TIMEOUT=5

# Neo4j
NEO4J_URL=bolt://localhost:7687
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
NEO4J_CONNECTION_TIMEOUT=30
NEO4J_POOL_SIZE=10
```

### Application Configuration

```env
# Environment
ENVIRONMENT=development  # development, production, testing
DEBUG=false
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false
API_ENABLE_DOCS=true
API_CORS_ORIGINS=["*"]

# MCP Settings
MCP_HOST=localhost
MCP_PORT=3000
MCP_TRANSPORT=stdio  # stdio, sse
MCP_TIMEOUT=30
MCP_LOG_LEVEL=INFO
```

### Embedding Configuration

```env
# Primary Embedding Model
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=auto  # auto, cpu, cuda
EMBEDDINGS_BATCH_SIZE=32
EMBEDDINGS_MAX_LENGTH=512

# Fallback Embedding Model
EMBEDDINGS_FALLBACK_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_FALLBACK_DEVICE=cpu
EMBEDDINGS_FALLBACK_BATCH_SIZE=16
EMBEDDINGS_FALLBACK_MAX_LENGTH=256

# Model Configuration - LOCAL ONLY
EMBEDDINGS_PRIMARY_PATH=./models/embeddings/e5-large-v2
EMBEDDINGS_FALLBACK_PATH=./models/embeddings/all-MiniLM-L12-v2
EMBEDDINGS_USE_LOCAL_FILES=true
RERANKER_MODEL_PATH=./models/cross-encoders/ms-marco-MiniLM-L-6-v2
RERANKER_USE_LOCAL_FILES=true

# Model Cache
MODEL_CACHE_DIR=./data/models
MODEL_CACHE_MAX_SIZE=50
MODEL_AUTO_DOWNLOAD=false  # Manual download required
MODEL_DOWNLOAD_TIMEOUT=300

# Enhanced Model Settings (Phase 1-2)
MULTIMODAL_MODELS_ENABLED=false  # Phase 1
CLIP_MODEL_PATH=./models/multimodal/clip-vit-base-patch32
WHISPER_MODEL_PATH=./models/audio/whisper-base
CONTEXTUAL_EMBEDDINGS_ENABLED=false  # Phase 2
```

### RAG Configuration

```env
# Current Retrieval Settings
RAG_TOP_K=20
RAG_MIN_CONFIDENCE=0.0
RAG_HYBRID_WEIGHT=0.7
RAG_DIVERSITY_PENALTY=0.3
RAG_MAX_CONTEXT_LENGTH=4000

# Current Reranking Settings
RAG_RERANKING_ENABLED=true
RAG_RERANKING_PROVIDER=cross_encoder
RAG_RERANKING_TOP_K=10
RAG_RERANKING_THRESHOLD=0.5
RAG_RERANKING_BATCH_SIZE=8

# Enhanced RAG Settings (Phase 1)
RAG_MULTI_MODAL_ENABLED=false  # Phase 1 rollout
RAG_CONTEXTUAL_LINKING_ENABLED=false
RAG_DYNAMIC_RERANKING_ENABLED=false
RAG_INTENT_CLASSIFICATION_ENABLED=false

# Anti-Hallucination Settings
RAG_HALLUCINATION_DETECTION_ENABLED=true
RAG_HALLUCINATION_THRESHOLD=0.75
RAG_SOURCE_GROUNDING_REQUIRED=true
RAG_CONFIDENCE_VALIDATION_ENABLED=true

# Hallucination Detection
RAG_HALLUCINATION_ENABLED=true
RAG_HALLUCINATION_THRESHOLD=75
RAG_GROUNDING_CHECK=true
RAG_EVIDENCE_COLLECTION=true

# Response Generation
RAG_MAX_TOKENS=2000
RAG_TEMPERATURE=0.1
RAG_INCLUDE_SOURCES=true
RAG_INCLUDE_CONFIDENCE=true
```

### Cache Configuration

```env
# Cache Settings
CACHE_ENABLED=true
CACHE_COMPRESSION=true
CACHE_COMPRESSION_THRESHOLD=1024
CACHE_DEFAULT_TTL=3600

# Cache TTL Settings (seconds)
CACHE_TTL_EMBEDDINGS=86400      # 24 hours
CACHE_TTL_SEARCH=3600           # 1 hour
CACHE_TTL_RERANK=1800           # 30 minutes
CACHE_TTL_HALLUCINATION=900     # 15 minutes
CACHE_TTL_GRAPH=7200            # 2 hours

# Cache Warming
CACHE_WARMING_ENABLED=false
```

### Security Configuration

```env
# Authentication
SECRET_KEY=your-secret-key-here-generate-with-openssl-rand-hex-32
API_KEY=your-api-key-here
JWT_SECRET=your-jwt-secret-here

# CORS
CORS_ENABLED=true
API_CORS_ORIGINS=["*"]

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_RPM=100
RATE_LIMIT_BURST=20

# SSL/TLS
SSL_ENABLED=false
SSL_CERT_PATH=/etc/ssl/certs/tyra.crt
SSL_KEY_PATH=/etc/ssl/private/tyra.key
```

### Observability Configuration

```env
# OpenTelemetry
OTEL_ENABLED=true
OTEL_SERVICE_NAME=tyra-mcp-memory-server
OTEL_SERVICE_VERSION=1.0.0
OTEL_ENVIRONMENT=development

# Tracing
OTEL_TRACES_ENABLED=true
OTEL_TRACES_EXPORTER=console  # console, jaeger, otlp
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=1.0

# Metrics
OTEL_METRICS_ENABLED=true
OTEL_METRICS_EXPORTER=console  # console, prometheus, otlp
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://localhost:4318/v1/metrics
OTEL_METRIC_EXPORT_INTERVAL=60000

# Logging
OTEL_LOGS_ENABLED=true
OTEL_LOGS_EXPORTER=console
LOG_FORMAT=json  # json, text
LOG_ROTATION_ENABLED=true
LOG_ROTATION_MAX_SIZE=100MB
LOG_ROTATION_MAX_FILES=10
LOG_ROTATION_MAX_AGE=30
```

### Self-Learning Configuration

```env
# Self-Learning
SELF_LEARNING_ENABLED=true
SELF_LEARNING_ANALYSIS_INTERVAL=3600
SELF_LEARNING_IMPROVEMENT_INTERVAL=86400
SELF_LEARNING_AUTO_OPTIMIZE=true

# Performance Tracking
PERFORMANCE_TRACKING_ENABLED=true
PERFORMANCE_SAMPLE_RATE=0.1
PERFORMANCE_SLOW_QUERY_THRESHOLD=1000
PERFORMANCE_METRICS_RETENTION=30

# Memory Health
MEMORY_HEALTH_ENABLED=true
MEMORY_HEALTH_CHECK_INTERVAL=3600
MEMORY_HEALTH_CLEANUP_INTERVAL=86400
MEMORY_HEALTH_STALE_THRESHOLD=30
MEMORY_HEALTH_REDUNDANCY_THRESHOLD=0.9

# A/B Testing
AB_TESTING_ENABLED=true
AB_TESTING_DEFAULT_SPLIT=0.5
AB_TESTING_MIN_SAMPLE_SIZE=100
AB_TESTING_SIGNIFICANCE_LEVEL=0.05
AB_TESTING_MAX_EXPERIMENTS=5
```

## 🎛️ Configuration Presets

### Development Environment

```env
# Development optimized for speed and debugging
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
API_RELOAD=true
API_ENABLE_DOCS=true

# Lightweight models for faster startup
EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=cpu
EMBEDDINGS_BATCH_SIZE=16

# Reduced caching for development
CACHE_TTL_EMBEDDINGS=3600
CACHE_TTL_SEARCH=300
RAG_RERANKING_ENABLED=false

# Disabled features for development
SELF_LEARNING_ENABLED=false
AB_TESTING_ENABLED=false
OTEL_TRACES_SAMPLER_ARG=0.1
```

### Production Environment

```env
# Production optimized for performance and reliability
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
API_RELOAD=false
API_ENABLE_DOCS=false

# High-quality models for accuracy
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=auto
EMBEDDINGS_BATCH_SIZE=32

# Optimized caching
CACHE_ENABLED=true
CACHE_TTL_EMBEDDINGS=86400
CACHE_TTL_SEARCH=3600
CACHE_WARMING_ENABLED=true

# Full feature set
RAG_RERANKING_ENABLED=true
RAG_HALLUCINATION_ENABLED=true
SELF_LEARNING_ENABLED=true
AB_TESTING_ENABLED=true

# Production monitoring
OTEL_TRACES_ENABLED=true
OTEL_METRICS_ENABLED=true
OTEL_TRACES_SAMPLER_ARG=0.1
PERFORMANCE_TRACKING_ENABLED=true
```

### Testing Environment

```env
# Testing optimized for isolation and speed
ENVIRONMENT=testing
DEBUG=true
LOG_LEVEL=DEBUG

# Separate test databases
DATABASE_URL=postgresql://test_user:test_password@localhost:5432/test_tyra_memory
REDIS_URL=redis://localhost:6379/1
NEO4J_URL=bolt://localhost:7687

# Fast models for testing
EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
EMBEDDINGS_DEVICE=cpu
EMBEDDINGS_BATCH_SIZE=8

# Minimal caching for testing
CACHE_ENABLED=false
RAG_RERANKING_ENABLED=false
SELF_LEARNING_ENABLED=false
AB_TESTING_ENABLED=false

# Minimal observability
OTEL_ENABLED=false
PERFORMANCE_TRACKING_ENABLED=false
```

## 🔧 Runtime Configuration

### API Configuration Updates

```bash
# Update configuration via API
curl -X POST http://localhost:8000/v1/admin/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "embeddings": {
      "primary": {
        "model": "intfloat/e5-large-v2",
        "device": "cuda"
      }
    }
  }'

# Reload configuration
curl -X POST http://localhost:8000/v1/admin/config/reload \
  -H "Authorization: Bearer your-api-key"
```

### Provider Swapping

```bash
# Switch to different embedding provider
curl -X POST http://localhost:8000/v1/admin/providers/embedding/switch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L12-v2"
  }'

# Switch reranking provider
curl -X POST http://localhost:8000/v1/admin/providers/reranker/switch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "provider": "vllm",
    "endpoint": "http://localhost:8001/v1"
  }'
```

## 🔍 Configuration Validation

### Validation Script

```bash
# Validate configuration
python scripts/validate_config.py

# Check specific configuration
python scripts/validate_config.py --config embeddings

# Validate environment variables
python scripts/validate_config.py --env
```

### Health Checks

```bash
# Configuration health check
curl http://localhost:8000/v1/admin/config/health

# Provider health check
curl http://localhost:8000/v1/admin/providers/health

# Database configuration check
curl http://localhost:8000/v1/admin/database/health
```

## 🚨 Troubleshooting Configuration

### Common Issues

#### Configuration File Not Found
```bash
# Check if config files exist
ls -la config/

# Create missing config files
cp config/config.yaml.example config/config.yaml
```

#### Environment Variable Issues
```bash
# Check environment variables
printenv | grep TYRA

# Validate .env file
cat .env | grep -v "^#" | grep -v "^$"
```

#### Database Connection Issues
```bash
# Test database URLs
python -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.getenv('DATABASE_URL'))
print('PostgreSQL connection:', engine.execute('SELECT 1').scalar())
"
```

#### Model Loading Issues
```bash
# Check model cache
ls -la data/models/

# Clear model cache
rm -rf data/models/*

# Test model loading
python -c "
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
print('Model loaded successfully')
"
```

### Debug Configuration

```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Check configuration loading
python -c "
from src.core.utils.config import get_config
config = get_config()
print('Configuration loaded successfully')
print('Active providers:', config.get('providers', {}))
"
```

## 📊 Performance Tuning

### Memory Optimization

```env
# Reduce memory usage
EMBEDDINGS_DEVICE=cpu
EMBEDDINGS_BATCH_SIZE=16
CACHE_L1_MAX_SIZE=500
POSTGRES_POOL_SIZE=10
REDIS_MAX_CONNECTIONS=25
```

### Speed Optimization

```env
# Optimize for speed
EMBEDDINGS_PRIMARY_MODEL=sentence-transformers/all-MiniLM-L12-v2
CACHE_ENABLED=true
CACHE_TTL_EMBEDDINGS=86400
RAG_RERANKING_ENABLED=false
SELF_LEARNING_ENABLED=false
```

### Accuracy Optimization

```env
# Optimize for accuracy
EMBEDDINGS_PRIMARY_MODEL=intfloat/e5-large-v2
EMBEDDINGS_DEVICE=cuda
RAG_RERANKING_ENABLED=true
RAG_HALLUCINATION_ENABLED=true
RAG_HALLUCINATION_THRESHOLD=85
```

## 📚 Additional Resources

### Related Documentation
- [Installation Guide](INSTALLATION.md)
- [Container Configuration](docs/CONTAINERS.md)
- [Provider Registry](docs/PROVIDER_REGISTRY.md)
- [API Documentation](API.md)

### Configuration Examples
- [Development Config](config/examples/development.yaml)
- [Production Config](config/examples/production.yaml)
- [Testing Config](config/examples/testing.yaml)

### Best Practices
1. Always validate configuration after changes
2. Use environment variables for sensitive data
3. Test configuration changes in development first
4. Monitor configuration impact on performance
5. Keep configuration files in version control
6. Document custom configuration changes

---

## 🚀 **Enhanced Configuration (Coming Soon)**

### **Phase 1: Advanced Intelligence Configuration**

```yaml
# Enhanced Intelligence Configuration (config/enhanced.yaml)
enhanced_intelligence:
  pydantic_ai:
    enabled: true
    structured_memory_operations: true
    validation_schemas:
      memory_synthesis: "strict"
      query_answering: "strict"
      entity_extraction: "permissive"
    
  anti_hallucination:
    multi_layer_validation: true
    inference_prevention:
      enabled: true
      threshold: 0.85
    output_validation:
      enabled: true
      field_validation: true
      schema_validation: true
    cross_reference_check:
      enabled: true
      knowledge_graph_validation: true
      source_grounding_required: true
    real_time_monitoring:
      enabled: true
      streaming_detection: true
      token_level_analysis: true
      
  memory_synthesis:
    deduplication:
      semantic_threshold: 0.85
      hash_fingerprinting: true
      merge_strategies: ["weighted_average", "confidence_based"]
    summarization:
      provider: "pydantic_ai"  # pydantic_ai, extractive, abstractive
      quality_scoring: "rouge_plus_factual"
      hallucination_prevention: true
    pattern_detection:
      algorithms: ["clustering", "lda", "collaborative_filtering"]
      topic_modeling: "gensim_lda"
      knowledge_gap_detection: true
    temporal_analysis:
      content_change_tracking: true
      concept_evolution: true
      lifecycle_analytics: true
      
  real_time_streams:
    websocket_infrastructure: true
    live_memory_updates: true
    progressive_search: true
    event_driven_triggers: true
    collaborative_features: true
    rate_limiting:
      connections_per_user: 10
      events_per_second: 100
      
  advanced_rag:
    multi_modal:
      image_processing: "clip"  # Local CLIP models
      audio_processing: "whisper"  # Local Whisper models
      video_processing: "combined"
      code_analysis: "ast_parser"
    contextual_linking:
      chunk_relationships: true
      coherence_scoring: true
      sequence_optimization: true
      context_expansion: true
    dynamic_reranking:
      intent_classification: true
      personalized_ranking: true
      real_time_adaptation: true
      ensemble_methods: ["cross_encoder", "vllm", "similarity"]
```

### **Phase 2: Predictive Intelligence Configuration**

```yaml
# Predictive Intelligence (Phase 2)
predictive_intelligence:
  usage_prediction:
    enabled: true
    ml_algorithms: ["markov_chains", "lstm", "random_forest"]
    prediction_horizon: "24h"
    accuracy_threshold: 0.85
    
  performance_prediction:
    enabled: true
    metrics: ["latency", "throughput", "error_rate"]
    forecasting_model: "prophet"
    alert_threshold: 0.1  # 10% degradation
    
  preloading:
    enabled: true
    cache_prediction: true
    priority_queuing: true
    success_tracking: true
    
  context_aware_embeddings:
    session_adaptation: true
    multi_perspective: true
    fine_tuning_enabled: true
    fusion_strategy: "attention_weighted"
    
  auto_optimization:
    hyperparameter_tuning: "bayesian"
    configuration_adaptation: true
    performance_monitoring: true
    rollback_protection: true
```

### **Phase 3: Self-Optimization Configuration**

```yaml
# Complete Self-Optimization (Phase 3)
self_optimization:
  continuous_learning:
    online_learning: true
    few_shot_adaptation: true
    transfer_learning: true
    catastrophic_forgetting_prevention: true
    
  ab_testing:
    framework: "local_statistical"
    max_concurrent_experiments: 5
    min_sample_size: 1000
    significance_level: 0.05
    auto_winner_selection: true
    
  autonomous_tuning:
    enabled: true
    safety_constraints: true
    performance_targets:
      latency_p95: "<100ms"
      accuracy: ">90%"
      error_rate: "<0.1%"
    optimization_algorithms: ["bayesian", "genetic", "gradient_free"]
    
  personalization:
    user_profiles: true
    adaptive_thresholds: true
    personalized_ranking: true
    privacy_preservation: "differential_privacy"
```

### **Enhanced Environment Variables**

```env
# Phase 1 Enhanced Features
PYDANTIC_AI_ENABLED=true
PYDANTIC_AI_VALIDATION_LEVEL=strict
ANTI_HALLUCINATION_ENABLED=true
ANTI_HALLUCINATION_THRESHOLD=0.85
MEMORY_SYNTHESIS_ENABLED=false  # Gradual rollout
REAL_TIME_STREAMS_ENABLED=false
MULTI_MODAL_RAG_ENABLED=false

# Phase 2 Predictive Features
PREDICTIVE_INTELLIGENCE_ENABLED=false
USAGE_PREDICTION_ENABLED=false
PERFORMANCE_PREDICTION_ENABLED=false
CONTEXT_AWARE_EMBEDDINGS_ENABLED=false
AUTO_OPTIMIZATION_ENABLED=false

# Phase 3 Self-Optimization
CONTINUOUS_LEARNING_ENABLED=false
AB_TESTING_FRAMEWORK_ENABLED=false
AUTONOMOUS_TUNING_ENABLED=false
PERSONALIZATION_ENABLED=false

# Enhanced Model Paths (Required for all phases)
CLIP_MODEL_PATH=./models/multimodal/clip-vit-base-patch32
WHISPER_MODEL_PATH=./models/audio/whisper-base
CROSS_ENCODER_PATH=./models/cross-encoders/ms-marco-MiniLM-L-6-v2

# vLLM Integration (Enhanced)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_RERANK_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_SYNTHESIS_MODEL=meta-llama/Llama-3.1-70B-Instruct
VLLM_API_KEY=dummy-key

# Enhanced Performance Tuning
ENHANCED_CACHE_ENABLED=true
PREDICTIVE_CACHE_ENABLED=false  # Phase 2
CACHE_PREDICTION_ACCURACY_TARGET=0.8
PERFORMANCE_MONITORING_ENHANCED=true
REAL_TIME_METRICS_ENABLED=true
```

### **Hot Configuration Reloading (Phase 1)**

```bash
# Reload enhanced configuration without restart
curl -X POST http://localhost:8000/v1/admin/config/reload-enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "components": ["pydantic_ai", "anti_hallucination", "memory_synthesis"],
    "validate_before_apply": true,
    "rollback_on_failure": true
  }'

# Enable specific enhanced features
curl -X POST http://localhost:8000/v1/admin/features/toggle \
  -H "Content-Type: application/json" \
  -d '{
    "memory_synthesis": true,
    "real_time_streams": false,
    "multi_modal_rag": false
  }'

# Monitor enhanced feature performance
curl http://localhost:8000/v1/admin/enhanced/metrics
```

### **Implementation Standards for Enhanced Configuration**

#### **Zero Tolerance Standards**
- ❌ **NO placeholders** in any configuration
- ❌ **NO TODO items** in config files
- ❌ **NO fake/mock settings** - all must be functional
- ❌ **NO stub configurations** - complete implementations only

#### **Quality Requirements**
- ✅ **Latest libraries only** - Most current stable versions
- ✅ **Production-ready defaults** - Optimized for real-world use
- ✅ **Hot-swappable components** - No restart required
- ✅ **Backward compatibility** - Existing configs always work
- ✅ **Comprehensive validation** - Pydantic schema validation

#### **Enhanced Configuration Architecture**
- ✅ **Modular configuration** - Each enhancement independently configurable
- ✅ **Performance-based adaptation** - Auto-tuning based on metrics
- ✅ **Feature flags** - Gradual rollout capability
- ✅ **A/B testing integration** - Built-in experimentation
- ✅ **Real-time monitoring** - Configuration impact tracking

---

## 🎆 **Configuration Evolution Timeline**

### **Current (Production Ready)**
✅ Complete configuration system with all current features

### **Phase 1 (Next 1-3 months)**
🚀 Enhanced intelligence configuration with Pydantic AI and anti-hallucination

### **Phase 2 (3-6 months)**
🤖 Predictive intelligence and auto-optimization configuration

### **Phase 3 (6-9 months)**
🎯 Complete self-optimization and autonomous configuration management

**All enhancements maintain 100% backward compatibility and include automated migration tools.**

---

🎉 **Configuration Complete!** Your Tyra MCP Memory Server is properly configured for current features and ready for upcoming enhancements.

For troubleshooting, see the [Installation Guide](INSTALLATION.md), [Enhanced Roadmap](ADVANCED_ENHANCEMENTS.md), or check the logs in `logs/memory-server.log`.