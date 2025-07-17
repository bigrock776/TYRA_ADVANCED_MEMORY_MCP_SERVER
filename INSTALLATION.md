# 📦 Tyra MCP Memory Server - Complete Installation & Usage Guide

**Version**: 3.0.0 (Production Ready)  
**Status**: ✅ Fully Integrated System  
**Estimated Setup Time**: 30-60 minutes  
**Features**: 16 MCP Tools + Dashboard + API + Suggestions System

> **🎉 PRODUCTION READY**: Complete enterprise-grade memory system with advanced AI capabilities, real-time analytics, intelligent suggestions, and comprehensive dashboard interface. All features fully integrated and operational.

## 🎯 Quick Start (10 Minutes)

**For experienced users wanting the fastest setup:**

```bash
# Clone and setup
git clone <your-repo-url>
cd tyra-mcp-memory-server

# One-command installation
./setup.sh

# Download required models
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./models/embeddings/all-MiniLM-L12-v2
huggingface-cli download intfloat/e5-large-v2 --local-dir ./models/embeddings/e5-large-v2

# Start all services
python main.py
```

**Verification**: 
- MCP Server: `python main.py` (should show "Tyra Memory MCP Server ready")
- REST API: Visit `http://localhost:8000/health`
- Dashboard: Visit `http://localhost:8050/dashboard`

---

## 📋 Prerequisites & System Requirements

### 🖥️ **System Requirements**

| Component | Minimum | Recommended | Purpose |
|-----------|---------|-------------|---------|
| **OS** | Ubuntu 20.04, macOS 12+ | Ubuntu 24.04 LTS | Core platform |
| **Python** | 3.11.0 | 3.12.2+ | Async features |
| **RAM** | 8GB | 16GB+ | ML models |
| **Storage** | 20GB free | 100GB+ | Models + data |
| **CPU** | 4 cores | 8+ cores | Processing |
| **GPU** | Optional | RTX 3060+ | 10x faster embeddings |

### 🛠️ **Required Services**

| Service | Version | Purpose | Installation |
|---------|---------|---------|-------------|
| **PostgreSQL** | 15+ | Vector storage + pgvector | `sudo apt install postgresql-15` |
| **Redis** | 6.0+ | Multi-layer caching | `sudo apt install redis-server` |
| **Neo4j** | 5.0+ | Knowledge graphs | Download from neo4j.com |

### 📦 **Required Python Packages**

**⚠️ CRITICAL**: All dependencies must be installed before model download. All models require manual download - no automatic downloads.

**Core Dependencies in `requirements.txt`:**
```bash
# MCP and API framework
mcp[cli]>=0.5.0              # Model Context Protocol
fastapi>=0.104.0             # REST API framework
uvicorn[standard]>=0.24.0    # ASGI server
pydantic>=2.5.0              # Data validation
pydantic-ai>=0.0.13          # AI-specific validation

# Database drivers
asyncpg>=0.29.0              # PostgreSQL async driver
psycopg2-binary>=2.9.9       # PostgreSQL sync driver
redis>=5.0.0                 # Cache layer
neo4j>=5.13.0               # Neo4j graph database

# AI/ML packages (for local model loading)
sentence-transformers>=2.7.0 # Embedding generation
transformers>=4.36.0         # HuggingFace transformers
torch>=2.1.0                 # PyTorch deep learning
sentencepiece>=0.1.99        # Tokenizer
protobuf>=3.20.0             # Model serialization
numpy>=1.24.0                # Numerical operations
scikit-learn>=1.3.0          # ML utilities

# Reranking and cross-encoders
rank_bm25>=0.2.2             # BM25 ranking algorithm
nltk>=3.8.1                  # Natural Language Toolkit
spacy>=3.7.2                 # Advanced NLP
langdetect>=1.0.9            # Language detection

# Dashboard & visualization
dash>=2.18.0                 # Analytics dashboards
plotly>=5.24.0               # Interactive visualization
streamlit>=1.42.0            # Web dashboards
bokeh>=3.7.0                 # Interactive visualization
altair>=5.5.0                # Statistical visualization

# Configuration and utilities
pyyaml>=6.0.1                # YAML configuration
python-dotenv>=1.0.0         # Environment variables
structlog>=23.2.0            # Structured logging
rich>=13.7.0                 # Enhanced terminal output
click>=8.1.0                 # CLI framework

# HTTP clients and file handling
httpx>=0.25.0                # Async HTTP client
aiohttp>=3.9.0               # HTTP server/client
aiofiles>=23.2.1             # Async file I/O

# Monitoring and observability
prometheus-client>=0.21.0    # Metrics collection
opentelemetry-api>=1.29.0    # Distributed tracing
opentelemetry-sdk>=1.29.0    # OpenTelemetry SDK

# Web crawling and document processing
crawl4ai>=0.7.0              # Advanced web crawling
beautifulsoup4>=4.12.3       # HTML/XML parsing
lxml>=5.3.0                  # XML/HTML processing
newspaper3k>=0.2.8           # News article extraction
selenium>=4.28.0             # Web browser automation
playwright>=1.50.0           # Modern web automation

# File ingestion
PyMuPDF>=1.22.0             # PDF file parsing
python-docx>=0.8.11         # Microsoft Word documents
chardet>=5.2.0              # Text encoding detection
unstructured>=0.10.0        # Universal document loader
markdown>=3.4.0             # Markdown parsing
html2text>=2020.1.16        # HTML to markdown
pandas>=2.1.0               # CSV/Excel files
jsonlines>=4.0.0            # JSONL format

# Security
cryptography>=41.0.0         # Encryption/decryption
passlib>=1.7.4               # Password hashing
```

**Model Requirements:**
- **HuggingFace CLI**: Required for model downloads
- **Git LFS**: Required for large model files
- **Storage**: 5GB+ for all models
- **RAM**: 8GB+ for model loading
- **GPU**: Optional but recommended (10x faster embeddings)

---

## 🚀 Installation Methods

### **Method 1: Automated Setup (Recommended)**

```bash
# 1. Clone repository
git clone <your-repo-url>
cd tyra-mcp-memory-server

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# 3. Download models (required)
source venv/bin/activate
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./models/embeddings/all-MiniLM-L12-v2
huggingface-cli download intfloat/e5-large-v2 --local-dir ./models/embeddings/e5-large-v2

# 4. Setup databases
./scripts/setup_databases.sh

# 5. Start system
python main.py
```

### **Method 2: Manual Installation**

#### **Step 1: Environment Setup**

```bash
# Python environment
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

#### **Step 2: Database Setup**

```bash
# PostgreSQL with pgvector
sudo apt install postgresql-15 postgresql-15-pgvector
sudo systemctl start postgresql
sudo -u postgres createuser -s $USER
createdb tyra_memory

# Redis
sudo apt install redis-server
sudo systemctl start redis

# Neo4j (download from neo4j.com)
# Follow official installation guide
```

#### **Step 3: Configuration**

```bash
# Copy and edit config
cp config/config.yaml.example config/config.yaml
# Edit database connections, API keys, etc.

# Initialize databases
python scripts/init_postgres.py
python scripts/init_neo4j.py
```

#### **Step 4: Model Installation (REQUIRED)**

**⚠️ CRITICAL**: All models must be manually downloaded. No automatic downloads in code.

**Prerequisites for Model Download:**
```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Install Git LFS (for large files)
sudo apt install git-lfs  # Ubuntu/Debian
brew install git-lfs      # macOS

# Login to HuggingFace (if using gated models)
huggingface-cli login
```

**Create Model Directories:**
```bash
mkdir -p models/embeddings models/cross-encoders models/rerankers
```

**A. Embedding Models (Required)**

**Primary Embedding Model (Recommended):**
```bash
# Download intfloat/e5-large-v2 (1024 dimensions, high quality)
huggingface-cli download intfloat/e5-large-v2 \
  --local-dir ./models/embeddings/e5-large-v2 \
  --local-dir-use-symlinks False

# Verify download
ls -la ./models/embeddings/e5-large-v2/
# Should contain: config.json, pytorch_model.bin, tokenizer files
```

**Fallback Embedding Model (Required):**
```bash
# Download all-MiniLM-L12-v2 (384 dimensions, fast)
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 \
  --local-dir ./models/embeddings/all-MiniLM-L12-v2 \
  --local-dir-use-symlinks False

# Verify download
ls -la ./models/embeddings/all-MiniLM-L12-v2/
```

**Alternative Embedding Models (Optional):**
```bash
# For multilingual support
huggingface-cli download intfloat/e5-small-v2 \
  --local-dir ./models/embeddings/e5-small-v2

# For very fast inference (CPU-optimized)
huggingface-cli download sentence-transformers/paraphrase-MiniLM-L3-v2 \
  --local-dir ./models/embeddings/paraphrase-MiniLM-L3-v2
```

**B. Cross-Encoder Models (For Reranking)**

**Primary Cross-Encoder (Recommended):**
```bash
# Download ms-marco-MiniLM-L-6-v2 (best quality/speed balance)
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2 \
  --local-dir-use-symlinks False

# Verify download
ls -la ./models/cross-encoders/ms-marco-MiniLM-L-6-v2/
```

**Alternative Cross-Encoders (Optional):**
```bash
# For general semantic similarity
huggingface-cli download cross-encoder/stsb-roberta-base \
  --local-dir ./models/cross-encoders/stsb-roberta-base

# For natural language inference
huggingface-cli download cross-encoder/nli-deberta-v3-base \
  --local-dir ./models/cross-encoders/nli-deberta-v3-base
```

**C. Model Size and Requirements**

| Model | Size | RAM | Purpose |
|-------|------|-----|----------|
| e5-large-v2 | ~1.2GB | 3GB+ | Primary embeddings (1024d) |
| all-MiniLM-L12-v2 | ~120MB | 1GB+ | Fallback embeddings (384d) |
| ms-marco-MiniLM-L-6-v2 | ~90MB | 1GB+ | Cross-encoder reranking |
| **Total** | **~1.4GB** | **5GB+** | **All models** |

**D. Test Model Installation**

```bash
# Test embedding models
python -c "
import torch
from sentence_transformers import SentenceTransformer

# Test primary model
print('Testing e5-large-v2...')
model = SentenceTransformer('./models/embeddings/e5-large-v2')
embedding = model.encode('Hello world')
print(f'✅ e5-large-v2: {embedding.shape}')

# Test fallback model
print('Testing all-MiniLM-L12-v2...')
model = SentenceTransformer('./models/embeddings/all-MiniLM-L12-v2')
embedding = model.encode('Hello world')
print(f'✅ all-MiniLM-L12-v2: {embedding.shape}')

print('🎉 All embedding models working!')
"

# Test cross-encoder
python -c "
from sentence_transformers import CrossEncoder

print('Testing cross-encoder...')
model = CrossEncoder('./models/cross-encoders/ms-marco-MiniLM-L-6-v2')
score = model.predict([('Hello', 'Hi there')])
print(f'✅ Cross-encoder score: {score}')
print('🎉 Cross-encoder working!')
"
```

**E. GPU Acceleration (Optional)**

```bash
# For NVIDIA GPUs (recommended for faster embeddings)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU only"}')"
```

**F. Offline Environment Setup**

For offline/secure environments:
```bash
# Download models on internet-connected machine
huggingface-cli download intfloat/e5-large-v2 --local-dir ./offline-models/e5-large-v2
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./offline-models/all-MiniLM-L12-v2

# Create tarball
tar -czf tyra-models.tar.gz offline-models/

# Transfer to offline machine and extract
scp tyra-models.tar.gz user@offline-machine:/path/to/tyra/
tar -xzf tyra-models.tar.gz
mv offline-models/* models/
```

---

## 🚀 Starting the System

### **All Services**

```bash
# Activate environment
source venv/bin/activate

# Start MCP server (primary interface)
python main.py

# In separate terminals:
# Start REST API server  
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Start dashboard
python src/dashboard/main.py

# Start file watcher (optional)
python -m src.core.services.file_watcher_service
```

### **Individual Components**

```bash
# MCP server only
python main.py

# API server only
python -c "from src.api.app import create_app; import uvicorn; uvicorn.run(create_app(), host='0.0.0.0', port=8000)"

# Dashboard only
python src/dashboard/main.py
```

---

## 🛠️ How to Use - Complete Feature Guide

### **🔧 MCP Tools (16 Available)**

Connect via MCP protocol for agent integration:

#### **Core Memory Operations**

**1. store_memory** - Store content with AI enhancement
```json
{
  "name": "store_memory",
  "arguments": {
    "content": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
    "agent_id": "claude",
    "session_id": "learning_session_1",
    "metadata": {"topic": "AI", "source": "textbook"},
    "extract_entities": true,
    "chunk_content": false
  }
}
```

**2. search_memory** - Advanced hybrid search with confidence scoring
```json
{
  "name": "search_memory", 
  "arguments": {
    "query": "What is machine learning?",
    "agent_id": "claude",
    "top_k": 10,
    "min_confidence": 0.7,
    "search_type": "hybrid",
    "include_analysis": true
  }
}
```

**3. delete_memory** - Remove specific memories
```json
{
  "name": "delete_memory",
  "arguments": {
    "memory_id": "mem_12345"
  }
}
```

#### **AI Analysis & Validation**

**4. analyze_response** - Hallucination detection with confidence scoring
```json
{
  "name": "analyze_response",
  "arguments": {
    "response": "Machine learning algorithms can achieve 100% accuracy on all datasets.",
    "query": "Tell me about ML accuracy",
    "retrieved_memories": [
      {"content": "ML accuracy varies by dataset...", "id": "mem_123"}
    ]
  }
}
```

#### **Advanced Memory Operations**

**5. deduplicate_memories** - Semantic deduplication with merge strategies
```json
{
  "name": "deduplicate_memories",
  "arguments": {
    "agent_id": "claude",
    "similarity_threshold": 0.85,
    "auto_merge": false,
    "merge_strategy": "preserve_newest"
  }
}
```

**6. summarize_memories** - AI-powered summarization with anti-hallucination
```json
{
  "name": "summarize_memories",
  "arguments": {
    "memory_ids": ["mem_123", "mem_456", "mem_789"],
    "summary_type": "extractive",
    "max_length": 200
  }
}
```

**7. detect_patterns** - Pattern recognition and knowledge gap detection
```json
{
  "name": "detect_patterns",
  "arguments": {
    "agent_id": "claude",
    "pattern_types": ["topic_clusters", "temporal_patterns"],
    "min_support": 3
  }
}
```

**8. analyze_temporal_evolution** - Track concept evolution over time
```json
{
  "name": "analyze_temporal_evolution",
  "arguments": {
    "concept": "artificial intelligence",
    "agent_id": "claude",
    "time_range_days": 30
  }
}
```

#### **Web Integration**

**9. crawl_website** - Natural language web crawling with AI extraction
```json
{
  "name": "crawl_website",
  "arguments": {
    "command": "crawl the latest AI research papers from arxiv.org",
    "max_pages": 5,
    "store_in_memory": true,
    "agent_id": "claude"
  }
}
```

#### **Intelligence Suggestions (NEW)**

**10. suggest_related_memories** - ML-powered related content suggestions
```json
{
  "name": "suggest_related_memories",
  "arguments": {
    "content": "I'm studying neural networks",
    "agent_id": "claude", 
    "limit": 10,
    "min_relevance": 0.5
  }
}
```

**11. detect_memory_connections** - Automatic connection detection
```json
{
  "name": "detect_memory_connections",
  "arguments": {
    "agent_id": "claude",
    "connection_types": ["semantic", "temporal", "entity"],
    "min_confidence": 0.6
  }
}
```

**12. recommend_memory_organization** - Structure optimization recommendations
```json
{
  "name": "recommend_memory_organization",
  "arguments": {
    "agent_id": "claude",
    "analysis_type": "clustering"
  }
}
```

**13. detect_knowledge_gaps** - Knowledge gap identification with learning paths
```json
{
  "name": "detect_knowledge_gaps",
  "arguments": {
    "agent_id": "claude",
    "domains": ["machine_learning", "neural_networks"],
    "gap_types": ["topic", "detail"]
  }
}
```

#### **System Operations**

**14. get_memory_stats** - Comprehensive system statistics
```json
{
  "name": "get_memory_stats",
  "arguments": {
    "agent_id": "claude",
    "include_performance": true,
    "include_recommendations": true
  }
}
```

**15. get_learning_insights** - Adaptive learning insights
```json
{
  "name": "get_learning_insights",
  "arguments": {
    "category": "performance",
    "days": 7
  }
}
```

**16. health_check** - Complete system health assessment
```json
{
  "name": "health_check",
  "arguments": {
    "detailed": true
  }
}
```

### **🌐 REST API Usage**

Access via HTTP endpoints at `http://localhost:8000`:

#### **Memory Operations**
```bash
# Store memory
curl -X POST http://localhost:8000/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "AI is transforming industries", "agent_id": "claude"}'

# Search memories  
curl -X POST http://localhost:8000/v1/search/memory \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "top_k": 5}'

# Get memory stats
curl http://localhost:8000/v1/memory/stats?agent_id=claude
```

#### **Trading Data Operations (NEW)**

**⚠️ CRITICAL SAFETY**: All trading operations require 95%+ confidence scores and include mandatory hallucination detection, audit logging, and safety disclaimers.

##### **OHLCV Data Storage**
```bash
# Store OHLCV data from exchanges (single)
curl -X POST http://localhost:8000/v1/trading/ohlcv \
  -H "Content-Type: application/json" \
  -d '{
    "exchange_code": "BINANCE",
    "symbol": "BTC/USDT",
    "timestamp": "2024-01-01T12:00:00Z",
    "timeframe": "1h",
    "open_price": "45000.00",
    "high_price": "45500.00",
    "low_price": "44800.00",
    "close_price": "45200.00",
    "volume": "125.45",
    "data_source": "ccxt"
  }'

# Store OHLCV data in batch
curl -X POST http://localhost:8000/v1/trading/ohlcv/batch \
  -H "Content-Type: application/json" \
  -d '{
    "exchange_code": "BINANCE",
    "symbol": "BTC/USDT",
    "data_source": "ccxt",
    "data": [{
      "timestamp": "2024-01-01T12:00:00Z",
      "timeframe": "1h",
      "open_price": "45000.00",
      "high_price": "45500.00",
      "low_price": "44800.00",
      "close_price": "45200.00",
      "volume": "125.45"
    }, {
      "timestamp": "2024-01-01T13:00:00Z",
      "timeframe": "1h",
      "open_price": "45200.00",
      "high_price": "45800.00",
      "low_price": "45100.00",
      "close_price": "45600.00",
      "volume": "98.32"
    }]
  }'

# Get OHLCV data
curl "http://localhost:8000/v1/trading/ohlcv?symbol=BTC/USDT&exchange_code=BINANCE&timeframe=1h&limit=100"
```

##### **Sentiment Analysis Data**
```bash
# Store sentiment data (single)
curl -X POST http://localhost:8000/v1/trading/sentiment \
  -H "Content-Type: application/json" \
  -d '{
    "source_name": "Fear & Greed Index",
    "symbol": "BTC/USDT",
    "timestamp": "2024-01-01T12:00:00Z",
    "timeframe": "1h",
    "sentiment_score": "0.75",
    "sentiment_label": "bullish",
    "confidence": 0.9,
    "keywords": ["bullish", "growth", "adoption"],
    "raw_data": {"fear_greed_value": 75}
  }'

# Store sentiment data in batch
curl -X POST http://localhost:8000/v1/trading/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{
    "source_name": "Fear & Greed Index",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "data": [{
      "timestamp": "2024-01-01T12:00:00Z",
      "sentiment_score": "0.75",
      "sentiment_label": "bullish",
      "confidence": 0.9,
      "keywords": ["bullish", "growth", "adoption"]
    }, {
      "timestamp": "2024-01-01T13:00:00Z",
      "sentiment_score": "0.65",
      "sentiment_label": "neutral",
      "confidence": 0.8,
      "keywords": ["consolidation", "sideways"]
    }]
  }'

# Get sentiment analysis
curl "http://localhost:8000/v1/trading/sentiment?symbol=BTC/USDT&source_name=Fear%20%26%20Greed%20Index&limit=50"
```

##### **News Articles Storage**
```bash
# Store news article (single)
curl -X POST http://localhost:8000/v1/trading/news \
  -H "Content-Type: application/json" \
  -d '{
    "source_name": "CoinDesk",
    "title": "Bitcoin Adoption Grows in Institutional Sector",
    "content": "Major institutions continue adopting Bitcoin as a treasury reserve asset...",
    "published_at": "2024-01-01T12:00:00Z",
    "url": "https://coindesk.com/article",
    "author": "Jane Crypto",
    "sentiment_score": "0.8",
    "sentiment_label": "bullish",
    "symbols_mentioned": ["BTC", "ETH"],
    "keywords": ["adoption", "institutional", "treasury"],
    "summary": "Institutional Bitcoin adoption continues to accelerate"
  }'

# Store news articles in batch
curl -X POST http://localhost:8000/v1/trading/news/batch \
  -H "Content-Type: application/json" \
  -d '{
    "source_name": "CoinDesk",
    "articles": [{
      "title": "Bitcoin Adoption Grows",
      "content": "Major institutions continue adopting Bitcoin...",
      "published_at": "2024-01-01T12:00:00Z",
      "sentiment_score": "0.8",
      "symbols_mentioned": ["BTC", "ETH"]
    }, {
      "title": "Ethereum 2.0 Update",
      "content": "Ethereum network sees major upgrade...",
      "published_at": "2024-01-01T13:00:00Z",
      "sentiment_score": "0.7",
      "symbols_mentioned": ["ETH"]
    }]
  }'

# Get news articles
curl "http://localhost:8000/v1/trading/news?symbols=BTC,ETH&source_name=CoinDesk&limit=20"
```

##### **Trading Positions Management**
```bash
# Update single position
curl -X POST http://localhost:8000/v1/trading/positions \
  -H "Content-Type: application/json" \
  -d '{
    "account_name": "trading_account_1",
    "exchange_code": "BINANCE",
    "symbol": "BTC/USDT",
    "side": "long",
    "quantity": "0.5",
    "entry_price": "45000.00",
    "current_price": "45200.00",
    "unrealized_pnl": "100.00",
    "realized_pnl": "0.00",
    "status": "open",
    "open_time": "2024-01-01T10:00:00Z"
  }'

# Update multiple positions
curl -X POST http://localhost:8000/v1/trading/positions/update \
  -H "Content-Type: application/json" \
  -d '[{
    "account_name": "trading_account_1",
    "exchange_code": "BINANCE",
    "symbol": "BTC/USDT",
    "side": "long",
    "quantity": "0.5",
    "entry_price": "45000.00",
    "current_price": "45200.00",
    "unrealized_pnl": "100.00",
    "status": "open"
  }, {
    "account_name": "trading_account_1",
    "exchange_code": "BINANCE",
    "symbol": "ETH/USDT",
    "side": "short",
    "quantity": "2.0",
    "entry_price": "3200.00",
    "current_price": "3150.00",
    "unrealized_pnl": "100.00",
    "status": "open"
  }]'

# Get positions
curl "http://localhost:8000/v1/trading/positions?account_name=trading_account_1&status=open"

# Close position
curl -X PUT http://localhost:8000/v1/trading/positions/close \
  -H "Content-Type: application/json" \
  -d '{
    "account_name": "trading_account_1",
    "symbol": "BTC/USDT",
    "close_price": "45500.00",
    "close_time": "2024-01-01T14:00:00Z",
    "realized_pnl": "250.00"
  }'
```

##### **Trading Signals**
```bash
# Store trading signal
curl -X POST http://localhost:8000/v1/trading/signals \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT",
    "exchange_code": "BINANCE",
    "signal_type": "buy",
    "signal_strength": "0.85",
    "confidence": "0.95",
    "timeframe": "4h",
    "source": "technical",
    "strategy_name": "RSI_MACD",
    "entry_price": "45000.00",
    "target_price": "47000.00",
    "stop_loss_price": "43000.00",
    "reasoning": "RSI oversold + MACD bullish crossover",
    "expires_at": "2024-01-01T16:00:00Z",
    "risk_level": "medium",
    "expected_duration": "2-4 hours"
  }'

# Get trading signals
curl "http://localhost:8000/v1/trading/signals?symbol=BTC/USDT&signal_type=buy&min_confidence=0.9"

# Update signal status
curl -X PUT http://localhost:8000/v1/trading/signals/{signal_id}/status \
  -H "Content-Type: application/json" \
  -d '{
    "status": "filled",
    "fill_price": "45100.00",
    "fill_time": "2024-01-01T13:30:00Z"
  }'
```

##### **Market Analysis & Aggregations**
```bash
# Get price analysis
curl "http://localhost:8000/v1/trading/analysis/price?symbol=BTC/USDT&timeframe=1h&periods=24"

# Get sentiment analysis aggregated
curl "http://localhost:8000/v1/trading/analysis/sentiment?symbol=BTC/USDT&timeframe=1h&periods=24"

# Get correlation analysis
curl "http://localhost:8000/v1/trading/analysis/correlation?symbols=BTC/USDT,ETH/USDT&timeframe=1h&periods=168"

# Get volume profile
curl "http://localhost:8000/v1/trading/analysis/volume?symbol=BTC/USDT&timeframe=1h&periods=24"
```

##### **Risk Management & Compliance**
```bash
# Get position risk metrics
curl "http://localhost:8000/v1/trading/risk/positions?account_name=trading_account_1"

# Get portfolio exposure
curl "http://localhost:8000/v1/trading/risk/exposure?account_name=trading_account_1"

# Get risk limits
curl "http://localhost:8000/v1/trading/risk/limits?account_name=trading_account_1"

# Set risk limits
curl -X POST http://localhost:8000/v1/trading/risk/limits \
  -H "Content-Type: application/json" \
  -d '{
    "account_name": "trading_account_1",
    "max_position_size": 10000,
    "max_daily_loss": 1000,
    "max_leverage": 3.0,
    "allowed_symbols": ["BTC/USDT", "ETH/USDT"]
  }'
```

##### **Trading System Health & Monitoring**
```bash
# Get comprehensive trading health
curl http://localhost:8000/v1/trading/health

# Get data freshness metrics
curl http://localhost:8000/v1/trading/health/data-freshness

# Get system performance metrics
curl http://localhost:8000/v1/trading/health/performance

# Get audit trail
curl "http://localhost:8000/v1/trading/audit?account_name=trading_account_1&action_type=trade&limit=100"
```

##### **CCXT Integration Endpoints**
```bash
# List supported exchanges
curl http://localhost:8000/v1/trading/exchanges

# Get exchange info
curl http://localhost:8000/v1/trading/exchanges/binance/info

# Get exchange markets
curl http://localhost:8000/v1/trading/exchanges/binance/markets

# Get real-time ticker
curl http://localhost:8000/v1/trading/exchanges/binance/ticker/BTC/USDT

# Get order book
curl http://localhost:8000/v1/trading/exchanges/binance/orderbook/BTC/USDT

# Get recent trades
curl http://localhost:8000/v1/trading/exchanges/binance/trades/BTC/USDT
```

##### **Webhook Integration (for n8n)**
```bash
# Register trading webhook
curl -X POST http://localhost:8000/v1/trading/webhooks/register \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_url": "https://your-n8n-instance.com/webhook/trading-data",
    "events": ["new_signal", "position_update", "sentiment_change"],
    "filters": {
      "symbols": ["BTC/USDT", "ETH/USDT"],
      "min_confidence": 0.8
    }
  }'

# Test webhook
curl -X POST http://localhost:8000/v1/trading/webhooks/test \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_id": "webhook_123",
    "test_event": "new_signal"
  }'
```

##### **Trading Data Schema**

**OHLCV Data Fields:**
- `exchange_code`: Exchange identifier (BINANCE, COINBASE, etc.)
- `symbol`: Trading pair (BTC/USDT, ETH/USDT, etc.)
- `timestamp`: ISO 8601 timestamp
- `timeframe`: 1m, 5m, 15m, 1h, 4h, 1d, etc.
- `open_price`, `high_price`, `low_price`, `close_price`: Decimal prices
- `volume`: Trading volume
- `data_source`: ccxt, websocket, api, etc.

**Sentiment Data Fields:**
- `source_name`: Data provider name
- `symbol`: Associated trading pair
- `sentiment_score`: -1.0 to 1.0 (bearish to bullish)
- `sentiment_label`: bearish, neutral, bullish
- `confidence`: 0.0 to 1.0 confidence score
- `keywords`: Array of relevant keywords
- `raw_data`: Original provider data (JSON)

**News Article Fields:**
- `source_name`: News provider
- `title`: Article headline
- `content`: Full article text
- `published_at`: Publication timestamp
- `url`: Article URL
- `author`: Article author
- `sentiment_score`: -1.0 to 1.0
- `symbols_mentioned`: Array of relevant symbols
- `keywords`: Extracted keywords
- `summary`: AI-generated summary

**Position Fields:**
- `account_name`: Trading account identifier
- `exchange_code`: Exchange name
- `symbol`: Trading pair
- `side`: long, short
- `quantity`: Position size
- `entry_price`: Average entry price
- `current_price`: Current market price
- `unrealized_pnl`: Unrealized profit/loss
- `realized_pnl`: Realized profit/loss
- `status`: open, closed, pending

**Signal Fields:**
- `symbol`: Trading pair
- `signal_type`: buy, sell, hold
- `signal_strength`: 0.0 to 1.0
- `confidence`: 0.0 to 1.0 (minimum 0.95 for execution)
- `timeframe`: Signal timeframe
- `source`: technical, fundamental, sentiment
- `strategy_name`: Strategy identifier
- `entry_price`: Suggested entry
- `target_price`: Profit target
- `stop_loss_price`: Stop loss level
- `reasoning`: Human-readable explanation
- `risk_level`: low, medium, high

#### **Suggestions & Analytics**
```bash
# Get related suggestions
curl -X POST http://localhost:8000/v1/suggestions/related \
  -H "Content-Type: application/json" \
  -d '{"content": "machine learning concepts", "limit": 10}'

# Detect connections
curl -X POST http://localhost:8000/v1/suggestions/connections \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "claude", "min_confidence": 0.7}'

# Organization recommendations
curl -X POST http://localhost:8000/v1/suggestions/organization \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "claude", "analysis_type": "clustering"}'

# Knowledge gaps
curl -X POST http://localhost:8000/v1/suggestions/gaps \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "claude", "gap_types": ["topic", "detail"]}'
```

#### **RAG & Synthesis**
```bash
# Rerank results
curl -X POST http://localhost:8000/v1/rag/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "AI ethics", "documents": ["doc1", "doc2"]}'

# Deduplicate memories
curl -X POST http://localhost:8000/v1/synthesis/deduplicate \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "claude", "similarity_threshold": 0.85}'

# Summarize memories
curl -X POST http://localhost:8000/v1/synthesis/summarize \
  -H "Content-Type: application/json" \
  -d '{"memory_ids": ["mem_1", "mem_2"], "summary_type": "abstractive"}'
```

### **📊 Dashboard Usage**

Access the interactive dashboard at `http://localhost:8050/dashboard`

#### **Available Dashboards:**

**1. Memory Network Visualization**
- Interactive 3D/2D graph of memory connections
- Real-time updates via WebSocket
- Node clustering and filtering
- Export capabilities

**2. Usage Analytics**
- Memory usage patterns and trends
- Agent activity analysis
- Performance metrics visualization
- ROI analysis with cost-benefit breakdowns

**3. Knowledge Gaps Analysis**
- Gap detection with severity scoring
- Learning path generation
- Progress tracking
- Impact assessment

**4. Performance Monitoring**
- System health metrics
- Response time analysis
- Cache performance
- Resource utilization

#### **Dashboard Features:**
- **Real-time updates** via WebSocket connections
- **Interactive filtering** by agent, time range, topics
- **Export functionality** for reports and data
- **Responsive design** for mobile and desktop
- **Custom dashboard creation** with templates

### **📁 File Ingestion System**

#### **Automatic File Processing**
```bash
# Drop files in the inbox for automatic processing
cp document.pdf tyra-ingest/inbox/
cp research_paper.txt tyra-ingest/inbox/
cp presentation.pptx tyra-ingest/inbox/

# Files are automatically:
# 1. Processed and chunked
# 2. Embedded with AI models
# 3. Stored in vector database
# 4. Linked in knowledge graph
# 5. Made searchable via MCP/API

# Check processing status
ls tyra-ingest/processed/  # Successfully processed files
ls tyra-ingest/failed/     # Failed processing files
```

#### **Supported File Formats**
- **Documents**: PDF, DOCX, TXT, MD, RTF
- **Presentations**: PPTX, PPT
- **Spreadsheets**: XLSX, CSV
- **Code**: PY, JS, TS, JSON, YAML
- **Web**: HTML, XML
- **Images**: PNG, JPG (with OCR)

### **🔄 Real-time Features**

#### **WebSocket Streaming**
```javascript
// Memory stream - real-time memory updates
const memoryWS = new WebSocket('ws://localhost:8000/v1/ws/memory-stream');
memoryWS.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Memory update:', update);
};

// Search stream - progressive search results
const searchWS = new WebSocket('ws://localhost:8000/v1/ws/search-stream');
searchWS.send(JSON.stringify({
  query: "artificial intelligence",
  agent_id: "claude"
}));

// Analytics stream - live performance metrics
const analyticsWS = new WebSocket('ws://localhost:8000/v1/ws/analytics-stream');
```

### **🤖 LLM Integration Examples**

#### **Local LLM Integration (Ollama, LM Studio, etc.)**

**Method 1: Ollama Integration**

**Step 1: Install Ollama MCP Client**
```bash
# Install MCP client for Ollama
pip install mcp-client ollama

# Or use the provided client
cd /path/to/tyra-mcp-memory-server
pip install -e .
```

**Step 2: Create Ollama Integration Script**
```python
# create_ollama_integration.py
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

class TyraOllamaClient:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.mcp_session = None
        
    async def connect_to_tyra(self):
        """Connect to Tyra MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            cwd="/path/to/tyra-mcp-memory-server"
        )
        
        self.mcp_session = await stdio_client(server_params)
        print("✅ Connected to Tyra MCP Memory Server")
        
        # List available tools
        tools = await self.mcp_session.list_tools()
        print(f"📋 Available tools: {len(tools.tools)}")
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")
    
    async def chat_with_memory(self, user_message):
        """Chat with Ollama using Tyra memory context."""
        
        # Search relevant memories
        search_result = await self.mcp_session.call_tool(
            "search_memory",
            {
                "query": user_message,
                "agent_id": "ollama",
                "top_k": 5,
                "min_confidence": 0.7
            }
        )
        
        # Build context from memories
        context = ""
        if search_result.content:
            memories = json.loads(search_result.content[0].text)
            if memories.get("results"):
                context = "\n".join([
                    f"Memory: {mem['content']}" 
                    for mem in memories["results"][:3]
                ])
        
        # Create enhanced prompt
        enhanced_prompt = f"""
Context from memory:
{context}

User question: {user_message}

Please provide a helpful response using the context if relevant.
"""
        
        # Get response from Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": enhanced_prompt}]
        )
        
        llm_response = response['message']['content']
        
        # Store the interaction in memory
        await self.mcp_session.call_tool(
            "store_memory",
            {
                "content": f"User: {user_message}\nResponse: {llm_response}",
                "agent_id": "ollama",
                "metadata": {
                    "model": self.model_name,
                    "interaction_type": "chat",
                    "timestamp": str(asyncio.get_event_loop().time())
                }
            }
        )
        
        return llm_response

# Usage example
async def main():
    client = TyraOllamaClient("llama3.1")
    await client.connect_to_tyra()
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = await client.chat_with_memory(user_input)
        print(f"Ollama: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Run the Integration**
```bash
# Start Tyra MCP server (separate terminal)
cd /path/to/tyra-mcp-memory-server
python main.py

# Start Ollama (separate terminal)
ollama serve

# Run the integration
python create_ollama_integration.py
```

**Method 2: LM Studio Integration**

**Create LM Studio Integration:**
```python
# lm_studio_integration.py
import asyncio
import json
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class TyraLMStudioClient:
    def __init__(self, lm_studio_url="http://localhost:1234"):
        self.lm_studio_url = lm_studio_url
        self.mcp_session = None
        
    async def connect_to_tyra(self):
        """Connect to Tyra MCP server."""
        server_params = StdioServerParameters(
            command="python", 
            args=["main.py"],
            cwd="/path/to/tyra-mcp-memory-server"
        )
        
        self.mcp_session = await stdio_client(server_params)
        print("✅ Connected to Tyra MCP Memory Server")
    
    def query_lm_studio(self, prompt, max_tokens=500):
        """Query LM Studio local server."""
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{self.lm_studio_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}"
    
    async def chat_with_memory(self, user_message):
        """Chat with LM Studio using Tyra memory."""
        
        # Get memory context
        search_result = await self.mcp_session.call_tool(
            "search_memory",
            {
                "query": user_message,
                "agent_id": "lm_studio", 
                "top_k": 3,
                "include_analysis": True
            }
        )
        
        # Build context
        context = ""
        if search_result.content:
            memories = json.loads(search_result.content[0].text)
            if memories.get("results"):
                context = "\n".join([
                    f"Relevant memory: {mem['content']}"
                    for mem in memories["results"]
                ])
        
        # Enhanced prompt with memory context
        enhanced_prompt = f"""
Previous conversation context:
{context}

Current question: {user_message}

Please respond helpfully using the context if relevant.
"""
        
        # Get LM Studio response
        llm_response = self.query_lm_studio(enhanced_prompt)
        
        # Store interaction in memory
        await self.mcp_session.call_tool(
            "store_memory",
            {
                "content": f"Q: {user_message}\nA: {llm_response}",
                "agent_id": "lm_studio",
                "extract_entities": True,
                "metadata": {"source": "lm_studio_chat"}
            }
        )
        
        return llm_response

# Usage
async def main():
    client = TyraLMStudioClient()
    await client.connect_to_tyra()
    
    print("LM Studio + Tyra Memory Chat")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = await client.chat_with_memory(user_input)
        print(f"LM Studio: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

**Method 3: Any OpenAI-Compatible API**

**Generic Local LLM Integration:**
```python
# generic_llm_integration.py
import asyncio
import json
import openai
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

class TyraGenericLLMClient:
    def __init__(self, api_base="http://localhost:8000", model_name="local-model"):
        # Configure for your local LLM API
        self.client = openai.AsyncOpenAI(
            api_key="not-needed",  # Many local APIs don't need keys
            base_url=api_base      # Your local LLM endpoint
        )
        self.model_name = model_name
        self.mcp_session = None
    
    async def connect_to_tyra(self):
        """Connect to Tyra MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"], 
            cwd="/path/to/tyra-mcp-memory-server"
        )
        
        self.mcp_session = await stdio_client(server_params)
        print("✅ Connected to Tyra MCP Memory Server")
    
    async def enhanced_chat(self, user_message, agent_id="local_llm"):
        """Chat with memory enhancement."""
        
        # Search for relevant memories
        memory_search = await self.mcp_session.call_tool(
            "search_memory",
            {
                "query": user_message,
                "agent_id": agent_id,
                "search_type": "hybrid",
                "top_k": 5,
                "min_confidence": 0.6
            }
        )
        
        # Get suggestions for additional context
        suggestions = await self.mcp_session.call_tool(
            "suggest_related_memories", 
            {
                "content": user_message,
                "agent_id": agent_id,
                "limit": 3
            }
        )
        
        # Build enhanced context
        context_parts = []
        
        if memory_search.content:
            memories = json.loads(memory_search.content[0].text)
            if memories.get("results"):
                context_parts.extend([
                    f"Memory: {mem['content']}" 
                    for mem in memories["results"][:3]
                ])
        
        if suggestions.content:
            sugg_data = json.loads(suggestions.content[0].text)
            if sugg_data.get("suggestions"):
                context_parts.extend([
                    f"Related: {sugg['content'][:100]}..." 
                    for sugg in sugg_data["suggestions"][:2]
                ])
        
        # Create enhanced prompt
        system_prompt = """You are an AI assistant with access to conversation memory. 
Use the provided context to give more informed and consistent responses."""
        
        context_text = "\n".join(context_parts) if context_parts else "No relevant previous context found."
        
        user_prompt = f"""
Context from previous conversations:
{context_text}

Current question: {user_message}
"""
        
        # Get LLM response
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content
            
            # Store the conversation in memory
            await self.mcp_session.call_tool(
                "store_memory",
                {
                    "content": f"User: {user_message}\nAssistant: {llm_response}",
                    "agent_id": agent_id,
                    "extract_entities": True,
                    "metadata": {
                        "model": self.model_name,
                        "context_used": len(context_parts) > 0,
                        "conversation_turn": True
                    }
                }
            )
            
            return llm_response
            
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

# Example usage for different local LLM setups
async def main():
    # Configure for your local LLM:
    
    # For Ollama:
    # client = TyraGenericLLMClient("http://localhost:11434", "llama3.1")
    
    # For LM Studio:
    # client = TyraGenericLLMClient("http://localhost:1234", "local-model")
    
    # For text-generation-webui:
    # client = TyraGenericLLMClient("http://localhost:5000", "your-model")
    
    # For LocalAI:
    # client = TyraGenericLLMClient("http://localhost:8080", "your-model")
    
    # For vLLM:
    # client = TyraGenericLLMClient("http://localhost:8000", "meta-llama/Llama-2-7b-chat-hf")
    
    # For TGI (Text Generation Inference):
    # client = TyraGenericLLMClient("http://localhost:3000", "your-model")
    
    client = TyraGenericLLMClient("http://localhost:1234", "local-model")
    
    await client.connect_to_tyra()
    
    print("Enhanced Local LLM Chat with Tyra Memory")
    print("Type 'exit' to quit, 'stats' for memory stats\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'stats':
            stats = await client.mcp_session.call_tool("get_memory_stats", {"agent_id": "local_llm"})
            print(f"Memory Stats: {stats.content[0].text}\n")
            continue
            
        response = await client.enhanced_chat(user_input)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

**Method 4: vLLM Integration**

**Step 1: Start vLLM Server**
```bash
# Install vLLM if not already installed
pip install vllm

# Start vLLM server with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000

# Or for a different model:
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/DialoGPT-large \
  --host 0.0.0.0 \
  --port 8000
```

**Step 2: Create vLLM Integration Script**
```python
# vllm_integration.py
import asyncio
import json
import openai
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

class TyravLLMClient:
    def __init__(self, vllm_url="http://localhost:8000", model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.client = openai.AsyncOpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=f"{vllm_url}/v1"  # vLLM OpenAI-compatible endpoint
        )
        self.model_name = model_name
        self.mcp_session = None
    
    async def connect_to_tyra(self):
        """Connect to Tyra MCP server."""
        server_params = StdioServerParameters(
            command="python",
            args=["main.py"],
            cwd="/path/to/tyra-mcp-memory-server"
        )
        
        self.mcp_session = await stdio_client(server_params)
        print("✅ Connected to Tyra MCP Memory Server")
        print("✅ Connected to vLLM Server")
    
    async def chat_with_vllm_memory(self, user_message, agent_id="vllm"):
        """Enhanced chat with vLLM using Tyra memory."""
        
        # Search for relevant memories
        memory_search = await self.mcp_session.call_tool(
            "search_memory",
            {
                "query": user_message,
                "agent_id": agent_id,
                "search_type": "hybrid",
                "top_k": 5,
                "min_confidence": 0.6
            }
        )
        
        # Get intelligent suggestions
        suggestions = await self.mcp_session.call_tool(
            "suggest_related_memories",
            {
                "content": user_message,
                "agent_id": agent_id,
                "limit": 3,
                "min_relevance": 0.5
            }
        )
        
        # Build enhanced context
        context_parts = []
        
        if memory_search.content:
            memories = json.loads(memory_search.content[0].text)
            if memories.get("results"):
                context_parts.extend([
                    f"Previous context: {mem['content'][:200]}..."
                    for mem in memories["results"][:3]
                ])
        
        if suggestions.content:
            sugg_data = json.loads(suggestions.content[0].text)
            if sugg_data.get("suggestions"):
                context_parts.extend([
                    f"Related topic: {sugg['content'][:150]}..."
                    for sugg in sugg_data["suggestions"][:2]
                ])
        
        # Create enhanced prompt for vLLM
        system_prompt = "You are a helpful AI assistant with access to conversation history. Use the provided context to give informed responses."
        
        context_text = "\n".join(context_parts) if context_parts else "No previous context available."
        
        user_prompt = f"""
Conversation Context:
{context_text}

Current Question: {user_message}

Please respond helpfully, using the context if relevant.
"""
        
        try:
            # Query vLLM
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=512,
                top_p=0.9
            )
            
            vllm_response = response.choices[0].message.content
            
            # Store interaction in Tyra memory
            await self.mcp_session.call_tool(
                "store_memory",
                {
                    "content": f"User: {user_message}\nvLLM: {vllm_response}",
                    "agent_id": agent_id,
                    "extract_entities": True,
                    "metadata": {
                        "model": self.model_name,
                        "context_used": len(context_parts) > 0,
                        "llm_provider": "vllm",
                        "conversation_turn": True
                    }
                }
            )
            
            return vllm_response
            
        except Exception as e:
            return f"Error querying vLLM: {str(e)}"

# Usage example
async def main():
    # Initialize vLLM client (adjust URL and model as needed)
    client = TyravLLMClient(
        vllm_url="http://localhost:8000",
        model_name="meta-llama/Llama-2-7b-chat-hf"
    )
    
    await client.connect_to_tyra()
    
    print("vLLM + Tyra Memory Enhanced Chat")
    print("Type 'exit' to quit, 'stats' for memory stats, 'gaps' for knowledge gaps\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'stats':
            stats = await client.mcp_session.call_tool("get_memory_stats", {"agent_id": "vllm"})
            print(f"📊 Memory Stats: {stats.content[0].text}\n")
            continue
        elif user_input.lower() == 'gaps':
            gaps = await client.mcp_session.call_tool("detect_knowledge_gaps", {"agent_id": "vllm"})
            print(f"🧠 Knowledge Gaps: {gaps.content[0].text}\n")
            continue
            
        response = await client.chat_with_vllm_memory(user_input)
        print(f"vLLM: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Run vLLM + Tyra Integration**
```bash
# Terminal 1: Start Tyra MCP server
cd /path/to/tyra-mcp-memory-server
python main.py

# Terminal 2: Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000

# Terminal 3: Run the integration
python vllm_integration.py
```

**vLLM Configuration Options:**
```bash
# For GPU acceleration
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9

# For different models
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/DialoGPT-large \
  --max-model-len 2048

# With custom port
python -m vllm.entrypoints.openai.api_server \
  --model your-model \
  --port 8080
```

**Method 5: Simple REST API Integration**

**Quick REST API Setup:**
```bash
# Start Tyra MCP server
python main.py

# In another terminal, use the REST API directly
curl -X POST http://localhost:8000/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello from my local LLM!",
    "agent_id": "my_llm"
  }'

# Search memories
curl -X POST http://localhost:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "hello",
    "agent_id": "my_llm"
  }'
```

#### **Claude Code Integration (Step-by-Step)**

**Step 1: Verify MCP Server is Running**
```bash
# Start the MCP server
cd /path/to/tyra-mcp-memory-server
source venv/bin/activate
python main.py

# Should see: "Tyra Memory MCP Server ready with 16 tools"
```

**Step 2: Configure Claude Code**

Add this to your Claude Code MCP configuration file:

**Option A: Using Claude Code Settings UI**
1. Open Claude Code
2. Go to Settings → MCP Servers
3. Add new server with these details:
   - **Name**: `tyra-memory`
   - **Command**: `python`
   - **Args**: `["main.py"]`
   - **Working Directory**: `/path/to/tyra-mcp-memory-server`

**Option B: Direct Configuration File**
```json
{
  "mcpServers": {
    "tyra-memory": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/absolute/path/to/tyra-mcp-memory-server",
      "env": {
        "PYTHONPATH": "/absolute/path/to/tyra-mcp-memory-server"
      }
    }
  }
}
```

**Step 3: Test Connection**
1. Restart Claude Code
2. In a conversation, type: "What MCP tools are available?"
3. You should see 16 Tyra memory tools listed
4. Test with: "Store this memory: Hello from Claude!"

**Step 4: Verify All 16 Tools**
Ask Claude to list available tools - you should see:
- `store_memory` - Store content with AI enhancement
- `search_memory` - Advanced search with confidence scoring
- `delete_memory` - Remove specific memories
- `analyze_response` - Hallucination detection
- `deduplicate_memories` - Semantic deduplication
- `summarize_memories` - AI-powered summarization
- `detect_patterns` - Pattern recognition
- `analyze_temporal_evolution` - Concept evolution tracking
- `crawl_website` - Natural language web crawling
- `suggest_related_memories` - ML-powered suggestions
- `detect_memory_connections` - Connection discovery
- `recommend_memory_organization` - Structure optimization
- `detect_knowledge_gaps` - Gap analysis with learning paths
- `get_memory_stats` - System statistics
- `get_learning_insights` - Adaptive learning insights
- `health_check` - System health assessment

#### **Multi-Agent Setup**
```bash
# Start server with multi-agent support
python main.py --multi-agent

# Each agent gets isolated memory space:
# - claude.* namespace (for Claude)
# - tyra.* namespace (for Tyra AI)
# - archon.* namespace (for Archon)
```

#### **Troubleshooting Connection Issues**

**Issue: "No MCP tools found"**
```bash
# Check if server is running
ps aux | grep "python main.py"

# Check Python path
which python
python --version  # Should be 3.11+

# Check dependencies
pip list | grep mcp
```

**Issue: "Connection refused"**
```bash
# Check if server is binding correctly
python main.py --verbose

# Check for port conflicts
netstat -tulpn | grep python
```

**Issue: "Tools not loading"**
```bash
# Verify configuration file path
cat /path/to/claude-code/mcp-config.json

# Check working directory is correct
cd /path/to/tyra-mcp-memory-server
ls main.py  # Should exist
```

#### **Advanced Claude Code Configuration**

**With API Key Authentication:**
```json
{
  "mcpServers": {
    "tyra-memory": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/path/to/tyra-mcp-memory-server",
      "env": {
        "TYRA_API_KEY": "your-secure-api-key",
        "TYRA_ENVIRONMENT": "production"
      }
    }
  }
}
```

**With Custom Configuration:**
```json
{
  "mcpServers": {
    "tyra-memory": {
      "command": "python",
      "args": ["main.py", "--config", "custom-config.yaml"],
      "cwd": "/path/to/tyra-mcp-memory-server",
      "env": {
        "TYRA_AGENT_ID": "claude",
        "TYRA_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

---

## 🔧 Configuration

### **Main Configuration** (`config/config.yaml`)

```yaml
app:
  name: "Tyra MCP Memory Server"
  version: "3.0.0"
  environment: production

memory:
  backend: postgres
  vector_dimensions: 1024
  chunk_size: 512
  max_memories_per_agent: 1000000

databases:
  postgres:
    host: localhost
    port: 5432
    database: tyra_memory
    user: tyra
    
  redis:
    host: localhost
    port: 6379
    db: 0
    
  neo4j:
    uri: bolt://localhost:7687
    user: neo4j
    password: password

api:
  host: 0.0.0.0
  port: 8000
  cors_origins: ["*"]

dashboard:
  host: 0.0.0.0
  port: 8050
  update_interval: 1000
```

### **Complete Configuration Reference**

#### **Model Configuration** (`config/models.yaml`)

```yaml
embeddings:
  primary:
    model_name: "intfloat/e5-large-v2"
    model_path: "./models/embeddings/e5-large-v2"
    dimensions: 1024
    device: "auto"  # "auto", "cuda", "cpu"
    batch_size: 32
    max_seq_length: 512
    normalize_embeddings: true
    use_local_files: true  # CRITICAL: No automatic downloads
    
  fallback:
    model_name: "sentence-transformers/all-MiniLM-L12-v2"
    model_path: "./models/embeddings/all-MiniLM-L12-v2"
    dimensions: 384
    device: "cpu"
    batch_size: 16
    max_seq_length: 256
    normalize_embeddings: true
    use_local_files: true

reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  model_path: "./models/cross-encoders/ms-marco-MiniLM-L-6-v2"
  device: "auto"
  batch_size: 8
  max_seq_length: 512
  use_local_files: true
  
  # Reranking configuration
  threshold: 0.7
  top_k: 5
  use_cross_encoder: true
  fallback_to_bm25: true
```

#### **Provider Configuration** (`config/providers.yaml`)

```yaml
embeddings:
  primary:
    provider: "sentence_transformers"
    model_name: "intfloat/e5-large-v2"
    model_path: "./models/embeddings/e5-large-v2"
    use_local_files: true
    device: "auto"
    
cache:
  redis:
    host: localhost
    port: 6379
    db: 0
    password: null
    max_connections: 10
    
  # Multi-layer cache configuration
  l1:
    type: "memory"
    max_size: 1000
    ttl: 3600  # 1 hour
    
  l2:
    type: "redis"
    ttl_embeddings: 86400  # 24 hours
    ttl_search: 3600       # 1 hour
    ttl_rerank: 1800       # 30 minutes
    
  l3:
    type: "postgres"
    materialized_views: true
    refresh_interval: 3600
```

#### **Observability Configuration** (`config/observability.yaml`)

```yaml
otel:
  enabled: true
  service_name: "tyra-mcp-memory-server"
  trace_all_operations: true
  export_endpoint: "http://localhost:14268/api/traces"
  sample_rate: 1.0  # 100% sampling for development
  
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  structured: true
  format: "json"
  
metrics:
  enabled: true
  endpoint: "/metrics"
  port: 9090
  collect_interval: 60
  
tracing:
  enabled: true
  jaeger_endpoint: "http://localhost:14268"
  service_name: "tyra-memory"
```

#### **Agent Configuration** (`config/agents.yaml`)

```yaml
agents:
  claude:
    memory_limit: 1000000
    confidence_threshold: 0.8
    hallucination_detection: true
    auto_deduplication: true
    
  tyra:
    memory_limit: 5000000
    confidence_threshold: 0.9
    hallucination_detection: true
    auto_deduplication: true
    
  archon:
    memory_limit: 2000000
    confidence_threshold: 0.85
    hallucination_detection: true
    auto_deduplication: false
    
default_agent_config:
  memory_limit: 1000000
  confidence_threshold: 0.8
  hallucination_detection: true
  auto_deduplication: true
  search_strategy: "hybrid"
  max_context_length: 4096
```

#### **Graphiti Configuration** (`config/graphiti.yaml`)

```yaml
graphiti:
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "password"
    database: "neo4j"
    max_pool_size: 20
    
  temporal:
    enable_temporal_validity: true
    default_validity_duration: "1y"
    auto_expire_nodes: true
    
  entity_extraction:
    enabled: true
    confidence_threshold: 0.7
    extract_relationships: true
    
  causal_inference:
    enabled: true
    max_hop_distance: 3
    confidence_threshold: 0.8
```

#### **Self-Learning Configuration** (`config/self_learning.yaml`)

```yaml
self_learning:
  enabled: true
  analysis_interval: "1h"
  auto_optimize: true
  
  online_learning:
    enabled: true
    learning_rate: 0.01
    batch_size: 32
    memory_replay: true
    
  ab_testing:
    enabled: true
    test_duration: "7d"
    significance_threshold: 0.05
    min_sample_size: 100
    
  optimization:
    bayesian_optimization: true
    hyperparameter_tuning: true
    auto_scaling: true
    
  catastrophic_forgetting:
    prevention_enabled: true
    rehearsal_ratio: 0.1
    elastic_weight_consolidation: true
```

#### **Trading Configuration** (`config/trading.yaml`)

```yaml
trading:
  confidence_requirements:
    minimum_confidence: 0.95  # 95% required for all trading operations
    hallucination_detection: true
    mandatory_disclaimers: true
    audit_logging: true
    
  exchanges:
    binance:
      enabled: true
      rate_limits:
        requests_per_minute: 1200
        orders_per_second: 10
        
    coinbase:
      enabled: true
      rate_limits:
        requests_per_minute: 600
        
  data_sources:
    ccxt:
      enabled: true
      timeout: 30000
      
    websocket:
      enabled: true
      reconnect_attempts: 5
      
  sentiment_sources:
    fear_greed_index:
      enabled: true
      update_frequency: 3600  # 1 hour
      
    news_apis:
      enabled: true
      sources: ["coindesk", "cointelegraph"]
      
  risk_management:
    max_position_size: 10000
    stop_loss_required: true
    risk_reward_ratio: 2.0
```

#### **Environment Variables** (`.env`)

```bash
# Core application
TYRA_ENV=production
TYRA_LOG_LEVEL=INFO
TYRA_DEBUG=false

# Database connections
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tyra_memory
POSTGRES_USER=tyra
POSTGRES_PASSWORD=secure_password

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
API_CORS_ORIGINS=*
API_ENABLE_DOCS=true
API_ENABLE_AUTH=false

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050

# Model paths
EMBEDDING_MODEL_PATH=./models/embeddings/e5-large-v2
FALLBACK_MODEL_PATH=./models/embeddings/all-MiniLM-L12-v2
CROSS_ENCODER_PATH=./models/cross-encoders/ms-marco-MiniLM-L-6-v2

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key
ENCRYPTION_KEY=your-32-byte-encryption-key

# Observability
OTEL_ENABLED=true
OTEL_SERVICE_NAME=tyra-mcp-memory-server
JAEGER_ENDPOINT=http://localhost:14268

# File watcher
FILE_WATCHER_ENABLED=true
INGEST_DIRECTORY=./tyra-ingest/inbox

# Trading (if using trading features)
TRADING_CONFIDENCE_THRESHOLD=0.95
TRADING_AUDIT_LOGGING=true
```

#### **Complete Configuration Files Structure**

```
config/
├── config.yaml              # Main application configuration
├── providers.yaml           # Provider and cache settings
├── models.yaml             # AI model configurations
├── observability.yaml      # Monitoring and logging
├── agents.yaml             # Agent-specific settings
├── graphiti.yaml           # Neo4j and graph settings
├── self_learning.yaml      # Adaptive learning configuration
└── trading.yaml            # Trading data settings (if used)

.env                        # Environment variables
.env.local                  # Local overrides (gitignored)
.env.production            # Production settings
```

#### **Configuration Priority Order**

1. **Environment Variables** (highest priority)
2. **Command Line Arguments**
3. **Local Config Files** (`.env.local`, `config-local.yaml`)
4. **Default Config Files** (`config.yaml`, etc.)
5. **Built-in Defaults** (lowest priority)

#### **Configuration Validation**

```bash
# Validate configuration
python -c "
from src.core.utils.config import get_settings
settings = get_settings()
print('✅ Configuration valid')
print(f'Database: {settings.databases.postgres.host}')
print(f'Models: {settings.models.embeddings.primary.model_name}')
"

# Test database connections
python scripts/test_connections.py

# Validate model paths
python scripts/validate_models.py
```

---

## 🧪 Testing & Validation

### **Basic System Test**
```bash
# Run basic functionality test
python -c "
import asyncio
from src.mcp.server import TyraMemoryServer

async def test():
    server = TyraMemoryServer()
    await server._initialize_components()
    print('✅ All systems operational')
    
asyncio.run(test())
"
```

### **Full Test Suite**
```bash
# Run comprehensive tests
python -m pytest tests/ -v --cov=src

# Test specific components
python -m pytest tests/test_mcp_server.py -v
python -m pytest tests/test_suggestions.py -v
python -m pytest tests/test_dashboard.py -v
```

### **Performance Validation**
```bash
# Test embedding performance
python scripts/test_embedding_model.py

# Test memory operations
python scripts/test_memory_pipeline.py

# Load testing
python scripts/test_performance.py --users 50 --duration 300
```

---

## 🚨 Comprehensive Troubleshooting Guide

### **Installation Issues**

**1. Models not found / Model loading errors**
```bash
# ⚠️ CRITICAL: Models must be manually downloaded
# Check if models exist
ls -la ./models/embeddings/e5-large-v2/
ls -la ./models/embeddings/all-MiniLM-L12-v2/
ls -la ./models/cross-encoders/ms-marco-MiniLM-L-6-v2/

# If missing, download required models
huggingface-cli download intfloat/e5-large-v2 --local-dir ./models/embeddings/e5-large-v2 --local-dir-use-symlinks False
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./models/embeddings/all-MiniLM-L12-v2 --local-dir-use-symlinks False
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2 --local-dir-use-symlinks False

# Test model loading
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./models/embeddings/e5-large-v2')
print('✅ Model loads successfully')
"
```

**2. HuggingFace authentication issues**
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token in environment
export HUGGINGFACE_HUB_TOKEN=your_token_here

# For offline environments, use --local-files-only
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./models/embeddings/e5-large-v2', local_files_only=True)
"
```

**3. Git LFS issues**
```bash
# Install Git LFS
sudo apt install git-lfs
git lfs install

# Download LFS files manually
cd ./models/embeddings/e5-large-v2/
git lfs pull

# Alternative: Direct download without Git LFS
huggingface-cli download intfloat/e5-large-v2 --local-dir ./models/embeddings/e5-large-v2 --local-dir-use-symlinks False
```

### **Database Connection Issues**

**1. PostgreSQL connection failures**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
sudo journalctl -u postgresql -f

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
CREATE DATABASE tyra_memory;
CREATE USER tyra WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE tyra_memory TO tyra;
\q

# Test connection
psql -h localhost -U tyra -d tyra_memory -c "SELECT version();"

# Install pgvector extension
sudo -u postgres psql -d tyra_memory -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**2. Redis connection issues**
```bash
# Check Redis status
sudo systemctl status redis
redis-cli ping

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Check Redis configuration
sudo nano /etc/redis/redis.conf
# Ensure: bind 127.0.0.1 ::1
# Ensure: port 6379

# Restart Redis after config changes
sudo systemctl restart redis
```

**3. Neo4j connection problems**
```bash
# Check Neo4j status
sudo systemctl status neo4j

# Start Neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Check Neo4j logs
sudo journalctl -u neo4j -f

# Access Neo4j browser
# http://localhost:7474
# Default: neo4j/neo4j (change password on first login)

# Test connection
cypher-shell -u neo4j -p your_password "RETURN 'Hello Neo4j' as message;"
```

### **Runtime Errors**

**1. Port conflicts**
```bash
# Check what's using ports
sudo netstat -tulpn | grep :8000  # API server
sudo netstat -tulpn | grep :8050  # Dashboard
sudo netstat -tulpn | grep :5432  # PostgreSQL
sudo netstat -tulpn | grep :6379  # Redis
sudo netstat -tulpn | grep :7687  # Neo4j

# Kill conflicting processes
sudo pkill -f "port 8000"
sudo fuser -k 8000/tcp

# Change ports in configuration
# Edit config/config.yaml:
api:
  port: 8001  # Use different port
dashboard:
  port: 8051
```

**2. Memory and resource issues**
```bash
# Check system resources
free -h
df -h
htop

# Monitor GPU usage (if using GPU)
nvidia-smi
watch -n 1 nvidia-smi

# Reduce memory usage - edit config/models.yaml
embeddings:
  primary:
    model_name: "sentence-transformers/all-MiniLM-L12-v2"  # Smaller model
    batch_size: 8  # Reduce batch size
  
# Reduce cache sizes in config/config.yaml
cache:
  l1:
    max_size: 500  # Reduce from 1000
  redis:
    max_memory: "1gb"  # Reduce from 2gb
```

**3. Permission issues**
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./models/
sudo chmod -R 755 ./models/

# Fix database permissions
sudo -u postgres createuser -s $USER
sudo -u postgres psql -c "ALTER USER $USER CREATEDB;"

# Fix Redis permissions
sudo usermod -a -G redis $USER
```

### **Performance Issues**

**1. Slow embedding generation**
```bash
# Check if GPU is being used
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# Install CUDA support for faster inference
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Enable GPU in config/models.yaml
embeddings:
  primary:
    device: "cuda"  # Force GPU usage
    batch_size: 64   # Increase batch size for GPU
```

**2. Slow database queries**
```bash
# Check PostgreSQL performance
sudo -u postgres psql -d tyra_memory
\x
SELECT * FROM pg_stat_activity WHERE state = 'active';

# Check for missing indexes
EXPLAIN ANALYZE SELECT * FROM memories WHERE agent_id = 'claude';

# Optimize PostgreSQL settings
sudo nano /etc/postgresql/15/main/postgresql.conf
# Increase:
# shared_buffers = 256MB
# effective_cache_size = 1GB
# work_mem = 4MB

# Restart PostgreSQL
sudo systemctl restart postgresql
```

**3. High memory usage**
```bash
# Monitor memory usage by component
python -c "
import psutil
print(f'System RAM: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1e9:.1f}GB')
"

# Optimize memory settings
# In config/config.yaml:
memory:
  chunk_size: 256        # Smaller chunks
  max_memories_per_agent: 500000  # Reduce limit
  
# Enable memory cleanup
cache:
  cleanup_interval: 3600  # Clean every hour
  max_age: 86400         # Remove old entries after 24h
```

### **MCP Integration Issues**

**1. MCP tools not available in Claude Code**
```bash
# Verify MCP server is running
python main.py
# Should show: "Tyra Memory MCP Server ready with 16 tools"

# Check MCP configuration in Claude Code
# File: ~/.config/claude-code/mcp_servers.json
{
  "mcpServers": {
    "tyra-memory": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/absolute/path/to/tyra-mcp-memory-server"
    }
  }
}

# Restart Claude Code after configuration changes
```

**2. MCP server crashes**
```bash
# Run with debug logging
TYRA_LOG_LEVEL=DEBUG python main.py

# Check for Python environment issues
which python
python --version  # Should be 3.11+
pip list | grep mcp

# Check dependency conflicts
pip check
```

### **Trading Data API Issues**

**1. Trading endpoints returning errors**
```bash
# Check if trading schema is installed
psql -h localhost -U tyra -d tyra_memory -c "
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name LIKE 'trading_%';
"

# Install trading schema if missing
psql -h localhost -U tyra -d tyra_memory -f src/migrations/sql/003_trading_data_schema.sql

# Test trading endpoint
curl -X GET http://localhost:8000/v1/trading/health
```

**2. High confidence requirements for trading**
```bash
# Trading operations require 95%+ confidence
# Check confidence scoring:
curl -X POST http://localhost:8000/v1/rag/check-hallucination \
  -H "Content-Type: application/json" \
  -d '{"text": "Buy Bitcoin at current price", "context": "Market analysis"}'

# Should return confidence score >= 0.95 for trading operations
```

### **Performance Optimization Strategies**

**1. GPU Acceleration (Recommended)**
```bash
# Install CUDA support for 10x faster embeddings
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU setup
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('Running on CPU - consider GPU for better performance')
"

# Enable GPU in configuration
# config/models.yaml:
embeddings:
  primary:
    device: "cuda"     # Use GPU
    batch_size: 64     # Larger batches on GPU
  fallback:
    device: "cuda"     # Use GPU for fallback too
    batch_size: 32
```

**2. Memory Optimization**
```yaml
# config/config.yaml - For systems with limited RAM
memory:
  chunk_size: 256              # Smaller chunks (default: 512)
  max_memories_per_agent: 500000  # Reduce if needed (default: 1M)
  vector_dimensions: 384       # Use smaller embeddings

cache:
  l1:
    max_size: 500             # Reduce in-memory cache
  redis:
    max_memory: "1gb"         # Reduce Redis memory
    ttl_embeddings: 3600      # 1 hour (reduce from 24h)
    
# Use smaller embedding model
embeddings:
  primary:
    model_name: "sentence-transformers/all-MiniLM-L12-v2"
    dimensions: 384
    batch_size: 16
```

**3. Database Optimization**
```sql
-- PostgreSQL performance tuning
-- /etc/postgresql/15/main/postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

-- Create additional indexes for better performance
CREATE INDEX CONCURRENTLY idx_memories_agent_created 
  ON memories(agent_id, created_at DESC);
CREATE INDEX CONCURRENTLY idx_memories_content_gin 
  ON memories USING gin(to_tsvector('english', content));
```

**4. Redis Optimization**
```bash
# /etc/redis/redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Restart Redis
sudo systemctl restart redis
```

**5. Application-Level Optimization**
```yaml
# config/config.yaml
api:
  workers: 4              # Increase for more concurrent requests
  max_connections: 100    # Database connection pool
  timeout: 30             # Request timeout
  
observability:
  enabled: false          # Disable in production for better performance
  
file_watcher:
  batch_size: 10          # Process files in batches
  check_interval: 30      # Check less frequently
```

### **Monitoring and Diagnostics**

**1. Health Monitoring**
```bash
# Check system health
curl http://localhost:8000/v1/health/detailed

# Monitor performance metrics
curl http://localhost:8000/v1/observability/metrics

# Check component status
curl http://localhost:8000/v1/health/components
```

**2. Log Analysis**
```bash
# Monitor application logs
tail -f logs/tyra-memory.log

# Check for errors
grep -i error logs/tyra-memory.log | tail -20

# Monitor database performance
sudo -u postgres tail -f /var/log/postgresql/postgresql-15-main.log
```

**3. Performance Profiling**
```bash
# Profile memory usage
python -m memory_profiler scripts/profile_memory.py

# Profile query performance
python scripts/benchmark_queries.py

# Load testing
python scripts/load_test.py --users 50 --duration 300
```

---

## 📚 Additional Resources

### **Documentation**
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture details
- [ADVANCED_ENHANCEMENTS.md](ADVANCED_ENHANCEMENTS.md) - Feature specifications
- [API.md](API.md) - Complete API reference
- [CONFIGURATION.md](CONFIGURATION.md) - Advanced configuration guide

### **Examples**
- [examples/n8n-workflows/](examples/n8n-workflows/) - n8n integration examples
- [examples/python-client/](examples/python-client/) - Python client usage
- [examples/api-integration/](examples/api-integration/) - REST API examples

### **Advanced Troubleshooting**

**1. Debug Mode**
```bash
# Run with maximum debugging
export TYRA_LOG_LEVEL=DEBUG
export TYRA_DEBUG=true
python main.py --verbose

# Enable SQL query logging
export TYRA_LOG_SQL=true

# Enable detailed tracing
export OTEL_LOG_LEVEL=debug
```

**2. Clean Reset (Nuclear Option)**
```bash
# WARNING: This will delete all data

# Stop all services
sudo systemctl stop postgresql redis neo4j

# Drop and recreate database
sudo -u postgres dropdb tyra_memory
sudo -u postgres createdb tyra_memory

# Clear Redis
redis-cli FLUSHALL

# Clear Neo4j
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;"

# Restart services
sudo systemctl start postgresql redis neo4j

# Reinitialize
python scripts/init_postgres.py
python scripts/init_neo4j.py
```

**3. Environment Validation Script**
```bash
# Create comprehensive validation script
cat > validate_environment.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
import subprocess
import importlib

def check_python():
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.11+")
        return False

def check_models():
    models = [
        "./models/embeddings/e5-large-v2",
        "./models/embeddings/all-MiniLM-L12-v2",
        "./models/cross-encoders/ms-marco-MiniLM-L-6-v2"
    ]
    
    all_good = True
    for model_path in models:
        if os.path.exists(model_path):
            print(f"✅ Model found: {model_path}")
        else:
            print(f"❌ Model missing: {model_path}")
            all_good = False
    return all_good

def check_services():
    services = ["postgresql", "redis", "neo4j"]
    all_good = True
    
    for service in services:
        try:
            result = subprocess.run(["systemctl", "is-active", service], 
                                  capture_output=True, text=True)
            if result.stdout.strip() == "active":
                print(f"✅ {service} is running")
            else:
                print(f"❌ {service} is not running")
                all_good = False
        except:
            print(f"❌ Could not check {service}")
            all_good = False
    return all_good

def check_dependencies():
    deps = ["torch", "sentence_transformers", "asyncpg", "redis", "neo4j"]
    all_good = True
    
    for dep in deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep} installed")
        except ImportError:
            print(f"❌ {dep} not installed")
            all_good = False
    return all_good

if __name__ == "__main__":
    print("Tyra MCP Memory Server Environment Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python),
        ("Dependencies", check_dependencies),
        ("Model Files", check_models),
        ("System Services", check_services)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All checks passed! System ready.")
    else:
        print("❌ Some checks failed. See above for details.")
        sys.exit(1)
EOF

# Run validation
python validate_environment.py
```

### **Getting Help**

**1. Diagnostic Information**
```bash
# Collect system information for support
echo "=== System Info ===" > debug_info.txt
uname -a >> debug_info.txt
python --version >> debug_info.txt
pip list >> debug_info.txt

echo "\n=== Service Status ===" >> debug_info.txt
systemctl status postgresql redis neo4j >> debug_info.txt

echo "\n=== Recent Logs ===" >> debug_info.txt
tail -50 logs/tyra-memory.log >> debug_info.txt

echo "\n=== Configuration ===" >> debug_info.txt
cat config/config.yaml >> debug_info.txt
```

**2. Support Channels**
- **Issues**: GitHub Issues (include debug_info.txt)
- **Discussions**: GitHub Discussions
- **Documentation**: Built-in help at `http://localhost:8000/docs`
- **API Reference**: [docs/API.md](docs/API.md)
- **Health Check**: `curl http://localhost:8000/v1/health/detailed`

---

## 🎉 What's Next?

After successful installation, you can:

1. **Start with basic memory operations** using MCP tools
2. **Explore the dashboard** for visual analytics
3. **Try the suggestions system** for intelligent recommendations
4. **Set up automated file ingestion** for document processing
5. **Integrate with n8n** for workflow automation
6. **Customize configurations** for your specific use case

**The system is ready for production use with all 16 MCP tools, comprehensive REST API, interactive dashboard, and intelligent suggestions system fully operational!**