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

All dependencies are in `requirements.txt`:

```bash
# Core packages
fastapi>=0.104.0
pydantic>=2.5.0
asyncpg>=0.29.0
redis>=5.0.0
neo4j>=5.15.0

# AI/ML packages  
sentence-transformers>=2.7.0
scikit-learn>=1.3.0
spacy>=3.7.2
numpy>=1.24.0

# Dashboard & visualization
dash>=2.15.0
plotly>=5.17.0
streamlit>=1.28.0

# MCP framework
mcp>=1.0.0
pydantic-ai>=0.0.13
```

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

#### **Step 4: Model Download**

```bash
# Required models (local operation)
mkdir -p models/embeddings models/cross-encoders

# Primary embedding model
huggingface-cli download intfloat/e5-large-v2 --local-dir ./models/embeddings/e5-large-v2

# Fallback embedding model  
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./models/embeddings/all-MiniLM-L12-v2

# Cross-encoder for reranking
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir ./models/cross-encoders/ms-marco-MiniLM-L-6-v2
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

Access via HTTP endpoints:

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

### **Model Configuration** (`config/models.yaml`)

```yaml
embeddings:
  primary:
    model_name: "intfloat/e5-large-v2"
    model_path: "./models/embeddings/e5-large-v2"
    dimensions: 1024
    device: "auto"
    
  fallback:
    model_name: "sentence-transformers/all-MiniLM-L12-v2"
    model_path: "./models/embeddings/all-MiniLM-L12-v2"
    dimensions: 384
    device: "cpu"

reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  model_path: "./models/cross-encoders/ms-marco-MiniLM-L-6-v2"
  device: "auto"
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

## 🚨 Troubleshooting

### **Common Issues**

**1. Models not found**
```bash
# Download required models
huggingface-cli download intfloat/e5-large-v2 --local-dir ./models/embeddings/e5-large-v2
huggingface-cli download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./models/embeddings/all-MiniLM-L12-v2
```

**2. Database connection errors**
```bash
# Check service status
sudo systemctl status postgresql redis neo4j

# Restart services
sudo systemctl restart postgresql redis neo4j

# Check connections
psql -h localhost -U tyra -d tyra_memory -c "SELECT 1;"
redis-cli ping
```

**3. Port conflicts**
```bash
# Check what's using ports
sudo netstat -tlnp | grep :8000
sudo netstat -tlnp | grep :8050

# Kill conflicting processes
sudo pkill -f "port 8000"
```

**4. Memory issues**
```bash
# Check memory usage
free -h
htop

# Reduce model size in config/models.yaml
# Use fallback model only:
embeddings:
  primary:
    model_name: "sentence-transformers/all-MiniLM-L12-v2"
    model_path: "./models/embeddings/all-MiniLM-L12-v2"
    dimensions: 384
```

### **Performance Optimization**

**GPU Acceleration**
```bash
# Install CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU usage
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Memory Optimization**
```yaml
# config/config.yaml
memory:
  chunk_size: 256  # Smaller chunks
  max_memories_per_agent: 500000  # Reduce if needed

cache:
  redis:
    max_memory: "2gb"
    ttl_embeddings: 3600  # 1 hour
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

### **Support**
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: Built-in help at `/docs` endpoint

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