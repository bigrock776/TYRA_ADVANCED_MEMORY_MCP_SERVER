# Local LLM Setup Guide for Tyra MCP Memory Server

## Overview
The Tyra MCP Memory Server supports multiple local LLM providers through an OpenAI-compatible API interface. This guide covers the best ways to connect your local LLM.

## Supported LLM Providers

### 1. vLLM (Recommended for Production)
vLLM provides the best performance for production deployments with features like continuous batching, PagedAttention, and quantization support.

#### Installation
```bash
pip install vllm
```

#### Starting vLLM Server
```bash
# For smaller models (8B parameters)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 4096

# For larger models with quantization
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --quantization awq \
  --gpu-memory-utilization 0.95
```

### 2. Ollama (Easiest Setup)
Ollama provides a simple way to run local models with automatic model management.

#### Installation
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama
```

#### Starting Ollama
```bash
# Start Ollama service
ollama serve

# Pull a model
ollama pull llama3.1:8b
ollama pull mixtral:8x7b
```

### 3. Text Generation WebUI (Most Features)
Provides a web interface with extensive customization options.

#### Installation
```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
python -m pip install -r requirements.txt
```

#### Starting with API
```bash
python server.py --api --api-port 8000 --model llama-3.1-8b
```

### 4. LM Studio (User-Friendly GUI)
Desktop application with built-in model management and API server.

1. Download from: https://lmstudio.ai/
2. Download your preferred model
3. Start the local server (usually on port 1234)

## Configuration

### 1. Environment Variables
Add these to your `.env` file:

```bash
# vLLM Configuration
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
VLLM_API_KEY=dummy-key  # Not needed for local deployment

# For Ollama (uses different port)
# VLLM_BASE_URL=http://localhost:11434/v1
# VLLM_MODEL_NAME=llama3.1:8b

# For LM Studio
# VLLM_BASE_URL=http://localhost:1234/v1
# VLLM_MODEL_NAME=local-model

# For Text Generation WebUI
# VLLM_BASE_URL=http://localhost:5000/v1
# VLLM_MODEL_NAME=llama-3.1-8b
```

### 2. Configuration Files
Update the following configuration files:

#### config/config.yaml
```yaml
# RAG Configuration
rag:
  reranking:
    vllm:
      enabled: true
      base_url: ${VLLM_BASE_URL:-http://localhost:8000/v1}
      model_name: ${VLLM_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
      api_key: ${VLLM_API_KEY:-dummy-key}
      max_tokens: 512
      temperature: 0.1
      timeout: 30
```

#### config/providers.yaml
```yaml
# Graphiti LLM Configuration
graphiti:
  llm_provider: "openai"  # Keep as openai for compatibility
  llm_base_url: ${VLLM_BASE_URL:-http://localhost:8000/v1}
  llm_model: ${VLLM_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
  llm_api_key: ${VLLM_API_KEY:-dummy-key}
  llm_temperature: 0.0
  llm_max_tokens: 1000
```

#### config/agents.yaml
```yaml
# Pydantic AI Agent Configuration
pydantic_ai:
  model_provider: "openai"  # Keep for compatibility
  base_url: ${VLLM_BASE_URL:-http://localhost:8000/v1}
  model_name: ${VLLM_MODEL_NAME:-llama2}
  api_key: ${VLLM_API_KEY:-dummy-key}
  temperature: 0.0
  max_retries: 3
  retry_delay: 1.0
```

## Model Recommendations

### For Best Performance (vLLM)
- **Small/Fast**: `meta-llama/Llama-3.1-8B-Instruct` (16GB VRAM)
- **Balanced**: `mistralai/Mixtral-8x7B-Instruct-v0.1` (48GB VRAM)
- **Large/Accurate**: `meta-llama/Llama-3.1-70B-Instruct` (140GB VRAM, use quantization)

### For Easy Setup (Ollama)
- **Small/Fast**: `llama3.1:8b` or `mistral:7b`
- **Balanced**: `mixtral:8x7b` or `llama3.1:40b`
- **Large/Accurate**: `llama3.1:70b`

### For Limited Resources
- **CPU-friendly**: `phi3:mini` (3.8B parameters)
- **Low VRAM**: `gemma2:2b` or `qwen2:1.5b`
- **Quantized**: Any model with Q4_K_M or Q5_K_M quantization

## Testing Your Connection

### 1. Check LLM Server
```bash
# For vLLM/Ollama/LM Studio
curl http://localhost:8000/v1/models

# Should return available models
```

### 2. Test Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
  }'
```

### 3. Test via Tyra API
```bash
# Start Tyra servers
python main.py  # In one terminal
python -m src.mcp.server  # In another terminal

# Test chat endpoint
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of France?",
    "agent_id": "test-agent",
    "mode": "fast"
  }'
```

## Integration Points

The local LLM is used in several key areas:

1. **Memory Synthesis**
   - Summarization of memory clusters
   - Pattern detection and analysis
   - Temporal evolution tracking

2. **Entity & Relationship Extraction**
   - Structured entity extraction via Pydantic AI
   - Relationship inference between entities
   - Confidence scoring

3. **Query Processing**
   - Intent classification
   - Query expansion and refinement
   - Semantic understanding

4. **Response Validation**
   - Hallucination detection
   - Source grounding verification
   - Confidence assessment

5. **Chat Interface**
   - Direct chat with memory-augmented responses
   - Trading mode with high-confidence requirements
   - Multi-turn conversation support

## Performance Optimization

### 1. Model Selection
- Use smaller models (8B) for speed-critical operations
- Use larger models (70B) for accuracy-critical tasks
- Consider quantization for larger models

### 2. Caching Strategy
- LLM responses are cached in Redis
- Cache TTL can be configured per operation type
- Semantic deduplication prevents redundant calls

### 3. Batch Processing
- The system supports batch inference
- Configure batch sizes based on your GPU memory
- Use streaming for real-time responses

### 4. Resource Management
```yaml
# Recommended settings for different GPUs
# RTX 3090/4090 (24GB VRAM)
--max-model-len 4096 --gpu-memory-utilization 0.9

# A100 (40GB VRAM)
--max-model-len 8192 --gpu-memory-utilization 0.95

# Multiple GPUs
--tensor-parallel-size 2 --pipeline-parallel-size 2
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure LLM server is running
   - Check firewall settings
   - Verify port configuration

2. **Out of Memory**
   - Use smaller models or quantization
   - Reduce max sequence length
   - Lower batch size

3. **Slow Response Times**
   - Enable GPU acceleration
   - Use vLLM for better performance
   - Check cache configuration

4. **Model Not Found**
   - Verify model name matches exactly
   - Ensure model is downloaded
   - Check API compatibility

### Debug Mode
Enable debug logging for LLM operations:
```bash
export LOG_LEVEL=DEBUG
export DEBUG_RAG=true
```

## Security Considerations

1. **Local-Only Operation**
   - All LLM calls stay within your network
   - No data leaves your infrastructure
   - Complete privacy and control

2. **API Key Management**
   - Use "dummy-key" for local deployments
   - Real keys only needed for cloud providers
   - Keys stored in environment variables

3. **Network Security**
   - Bind to localhost only for single-machine setups
   - Use VPN or private network for distributed setups
   - Enable TLS for production deployments

## Next Steps

1. Choose your LLM provider based on your requirements
2. Install and start the LLM server
3. Update configuration files
4. Test the connection
5. Monitor performance and adjust settings

For production deployments, we recommend vLLM with a quantized 70B model for the best balance of performance and quality.