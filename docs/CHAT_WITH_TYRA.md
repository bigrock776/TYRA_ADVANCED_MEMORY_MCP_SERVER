# Chatting with Tyra Through the MCP Memory Server

## Overview
Once you have the Tyra MCP Memory Server connected, you can chat with Tyra through multiple interfaces. The system provides both REST API endpoints and MCP tools with memory integration, context awareness, and trading safety features.

## Methods to Chat with Tyra

### 1. **REST API Chat Endpoints**

The server provides several chat endpoints accessible via HTTP:

#### Standard Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the current market conditions for AAPL?"}
    ],
    "model": "balanced",
    "temperature": 0.7,
    "use_memory": true,
    "memory_limit": 10,
    "agent_id": "tyra"
  }'
```

#### Trading-Specific Chat (High Confidence)
For trading-related queries, use the specialized endpoint with 95% confidence requirements:

```bash
curl -X POST http://localhost:8000/v1/chat/trading \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Should I buy AAPL at the current price?"}
    ],
    "model": "accurate",
    "temperature": 0.3,
    "use_memory": true,
    "memory_limit": 20,
    "agent_id": "tyra",
    "confirm_high_confidence": true
  }'
```

#### Streaming Chat
For real-time responses:

```bash
curl -X POST http://localhost:8000/v1/chat/completions/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Analyze the tech sector trends"}
    ],
    "stream": true,
    "agent_id": "tyra"
  }'
```

### 2. **Python Client Integration**

```python
import aiohttp
import asyncio
import json

class TyraChatClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.agent_id = "tyra"
    
    async def chat(self, message, use_trading_mode=False):
        """Chat with Tyra using the memory-enhanced API."""
        
        endpoint = "/v1/chat/trading" if use_trading_mode else "/v1/chat/completions"
        
        payload = {
            "messages": [{"role": "user", "content": message}],
            "agent_id": self.agent_id,
            "use_memory": True,
            "memory_limit": 20 if use_trading_mode else 10,
            "model": "accurate" if use_trading_mode else "balanced"
        }
        
        if use_trading_mode:
            payload["confirm_high_confidence"] = True
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}{endpoint}",
                json=payload
            ) as response:
                result = await response.json()
                return result

# Usage
async def main():
    client = TyraChatClient()
    
    # Regular chat
    response = await client.chat("What's the market outlook for tech stocks?")
    print(f"Tyra: {response['message']['content']}")
    print(f"Confidence: {response['confidence']}%")
    
    # Trading mode (high confidence)
    trading_response = await client.chat(
        "Should I enter a long position on AAPL?", 
        use_trading_mode=True
    )
    print(f"\nTrading Analysis:")
    print(f"Response: {trading_response['message']['content']}")
    print(f"Trading Approved: {trading_response['trading_approved']}")
    print(f"Confidence: {trading_response['confidence']}%")
    print(f"Warnings: {trading_response.get('warnings', [])}")

asyncio.run(main())
```

### 3. **WebSocket Real-time Chat**

```javascript
// Connect to WebSocket for real-time chat
const ws = new WebSocket('ws://localhost:8001/v1/ws/chat?agent=tyra');

ws.onopen = () => {
    console.log('Connected to Tyra chat');
    
    // Send a message
    ws.send(JSON.stringify({
        type: 'message',
        content: 'What are the key support levels for SPY?',
        use_memory: true,
        require_high_confidence: false
    }));
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log('Tyra:', response.content);
    console.log('Confidence:', response.confidence);
    
    // Handle memory-enhanced responses
    if (response.memories_used) {
        console.log('Based on', response.memories_used.length, 'memories');
    }
};
```

### 4. **Through MCP Tools in Claude**

When using Claude with the MCP server connected:

```python
# Claude can use MCP tools to interact with Tyra's memory
# Example: Store a conversation with Tyra
result = await mcp_client.call("store_memory", {
    "content": "Tyra analyzed AAPL and identified bullish momentum with RSI at 65",
    "agent_id": "tyra",
    "metadata": {
        "type": "trading_analysis",
        "symbol": "AAPL",
        "confidence": 87
    }
})

# Search Tyra's memories for context
memories = await mcp_client.call("search_memory", {
    "query": "AAPL technical analysis momentum",
    "agent_id": "tyra",
    "min_confidence": 0.8
})
```

### 5. **Interactive CLI Chat**

Create a simple CLI interface:

```python
#!/usr/bin/env python3
import asyncio
import aiohttp
from datetime import datetime

class TyraInteractiveChat:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.session_messages = []
        self.agent_id = "tyra"
    
    async def chat_loop(self):
        print("🤖 Tyra Trading Assistant")
        print("=" * 50)
        print("Commands:")
        print("  /trading - Switch to high-confidence trading mode")
        print("  /normal  - Switch to normal chat mode")
        print("  /memory  - Show memory usage")
        print("  /quit    - Exit")
        print("=" * 50)
        
        trading_mode = False
        
        while True:
            try:
                user_input = input(f"\n{'[TRADING]' if trading_mode else '[CHAT]'} You: ")
                
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        break
                    elif user_input == '/trading':
                        trading_mode = True
                        print("✅ Switched to trading mode (95% confidence required)")
                    elif user_input == '/normal':
                        trading_mode = False
                        print("✅ Switched to normal chat mode")
                    elif user_input == '/memory':
                        await self.show_memory_stats()
                    continue
                
                # Add message to session
                self.session_messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Get response
                response = await self.get_chat_response(trading_mode)
                
                # Display response
                print(f"\n🤖 Tyra: {response['content']}")
                print(f"   Confidence: {response['confidence']:.1f}%")
                
                if trading_mode:
                    print(f"   Trading Approved: {'✅' if response.get('trading_approved') else '❌'}")
                    if response.get('warnings'):
                        print(f"   ⚠️  Warnings: {', '.join(response['warnings'])}")
                
                # Add to session
                self.session_messages.append({
                    "role": "assistant",
                    "content": response['content']
                })
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def get_chat_response(self, trading_mode=False):
        """Get response from Tyra."""
        endpoint = "/v1/chat/trading" if trading_mode else "/v1/chat/completions"
        
        payload = {
            "messages": self.session_messages[-10:],  # Last 10 messages for context
            "agent_id": self.agent_id,
            "use_memory": True,
            "memory_limit": 20 if trading_mode else 10,
            "model": "accurate" if trading_mode else "balanced",
            "temperature": 0.3 if trading_mode else 0.7
        }
        
        if trading_mode:
            payload["confirm_high_confidence"] = True
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}{endpoint}",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if trading_mode:
                        return {
                            "content": result["message"]["content"],
                            "confidence": result["confidence"],
                            "trading_approved": result["trading_approved"],
                            "warnings": result.get("warnings", [])
                        }
                    else:
                        return {
                            "content": result["message"]["content"],
                            "confidence": result["confidence"]
                        }
                else:
                    raise Exception(f"API error: {response.status}")
    
    async def show_memory_stats(self):
        """Show memory usage statistics."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/memory/stats",
                params={"agent_id": self.agent_id}
            ) as response:
                if response.status == 200:
                    stats = await response.json()
                    print(f"\n📊 Memory Stats for Tyra:")
                    print(f"   Total Memories: {stats.get('total_memories', 0)}")
                    print(f"   Trading Memories: {stats.get('trading_memories', 0)}")
                    print(f"   Last Updated: {stats.get('last_updated', 'N/A')}")

# Run the interactive chat
if __name__ == "__main__":
    chat = TyraInteractiveChat()
    asyncio.run(chat.chat_loop())
```

### 6. **n8n Workflow Integration**

Create an n8n workflow to chat with Tyra:

```json
{
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "tyra-chat",
        "responseMode": "lastNode",
        "options": {}
      }
    },
    {
      "name": "Chat with Tyra",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8000/v1/chat/completions",
        "options": {},
        "bodyParametersUi": {
          "parameter": [
            {
              "name": "messages",
              "value": "=[{\"role\":\"user\",\"content\":\"{{$json.query}}\"}]"
            },
            {
              "name": "agent_id",
              "value": "tyra"
            },
            {
              "name": "use_memory",
              "value": true
            }
          ]
        }
      }
    }
  ]
}
```

## Chat Configuration Options

### Model Selection
- **fast**: Quick responses, lower accuracy (8B model)
- **balanced**: Good balance of speed and accuracy (40B model)
- **accurate**: Highest accuracy, required for trading (70B model)

### Temperature Settings
- **0.0-0.3**: Deterministic, consistent (recommended for trading)
- **0.4-0.7**: Balanced creativity (default for general chat)
- **0.8-1.0**: Creative, varied responses

### Memory Integration
- **use_memory**: Enable/disable memory context
- **memory_limit**: Number of memories to retrieve (1-50)
- **agent_id**: Always use "tyra" for Tyra-specific context

## Trading Mode Requirements

When chatting about trading decisions:

1. **Use the `/v1/chat/trading` endpoint**
2. **Confidence must be ≥95% for approval**
3. **Lower temperature (≤0.5) is enforced**
4. **More memory context is used (20+ memories)**
5. **Hallucination detection is stricter**
6. **All decisions are logged for audit**

## Best Practices

### 1. Context Building
```python
# First, store relevant context
await store_memory("AAPL showing bullish divergence on daily chart", agent_id="tyra")
await store_memory("Tech sector momentum increasing this week", agent_id="tyra")

# Then chat with context awareness
response = await chat("What's your view on AAPL?", use_memory=True)
```

### 2. Session Management
- Create conversation sessions for multi-turn chats
- Maintain context across messages
- Use conversation IDs for tracking

### 3. Error Handling
```python
try:
    response = await chat_with_tyra(message)
except Exception as e:
    if "confidence" in str(e):
        print("Confidence too low for trading decision")
    else:
        print(f"Chat error: {e}")
```

### 4. Monitoring Confidence
Always check confidence levels, especially for trading:
```python
if response['confidence'] >= 95 and response['trading_approved']:
    # Safe to consider for trading
    execute_trade(response['recommendation'])
else:
    # Need human review
    flag_for_review(response)
```

## Advanced Features

### 1. Multi-Agent Conversations
```python
# Tyra can reference memories from other agents
response = await chat(
    "What did Claude say about the market yesterday?",
    cross_agent_search=True
)
```

### 2. Streaming Responses
```python
async for chunk in stream_chat("Analyze the forex market"):
    print(chunk['content'], end='', flush=True)
```

### 3. Voice Integration
Connect Tyra to voice interfaces using the chat API with speech-to-text and text-to-speech services.

## Troubleshooting

### Common Issues

1. **"LLM service unavailable"**
   - Ensure your local LLM (vLLM/Ollama) is running
   - Check the LLM endpoint configuration

2. **"Confidence below threshold"**
   - For trading queries, ensure you're using the trading endpoint
   - Provide more context in your query
   - Check if relevant memories exist

3. **"No memories found"**
   - Verify agent_id is set to "tyra"
   - Check if memories have been stored
   - Ensure memory search is enabled

4. **Slow responses**
   - Check LLM model size and GPU availability
   - Consider using the "fast" model for non-trading queries
   - Enable response caching

## Example Trading Conversation

```python
# Start a trading analysis session
chat = TyraChatClient()

# Build context
await chat.store_context("Market opened with bullish sentiment")
await chat.store_context("Fed meeting notes were dovish")

# Ask for analysis
response = await chat.chat(
    "Given the current market conditions, should I increase my tech allocation?",
    use_trading_mode=True
)

print(f"Tyra's Analysis: {response['message']['content']}")
print(f"Confidence: {response['confidence']}%")
print(f"Safe for Trading: {response['trading_approved']}")

# Follow-up with specific stock
follow_up = await chat.chat(
    "What about NVDA specifically? Entry points?",
    use_trading_mode=True
)

print(f"\nNVDA Analysis: {follow_up['message']['content']}")
```

This completes the setup for chatting with Tyra through the MCP Memory Server!