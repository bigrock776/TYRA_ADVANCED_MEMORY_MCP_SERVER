# CCXT MCP Integration Guide

## Overview
This guide shows how to integrate the Tyra MCP Memory Server with CCXT MCP for comprehensive trading data ingestion. The system automatically stores OHLCV data, sentiment analysis, and trading information into both PostgreSQL and the memory system for AI analysis.

## Prerequisites

1. **Database Setup**
   ```bash
   # Run the trading data migration
   python scripts/run_migration.py 003_trading_data_schema.sql
   ```

2. **CCXT MCP Server** (if using external CCXT MCP)
   - Install and configure CCXT MCP server
   - Or use the built-in CCXT integration endpoints

3. **API Keys** (for data sources)
   - Exchange API keys for position tracking
   - News API keys for sentiment data
   - Social media API keys (optional)

## Table Structure Created

The migration creates 18 comprehensive tables:

### **Core Trading Tables**
- `trading_exchanges` - Exchange definitions
- `trading_instruments` - Symbol/pair definitions  
- `ohlcv_data` - Price and volume data
- `technical_indicators` - Calculated indicators

### **Sentiment & News**
- `sentiment_sources` - Data source definitions
- `sentiment_data` - Sentiment analysis results
- `market_news` - News articles with sentiment

### **Portfolio & Positions**
- `trading_accounts` - Portfolio accounts
- `trading_positions` - Open/closed positions
- `trading_orders` - Order history
- `trading_signals` - AI-generated signals
- `trading_metrics` - Performance analytics

### **Monitoring**
- `data_ingestion_logs` - Ingestion tracking
- `data_quality_metrics` - Data quality monitoring

## Integration Methods

### 1. **Direct API Integration**

#### OHLCV Data Ingestion
```python
import aiohttp
import asyncio
from datetime import datetime
from decimal import Decimal

async def send_ohlcv_data_to_tyra(exchange_code, symbol, ohlcv_data):
    """Send OHLCV data to Tyra memory server."""
    
    # Format data for Tyra API
    formatted_data = []
    for candle in ohlcv_data:
        formatted_data.append({
            "timestamp": datetime.fromtimestamp(candle[0] / 1000).isoformat(),
            "timeframe": "1h",
            "open_price": str(Decimal(str(candle[1]))),
            "high_price": str(Decimal(str(candle[2]))),
            "low_price": str(Decimal(str(candle[3]))),
            "close_price": str(Decimal(str(candle[4]))),
            "volume": str(Decimal(str(candle[5])))
        })
    
    payload = {
        "exchange_code": exchange_code,
        "symbol": symbol,
        "data": formatted_data,
        "data_source": "ccxt"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/v1/trading/ohlcv/batch",
            json=payload
        ) as response:
            result = await response.json()
            print(f"OHLCV ingestion result: {result}")
            return result

# Example usage
ohlcv_data = [
    [1640995200000, 47000.5, 47200.0, 46800.0, 47100.0, 125.5],  # [timestamp, O, H, L, C, V]
    [1640998800000, 47100.0, 47300.0, 46900.0, 47150.0, 98.2]
]

await send_ohlcv_data_to_tyra("BINANCE", "BTC/USDT", ohlcv_data)
```

#### Sentiment Data Ingestion
```python
async def send_sentiment_data_to_tyra(source_name, symbol, sentiment_data):
    """Send sentiment analysis data to Tyra."""
    
    formatted_data = []
    for item in sentiment_data:
        formatted_data.append({
            "timestamp": item["timestamp"],
            "sentiment_score": str(Decimal(str(item["score"]))),
            "sentiment_label": item["label"],  # bullish, bearish, neutral
            "confidence": item["confidence"],
            "volume_mentions": item.get("mentions", 0),
            "keywords": item.get("keywords", [])
        })
    
    payload = {
        "source_name": source_name,
        "symbol": symbol,
        "timeframe": "1h",
        "data": formatted_data
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/v1/trading/sentiment/batch",
            json=payload
        ) as response:
            result = await response.json()
            return result

# Example usage
sentiment_data = [
    {
        "timestamp": "2024-01-01T12:00:00Z",
        "score": 0.75,
        "label": "bullish",
        "confidence": 0.85,
        "mentions": 150,
        "keywords": ["bitcoin", "rally", "bull market"]
    }
]

await send_sentiment_data_to_tyra("Fear & Greed Index", "BTC/USDT", sentiment_data)
```

#### Position Updates
```python
async def send_position_update_to_tyra(positions):
    """Send position updates to Tyra."""
    
    formatted_positions = []
    for pos in positions:
        formatted_positions.append({
            "account_name": pos["account"],
            "exchange_code": pos["exchange"],
            "symbol": pos["symbol"],
            "position_id": pos.get("position_id"),
            "side": pos["side"],  # long or short
            "quantity": str(Decimal(str(pos["quantity"]))),
            "entry_price": str(Decimal(str(pos["entry_price"]))),
            "current_price": str(Decimal(str(pos["current_price"]))),
            "unrealized_pnl": str(Decimal(str(pos["unrealized_pnl"]))),
            "status": pos["status"]  # open, closed, liquidated
        })
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/v1/trading/positions/update",
            json=formatted_positions
        ) as response:
            result = await response.json()
            return result
```

### 2. **CCXT MCP Integration**

#### Using MCP Tools
```python
# Example of calling CCXT MCP tools and forwarding to Tyra
import mcp

async def fetch_and_store_market_data():
    """Fetch data from CCXT MCP and store in Tyra."""
    
    # Initialize CCXT MCP client
    ccxt_client = mcp.Client("ccxt-server")
    
    # Fetch OHLCV data using CCXT MCP
    ohlcv_result = await ccxt_client.call("fetch_ohlcv", {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "limit": 100
    })
    
    if ohlcv_result["success"]:
        # Forward to Tyra for storage and AI analysis
        await send_ohlcv_data_to_tyra(
            "BINANCE", 
            "BTC/USDT", 
            ohlcv_result["data"]
        )
    
    # Fetch positions using CCXT MCP
    positions_result = await ccxt_client.call("fetch_positions", {
        "exchange": "binance"
    })
    
    if positions_result["success"]:
        # Forward to Tyra
        await send_position_update_to_tyra(positions_result["data"])
```

### 3. **Automated Data Pipeline**

#### Background Data Fetcher
```python
import asyncio
import ccxt
from datetime import datetime, timedelta

class TradingDataPipeline:
    def __init__(self):
        self.exchanges = {
            "binance": ccxt.binance(),
            "coinbase": ccxt.coinbasepro(),
            "kraken": ccxt.kraken()
        }
        self.symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        self.tyra_base_url = "http://localhost:8000"
    
    async def fetch_and_store_ohlcv(self, exchange_name, symbol, timeframe="1h"):
        """Fetch OHLCV data and store in Tyra."""
        try:
            exchange = self.exchanges[exchange_name]
            
            # Fetch recent OHLCV data
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=24)
            
            # Send to Tyra
            await send_ohlcv_data_to_tyra(
                exchange_name.upper(), 
                symbol, 
                ohlcv
            )
            
            print(f"✅ Stored OHLCV data for {symbol} from {exchange_name}")
            
        except Exception as e:
            print(f"❌ Error fetching OHLCV for {symbol}: {e}")
    
    async def fetch_and_store_sentiment(self):
        """Fetch sentiment data from APIs and store in Tyra."""
        try:
            # Example: Fear & Greed Index
            fear_greed_data = await self.fetch_fear_greed_index()
            await send_sentiment_data_to_tyra(
                "Fear & Greed Index",
                None,  # General market sentiment
                fear_greed_data
            )
            
            print("✅ Stored Fear & Greed sentiment data")
            
        except Exception as e:
            print(f"❌ Error fetching sentiment: {e}")
    
    async def fetch_fear_greed_index(self):
        """Fetch Fear & Greed Index data."""
        # This would call the actual Fear & Greed API
        return [{
            "timestamp": datetime.utcnow().isoformat(),
            "score": 0.65,  # Example score
            "label": "neutral",
            "confidence": 0.8,
            "mentions": 1,
            "keywords": ["fear", "greed", "market"]
        }]
    
    async def run_pipeline(self):
        """Run the complete data pipeline."""
        while True:
            try:
                # Fetch OHLCV data for all symbols and exchanges
                tasks = []
                for exchange_name in self.exchanges:
                    for symbol in self.symbols:
                        tasks.append(
                            self.fetch_and_store_ohlcv(exchange_name, symbol)
                        )
                
                # Add sentiment data task
                tasks.append(self.fetch_and_store_sentiment())
                
                # Execute all tasks concurrently
                await asyncio.gather(*tasks, return_exceptions=True)
                
                print(f"🔄 Data pipeline completed at {datetime.utcnow()}")
                
                # Wait 1 hour before next run
                await asyncio.sleep(3600)
                
            except Exception as e:
                print(f"❌ Pipeline error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

# Run the pipeline
pipeline = TradingDataPipeline()
asyncio.run(pipeline.run_pipeline())
```

### 4. **Trading Signals Integration**

#### Generate and Store Trading Signals
```python
async def generate_and_store_signal(symbol, analysis_result):
    """Generate trading signal from analysis and store in Tyra."""
    
    signal_data = {
        "symbol": symbol,
        "exchange_code": "BINANCE",
        "signal_type": analysis_result["recommendation"],  # buy, sell, hold
        "signal_strength": str(Decimal(str(analysis_result["strength"]))),
        "confidence": analysis_result["confidence"],
        "timeframe": "1h",
        "source": "technical",
        "strategy_name": "RSI_MACD_Crossover",
        "entry_price": str(Decimal(str(analysis_result["entry_price"]))),
        "target_price": str(Decimal(str(analysis_result["target_price"]))),
        "stop_loss_price": str(Decimal(str(analysis_result["stop_loss"]))),
        "reasoning": analysis_result["reasoning"],
        "indicators_used": ["RSI", "MACD", "Volume"]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/v1/trading/signals",
            json=signal_data
        ) as response:
            result = await response.json()
            return result
```

## Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CCXT MCP      │    │  External APIs  │    │  News/Social    │
│   • OHLCV       │    │  • Positions    │    │  • Sentiment    │
│   • Orders      │    │  • Account      │    │  • Articles     │
│   • Positions   │    │  • Metrics      │    │  • Social       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Tyra Trading APIs      │
                    │  • /v1/trading/ohlcv/     │
                    │  • /v1/trading/sentiment/ │
                    │  • /v1/trading/positions/ │
                    │  • /v1/trading/signals/   │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      Data Storage         │
                    │  ┌─────────┬─────────┐    │
                    │  │PostgreSQL│Memory │    │
                    │  │Trading   │System │    │
                    │  │Tables    │(Tyra) │    │
                    │  └─────────┴─────────┘    │
                    └───────────────────────────┘
```

## Configuration

### Environment Variables
```bash
# Add to .env file
TRADING_DATA_ENABLED=true
TRADING_API_RATE_LIMIT=1000
TRADING_DATA_RETENTION_DAYS=365

# Exchange API Keys (if needed)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_secret
COINBASE_API_KEY=your_api_key
COINBASE_SECRET=your_secret

# News/Sentiment APIs
NEWS_API_KEY=your_news_api_key
SOCIAL_API_KEY=your_social_api_key
```

### Trading Configuration
```yaml
# Add to config/config.yaml
trading:
  enabled: true
  exchanges:
    - binance
    - coinbase
    - kraken
  
  symbols:
    - BTC/USDT
    - ETH/USDT
    - ADA/USDT
  
  data_sources:
    ohlcv:
      timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]
      batch_size: 1000
      retention_days: 365
    
    sentiment:
      sources: ["fear_greed", "news", "social"]
      update_interval: 3600  # 1 hour
      retention_days: 90
  
  memory_integration:
    store_as_memories: true
    agent_id: "tyra"
    include_metadata: true
```

## Testing the Integration

### 1. **Test Database Setup**
```bash
# Run migration
python scripts/run_migration.py 003_trading_data_schema.sql

# Verify tables created
psql -d tyra_memory -c "\dt trading_*"
```

### 2. **Test API Endpoints**
```bash
# Test OHLCV ingestion
curl -X POST http://localhost:8000/v1/trading/ohlcv/batch \
  -H "Content-Type: application/json" \
  -d '{
    "exchange_code": "BINANCE",
    "symbol": "BTC/USDT",
    "data": [{
      "timestamp": "2024-01-01T12:00:00Z",
      "timeframe": "1h",
      "open_price": "47000.0",
      "high_price": "47200.0",
      "low_price": "46800.0",
      "close_price": "47100.0",
      "volume": "125.5"
    }],
    "data_source": "test"
  }'

# Test sentiment ingestion
curl -X POST http://localhost:8000/v1/trading/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{
    "source_name": "Test Source",
    "symbol": "BTC/USDT",
    "data": [{
      "timestamp": "2024-01-01T12:00:00Z",
      "sentiment_score": "0.75",
      "sentiment_label": "bullish",
      "confidence": 0.85,
      "keywords": ["bitcoin", "rally"]
    }]
  }'
```

### 3. **Test Memory Integration**
```bash
# Check if data appears in memory system
curl http://localhost:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "BTC OHLCV",
    "agent_id": "tyra",
    "limit": 10
  }'
```

## Monitoring and Maintenance

### 1. **Data Quality Monitoring**
```python
# Check data ingestion logs
SELECT * FROM data_ingestion_logs 
WHERE status = 'error' 
ORDER BY started_at DESC 
LIMIT 10;

# Check data quality metrics
SELECT * FROM data_quality_metrics 
WHERE status != 'good' 
ORDER BY measured_at DESC;
```

### 2. **Performance Monitoring**
```python
# Monitor API endpoint performance
SELECT 
    data_type,
    AVG(execution_time_ms) as avg_time,
    COUNT(*) as total_calls
FROM data_ingestion_logs 
WHERE started_at > NOW() - INTERVAL '24 hours'
GROUP BY data_type;
```

### 3. **Memory System Health**
```bash
# Check memory system for trading data
curl http://localhost:8000/v1/memory/stats?agent_id=tyra
```

## Troubleshooting

### Common Issues

1. **Migration Fails**
   - Ensure PostgreSQL has vector extension
   - Check database permissions
   - Verify no conflicting table names

2. **API Ingestion Errors**
   - Check JSON format and required fields
   - Verify exchange codes match database
   - Check decimal precision limits

3. **Memory Integration Issues**
   - Verify memory manager is running
   - Check agent_id matches configuration
   - Ensure metadata is valid JSON

4. **Performance Issues**
   - Add indexes for your query patterns
   - Consider partitioning large tables
   - Monitor connection pool usage

This completes the comprehensive trading data integration system!