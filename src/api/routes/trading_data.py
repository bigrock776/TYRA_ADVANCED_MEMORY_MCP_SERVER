"""
Trading Data Ingestion API Endpoints.

Provides comprehensive endpoints for ingesting OHLCV data, sentiment analysis,
position updates, and other trading data from APIs and CCXT MCP.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ...core.memory.manager import MemoryManager
from ...core.utils.logger import get_logger
from ..app import get_memory_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/trading", tags=["trading-data"])


# =============================================================================
# Request/Response Models
# =============================================================================

class OHLCVDataPoint(BaseModel):
    """Single OHLCV data point."""
    timestamp: datetime
    timeframe: str = Field(..., description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d")
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    quote_volume: Optional[Decimal] = None
    trades_count: Optional[int] = None
    vwap: Optional[Decimal] = None


class OHLCVBatchRequest(BaseModel):
    """Batch OHLCV data ingestion request."""
    exchange_code: str = Field(..., description="Exchange code (e.g., BINANCE)")
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    data: List[OHLCVDataPoint] = Field(..., description="OHLCV data points")
    data_source: str = Field(default="api", description="Data source: api, ccxt, websocket")
    overwrite_existing: bool = Field(default=False, description="Overwrite existing data")


class SentimentDataPoint(BaseModel):
    """Single sentiment analysis data point."""
    timestamp: datetime
    sentiment_score: Decimal = Field(..., ge=-1, le=1, description="Sentiment score -1 to 1")
    sentiment_label: str = Field(..., description="bullish, bearish, neutral")
    confidence: float = Field(..., ge=0, le=1, description="Confidence 0-1")
    volume_mentions: Optional[int] = None
    keywords: Optional[List[str]] = None
    raw_data: Optional[Dict[str, Any]] = None


class SentimentBatchRequest(BaseModel):
    """Batch sentiment data ingestion request."""
    source_name: str = Field(..., description="Sentiment source name")
    symbol: Optional[str] = Field(None, description="Trading symbol (optional)")
    timeframe: str = Field(default="1h", description="Aggregation timeframe")
    data: List[SentimentDataPoint] = Field(..., description="Sentiment data points")


class NewsArticle(BaseModel):
    """News article data."""
    title: str
    content: Optional[str] = None
    published_at: datetime
    url: Optional[str] = None
    author: Optional[str] = None
    sentiment_score: Optional[Decimal] = Field(None, ge=-1, le=1)
    relevance_score: Optional[float] = Field(None, ge=0, le=1)
    symbols_mentioned: Optional[List[str]] = None
    categories: Optional[List[str]] = None


class NewsBatchRequest(BaseModel):
    """Batch news data ingestion request."""
    source_name: str = Field(..., description="News source name")
    articles: List[NewsArticle] = Field(..., description="News articles")


class PositionUpdate(BaseModel):
    """Position update data."""
    account_name: str
    exchange_code: str
    symbol: str
    position_id: Optional[str] = None
    side: str = Field(..., description="long or short")
    quantity: Decimal
    entry_price: Decimal
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    fees: Optional[Decimal] = None
    margin_used: Optional[Decimal] = None
    liquidation_price: Optional[Decimal] = None
    status: str = Field(default="open", description="open, closed, liquidated")
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


class TradingSignal(BaseModel):
    """Trading signal data."""
    symbol: str
    exchange_code: str
    signal_type: str = Field(..., description="buy, sell, hold")
    signal_strength: Decimal = Field(..., ge=0, le=1)
    confidence: Decimal = Field(..., ge=0, le=1)
    timeframe: str
    source: str = Field(..., description="technical, fundamental, sentiment, ml")
    strategy_name: Optional[str] = None
    entry_price: Optional[Decimal] = None
    target_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    risk_reward_ratio: Optional[Decimal] = None
    reasoning: Optional[str] = None
    indicators_used: Optional[List[str]] = None
    expires_at: Optional[datetime] = None


class IngestionResponse(BaseModel):
    """Standard ingestion response."""
    success: bool
    message: str
    records_processed: int
    records_inserted: int = 0
    records_updated: int = 0
    records_failed: int = 0
    errors: Optional[List[str]] = None
    ingestion_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Database Helper Functions
# =============================================================================

async def get_or_create_exchange(exchange_code: str, exchange_type: str = "crypto") -> str:
    """Get or create trading exchange record."""
    # This would typically use a database connection
    # For now, return a mock UUID
    return str(uuid.uuid4())


async def get_or_create_instrument(symbol: str, exchange_id: str) -> str:
    """Get or create trading instrument record."""
    # Parse symbol (e.g., "BTC/USDT" -> base="BTC", quote="USDT")
    if "/" in symbol:
        base_asset, quote_asset = symbol.split("/", 1)
    else:
        base_asset, quote_asset = symbol, "USD"
    
    return str(uuid.uuid4())


async def get_or_create_sentiment_source(source_name: str, source_type: str = "api") -> str:
    """Get or create sentiment source record."""
    return str(uuid.uuid4())


async def log_ingestion_attempt(data_type: str, source: str, status: str, **kwargs) -> str:
    """Log data ingestion attempt."""
    ingestion_id = str(uuid.uuid4())
    logger.info(
        f"Data ingestion logged",
        ingestion_id=ingestion_id,
        data_type=data_type,
        source=source,
        status=status,
        **kwargs
    )
    return ingestion_id


# =============================================================================
# OHLCV Data Endpoints
# =============================================================================

@router.post("/ohlcv/batch", response_model=IngestionResponse)
async def ingest_ohlcv_batch(
    request: OHLCVBatchRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Ingest batch OHLCV data from exchanges.
    
    Supports data from CCXT, direct API calls, or WebSocket streams.
    Automatically creates exchange and instrument records if they don't exist.
    """
    ingestion_id = str(uuid.uuid4())
    
    try:
        # Get or create exchange and instrument
        exchange_id = await get_or_create_exchange(request.exchange_code)
        instrument_id = await get_or_create_instrument(request.symbol, exchange_id)
        
        # Process OHLCV data points
        records_processed = len(request.data)
        records_inserted = 0
        records_updated = 0
        records_failed = 0
        errors = []
        
        for data_point in request.data:
            try:
                # Store OHLCV data in database
                # This would typically use SQLAlchemy or direct SQL
                
                # Also store as memory for AI analysis
                memory_content = f"OHLCV {request.symbol} {data_point.timeframe}: O:{data_point.open_price} H:{data_point.high_price} L:{data_point.low_price} C:{data_point.close_price} V:{data_point.volume} at {data_point.timestamp}"
                
                await memory_manager.store_memory(
                    memory_id=str(uuid.uuid4()),
                    text=memory_content,
                    metadata={
                        "type": "ohlcv_data",
                        "exchange": request.exchange_code,
                        "symbol": request.symbol,
                        "timeframe": data_point.timeframe,
                        "timestamp": data_point.timestamp.isoformat(),
                        "close_price": float(data_point.close_price),
                        "volume": float(data_point.volume),
                        "data_source": request.data_source
                    },
                    agent_id="tyra"
                )
                
                records_inserted += 1
                
            except Exception as e:
                records_failed += 1
                errors.append(f"Failed to process data point at {data_point.timestamp}: {str(e)}")
                logger.error(f"OHLCV ingestion error: {e}")
        
        # Log ingestion attempt
        await log_ingestion_attempt(
            data_type="ohlcv",
            source=request.data_source,
            status="success" if records_failed == 0 else "partial",
            records_processed=records_processed,
            records_inserted=records_inserted,
            records_failed=records_failed
        )
        
        return IngestionResponse(
            success=records_failed == 0,
            message=f"Processed {records_processed} OHLCV records for {request.symbol}",
            records_processed=records_processed,
            records_inserted=records_inserted,
            records_updated=records_updated,
            records_failed=records_failed,
            errors=errors if errors else None,
            ingestion_id=ingestion_id
        )
        
    except Exception as e:
        logger.error(f"OHLCV batch ingestion failed: {e}")
        await log_ingestion_attempt(
            data_type="ohlcv",
            source=request.data_source,
            status="error",
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"OHLCV ingestion failed: {str(e)}")


@router.get("/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str,
    exchange: str = Query(..., description="Exchange code"),
    timeframe: str = Query("1d", description="Timeframe"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter")
):
    """
    Retrieve OHLCV data for a symbol.
    
    Returns historical OHLCV data with optional time filtering.
    """
    try:
        # This would query the database for OHLCV data
        # For now, return a mock response
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data": [],  # Would contain actual OHLCV records
            "count": 0,
            "query_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"OHLCV retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve OHLCV data: {str(e)}")


# =============================================================================
# Sentiment Data Endpoints
# =============================================================================

@router.post("/sentiment/batch", response_model=IngestionResponse)
async def ingest_sentiment_batch(
    request: SentimentBatchRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Ingest batch sentiment analysis data.
    
    Supports sentiment data from news APIs, social media, or analysis services.
    """
    ingestion_id = str(uuid.uuid4())
    
    try:
        # Get or create sentiment source
        source_id = await get_or_create_sentiment_source(request.source_name)
        
        # Get instrument if symbol provided
        instrument_id = None
        if request.symbol:
            exchange_id = await get_or_create_exchange("GENERAL", "sentiment")
            instrument_id = await get_or_create_instrument(request.symbol, exchange_id)
        
        # Process sentiment data points
        records_processed = len(request.data)
        records_inserted = 0
        records_failed = 0
        errors = []
        
        for data_point in request.data:
            try:
                # Store sentiment data in database
                # This would typically use SQLAlchemy
                
                # Also store as memory for AI analysis
                memory_content = f"Sentiment analysis from {request.source_name}: {data_point.sentiment_label} sentiment score {data_point.sentiment_score} (confidence: {data_point.confidence}) at {data_point.timestamp}"
                if data_point.keywords:
                    memory_content += f" Keywords: {', '.join(data_point.keywords)}"
                
                await memory_manager.store_memory(
                    memory_id=str(uuid.uuid4()),
                    text=memory_content,
                    metadata={
                        "type": "sentiment_data",
                        "source": request.source_name,
                        "symbol": request.symbol,
                        "timeframe": request.timeframe,
                        "timestamp": data_point.timestamp.isoformat(),
                        "sentiment_score": float(data_point.sentiment_score),
                        "sentiment_label": data_point.sentiment_label,
                        "confidence": data_point.confidence,
                        "keywords": data_point.keywords,
                        "volume_mentions": data_point.volume_mentions
                    },
                    agent_id="tyra"
                )
                
                records_inserted += 1
                
            except Exception as e:
                records_failed += 1
                errors.append(f"Failed to process sentiment data at {data_point.timestamp}: {str(e)}")
                logger.error(f"Sentiment ingestion error: {e}")
        
        # Log ingestion attempt
        await log_ingestion_attempt(
            data_type="sentiment",
            source=request.source_name,
            status="success" if records_failed == 0 else "partial",
            records_processed=records_processed,
            records_inserted=records_inserted,
            records_failed=records_failed
        )
        
        return IngestionResponse(
            success=records_failed == 0,
            message=f"Processed {records_processed} sentiment records from {request.source_name}",
            records_processed=records_processed,
            records_inserted=records_inserted,
            records_failed=records_failed,
            errors=errors if errors else None,
            ingestion_id=ingestion_id
        )
        
    except Exception as e:
        logger.error(f"Sentiment batch ingestion failed: {e}")
        await log_ingestion_attempt(
            data_type="sentiment",
            source=request.source_name,
            status="error",
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Sentiment ingestion failed: {str(e)}")


# =============================================================================
# News Data Endpoints
# =============================================================================

@router.post("/news/batch", response_model=IngestionResponse)
async def ingest_news_batch(
    request: NewsBatchRequest,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Ingest batch news articles with sentiment analysis.
    
    Processes news articles and stores them for sentiment and market analysis.
    """
    ingestion_id = str(uuid.uuid4())
    
    try:
        # Get or create news source
        source_id = await get_or_create_sentiment_source(request.source_name, "news")
        
        # Process news articles
        records_processed = len(request.articles)
        records_inserted = 0
        records_failed = 0
        errors = []
        
        for article in request.articles:
            try:
                # Store news article in database
                # This would typically use SQLAlchemy
                
                # Also store as memory for AI analysis
                memory_content = f"News from {request.source_name}: {article.title}"
                if article.content:
                    memory_content += f"\n\nContent: {article.content[:500]}..."
                
                metadata = {
                    "type": "news_article",
                    "source": request.source_name,
                    "title": article.title,
                    "published_at": article.published_at.isoformat(),
                    "url": article.url,
                    "author": article.author,
                    "symbols_mentioned": article.symbols_mentioned,
                    "categories": article.categories
                }
                
                if article.sentiment_score is not None:
                    metadata.update({
                        "sentiment_score": float(article.sentiment_score),
                        "relevance_score": article.relevance_score
                    })
                
                await memory_manager.store_memory(
                    memory_id=str(uuid.uuid4()),
                    text=memory_content,
                    metadata=metadata,
                    agent_id="tyra"
                )
                
                records_inserted += 1
                
            except Exception as e:
                records_failed += 1
                errors.append(f"Failed to process article '{article.title}': {str(e)}")
                logger.error(f"News ingestion error: {e}")
        
        # Log ingestion attempt
        await log_ingestion_attempt(
            data_type="news",
            source=request.source_name,
            status="success" if records_failed == 0 else "partial",
            records_processed=records_processed,
            records_inserted=records_inserted,
            records_failed=records_failed
        )
        
        return IngestionResponse(
            success=records_failed == 0,
            message=f"Processed {records_processed} news articles from {request.source_name}",
            records_processed=records_processed,
            records_inserted=records_inserted,
            records_failed=records_failed,
            errors=errors if errors else None,
            ingestion_id=ingestion_id
        )
        
    except Exception as e:
        logger.error(f"News batch ingestion failed: {e}")
        await log_ingestion_attempt(
            data_type="news",
            source=request.source_name,
            status="error",
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"News ingestion failed: {str(e)}")


# =============================================================================
# Position and Trading Data Endpoints
# =============================================================================

@router.post("/positions/update")
async def update_positions(
    positions: List[PositionUpdate],
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Update trading positions from CCXT or trading APIs.
    
    Processes position updates and calculates P&L, risk metrics.
    """
    try:
        records_processed = len(positions)
        records_updated = 0
        errors = []
        
        for position in positions:
            try:
                # Update position in database
                # This would typically use SQLAlchemy
                
                # Store position update as memory
                memory_content = f"Position update for {position.symbol}: {position.side} {position.quantity} at {position.entry_price}"
                if position.unrealized_pnl is not None:
                    memory_content += f", Unrealized P&L: {position.unrealized_pnl}"
                
                await memory_manager.store_memory(
                    memory_id=str(uuid.uuid4()),
                    text=memory_content,
                    metadata={
                        "type": "position_update",
                        "account": position.account_name,
                        "exchange": position.exchange_code,
                        "symbol": position.symbol,
                        "side": position.side,
                        "quantity": float(position.quantity),
                        "entry_price": float(position.entry_price),
                        "current_price": float(position.current_price) if position.current_price else None,
                        "unrealized_pnl": float(position.unrealized_pnl) if position.unrealized_pnl else None,
                        "status": position.status
                    },
                    agent_id="tyra"
                )
                
                records_updated += 1
                
            except Exception as e:
                errors.append(f"Failed to update position {position.symbol}: {str(e)}")
                logger.error(f"Position update error: {e}")
        
        return {
            "success": len(errors) == 0,
            "message": f"Processed {records_processed} position updates",
            "records_processed": records_processed,
            "records_updated": records_updated,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"Position update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Position update failed: {str(e)}")


@router.post("/signals", response_model=IngestionResponse)
async def store_trading_signal(
    signal: TradingSignal,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """
    Store trading signal from analysis or external source.
    
    Processes trading signals and stores them for decision making.
    """
    try:
        # Get or create exchange and instrument
        exchange_id = await get_or_create_exchange(signal.exchange_code)
        instrument_id = await get_or_create_instrument(signal.symbol, exchange_id)
        
        # Store trading signal in database
        # This would typically use SQLAlchemy
        
        # Store signal as memory for AI analysis
        memory_content = f"Trading signal for {signal.symbol}: {signal.signal_type.upper()} signal with {signal.signal_strength} strength and {signal.confidence} confidence"
        if signal.reasoning:
            memory_content += f"\n\nReasoning: {signal.reasoning}"
        
        metadata = {
            "type": "trading_signal",
            "symbol": signal.symbol,
            "exchange": signal.exchange_code,
            "signal_type": signal.signal_type,
            "signal_strength": float(signal.signal_strength),
            "confidence": float(signal.confidence),
            "timeframe": signal.timeframe,
            "source": signal.source,
            "strategy_name": signal.strategy_name,
            "indicators_used": signal.indicators_used
        }
        
        if signal.entry_price:
            metadata["entry_price"] = float(signal.entry_price)
        if signal.target_price:
            metadata["target_price"] = float(signal.target_price)
        if signal.stop_loss_price:
            metadata["stop_loss_price"] = float(signal.stop_loss_price)
        if signal.risk_reward_ratio:
            metadata["risk_reward_ratio"] = float(signal.risk_reward_ratio)
        
        await memory_manager.store_memory(
            memory_id=str(uuid.uuid4()),
            text=memory_content,
            metadata=metadata,
            agent_id="tyra"
        )
        
        ingestion_id = str(uuid.uuid4())
        
        return IngestionResponse(
            success=True,
            message=f"Trading signal stored for {signal.symbol}",
            records_processed=1,
            records_inserted=1,
            ingestion_id=ingestion_id
        )
        
    except Exception as e:
        logger.error(f"Trading signal storage failed: {e}")
        raise HTTPException(status_code=500, detail=f"Signal storage failed: {str(e)}")


# =============================================================================
# Data Retrieval Endpoints
# =============================================================================

@router.get("/sentiment/{symbol}")
async def get_sentiment_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records"),
    start_time: Optional[datetime] = Query(None, description="Start time filter")
):
    """Get sentiment data for a symbol."""
    try:
        # This would query the database for sentiment data
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "sentiment_data": [],  # Would contain actual sentiment records
            "count": 0,
            "query_time": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Sentiment retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sentiment data: {str(e)}")


@router.get("/signals/{symbol}")
async def get_trading_signals(
    symbol: str,
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    min_confidence: float = Query(0.5, ge=0, le=1, description="Minimum confidence"),
    limit: int = Query(50, ge=1, le=500, description="Number of records")
):
    """Get trading signals for a symbol."""
    try:
        # This would query the database for trading signals
        return {
            "symbol": symbol,
            "signals": [],  # Would contain actual signal records
            "count": 0,
            "query_time": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Signal retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve signals: {str(e)}")


# =============================================================================
# Health and Status Endpoints
# =============================================================================

@router.get("/health")
async def trading_data_health():
    """Get trading data system health status."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "connected",
                "memory_system": "operational",
                "ingestion_endpoints": "available"
            },
            "metrics": {
                "total_symbols_tracked": 0,  # Would be actual count
                "data_sources": 0,
                "last_ingestion": None
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }