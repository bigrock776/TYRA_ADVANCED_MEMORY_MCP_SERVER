-- =============================================================================
-- Tyra MCP Memory Server - Trading Data Schema
-- =============================================================================
-- This migration adds comprehensive trading data tables for OHLCV data,
-- sentiment analysis, positions, and trading analytics from APIs and CCXT MCP.

-- =============================================================================
-- Trading Markets and Instruments
-- =============================================================================

-- Trading exchanges/markets
CREATE TABLE IF NOT EXISTS trading_exchanges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    code VARCHAR(20) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL, -- 'crypto', 'forex', 'stocks', 'futures'
    api_endpoints JSONB DEFAULT '{}'::jsonb,
    rate_limits JSONB DEFAULT '{}'::jsonb,
    fees JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Trading instruments/symbols
CREATE TABLE IF NOT EXISTS trading_instruments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,
    exchange_id UUID NOT NULL REFERENCES trading_exchanges(id) ON DELETE CASCADE,
    base_asset VARCHAR(20) NOT NULL,
    quote_asset VARCHAR(20) NOT NULL,
    instrument_type VARCHAR(50) NOT NULL, -- 'spot', 'future', 'option', 'perpetual'
    min_quantity DECIMAL(20,10),
    max_quantity DECIMAL(20,10),
    quantity_step DECIMAL(20,10),
    min_price DECIMAL(20,10),
    max_price DECIMAL(20,10),
    price_step DECIMAL(20,10),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(symbol, exchange_id)
);

-- =============================================================================
-- OHLCV Data Tables
-- =============================================================================

-- Main OHLCV data table with partitioning support
CREATE TABLE IF NOT EXISTS ohlcv_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id UUID NOT NULL REFERENCES trading_instruments(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '4h', '1d', etc.
    open_price DECIMAL(20,10) NOT NULL,
    high_price DECIMAL(20,10) NOT NULL,
    low_price DECIMAL(20,10) NOT NULL,
    close_price DECIMAL(20,10) NOT NULL,
    volume DECIMAL(25,10) NOT NULL,
    quote_volume DECIMAL(25,10),
    trades_count INTEGER,
    vwap DECIMAL(20,10), -- Volume Weighted Average Price
    data_source VARCHAR(50) NOT NULL DEFAULT 'api', -- 'api', 'ccxt', 'websocket'
    quality_score FLOAT DEFAULT 1.0, -- Data quality indicator
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(instrument_id, timestamp, timeframe)
);

-- Technical indicators cache
CREATE TABLE IF NOT EXISTS technical_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id UUID NOT NULL REFERENCES trading_instruments(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value DECIMAL(20,10),
    indicator_data JSONB, -- For complex indicators with multiple values
    calculation_params JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(instrument_id, timestamp, timeframe, indicator_name)
);

-- =============================================================================
-- Sentiment Analysis Tables
-- =============================================================================

-- Sentiment data sources
CREATE TABLE IF NOT EXISTS sentiment_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    source_type VARCHAR(50) NOT NULL, -- 'news', 'social', 'analysis', 'fear_greed'
    api_endpoint VARCHAR(500),
    update_frequency INTEGER, -- in minutes
    reliability_score FLOAT DEFAULT 0.8,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Sentiment analysis results
CREATE TABLE IF NOT EXISTS sentiment_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id UUID REFERENCES trading_instruments(id) ON DELETE CASCADE,
    source_id UUID NOT NULL REFERENCES sentiment_sources(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) DEFAULT '1h', -- Aggregation timeframe
    sentiment_score DECIMAL(5,4), -- -1.0 to 1.0
    sentiment_label VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
    confidence FLOAT NOT NULL,
    volume_mentions INTEGER, -- Number of mentions/articles
    raw_data JSONB, -- Raw sentiment data
    keywords JSONB DEFAULT '[]'::jsonb, -- Extracted keywords
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- News and social media data
CREATE TABLE IF NOT EXISTS market_news (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT,
    source_id UUID NOT NULL REFERENCES sentiment_sources(id) ON DELETE CASCADE,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    url VARCHAR(1000),
    author VARCHAR(255),
    sentiment_score DECIMAL(5,4),
    relevance_score FLOAT,
    symbols_mentioned JSONB DEFAULT '[]'::jsonb,
    categories JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =============================================================================
-- Trading Positions and Portfolio
-- =============================================================================

-- Trading accounts/portfolios
CREATE TABLE IF NOT EXISTS trading_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    account_type VARCHAR(50) NOT NULL, -- 'demo', 'live', 'paper'
    exchange_id UUID NOT NULL REFERENCES trading_exchanges(id) ON DELETE CASCADE,
    agent_id VARCHAR(255) NOT NULL DEFAULT 'tyra',
    balance JSONB DEFAULT '{}'::jsonb, -- Balance by currency
    equity DECIMAL(20,10),
    margin_used DECIMAL(20,10),
    margin_available DECIMAL(20,10),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Trading positions
CREATE TABLE IF NOT EXISTS trading_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id) ON DELETE CASCADE,
    instrument_id UUID NOT NULL REFERENCES trading_instruments(id) ON DELETE CASCADE,
    position_id VARCHAR(100), -- External position ID
    side VARCHAR(10) NOT NULL, -- 'long', 'short'
    quantity DECIMAL(20,10) NOT NULL,
    entry_price DECIMAL(20,10) NOT NULL,
    current_price DECIMAL(20,10),
    mark_price DECIMAL(20,10),
    unrealized_pnl DECIMAL(20,10),
    realized_pnl DECIMAL(20,10),
    fees DECIMAL(20,10) DEFAULT 0,
    margin_used DECIMAL(20,10),
    liquidation_price DECIMAL(20,10),
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL,
    closed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'liquidated'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Trading orders
CREATE TABLE IF NOT EXISTS trading_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id) ON DELETE CASCADE,
    instrument_id UUID NOT NULL REFERENCES trading_instruments(id) ON DELETE CASCADE,
    position_id UUID REFERENCES trading_positions(id) ON DELETE SET NULL,
    order_id VARCHAR(100), -- External order ID
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop', 'stop_limit'
    side VARCHAR(10) NOT NULL, -- 'buy', 'sell'
    quantity DECIMAL(20,10) NOT NULL,
    filled_quantity DECIMAL(20,10) DEFAULT 0,
    price DECIMAL(20,10),
    stop_price DECIMAL(20,10),
    average_price DECIMAL(20,10),
    fees DECIMAL(20,10) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'filled', 'cancelled', 'rejected'
    time_in_force VARCHAR(10) DEFAULT 'GTC', -- 'GTC', 'IOC', 'FOK'
    placed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =============================================================================
-- Trading Analytics and Signals
-- =============================================================================

-- Trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id UUID NOT NULL REFERENCES trading_instruments(id) ON DELETE CASCADE,
    signal_type VARCHAR(50) NOT NULL, -- 'buy', 'sell', 'hold'
    signal_strength DECIMAL(5,4) NOT NULL, -- 0.0 to 1.0
    confidence DECIMAL(5,4) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    source VARCHAR(100) NOT NULL, -- 'technical', 'fundamental', 'sentiment', 'ml'
    strategy_name VARCHAR(100),
    entry_price DECIMAL(20,10),
    target_price DECIMAL(20,10),
    stop_loss_price DECIMAL(20,10),
    risk_reward_ratio DECIMAL(10,4),
    reasoning TEXT,
    indicators_used JSONB DEFAULT '[]'::jsonb,
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Trading performance metrics
CREATE TABLE IF NOT EXISTS trading_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES trading_accounts(id) ON DELETE CASCADE,
    instrument_id UUID REFERENCES trading_instruments(id) ON DELETE CASCADE,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_pnl DECIMAL(20,10),
    realized_pnl DECIMAL(20,10),
    unrealized_pnl DECIMAL(20,10),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(10,4),
    max_drawdown DECIMAL(20,10),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    calmar_ratio DECIMAL(10,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(account_id, instrument_id, timeframe, timestamp)
);

-- =============================================================================
-- Data Quality and Monitoring
-- =============================================================================

-- Data ingestion logs
CREATE TABLE IF NOT EXISTS data_ingestion_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    data_type VARCHAR(50) NOT NULL, -- 'ohlcv', 'sentiment', 'news', 'position'
    source VARCHAR(100) NOT NULL, -- 'ccxt', 'api', 'websocket'
    status VARCHAR(20) NOT NULL, -- 'success', 'error', 'partial'
    records_processed INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    error_message TEXT,
    execution_time_ms INTEGER,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Data quality metrics
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,10),
    threshold_value DECIMAL(20,10),
    status VARCHAR(20) NOT NULL, -- 'good', 'warning', 'critical'
    measured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Exchange and instrument indexes
CREATE INDEX IF NOT EXISTS idx_trading_exchanges_code ON trading_exchanges(code);
CREATE INDEX IF NOT EXISTS idx_trading_instruments_symbol ON trading_instruments(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_instruments_exchange ON trading_instruments(exchange_id);

-- OHLCV data indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_instrument_time ON ohlcv_data(instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe ON ohlcv_data(timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp DESC);

-- Technical indicators indexes
CREATE INDEX IF NOT EXISTS idx_technical_indicators_instrument ON technical_indicators(instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_name ON technical_indicators(indicator_name, timestamp DESC);

-- Sentiment data indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_data_instrument ON sentiment_data(instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_source ON sentiment_data(source_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_score ON sentiment_data(sentiment_score, timestamp DESC);

-- News indexes
CREATE INDEX IF NOT EXISTS idx_market_news_published ON market_news(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_news_source ON market_news(source_id, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_news_symbols ON market_news USING GIN(symbols_mentioned);

-- Trading position indexes
CREATE INDEX IF NOT EXISTS idx_trading_positions_account ON trading_positions(account_id, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_positions_instrument ON trading_positions(instrument_id, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_positions_status ON trading_positions(status, opened_at DESC);

-- Trading order indexes
CREATE INDEX IF NOT EXISTS idx_trading_orders_account ON trading_orders(account_id, placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_orders_instrument ON trading_orders(instrument_id, placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_orders_status ON trading_orders(status, placed_at DESC);

-- Trading signal indexes
CREATE INDEX IF NOT EXISTS idx_trading_signals_instrument ON trading_signals(instrument_id, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_type ON trading_signals(signal_type, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_confidence ON trading_signals(confidence DESC, generated_at DESC);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_trading_metrics_account ON trading_metrics(account_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_instrument ON trading_metrics(instrument_id, timestamp DESC);

-- Data ingestion indexes
CREATE INDEX IF NOT EXISTS idx_data_ingestion_logs_type ON data_ingestion_logs(data_type, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_ingestion_logs_status ON data_ingestion_logs(status, started_at DESC);

-- =============================================================================
-- Triggers for Updated At
-- =============================================================================

CREATE TRIGGER update_trading_exchanges_updated_at BEFORE UPDATE ON trading_exchanges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_instruments_updated_at BEFORE UPDATE ON trading_instruments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON trading_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_positions_updated_at BEFORE UPDATE ON trading_positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_orders_updated_at BEFORE UPDATE ON trading_orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Useful Functions
-- =============================================================================

-- Function to get latest OHLCV data for an instrument
CREATE OR REPLACE FUNCTION get_latest_ohlcv(p_instrument_id UUID, p_timeframe VARCHAR)
RETURNS TABLE (
    timestamp TIMESTAMP WITH TIME ZONE,
    open_price DECIMAL(20,10),
    high_price DECIMAL(20,10),
    low_price DECIMAL(20,10),
    close_price DECIMAL(20,10),
    volume DECIMAL(25,10)
) AS $$
BEGIN
    RETURN QUERY
    SELECT od.timestamp, od.open_price, od.high_price, od.low_price, od.close_price, od.volume
    FROM ohlcv_data od
    WHERE od.instrument_id = p_instrument_id AND od.timeframe = p_timeframe
    ORDER BY od.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate basic trading statistics
CREATE OR REPLACE FUNCTION get_trading_stats(p_account_id UUID, p_days INTEGER DEFAULT 30)
RETURNS JSONB AS $$
DECLARE
    stats JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_trades', COUNT(*),
        'winning_trades', COUNT(*) FILTER (WHERE realized_pnl > 0),
        'losing_trades', COUNT(*) FILTER (WHERE realized_pnl < 0),
        'win_rate', ROUND(
            (COUNT(*) FILTER (WHERE realized_pnl > 0)::DECIMAL / NULLIF(COUNT(*), 0)) * 100, 2
        ),
        'total_pnl', COALESCE(SUM(realized_pnl), 0),
        'avg_win', COALESCE(AVG(realized_pnl) FILTER (WHERE realized_pnl > 0), 0),
        'avg_loss', COALESCE(AVG(realized_pnl) FILTER (WHERE realized_pnl < 0), 0),
        'largest_win', COALESCE(MAX(realized_pnl), 0),
        'largest_loss', COALESCE(MIN(realized_pnl), 0),
        'profit_factor', CASE
            WHEN ABS(COALESCE(SUM(realized_pnl) FILTER (WHERE realized_pnl < 0), 0)) > 0 THEN
                COALESCE(SUM(realized_pnl) FILTER (WHERE realized_pnl > 0), 0) / 
                ABS(COALESCE(SUM(realized_pnl) FILTER (WHERE realized_pnl < 0), 0))
            ELSE 0
        END
    ) INTO stats
    FROM trading_positions
    WHERE account_id = p_account_id
    AND status = 'closed'
    AND closed_at > CURRENT_TIMESTAMP - (p_days || ' days')::INTERVAL;

    RETURN stats;
END;
$$ LANGUAGE plpgsql;

-- Function to get current sentiment for an instrument
CREATE OR REPLACE FUNCTION get_current_sentiment(p_instrument_id UUID)
RETURNS JSONB AS $$
DECLARE
    sentiment JSONB;
BEGIN
    SELECT jsonb_build_object(
        'overall_sentiment', AVG(sentiment_score),
        'confidence', AVG(confidence),
        'sources_count', COUNT(DISTINCT source_id),
        'latest_timestamp', MAX(timestamp),
        'sentiment_distribution', jsonb_build_object(
            'bullish', COUNT(*) FILTER (WHERE sentiment_label = 'bullish'),
            'bearish', COUNT(*) FILTER (WHERE sentiment_label = 'bearish'),
            'neutral', COUNT(*) FILTER (WHERE sentiment_label = 'neutral')
        )
    ) INTO sentiment
    FROM sentiment_data
    WHERE instrument_id = p_instrument_id
    AND timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours';

    RETURN sentiment;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert common exchanges
INSERT INTO trading_exchanges (name, code, type, metadata) VALUES
    ('Binance', 'BINANCE', 'crypto', '{"website": "https://binance.com", "api_version": "v3"}'),
    ('Coinbase Pro', 'COINBASE', 'crypto', '{"website": "https://pro.coinbase.com", "api_version": "v2"}'),
    ('Kraken', 'KRAKEN', 'crypto', '{"website": "https://kraken.com", "api_version": "v1"}'),
    ('Bybit', 'BYBIT', 'crypto', '{"website": "https://bybit.com", "api_version": "v5"}'),
    ('OKX', 'OKX', 'crypto', '{"website": "https://okx.com", "api_version": "v5"}')
ON CONFLICT (code) DO NOTHING;

-- Insert common sentiment sources
INSERT INTO sentiment_sources (name, source_type, reliability_score, metadata) VALUES
    ('Fear & Greed Index', 'analysis', 0.8, '{"provider": "CNN", "scale": "0-100"}'),
    ('CoinTelegraph', 'news', 0.7, '{"provider": "CoinTelegraph", "language": "en"}'),
    ('CoinDesk', 'news', 0.8, '{"provider": "CoinDesk", "language": "en"}'),
    ('Reddit Sentiment', 'social', 0.6, '{"provider": "Reddit", "subreddits": ["cryptocurrency", "bitcoin"]}'),
    ('Twitter Sentiment', 'social', 0.5, '{"provider": "Twitter", "keywords": ["bitcoin", "crypto"]"}')
ON CONFLICT (name) DO NOTHING;

-- Log schema creation
INSERT INTO config_changes (parameter_name, old_value, new_value, change_reason)
VALUES
    ('trading_schema_version', NULL, '1.0.0', 'Initial trading schema creation'),
    ('trading_tables_created', NULL, 'true', 'Trading data tables created')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Completion
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Trading data schema v1.0.0 created successfully';
    RAISE NOTICE 'Trading tables created: 18';
    RAISE NOTICE 'Ready for OHLCV data, sentiment analysis, and position tracking';
END $$;