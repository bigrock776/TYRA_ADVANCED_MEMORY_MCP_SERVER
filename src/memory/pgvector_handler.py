"""
Enhanced PostgreSQL Vector Handler for Tyra Web Memory System.

Advanced PostgreSQL handler with pgvector integration for semantic memory storage,
similarity search, staleness detection, and comprehensive vector operations.
Built on SQLAlchemy with full async support and zero external dependencies.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    String, Text, DateTime, Float, Integer, JSON, Boolean,
    select, delete, update, func, and_, or_, text
)
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import structlog
from pydantic import BaseModel, Field, ConfigDict

from ..core.utils.config import settings

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class ChunkStatus(str, Enum):
    """Status of memory chunks."""
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"
    FLAGGED = "flagged"


class MemoryChunk(Base):
    """
    Memory chunk model with vector embeddings.
    
    Stores text chunks with their vector embeddings, metadata, and tracking information
    for the Tyra web memory system.
    """
    __tablename__ = "memory_chunks"
    
    # Primary identification
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(500))
    source: Mapped[Optional[str]] = mapped_column(String(2000))
    
    # Vector embedding (adjustable dimension)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(dim=1024))  # Default for e5-large-v2
    
    # Metadata
    metadata_: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, name='metadata')
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Quality and confidence metrics
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    freshness_score: Mapped[float] = mapped_column(Float, default=1.0)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Status and tracking
    status: Mapped[str] = mapped_column(String(50), default=ChunkStatus.ACTIVE.value)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    
    # Content characteristics
    content_length: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    content_type: Mapped[str] = mapped_column(String(50), default="text")
    
    # Query and search context
    original_query: Mapped[Optional[str]] = mapped_column(Text)
    search_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    def __repr__(self) -> str:
        return f"<MemoryChunk(id='{self.id}', title='{self.title[:50]}...', status='{self.status}')>"


@dataclass
class ChunkSearchResult:
    """Result from chunk similarity search."""
    chunk: MemoryChunk
    similarity_score: float
    distance: float
    relevance_score: float
    combined_score: float


@dataclass
class ChunkStats:
    """Statistics about memory chunks."""
    total_chunks: int
    active_chunks: int
    stale_chunks: int
    archived_chunks: int
    flagged_chunks: int
    average_confidence: float
    average_quality: float
    total_size_mb: float
    oldest_chunk_age_days: int
    newest_chunk_age_hours: int


class SearchConfig(BaseModel):
    """Configuration for similarity search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    threshold: float = Field(0.75, ge=0.0, le=1.0, description="Similarity threshold")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    include_stale: bool = Field(False, description="Include stale chunks in results")
    include_archived: bool = Field(False, description="Include archived chunks")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence score")
    boost_recent: bool = Field(True, description="Boost scores for recent content")
    boost_factor: float = Field(0.1, ge=0.0, le=0.5, description="Boost factor for recent content")


class PgVectorHandler:
    """
    Enhanced PostgreSQL Vector Handler with pgvector integration.
    
    Provides comprehensive vector storage, similarity search, staleness detection,
    and memory management for the Tyra web memory system.
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        embedding_dim: int = 1024,
        pool_size: int = 20,
        max_overflow: int = 40,
        stale_threshold_days: int = 30
    ):
        """
        Initialize PostgreSQL vector handler.
        
        Args:
            database_url: PostgreSQL connection URL
            embedding_dim: Dimension of vector embeddings
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
            stale_threshold_days: Days after which content is considered stale
        """
        self.database_url = database_url or self._build_database_url()
        self.embedding_dim = embedding_dim
        self.stale_threshold_days = stale_threshold_days
        
        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Performance tracking
        self.operation_stats = {
            'total_stores': 0,
            'total_queries': 0,
            'total_updates': 0,
            'total_deletes': 0,
            'average_store_time': 0.0,
            'average_query_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(
            "Initialized PgVectorHandler",
            embedding_dim=embedding_dim,
            stale_threshold_days=stale_threshold_days
        )
    
    def _build_database_url(self) -> str:
        """Build database URL from settings."""
        db_config = settings.memory.postgres
        return (
            f"postgresql+asyncpg://"
            f"{db_config.user}:{db_config.password}@"
            f"{db_config.host}:{db_config.port}/"
            f"{db_config.database}"
        )
    
    async def initialize(self) -> None:
        """Initialize database schema and extensions."""
        try:
            async with self.engine.begin() as conn:
                # Enable pgvector extension
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
                
                # Create vector index for similarity search
                index_sql = f"""
                CREATE INDEX IF NOT EXISTS memory_chunks_embedding_idx 
                ON memory_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
                await conn.execute(text(index_sql))
                
                # Create additional indexes for common queries
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_status ON memory_chunks(status)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_created_at ON memory_chunks(created_at DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_confidence ON memory_chunks(confidence_score DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_source ON memory_chunks(source)",
                    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_query ON memory_chunks USING gin(to_tsvector('english', original_query))",
                    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_text ON memory_chunks USING gin(to_tsvector('english', text))",
                ]
                
                for index_sql in indexes:
                    await conn.execute(text(index_sql))
                
                await conn.commit()
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            raise
    
    async def store_chunk(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
        original_query: Optional[str] = None,
        confidence_score: float = 0.0,
        quality_score: float = 0.0
    ) -> str:
        """
        Store a memory chunk with vector embedding.
        
        Args:
            text: Text content of the chunk
            embedding: Vector embedding of the text
            metadata: Additional metadata
            title: Optional title
            source: Source URL or identifier
            original_query: Original query that led to this content
            confidence_score: Confidence score for the content
            quality_score: Quality score for the content
            
        Returns:
            ID of the stored chunk
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate embedding dimension
            if len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension {len(embedding)} does not match "
                    f"expected dimension {self.embedding_dim}"
                )
            
            chunk_id = str(uuid.uuid4())
            
            chunk = MemoryChunk(
                id=chunk_id,
                text=text,
                title=title,
                source=source,
                embedding=embedding,
                metadata_=metadata or {},
                original_query=original_query,
                confidence_score=confidence_score,
                quality_score=quality_score,
                content_length=len(text),
                freshness_score=1.0,  # New content is fresh
                relevance_score=confidence_score,  # Use confidence as initial relevance
                language=self._detect_language(text),
                content_type="text"
            )
            
            async with self.async_session() as session:
                session.add(chunk)
                await session.commit()
                await session.refresh(chunk)
            
            store_time = (datetime.utcnow() - start_time).total_seconds()
            self.operation_stats['total_stores'] += 1
            self.operation_stats['average_store_time'] = (
                (self.operation_stats['average_store_time'] * (self.operation_stats['total_stores'] - 1) + 
                 store_time) / self.operation_stats['total_stores']
            )
            
            logger.info(
                "Stored memory chunk successfully",
                chunk_id=chunk_id,
                text_length=len(text),
                source=source,
                store_time_seconds=store_time
            )
            
            return chunk_id
            
        except Exception as e:
            logger.error("Failed to store memory chunk", error=str(e))
            raise
    
    async def query_similar_chunks(
        self,
        query_embedding: List[float],
        config: Optional[SearchConfig] = None,
        threshold: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[ChunkSearchResult]:
        """
        Query for similar memory chunks using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            config: Search configuration
            threshold: Similarity threshold (overrides config)
            limit: Result limit (overrides config)
            
        Returns:
            List of similar chunks with similarity scores
        """
        start_time = datetime.utcnow()
        
        try:
            # Use provided config or create default
            search_config = config or SearchConfig()
            if threshold is not None:
                search_config.threshold = threshold
            if limit is not None:
                search_config.limit = limit
            
            # Validate embedding dimension
            if len(query_embedding) != self.embedding_dim:
                raise ValueError(
                    f"Query embedding dimension {len(query_embedding)} does not match "
                    f"expected dimension {self.embedding_dim}"
                )
            
            async with self.async_session() as session:
                # Build base query
                query = select(
                    MemoryChunk,
                    (1 - MemoryChunk.embedding.cosine_distance(query_embedding)).label('similarity'),
                    MemoryChunk.embedding.cosine_distance(query_embedding).label('distance')
                )
                
                # Apply status filters
                status_filters = [ChunkStatus.ACTIVE.value]
                if search_config.include_stale:
                    status_filters.append(ChunkStatus.STALE.value)
                if search_config.include_archived:
                    status_filters.append(ChunkStatus.ARCHIVED.value)
                
                query = query.filter(MemoryChunk.status.in_(status_filters))
                
                # Apply confidence filter
                if search_config.min_confidence > 0:
                    query = query.filter(MemoryChunk.confidence_score >= search_config.min_confidence)
                
                # Apply similarity threshold
                query = query.filter(
                    (1 - MemoryChunk.embedding.cosine_distance(query_embedding)) >= search_config.threshold
                )
                
                # Order by similarity and limit
                query = query.order_by(
                    (1 - MemoryChunk.embedding.cosine_distance(query_embedding)).desc()
                ).limit(search_config.limit)
                
                result = await session.execute(query)
                rows = result.fetchall()
            
            # Process results
            search_results = []
            for row in rows:
                chunk, similarity, distance = row
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(
                    chunk, similarity, search_config.boost_recent, search_config.boost_factor
                )
                
                # Calculate combined score
                combined_score = (similarity + relevance_score + chunk.confidence_score) / 3
                
                search_result = ChunkSearchResult(
                    chunk=chunk,
                    similarity_score=similarity,
                    distance=distance,
                    relevance_score=relevance_score,
                    combined_score=combined_score
                )
                search_results.append(search_result)
                
                # Update access tracking
                await self._update_access_tracking(chunk.id)
            
            # Sort by combined score
            search_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self.operation_stats['total_queries'] += 1
            self.operation_stats['average_query_time'] = (
                (self.operation_stats['average_query_time'] * (self.operation_stats['total_queries'] - 1) + 
                 query_time) / self.operation_stats['total_queries']
            )
            
            logger.info(
                "Similarity query completed",
                results_count=len(search_results),
                threshold=search_config.threshold,
                query_time_seconds=query_time
            )
            
            return search_results
            
        except Exception as e:
            logger.error("Similarity query failed", error=str(e))
            raise
    
    async def flag_stale_chunks(
        self,
        days: Optional[int] = None,
        batch_size: int = 1000
    ) -> int:
        """
        Flag chunks as stale based on age.
        
        Args:
            days: Days threshold (uses default if not provided)
            batch_size: Number of chunks to process in each batch
            
        Returns:
            Number of chunks flagged as stale
        """
        days = days or self.stale_threshold_days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        total_flagged = 0
        
        try:
            async with self.async_session() as session:
                # Find stale chunks in batches
                offset = 0
                while True:
                    query = select(MemoryChunk.id).filter(
                        and_(
                            MemoryChunk.created_at < cutoff_date,
                            MemoryChunk.status == ChunkStatus.ACTIVE.value
                        )
                    ).offset(offset).limit(batch_size)
                    
                    result = await session.execute(query)
                    chunk_ids = [row[0] for row in result.fetchall()]
                    
                    if not chunk_ids:
                        break
                    
                    # Update status to stale
                    update_query = update(MemoryChunk).where(
                        MemoryChunk.id.in_(chunk_ids)
                    ).values(
                        status=ChunkStatus.STALE.value,
                        updated_at=datetime.utcnow()
                    )
                    
                    result = await session.execute(update_query)
                    flagged_count = result.rowcount
                    total_flagged += flagged_count
                    
                    await session.commit()
                    
                    logger.info(
                        "Flagged batch of stale chunks",
                        batch_size=flagged_count,
                        total_flagged=total_flagged
                    )
                    
                    offset += batch_size
                    
                    if flagged_count < batch_size:
                        break
            
            logger.info(
                "Stale chunk flagging completed",
                total_flagged=total_flagged,
                days_threshold=days
            )
            
            return total_flagged
            
        except Exception as e:
            logger.error("Failed to flag stale chunks", error=str(e))
            raise
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a specific chunk by ID."""
        try:
            async with self.async_session() as session:
                query = delete(MemoryChunk).where(MemoryChunk.id == chunk_id)
                result = await session.execute(query)
                await session.commit()
                
                deleted = result.rowcount > 0
                if deleted:
                    self.operation_stats['total_deletes'] += 1
                    logger.info("Deleted memory chunk", chunk_id=chunk_id)
                else:
                    logger.warning("Chunk not found for deletion", chunk_id=chunk_id)
                
                return deleted
                
        except Exception as e:
            logger.error("Failed to delete chunk", chunk_id=chunk_id, error=str(e))
            raise
    
    async def update_chunk(
        self,
        chunk_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update chunk fields."""
        try:
            # Add updated timestamp
            updates['updated_at'] = datetime.utcnow()
            
            async with self.async_session() as session:
                query = update(MemoryChunk).where(
                    MemoryChunk.id == chunk_id
                ).values(**updates)
                
                result = await session.execute(query)
                await session.commit()
                
                updated = result.rowcount > 0
                if updated:
                    self.operation_stats['total_updates'] += 1
                    logger.info("Updated memory chunk", chunk_id=chunk_id, updates=list(updates.keys()))
                else:
                    logger.warning("Chunk not found for update", chunk_id=chunk_id)
                
                return updated
                
        except Exception as e:
            logger.error("Failed to update chunk", chunk_id=chunk_id, error=str(e))
            raise
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[MemoryChunk]:
        """Get chunk by ID."""
        try:
            async with self.async_session() as session:
                query = select(MemoryChunk).where(MemoryChunk.id == chunk_id)
                result = await session.execute(query)
                chunk = result.scalar_one_or_none()
                
                if chunk:
                    await self._update_access_tracking(chunk_id)
                
                return chunk
                
        except Exception as e:
            logger.error("Failed to get chunk", chunk_id=chunk_id, error=str(e))
            raise
    
    async def get_chunks_by_source(
        self,
        source: str,
        limit: int = 100
    ) -> List[MemoryChunk]:
        """Get chunks from a specific source."""
        try:
            async with self.async_session() as session:
                query = select(MemoryChunk).filter(
                    MemoryChunk.source == source
                ).order_by(MemoryChunk.created_at.desc()).limit(limit)
                
                result = await session.execute(query)
                chunks = result.scalars().all()
                
                return list(chunks)
                
        except Exception as e:
            logger.error("Failed to get chunks by source", source=source, error=str(e))
            raise
    
    async def get_stats(self) -> ChunkStats:
        """Get comprehensive statistics about stored chunks."""
        try:
            async with self.async_session() as session:
                # Count by status
                status_query = select(
                    MemoryChunk.status,
                    func.count(MemoryChunk.id).label('count')
                ).group_by(MemoryChunk.status)
                
                status_result = await session.execute(status_query)
                status_counts = {row.status: row.count for row in status_result.fetchall()}
                
                # Overall statistics
                stats_query = select(
                    func.count(MemoryChunk.id).label('total'),
                    func.avg(MemoryChunk.confidence_score).label('avg_confidence'),
                    func.avg(MemoryChunk.quality_score).label('avg_quality'),
                    func.sum(MemoryChunk.content_length).label('total_size'),
                    func.min(MemoryChunk.created_at).label('oldest'),
                    func.max(MemoryChunk.created_at).label('newest')
                )
                
                stats_result = await session.execute(stats_query)
                stats_row = stats_result.fetchone()
                
                # Calculate ages
                now = datetime.utcnow()
                oldest_age_days = (now - stats_row.oldest).days if stats_row.oldest else 0
                newest_age_hours = (now - stats_row.newest).total_seconds() / 3600 if stats_row.newest else 0
                
                return ChunkStats(
                    total_chunks=stats_row.total or 0,
                    active_chunks=status_counts.get(ChunkStatus.ACTIVE.value, 0),
                    stale_chunks=status_counts.get(ChunkStatus.STALE.value, 0),
                    archived_chunks=status_counts.get(ChunkStatus.ARCHIVED.value, 0),
                    flagged_chunks=status_counts.get(ChunkStatus.FLAGGED.value, 0),
                    average_confidence=float(stats_row.avg_confidence or 0),
                    average_quality=float(stats_row.avg_quality or 0),
                    total_size_mb=float(stats_row.total_size or 0) / (1024 * 1024),
                    oldest_chunk_age_days=oldest_age_days,
                    newest_chunk_age_hours=int(newest_age_hours)
                )
                
        except Exception as e:
            logger.error("Failed to get statistics", error=str(e))
            raise
    
    async def _update_access_tracking(self, chunk_id: str) -> None:
        """Update access tracking for a chunk."""
        try:
            async with self.async_session() as session:
                query = update(MemoryChunk).where(
                    MemoryChunk.id == chunk_id
                ).values(
                    last_accessed=datetime.utcnow(),
                    access_count=MemoryChunk.access_count + 1
                )
                
                await session.execute(query)
                await session.commit()
                
        except Exception as e:
            logger.warning("Failed to update access tracking", chunk_id=chunk_id, error=str(e))
            # Don't raise - access tracking failure shouldn't break queries
    
    def _calculate_relevance_score(
        self,
        chunk: MemoryChunk,
        similarity: float,
        boost_recent: bool,
        boost_factor: float
    ) -> float:
        """Calculate relevance score with optional recency boost."""
        base_score = (similarity + chunk.confidence_score + chunk.quality_score) / 3
        
        if boost_recent:
            # Apply recency boost (newer content gets higher scores)
            age_hours = (datetime.utcnow() - chunk.created_at).total_seconds() / 3600
            # Boost decays over 30 days
            recency_boost = boost_factor * max(0, 1 - (age_hours / (30 * 24)))
            base_score += recency_boost
        
        return min(base_score, 1.0)
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Simple language detection (could be enhanced with proper language detection)."""
        # Very basic heuristic - could integrate langdetect library if needed
        if len(text) < 10:
            return None
        
        # Check for common English patterns
        english_indicators = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']
        text_lower = text.lower()
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        if english_count >= 3:
            return 'en'
        
        return None
    
    async def cleanup_stale_chunks(
        self,
        days: int = 90,
        batch_size: int = 1000
    ) -> int:
        """Permanently delete very old stale chunks."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        total_deleted = 0
        
        try:
            async with self.async_session() as session:
                offset = 0
                while True:
                    # Find very old stale chunks
                    query = select(MemoryChunk.id).filter(
                        and_(
                            MemoryChunk.created_at < cutoff_date,
                            MemoryChunk.status == ChunkStatus.STALE.value
                        )
                    ).offset(offset).limit(batch_size)
                    
                    result = await session.execute(query)
                    chunk_ids = [row[0] for row in result.fetchall()]
                    
                    if not chunk_ids:
                        break
                    
                    # Delete chunks
                    delete_query = delete(MemoryChunk).where(
                        MemoryChunk.id.in_(chunk_ids)
                    )
                    
                    result = await session.execute(delete_query)
                    deleted_count = result.rowcount
                    total_deleted += deleted_count
                    
                    await session.commit()
                    
                    offset += batch_size
                    
                    if deleted_count < batch_size:
                        break
            
            logger.info(
                "Cleanup completed",
                total_deleted=total_deleted,
                days_threshold=days
            )
            
            return total_deleted
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            raise
    
    async def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation performance statistics."""
        return {
            **self.operation_stats,
            'success_rate_queries': 1.0,  # Could track failures separately
            'average_results_per_query': 0.0,  # Could track this
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of PostgreSQL connection and operations."""
        try:
            async with self.async_session() as session:
                # Test basic connectivity
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
                
                # Test vector extension
                result = await session.execute(text("SELECT vector_dims('[1,2,3]'::vector)"))
                dims = result.scalar()
                
                # Get basic stats
                stats = await self.get_stats()
                
                return {
                    'status': 'healthy',
                    'database_connected': True,
                    'vector_extension_available': dims == 3,
                    'total_chunks': stats.total_chunks,
                    'active_chunks': stats.active_chunks,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()
        logger.info("Database connections closed")


# Example usage
async def example_usage():
    """Example of using PgVectorHandler."""
    handler = PgVectorHandler()
    
    try:
        # Initialize
        await handler.initialize()
        
        # Store a chunk
        embedding = [0.1] * 1024  # Example embedding
        chunk_id = await handler.store_chunk(
            text="This is a test memory chunk",
            embedding=embedding,
            title="Test Chunk",
            source="example://test",
            confidence_score=0.8
        )
        
        # Query similar chunks
        results = await handler.query_similar_chunks(
            query_embedding=embedding,
            threshold=0.5
        )
        
        print(f"Stored chunk: {chunk_id}")
        print(f"Found {len(results)} similar chunks")
        
        # Get statistics
        stats = await handler.get_stats()
        print(f"Total chunks: {stats.total_chunks}")
        
    finally:
        await handler.close()


if __name__ == "__main__":
    asyncio.run(example_usage())