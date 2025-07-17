"""
Intent-Aware Retrieval System for Advanced RAG Pipeline.

This module provides intelligent retrieval that adapts to query intent, context,
and user preferences for optimal memory selection. All processing is performed
locally with zero external API calls.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import json
import math
import heapq

# ML and retrieval imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import pandas as pd

# Graph analysis
import networkx as nx

# NLP imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import structlog
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent

from ...models.memory import Memory
from ..embeddings.embedder import Embedder
from ..cache.redis_cache import RedisCache
from .dynamic_reranking import QueryIntent, QueryContext, DynamicReranker
from .chunk_linking import ContextualChunkLinker, TextChunk, LinkType
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class RetrievalStrategy(str, Enum):
    """Different retrieval strategies."""
    VECTOR_SIMILARITY = "vector_similarity"     # Pure vector search
    HYBRID_SEARCH = "hybrid_search"            # Vector + keyword search
    INTENT_GUIDED = "intent_guided"            # Intent-aware retrieval
    CONTEXTUAL_EXPANSION = "contextual_expansion"  # Context-expanded search
    MULTI_HOP = "multi_hop"                    # Multi-hop reasoning
    TEMPORAL_AWARE = "temporal_aware"          # Time-aware retrieval
    PERSONALIZED = "personalized"             # User-specific retrieval
    ENSEMBLE = "ensemble"                      # Multiple strategy combination


class RetrievalMode(str, Enum):
    """Retrieval modes for different use cases."""
    PRECISE = "precise"        # High precision, low recall
    COMPREHENSIVE = "comprehensive"  # High recall, broader results
    BALANCED = "balanced"      # Balanced precision/recall
    EXPLORATORY = "exploratory"  # Discovery-focused
    FOCUSED = "focused"        # Narrow, specific results


class FilterCriteria(str, Enum):
    """Criteria for filtering memories."""
    RECENCY = "recency"
    QUALITY = "quality"
    RELEVANCE = "relevance"
    POPULARITY = "popularity"
    SOURCE = "source"
    DOMAIN = "domain"
    ENTITY = "entity"
    SENTIMENT = "sentiment"


@dataclass
class RetrievalContext:
    """Extended context for retrieval operations."""
    query_intent: QueryIntent
    conversation_history: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    domain_context: Optional[str] = None
    temporal_context: Optional[datetime] = None
    spatial_context: Optional[str] = None
    task_context: Optional[str] = None
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    filter_criteria: List[FilterCriteria] = field(default_factory=list)
    excluded_memory_ids: Set[str] = field(default_factory=set)
    required_entities: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Result from intent-aware retrieval."""
    memory: Memory
    relevance_score: float
    intent_alignment: float
    context_score: float
    final_score: float
    retrieval_path: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalAnalytics(BaseModel):
    """Analytics from retrieval operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    total_candidates: int = Field(description="Total candidate memories")
    filtered_count: int = Field(description="Memories after filtering")
    final_count: int = Field(description="Final retrieved memories")
    processing_time: float = Field(description="Processing time in seconds")
    strategy_used: RetrievalStrategy = Field(description="Retrieval strategy used")
    intent_detected: QueryIntent = Field(description="Detected query intent")
    expansion_performed: bool = Field(description="Whether context expansion was used")
    filters_applied: List[str] = Field(description="Filters applied")


class IntentAwareRetriever:
    """
    Advanced intent-aware retrieval system for RAG pipeline.
    
    Features:
    - Multi-strategy retrieval with intent adaptation
    - Context-aware query expansion
    - Temporal and spatial awareness
    - Multi-hop reasoning through chunk relationships
    - Personalized retrieval based on user preferences
    - Quality-based filtering and ranking
    - Ensemble retrieval combining multiple approaches
    """
    
    def __init__(
        self,
        embedder: Embedder,
        reranker: DynamicReranker,
        chunk_linker: Optional[ContextualChunkLinker] = None,
        cache: Optional[RedisCache] = None,
        spacy_model: str = "en_core_web_sm",
        default_top_k: int = 20,
        expansion_factor: float = 1.5,
        temporal_weight: float = 0.1
    ):
        """
        Initialize the intent-aware retriever.
        
        Args:
            embedder: Text embedder for similarity calculation
            reranker: Dynamic reranker for result optimization
            chunk_linker: Optional chunk linker for contextual expansion
            cache: Optional Redis cache for performance
            spacy_model: spaCy model for NLP processing
            default_top_k: Default number of results to retrieve
            expansion_factor: Factor for query expansion
            temporal_weight: Weight for temporal relevance
        """
        self.embedder = embedder
        self.reranker = reranker
        self.chunk_linker = chunk_linker
        self.cache = cache
        self.default_top_k = default_top_k
        self.expansion_factor = expansion_factor
        self.temporal_weight = temporal_weight
        
        # Initialize spaCy for NLP processing
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(f"SpaCy model {spacy_model} not found")
        
        # Initialize TF-IDF for keyword-based retrieval
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=2
        )
        self.tfidf_fitted = False
        
        # Initialize intent-specific strategies
        self._init_intent_strategies()
        
        # User interaction tracking
        self.user_profiles = defaultdict(lambda: {
            'preferred_domains': Counter(),
            'query_patterns': Counter(),
            'interaction_history': [],
            'feedback_scores': []
        })
        
        # Performance analytics
        self.retrieval_analytics = []
        
        logger.info(
            "Initialized intent-aware retriever",
            spacy_available=SPACY_AVAILABLE,
            default_top_k=default_top_k,
            expansion_factor=expansion_factor
        )
    
    def _init_intent_strategies(self):
        """Initialize intent-specific retrieval strategies."""
        self.intent_strategies = {
            QueryIntent.FACTUAL: {
                'strategy': RetrievalStrategy.HYBRID_SEARCH,
                'mode': RetrievalMode.PRECISE,
                'weights': {'semantic': 0.6, 'keyword': 0.4},
                'expansion': False,
                'filters': [FilterCriteria.QUALITY, FilterCriteria.RELEVANCE]
            },
            QueryIntent.PROCEDURAL: {
                'strategy': RetrievalStrategy.CONTEXTUAL_EXPANSION,
                'mode': RetrievalMode.COMPREHENSIVE,
                'weights': {'semantic': 0.5, 'keyword': 0.3, 'structure': 0.2},
                'expansion': True,
                'filters': [FilterCriteria.QUALITY]
            },
            QueryIntent.ANALYTICAL: {
                'strategy': RetrievalStrategy.MULTI_HOP,
                'mode': RetrievalMode.COMPREHENSIVE,
                'weights': {'semantic': 0.7, 'context': 0.3},
                'expansion': True,
                'filters': [FilterCriteria.QUALITY, FilterCriteria.RELEVANCE]
            },
            QueryIntent.TEMPORAL: {
                'strategy': RetrievalStrategy.TEMPORAL_AWARE,
                'mode': RetrievalMode.BALANCED,
                'weights': {'semantic': 0.4, 'temporal': 0.4, 'keyword': 0.2},
                'expansion': False,
                'filters': [FilterCriteria.RECENCY, FilterCriteria.RELEVANCE]
            },
            QueryIntent.COMPARATIVE: {
                'strategy': RetrievalStrategy.HYBRID_SEARCH,
                'mode': RetrievalMode.COMPREHENSIVE,
                'weights': {'semantic': 0.6, 'keyword': 0.4},
                'expansion': True,
                'filters': [FilterCriteria.QUALITY]
            },
            QueryIntent.EXPLORATORY: {
                'strategy': RetrievalStrategy.ENSEMBLE,
                'mode': RetrievalMode.EXPLORATORY,
                'weights': {'semantic': 0.4, 'diversity': 0.3, 'keyword': 0.3},
                'expansion': True,
                'filters': [FilterCriteria.QUALITY]
            }
        }
    
    async def retrieve_memories(
        self,
        query: str,
        memories: List[Memory],
        context: Optional[RetrievalContext] = None,
        strategy: Optional[RetrievalStrategy] = None,
        mode: Optional[RetrievalMode] = None,
        top_k: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Tuple[List[RetrievalResult], RetrievalAnalytics]:
        """
        Retrieve memories using intent-aware strategies.
        
        Args:
            query: Search query
            memories: List of candidate memories
            context: Optional retrieval context
            strategy: Optional override for retrieval strategy
            mode: Optional override for retrieval mode
            top_k: Number of results to return
            user_id: Optional user ID for personalization
            
        Returns:
            Tuple of (retrieval results, analytics)
        """
        start_time = datetime.utcnow()
        
        # Initialize context if not provided
        if context is None:
            intent = await self._detect_query_intent(query)
            context = RetrievalContext(query_intent=intent)
        
        # Determine strategy and mode
        if strategy is None or mode is None:
            intent_config = self.intent_strategies.get(
                context.query_intent, 
                self.intent_strategies[QueryIntent.EXPLORATORY]
            )
            strategy = strategy or intent_config['strategy']
            mode = mode or intent_config['mode']
        
        top_k = top_k or self.default_top_k
        
        # Apply initial filtering
        filtered_memories = await self._apply_filters(memories, context)
        
        # Perform retrieval based on strategy
        candidates = await self._execute_retrieval_strategy(
            query, filtered_memories, context, strategy, mode
        )
        
        # Apply personalization if user provided
        if user_id:
            candidates = await self._apply_personalization(candidates, user_id, context)
        
        # Final ranking and selection
        final_results = await self._final_ranking_and_selection(
            query, candidates, context, top_k
        )
        
        # Generate analytics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        analytics = RetrievalAnalytics(
            total_candidates=len(memories),
            filtered_count=len(filtered_memories),
            final_count=len(final_results),
            processing_time=processing_time,
            strategy_used=strategy,
            intent_detected=context.query_intent,
            expansion_performed=strategy in [
                RetrievalStrategy.CONTEXTUAL_EXPANSION,
                RetrievalStrategy.MULTI_HOP,
                RetrievalStrategy.ENSEMBLE
            ],
            filters_applied=[f.value for f in context.filter_criteria]
        )
        
        logger.info(
            "Completed intent-aware retrieval",
            query_intent=context.query_intent.value,
            strategy=strategy.value,
            mode=mode.value,
            total_candidates=len(memories),
            final_results=len(final_results),
            processing_time=processing_time
        )
        
        return final_results, analytics
    
    async def _detect_query_intent(self, query: str) -> QueryIntent:
        """Detect query intent using pattern matching and NLP."""
        # Use the reranker's intent detection
        return await self.reranker._detect_query_intent(query)
    
    async def _apply_filters(
        self, 
        memories: List[Memory], 
        context: RetrievalContext
    ) -> List[Memory]:
        """Apply filtering criteria to memories."""
        filtered = memories.copy()
        
        # Exclude specified memories
        if context.excluded_memory_ids:
            filtered = [m for m in filtered if m.id not in context.excluded_memory_ids]
        
        # Apply filter criteria
        for criterion in context.filter_criteria:
            if criterion == FilterCriteria.RECENCY:
                filtered = await self._filter_by_recency(filtered, context)
            elif criterion == FilterCriteria.QUALITY:
                filtered = await self._filter_by_quality(filtered)
            elif criterion == FilterCriteria.DOMAIN:
                filtered = await self._filter_by_domain(filtered, context)
            elif criterion == FilterCriteria.ENTITY:
                filtered = await self._filter_by_entities(filtered, context)
        
        return filtered
    
    async def _filter_by_recency(
        self, 
        memories: List[Memory], 
        context: RetrievalContext
    ) -> List[Memory]:
        """Filter memories by recency."""
        if not context.temporal_context:
            return memories
        
        # Keep memories from last 30 days by default
        cutoff_date = context.temporal_context - timedelta(days=30)
        
        return [
            m for m in memories 
            if hasattr(m, 'created_at') and m.created_at and m.created_at >= cutoff_date
        ]
    
    async def _filter_by_quality(self, memories: List[Memory]) -> List[Memory]:
        """Filter memories by quality score."""
        # Calculate quality scores and keep top 80%
        quality_scores = []
        
        for memory in memories:
            score = self._calculate_quality_score(memory)
            quality_scores.append((memory, score))
        
        # Sort by quality and keep top 80%
        quality_scores.sort(key=lambda x: x[1], reverse=True)
        top_80_percent = int(len(quality_scores) * 0.8)
        
        return [item[0] for item in quality_scores[:top_80_percent]]
    
    async def _filter_by_domain(
        self, 
        memories: List[Memory], 
        context: RetrievalContext
    ) -> List[Memory]:
        """Filter memories by domain context."""
        if not context.domain_context:
            return memories
        
        return [
            m for m in memories
            if hasattr(m, 'metadata') and 
               m.metadata and 
               m.metadata.get('domain') == context.domain_context
        ]
    
    async def _filter_by_entities(
        self, 
        memories: List[Memory], 
        context: RetrievalContext
    ) -> List[Memory]:
        """Filter memories by required entities."""
        if not context.required_entities:
            return memories
        
        filtered = []
        for memory in memories:
            # Extract entities from memory content
            entities = await self._extract_entities(memory.content)
            
            # Check if any required entity is present
            if any(entity.lower() in [e.lower() for e in entities] 
                   for entity in context.required_entities):
                filtered.append(memory)
        
        return filtered
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]
    
    def _calculate_quality_score(self, memory: Memory) -> float:
        """Calculate quality score for a memory."""
        # Use the reranker's quality calculation
        return self.reranker._calculate_quality_score(memory)
    
    async def _execute_retrieval_strategy(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        strategy: RetrievalStrategy,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Execute the specified retrieval strategy."""
        
        if strategy == RetrievalStrategy.VECTOR_SIMILARITY:
            return await self._vector_similarity_retrieval(query, memories, context, mode)
        
        elif strategy == RetrievalStrategy.HYBRID_SEARCH:
            return await self._hybrid_search_retrieval(query, memories, context, mode)
        
        elif strategy == RetrievalStrategy.INTENT_GUIDED:
            return await self._intent_guided_retrieval(query, memories, context, mode)
        
        elif strategy == RetrievalStrategy.CONTEXTUAL_EXPANSION:
            return await self._contextual_expansion_retrieval(query, memories, context, mode)
        
        elif strategy == RetrievalStrategy.MULTI_HOP:
            return await self._multi_hop_retrieval(query, memories, context, mode)
        
        elif strategy == RetrievalStrategy.TEMPORAL_AWARE:
            return await self._temporal_aware_retrieval(query, memories, context, mode)
        
        elif strategy == RetrievalStrategy.PERSONALIZED:
            return await self._personalized_retrieval(query, memories, context, mode)
        
        elif strategy == RetrievalStrategy.ENSEMBLE:
            return await self._ensemble_retrieval(query, memories, context, mode)
        
        else:
            # Default to vector similarity
            return await self._vector_similarity_retrieval(query, memories, context, mode)
    
    async def _vector_similarity_retrieval(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Pure vector similarity-based retrieval."""
        query_embedding = await self.embedder.embed(query)
        results = []
        
        for memory in memories:
            # Calculate similarity
            if hasattr(memory, 'embedding') and memory.embedding is not None:
                similarity = float(cosine_similarity([query_embedding], [memory.embedding])[0][0])
            else:
                memory_embedding = await self.embedder.embed(memory.content)
                similarity = float(cosine_similarity([query_embedding], [memory_embedding])[0][0])
            
            result = RetrievalResult(
                memory=memory,
                relevance_score=similarity,
                intent_alignment=0.5,  # Neutral for pure vector search
                context_score=0.0,
                final_score=similarity,
                explanation="Vector similarity match"
            )
            results.append(result)
        
        return results
    
    async def _hybrid_search_retrieval(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Hybrid vector + keyword search retrieval."""
        
        # Vector similarity component
        vector_results = await self._vector_similarity_retrieval(query, memories, context, mode)
        
        # Keyword similarity component
        keyword_scores = await self._calculate_keyword_scores(query, memories)
        
        # Combine scores
        intent_config = self.intent_strategies.get(context.query_intent, {})
        weights = intent_config.get('weights', {'semantic': 0.7, 'keyword': 0.3})
        
        results = []
        for i, vector_result in enumerate(vector_results):
            keyword_score = keyword_scores[i]
            
            hybrid_score = (
                vector_result.relevance_score * weights.get('semantic', 0.7) +
                keyword_score * weights.get('keyword', 0.3)
            )
            
            result = RetrievalResult(
                memory=vector_result.memory,
                relevance_score=vector_result.relevance_score,
                intent_alignment=keyword_score,  # Use keyword score as intent proxy
                context_score=0.0,
                final_score=hybrid_score,
                explanation=f"Hybrid search (vector: {vector_result.relevance_score:.3f}, keyword: {keyword_score:.3f})"
            )
            results.append(result)
        
        return results
    
    async def _calculate_keyword_scores(self, query: str, memories: List[Memory]) -> List[float]:
        """Calculate keyword-based similarity scores."""
        # Prepare documents
        documents = [query] + [memory.content for memory in memories]
        
        # Fit TF-IDF if not already fitted or refit if needed
        if not self.tfidf_fitted or len(documents) > 1000:
            self.tfidf_vectorizer.fit(documents)
            self.tfidf_fitted = True
        
        # Transform documents
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform(documents)
        except:
            # Fallback: refit and transform
            self.tfidf_vectorizer.fit(documents)
            tfidf_matrix = self.tfidf_vectorizer.transform(documents)
        
        # Calculate similarities
        query_vector = tfidf_matrix[0:1]
        memory_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, memory_vectors)[0]
        return similarities.tolist()
    
    async def _intent_guided_retrieval(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Intent-guided retrieval with intent-specific scoring."""
        
        # Start with hybrid search
        results = await self._hybrid_search_retrieval(query, memories, context, mode)
        
        # Apply intent-specific adjustments
        for result in results:
            intent_score = await self._calculate_intent_specific_score(
                result.memory, context.query_intent
            )
            
            # Adjust final score based on intent alignment
            result.intent_alignment = intent_score
            result.final_score = (
                result.relevance_score * 0.6 +
                result.intent_alignment * 0.4
            )
            result.explanation += f"; Intent alignment: {intent_score:.3f}"
        
        return results
    
    async def _calculate_intent_specific_score(
        self, 
        memory: Memory, 
        intent: QueryIntent
    ) -> float:
        """Calculate intent-specific relevance score."""
        # Use the reranker's intent alignment calculation
        return await self.reranker._calculate_intent_alignment(memory, intent)
    
    async def _contextual_expansion_retrieval(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Retrieval with contextual query expansion."""
        
        # Expand query with context
        expanded_query = await self._expand_query_with_context(query, context)
        
        # Perform hybrid search with expanded query
        results = await self._hybrid_search_retrieval(expanded_query, memories, context, mode)
        
        # Add context scores
        for result in results:
            context_score = await self._calculate_context_score(result.memory, context)
            result.context_score = context_score
            result.final_score = (
                result.relevance_score * 0.5 +
                result.intent_alignment * 0.3 +
                result.context_score * 0.2
            )
            result.explanation += f"; Context expansion used"
        
        return results
    
    async def _expand_query_with_context(
        self, 
        query: str, 
        context: RetrievalContext
    ) -> str:
        """Expand query using contextual information."""
        expanded_parts = [query]
        
        # Add conversation context
        if context.conversation_history:
            recent_context = " ".join(context.conversation_history[-2:])
            expanded_parts.append(recent_context)
        
        # Add domain context
        if context.domain_context:
            expanded_parts.append(context.domain_context)
        
        # Add task context
        if context.task_context:
            expanded_parts.append(context.task_context)
        
        # Add required entities
        if context.required_entities:
            expanded_parts.extend(context.required_entities)
        
        return " ".join(expanded_parts)
    
    async def _calculate_context_score(
        self, 
        memory: Memory, 
        context: RetrievalContext
    ) -> float:
        """Calculate context relevance score."""
        # Use the reranker's context relevance calculation
        query_context = QueryContext(
            intent=context.query_intent,
            conversation_history=context.conversation_history,
            user_preferences=context.user_preferences,
            domain=context.domain_context,
            temporal_context=context.temporal_context,
            task_context=context.task_context
        )
        return await self.reranker._calculate_context_relevance(memory, query_context)
    
    async def _multi_hop_retrieval(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Multi-hop retrieval using chunk relationships."""
        
        if not self.chunk_linker:
            # Fallback to contextual expansion
            return await self._contextual_expansion_retrieval(query, memories, context, mode)
        
        # Initial retrieval
        initial_results = await self._hybrid_search_retrieval(query, memories, context, mode)
        
        # Expand through chunk relationships
        expanded_results = []
        processed_ids = set()
        
        for result in initial_results[:10]:  # Limit initial candidates
            if result.memory.id in processed_ids:
                continue
            
            # Get related chunks
            related_chunks = await self._get_related_chunks(result.memory, memories)
            
            for related_memory, relation_score in related_chunks:
                if related_memory.id not in processed_ids:
                    # Calculate multi-hop score
                    multi_hop_score = result.final_score * relation_score * 0.8
                    
                    expanded_result = RetrievalResult(
                        memory=related_memory,
                        relevance_score=multi_hop_score,
                        intent_alignment=result.intent_alignment,
                        context_score=relation_score,
                        final_score=multi_hop_score,
                        retrieval_path=[result.memory.id, related_memory.id],
                        explanation=f"Multi-hop via {result.memory.id} (relation: {relation_score:.3f})"
                    )
                    expanded_results.append(expanded_result)
                    processed_ids.add(related_memory.id)
            
            # Add original result
            expanded_results.append(result)
            processed_ids.add(result.memory.id)
        
        return expanded_results
    
    async def _get_related_chunks(
        self, 
        memory: Memory, 
        all_memories: List[Memory]
    ) -> List[Tuple[Memory, float]]:
        """Get related chunks using chunk linking."""
        # This is a simplified version - in practice, would use chunk_linker
        related = []
        
        for other_memory in all_memories:
            if other_memory.id == memory.id:
                continue
            
            # Simple similarity-based relationship
            if hasattr(memory, 'embedding') and hasattr(other_memory, 'embedding'):
                if memory.embedding is not None and other_memory.embedding is not None:
                    similarity = float(cosine_similarity([memory.embedding], [other_memory.embedding])[0][0])
                    if similarity > 0.7:  # High similarity threshold
                        related.append((other_memory, similarity))
        
        # Sort by similarity and return top 3
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:3]
    
    async def _temporal_aware_retrieval(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Temporal-aware retrieval with time-based scoring."""
        
        # Start with hybrid search
        results = await self._hybrid_search_retrieval(query, memories, context, mode)
        
        # Apply temporal scoring
        for result in results:
            temporal_score = self._calculate_temporal_score(result.memory, context)
            
            # Adjust final score with temporal component
            result.final_score = (
                result.relevance_score * 0.6 +
                result.intent_alignment * 0.2 +
                temporal_score * self.temporal_weight * 2
            )
            result.explanation += f"; Temporal score: {temporal_score:.3f}"
        
        return results
    
    def _calculate_temporal_score(self, memory: Memory, context: RetrievalContext) -> float:
        """Calculate temporal relevance score."""
        if not hasattr(memory, 'created_at') or not memory.created_at:
            return 0.5  # Neutral score for unknown timestamps
        
        if not context.temporal_context:
            # Use current time as reference
            reference_time = datetime.utcnow()
        else:
            reference_time = context.temporal_context
        
        # Calculate days difference
        days_diff = abs((reference_time - memory.created_at).days)
        
        # Exponential decay: more recent = higher score
        temporal_score = math.exp(-days_diff / 30.0)  # 30-day half-life
        return min(temporal_score, 1.0)
    
    async def _personalized_retrieval(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Personalized retrieval based on user preferences."""
        
        # Start with intent-guided retrieval
        results = await self._intent_guided_retrieval(query, memories, context, mode)
        
        # Apply personalization if user preferences available
        if context.user_preferences:
            for result in results:
                personalization_score = self._calculate_personalization_score(
                    result.memory, context.user_preferences
                )
                
                # Adjust final score with personalization
                result.final_score = (
                    result.final_score * 0.8 +
                    personalization_score * 0.2
                )
                result.explanation += f"; Personalization: {personalization_score:.3f}"
        
        return results
    
    def _calculate_personalization_score(
        self, 
        memory: Memory, 
        user_preferences: Dict[str, Any]
    ) -> float:
        """Calculate personalization score based on user preferences."""
        score = 0.5  # Neutral baseline
        
        # Domain preferences
        preferred_domains = user_preferences.get('domains', [])
        if preferred_domains and hasattr(memory, 'metadata') and memory.metadata:
            memory_domain = memory.metadata.get('domain', '')
            if memory_domain in preferred_domains:
                score += 0.3
        
        # Content type preferences
        preferred_types = user_preferences.get('content_types', [])
        if preferred_types and hasattr(memory, 'metadata') and memory.metadata:
            content_type = memory.metadata.get('type', '')
            if content_type in preferred_types:
                score += 0.2
        
        return min(score, 1.0)
    
    async def _ensemble_retrieval(
        self,
        query: str,
        memories: List[Memory],
        context: RetrievalContext,
        mode: RetrievalMode
    ) -> List[RetrievalResult]:
        """Ensemble retrieval combining multiple strategies."""
        
        # Run multiple strategies
        strategies_to_run = [
            RetrievalStrategy.HYBRID_SEARCH,
            RetrievalStrategy.INTENT_GUIDED,
            RetrievalStrategy.CONTEXTUAL_EXPANSION
        ]
        
        all_results = []
        for strategy in strategies_to_run:
            strategy_results = await self._execute_retrieval_strategy(
                query, memories, context, strategy, mode
            )
            
            # Weight results by strategy
            weight = 1.0 / len(strategies_to_run)
            for result in strategy_results:
                result.final_score *= weight
                result.explanation += f" (ensemble: {strategy.value})"
            
            all_results.extend(strategy_results)
        
        # Merge results by memory ID
        merged_results = {}
        for result in all_results:
            memory_id = result.memory.id
            if memory_id in merged_results:
                # Combine scores
                existing = merged_results[memory_id]
                existing.final_score += result.final_score
                existing.explanation += "; " + result.explanation
            else:
                merged_results[memory_id] = result
        
        return list(merged_results.values())
    
    async def _apply_personalization(
        self,
        candidates: List[RetrievalResult],
        user_id: str,
        context: RetrievalContext
    ) -> List[RetrievalResult]:
        """Apply user-specific personalization to candidates."""
        
        user_profile = self.user_profiles[user_id]
        
        # Update user preferences based on context
        if context.domain_context:
            user_profile['preferred_domains'][context.domain_context] += 1
        
        # Apply personalization scoring
        for result in candidates:
            # Domain preference scoring
            if hasattr(result.memory, 'metadata') and result.memory.metadata:
                memory_domain = result.memory.metadata.get('domain', '')
                domain_score = user_profile['preferred_domains'].get(memory_domain, 0)
                domain_boost = min(domain_score / 10.0, 0.2)  # Max 20% boost
                result.final_score += domain_boost
        
        return candidates
    
    async def _final_ranking_and_selection(
        self,
        query: str,
        candidates: List[RetrievalResult],
        context: RetrievalContext,
        top_k: int
    ) -> List[RetrievalResult]:
        """Final ranking and selection of top results."""
        
        # Sort by final score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply diversity if in exploratory mode
        if context.query_intent == QueryIntent.EXPLORATORY:
            candidates = await self._apply_diversity_selection(candidates, top_k)
        
        # Select top-k
        final_results = candidates[:top_k]
        
        # Add final explanations and metadata
        for i, result in enumerate(final_results):
            result.metadata.update({
                'rank': i + 1,
                'total_candidates': len(candidates),
                'query_intent': context.query_intent.value,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return final_results
    
    async def _apply_diversity_selection(
        self,
        candidates: List[RetrievalResult],
        target_count: int
    ) -> List[RetrievalResult]:
        """Apply diversity-aware selection to avoid redundant results."""
        
        if len(candidates) <= target_count:
            return candidates
        
        # Use MMR (Maximal Marginal Relevance) approach
        selected = []
        remaining = candidates.copy()
        
        # Select first result (highest score)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining results balancing relevance and diversity
        while len(selected) < target_count and remaining:
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Calculate diversity score
                diversity_score = 1.0
                for selected_result in selected:
                    if (hasattr(candidate.memory, 'embedding') and 
                        hasattr(selected_result.memory, 'embedding') and
                        candidate.memory.embedding is not None and
                        selected_result.memory.embedding is not None):
                        
                        similarity = float(cosine_similarity(
                            [candidate.memory.embedding],
                            [selected_result.memory.embedding]
                        )[0][0])
                        diversity_score *= (1.0 - similarity)
                
                # MMR score: balance relevance and diversity
                mmr_score = 0.7 * candidate.final_score + 0.3 * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        
        return selected
    
    async def update_user_feedback(
        self,
        user_id: str,
        query: str,
        selected_results: List[str],
        feedback_scores: Optional[List[float]] = None
    ):
        """Update user profile based on feedback."""
        
        user_profile = self.user_profiles[user_id]
        
        # Update interaction history
        interaction = {
            'query': query,
            'selected_ids': selected_results,
            'feedback_scores': feedback_scores,
            'timestamp': datetime.utcnow()
        }
        user_profile['interaction_history'].append(interaction)
        
        # Keep only recent interactions
        if len(user_profile['interaction_history']) > 100:
            user_profile['interaction_history'] = user_profile['interaction_history'][-100:]
        
        # Update feedback scores
        if feedback_scores:
            user_profile['feedback_scores'].extend(feedback_scores)
            if len(user_profile['feedback_scores']) > 1000:
                user_profile['feedback_scores'] = user_profile['feedback_scores'][-1000:]
    
    async def get_retrieval_analytics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval analytics."""
        
        if not self.retrieval_analytics:
            return {}
        
        # Calculate aggregate metrics
        total_retrievals = len(self.retrieval_analytics)
        avg_processing_time = np.mean([a.processing_time for a in self.retrieval_analytics])
        
        # Strategy usage distribution
        strategy_usage = Counter([a.strategy_used.value for a in self.retrieval_analytics])
        
        # Intent distribution
        intent_distribution = Counter([a.intent_detected.value for a in self.retrieval_analytics])
        
        return {
            'total_retrievals': total_retrievals,
            'avg_processing_time': avg_processing_time,
            'strategy_usage': dict(strategy_usage),
            'intent_distribution': dict(intent_distribution),
            'avg_candidates_filtered': np.mean([a.filtered_count for a in self.retrieval_analytics]),
            'avg_final_results': np.mean([a.final_count for a in self.retrieval_analytics]),
            'expansion_usage_rate': np.mean([a.expansion_performed for a in self.retrieval_analytics])
        }