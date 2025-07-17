"""Memory retrieval system with multi-strategy support."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.logger import get_logger
from ..cache.redis_cache import CacheManager
from ..interfaces.embeddings import EmbeddingProvider
from ..interfaces.graph_engine import GraphEngine
from ..interfaces.reranker import Reranker
from ..observability.tracing import trace_method
from ..rag.hallucination_detector import HallucinationDetector
from .models import (
    MemorySearchResult,
    RetrievalContext,
    Memory,
    ConfidenceLevel
)
from .postgres_client import PostgresClient

logger = get_logger(__name__)


class MemoryRetriever:
    """Advanced memory retrieval system with multiple strategies."""
    
    def __init__(
        self,
        postgres_client: PostgresClient,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[Reranker] = None,
        graph_engine: Optional[GraphEngine] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
        cache_manager: Optional[CacheManager] = None,
        vector_weight: float = 0.7,
        diversity_penalty: float = 0.1
    ):
        """Initialize memory retriever.
        
        Args:
            postgres_client: PostgreSQL client for vector operations
            embedding_provider: Embedding generation provider
            reranker: Optional reranking model
            graph_engine: Optional graph database engine
            hallucination_detector: Optional hallucination detection
            cache_manager: Optional cache for results
            vector_weight: Weight for vector similarity in hybrid search
            diversity_penalty: Penalty for similar results
        """
        self.postgres_client = postgres_client
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.graph_engine = graph_engine
        self.hallucination_detector = hallucination_detector
        self.cache_manager = cache_manager
        self.vector_weight = vector_weight
        self.diversity_penalty = diversity_penalty
        
    @trace_method("memory_retrieve")
    async def retrieve(
        self,
        context: RetrievalContext,
        strategy: str = "hybrid"
    ) -> List[MemorySearchResult]:
        """Retrieve memories using specified strategy.
        
        Args:
            context: Retrieval context with query and filters
            strategy: Retrieval strategy (vector, text, hybrid, graph-enhanced)
            
        Returns:
            List of memory search results
        """
        # Check cache first
        cache_key = f"retrieval:{strategy}:{context.query}:{context.agent_id}:{context.top_k}"
        if self.cache_manager:
            cached = await self.cache_manager.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for retrieval: {cache_key}")
                return cached
                
        # Generate embedding for query
        query_embedding = await self._get_query_embedding(context.query)
        
        # Execute retrieval strategy
        if strategy == "vector":
            results = await self._vector_retrieve(context, query_embedding)
        elif strategy == "text":
            results = await self._text_retrieve(context)
        elif strategy == "hybrid":
            results = await self._hybrid_retrieve(context, query_embedding)
        elif strategy == "graph-enhanced":
            results = await self._graph_enhanced_retrieve(context, query_embedding)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
            
        # Apply reranking if enabled
        if context.rerank and self.reranker and len(results) > 0:
            results = await self._rerank_results(context.query, results)
            
        # Apply diversity penalty
        if self.diversity_penalty > 0:
            results = self._apply_diversity_penalty(results)
            
        # Filter by minimum score
        results = [r for r in results if r.score >= context.min_score]
        
        # Limit to top_k
        results = results[:context.top_k]
        
        # Cache results
        if self.cache_manager and results:
            await self.cache_manager.set(cache_key, results, ttl=3600)
            
        logger.info(f"Retrieved {len(results)} memories using {strategy} strategy")
        return results
        
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query with caching."""
        cache_key = f"embedding:query:{query}"
        
        if self.cache_manager:
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return np.array(cached)
                
        embedding = await self.embedding_provider.embed_query(query)
        
        if self.cache_manager:
            await self.cache_manager.set(cache_key, embedding.tolist(), ttl=86400)
            
        return embedding
        
    async def _vector_retrieve(
        self,
        context: RetrievalContext,
        query_embedding: np.ndarray
    ) -> List[MemorySearchResult]:
        """Pure vector similarity retrieval."""
        results = await self.postgres_client.search(
            query_embedding=query_embedding,
            top_k=context.top_k * 2,  # Get extra for filtering
            filters=context.filters,
            agent_id=context.agent_id
        )
        
        return [
            MemorySearchResult(
                memory=Memory(
                    id=result[0],
                    text=result[2]['text'],
                    agent_id=result[2].get('agent_id'),
                    metadata={k: v for k, v in result[2].items() 
                             if k not in ['text', 'agent_id', 'created_at']},
                    created_at=result[2]['created_at']
                ),
                score=result[1],
                distance=1 - result[1],
                source="vector"
            )
            for result in results
        ]
        
    async def _text_retrieve(self, context: RetrievalContext) -> List[MemorySearchResult]:
        """Pure text-based retrieval using PostgreSQL full-text search."""
        # This would require implementing text search in PostgresClient
        # For now, return empty list
        logger.warning("Text-only retrieval not fully implemented")
        return []
        
    async def _hybrid_retrieve(
        self,
        context: RetrievalContext,
        query_embedding: np.ndarray
    ) -> List[MemorySearchResult]:
        """Hybrid retrieval combining vector and text search."""
        results = await self.postgres_client.hybrid_search(
            query_text=context.query,
            query_embedding=query_embedding,
            top_k=context.top_k * 2,
            vector_weight=self.vector_weight,
            agent_id=context.agent_id
        )
        
        return [
            MemorySearchResult(
                memory=Memory(
                    id=result[0],
                    text=result[2]['text'],
                    agent_id=result[2].get('agent_id'),
                    metadata={k: v for k, v in result[2].items() 
                             if k not in ['text', 'agent_id', 'created_at', 'vector_score', 'text_score']},
                    created_at=result[2]['created_at']
                ),
                score=result[1],
                distance=1 - result[2].get('vector_score', 0),
                source="hybrid"
            )
            for result in results
        ]
        
    async def _graph_enhanced_retrieve(
        self,
        context: RetrievalContext,
        query_embedding: np.ndarray
    ) -> List[MemorySearchResult]:
        """Graph-enhanced retrieval using knowledge graph relationships."""
        # First get vector results
        vector_results = await self._vector_retrieve(context, query_embedding)
        
        if not self.graph_engine or not vector_results:
            return vector_results
            
        # Extract entities from query and top results
        entities = await self._extract_entities_from_results(
            context.query,
            vector_results[:5]
        )
        
        if not entities:
            return vector_results
            
        # Query graph for related memories
        graph_memories = await self._query_graph_for_memories(entities)
        
        # Merge and re-score results
        merged_results = self._merge_graph_results(vector_results, graph_memories)
        
        return merged_results
        
    async def _rerank_results(
        self,
        query: str,
        results: List[MemorySearchResult]
    ) -> List[MemorySearchResult]:
        """Apply reranking to improve result relevance."""
        if not results:
            return results
            
        # Prepare texts for reranking
        texts = [r.memory.text for r in results]
        
        # Get reranking scores
        scores = await self.reranker.rerank(query, texts)
        
        # Update results with rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = score
            # Combine original score with rerank score
            result.score = (result.score + score) / 2
            
        # Re-sort by new combined score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
        
    def _apply_diversity_penalty(
        self,
        results: List[MemorySearchResult]
    ) -> List[MemorySearchResult]:
        """Apply penalty to similar results to increase diversity."""
        if len(results) <= 1:
            return results
            
        # Simple similarity check based on text overlap
        penalized_results = []
        seen_texts = set()
        
        for result in results:
            # Create text fingerprint (first 100 chars)
            fingerprint = result.memory.text[:100].lower()
            
            # Check similarity with seen results
            similarity_penalty = 0
            for seen in seen_texts:
                if self._text_similarity(fingerprint, seen) > 0.8:
                    similarity_penalty = self.diversity_penalty
                    break
                    
            # Apply penalty
            result.score *= (1 - similarity_penalty)
            penalized_results.append(result)
            seen_texts.add(fingerprint)
            
        # Re-sort after penalties
        penalized_results.sort(key=lambda x: x.score, reverse=True)
        
        return penalized_results
        
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on character overlap."""
        if not text1 or not text2:
            return 0.0
            
        # Simple Jaccard similarity on character n-grams
        ngrams1 = set(text1[i:i+3] for i in range(len(text1)-2))
        ngrams2 = set(text2[i:i+3] for i in range(len(text2)-2))
        
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
        
    async def _extract_entities_from_results(
        self,
        query: str,
        results: List[MemorySearchResult]
    ) -> List[str]:
        """Extract entities from query and results for graph lookup."""
        entities = set()
        
        # Extract entities from query using improved keyword extraction
        query_entities = self._extract_query_entities(query)
        entities.update(query_entities)
        
        # Extract entities from memory results content
        for result in results[:3]:  # Limit to top 3 results for performance
            content_entities = self._extract_content_entities(result.content)
            entities.update(content_entities)
        
        # Return sorted list, prioritizing longer entities
        entity_list = sorted(list(entities), key=len, reverse=True)
        return entity_list[:8]  # Limit to 8 most relevant entities
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entities from query text."""
        entities = []
        
        # Remove common question words and prepositions
        stop_words = {
            'what', 'when', 'where', 'how', 'why', 'who', 'which', 'whose',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'about'
        }
        
        # Clean and tokenize query
        words = query.lower().replace('?', '').replace('!', '').split()
        
        # Extract meaningful words as potential entities
        for word in words:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            
            # Filter by length and stop words
            if (len(clean_word) >= 3 and 
                clean_word not in stop_words and
                clean_word.isalpha()):
                entities.append(clean_word)
        
        # Look for potential multi-word entities (simple approach)
        query_lower = query.lower()
        # Common patterns for names, places, etc.
        import re
        
        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend([name.lower() for name in capitalized])
        
        # Extract quoted phrases as entities
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend([phrase.lower() for phrase in quoted if len(phrase) > 2])
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_content_entities(self, content: str) -> List[str]:
        """Extract potential entities from memory content."""
        if not content or len(content) < 10:
            return []
        
        entities = []
        
        # Extract capitalized words (likely proper nouns)
        import re
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # Filter and clean capitalized words
        for entity in capitalized:
            clean_entity = entity.strip().lower()
            if (len(clean_entity) >= 3 and 
                not clean_entity.startswith(('the', 'a ', 'an ')) and
                clean_entity not in ['this', 'that', 'these', 'those']):
                entities.append(clean_entity)
        
        # Extract common entity patterns
        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content)
        entities.extend(dates)
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        entities.extend([email.lower() for email in emails])
        
        # URLs
        urls = re.findall(r'https?://[^\s]+', content)
        entities.extend(urls)
        
        # Remove duplicates and return
        return list(set(entities))[:5]  # Limit content entities
        
    async def _query_graph_for_memories(
        self,
        entities: List[str]
    ) -> List[Tuple[str, float]]:
        """Query knowledge graph for memories related to entities."""
        if not self.graph_engine:
            return []
            
        try:
            related_memories = []
            
            if entities:
                # Query graph for memories connected to any of the entities
                entity_list = "', '".join(entities)
                query = f"""
                MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN ['{entity_list}']
                RETURN DISTINCT m.id as memory_id, 
                       COALESCE(m.relevance_score, 0.5) as relevance_score,
                       COUNT(e) as entity_matches
                ORDER BY entity_matches DESC, relevance_score DESC
                LIMIT 20
                """
                
                results = await self.graph_engine.execute_query(query)
                
                for record in results:
                    memory_id = record.get("memory_id")
                    base_score = float(record.get("relevance_score", 0.5))
                    entity_matches = int(record.get("entity_matches", 1))
                    
                    # Boost score based on number of matching entities
                    boosted_score = min(1.0, base_score * (1 + 0.1 * entity_matches))
                    related_memories.append((memory_id, boosted_score))
                    
            return related_memories
            
        except Exception as e:
            logger.error(f"Error querying graph for memories: {e}")
            return []
        
    def _merge_graph_results(
        self,
        vector_results: List[MemorySearchResult],
        graph_memories: List[Tuple[str, float]]
    ) -> List[MemorySearchResult]:
        """Merge vector and graph results with boosted scores."""
        # Create mapping of memory IDs to results
        result_map = {str(r.memory.id): r for r in vector_results}
        
        # Boost scores for memories found in graph
        for memory_id, graph_score in graph_memories:
            if memory_id in result_map:
                # Boost existing result
                result_map[memory_id].score *= (1 + graph_score * 0.2)
                
        # Re-sort by boosted scores
        merged = list(result_map.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        
        return merged
        
    async def check_hallucination(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[MemorySearchResult]
    ) -> Tuple[bool, float, ConfidenceLevel]:
        """Check if response contains hallucinations.
        
        Args:
            query: Original query
            response: Generated response
            retrieved_chunks: Retrieved memory chunks
            
        Returns:
            Tuple of (has_hallucination, confidence_score, confidence_level)
        """
        if not self.hallucination_detector:
            # No detector available, assume no hallucination but low confidence
            return False, 60.0, ConfidenceLevel.FUZZY
            
        # Prepare source texts
        sources = [r.memory.text for r in retrieved_chunks]
        
        # Check for hallucination
        result = await self.hallucination_detector.detect(
            query=query,
            response=response,
            sources=sources
        )
        
        # Determine confidence level
        if result.confidence_score >= 95:
            level = ConfidenceLevel.ROCK_SOLID
        elif result.confidence_score >= 80:
            level = ConfidenceLevel.HIGH
        elif result.confidence_score >= 60:
            level = ConfidenceLevel.FUZZY
        else:
            level = ConfidenceLevel.LOW
            
        return result.has_hallucination, result.confidence_score, level