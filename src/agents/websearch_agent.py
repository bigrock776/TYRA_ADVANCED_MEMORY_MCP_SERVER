"""
Local Web Search Agent for Tyra Memory System.

Advanced web search agent that performs local web search using duckduckgo-search,
content extraction with trafilatura and newspaper3k, local LLM summarization,
and integration with the memory system. All operations are performed locally
with zero external API dependencies.
"""

import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urljoin, urlparse
import aiohttp
import trafilatura
from newspaper import Article
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ..core.clients.vllm_client import VLLMClient, ChatMessage, Role
from ..core.providers.embeddings.embedder import Embedder
from ..memory.pgvector_handler import PgVectorHandler
from ..memory.neo4j_linker import Neo4jLinker
from ..validators.memory_confidence import MemoryConfidenceAgent
from ..core.utils.config import settings

logger = structlog.get_logger(__name__)


class SearchMethod(str, Enum):
    """Web search methods."""
    DUCKDUCKGO = "duckduckgo"
    LOCAL_FALLBACK = "local_fallback"
    CACHED_RESULTS = "cached_results"


class ContentExtractor(str, Enum):
    """Content extraction methods."""
    TRAFILATURA = "trafilatura"
    NEWSPAPER = "newspaper3k"
    BEAUTIFULSOUP = "beautifulsoup"
    HYBRID = "hybrid"


class ExtractionQuality(str, Enum):
    """Quality levels for extracted content."""
    EXCELLENT = "excellent"  # High confidence, clean extraction
    GOOD = "good"           # Good quality, minor issues
    FAIR = "fair"           # Acceptable quality, some problems
    POOR = "poor"           # Low quality, extraction issues


class WebSearchResult(BaseModel):
    """Structured web search result with comprehensive metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Core content
    text: str = Field(..., min_length=10, description="Extracted and processed text content")
    title: str = Field(..., min_length=1, description="Page title")
    source: str = Field(..., description="Source URL")
    
    # Embeddings and processing
    embedding: List[float] = Field(..., min_items=384, description="Content embedding vector")
    summary: Optional[str] = Field(None, description="LLM-generated summary")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Extraction timestamp")
    extraction_method: ContentExtractor = Field(..., description="Method used for content extraction")
    extraction_quality: ExtractionQuality = Field(..., description="Quality assessment of extraction")
    
    # Confidence and validation
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    freshness_score: float = Field(..., ge=0.0, le=1.0, description="Content freshness score")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Query relevance score")
    
    # Technical metadata
    content_length: int = Field(..., ge=0, description="Original content length")
    processed_length: int = Field(..., ge=0, description="Processed content length")
    language: Optional[str] = Field(None, description="Detected language")
    encoding: Optional[str] = Field(None, description="Character encoding")
    
    # Extraction details
    extraction_time_seconds: float = Field(..., ge=0.0, description="Time taken for extraction")
    error_count: int = Field(default=0, ge=0, description="Number of extraction errors")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    
    @field_validator('source')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format."""
        if not re.match(r'^https?://', v):
            raise ValueError(f"Invalid URL format: {v}")
        return v
    
    @field_validator('text')
    @classmethod
    def validate_content_quality(cls, v):
        """Basic content quality validation."""
        if len(v.strip()) < 10:
            raise ValueError("Content too short after extraction")
        # Check for common extraction failures
        if v.count('\n') / len(v) > 0.1:  # Too many newlines
            raise ValueError("Content appears to have extraction issues")
        return v


class LocalWebSearcher:
    """
    Local web search implementation using DuckDuckGo.
    
    Provides web search capabilities without external API keys or dependencies.
    """
    
    def __init__(
        self,
        max_results: int = 10,
        timeout: int = 30,
        user_agent: str = None
    ):
        """
        Initialize local web searcher.
        
        Args:
            max_results: Maximum search results to return
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.max_results = max_results
        self.timeout = timeout
        self.user_agent = user_agent or "Tyra-Local-Search/1.0"
        
        # Search history for deduplication
        self.search_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("Initialized LocalWebSearcher", max_results=max_results)
    
    async def search(
        self,
        query: str,
        region: str = "us-en",
        safesearch: str = "moderate",
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform local web search using DuckDuckGo.
        
        Args:
            query: Search query
            region: Search region (us-en, uk-en, etc.)
            safesearch: Safe search setting (strict, moderate, off)
            max_results: Override default max results
            
        Returns:
            List of search results with title, href, body
        """
        start_time = datetime.utcnow()
        max_results = max_results or self.max_results
        
        try:
            # Check search history for recent duplicates
            query_hash = hashlib.md5(query.encode()).hexdigest()
            if query_hash in self.search_history:
                recent_search = self.search_history[query_hash]
                if len(recent_search) > 0:
                    search_time = datetime.fromisoformat(recent_search[0].get('search_time', '1970-01-01T00:00:00'))
                    if datetime.utcnow() - search_time < timedelta(minutes=10):
                        logger.info("Using cached search results", query=query)
                        return recent_search
            
            logger.info("Performing local web search", query=query, max_results=max_results)
            
            # Use DuckDuckGo search
            with DDGS() as ddgs:
                search_results = []
                
                # Perform text search
                try:
                    results = ddgs.text(
                        keywords=query,
                        region=region,
                        safesearch=safesearch,
                        max_results=max_results
                    )
                    
                    for result in results:
                        search_result = {
                            'title': result.get('title', ''),
                            'href': result.get('href', ''),
                            'body': result.get('body', ''),
                            'search_time': datetime.utcnow().isoformat(),
                            'search_method': SearchMethod.DUCKDUCKGO.value
                        }
                        search_results.append(search_result)
                        
                        if len(search_results) >= max_results:
                            break
                
                except Exception as e:
                    logger.error("DuckDuckGo search failed", error=str(e))
                    # Could implement fallback search methods here
                    raise
            
            # Cache results
            self.search_history[query_hash] = search_results
            
            search_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                "Search completed",
                query=query,
                results_count=len(search_results),
                search_time_seconds=search_time
            )
            
            return search_results
            
        except Exception as e:
            logger.error("Web search failed", query=query, error=str(e))
            raise


class ContentExtractorEngine:
    """
    Advanced content extraction engine supporting multiple methods.
    
    Provides robust content extraction with fallback methods and quality assessment.
    """
    
    def __init__(self, timeout: int = 30):
        """Initialize content extractor."""
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'User-Agent': 'Tyra-Content-Extractor/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def extract_content(
        self,
        url: str,
        method: ContentExtractor = ContentExtractor.HYBRID,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Extract content from URL using specified method.
        
        Args:
            url: URL to extract content from
            method: Extraction method to use
            max_retries: Maximum retry attempts
            
        Returns:
            Dictionary with extracted content and metadata
        """
        start_time = datetime.utcnow()
        
        for attempt in range(max_retries + 1):
            try:
                if method == ContentExtractor.TRAFILATURA:
                    result = await self._extract_with_trafilatura(url)
                elif method == ContentExtractor.NEWSPAPER:
                    result = await self._extract_with_newspaper(url)
                elif method == ContentExtractor.BEAUTIFULSOUP:
                    result = await self._extract_with_beautifulsoup(url)
                elif method == ContentExtractor.HYBRID:
                    result = await self._extract_hybrid(url)
                else:
                    raise ValueError(f"Unknown extraction method: {method}")
                
                extraction_time = (datetime.utcnow() - start_time).total_seconds()
                result['extraction_time_seconds'] = extraction_time
                result['retry_count'] = attempt
                
                return result
                
            except Exception as e:
                logger.warning(
                    "Content extraction attempt failed",
                    url=url,
                    method=method.value,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt == max_retries:
                    logger.error("All extraction attempts failed", url=url)
                    raise
                
                # Wait before retry
                await asyncio.sleep(1 * (attempt + 1))
    
    async def _fetch_page(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Fetch page content with metadata."""
        if not self.session:
            raise RuntimeError("ContentExtractor not initialized as async context manager")
        
        async with self.session.get(url) as response:
            response.raise_for_status()
            
            content = await response.text()
            metadata = {
                'status_code': response.status,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(content),
                'encoding': response.charset or 'utf-8',
                'url': str(response.url),  # Final URL after redirects
            }
            
            return content, metadata
    
    async def _extract_with_trafilatura(self, url: str) -> Dict[str, Any]:
        """Extract content using trafilatura."""
        html_content, metadata = await self._fetch_page(url)
        
        # Extract with trafilatura
        extracted = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            include_formatting=False
        )
        
        if not extracted:
            raise ValueError("Trafilatura extraction returned empty content")
        
        # Extract metadata
        doc_metadata = trafilatura.extract_metadata(html_content)
        title = doc_metadata.title if doc_metadata else ""
        
        if not title:
            # Fallback title extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "Untitled"
        
        return {
            'text': extracted,
            'title': title,
            'extraction_method': ContentExtractor.TRAFILATURA.value,
            'extraction_quality': self._assess_quality(extracted, html_content),
            'language': doc_metadata.language if doc_metadata else None,
            'content_length': len(html_content),
            'processed_length': len(extracted),
            'error_count': 0,
            **metadata
        }
    
    async def _extract_with_newspaper(self, url: str) -> Dict[str, Any]:
        """Extract content using newspaper3k."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if not article.text:
                raise ValueError("Newspaper extraction returned empty content")
            
            return {
                'text': article.text,
                'title': article.title or "Untitled",
                'extraction_method': ContentExtractor.NEWSPAPER.value,
                'extraction_quality': self._assess_quality(article.text, article.html),
                'language': article.meta_lang,
                'content_length': len(article.html) if article.html else 0,
                'processed_length': len(article.text),
                'error_count': 0,
                'url': url,
                'encoding': 'utf-8'
            }
            
        except Exception as e:
            logger.error("Newspaper extraction failed", url=url, error=str(e))
            raise
    
    async def _extract_with_beautifulsoup(self, url: str) -> Dict[str, Any]:
        """Extract content using BeautifulSoup (fallback method)."""
        html_content, metadata = await self._fetch_page(url)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "Untitled"
        
        # Extract main content
        # Try to find main content areas
        content_candidates = []
        for tag in ['main', 'article', 'div[class*="content"]', 'div[class*="main"]']:
            elements = soup.select(tag)
            content_candidates.extend(elements)
        
        if content_candidates:
            # Use the largest content block
            main_content = max(content_candidates, key=lambda x: len(x.get_text()))
            text = main_content.get_text()
        else:
            # Fallback to body
            body = soup.find('body')
            text = body.get_text() if body else soup.get_text()
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text or len(text) < 10:
            raise ValueError("BeautifulSoup extraction returned insufficient content")
        
        return {
            'text': text,
            'title': title,
            'extraction_method': ContentExtractor.BEAUTIFULSOUP.value,
            'extraction_quality': self._assess_quality(text, html_content),
            'language': None,
            'content_length': len(html_content),
            'processed_length': len(text),
            'error_count': 0,
            **metadata
        }
    
    async def _extract_hybrid(self, url: str) -> Dict[str, Any]:
        """Hybrid extraction using multiple methods with quality comparison."""
        methods = [
            ContentExtractor.TRAFILATURA,
            ContentExtractor.NEWSPAPER,
            ContentExtractor.BEAUTIFULSOUP
        ]
        
        results = []
        errors = []
        
        for method in methods:
            try:
                result = await self.extract_content(url, method, max_retries=0)
                results.append(result)
            except Exception as e:
                errors.append(f"{method.value}: {str(e)}")
                continue
        
        if not results:
            raise ValueError(f"All extraction methods failed: {'; '.join(errors)}")
        
        # Select best result based on quality and length
        best_result = max(
            results,
            key=lambda x: (
                self._quality_score(x['extraction_quality']),
                x['processed_length']
            )
        )
        
        best_result['extraction_method'] = ContentExtractor.HYBRID.value
        best_result['error_count'] = len(errors)
        
        return best_result
    
    def _assess_quality(self, text: str, html: str) -> ExtractionQuality:
        """Assess extraction quality based on various factors."""
        if not text or len(text) < 10:
            return ExtractionQuality.POOR
        
        # Calculate metrics
        text_length = len(text)
        html_length = len(html) if html else 1
        extraction_ratio = text_length / html_length
        
        # Check for quality indicators
        has_proper_sentences = len(re.findall(r'[.!?]+', text)) > 0
        has_reasonable_words = len(text.split()) > 5
        not_too_much_whitespace = text.count('\n') / len(text) < 0.05
        
        # Determine quality
        if (extraction_ratio > 0.05 and has_proper_sentences and 
            has_reasonable_words and not_too_much_whitespace and text_length > 100):
            return ExtractionQuality.EXCELLENT
        elif (extraction_ratio > 0.02 and has_proper_sentences and 
              has_reasonable_words and text_length > 50):
            return ExtractionQuality.GOOD
        elif has_reasonable_words and text_length > 20:
            return ExtractionQuality.FAIR
        else:
            return ExtractionQuality.POOR
    
    def _quality_score(self, quality: ExtractionQuality) -> float:
        """Convert quality enum to numeric score."""
        scores = {
            ExtractionQuality.EXCELLENT: 1.0,
            ExtractionQuality.GOOD: 0.75,
            ExtractionQuality.FAIR: 0.5,
            ExtractionQuality.POOR: 0.25
        }
        return scores.get(quality, 0.0)


class WebSearchAgent:
    """
    Advanced Local Web Search Agent for Tyra Memory System.
    
    Provides comprehensive web search capabilities with content extraction,
    local LLM summarization, embedding generation, and memory integration.
    All operations are performed locally with zero external API dependencies.
    """
    
    def __init__(
        self,
        vllm_client: Optional[VLLMClient] = None,
        embedder: Optional[Embedder] = None,
        pgvector_handler: Optional[PgVectorHandler] = None,
        neo4j_linker: Optional[Neo4jLinker] = None,
        confidence_agent: Optional[MemoryConfidenceAgent] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize Web Search Agent.
        
        Args:
            vllm_client: Local LLM client for summarization
            embedder: Embedding provider for vector generation
            pgvector_handler: PostgreSQL vector handler
            neo4j_linker: Neo4j graph linker
            confidence_agent: Memory confidence scoring agent
            max_results: Maximum search results to process
            similarity_threshold: Similarity threshold for memory storage
        """
        self.vllm_client = vllm_client or VLLMClient()
        self.embedder = embedder
        self.pgvector_handler = pgvector_handler
        self.neo4j_linker = neo4j_linker
        self.confidence_agent = confidence_agent
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.searcher = LocalWebSearcher(max_results=max_results)
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'total_extractions': 0,
            'successful_extractions': 0,
            'average_search_time': 0.0,
            'average_extraction_time': 0.0,
        }
        
        logger.info(
            "Initialized WebSearchAgent",
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )
    
    async def search_and_integrate(
        self,
        query: str,
        max_results: Optional[int] = None,
        force_refresh: bool = False
    ) -> List[WebSearchResult]:
        """
        Perform comprehensive web search with memory integration.
        
        Args:
            query: Search query
            max_results: Override default max results
            force_refresh: Force new search even if cached results exist
            
        Returns:
            List of processed and integrated search results
        """
        start_time = datetime.utcnow()
        max_results = max_results or self.max_results
        self.search_stats['total_searches'] += 1
        
        try:
            logger.info("Starting web search and integration", query=query)
            
            # Check for existing similar memories first (unless force refresh)
            if not force_refresh and self.pgvector_handler and self.embedder:
                query_embedding = await self.embedder.embed_text(query)
                similar_memories = await self.pgvector_handler.query_similar_chunks(
                    query_embedding=query_embedding,
                    threshold=self.similarity_threshold,
                    limit=5
                )
                
                if similar_memories:
                    logger.info(
                        "Found similar existing memories, skipping web search",
                        query=query,
                        similar_count=len(similar_memories)
                    )
                    # Convert existing memories to WebSearchResult format
                    return await self._convert_memories_to_results(similar_memories, query)
            
            # Perform web search
            search_results = await self.searcher.search(query, max_results=max_results)
            
            if not search_results:
                logger.warning("No search results found", query=query)
                return []
            
            # Process search results
            processed_results = []
            
            async with ContentExtractorEngine() as extractor:
                for search_result in search_results:
                    try:
                        processed_result = await self._process_search_result(
                            search_result, query, extractor
                        )
                        if processed_result:
                            processed_results.append(processed_result)
                    except Exception as e:
                        logger.error(
                            "Failed to process search result",
                            url=search_result.get('href', 'unknown'),
                            error=str(e)
                        )
                        continue
            
            # Store results in memory system
            if processed_results and self.pgvector_handler:
                await self._store_results_in_memory(processed_results, query)
            
            search_time = (datetime.utcnow() - start_time).total_seconds()
            self.search_stats['successful_searches'] += 1
            self.search_stats['average_search_time'] = (
                (self.search_stats['average_search_time'] * (self.search_stats['successful_searches'] - 1) + 
                 search_time) / self.search_stats['successful_searches']
            )
            
            logger.info(
                "Web search and integration completed",
                query=query,
                results_processed=len(processed_results),
                total_time_seconds=search_time
            )
            
            return processed_results
            
        except Exception as e:
            logger.error("Web search and integration failed", query=query, error=str(e))
            raise
    
    async def _process_search_result(
        self,
        search_result: Dict[str, Any],
        query: str,
        extractor: ContentExtractorEngine
    ) -> Optional[WebSearchResult]:
        """Process individual search result."""
        url = search_result.get('href', '')
        if not url:
            return None
        
        start_time = datetime.utcnow()
        self.search_stats['total_extractions'] += 1
        
        try:
            # Extract content
            content_data = await extractor.extract_content(url)
            
            # Generate summary if LLM is available
            summary = None
            if self.vllm_client:
                summary = await self._generate_summary(content_data['text'], query)
            
            # Generate embedding
            embedding = []
            if self.embedder:
                text_for_embedding = summary or content_data['text']
                embedding = await self.embedder.embed_text(text_for_embedding)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                content_data['text'], query, search_result.get('body', '')
            )
            
            # Calculate freshness score (could be enhanced with actual date detection)
            freshness_score = 0.8  # Default assumption of relatively fresh content
            
            # Calculate overall confidence
            quality_score = extractor._quality_score(
                ExtractionQuality(content_data['extraction_quality'])
            )
            confidence_score = (quality_score + relevance_score + freshness_score) / 3
            
            result = WebSearchResult(
                text=content_data['text'],
                title=content_data['title'],
                source=url,
                embedding=embedding,
                summary=summary,
                extraction_method=ContentExtractor(content_data['extraction_method']),
                extraction_quality=ExtractionQuality(content_data['extraction_quality']),
                confidence_score=confidence_score,
                freshness_score=freshness_score,
                relevance_score=relevance_score,
                content_length=content_data['content_length'],
                processed_length=content_data['processed_length'],
                language=content_data.get('language'),
                encoding=content_data.get('encoding'),
                extraction_time_seconds=content_data['extraction_time_seconds'],
                error_count=content_data['error_count'],
                retry_count=content_data['retry_count']
            )
            
            self.search_stats['successful_extractions'] += 1
            extraction_time = (datetime.utcnow() - start_time).total_seconds()
            self.search_stats['average_extraction_time'] = (
                (self.search_stats['average_extraction_time'] * (self.search_stats['successful_extractions'] - 1) + 
                 extraction_time) / self.search_stats['successful_extractions']
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to process search result", url=url, error=str(e))
            return None
    
    async def _generate_summary(self, text: str, query: str) -> Optional[str]:
        """Generate summary using local LLM."""
        try:
            if not self.vllm_client:
                return None
            
            # Truncate text if too long
            max_length = 2000  # Adjust based on LLM context window
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            messages = [
                ChatMessage(
                    Role.SYSTEM,
                    "You are a helpful assistant that creates concise, accurate summaries. "
                    "Focus on information relevant to the user's query."
                ),
                ChatMessage(
                    Role.USER,
                    f"Please create a concise summary of the following text, "
                    f"focusing on information relevant to this query: '{query}'\n\n"
                    f"Text:\n{text}"
                )
            ]
            
            response = await self.vllm_client.chat(
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )
            
            summary = response.get_text().strip()
            return summary if summary else None
            
        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))
            return None
    
    def _calculate_relevance_score(
        self, 
        content: str, 
        query: str, 
        snippet: str = ""
    ) -> float:
        """Calculate relevance score based on query terms."""
        try:
            # Simple relevance scoring based on term frequency
            query_terms = set(query.lower().split())
            content_lower = content.lower()
            snippet_lower = snippet.lower()
            
            # Count matches in content and snippet
            content_matches = sum(1 for term in query_terms if term in content_lower)
            snippet_matches = sum(1 for term in query_terms if term in snippet_lower)
            
            # Calculate scores
            content_score = content_matches / len(query_terms) if query_terms else 0
            snippet_score = snippet_matches / len(query_terms) if query_terms else 0
            
            # Combine scores (snippet from search engine is pre-filtered)
            relevance_score = (content_score * 0.7) + (snippet_score * 0.3)
            
            return min(relevance_score, 1.0)
            
        except Exception:
            return 0.5  # Default neutral score
    
    async def _convert_memories_to_results(
        self, 
        memories: List[Any], 
        query: str
    ) -> List[WebSearchResult]:
        """Convert existing memory chunks to WebSearchResult format."""
        results = []
        
        for memory in memories:
            try:
                # Extract data from memory object
                # This would depend on the actual memory structure
                result = WebSearchResult(
                    text=getattr(memory, 'text', str(memory)),
                    title=getattr(memory, 'title', 'Cached Memory'),
                    source=getattr(memory, 'source', 'memory://local'),
                    embedding=getattr(memory, 'embedding', []),
                    extraction_method=ContentExtractor.CACHED_RESULTS,
                    extraction_quality=ExtractionQuality.GOOD,
                    confidence_score=getattr(memory, 'confidence', 0.8),
                    freshness_score=0.9,  # Cached results are "fresh" in terms of availability
                    relevance_score=getattr(memory, 'similarity', 0.8),
                    content_length=len(str(memory)),
                    processed_length=len(str(memory)),
                    extraction_time_seconds=0.0
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to convert memory to result", error=str(e))
                continue
        
        return results
    
    async def _store_results_in_memory(
        self,
        results: List[WebSearchResult],
        query: str
    ) -> None:
        """Store search results in memory system."""
        try:
            for result in results:
                # Store in PostgreSQL vector store
                if self.pgvector_handler and result.embedding:
                    metadata = {
                        'title': result.title,
                        'source': result.source,
                        'query': query,
                        'timestamp': result.timestamp.isoformat(),
                        'extraction_method': result.extraction_method.value,
                        'confidence_score': result.confidence_score,
                        'relevance_score': result.relevance_score,
                        'freshness_score': result.freshness_score
                    }
                    
                    await self.pgvector_handler.store_chunk(
                        text=result.text,
                        embedding=result.embedding,
                        metadata=metadata
                    )
                
                # Store in Neo4j graph
                if self.neo4j_linker:
                    await self.neo4j_linker.add_web_search_result(result, query)
                    
        except Exception as e:
            logger.error("Failed to store results in memory", error=str(e))
            # Don't raise - storage failure shouldn't break search
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics."""
        return {
            **self.search_stats,
            'success_rate_search': (
                self.search_stats['successful_searches'] / 
                max(self.search_stats['total_searches'], 1)
            ),
            'success_rate_extraction': (
                self.search_stats['successful_extractions'] / 
                max(self.search_stats['total_extractions'], 1)
            ),
        }
    
    async def clear_cache(self) -> None:
        """Clear search cache."""
        self.searcher.search_history.clear()
        logger.info("Search cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of web search components."""
        health_status = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check VLLMClient
        if self.vllm_client:
            try:
                llm_healthy = await self.vllm_client.health_check()
                health_status['components']['vllm_client'] = {
                    'status': 'healthy' if llm_healthy else 'unhealthy'
                }
            except Exception as e:
                health_status['components']['vllm_client'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Check other components
        for component_name, component in [
            ('embedder', self.embedder),
            ('pgvector_handler', self.pgvector_handler),
            ('neo4j_linker', self.neo4j_linker),
            ('confidence_agent', self.confidence_agent)
        ]:
            if component:
                if hasattr(component, 'health_check'):
                    try:
                        component_health = await component.health_check()
                        health_status['components'][component_name] = component_health
                    except Exception as e:
                        health_status['components'][component_name] = {
                            'status': 'unhealthy',
                            'error': str(e)
                        }
                else:
                    health_status['components'][component_name] = {
                        'status': 'healthy',
                        'note': 'No health check method available'
                    }
        
        # Overall status
        unhealthy_components = [
            name for name, status in health_status['components'].items()
            if status.get('status') == 'unhealthy'
        ]
        
        if unhealthy_components:
            health_status['status'] = 'degraded'
            health_status['unhealthy_components'] = unhealthy_components
        
        return health_status


# Example usage
async def example_usage():
    """Example of using WebSearchAgent."""
    async with VLLMClient() as vllm_client:
        # Initialize agent
        agent = WebSearchAgent(vllm_client=vllm_client)
        
        # Perform search
        results = await agent.search_and_integrate(
            query="latest developments in AI memory systems",
            max_results=5
        )
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Title: {result.title}")
            print(f"Source: {result.source}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Summary: {result.summary}")
            print(f"Text length: {result.processed_length} chars")


if __name__ == "__main__":
    asyncio.run(example_usage())