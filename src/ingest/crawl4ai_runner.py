"""
Crawl4AI Runner for Tyra Web Memory System.

Advanced web crawling integration using Crawl4AI library for intelligent content
extraction, domain-allowlisted crawling, and seamless integration with the memory
system including PostgreSQL and Neo4j storage.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urljoin, urlparse, urlunparse
import time

import structlog
from crawl4ai import AsyncWebCrawler, CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking, NlpSentenceChunking
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ..core.providers.embeddings.embedder import Embedder
from ..memory.pgvector_handler import PgVectorHandler
from ..memory.neo4j_linker import Neo4jLinker
from ..agents.websearch_agent import WebSearchResult, ContentExtractor, ExtractionQuality
from ..validators.memory_confidence import MemoryConfidenceAgent
from ..core.utils.config import settings

logger = structlog.get_logger(__name__)


class CrawlStrategy(str, Enum):
    """Crawling strategies for different content types."""
    SINGLE_PAGE = "single_page"
    SITE_MAP = "site_map"
    RECURSIVE = "recursive"
    RSS_FEED = "rss_feed"
    API_ENDPOINT = "api_endpoint"


class ContentType(str, Enum):
    """Types of content that can be crawled."""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    DOCUMENTATION = "documentation"
    NEWS = "news"
    RESEARCH_PAPER = "research_paper"
    FORUM_POST = "forum_post"
    PRODUCT_PAGE = "product_page"
    GENERAL = "general"


class CrawlStatus(str, Enum):
    """Status of crawl operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"


@dataclass
class CrawlResult:
    """Result of a crawl operation."""
    url: str
    success: bool
    content: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    crawl_time_seconds: float = 0.0
    content_length: int = 0
    extraction_method: str = "crawl4ai"


class DomainPolicy(BaseModel):
    """Domain crawling policy configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    domain: str = Field(..., description="Domain name")
    allowed: bool = Field(True, description="Whether domain is allowed")
    max_pages: int = Field(10, ge=1, le=1000, description="Maximum pages to crawl")
    rate_limit_seconds: float = Field(1.0, ge=0.1, le=60.0, description="Rate limit between requests")
    content_types: List[ContentType] = Field(default_factory=list, description="Allowed content types")
    excluded_paths: List[str] = Field(default_factory=list, description="Excluded URL paths")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    
    @field_validator('domain')
    @classmethod
    def validate_domain(cls, v):
        """Validate domain format."""
        if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError(f"Invalid domain format: {v}")
        return v.lower()


class Crawl4aiRunner:
    """
    Advanced Crawl4AI Runner for Tyra Web Memory System.
    
    Provides intelligent web crawling with domain allowlisting, content extraction,
    embedding generation, and seamless integration with PostgreSQL and Neo4j storage.
    """
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        pgvector_handler: Optional[PgVectorHandler] = None,
        neo4j_linker: Optional[Neo4jLinker] = None,
        confidence_agent: Optional[MemoryConfidenceAgent] = None,
        headless: bool = True,
        browser_type: str = "chromium",
        max_concurrent: int = 3,
        default_delay: float = 1.0
    ):
        """
        Initialize Crawl4AI Runner.
        
        Args:
            embedder: Embedding provider for vector generation
            pgvector_handler: PostgreSQL vector handler
            neo4j_linker: Neo4j graph linker
            confidence_agent: Memory confidence scoring agent
            headless: Run browser in headless mode
            browser_type: Browser type (chromium, firefox, webkit)
            max_concurrent: Maximum concurrent crawl operations
            default_delay: Default delay between requests in seconds
        """
        self.embedder = embedder
        self.pgvector_handler = pgvector_handler
        self.neo4j_linker = neo4j_linker
        self.confidence_agent = confidence_agent
        self.headless = headless
        self.browser_type = browser_type
        self.max_concurrent = max_concurrent
        self.default_delay = default_delay
        
        # Domain policies
        self.domain_policies: Dict[str, DomainPolicy] = {}
        self._load_default_domain_policies()
        
        # Crawl tracking
        self.crawl_history: Dict[str, datetime] = {}
        self.failed_urls: Set[str] = set()
        self.rate_limits: Dict[str, datetime] = {}
        
        # Performance tracking
        self.crawl_stats = {
            'total_crawls': 0,
            'successful_crawls': 0,
            'failed_crawls': 0,
            'skipped_crawls': 0,
            'total_pages_processed': 0,
            'total_content_bytes': 0,
            'average_crawl_time': 0.0,
            'rate_limit_hits': 0,
        }
        
        # Initialize crawler (will be created when needed)
        self.crawler: Optional[AsyncWebCrawler] = None
        
        logger.info(
            "Initialized Crawl4aiRunner",
            headless=headless,
            browser_type=browser_type,
            max_concurrent=max_concurrent
        )
    
    def _load_default_domain_policies(self) -> None:
        """Load default domain policies for common sites."""
        default_policies = [
            DomainPolicy(
                domain="wikipedia.org",
                allowed=True,
                max_pages=50,
                rate_limit_seconds=0.5,
                content_types=[ContentType.ARTICLE, ContentType.GENERAL],
                excluded_paths=["/wiki/Special:", "/wiki/Talk:", "/wiki/File:"]
            ),
            DomainPolicy(
                domain="github.com",
                allowed=True,
                max_pages=20,
                rate_limit_seconds=1.0,
                content_types=[ContentType.DOCUMENTATION, ContentType.GENERAL],
                excluded_paths=["/issues/", "/pulls/", "/commits/"]
            ),
            DomainPolicy(
                domain="stackoverflow.com",
                allowed=True,
                max_pages=30,
                rate_limit_seconds=1.5,
                content_types=[ContentType.FORUM_POST, ContentType.GENERAL],
                excluded_paths=["/users/", "/tags/"]
            ),
            DomainPolicy(
                domain="medium.com",
                allowed=True,
                max_pages=25,
                rate_limit_seconds=2.0,
                content_types=[ContentType.BLOG_POST, ContentType.ARTICLE],
                excluded_paths=["/tag/", "/topics/"]
            ),
            DomainPolicy(
                domain="arxiv.org",
                allowed=True,
                max_pages=15,
                rate_limit_seconds=3.0,
                content_types=[ContentType.RESEARCH_PAPER],
                excluded_paths=["/list/", "/find/"]
            ),
        ]
        
        for policy in default_policies:
            self.domain_policies[policy.domain] = policy
    
    async def initialize(self) -> None:
        """Initialize the Crawl4AI crawler."""
        try:
            self.crawler = AsyncWebCrawler(
                headless=self.headless,
                browser_type=self.browser_type,
                verbose=False
            )
            await self.crawler.astart()
            
            logger.info("Crawl4AI crawler initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Crawl4AI crawler", error=str(e))
            raise
    
    async def add_domain_policy(self, policy: DomainPolicy) -> None:
        """Add or update a domain crawling policy."""
        self.domain_policies[policy.domain] = policy
        logger.info("Added domain policy", domain=policy.domain, allowed=policy.allowed)
    
    async def crawl_url(
        self,
        url: str,
        strategy: CrawlStrategy = CrawlStrategy.SINGLE_PAGE,
        extract_content: bool = True,
        store_in_memory: bool = True,
        force_recrawl: bool = False
    ) -> List[CrawlResult]:
        """
        Crawl a single URL or site using specified strategy.
        
        Args:
            url: URL to crawl
            strategy: Crawling strategy to use
            extract_content: Whether to extract and process content
            store_in_memory: Whether to store results in memory system
            force_recrawl: Force recrawl even if recently crawled
            
        Returns:
            List of crawl results
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate URL and domain policy
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            if not await self._is_url_allowed(url):
                logger.warning("URL not allowed by domain policy", url=url)
                return [CrawlResult(
                    url=url,
                    success=False,
                    error="Domain not allowed",
                    status_code=403
                )]
            
            # Check if recently crawled
            if not force_recrawl and url in self.crawl_history:
                last_crawl = self.crawl_history[url]
                if datetime.utcnow() - last_crawl < timedelta(hours=1):
                    logger.info("URL recently crawled, skipping", url=url)
                    self.crawl_stats['skipped_crawls'] += 1
                    return [CrawlResult(
                        url=url,
                        success=False,
                        error="Recently crawled",
                        status_code=429
                    )]
            
            # Apply rate limiting
            await self._apply_rate_limit(domain)
            
            # Perform crawl based on strategy
            if strategy == CrawlStrategy.SINGLE_PAGE:
                results = await self._crawl_single_page(url, extract_content)
            elif strategy == CrawlStrategy.RECURSIVE:
                results = await self._crawl_recursive(url, extract_content)
            elif strategy == CrawlStrategy.SITE_MAP:
                results = await self._crawl_sitemap(url, extract_content)
            else:
                raise ValueError(f"Unsupported crawl strategy: {strategy}")
            
            # Store in memory system if requested
            if store_in_memory and results:
                await self._store_crawl_results(results, url)
            
            # Update statistics
            self.crawl_stats['total_crawls'] += 1
            if any(r.success for r in results):
                self.crawl_stats['successful_crawls'] += 1
            else:
                self.crawl_stats['failed_crawls'] += 1
            
            crawl_time = (datetime.utcnow() - start_time).total_seconds()
            self.crawl_stats['average_crawl_time'] = (
                (self.crawl_stats['average_crawl_time'] * (self.crawl_stats['total_crawls'] - 1) + 
                 crawl_time) / self.crawl_stats['total_crawls']
            )
            
            # Update crawl history
            self.crawl_history[url] = datetime.utcnow()
            
            logger.info(
                "Crawl completed",
                url=url,
                strategy=strategy.value,
                results_count=len(results),
                successful=sum(1 for r in results if r.success),
                crawl_time_seconds=crawl_time
            )
            
            return results
            
        except Exception as e:
            logger.error("Crawl failed", url=url, error=str(e))
            self.crawl_stats['failed_crawls'] += 1
            self.failed_urls.add(url)
            
            return [CrawlResult(
                url=url,
                success=False,
                error=str(e)
            )]
    
    async def _crawl_single_page(self, url: str, extract_content: bool = True) -> List[CrawlResult]:
        """Crawl a single page."""
        if not self.crawler:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Configure extraction strategy
            extraction_strategy = None
            if extract_content:
                extraction_strategy = LLMExtractionStrategy(
                    provider="ollama/llama2",  # Local LLM
                    api_token="",  # No token needed for local
                    instruction="Extract the main content, title, and key information from this page."
                )
            
            # Perform crawl
            result = await self.crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                chunking_strategy=NlpSentenceChunking(overlap=100),
                bypass_cache=True,
                include_raw_html=False
            )
            
            crawl_time = time.time() - start_time
            
            if result.success:
                return [CrawlResult(
                    url=url,
                    success=True,
                    content=result.markdown or result.cleaned_html,
                    title=self._extract_title(result.cleaned_html),
                    metadata={
                        'links': result.links,
                        'media': result.media,
                        'extracted_content': result.extracted_content,
                        'success': result.success,
                        'status_code': result.status_code,
                        'fit_markdown': getattr(result, 'fit_markdown', False)
                    },
                    status_code=result.status_code,
                    crawl_time_seconds=crawl_time,
                    content_length=len(result.markdown or result.cleaned_html or "")
                )]
            else:
                return [CrawlResult(
                    url=url,
                    success=False,
                    error=f"Crawl failed: {result.status_code}",
                    status_code=result.status_code,
                    crawl_time_seconds=crawl_time
                )]
                
        except Exception as e:
            crawl_time = time.time() - start_time
            logger.error("Single page crawl failed", url=url, error=str(e))
            
            return [CrawlResult(
                url=url,
                success=False,
                error=str(e),
                crawl_time_seconds=crawl_time
            )]
    
    async def _crawl_recursive(self, base_url: str, extract_content: bool = True, max_depth: int = 2) -> List[CrawlResult]:
        """Crawl recursively following links."""
        if not self.crawler:
            await self.initialize()
        
        crawled_urls = set()
        results = []
        url_queue = [(base_url, 0)]  # (url, depth)
        
        parsed_base = urlparse(base_url)
        domain = parsed_base.netloc
        policy = self.domain_policies.get(domain)
        max_pages = policy.max_pages if policy else 10
        
        while url_queue and len(results) < max_pages:
            current_url, depth = url_queue.pop(0)
            
            if current_url in crawled_urls or depth > max_depth:
                continue
            
            # Crawl current page
            page_results = await self._crawl_single_page(current_url, extract_content)
            results.extend(page_results)
            crawled_urls.add(current_url)
            
            # Extract links for next level
            if depth < max_depth and page_results and page_results[0].success:
                try:
                    links = page_results[0].metadata.get('links', [])
                    for link in links[:5]:  # Limit links per page
                        absolute_url = urljoin(current_url, link['href'])
                        parsed_link = urlparse(absolute_url)
                        
                        # Only follow same-domain links
                        if parsed_link.netloc == domain and absolute_url not in crawled_urls:
                            url_queue.append((absolute_url, depth + 1))
                            
                except Exception as e:
                    logger.warning("Failed to extract links", url=current_url, error=str(e))
            
            # Apply rate limiting between pages
            if policy:
                await asyncio.sleep(policy.rate_limit_seconds)
        
        return results
    
    async def _crawl_sitemap(self, base_url: str, extract_content: bool = True) -> List[CrawlResult]:
        """Crawl using sitemap.xml if available."""
        parsed_url = urlparse(base_url)
        sitemap_urls = [
            f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap.xml",
            f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap_index.xml",
            f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        ]
        
        urls_to_crawl = []
        
        # Try to find sitemap
        for sitemap_url in sitemap_urls:
            try:
                sitemap_result = await self._crawl_single_page(sitemap_url, False)
                if sitemap_result and sitemap_result[0].success:
                    content = sitemap_result[0].content
                    
                    if sitemap_url.endswith('robots.txt'):
                        # Extract sitemap URLs from robots.txt
                        sitemap_matches = re.findall(r'Sitemap:\s*(.+)', content, re.IGNORECASE)
                        for match in sitemap_matches:
                            try:
                                nested_result = await self._crawl_single_page(match.strip(), False)
                                if nested_result and nested_result[0].success:
                                    urls_to_crawl.extend(self._parse_sitemap(nested_result[0].content))
                            except:
                                continue
                    else:
                        # Parse XML sitemap
                        urls_to_crawl.extend(self._parse_sitemap(content))
                    
                    if urls_to_crawl:
                        break
                        
            except Exception as e:
                logger.debug("Failed to fetch sitemap", url=sitemap_url, error=str(e))
                continue
        
        # If no sitemap found, fallback to recursive crawl
        if not urls_to_crawl:
            logger.info("No sitemap found, falling back to recursive crawl", base_url=base_url)
            return await self._crawl_recursive(base_url, extract_content, max_depth=1)
        
        # Crawl URLs from sitemap
        results = []
        domain = urlparse(base_url).netloc
        policy = self.domain_policies.get(domain)
        max_pages = policy.max_pages if policy else 20
        
        for url in urls_to_crawl[:max_pages]:
            try:
                page_results = await self._crawl_single_page(url, extract_content)
                results.extend(page_results)
                
                # Rate limiting
                if policy:
                    await asyncio.sleep(policy.rate_limit_seconds)
                    
            except Exception as e:
                logger.warning("Failed to crawl sitemap URL", url=url, error=str(e))
                continue
        
        return results
    
    def _parse_sitemap(self, sitemap_content: str) -> List[str]:
        """Parse URLs from sitemap XML content."""
        urls = []
        
        # Simple regex-based XML parsing
        url_matches = re.findall(r'<loc>([^<]+)</loc>', sitemap_content)
        for match in url_matches:
            urls.append(match.strip())
        
        return urls
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content."""
        if not html_content:
            return "Untitled"
        
        # Look for title tag
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        # Look for h1 tag
        h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html_content, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()
        
        return "Untitled"
    
    async def _is_url_allowed(self, url: str) -> bool:
        """Check if URL is allowed by domain policy."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Check domain policy
        policy = self.domain_policies.get(domain)
        if policy is None:
            # Unknown domain - check default policy
            return await self._check_default_policy(domain)
        
        if not policy.allowed:
            return False
        
        # Check excluded paths
        for excluded_path in policy.excluded_paths:
            if excluded_path in parsed_url.path:
                return False
        
        return True
    
    async def _check_default_policy(self, domain: str) -> bool:
        """Check default policy for unknown domains."""
        # Simple heuristics for allowing domains
        suspicious_tlds = ['.onion', '.bit', '.i2p']
        suspicious_keywords = ['spam', 'malware', 'phishing', 'scam']
        
        # Block suspicious TLDs
        for tld in suspicious_tlds:
            if domain.endswith(tld):
                return False
        
        # Block suspicious keywords
        for keyword in suspicious_keywords:
            if keyword in domain.lower():
                return False
        
        # Allow most domains with rate limiting
        default_policy = DomainPolicy(
            domain=domain,
            allowed=True,
            max_pages=5,
            rate_limit_seconds=2.0
        )
        self.domain_policies[domain] = default_policy
        
        return True
    
    async def _apply_rate_limit(self, domain: str) -> None:
        """Apply rate limiting for domain."""
        if domain in self.rate_limits:
            last_request = self.rate_limits[domain]
            policy = self.domain_policies.get(domain)
            delay = policy.rate_limit_seconds if policy else self.default_delay
            
            elapsed = (datetime.utcnow() - last_request).total_seconds()
            if elapsed < delay:
                sleep_time = delay - elapsed
                await asyncio.sleep(sleep_time)
                self.crawl_stats['rate_limit_hits'] += 1
        
        self.rate_limits[domain] = datetime.utcnow()
    
    async def _store_crawl_results(self, results: List[CrawlResult], original_query: str) -> None:
        """Store crawl results in memory system."""
        try:
            for result in results:
                if not result.success or not result.content:
                    continue
                
                # Generate embedding if embedder is available
                embedding = []
                if self.embedder:
                    text_to_embed = f"{result.title} {result.content}"[:2000]  # Truncate
                    embedding = await self.embedder.embed_text(text_to_embed)
                
                # Calculate confidence scores
                confidence_score = 0.8  # Default confidence
                if self.confidence_agent:
                    confidence_score = await self.confidence_agent.calculate_confidence(
                        text=result.content,
                        source=result.url,
                        metadata=result.metadata or {}
                    )
                
                # Store in PostgreSQL
                if self.pgvector_handler and embedding:
                    await self.pgvector_handler.store_chunk(
                        text=result.content,
                        embedding=embedding,
                        metadata={
                            'title': result.title,
                            'source': result.url,
                            'crawl_method': 'crawl4ai',
                            'crawl_time': datetime.utcnow().isoformat(),
                            'original_query': original_query,
                            'status_code': result.status_code,
                            'content_length': result.content_length
                        },
                        title=result.title,
                        source=result.url,
                        original_query=original_query,
                        confidence_score=confidence_score
                    )
                
                # Store in Neo4j
                if self.neo4j_linker:
                    # Convert to WebSearchResult format
                    web_result = WebSearchResult(
                        text=result.content,
                        title=result.title or "Untitled",
                        source=result.url,
                        embedding=embedding,
                        extraction_method=ContentExtractor.TRAFILATURA,  # Default
                        extraction_quality=ExtractionQuality.GOOD,
                        confidence_score=confidence_score,
                        freshness_score=0.9,  # Assume fresh since just crawled
                        relevance_score=0.8,  # Default relevance
                        content_length=len(result.content or ""),
                        processed_length=len(result.content or ""),
                        extraction_time_seconds=result.crawl_time_seconds
                    )
                    
                    await self.neo4j_linker.add_web_search_result(web_result, original_query)
                
                self.crawl_stats['total_pages_processed'] += 1
                self.crawl_stats['total_content_bytes'] += result.content_length
                
        except Exception as e:
            logger.error("Failed to store crawl results", error=str(e))
            # Don't raise - storage failure shouldn't break crawling
    
    async def batch_crawl(
        self,
        urls: List[str],
        strategy: CrawlStrategy = CrawlStrategy.SINGLE_PAGE,
        max_concurrent: Optional[int] = None
    ) -> List[List[CrawlResult]]:
        """Crawl multiple URLs concurrently."""
        max_concurrent = max_concurrent or self.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def crawl_with_semaphore(url: str) -> List[CrawlResult]:
            async with semaphore:
                return await self.crawl_url(url, strategy)
        
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch crawl failed for URL", url=urls[i], error=str(result))
                valid_results.append([])
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def get_crawl_stats(self) -> Dict[str, Any]:
        """Get crawling statistics."""
        return {
            **self.crawl_stats,
            'domain_policies_count': len(self.domain_policies),
            'failed_urls_count': len(self.failed_urls),
            'crawl_history_count': len(self.crawl_history),
            'success_rate': (
                self.crawl_stats['successful_crawls'] / 
                max(self.crawl_stats['total_crawls'], 1)
            )
        }
    
    async def clear_cache(self) -> None:
        """Clear crawl cache and history."""
        self.crawl_history.clear()
        self.failed_urls.clear()
        self.rate_limits.clear()
        logger.info("Crawl cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of crawler components."""
        health_status = {
            'status': 'healthy',
            'components': {},
            'crawler_initialized': self.crawler is not None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check Crawl4AI crawler
        if self.crawler:
            try:
                # Simple test crawl
                test_result = await self.crawler.arun(
                    url="https://httpbin.org/get",
                    bypass_cache=True
                )
                health_status['components']['crawl4ai'] = {
                    'status': 'healthy' if test_result.success else 'unhealthy',
                    'test_url_success': test_result.success
                }
            except Exception as e:
                health_status['components']['crawl4ai'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            health_status['components']['crawl4ai'] = {
                'status': 'unhealthy',
                'error': 'Crawler not initialized'
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
    
    async def close(self) -> None:
        """Close crawler and clean up resources."""
        if self.crawler:
            await self.crawler.aclose()
        logger.info("Crawl4AI runner closed")


# Example usage
async def example_usage():
    """Example of using Crawl4aiRunner."""
    runner = Crawl4aiRunner()
    
    try:
        # Initialize
        await runner.initialize()
        
        # Add custom domain policy
        policy = DomainPolicy(
            domain="example.com",
            allowed=True,
            max_pages=5,
            rate_limit_seconds=1.0
        )
        await runner.add_domain_policy(policy)
        
        # Crawl single page
        results = await runner.crawl_url(
            "https://example.com",
            strategy=CrawlStrategy.SINGLE_PAGE
        )
        
        for result in results:
            print(f"Crawled: {result.url}")
            print(f"Success: {result.success}")
            print(f"Title: {result.title}")
            print(f"Content length: {result.content_length}")
        
        # Get statistics
        stats = await runner.get_crawl_stats()
        print(f"Crawl stats: {stats}")
        
        # Health check
        health = await runner.health_check()
        print(f"Health status: {health}")
        
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(example_usage())