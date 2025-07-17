"""
API routes for web crawling operations.

Provides REST endpoints for website crawling with natural language commands,
crawl management, and statistics.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ...core.crawling.natural_language_parser import NaturalLanguageCrawlParser
from ...ingest.crawl4ai_runner import Crawl4aiRunner, CrawlStrategy, CrawlStatus
from ...core.memory.manager import MemoryManager
from ...core.utils.simple_logger import get_logger
from ..middleware.auth import verify_api_key
from ..middleware.rate_limit import rate_limit

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/crawl", tags=["crawling"])


class CrawlRequest(BaseModel):
    """Request model for crawling websites."""
    command: str = Field(..., description="Natural language crawl command")
    max_pages: Optional[int] = Field(10, ge=1, le=100, description="Maximum pages to crawl")
    strategy: Optional[str] = Field(None, description="Override crawl strategy")
    store_in_memory: bool = Field(True, description="Store crawled content in memory")
    agent_id: str = Field("tyra", description="Agent ID for memory storage")
    extract_entities: bool = Field(True, description="Extract entities from content")
    force_recrawl: bool = Field(False, description="Force recrawl even if recently crawled")


class BatchCrawlRequest(BaseModel):
    """Request model for batch crawling."""
    urls: List[str] = Field(..., description="List of URLs to crawl")
    strategy: str = Field("single_page", description="Crawl strategy to use")
    max_concurrent: int = Field(3, ge=1, le=10, description="Maximum concurrent crawls")
    store_in_memory: bool = Field(True, description="Store crawled content in memory")
    agent_id: str = Field("tyra", description="Agent ID for memory storage")


class CrawlResponse(BaseModel):
    """Response model for crawl operations."""
    success: bool
    command: Optional[str] = None
    parsed_intent: Optional[str] = None
    strategy_used: Optional[str] = None
    confidence: Optional[float] = None
    results: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    timestamp: datetime


class CrawlStatsResponse(BaseModel):
    """Response model for crawl statistics."""
    total_crawls: int
    successful_crawls: int
    failed_crawls: int
    skipped_crawls: int
    total_pages_processed: int
    total_content_bytes: int
    average_crawl_time: float
    success_rate: float
    domain_policies_count: int
    timestamp: datetime


class DomainPolicyRequest(BaseModel):
    """Request model for domain policies."""
    domain: str = Field(..., description="Domain name")
    allowed: bool = Field(True, description="Whether domain is allowed")
    max_pages: int = Field(10, ge=1, le=1000, description="Maximum pages to crawl")
    rate_limit_seconds: float = Field(1.0, ge=0.1, le=60.0, description="Rate limit")
    excluded_paths: List[str] = Field(default_factory=list, description="Excluded URL paths")


# Dependency to get crawl runner
_crawl_runner: Optional[Crawl4aiRunner] = None
_crawl_parser: NaturalLanguageCrawlParser = NaturalLanguageCrawlParser()
_memory_manager: Optional[MemoryManager] = None


async def get_crawl_runner() -> Crawl4aiRunner:
    """Get or create crawl runner instance."""
    global _crawl_runner, _memory_manager
    
    if _crawl_runner is None:
        # Initialize memory manager if needed
        if _memory_manager is None:
            _memory_manager = MemoryManager()
            await _memory_manager.initialize()
        
        # Initialize crawl runner
        _crawl_runner = Crawl4aiRunner(
            embedder=_memory_manager.embedding_provider,
            pgvector_handler=getattr(_memory_manager, 'postgres_handler', None),
            neo4j_linker=getattr(_memory_manager, 'graph_handler', None),
            headless=True,
            max_concurrent=3
        )
        await _crawl_runner.initialize()
        
    return _crawl_runner


@router.post("/", response_model=CrawlResponse)
@rate_limit(calls=10, period=60)
async def crawl_website(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    crawl_runner: Crawl4aiRunner = Depends(get_crawl_runner)
) -> CrawlResponse:
    """
    Crawl a website using natural language commands.
    
    Examples:
    - "Crawl the Wikipedia page about machine learning"
    - "Read the entire documentation at docs.python.org"
    - "Get all blog posts from medium.com about AI"
    """
    try:
        # Parse natural language command
        parsed_command = _crawl_parser.parse(request.command)
        
        # Override with explicit parameters
        if request.max_pages:
            parsed_command.max_pages = request.max_pages
        if request.strategy:
            parsed_command.strategy = CrawlStrategy(request.strategy)
        parsed_command.store_in_memory = request.store_in_memory
        parsed_command.extract_entities = request.extract_entities
        
        # Check if URLs were found
        if not parsed_command.urls:
            suggestions = _crawl_parser.suggest_improvements(parsed_command)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No URLs found in command",
                    "suggestions": suggestions,
                    "parsed_intent": parsed_command.intent.value,
                    "confidence": parsed_command.confidence,
                }
            )
        
        # Perform crawling
        all_results = []
        crawl_stats = {
            "total_urls": len(parsed_command.urls),
            "successful_crawls": 0,
            "failed_crawls": 0,
            "pages_stored": 0,
            "total_content_bytes": 0,
        }
        
        for url in parsed_command.urls:
            try:
                # Crawl URL
                crawl_results = await crawl_runner.crawl_url(
                    url=url,
                    strategy=parsed_command.strategy,
                    extract_content=True,
                    store_in_memory=request.store_in_memory,
                    force_recrawl=request.force_recrawl
                )
                
                # Process results
                for result in crawl_results:
                    if result.success:
                        crawl_stats["successful_crawls"] += 1
                        crawl_stats["total_content_bytes"] += result.content_length
                        if request.store_in_memory:
                            crawl_stats["pages_stored"] += 1
                        
                        all_results.append({
                            "url": result.url,
                            "title": result.title,
                            "success": True,
                            "content_length": result.content_length,
                            "stored": request.store_in_memory,
                        })
                    else:
                        crawl_stats["failed_crawls"] += 1
                        all_results.append({
                            "url": result.url,
                            "success": False,
                            "error": result.error,
                        })
                
            except Exception as e:
                logger.error(f"Failed to crawl {url}: {e}")
                crawl_stats["failed_crawls"] += 1
                all_results.append({
                    "url": url,
                    "success": False,
                    "error": str(e),
                })
            
            # Respect max pages
            if parsed_command.max_pages and crawl_stats["successful_crawls"] >= parsed_command.max_pages:
                break
        
        return CrawlResponse(
            success=True,
            command=request.command,
            parsed_intent=parsed_command.intent.value,
            strategy_used=parsed_command.strategy.value,
            confidence=parsed_command.confidence,
            results=all_results,
            statistics=crawl_stats,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crawl request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=CrawlResponse)
@rate_limit(calls=5, period=60)
async def batch_crawl(
    request: BatchCrawlRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    crawl_runner: Crawl4aiRunner = Depends(get_crawl_runner)
) -> CrawlResponse:
    """
    Crawl multiple URLs concurrently.
    
    Useful for crawling a list of specific URLs with the same strategy.
    """
    try:
        strategy = CrawlStrategy(request.strategy)
        
        # Batch crawl
        results = await crawl_runner.batch_crawl(
            urls=request.urls,
            strategy=strategy,
            max_concurrent=request.max_concurrent
        )
        
        # Process results
        all_results = []
        crawl_stats = {
            "total_urls": len(request.urls),
            "successful_crawls": 0,
            "failed_crawls": 0,
            "pages_stored": 0,
            "total_content_bytes": 0,
        }
        
        for url, url_results in zip(request.urls, results):
            if url_results:
                for result in url_results:
                    if result.success:
                        crawl_stats["successful_crawls"] += 1
                        crawl_stats["total_content_bytes"] += result.content_length
                        if request.store_in_memory:
                            crawl_stats["pages_stored"] += 1
                        
                        all_results.append({
                            "url": result.url,
                            "title": result.title,
                            "success": True,
                            "content_length": result.content_length,
                            "stored": request.store_in_memory,
                        })
                    else:
                        crawl_stats["failed_crawls"] += 1
                        all_results.append({
                            "url": result.url,
                            "success": False,
                            "error": result.error,
                        })
            else:
                crawl_stats["failed_crawls"] += 1
                all_results.append({
                    "url": url,
                    "success": False,
                    "error": "No results returned",
                })
        
        return CrawlResponse(
            success=True,
            strategy_used=strategy.value,
            results=all_results,
            statistics=crawl_stats,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Batch crawl failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=CrawlStatsResponse)
async def get_crawl_statistics(
    api_key: str = Depends(verify_api_key),
    crawl_runner: Crawl4aiRunner = Depends(get_crawl_runner)
) -> CrawlStatsResponse:
    """Get crawling statistics and performance metrics."""
    try:
        stats = await crawl_runner.get_crawl_stats()
        
        return CrawlStatsResponse(
            total_crawls=stats["total_crawls"],
            successful_crawls=stats["successful_crawls"],
            failed_crawls=stats["failed_crawls"],
            skipped_crawls=stats["skipped_crawls"],
            total_pages_processed=stats["total_pages_processed"],
            total_content_bytes=stats["total_content_bytes"],
            average_crawl_time=stats["average_crawl_time"],
            success_rate=stats["success_rate"],
            domain_policies_count=stats["domain_policies_count"],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get crawl stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/domain-policy")
async def add_domain_policy(
    request: DomainPolicyRequest,
    api_key: str = Depends(verify_api_key),
    crawl_runner: Crawl4aiRunner = Depends(get_crawl_runner)
) -> Dict[str, Any]:
    """Add or update a domain crawling policy."""
    try:
        from ...ingest.crawl4ai_runner import DomainPolicy
        
        policy = DomainPolicy(
            domain=request.domain,
            allowed=request.allowed,
            max_pages=request.max_pages,
            rate_limit_seconds=request.rate_limit_seconds,
            excluded_paths=request.excluded_paths
        )
        
        await crawl_runner.add_domain_policy(policy)
        
        return {
            "success": True,
            "domain": request.domain,
            "policy": {
                "allowed": request.allowed,
                "max_pages": request.max_pages,
                "rate_limit_seconds": request.rate_limit_seconds,
                "excluded_paths": request.excluded_paths,
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to add domain policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_crawl_cache(
    api_key: str = Depends(verify_api_key),
    crawl_runner: Crawl4aiRunner = Depends(get_crawl_runner)
) -> Dict[str, Any]:
    """Clear crawl cache and history."""
    try:
        await crawl_runner.clear_cache()
        
        return {
            "success": True,
            "message": "Crawl cache cleared successfully",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear crawl cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def crawl_health_check(
    api_key: str = Depends(verify_api_key),
    crawl_runner: Crawl4aiRunner = Depends(get_crawl_runner)
) -> Dict[str, Any]:
    """Check health of crawling components."""
    try:
        health = await crawl_runner.health_check()
        return health
        
    except Exception as e:
        logger.error(f"Crawl health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }