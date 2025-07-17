"""
Advanced MCP server for Tyra's memory system.

Provides Model Context Protocol (MCP) tools for memory storage, retrieval,
analytics, and adaptive learning capabilities with comprehensive error handling
and performance monitoring.
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    GetPromptRequest,
    GetPromptResult,
    ListToolsRequest,
    Prompt,
    PromptMessage,
    Role,
    TextContent,
    Tool,
)

from ..core.adaptation.learning_engine import LearningEngine
from ..core.analytics.performance_tracker import MetricType, PerformanceTracker
from ..core.memory.manager import MemoryManager, MemorySearchRequest, MemoryStoreRequest
from ..core.rag.hallucination_detector import HallucinationDetector
from ..core.synthesis import (
    DeduplicationEngine,
    SummarizationEngine,
    PatternDetector,
    TemporalAnalyzer,
    SummarizationType,
)
from ..core.clients.vllm_client import VLLMClient
from ..core.utils.simple_config import get_setting, get_settings
from ..core.utils.simple_logger import get_logger
from ..core.crawling.natural_language_parser import NaturalLanguageCrawlParser, CrawlCommand
from ..ingest.crawl4ai_runner import Crawl4aiRunner, CrawlStrategy, CrawlStatus
from ..core.services.file_watcher_service import get_file_watcher_manager
from ..suggestions.related.local_suggester import LocalSuggester
from ..suggestions.connections.local_connector import LocalConnector
from ..suggestions.organization.local_recommender import LocalRecommender
from ..suggestions.gaps.local_detector import LocalDetector
from ..agents.websearch_agent import WebSearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class TyraMemoryServer:
    """
    Advanced MCP server for Tyra's memory system.

    Features:
    - Memory storage and retrieval tools
    - Advanced search with hallucination detection
    - Performance analytics and monitoring
    - Adaptive learning and optimization
    - Multi-agent memory isolation
    - Comprehensive error handling
    """

    def __init__(self):
        self.server = Server("tyra-memory-server")

        # Core components
        self.memory_manager: Optional[MemoryManager] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.learning_engine: Optional[LearningEngine] = None
        self.hallucination_detector: Optional[HallucinationDetector] = None
        
        # Synthesis components
        self.deduplication_engine: Optional[DeduplicationEngine] = None
        self.summarization_engine: Optional[SummarizationEngine] = None
        self.pattern_detector: Optional[PatternDetector] = None
        self.temporal_analyzer: Optional[TemporalAnalyzer] = None
        self.vllm_client: Optional[VLLMClient] = None
        
        # Crawling components
        self.crawl_parser: NaturalLanguageCrawlParser = NaturalLanguageCrawlParser()
        self.crawl_runner: Optional[Crawl4aiRunner] = None
        
        # File watcher service
        self.file_watcher_manager = get_file_watcher_manager()
        
        # Suggestions components
        self.local_suggester: Optional[LocalSuggester] = None
        self.local_connector: Optional[LocalConnector] = None
        self.local_recommender: Optional[LocalRecommender] = None
        self.local_detector: Optional[LocalDetector] = None
        
        # Web search agent
        self.web_search_agent: Optional[WebSearchAgent] = None

        # Server state
        self._initialized = False
        self._total_requests = 0
        self._successful_requests = 0
        self._start_time = datetime.utcnow()

        # Register handlers
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available memory tools."""
            return [
                Tool(
                    name="store_memory",
                    description="Store information in memory with optional metadata and entity extraction",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to store in memory",
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "ID of the agent storing the memory (e.g., 'tyra', 'claude', 'archon')",
                                "default": "tyra",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session ID for grouping related memories",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata to store with the memory",
                                "additionalProperties": True,
                            },
                            "extract_entities": {
                                "type": "boolean",
                                "description": "Whether to extract entities and relationships",
                                "default": True,
                            },
                            "chunk_content": {
                                "type": "boolean",
                                "description": "Whether to chunk large content",
                                "default": False,
                            },
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="search_memory",
                    description="Search memories with advanced filtering and confidence scoring",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for finding relevant memories",
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Filter by agent ID (e.g., 'tyra', 'claude', 'archon')",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Filter by session ID",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "min_confidence": {
                                "type": "number",
                                "description": "Minimum confidence score (0.0-1.0)",
                                "default": 0.0,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "search_type": {
                                "type": "string",
                                "description": "Type of search to perform",
                                "enum": ["vector", "graph", "hybrid"],
                                "default": "hybrid",
                            },
                            "include_analysis": {
                                "type": "boolean",
                                "description": "Whether to include hallucination analysis",
                                "default": True,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="analyze_response",
                    description="Analyze a response for hallucinations and confidence scoring",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "string",
                                "description": "The response text to analyze",
                            },
                            "query": {
                                "type": "string",
                                "description": "Original query that generated the response",
                            },
                            "retrieved_memories": {
                                "type": "array",
                                "description": "Memory chunks used to generate the response",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "id": {"type": "string"},
                                        "metadata": {"type": "object"},
                                    },
                                },
                            },
                        },
                        "required": ["response"],
                    },
                ),
                Tool(
                    name="get_memory_stats",
                    description="Get comprehensive memory system statistics and health metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Filter stats by agent ID",
                            },
                            "include_performance": {
                                "type": "boolean",
                                "description": "Include performance analytics",
                                "default": True,
                            },
                            "include_recommendations": {
                                "type": "boolean",
                                "description": "Include optimization recommendations",
                                "default": True,
                            },
                        },
                    },
                ),
                Tool(
                    name="get_learning_insights",
                    description="Get insights from the adaptive learning system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter insights by category",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days to look back",
                                "default": 7,
                                "minimum": 1,
                                "maximum": 90,
                            },
                        },
                    },
                ),
                Tool(
                    name="delete_memory",
                    description="Delete a specific memory by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the memory to delete",
                            }
                        },
                        "required": ["memory_id"],
                    },
                ),
                Tool(
                    name="deduplicate_memories",
                    description="Identify and merge duplicate memories using semantic similarity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Filter by agent ID",
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Similarity threshold for duplicates (0.7-1.0)",
                                "default": 0.85,
                                "minimum": 0.7,
                                "maximum": 1.0,
                            },
                            "auto_merge": {
                                "type": "boolean",
                                "description": "Automatically merge duplicates",
                                "default": False,
                            },
                        },
                    },
                ),
                Tool(
                    name="summarize_memories",
                    description="Generate AI-powered summaries of memory clusters with anti-hallucination",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_ids": {
                                "type": "array",
                                "description": "List of memory IDs to summarize",
                                "items": {"type": "string"},
                            },
                            "summary_type": {
                                "type": "string",
                                "description": "Type of summarization",
                                "enum": ["extractive", "abstractive", "hybrid", "progressive"],
                                "default": "hybrid",
                            },
                            "max_length": {
                                "type": "integer",
                                "description": "Maximum summary length in tokens",
                                "default": 200,
                                "minimum": 50,
                                "maximum": 500,
                            },
                        },
                        "required": ["memory_ids"],
                    },
                ),
                Tool(
                    name="detect_patterns",
                    description="Detect patterns and knowledge gaps across memories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Filter by agent ID",
                            },
                            "min_cluster_size": {
                                "type": "integer",
                                "description": "Minimum memories per pattern cluster",
                                "default": 3,
                                "minimum": 2,
                            },
                            "include_recommendations": {
                                "type": "boolean",
                                "description": "Include learning recommendations",
                                "default": True,
                            },
                        },
                    },
                ),
                Tool(
                    name="analyze_temporal_evolution",
                    description="Analyze how memories and concepts evolve over time",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string",
                                "description": "Concept to track evolution (optional)",
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Filter by agent ID",
                            },
                            "time_window_days": {
                                "type": "integer",
                                "description": "Time window for analysis in days",
                                "default": 30,
                                "minimum": 1,
                            },
                        },
                    },
                ),
                Tool(
                    name="health_check",
                    description="Perform comprehensive system health check",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "detailed": {
                                "type": "boolean",
                                "description": "Include detailed component health",
                                "default": False,
                            }
                        },
                    },
                ),
                Tool(
                    name="crawl_website",
                    description="Crawl websites using natural language commands and store content in memory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Natural language crawl command (e.g., 'crawl the Wikipedia page about AI', 'read all documentation at docs.python.org')",
                            },
                            "max_pages": {
                                "type": "integer",
                                "description": "Maximum number of pages to crawl",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "strategy": {
                                "type": "string",
                                "description": "Override crawl strategy",
                                "enum": ["single_page", "site_map", "recursive", "rss_feed"],
                            },
                            "store_in_memory": {
                                "type": "boolean",
                                "description": "Whether to store crawled content in memory",
                                "default": True,
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID for memory storage",
                                "default": "tyra",
                            },
                        },
                        "required": ["command"],
                    },
                ),
                Tool(
                    name="suggest_related_memories",
                    description="Get intelligent suggestions for related memories using ML algorithms",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the memory to find related suggestions for",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to find related suggestions for (alternative to memory_id)",
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID to filter suggestions",
                                "default": "tyra",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of suggestions to return",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50,
                            },
                            "min_relevance": {
                                "type": "number",
                                "description": "Minimum relevance score (0.0-1.0)",
                                "default": 0.3,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="detect_memory_connections",
                    description="Automatically detect and suggest connections between memories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID to analyze connections for",
                                "default": "tyra",
                            },
                            "connection_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Types of connections to detect",
                                "default": ["semantic", "temporal", "entity"],
                            },
                            "min_confidence": {
                                "type": "number",
                                "description": "Minimum confidence score for connections",
                                "default": 0.5,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="recommend_memory_organization",
                    description="Analyze memory structure and recommend organization improvements",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID to analyze organization for",
                                "default": "tyra",
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of organization analysis",
                                "enum": ["clustering", "hierarchy", "topics", "all"],
                                "default": "all",
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="detect_knowledge_gaps",
                    description="Identify knowledge gaps and suggest content to fill them",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID to analyze gaps for",
                                "default": "tyra",
                            },
                            "domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific domains to analyze for gaps",
                            },
                            "gap_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Types of gaps to detect",
                                "default": ["topic", "temporal", "detail"],
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="web_search",
                    description="Perform local web search with content extraction and memory integration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for web search",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of search results to process",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "store_in_memory": {
                                "type": "boolean",
                                "description": "Whether to store results in memory system",
                                "default": True,
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Agent ID for memory storage",
                                "default": "tyra",
                            },
                            "force_refresh": {
                                "type": "boolean",
                                "description": "Force new search even if similar content exists",
                                "default": False,
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> CallToolResult:
            """Handle tool execution with comprehensive error handling."""
            start_time = datetime.utcnow()
            self._total_requests += 1

            try:
                # Ensure system is initialized
                if not self._initialized:
                    await self._initialize_components()

                # Route to appropriate handler
                if name == "store_memory":
                    result = await self._handle_store_memory(arguments)
                elif name == "search_memory":
                    result = await self._handle_search_memory(arguments)
                elif name == "analyze_response":
                    result = await self._handle_analyze_response(arguments)
                elif name == "get_memory_stats":
                    result = await self._handle_get_memory_stats(arguments)
                elif name == "get_learning_insights":
                    result = await self._handle_get_learning_insights(arguments)
                elif name == "delete_memory":
                    result = await self._handle_delete_memory(arguments)
                elif name == "deduplicate_memories":
                    result = await self._handle_deduplicate_memories(arguments)
                elif name == "summarize_memories":
                    result = await self._handle_summarize_memories(arguments)
                elif name == "detect_patterns":
                    result = await self._handle_detect_patterns(arguments)
                elif name == "analyze_temporal_evolution":
                    result = await self._handle_analyze_temporal_evolution(arguments)
                elif name == "health_check":
                    result = await self._handle_health_check(arguments)
                elif name == "crawl_website":
                    result = await self._handle_crawl_website(arguments)
                elif name == "suggest_related_memories":
                    result = await self._handle_suggest_related_memories(arguments)
                elif name == "detect_memory_connections":
                    result = await self._handle_detect_memory_connections(arguments)
                elif name == "recommend_memory_organization":
                    result = await self._handle_recommend_memory_organization(arguments)
                elif name == "detect_knowledge_gaps":
                    result = await self._handle_detect_knowledge_gaps(arguments)
                elif name == "web_search":
                    result = await self._handle_web_search(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")

                # Record successful request
                self._successful_requests += 1

                # Track performance
                response_time = (datetime.utcnow() - start_time).total_seconds()
                if self.performance_tracker:
                    await self.performance_tracker.record_metric(
                        MetricType.RESPONSE_TIME,
                        response_time,
                        context={"tool": name, "success": True},
                    )

                return CallToolResult(
                    content=[
                        TextContent(type="text", text=json.dumps(result, indent=2))
                    ]
                )

            except Exception as e:
                # Log error
                error_msg = f"Tool '{name}' failed: {str(e)}"
                logger.error(
                    "Tool execution failed",
                    tool=name,
                    arguments=arguments,
                    error=str(e),
                    traceback=traceback.format_exc(),
                )

                # Track error metrics
                response_time = (datetime.utcnow() - start_time).total_seconds()
                if self.performance_tracker:
                    await self.performance_tracker.record_metric(
                        MetricType.RESPONSE_TIME,
                        response_time,
                        context={"tool": name, "success": False, "error": str(e)},
                    )

                # Return error response
                error_result = {
                    "success": False,
                    "error": error_msg,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tool": name,
                }

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=json.dumps(error_result, indent=2)
                        )
                    ]
                )

    async def _initialize_components(self) -> None:
        """Initialize all server components."""
        try:
            logger.info("Initializing Tyra memory server components...")

            # Initialize memory manager
            self.memory_manager = MemoryManager()
            await self.memory_manager.initialize()

            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker()

            # Initialize hallucination detector
            if self.memory_manager.embedding_provider:
                self.hallucination_detector = HallucinationDetector(
                    self.memory_manager.embedding_provider
                )

            # Initialize learning engine
            self.learning_engine = LearningEngine(self.performance_tracker)

            # Initialize vLLM client if configured
            try:
                self.vllm_client = VLLMClient()
                logger.info("Initialized vLLM client successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize vLLM client: {e}")
                self.vllm_client = None

            # Initialize synthesis components
            if self.memory_manager.embedding_provider:
                # Deduplication engine
                self.deduplication_engine = DeduplicationEngine(
                    embedder=self.memory_manager.embedding_provider,
                    cache=None  # Could add cache if available
                )
                
                # Summarization engine
                self.summarization_engine = SummarizationEngine(
                    embedder=self.memory_manager.embedding_provider,
                    vllm_client=self.vllm_client,
                    hallucination_detector=self.hallucination_detector,
                    cache=None  # Could add cache if available
                )
                
                # Pattern detector
                self.pattern_detector = PatternDetector(
                    embedder=self.memory_manager.embedding_provider,
                    cache=None
                )
                
                # Temporal analyzer
                self.temporal_analyzer = TemporalAnalyzer(
                    embedder=self.memory_manager.embedding_provider,
                    cache=None
                )
                
                logger.info("Initialized all synthesis components successfully")
            else:
                logger.warning("Embedding provider not available, synthesis features disabled")
            
            # Initialize suggestions components
            if self.memory_manager.embedding_provider:
                try:
                    # Related memory suggester
                    self.local_suggester = LocalSuggester(
                        embedder=self.memory_manager.embedding_provider,
                        memory_manager=self.memory_manager
                    )
                    
                    # Memory connections detector
                    self.local_connector = LocalConnector(
                        embedder=self.memory_manager.embedding_provider,
                        memory_manager=self.memory_manager
                    )
                    
                    # Organization recommender
                    self.local_recommender = LocalRecommender(
                        embedder=self.memory_manager.embedding_provider,
                        memory_manager=self.memory_manager
                    )
                    
                    # Knowledge gaps detector
                    self.local_detector = LocalDetector(
                        embedder=self.memory_manager.embedding_provider,
                        memory_manager=self.memory_manager
                    )
                    
                    logger.info("Initialized all suggestions components successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize some suggestions components: {e}")
            else:
                logger.warning("Embedding provider not available, suggestions features disabled")
            
            # Initialize web search agent
            if self.memory_manager.embedding_provider:
                try:
                    self.web_search_agent = WebSearchAgent(
                        vllm_client=self.vllm_client,
                        embedder=self.memory_manager.embedding_provider,
                        pgvector_handler=self.memory_manager.postgres_handler if hasattr(self.memory_manager, 'postgres_handler') else None,
                        neo4j_linker=self.memory_manager.graph_handler if hasattr(self.memory_manager, 'graph_handler') else None,
                        max_results=10,
                        similarity_threshold=0.75
                    )
                    logger.info("Initialized WebSearchAgent successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize WebSearchAgent: {e}")
                    self.web_search_agent = None
            else:
                logger.warning("Embedding provider not available, web search features disabled")
            
            # Initialize crawl4ai runner
            try:
                self.crawl_runner = Crawl4aiRunner(
                    embedder=self.memory_manager.embedding_provider,
                    pgvector_handler=self.memory_manager.postgres_handler if hasattr(self.memory_manager, 'postgres_handler') else None,
                    neo4j_linker=self.memory_manager.graph_handler if hasattr(self.memory_manager, 'graph_handler') else None,
                    confidence_agent=None,  # Could add confidence agent if available
                    headless=True,
                    max_concurrent=3
                )
                await self.crawl_runner.initialize()
                logger.info("Initialized Crawl4AI runner successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Crawl4AI runner: {e}")
                self.crawl_runner = None

            # Start learning cycle in background
            asyncio.create_task(self.learning_engine.start_learning_cycle())
            
            # Start file watcher service if enabled
            try:
                file_watcher_enabled = get_setting("file_watcher.enabled", True)
                file_watcher_auto_start = get_setting("file_watcher.processing.auto_start", True)
                
                if file_watcher_enabled and file_watcher_auto_start:
                    asyncio.create_task(self.file_watcher_manager.start())
                    logger.info("File watcher service started automatically")
                else:
                    logger.info("File watcher service not started (disabled or auto_start=false)")
            except Exception as e:
                logger.warning(f"Failed to start file watcher service: {e}")

            self._initialized = True
            logger.info("Tyra memory server components initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize server components", error=str(e))
            raise

    async def _handle_store_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory storage requests."""
        try:
            request = MemoryStoreRequest(
                content=arguments["content"],
                agent_id=arguments.get("agent_id", "tyra"),
                session_id=arguments.get("session_id"),
                metadata=arguments.get("metadata"),
                extract_entities=arguments.get("extract_entities", True),
                chunk_content=arguments.get("chunk_content", False),
            )

            result = await self.memory_manager.store_memory(request)

            return {
                "success": True,
                "memory_id": result.memory_id,
                "chunk_ids": result.chunk_ids,
                "entities_created": len(result.entities_created),
                "relationships_created": len(result.relationships_created),
                "processing_time": {
                    "embedding": result.embedding_time,
                    "storage": result.storage_time,
                    "graph": result.graph_time,
                    "total": result.total_time,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Memory storage failed", arguments=arguments, error=str(e))
            raise

    async def _handle_search_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory search requests."""
        try:
            request = MemorySearchRequest(
                query=arguments["query"],
                agent_id=arguments.get("agent_id"),
                session_id=arguments.get("session_id"),
                top_k=arguments.get("top_k", 10),
                min_confidence=arguments.get("min_confidence", 0.0),
                search_type=arguments.get("search_type", "hybrid"),
                include_graph=arguments.get("search_type", "hybrid")
                in ["graph", "hybrid"],
            )

            # Perform search
            search_results = await self.memory_manager.search_memory(request)

            # Prepare results
            results = []
            retrieved_chunks = []

            for result in search_results:
                result_data = {
                    "id": result.id,
                    "content": result.content,
                    "score": result.score,
                    "confidence": result.confidence,
                    "source_type": result.source_type,
                    "metadata": result.metadata,
                }

                if result.rerank_explanation:
                    result_data["rerank_explanation"] = result.rerank_explanation

                if result.entities:
                    result_data["entities"] = [
                        {
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.entity_type,
                            "confidence": entity.confidence,
                        }
                        for entity in result.entities
                    ]

                results.append(result_data)

                # Prepare for hallucination analysis
                retrieved_chunks.append(
                    {
                        "content": result.content,
                        "id": result.id,
                        "metadata": result.metadata,
                    }
                )

            response = {
                "success": True,
                "query": request.query,
                "results": results,
                "total_results": len(results),
                "search_type": request.search_type,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add hallucination analysis if requested
            if (
                arguments.get("include_analysis", True)
                and self.hallucination_detector
                and results
            ):
                # Create a summary response from top results
                summary_content = "\n".join([r["content"] for r in results[:3]])

                analysis = await self.hallucination_detector.analyze_response(
                    summary_content, retrieved_chunks, request.query
                )

                response["hallucination_analysis"] = {
                    "confidence": analysis.overall_confidence,
                    "confidence_level": analysis.confidence_level.value,
                    "hallucination_flag": analysis.hallucination_flag,
                    "safe_to_act_on": analysis.safe_to_act_on,
                    "evidence_count": analysis.evidence_count,
                    "reasoning": analysis.reasoning,
                    "warnings": analysis.warnings,
                }

            return response

        except Exception as e:
            logger.error("Memory search failed", arguments=arguments, error=str(e))
            raise

    async def _handle_analyze_response(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle response analysis for hallucination detection."""
        try:
            if not self.hallucination_detector:
                return {
                    "success": False,
                    "error": "Hallucination detector not available",
                }

            response_text = arguments["response"]
            query = arguments.get("query", "")
            retrieved_memories = arguments.get("retrieved_memories", [])

            # Perform analysis
            analysis = await self.hallucination_detector.analyze_response(
                response_text, retrieved_memories, query
            )

            return {
                "success": True,
                "response_text": response_text,
                "analysis": {
                    "overall_confidence": analysis.overall_confidence,
                    "confidence_level": analysis.confidence_level.value,
                    "confidence_emoji": self.hallucination_detector.get_confidence_emoji(
                        analysis.confidence_level
                    ),
                    "hallucination_flag": analysis.hallucination_flag,
                    "safe_to_act_on": analysis.safe_to_act_on,
                    "grounding_score": analysis.grounding_score,
                    "evidence_count": analysis.evidence_count,
                    "reasoning": analysis.reasoning,
                    "warnings": analysis.warnings,
                    "analysis_time": analysis.analysis_time,
                },
                "evidence": [
                    {
                        "source_chunk": evidence.source_chunk,
                        "similarity_score": evidence.similarity_score,
                        "relevance_score": evidence.relevance_score,
                        "confidence": evidence.confidence,
                        "chunk_id": evidence.chunk_id,
                    }
                    for evidence in analysis.evidence[:5]  # Top 5 evidence pieces
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Response analysis failed", arguments=arguments, error=str(e))
            raise

    async def _handle_get_memory_stats(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle memory statistics requests."""
        try:
            # Get memory manager stats
            memory_stats = await self.memory_manager.get_stats()

            # Get health check
            health_status = await self.memory_manager.health_check()

            response = {
                "success": True,
                "memory_stats": memory_stats,
                "health_status": health_status,
                "server_stats": {
                    "total_requests": self._total_requests,
                    "successful_requests": self._successful_requests,
                    "success_rate": self._successful_requests
                    / max(self._total_requests, 1),
                    "uptime_seconds": (
                        datetime.utcnow() - self._start_time
                    ).total_seconds(),
                    "initialized": self._initialized,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add performance analytics if requested
            if arguments.get("include_performance", True) and self.performance_tracker:
                performance_summary = (
                    await self.performance_tracker.get_performance_summary(days=7)
                )
                response["performance_analytics"] = performance_summary

            # Add optimization recommendations if requested
            if arguments.get("include_recommendations", True) and self.learning_engine:
                recommendations = await self.learning_engine.get_recommendations()
                response["optimization_recommendations"] = [
                    {
                        "category": rec.category,
                        "priority": rec.priority,
                        "title": rec.title,
                        "description": rec.description,
                        "expected_impact": rec.expected_impact,
                        "confidence": rec.confidence,
                    }
                    for rec in recommendations[:5]  # Top 5 recommendations
                ]

            return response

        except Exception as e:
            logger.error(
                "Memory stats request failed", arguments=arguments, error=str(e)
            )
            raise

    async def _handle_get_learning_insights(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle learning insights requests."""
        try:
            if not self.learning_engine:
                return {"success": False, "error": "Learning engine not available"}

            category = arguments.get("category")
            insights = await self.learning_engine.get_learning_insights(category)

            # Get experiment history
            days = arguments.get("days", 7)
            experiments = await self.learning_engine.get_experiment_history(days)

            return {
                "success": True,
                "insights": [
                    {
                        "category": insight.category,
                        "insight": insight.insight,
                        "confidence": insight.confidence,
                        "actionable": insight.actionable,
                        "impact_estimate": insight.impact_estimate,
                        "timestamp": insight.timestamp.isoformat(),
                    }
                    for insight in insights
                ],
                "recent_experiments": [
                    {
                        "id": exp.id,
                        "type": exp.adaptation_type.value,
                        "status": exp.status.value,
                        "success": exp.success,
                        "improvements": exp.improvement,
                        "confidence": exp.confidence,
                        "start_time": exp.start_time.isoformat(),
                    }
                    for exp in experiments
                ],
                "learning_stats": self.learning_engine.get_learning_stats(),
                "period_days": days,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                "Learning insights request failed", arguments=arguments, error=str(e)
            )
            raise

    async def _handle_delete_memory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory deletion requests."""
        try:
            memory_id = arguments["memory_id"]
            success = await self.memory_manager.delete_memory(memory_id)

            return {
                "success": success,
                "memory_id": memory_id,
                "deleted": success,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Memory deletion failed", arguments=arguments, error=str(e))
            raise

    async def _handle_health_check(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check requests."""
        try:
            detailed = arguments.get("detailed", False)

            # Basic health check
            health_status = {
                "success": True,
                "status": "healthy",
                "initialized": self._initialized,
                "uptime_seconds": (
                    datetime.utcnow() - self._start_time
                ).total_seconds(),
                "total_requests": self._total_requests,
                "success_rate": self._successful_requests
                / max(self._total_requests, 1),
                "timestamp": datetime.utcnow().isoformat(),
            }

            if detailed and self._initialized:
                # Detailed component health
                components = {}

                if self.memory_manager:
                    components["memory_manager"] = (
                        await self.memory_manager.health_check()
                    )

                if self.performance_tracker:
                    components["performance_tracker"] = {
                        "status": "healthy",
                        "stats": self.performance_tracker.get_analytics_stats(),
                    }

                if self.learning_engine:
                    components["learning_engine"] = {
                        "status": "healthy",
                        "stats": self.learning_engine.get_learning_stats(),
                    }

                health_status["components"] = components

            return health_status

        except Exception as e:
            logger.error("Health check failed", arguments=arguments, error=str(e))
            return {
                "success": False,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _handle_deduplicate_memories(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory deduplication requests."""
        try:
            if not self.deduplication_engine:
                return {
                    "success": False,
                    "error": "Deduplication engine not initialized",
                }

            # Search for memories to deduplicate
            search_request = MemorySearchRequest(
                query="*",  # Get all memories
                agent_id=arguments.get("agent_id"),
                top_k=1000,  # Process up to 1000 memories
            )
            
            search_results = await self.memory_manager.search_memory(search_request)
            
            if not search_results.results:
                return {
                    "success": True,
                    "duplicates_found": 0,
                    "message": "No memories found to deduplicate",
                }
            
            # Convert to Memory objects for deduplication
            from ...models.memory import Memory
            memories = [
                Memory(
                    id=result.id,
                    content=result.content,
                    agent_id=result.metadata.get("agent_id", "unknown"),
                    created_at=datetime.fromisoformat(result.metadata.get("created_at", datetime.utcnow().isoformat())),
                    metadata=result.metadata
                )
                for result in search_results.results
            ]
            
            # Find duplicates
            duplicate_groups = await self.deduplication_engine.find_duplicates(
                memories=memories,
                threshold=arguments.get("similarity_threshold", 0.85)
            )
            
            result = {
                "success": True,
                "duplicates_found": len(duplicate_groups),
                "duplicate_groups": [],
                "total_memories_processed": len(memories),
            }
            
            # Process duplicate groups
            for group in duplicate_groups[:10]:  # Limit to first 10 groups
                group_info = {
                    "duplicate_type": group.duplicate_type.value,
                    "confidence": group.confidence,
                    "memory_ids": group.memory_ids,
                    "suggested_merge": group.suggested_merge,
                }
                
                if arguments.get("auto_merge", False) and group.confidence > 0.9:
                    # Auto-merge high confidence duplicates
                    merged = await self.deduplication_engine.merge_duplicates(
                        group, strategy=group.merge_strategy
                    )
                    group_info["merged"] = True
                    group_info["merged_content"] = merged.content[:200] + "..."
                
                result["duplicate_groups"].append(group_info)
            
            return result
            
        except Exception as e:
            logger.error("Deduplication failed", arguments=arguments, error=str(e))
            raise

    async def _handle_summarize_memories(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory summarization requests."""
        try:
            if not self.summarization_engine:
                return {
                    "success": False,
                    "error": "Summarization engine not initialized",
                }

            memory_ids = arguments.get("memory_ids", [])
            if not memory_ids:
                return {
                    "success": False,
                    "error": "No memory IDs provided",
                }
            
            # Fetch memories by IDs
            memories_to_summarize = []
            for memory_id in memory_ids:
                # Search for specific memory by ID
                search_request = MemorySearchRequest(
                    query=memory_id,
                    top_k=1,
                    search_type="vector"
                )
                search_results = await self.memory_manager.search_memory(search_request)
                
                if search_results.results:
                    memories_to_summarize.append(search_results.results[0].content)
            
            if not memories_to_summarize:
                return {
                    "success": False,
                    "error": "No memories found for provided IDs",
                }
            
            # Generate summary
            summary_type = SummarizationType(arguments.get("summary_type", "hybrid"))
            summary = await self.summarization_engine.summarize(
                texts=memories_to_summarize,
                summary_type=summary_type,
                max_length=arguments.get("max_length", 200)
            )
            
            # Evaluate quality
            quality_metrics = await self.summarization_engine.evaluate_summary(
                summary.summary,
                memories_to_summarize
            )
            
            return {
                "success": True,
                "summary": summary.summary,
                "key_points": summary.key_points,
                "confidence_score": summary.confidence_score,
                "quality_metrics": {
                    "rouge_l": quality_metrics.rouge_l_score,
                    "factual_consistency": quality_metrics.factual_consistency,
                    "hallucination_score": quality_metrics.hallucination_score,
                    "overall_quality": quality_metrics.overall_quality.value,
                },
                "source_memory_count": len(memories_to_summarize),
                "summary_type": summary_type.value,
            }
            
        except Exception as e:
            logger.error("Summarization failed", arguments=arguments, error=str(e))
            raise

    async def _handle_detect_patterns(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pattern detection requests."""
        try:
            if not self.pattern_detector:
                return {
                    "success": False,
                    "error": "Pattern detector not initialized",
                }

            # Search for memories to analyze
            search_request = MemorySearchRequest(
                query="*",
                agent_id=arguments.get("agent_id"),
                top_k=500,  # Analyze up to 500 memories
            )
            
            search_results = await self.memory_manager.search_memory(search_request)
            
            if not search_results.results:
                return {
                    "success": True,
                    "patterns_found": 0,
                    "message": "No memories found to analyze",
                }
            
            # Convert to Memory objects
            from ...models.memory import Memory
            memories = [
                Memory(
                    id=result.id,
                    content=result.content,
                    agent_id=result.metadata.get("agent_id", "unknown"),
                    created_at=datetime.fromisoformat(result.metadata.get("created_at", datetime.utcnow().isoformat())),
                    metadata=result.metadata
                )
                for result in search_results.results
            ]
            
            # Detect patterns
            pattern_result = await self.pattern_detector.detect_patterns(
                memories=memories,
                min_cluster_size=arguments.get("min_cluster_size", 3)
            )
            
            result = {
                "success": True,
                "total_memories_analyzed": len(memories),
                "patterns_found": len(pattern_result.pattern_clusters),
                "knowledge_gaps_found": len(pattern_result.knowledge_gaps),
                "pattern_clusters": [],
                "knowledge_gaps": [],
                "insights": [],
            }
            
            # Add pattern clusters (limit to first 5)
            for cluster in pattern_result.pattern_clusters[:5]:
                result["pattern_clusters"].append({
                    "pattern_type": cluster.pattern_type.value,
                    "memory_count": len(cluster.memory_ids),
                    "confidence": cluster.confidence,
                    "representative_content": cluster.representative_content[:200] + "...",
                    "common_themes": cluster.common_themes[:3],
                })
            
            # Add knowledge gaps (limit to first 5)
            for gap in pattern_result.knowledge_gaps[:5]:
                result["knowledge_gaps"].append({
                    "topic": gap.topic,
                    "importance_score": gap.importance_score,
                    "related_memories": len(gap.related_memories),
                    "suggested_questions": gap.suggested_questions[:3],
                })
            
            # Add insights
            for insight in pattern_result.insights[:5]:
                result["insights"].append({
                    "type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                })
            
            # Add recommendations if requested
            if arguments.get("include_recommendations", True):
                recommendations = await self.pattern_detector.generate_recommendations(
                    pattern_result
                )
                result["recommendations"] = [
                    {
                        "action": rec.action,
                        "reason": rec.reason,
                        "priority": rec.priority,
                    }
                    for rec in recommendations[:5]
                ]
            
            return result
            
        except Exception as e:
            logger.error("Pattern detection failed", arguments=arguments, error=str(e))
            raise

    async def _handle_analyze_temporal_evolution(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle temporal evolution analysis requests."""
        try:
            if not self.temporal_analyzer:
                return {
                    "success": False,
                    "error": "Temporal analyzer not initialized",
                }

            # Search for memories in time window
            search_request = MemorySearchRequest(
                query=arguments.get("concept", "*"),
                agent_id=arguments.get("agent_id"),
                top_k=1000,
            )
            
            search_results = await self.memory_manager.search_memory(search_request)
            
            if not search_results.results:
                return {
                    "success": True,
                    "evolutions_found": 0,
                    "message": "No memories found to analyze",
                }
            
            # Convert to Memory objects
            from ...models.memory import Memory
            memories = [
                Memory(
                    id=result.id,
                    content=result.content,
                    agent_id=result.metadata.get("agent_id", "unknown"),
                    created_at=datetime.fromisoformat(result.metadata.get("created_at", datetime.utcnow().isoformat())),
                    metadata=result.metadata
                )
                for result in search_results.results
            ]
            
            # Filter by time window
            time_window_days = arguments.get("time_window_days", 30)
            cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
            memories = [m for m in memories if m.created_at >= cutoff_date]
            
            if not memories:
                return {
                    "success": True,
                    "evolutions_found": 0,
                    "message": f"No memories found in the last {time_window_days} days",
                }
            
            # Analyze temporal evolution
            evolution_result = await self.temporal_analyzer.analyze_evolution(
                memories=memories,
                time_window_days=time_window_days
            )
            
            result = {
                "success": True,
                "total_memories_analyzed": len(memories),
                "time_window_days": time_window_days,
                "concept_evolutions": [],
                "learning_progressions": [],
                "temporal_insights": [],
            }
            
            # Add concept evolutions (limit to first 5)
            for evolution in evolution_result.concept_evolutions[:5]:
                result["concept_evolutions"].append({
                    "concept": evolution.concept,
                    "evolution_type": evolution.evolution_type.value,
                    "confidence": evolution.confidence,
                    "start_understanding": evolution.start_understanding[:100] + "...",
                    "current_understanding": evolution.current_understanding[:100] + "...",
                    "key_transitions": len(evolution.key_transitions),
                })
            
            # Add learning progressions
            for progression in evolution_result.learning_progressions[:5]:
                result["learning_progressions"].append({
                    "topic": progression.topic,
                    "mastery_level": progression.mastery_level,
                    "learning_velocity": progression.learning_velocity,
                    "milestone_count": len(progression.milestones),
                    "next_concepts": progression.next_concepts[:3],
                })
            
            # Add temporal insights
            for insight in evolution_result.temporal_insights[:5]:
                result["temporal_insights"].append({
                    "type": insight.insight_type,
                    "description": insight.description,
                    "trend_direction": insight.trend_direction.value if hasattr(insight, "trend_direction") else None,
                    "confidence": insight.confidence,
                })
            
            # Add predictions if available
            if hasattr(evolution_result, "future_predictions"):
                result["future_predictions"] = [
                    {
                        "concept": pred.concept,
                        "predicted_evolution": pred.predicted_evolution,
                        "confidence": pred.confidence,
                        "timeframe_days": pred.timeframe_days,
                    }
                    for pred in evolution_result.future_predictions[:3]
                ]
            
            return result
            
        except Exception as e:
            logger.error("Temporal analysis failed", arguments=arguments, error=str(e))
            raise
    
    async def _handle_crawl_website(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle website crawling requests with natural language."""
        try:
            if not self.crawl_runner:
                return {
                    "success": False,
                    "error": "Crawl4AI runner not initialized",
                }
            
            # Parse natural language command
            command_text = arguments["command"]
            parsed_command = self.crawl_parser.parse(command_text)
            
            # Override with explicit parameters if provided
            if "max_pages" in arguments:
                parsed_command.max_pages = arguments["max_pages"]
            if "strategy" in arguments:
                parsed_command.strategy = CrawlStrategy(arguments["strategy"])
            if "store_in_memory" in arguments:
                parsed_command.store_in_memory = arguments["store_in_memory"]
            
            agent_id = arguments.get("agent_id", "tyra")
            
            # Check if URLs were extracted
            if not parsed_command.urls:
                suggestions = self.crawl_parser.suggest_improvements(parsed_command)
                return {
                    "success": False,
                    "error": "No URLs found in command",
                    "suggestions": suggestions,
                    "parsed_intent": parsed_command.intent.value,
                    "confidence": parsed_command.confidence,
                }
            
            # Start crawling
            all_results = []
            crawl_stats = {
                "total_urls": len(parsed_command.urls),
                "successful_crawls": 0,
                "failed_crawls": 0,
                "pages_stored": 0,
                "total_content_bytes": 0,
            }
            
            for url in parsed_command.urls:
                logger.info(
                    "Starting crawl",
                    url=url,
                    strategy=parsed_command.strategy.value,
                    max_pages=parsed_command.max_pages,
                )
                
                try:
                    # Perform crawl
                    crawl_results = await self.crawl_runner.crawl_url(
                        url=url,
                        strategy=parsed_command.strategy,
                        extract_content=True,
                        store_in_memory=False,  # We'll handle storage ourselves
                        force_recrawl=True
                    )
                    
                    # Process results
                    for result in crawl_results:
                        if result.success:
                            crawl_stats["successful_crawls"] += 1
                            crawl_stats["total_content_bytes"] += result.content_length
                            
                            # Store in memory if requested
                            if parsed_command.store_in_memory and result.content:
                                try:
                                    # Prepare metadata
                                    metadata = {
                                        "source": result.url,
                                        "title": result.title,
                                        "crawl_method": "crawl4ai",
                                        "crawl_command": command_text,
                                        "content_type": parsed_command.content_type.value,
                                        "crawl_time": datetime.utcnow().isoformat(),
                                    }
                                    
                                    if result.metadata:
                                        metadata.update(result.metadata)
                                    
                                    # Store using memory manager
                                    store_request = MemoryStoreRequest(
                                        content=result.content,
                                        agent_id=agent_id,
                                        metadata=metadata,
                                        extract_entities=parsed_command.extract_entities,
                                        chunk_content=True,  # Chunk large web content
                                    )
                                    
                                    store_result = await self.memory_manager.store_memory(store_request)
                                    crawl_stats["pages_stored"] += 1
                                    
                                    # Add to results
                                    all_results.append({
                                        "url": result.url,
                                        "title": result.title,
                                        "success": True,
                                        "content_length": result.content_length,
                                        "memory_id": store_result.memory_id,
                                        "chunks_created": len(store_result.chunk_ids),
                                        "entities_extracted": len(store_result.entities_created),
                                    })
                                    
                                except Exception as e:
                                    logger.error("Failed to store crawled content", url=result.url, error=str(e))
                                    all_results.append({
                                        "url": result.url,
                                        "title": result.title,
                                        "success": True,
                                        "content_length": result.content_length,
                                        "storage_error": str(e),
                                    })
                            else:
                                all_results.append({
                                    "url": result.url,
                                    "title": result.title,
                                    "success": True,
                                    "content_length": result.content_length,
                                    "stored": False,
                                })
                        else:
                            crawl_stats["failed_crawls"] += 1
                            all_results.append({
                                "url": result.url,
                                "success": False,
                                "error": result.error,
                            })
                    
                except Exception as e:
                    logger.error("Crawl failed", url=url, error=str(e))
                    crawl_stats["failed_crawls"] += 1
                    all_results.append({
                        "url": url,
                        "success": False,
                        "error": str(e),
                    })
                
                # Respect max pages limit
                if parsed_command.max_pages and crawl_stats["successful_crawls"] >= parsed_command.max_pages:
                    break
            
            # Get crawl statistics
            runner_stats = await self.crawl_runner.get_crawl_stats()
            
            return {
                "success": True,
                "command": command_text,
                "parsed_intent": parsed_command.intent.value,
                "strategy_used": parsed_command.strategy.value,
                "confidence": parsed_command.confidence,
                "results": all_results,
                "statistics": crawl_stats,
                "runner_stats": runner_stats,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error("Website crawl failed", arguments=arguments, error=str(e))
            raise

    async def _handle_suggest_related_memories(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle related memory suggestions requests."""
        try:
            if not self.local_suggester:
                return {
                    "success": False,
                    "error": "Related memory suggester not initialized",
                }
            
            # Get suggestions based on memory_id or content
            memory_id = arguments.get("memory_id")
            content = arguments.get("content")
            agent_id = arguments.get("agent_id", "tyra")
            limit = arguments.get("limit", 10)
            min_relevance = arguments.get("min_relevance", 0.3)
            
            if memory_id:
                suggestions = await self.local_suggester.suggest_for_memory(
                    memory_id=memory_id,
                    agent_id=agent_id,
                    limit=limit,
                    min_relevance=min_relevance
                )
            elif content:
                suggestions = await self.local_suggester.suggest_for_content(
                    content=content,
                    agent_id=agent_id,
                    limit=limit,
                    min_relevance=min_relevance
                )
            else:
                return {
                    "success": False,
                    "error": "Either memory_id or content must be provided",
                }
            
            return {
                "success": True,
                "suggestions": [
                    {
                        "memory_id": s.memory_id,
                        "relevance_score": s.relevance_score,
                        "suggestion_type": s.suggestion_type.value,
                        "content_preview": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                        "reasoning": s.reasoning,
                    }
                    for s in suggestions
                ],
                "total_found": len(suggestions),
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error("Related memory suggestions failed", arguments=arguments, error=str(e))
            raise

    async def _handle_detect_memory_connections(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory connections detection requests."""
        try:
            if not self.local_connector:
                return {
                    "success": False,
                    "error": "Memory connector not initialized",
                }
            
            agent_id = arguments.get("agent_id", "tyra")
            connection_types = arguments.get("connection_types", ["semantic", "temporal", "entity"])
            min_confidence = arguments.get("min_confidence", 0.5)
            
            connections = await self.local_connector.detect_connections(
                agent_id=agent_id,
                connection_types=connection_types,
                min_confidence=min_confidence
            )
            
            return {
                "success": True,
                "connections": [
                    {
                        "source_memory_id": c.source_memory_id,
                        "target_memory_id": c.target_memory_id,
                        "connection_type": c.connection_type.value,
                        "confidence": c.confidence,
                        "strength": c.strength,
                        "reasoning": c.reasoning,
                        "evidence": c.evidence,
                    }
                    for c in connections
                ],
                "total_found": len(connections),
                "agent_id": agent_id,
                "connection_types_analyzed": connection_types,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error("Memory connections detection failed", arguments=arguments, error=str(e))
            raise

    async def _handle_recommend_memory_organization(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory organization recommendations requests."""
        try:
            if not self.local_recommender:
                return {
                    "success": False,
                    "error": "Memory organization recommender not initialized",
                }
            
            agent_id = arguments.get("agent_id", "tyra")
            analysis_type = arguments.get("analysis_type", "all")
            
            recommendations = await self.local_recommender.analyze_organization(
                agent_id=agent_id,
                analysis_type=analysis_type
            )
            
            return {
                "success": True,
                "recommendations": [
                    {
                        "type": r.recommendation_type.value,
                        "priority": r.priority.value,
                        "impact_score": r.impact_score,
                        "description": r.description,
                        "suggested_actions": r.suggested_actions,
                        "affected_memories": r.affected_memory_ids,
                        "implementation_effort": r.implementation_effort.value,
                    }
                    for r in recommendations
                ],
                "total_recommendations": len(recommendations),
                "agent_id": agent_id,
                "analysis_type": analysis_type,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error("Memory organization recommendations failed", arguments=arguments, error=str(e))
            raise

    async def _handle_detect_knowledge_gaps(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle knowledge gaps detection requests."""
        try:
            if not self.local_detector:
                return {
                    "success": False,
                    "error": "Knowledge gaps detector not initialized",
                }
            
            agent_id = arguments.get("agent_id", "tyra")
            domains = arguments.get("domains")
            gap_types = arguments.get("gap_types", ["topic", "temporal", "detail"])
            
            gaps = await self.local_detector.detect_gaps(
                agent_id=agent_id,
                domains=domains,
                gap_types=gap_types
            )
            
            return {
                "success": True,
                "knowledge_gaps": [
                    {
                        "gap_type": g.gap_type.value,
                        "severity": g.severity.value,
                        "domain": g.domain,
                        "description": g.description,
                        "suggested_content": g.suggested_content,
                        "priority_score": g.priority_score,
                        "learning_path": g.learning_path,
                        "related_memories": g.related_memory_ids,
                    }
                    for g in gaps
                ],
                "total_gaps_found": len(gaps),
                "agent_id": agent_id,
                "domains_analyzed": domains or "all",
                "gap_types_analyzed": gap_types,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error("Knowledge gaps detection failed", arguments=arguments, error=str(e))
            raise

    async def _handle_web_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web search requests."""
        try:
            if not self.web_search_agent:
                return {
                    "success": False,
                    "error": "Web search agent not initialized",
                }
            
            query = arguments["query"]
            max_results = arguments.get("max_results", 5)
            force_refresh = arguments.get("force_refresh", False)
            agent_id = arguments.get("agent_id", "tyra")
            store_in_memory = arguments.get("store_in_memory", True)
            
            # Perform web search and integration
            search_results = await self.web_search_agent.search_and_integrate(
                query=query,
                max_results=max_results,
                force_refresh=force_refresh
            )
            
            # Format results for response
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "title": result.title,
                    "source": result.source,
                    "content_preview": result.text[:500] + "..." if len(result.text) > 500 else result.text,
                    "summary": result.summary,
                    "confidence_score": result.confidence_score,
                    "relevance_score": result.relevance_score,
                    "freshness_score": result.freshness_score,
                    "extraction_method": result.extraction_method.value,
                    "extraction_quality": result.extraction_quality.value,
                    "content_length": result.processed_length,
                    "timestamp": result.timestamp.isoformat(),
                }
                formatted_results.append(formatted_result)
            
            # Get search statistics
            search_stats = await self.web_search_agent.get_search_stats()
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "stored_in_memory": store_in_memory,
                "agent_id": agent_id,
                "search_statistics": search_stats,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error("Web search failed", arguments=arguments, error=str(e))
            raise

    async def run(self) -> None:
        """Run the MCP server."""
        try:
            logger.info("Starting Tyra Memory MCP Server...")

            # Pre-initialize components
            await self._initialize_components()

            logger.info("Tyra Memory MCP Server ready for connections")

            # Run the server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="tyra-memory-server",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=None, experimental_capabilities=None
                        ),
                    ),
                )

        except Exception as e:
            logger.error("Server startup failed", error=str(e))
            raise
        finally:
            # Cleanup
            if self.memory_manager:
                await self.memory_manager.close()
            
            # Stop file watcher service
            if self.file_watcher_manager:
                await self.file_watcher_manager.stop()

            logger.info("Tyra Memory MCP Server stopped")


async def main():
    """Main entry point for the server."""
    server = TyraMemoryServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
