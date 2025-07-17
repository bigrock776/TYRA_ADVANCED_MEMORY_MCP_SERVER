"""
API route modules for Tyra MCP Memory Server.

Provides comprehensive REST API endpoints for all memory system operations:
- Memory CRUD operations
- Advanced search and retrieval  
- Memory synthesis and analysis
- RAG pipeline features
- Real-time chat interfaces
- Graph-based operations
- System administration
- Analytics and insights
- Webhook integrations
"""

from .admin import router as admin_router
from .analytics import router as analytics_router
from .chat import router as chat_router
from .crawling import router as crawling_router
from .embeddings import router as embeddings_router
from .file_watcher import router as file_watcher_router
from .graph import router as graph_router
from .health import router as health_router
from .ingestion import router as ingestion_router
from .memory import router as memory_router
from .observability import router as observability_router
from .performance import router as performance_router
from .personalization import router as personalization_router
from .prediction import router as prediction_router
from .rag import router as rag_router
from .search import router as search_router
from .security import router as security_router
from .suggestions import router as suggestions_router
from .synthesis import router as synthesis_router
from .trading_data import router as trading_data_router
from .webhooks import router as webhooks_router

__all__ = [
    "admin_router",
    "analytics_router", 
    "chat_router",
    "crawling_router",
    "embeddings_router",
    "file_watcher_router",
    "graph_router",
    "health_router",
    "ingestion_router",
    "memory_router",
    "observability_router",
    "performance_router",
    "personalization_router",
    "prediction_router",
    "rag_router", 
    "search_router",
    "security_router",
    "suggestions_router",
    "synthesis_router",
    "trading_data_router",
    "webhooks_router",
]