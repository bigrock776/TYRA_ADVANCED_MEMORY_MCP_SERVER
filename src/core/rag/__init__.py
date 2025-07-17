"""
Advanced RAG Pipeline Module.

This module provides comprehensive RAG (Retrieval-Augmented Generation) capabilities including:
- Multi-modal content processing with CLIP integration
- Contextual chunk linking and relationship analysis
- Dynamic reranking with query intent detection
- Intent-aware retrieval with personalization
- Hallucination detection and confidence scoring

All operations are performed locally with zero external API calls.
"""

from .multimodal import (
    MultiModalProcessor,
    MultiModalContent,
    MultiModalSearchResult,
    ModalityType,
    DocumentType,
    CodeLanguage
)

from .chunk_linking import (
    ContextualChunkLinker,
    TextChunk,
    ContextualLink,
    ChunkLinkingResult,
    LinkType,
    ChunkType,
    LinkStrength
)

from .dynamic_reranking import (
    DynamicReranker,
    QueryIntent,
    QueryContext,
    RerankingStrategy,
    ContextType,
    RankingFeatures,
    RerankingResult
)

from .intent_detector import (
    IntentAwareRetriever,
    RetrievalStrategy,
    RetrievalMode,
    FilterCriteria,
    RetrievalContext,
    RetrievalResult,
    RetrievalAnalytics
)

from .reranker import (
    Reranker,
    RerankingResult as BaseRerankingResult
)

from .hallucination_detector import (
    HallucinationDetector,
    HallucinationResult,
    HallucinationType,
    ConfidenceLevel,
    ValidationMetrics
)

__all__ = [
    # Multi-modal processing
    "MultiModalProcessor",
    "MultiModalContent",
    "MultiModalSearchResult",
    "ModalityType",
    "DocumentType", 
    "CodeLanguage",
    
    # Chunk linking
    "ContextualChunkLinker",
    "TextChunk",
    "ContextualLink",
    "ChunkLinkingResult",
    "LinkType",
    "ChunkType",
    "LinkStrength",
    
    # Dynamic reranking
    "DynamicReranker",
    "QueryIntent",
    "QueryContext",
    "RerankingStrategy",
    "ContextType",
    "RankingFeatures",
    "RerankingResult",
    
    # Intent-aware retrieval
    "IntentAwareRetriever",
    "RetrievalStrategy",
    "RetrievalMode",
    "FilterCriteria",
    "RetrievalContext",
    "RetrievalResult",
    "RetrievalAnalytics",
    
    # Base reranking
    "Reranker",
    "BaseRerankingResult",
    
    # Hallucination detection
    "HallucinationDetector",
    "HallucinationResult",
    "HallucinationType",
    "ConfidenceLevel",
    "ValidationMetrics"
]