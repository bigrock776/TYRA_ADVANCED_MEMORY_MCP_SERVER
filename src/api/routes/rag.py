"""
RAG (Retrieval-Augmented Generation) API endpoints.

Provides advanced retrieval features including reranking,
hallucination detection, and confidence scoring.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.rag.pipeline_manager import (
    IntegratedRAGPipeline, 
    RAGPipelineResult, 
    PipelineConfig, 
    PipelineMode,
    PipelineStage
)
from ...core.rag.hallucination_detector import HallucinationDetector
from ...core.rag.reranker import Reranker
from ...core.memory.manager import MemoryManager
from ...core.embeddings.embedder import Embedder
from ...core.graph.neo4j_client import Neo4jClient
from ...core.cache.redis_cache import RedisCache
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Enums
class ConfidenceLevel(str, Enum):
    """Confidence levels for responses."""

    ROCK_SOLID = "rock_solid"  # 95%+ confidence
    HIGH = "high"  # 80-95% confidence
    FUZZY = "fuzzy"  # 60-80% confidence
    LOW = "low"  # Below 60% confidence


class RerankingModel(str, Enum):
    """Available reranking models."""

    CROSS_ENCODER = "cross_encoder"
    VLLM = "vllm"
    HYBRID = "hybrid"


# Request/Response Models
class RetrievalRequest(BaseModel):
    """RAG retrieval request."""

    query: str = Field(..., description="Query for retrieval")
    limit: int = Field(10, ge=1, le=50, description="Number of documents to retrieve")
    rerank: bool = Field(True, description="Apply reranking")
    detect_hallucination: bool = Field(True, description="Check for hallucination")
    min_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    include_graph_context: bool = Field(
        False, description="Include knowledge graph context"
    )


class Document(BaseModel):
    """Retrieved document with metadata."""

    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score")
    confidence: float = Field(..., description="Confidence score (0-1)")
    confidence_level: ConfidenceLevel = Field(
        ..., description="Confidence level category"
    )
    metadata: Dict[str, Any] = Field(default={}, description="Document metadata")
    graph_context: Optional[Dict[str, Any]] = Field(
        None, description="Related graph entities"
    )


class RetrievalResponse(BaseModel):
    """RAG retrieval response."""

    query: str = Field(..., description="Original query")
    documents: List[Document] = Field(..., description="Retrieved documents")
    overall_confidence: float = Field(..., description="Overall confidence score")
    hallucination_risk: float = Field(..., description="Hallucination risk score (0-1)")
    warnings: List[str] = Field(default=[], description="Any warnings or issues")


class RerankRequest(BaseModel):
    """Reranking request."""

    query: str = Field(..., description="Query for reranking")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to rerank")
    model: RerankingModel = Field(
        RerankingModel.CROSS_ENCODER, description="Reranking model"
    )
    top_k: int = Field(10, ge=1, le=50, description="Number of top documents to return")


class HallucinationCheckRequest(BaseModel):
    """Hallucination check request."""

    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response to check")
    documents: List[Dict[str, Any]] = Field(..., description="Source documents")
    threshold: float = Field(0.75, ge=0.0, le=1.0, description="Confidence threshold")


class HallucinationCheckResponse(BaseModel):
    """Hallucination check response."""

    is_grounded: bool = Field(
        ..., description="Whether response is grounded in documents"
    )
    confidence: float = Field(..., description="Grounding confidence score")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level")
    problematic_claims: List[str] = Field(
        default=[], description="Claims that lack grounding"
    )
    supporting_evidence: List[Dict[str, Any]] = Field(
        default=[], description="Supporting evidence from documents"
    )


class AnswerGenerationRequest(BaseModel):
    """Request for answer generation with RAG."""

    query: str = Field(..., description="User query")
    context_limit: int = Field(10, ge=1, le=50, description="Maximum context documents")
    temperature: float = Field(
        0.1, ge=0.0, le=1.0, description="Generation temperature"
    )
    require_confidence: ConfidenceLevel = Field(
        ConfidenceLevel.HIGH, description="Required confidence level"
    )
    include_citations: bool = Field(True, description="Include source citations")


class IntegratedRAGRequest(BaseModel):
    """Request for integrated RAG pipeline execution."""
    
    query: str = Field(..., description="User query")
    user_id: Optional[str] = Field(None, description="User identifier")
    pipeline_mode: PipelineMode = Field(PipelineMode.STANDARD, description="Pipeline execution mode")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results to return")
    min_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    enable_multimodal: bool = Field(True, description="Enable multi-modal processing")
    enable_reranking: bool = Field(True, description="Enable dynamic reranking")
    enable_hallucination_check: bool = Field(True, description="Enable hallucination detection")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class PipelineAnalyticsResponse(BaseModel):
    """Response for pipeline analytics."""
    
    total_executions: int = Field(..., description="Total pipeline executions")
    avg_execution_time_ms: float = Field(..., description="Average execution time")
    stage_success_rates: Dict[str, float] = Field(..., description="Success rate per stage")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    avg_retrieval_count: float = Field(..., description="Average retrieval count")
    avg_final_result_count: float = Field(..., description="Average final result count")


# Dependencies
async def get_reranker() -> Reranker:
    """Get reranker instance."""
    try:
        return get_provider(ProviderType.RERANKER, "default")
    except Exception as e:
        logger.error(f"Failed to get reranker: {e}")
        raise HTTPException(status_code=500, detail="Reranker unavailable")


async def get_hallucination_detector() -> HallucinationDetector:
    """Get hallucination detector instance."""
    try:
        return get_provider(ProviderType.HALLUCINATION_DETECTOR, "default")
    except Exception as e:
        logger.error(f"Failed to get hallucination detector: {e}")
        raise HTTPException(
            status_code=500, detail="Hallucination detector unavailable"
        )


async def get_searcher() -> Searcher:
    """Get searcher instance."""
    try:
        return get_provider(ProviderType.SEARCHER, "default")
    except Exception as e:
        logger.error(f"Failed to get searcher: {e}")
        raise HTTPException(status_code=500, detail="Searcher unavailable")


async def get_integrated_pipeline() -> IntegratedRAGPipeline:
    """Get integrated RAG pipeline instance."""
    try:
        # Get required dependencies
        memory_manager = get_provider(ProviderType.MEMORY_MANAGER, "default")
        embedder = get_provider(ProviderType.EMBEDDER, "default")
        graph_client = get_provider(ProviderType.GRAPH_CLIENT, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        
        return IntegratedRAGPipeline(
            memory_manager=memory_manager,
            embedder=embedder, 
            graph_client=graph_client,
            cache=cache
        )
    except Exception as e:
        logger.error(f"Failed to get integrated pipeline: {e}")
        raise HTTPException(status_code=500, detail="RAG pipeline unavailable")


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_with_rag(
    request: RetrievalRequest,
    searcher: Searcher = Depends(get_searcher),
    reranker: Reranker = Depends(get_reranker),
    hallucination_detector: HallucinationDetector = Depends(get_hallucination_detector),
):
    """
    Retrieve documents with advanced RAG features.

    Performs retrieval with optional reranking and hallucination detection.
    Returns documents with confidence scores and warnings.
    """
    try:
        # Initial retrieval
        initial_results = await searcher.search(
            query=request.query,
            strategy="hybrid",
            limit=request.limit * 2 if request.rerank else request.limit,
        )

        # Rerank if requested
        if request.rerank and initial_results:
            reranked_results = await reranker.rerank(
                query=request.query,
                documents=[
                    {"id": r["memory_id"], "text": r["text"]} for r in initial_results
                ],
                top_k=request.limit,
            )
            # Map reranked scores back
            result_map = {r["memory_id"]: r for r in initial_results}
            results = []
            for reranked in reranked_results:
                original = result_map[reranked["id"]]
                original["score"] = reranked["score"]
                results.append(original)
        else:
            results = initial_results[: request.limit]

        # Process results with confidence scoring
        documents = []
        confidence_scores = []
        warnings = []

        for result in results:
            # Calculate confidence
            confidence = await _calculate_confidence(result, request.query)
            confidence_scores.append(confidence)

            # Apply minimum confidence filter
            if request.min_confidence and confidence < request.min_confidence:
                continue

            # Add graph context if requested
            graph_context = None
            if request.include_graph_context:
                graph_context = await _get_graph_context(result["memory_id"])

            documents.append(
                Document(
                    id=result["memory_id"],
                    text=result["text"],
                    score=result["score"],
                    confidence=confidence,
                    confidence_level=_get_confidence_level(confidence),
                    metadata=result.get("metadata", {}),
                    graph_context=graph_context,
                )
            )

        # Calculate overall confidence
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        # Assess hallucination risk
        hallucination_risk = 0.0
        if request.detect_hallucination and documents:
            hallucination_assessment = await hallucination_detector.assess_risk(
                query=request.query, documents=[d.text for d in documents]
            )
            hallucination_risk = hallucination_assessment["risk_score"]

            if hallucination_risk > 0.5:
                warnings.append(
                    f"High hallucination risk detected: {hallucination_risk:.2f}"
                )

        # Add warnings for low confidence
        if overall_confidence < 0.6:
            warnings.append(f"Low overall confidence: {overall_confidence:.2f}")

        return RetrievalResponse(
            query=request.query,
            documents=documents,
            overall_confidence=overall_confidence,
            hallucination_risk=hallucination_risk,
            warnings=warnings,
        )

    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rerank", response_model=List[Document])
async def rerank_documents(
    request: RerankRequest, reranker: Reranker = Depends(get_reranker)
):
    """
    Rerank a set of documents for a query.

    Uses advanced reranking models to improve relevance ordering.
    """
    try:
        # Perform reranking
        reranked = await reranker.rerank(
            query=request.query,
            documents=request.documents,
            model=request.model,
            top_k=request.top_k,
        )

        # Convert to response format
        documents = []
        for doc in reranked:
            confidence = await _calculate_confidence(doc, request.query)

            documents.append(
                Document(
                    id=doc["id"],
                    text=doc["text"],
                    score=doc["score"],
                    confidence=confidence,
                    confidence_level=_get_confidence_level(confidence),
                    metadata=doc.get("metadata", {}),
                )
            )

        return documents

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-hallucination", response_model=HallucinationCheckResponse)
async def check_hallucination(
    request: HallucinationCheckRequest,
    detector: HallucinationDetector = Depends(get_hallucination_detector),
):
    """
    Check if a response is grounded in the provided documents.

    Detects potential hallucinations and returns confidence scores.
    """
    try:
        # Perform hallucination check
        result = await detector.check_grounding(
            query=request.query,
            response=request.response,
            documents=request.documents,
            threshold=request.threshold,
        )

        # Determine confidence level
        confidence_level = _get_confidence_level(result["confidence"])

        return HallucinationCheckResponse(
            is_grounded=result["is_grounded"],
            confidence=result["confidence"],
            confidence_level=confidence_level,
            problematic_claims=result.get("problematic_claims", []),
            supporting_evidence=result.get("supporting_evidence", []),
        )

    except Exception as e:
        logger.error(f"Hallucination check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-answer")
async def generate_answer_with_rag(
    request: AnswerGenerationRequest,
    searcher: Searcher = Depends(get_searcher),
    reranker: Reranker = Depends(get_reranker),
    detector: HallucinationDetector = Depends(get_hallucination_detector),
):
    """
    Generate an answer using RAG with confidence guarantees.

    Retrieves relevant context, generates an answer, and ensures
    it meets the required confidence level.
    """
    try:
        # Retrieve relevant context
        initial_results = await searcher.search(
            query=request.query, strategy="hybrid", limit=request.context_limit * 2
        )

        # Rerank for better relevance
        reranked = await reranker.rerank(
            query=request.query,
            documents=[
                {"id": r["memory_id"], "text": r["text"]} for r in initial_results
            ],
            top_k=request.context_limit,
        )

        # Filter by required confidence level
        min_confidence = _get_min_confidence_for_level(request.require_confidence)
        context_docs = []

        for doc in reranked:
            confidence = await _calculate_confidence(doc, request.query)
            if confidence >= min_confidence:
                context_docs.append(doc)

        if not context_docs:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found meeting confidence requirement: {request.require_confidence}",
            )

        # Generate answer using LLM with context
        answer = await _generate_rag_answer(request.query, context_docs)

        # Check for hallucination
        hallucination_check = await detector.check_grounding(
            query=request.query, response=answer, documents=context_docs
        )

        # Add citations if requested
        citations = []
        if request.include_citations:
            citations = [
                {"id": doc["id"], "text": doc["text"][:200] + "..."}
                for doc in context_docs[:3]
            ]

        return {
            "query": request.query,
            "answer": answer,
            "confidence": hallucination_check["confidence"],
            "confidence_level": _get_confidence_level(
                hallucination_check["confidence"]
            ),
            "context_used": len(context_docs),
            "citations": citations,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confidence-thresholds")
async def get_confidence_thresholds():
    """
    Get confidence threshold definitions.

    Returns the thresholds used for different confidence levels.
    """
    return {
        "levels": {
            "rock_solid": {
                "min_score": 0.95,
                "description": "Extremely high confidence, safe for automated actions",
            },
            "high": {
                "min_score": 0.80,
                "description": "High confidence, generally reliable",
            },
            "fuzzy": {
                "min_score": 0.60,
                "description": "Moderate confidence, may need verification",
            },
            "low": {
                "min_score": 0.0,
                "description": "Low confidence, use with caution",
            },
        }
    }


async def _calculate_confidence(document: Dict[str, Any], query: str) -> float:
    """Calculate confidence score for a document."""
    # Simple confidence calculation based on relevance score
    # In production, this would use more sophisticated methods
    base_score = document.get("score", 0.5)

    # Adjust based on metadata quality
    has_metadata = bool(document.get("metadata"))
    metadata_boost = 0.1 if has_metadata else 0.0

    # Adjust based on recency (recent content gets boost)
    recency_boost = _calculate_recency_boost(document)

    confidence = min(1.0, base_score + metadata_boost + recency_boost)
    return confidence


def _calculate_recency_boost(document: Dict[str, Any]) -> float:
    """Calculate recency boost based on document age."""
    try:
        from datetime import datetime, timedelta
        
        # Get document timestamp
        created_at = document.get("created_at")
        if not created_at:
            # Try alternative timestamp fields
            created_at = document.get("timestamp") or document.get("date")
        
        if not created_at:
            return 0.0  # No timestamp available, no boost
        
        # Parse timestamp if it's a string
        if isinstance(created_at, str):
            try:
                from dateutil.parser import parse
                created_at = parse(created_at)
            except:
                return 0.0  # Invalid timestamp format
        
        # Calculate age in days
        now = datetime.utcnow()
        if created_at.tzinfo:
            # Make now timezone-aware if created_at is
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        
        age_delta = now - created_at
        age_days = age_delta.total_seconds() / (24 * 3600)
        
        # Calculate recency boost (exponential decay)
        # Recent documents (0-7 days) get full boost (0.1)
        # Older documents get decreasing boost
        if age_days < 0:
            return 0.1  # Future dates get max boost
        elif age_days <= 1:
            return 0.1  # 1 day: full boost
        elif age_days <= 7:
            return 0.08  # 1 week: high boost
        elif age_days <= 30:
            return 0.05  # 1 month: medium boost
        elif age_days <= 90:
            return 0.02  # 3 months: small boost
        else:
            return 0.0  # Older than 3 months: no boost
            
    except Exception as e:
        # If any error occurs, return neutral boost
        return 0.0


def _get_confidence_level(confidence: float) -> ConfidenceLevel:
    """Map confidence score to level."""
    if confidence >= 0.95:
        return ConfidenceLevel.ROCK_SOLID
    elif confidence >= 0.80:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.60:
        return ConfidenceLevel.FUZZY
    else:
        return ConfidenceLevel.LOW


def _get_min_confidence_for_level(level: ConfidenceLevel) -> float:
    """Get minimum confidence score for a level."""
    mapping = {
        ConfidenceLevel.ROCK_SOLID: 0.95,
        ConfidenceLevel.HIGH: 0.80,
        ConfidenceLevel.FUZZY: 0.60,
        ConfidenceLevel.LOW: 0.0,
    }
    return mapping[level]


async def _get_graph_context(memory_id: str) -> Optional[Dict[str, Any]]:
    """Get knowledge graph context for a memory."""
    try:
        from core.graph.neo4j_client import get_neo4j_client
        
        client = await get_neo4j_client()
        if not client:
            return None
            
        # Query for entities and relationships connected to this memory
        entities = await client.get_memory_entities(memory_id)
        relationships = await client.get_memory_relationships(memory_id)
        
        return {
            "entities": [e.get("name", e.get("id", "unknown")) for e in entities[:5]],
            "relationships": [r.get("type", "RELATED") for r in relationships[:5]],
            "temporal_context": datetime.utcnow().strftime("%Y-%m"),
        }
    except Exception as e:
        logger.error(f"Error getting graph context for memory {memory_id}: {e}")
        return None


async def _generate_rag_answer(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """Generate an answer using LLM with RAG context."""
    try:
        import aiohttp
        
        # Prepare context from documents
        context_text = "\n\n".join([
            f"Document {i+1}: {doc.get('text', '')[:500]}"
            for i, doc in enumerate(context_docs[:5])
        ])
        
        # Build the prompt
        system_prompt = """You are Tyra, an AI assistant that provides accurate answers based on the provided context documents. 
        Use only the information in the context to answer questions. If the context doesn't contain enough information to answer fully, say so clearly."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        # Call local vLLM server
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for factual accuracy
            "max_tokens": 500,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.error(f"LLM request failed with status {response.status}")
                    return _generate_fallback_rag_answer(query, context_docs)
                    
    except Exception as e:
        logger.error(f"Error generating RAG answer: {e}")
        return _generate_fallback_rag_answer(query, context_docs)


def _generate_fallback_rag_answer(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """Generate a fallback RAG answer when LLM is unavailable."""
    if not context_docs:
        return f"I found no relevant context to answer your query: {query}"
    
    # Extract key information from context
    context_snippets = []
    for doc in context_docs[:3]:
        text = doc.get("text", "")
        if len(text) > 200:
            text = text[:197] + "..."
        context_snippets.append(text)
    
    answer = f"Based on {len(context_docs)} relevant documents from the memory system:\n\n"
    answer += "\n\n".join(f"• {snippet}" for snippet in context_snippets)
    answer += f"\n\n(Note: LLM service unavailable - showing relevant context for: {query})"
    
    return answer


# New Integrated Pipeline Endpoints

@router.post("/pipeline/execute", response_model=RAGPipelineResult)
async def execute_integrated_pipeline(
    request: IntegratedRAGRequest,
    pipeline: IntegratedRAGPipeline = Depends(get_integrated_pipeline)
):
    """
    Execute the complete integrated RAG pipeline.
    
    Runs the full pipeline including intent detection, multi-modal processing,
    retrieval, chunk linking, reranking, and hallucination detection.
    """
    try:
        # Create pipeline configuration
        enabled_stages = [stage for stage in PipelineStage]
        
        if not request.enable_multimodal:
            enabled_stages.remove(PipelineStage.MULTIMODAL_PROCESSING)
        if not request.enable_reranking:
            enabled_stages.remove(PipelineStage.RERANKING)
        if not request.enable_hallucination_check:
            enabled_stages.remove(PipelineStage.HALLUCINATION_CHECK)
        
        config = PipelineConfig(
            mode=request.pipeline_mode,
            enabled_stages=enabled_stages,
            max_retrieval_results=request.max_results,
            min_confidence_threshold=request.min_confidence,
            enable_hallucination_detection=request.enable_hallucination_check,
            enable_multi_modal=request.enable_multimodal,
            enable_reranking=request.enable_reranking
        )
        
        # Execute pipeline
        result = await pipeline.execute_pipeline(
            query=request.query,
            user_id=request.user_id,
            config=config,
            context=request.context
        )
        
        logger.info(
            "Integrated RAG pipeline executed",
            query=request.query[:50],
            result_count=len(result.results),
            total_time_ms=result.metrics.total_time_ms
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Integrated pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/analytics", response_model=PipelineAnalyticsResponse)
async def get_pipeline_analytics(
    pipeline: IntegratedRAGPipeline = Depends(get_integrated_pipeline)
):
    """
    Get analytics and performance metrics for the RAG pipeline.
    
    Returns statistics about pipeline execution including success rates,
    performance metrics, and cache efficiency.
    """
    try:
        analytics = await pipeline.get_pipeline_analytics()
        
        return PipelineAnalyticsResponse(
            total_executions=analytics["total_executions"],
            avg_execution_time_ms=analytics["avg_execution_time_ms"],
            stage_success_rates=analytics["stage_success_rates"],
            cache_hit_rate=analytics["cache_hit_rate"],
            avg_retrieval_count=analytics["avg_retrieval_count"],
            avg_final_result_count=analytics["avg_final_result_count"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get pipeline analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/config")
async def update_pipeline_config(
    config: Dict[str, Any],
    pipeline: IntegratedRAGPipeline = Depends(get_integrated_pipeline)
):
    """
    Update pipeline configuration.
    
    Allows dynamic updates to pipeline behavior including enabled stages,
    performance thresholds, and quality settings.
    """
    try:
        # Validate and update configuration
        if "default_config" in config:
            pipeline.default_config = PipelineConfig(**config["default_config"])
        
        return {"status": "success", "message": "Pipeline configuration updated"}
        
    except Exception as e:
        logger.error(f"Failed to update pipeline config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
