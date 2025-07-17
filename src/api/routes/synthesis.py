"""
Memory Synthesis API endpoints.

Provides REST API for advanced memory synthesis operations including:
- Memory deduplication and merging
- AI-powered summarization with anti-hallucination
- Pattern detection and knowledge gap analysis
- Temporal evolution tracking
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict

from ...core.synthesis import (
    DeduplicationEngine,
    SummarizationEngine,
    PatternDetector,
    TemporalAnalyzer,
    SummarizationType,
    DuplicateType,
    MergeStrategy,
    PatternType,
)
from ...core.memory.manager import MemoryManager, MemorySearchRequest
from ...core.clients.vllm_client import VLLMClient
from ...core.rag.hallucination_detector import HallucinationDetector
from ...core.providers.embeddings.embedder import Embedder
from ...models.memory import Memory
from ...core.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/synthesis", tags=["Memory Synthesis"])


# Request/Response Models
class DeduplicationRequest(BaseModel):
    """Request for memory deduplication."""
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    similarity_threshold: float = Field(0.85, ge=0.7, le=1.0, description="Similarity threshold for duplicates")
    max_memories: int = Field(1000, ge=10, le=5000, description="Maximum memories to process")
    auto_merge: bool = Field(False, description="Automatically merge high confidence duplicates")
    merge_strategy: str = Field("preserve_newest", description="Strategy for merging duplicates")


class SummarizationRequest(BaseModel):
    """Request for memory summarization."""
    memory_ids: List[str] = Field(..., min_items=1, description="Memory IDs to summarize")
    summary_type: str = Field("hybrid", description="Type of summarization")
    max_length: int = Field(200, ge=50, le=500, description="Maximum summary length")
    min_length: int = Field(50, ge=10, le=200, description="Minimum summary length")
    include_quality_metrics: bool = Field(True, description="Include quality assessment")


class PatternDetectionRequest(BaseModel):
    """Request for pattern detection."""
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    min_cluster_size: int = Field(3, ge=2, le=10, description="Minimum memories per pattern")
    max_memories: int = Field(500, ge=50, le=2000, description="Maximum memories to analyze")
    include_recommendations: bool = Field(True, description="Include learning recommendations")
    pattern_types: List[str] = Field(default_factory=list, description="Specific pattern types to detect")


class TemporalAnalysisRequest(BaseModel):
    """Request for temporal evolution analysis."""
    concept: Optional[str] = Field(None, description="Specific concept to track")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    time_window_days: int = Field(30, ge=1, le=365, description="Analysis time window")
    include_predictions: bool = Field(True, description="Include future predictions")


# Response Models
class DeduplicationResult(BaseModel):
    """Result from deduplication operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    total_memories_processed: int
    duplicates_found: int
    duplicates_merged: int = 0
    duplicate_groups: List[Dict[str, Any]]
    processing_time_seconds: float
    savings_estimate: Dict[str, Union[int, float]]


class SummarizationResult(BaseModel):
    """Result from summarization operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    summary: str
    key_points: List[str]
    confidence_score: float
    quality_metrics: Optional[Dict[str, float]] = None
    source_memory_count: int
    summary_type: str
    processing_time_seconds: float


class PatternDetectionResult(BaseModel):
    """Result from pattern detection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    total_memories_analyzed: int
    patterns_found: int
    pattern_clusters: List[Dict[str, Any]]
    knowledge_gaps: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    recommendations: Optional[List[Dict[str, Any]]] = None
    processing_time_seconds: float


class TemporalAnalysisResult(BaseModel):
    """Result from temporal analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    total_memories_analyzed: int
    time_window_days: int
    concept_evolutions: List[Dict[str, Any]]
    learning_progressions: List[Dict[str, Any]]
    temporal_insights: List[Dict[str, Any]]
    future_predictions: Optional[List[Dict[str, Any]]] = None
    processing_time_seconds: float


# Dependency injection for components
async def get_memory_manager() -> MemoryManager:
    """Get memory manager instance."""
    manager = MemoryManager()
    await manager.initialize()
    return manager


async def get_vllm_client() -> Optional[VLLMClient]:
    """Get vLLM client instance."""
    try:
        return VLLMClient()
    except Exception as e:
        logger.warning(f"Failed to initialize vLLM client: {e}")
        return None


async def get_synthesis_components(
    memory_manager: MemoryManager = Depends(get_memory_manager),
    vllm_client: Optional[VLLMClient] = Depends(get_vllm_client)
) -> Dict[str, Any]:
    """Get initialized synthesis components."""
    components = {}
    
    if not memory_manager.embedding_provider:
        raise HTTPException(
            status_code=503,
            detail="Embedding provider not available - synthesis features disabled"
        )
    
    embedder = memory_manager.embedding_provider
    
    # Initialize hallucination detector
    hallucination_detector = None
    try:
        hallucination_detector = HallucinationDetector(embedder)
    except Exception as e:
        logger.warning(f"Failed to initialize hallucination detector: {e}")
    
    # Initialize synthesis components
    try:
        components["deduplication"] = DeduplicationEngine(
            embedder=embedder,
            cache=None  # Could add cache if available
        )
        
        components["summarization"] = SummarizationEngine(
            embedder=embedder,
            vllm_client=vllm_client,
            hallucination_detector=hallucination_detector,
            cache=None
        )
        
        components["pattern_detection"] = PatternDetector(
            embedder=embedder,
            cache=None
        )
        
        components["temporal_analysis"] = TemporalAnalyzer(
            embedder=embedder,
            cache=None
        )
        
        components["memory_manager"] = memory_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize synthesis components: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Synthesis components initialization failed: {str(e)}"
        )
    
    return components


# API Endpoints

@router.post("/deduplicate", response_model=DeduplicationResult)
async def deduplicate_memories(
    request: DeduplicationRequest,
    background_tasks: BackgroundTasks,
    components: Dict[str, Any] = Depends(get_synthesis_components)
) -> DeduplicationResult:
    """
    Identify and optionally merge duplicate memories using semantic similarity.
    
    This endpoint analyzes memories for duplicates using advanced embedding-based
    similarity detection and provides options for automated merging.
    """
    start_time = datetime.utcnow()
    
    try:
        deduplication_engine = components["deduplication"]
        memory_manager = components["memory_manager"]
        
        # Search for memories to deduplicate
        search_request = MemorySearchRequest(
            query="*",  # Get all memories
            agent_id=request.agent_id,
            session_id=request.session_id,
            top_k=request.max_memories,
        )
        
        search_results = await memory_manager.search_memory(search_request)
        
        if not search_results.results:
            return DeduplicationResult(
                success=True,
                total_memories_processed=0,
                duplicates_found=0,
                duplicate_groups=[],
                processing_time_seconds=0.0,
                savings_estimate={"memory_count": 0, "storage_bytes": 0}
            )
        
        # Convert to Memory objects
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
        
        logger.info(f"Starting deduplication of {len(memories)} memories")
        
        # Find duplicates
        duplicate_groups = await deduplication_engine.find_duplicates(
            memories=memories,
            threshold=request.similarity_threshold
        )
        
        duplicates_merged = 0
        processed_groups = []
        
        # Process duplicate groups
        for group in duplicate_groups:
            group_info = {
                "duplicate_type": group.duplicate_type.value,
                "confidence": group.confidence,
                "memory_ids": group.memory_ids,
                "memory_count": len(group.memory_ids),
                "representative_content": group.suggested_merge[:200] + "..." if group.suggested_merge else "",
                "merge_strategy": group.merge_strategy.value,
                "merged": False
            }
            
            # Auto-merge if requested and high confidence
            if request.auto_merge and group.confidence > 0.9:
                try:
                    merged_strategy = MergeStrategy(request.merge_strategy)
                    merged = await deduplication_engine.merge_duplicates(
                        group, strategy=merged_strategy
                    )
                    
                    group_info.update({
                        "merged": True,
                        "merged_content": merged.content[:200] + "...",
                        "merged_id": merged.id
                    })
                    
                    duplicates_merged += 1
                    
                    # Schedule background storage update
                    # background_tasks.add_task(update_merged_memory, merged)
                    
                except Exception as e:
                    logger.error(f"Failed to merge duplicate group: {e}")
                    group_info["merge_error"] = str(e)
            
            processed_groups.append(group_info)
        
        # Calculate savings estimate
        total_duplicate_memories = sum(len(group.memory_ids) - 1 for group in duplicate_groups)
        estimated_storage_savings = total_duplicate_memories * 1024  # Rough estimate
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = DeduplicationResult(
            success=True,
            total_memories_processed=len(memories),
            duplicates_found=len(duplicate_groups),
            duplicates_merged=duplicates_merged,
            duplicate_groups=processed_groups,
            processing_time_seconds=processing_time,
            savings_estimate={
                "memory_count": total_duplicate_memories,
                "storage_bytes": estimated_storage_savings,
                "storage_mb": round(estimated_storage_savings / (1024 * 1024), 2)
            }
        )
        
        logger.info(
            "Deduplication completed",
            total_processed=len(memories),
            duplicates_found=len(duplicate_groups),
            duplicates_merged=duplicates_merged,
            processing_time=processing_time
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Deduplication operation failed: {str(e)}"
        )


@router.post("/summarize", response_model=SummarizationResult)
async def summarize_memories(
    request: SummarizationRequest,
    components: Dict[str, Any] = Depends(get_synthesis_components)
) -> SummarizationResult:
    """
    Generate AI-powered summaries of memory clusters with anti-hallucination.
    
    This endpoint creates structured summaries using advanced AI with comprehensive
    validation to prevent hallucinations and ensure factual consistency.
    """
    start_time = datetime.utcnow()
    
    try:
        summarization_engine = components["summarization"]
        memory_manager = components["memory_manager"]
        
        # Fetch memories by IDs
        memories_to_summarize = []
        for memory_id in request.memory_ids:
            search_request = MemorySearchRequest(
                query=memory_id,
                top_k=1,
                search_type="vector"
            )
            search_results = await memory_manager.search_memory(search_request)
            
            if search_results.results:
                result = search_results.results[0]
                memory = Memory(
                    id=result.id,
                    content=result.content,
                    agent_id=result.metadata.get("agent_id", "unknown"),
                    created_at=datetime.fromisoformat(result.metadata.get("created_at", datetime.utcnow().isoformat())),
                    metadata=result.metadata
                )
                memories_to_summarize.append(memory)
        
        if not memories_to_summarize:
            raise HTTPException(
                status_code=404,
                detail="No memories found for provided IDs"
            )
        
        logger.info(f"Starting summarization of {len(memories_to_summarize)} memories")
        
        # Generate summary
        summary_type = SummarizationType(request.summary_type)
        summary = await summarization_engine.summarize_memories(
            memories=memories_to_summarize,
            summarization_type=summary_type,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        # Evaluate quality if requested
        quality_metrics = None
        if request.include_quality_metrics:
            source_texts = [m.content for m in memories_to_summarize]
            quality_metrics = await summarization_engine.evaluate_summary(
                summary.summary,
                source_texts
            )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = SummarizationResult(
            success=True,
            summary=summary.summary,
            key_points=summary.key_points,
            confidence_score=summary.confidence_score,
            quality_metrics={
                "rouge_l": quality_metrics.rouge_l_score,
                "factual_consistency": quality_metrics.factual_consistency,
                "hallucination_score": quality_metrics.hallucination_score,
                "overall_quality": quality_metrics.overall_quality.value,
            } if quality_metrics else None,
            source_memory_count=len(memories_to_summarize),
            summary_type=summary_type.value,
            processing_time_seconds=processing_time
        )
        
        logger.info(
            "Summarization completed",
            memory_count=len(memories_to_summarize),
            summary_length=len(summary.summary),
            confidence=summary.confidence_score,
            processing_time=processing_time
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization operation failed: {str(e)}"
        )


@router.post("/detect-patterns", response_model=PatternDetectionResult)
async def detect_patterns(
    request: PatternDetectionRequest,
    components: Dict[str, Any] = Depends(get_synthesis_components)
) -> PatternDetectionResult:
    """
    Detect patterns and knowledge gaps across memories.
    
    This endpoint analyzes memory collections to identify recurring patterns,
    knowledge clusters, and gaps in understanding.
    """
    start_time = datetime.utcnow()
    
    try:
        pattern_detector = components["pattern_detection"]
        memory_manager = components["memory_manager"]
        
        # Search for memories to analyze
        search_request = MemorySearchRequest(
            query="*",
            agent_id=request.agent_id,
            session_id=request.session_id,
            top_k=request.max_memories,
        )
        
        search_results = await memory_manager.search_memory(search_request)
        
        if not search_results.results:
            return PatternDetectionResult(
                success=True,
                total_memories_analyzed=0,
                patterns_found=0,
                pattern_clusters=[],
                knowledge_gaps=[],
                insights=[],
                processing_time_seconds=0.0
            )
        
        # Convert to Memory objects
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
        
        logger.info(f"Starting pattern detection on {len(memories)} memories")
        
        # Detect patterns
        pattern_result = await pattern_detector.detect_patterns(
            memories=memories,
            min_cluster_size=request.min_cluster_size
        )
        
        # Process results
        pattern_clusters = []
        for cluster in pattern_result.pattern_clusters[:10]:  # Limit results
            pattern_clusters.append({
                "pattern_type": cluster.pattern_type.value,
                "memory_count": len(cluster.memory_ids),
                "confidence": cluster.confidence,
                "representative_content": cluster.representative_content[:200] + "...",
                "common_themes": cluster.common_themes[:3],
                "memory_ids": cluster.memory_ids[:5],  # Limit for response size
            })
        
        knowledge_gaps = []
        for gap in pattern_result.knowledge_gaps[:10]:
            knowledge_gaps.append({
                "topic": gap.topic,
                "importance_score": gap.importance_score,
                "related_memory_count": len(gap.related_memories),
                "suggested_questions": gap.suggested_questions[:3],
                "confidence": getattr(gap, 'confidence', 0.8),
            })
        
        insights = []
        for insight in pattern_result.insights[:10]:
            insights.append({
                "type": insight.insight_type,
                "description": insight.description,
                "confidence": insight.confidence,
                "supporting_evidence": getattr(insight, 'supporting_evidence', [])[:3],
            })
        
        # Generate recommendations if requested
        recommendations = None
        if request.include_recommendations:
            try:
                recs = await pattern_detector.generate_recommendations(pattern_result)
                recommendations = [
                    {
                        "action": rec.action,
                        "reason": rec.reason,
                        "priority": rec.priority,
                        "expected_impact": getattr(rec, 'expected_impact', 'medium'),
                    }
                    for rec in recs[:5]
                ]
            except Exception as e:
                logger.warning(f"Failed to generate recommendations: {e}")
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = PatternDetectionResult(
            success=True,
            total_memories_analyzed=len(memories),
            patterns_found=len(pattern_result.pattern_clusters),
            pattern_clusters=pattern_clusters,
            knowledge_gaps=knowledge_gaps,
            insights=insights,
            recommendations=recommendations,
            processing_time_seconds=processing_time
        )
        
        logger.info(
            "Pattern detection completed",
            memories_analyzed=len(memories),
            patterns_found=len(pattern_result.pattern_clusters),
            gaps_found=len(pattern_result.knowledge_gaps),
            processing_time=processing_time
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pattern detection operation failed: {str(e)}"
        )


@router.post("/analyze-temporal", response_model=TemporalAnalysisResult)
async def analyze_temporal_evolution(
    request: TemporalAnalysisRequest,
    components: Dict[str, Any] = Depends(get_synthesis_components)
) -> TemporalAnalysisResult:
    """
    Analyze how memories and concepts evolve over time.
    
    This endpoint tracks the evolution of concepts and learning patterns
    across time periods to identify trends and predict future developments.
    """
    start_time = datetime.utcnow()
    
    try:
        temporal_analyzer = components["temporal_analysis"]
        memory_manager = components["memory_manager"]
        
        # Search for memories in time window
        query = request.concept if request.concept else "*"
        search_request = MemorySearchRequest(
            query=query,
            agent_id=request.agent_id,
            top_k=1000,
        )
        
        search_results = await memory_manager.search_memory(search_request)
        
        if not search_results.results:
            return TemporalAnalysisResult(
                success=True,
                total_memories_analyzed=0,
                time_window_days=request.time_window_days,
                concept_evolutions=[],
                learning_progressions=[],
                temporal_insights=[],
                processing_time_seconds=0.0
            )
        
        # Convert to Memory objects and filter by time window
        cutoff_date = datetime.utcnow() - timedelta(days=request.time_window_days)
        memories = []
        
        for result in search_results.results:
            try:
                created_at = datetime.fromisoformat(result.metadata.get("created_at", datetime.utcnow().isoformat()))
                if created_at >= cutoff_date:
                    memory = Memory(
                        id=result.id,
                        content=result.content,
                        agent_id=result.metadata.get("agent_id", "unknown"),
                        created_at=created_at,
                        metadata=result.metadata
                    )
                    memories.append(memory)
            except Exception as e:
                logger.warning(f"Failed to parse memory date: {e}")
                continue
        
        if not memories:
            return TemporalAnalysisResult(
                success=True,
                total_memories_analyzed=0,
                time_window_days=request.time_window_days,
                concept_evolutions=[],
                learning_progressions=[],
                temporal_insights=[],
                processing_time_seconds=0.0
            )
        
        logger.info(f"Starting temporal analysis of {len(memories)} memories over {request.time_window_days} days")
        
        # Analyze temporal evolution
        evolution_result = await temporal_analyzer.analyze_evolution(
            memories=memories,
            time_window_days=request.time_window_days
        )
        
        # Process concept evolutions
        concept_evolutions = []
        for evolution in evolution_result.concept_evolutions[:5]:
            concept_evolutions.append({
                "concept": evolution.concept,
                "evolution_type": evolution.evolution_type.value,
                "confidence": evolution.confidence,
                "start_understanding": evolution.start_understanding[:100] + "...",
                "current_understanding": evolution.current_understanding[:100] + "...",
                "key_transition_count": len(evolution.key_transitions),
                "trend_direction": getattr(evolution, 'trend_direction', 'unknown'),
            })
        
        # Process learning progressions
        learning_progressions = []
        for progression in evolution_result.learning_progressions[:5]:
            learning_progressions.append({
                "topic": progression.topic,
                "mastery_level": progression.mastery_level,
                "learning_velocity": progression.learning_velocity,
                "milestone_count": len(progression.milestones),
                "next_concepts": progression.next_concepts[:3],
                "confidence": getattr(progression, 'confidence', 0.8),
            })
        
        # Process temporal insights
        temporal_insights = []
        for insight in evolution_result.temporal_insights[:5]:
            temporal_insights.append({
                "type": insight.insight_type,
                "description": insight.description,
                "confidence": insight.confidence,
                "time_period": getattr(insight, 'time_period', None),
                "supporting_data": getattr(insight, 'supporting_data', {})
            })
        
        # Generate predictions if requested
        future_predictions = None
        if request.include_predictions and hasattr(evolution_result, 'future_predictions'):
            future_predictions = [
                {
                    "concept": pred.concept,
                    "predicted_evolution": pred.predicted_evolution,
                    "confidence": pred.confidence,
                    "timeframe_days": pred.timeframe_days,
                    "prediction_basis": getattr(pred, 'prediction_basis', [])[:3],
                }
                for pred in evolution_result.future_predictions[:3]
            ]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = TemporalAnalysisResult(
            success=True,
            total_memories_analyzed=len(memories),
            time_window_days=request.time_window_days,
            concept_evolutions=concept_evolutions,
            learning_progressions=learning_progressions,
            temporal_insights=temporal_insights,
            future_predictions=future_predictions,
            processing_time_seconds=processing_time
        )
        
        logger.info(
            "Temporal analysis completed",
            memories_analyzed=len(memories),
            evolutions_found=len(evolution_result.concept_evolutions),
            progressions_found=len(evolution_result.learning_progressions),
            processing_time=processing_time
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Temporal analysis operation failed: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def synthesis_health_check(
    components: Dict[str, Any] = Depends(get_synthesis_components)
) -> Dict[str, Any]:
    """Check health of synthesis components."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check each component
        for name, component in components.items():
            if name == "memory_manager":
                continue  # Skip memory manager, checked separately
            
            try:
                if hasattr(component, 'health_check'):
                    component_health = await component.health_check()
                else:
                    component_health = {"status": "healthy", "initialized": component is not None}
                
                health_status["components"][name] = component_health
                
            except Exception as e:
                health_status["components"][name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }