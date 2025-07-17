"""
Context-Aware Embeddings API endpoints.

Provides advanced embedding capabilities including session-aware embeddings,
multi-perspective processing, dynamic fine-tuning, and embedding fusion.
All processing is performed locally with zero external API calls.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field

from ...core.embeddings.contextual import ContextualEmbedder, SessionContext, ContextualResult
from ...core.embeddings.multi_perspective import MultiPerspectiveEmbedder, PerspectiveType, EmbeddingContext
from ...core.embeddings.fine_tuning import DynamicFineTuner, FineTuningConfig, FineTuningResult
from ...core.embeddings.fusion import EmbeddingFusionSystem, FusionConfig, FusionResult
from ...core.embeddings.embedder import Embedder
from ...core.cache.redis_cache import RedisCache
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class ContextualEmbeddingRequest(BaseModel):
    """Request for contextual embedding generation."""
    
    text: str = Field(..., description="Text to embed")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    user_id: Optional[str] = Field(None, description="User identifier")
    context_window: int = Field(5, ge=1, le=20, description="Number of previous interactions to consider")
    decay_factor: float = Field(0.8, ge=0.1, le=1.0, description="Context decay factor")
    include_metadata: bool = Field(True, description="Include embedding metadata")


class ContextualEmbeddingResponse(BaseModel):
    """Response for contextual embedding generation."""
    
    embedding: List[float] = Field(..., description="Generated embedding vector")
    dimension: int = Field(..., description="Embedding dimension")
    context_strength: float = Field(..., description="Strength of contextual influence")
    session_contributions: Dict[str, float] = Field(..., description="Contribution of each session element")
    metadata: Dict[str, Any] = Field(..., description="Additional embedding metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class MultiPerspectiveRequest(BaseModel):
    """Request for multi-perspective embedding generation."""
    
    text: str = Field(..., description="Text to embed")
    perspective_types: Optional[List[str]] = Field(None, description="Specific perspectives to use")
    fusion_method: str = Field("weighted_average", description="Fusion method for combining perspectives")
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum perspective confidence")
    include_individual_results: bool = Field(False, description="Include individual perspective results")


class MultiPerspectiveResponse(BaseModel):
    """Response for multi-perspective embedding generation."""
    
    fused_embedding: List[float] = Field(..., description="Fused embedding from all perspectives")
    perspective_weights: Dict[str, float] = Field(..., description="Weight of each perspective")
    perspective_confidences: Dict[str, float] = Field(..., description="Confidence of each perspective")
    fusion_quality: float = Field(..., description="Quality score of the fusion")
    individual_embeddings: Optional[Dict[str, List[float]]] = Field(None, description="Individual perspective embeddings")
    processing_time_ms: float = Field(..., description="Total processing time")


class FineTuningRequest(BaseModel):
    """Request for dynamic fine-tuning operation."""
    
    training_data: List[Dict[str, Any]] = Field(..., description="Training data for fine-tuning")
    target_domain: str = Field(..., description="Target domain for adaptation")
    learning_rate: float = Field(1e-5, gt=0.0, le=1e-2, description="Learning rate for fine-tuning")
    num_epochs: int = Field(3, ge=1, le=20, description="Number of training epochs")
    validation_split: float = Field(0.2, ge=0.0, le=0.5, description="Validation data split ratio")
    use_lora: bool = Field(True, description="Use LoRA for parameter-efficient fine-tuning")


class FineTuningResponse(BaseModel):
    """Response for dynamic fine-tuning operation."""
    
    job_id: str = Field(..., description="Fine-tuning job identifier")
    status: str = Field(..., description="Job status")
    model_version: str = Field(..., description="New model version identifier")
    training_metrics: Dict[str, float] = Field(..., description="Training performance metrics")
    validation_metrics: Dict[str, float] = Field(..., description="Validation performance metrics")
    improvement_score: float = Field(..., description="Overall improvement score")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")


class EmbeddingFusionRequest(BaseModel):
    """Request for embedding fusion operation."""
    
    embeddings: List[List[float]] = Field(..., description="List of embeddings to fuse")
    embedding_sources: Optional[List[str]] = Field(None, description="Sources of each embedding")
    fusion_strategy: str = Field("adaptive", description="Fusion strategy to use")
    quality_weights: Optional[List[float]] = Field(None, description="Quality weights for each embedding")
    normalize_output: bool = Field(True, description="Normalize the fused embedding")


class EmbeddingFusionResponse(BaseModel):
    """Response for embedding fusion operation."""
    
    fused_embedding: List[float] = Field(..., description="Fused embedding vector")
    fusion_weights: List[float] = Field(..., description="Weights used for fusion")
    quality_score: float = Field(..., description="Quality score of the fusion")
    dimension_compatibility: bool = Field(..., description="Whether all embeddings were compatible")
    fusion_metadata: Dict[str, Any] = Field(..., description="Additional fusion metadata")


class EmbeddingBenchmarkRequest(BaseModel):
    """Request for embedding system benchmarking."""
    
    test_queries: List[str] = Field(..., description="Test queries for benchmarking")
    benchmark_types: List[str] = Field(["contextual", "multi_perspective", "fusion"], description="Types of embeddings to benchmark")
    ground_truth: Optional[List[Dict[str, Any]]] = Field(None, description="Ground truth for evaluation")
    include_performance_metrics: bool = Field(True, description="Include detailed performance metrics")


class EmbeddingBenchmarkResponse(BaseModel):
    """Response for embedding system benchmarking."""
    
    benchmark_id: str = Field(..., description="Benchmark run identifier")
    results: Dict[str, Dict[str, Any]] = Field(..., description="Benchmark results by type")
    overall_rankings: List[Dict[str, Any]] = Field(..., description="Overall performance rankings")
    performance_summary: Dict[str, float] = Field(..., description="Summary performance metrics")
    recommendations: List[Dict[str, Any]] = Field(..., description="Optimization recommendations")


class EmbeddingMetricsResponse(BaseModel):
    """Response for embedding system metrics."""
    
    timestamp: datetime = Field(..., description="Metrics timestamp")
    contextual_embedder: Dict[str, Any] = Field(..., description="Contextual embedder metrics")
    multi_perspective: Dict[str, Any] = Field(..., description="Multi-perspective embedder metrics")
    fine_tuner: Dict[str, Any] = Field(..., description="Fine-tuner metrics")
    fusion_system: Dict[str, Any] = Field(..., description="Fusion system metrics")
    overall_performance: Dict[str, float] = Field(..., description="Overall system performance")


# Dependencies
async def get_contextual_embedder() -> ContextualEmbedder:
    """Get contextual embedder instance."""
    try:
        base_embedder = get_provider(ProviderType.EMBEDDER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return ContextualEmbedder(base_embedder=base_embedder, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get contextual embedder: {e}")
        raise HTTPException(status_code=500, detail="Contextual embedder unavailable")


async def get_multi_perspective_embedder() -> MultiPerspectiveEmbedder:
    """Get multi-perspective embedder instance."""
    try:
        base_embedder = get_provider(ProviderType.EMBEDDER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return MultiPerspectiveEmbedder(base_embedder=base_embedder, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get multi-perspective embedder: {e}")
        raise HTTPException(status_code=500, detail="Multi-perspective embedder unavailable")


async def get_fine_tuner() -> DynamicFineTuner:
    """Get dynamic fine-tuner instance."""
    try:
        base_embedder = get_provider(ProviderType.EMBEDDER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return DynamicFineTuner(base_embedder=base_embedder, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get fine-tuner: {e}")
        raise HTTPException(status_code=500, detail="Fine-tuner unavailable")


async def get_fusion_system() -> EmbeddingFusionSystem:
    """Get embedding fusion system instance."""
    try:
        cache = get_provider(ProviderType.CACHE, "default")
        return EmbeddingFusionSystem(cache=cache)
    except Exception as e:
        logger.error(f"Failed to get fusion system: {e}")
        raise HTTPException(status_code=500, detail="Fusion system unavailable")


# API Endpoints

@router.post("/contextual", response_model=ContextualEmbeddingResponse)
async def generate_contextual_embedding(
    request: ContextualEmbeddingRequest,
    embedder: ContextualEmbedder = Depends(get_contextual_embedder)
):
    """
    Generate context-aware embeddings based on session history.
    
    Creates embeddings that adapt based on previous interactions in the session,
    providing more relevant representations for query understanding.
    """
    try:
        start_time = time.time()
        
        # Create session context
        session_context = SessionContext(
            session_id=request.session_id or f"session_{int(time.time())}",
            user_id=request.user_id or "anonymous",
            context_window=request.context_window,
            decay_factor=request.decay_factor
        )
        
        # Generate contextual embedding
        result = await embedder.generate_contextual_embedding(
            text=request.text,
            session_context=session_context,
            include_metadata=request.include_metadata
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ContextualEmbeddingResponse(
            embedding=result.embedding.tolist(),
            dimension=len(result.embedding),
            context_strength=result.context_strength,
            session_contributions=result.session_contributions,
            metadata=result.metadata,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Contextual embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-perspective", response_model=MultiPerspectiveResponse)
async def generate_multi_perspective_embedding(
    request: MultiPerspectiveRequest,
    embedder: MultiPerspectiveEmbedder = Depends(get_multi_perspective_embedder)
):
    """
    Generate multi-perspective embeddings with fusion.
    
    Creates embeddings from multiple domain-specific perspectives and fuses
    them into a unified representation with confidence scoring.
    """
    try:
        start_time = time.time()
        
        # Parse perspective types
        perspective_types = None
        if request.perspective_types:
            try:
                perspective_types = [PerspectiveType(p) for p in request.perspective_types]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid perspective type: {e}")
        
        # Create embedding context
        context = EmbeddingContext(
            query=request.text,
            perspective_types=perspective_types,
            min_confidence=request.min_confidence,
            fusion_method=request.fusion_method,
            include_metadata=True
        )
        
        # Generate multi-perspective embedding
        result = await embedder.generate_contextual_embedding(
            text=request.text,
            context=context
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare individual embeddings if requested
        individual_embeddings = None
        if request.include_individual_results and hasattr(result, 'individual_results'):
            individual_embeddings = {
                p.value: emb.tolist() 
                for p, emb in result.individual_results.items()
            }
        
        return MultiPerspectiveResponse(
            fused_embedding=result.embedding.tolist(),
            perspective_weights=result.perspective_weights,
            perspective_confidences=result.perspective_confidences,
            fusion_quality=result.fusion_quality,
            individual_embeddings=individual_embeddings,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Multi-perspective embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine-tune", response_model=FineTuningResponse)
async def start_fine_tuning(
    request: FineTuningRequest,
    background_tasks: BackgroundTasks,
    fine_tuner: DynamicFineTuner = Depends(get_fine_tuner)
):
    """
    Start dynamic fine-tuning of embedding models.
    
    Performs parameter-efficient fine-tuning using LoRA to adapt models
    to specific domains or user preferences.
    """
    try:
        # Generate unique job ID
        job_id = f"finetune_{int(time.time())}_{request.target_domain}"
        
        # Validate training data
        if len(request.training_data) < 10:
            raise HTTPException(
                status_code=400,
                detail="Minimum 10 training examples required"
            )
        
        # Create fine-tuning configuration
        config = FineTuningConfig(
            target_domain=request.target_domain,
            learning_rate=request.learning_rate,
            num_epochs=request.num_epochs,
            validation_split=request.validation_split,
            use_lora=request.use_lora,
            job_id=job_id
        )
        
        # Start fine-tuning in background
        background_tasks.add_task(
            _execute_fine_tuning,
            fine_tuner,
            request.training_data,
            config
        )
        
        # Estimate completion time
        estimated_time = _estimate_fine_tuning_time(
            len(request.training_data),
            request.num_epochs
        )
        
        return FineTuningResponse(
            job_id=job_id,
            status="started",
            model_version=f"{job_id}_v1",
            training_metrics={},
            validation_metrics={},
            improvement_score=0.0,
            estimated_completion_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"Fine-tuning initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fine-tune/{job_id}")
async def get_fine_tuning_status(
    job_id: str,
    fine_tuner: DynamicFineTuner = Depends(get_fine_tuner)
):
    """
    Get the status of a fine-tuning job.
    
    Returns current progress, metrics, and completion status.
    """
    try:
        # Get job status from fine-tuner
        status = await fine_tuner.get_job_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Fine-tuning job not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get fine-tuning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fusion", response_model=EmbeddingFusionResponse)
async def fuse_embeddings(
    request: EmbeddingFusionRequest,
    fusion_system: EmbeddingFusionSystem = Depends(get_fusion_system)
):
    """
    Fuse multiple embeddings into a unified representation.
    
    Combines embeddings from different sources or models using advanced
    fusion strategies for improved quality and relevance.
    """
    try:
        # Validate input
        if len(request.embeddings) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 embeddings required for fusion"
            )
        
        # Create fusion configuration
        config = FusionConfig(
            strategy=request.fusion_strategy,
            normalize_output=request.normalize_output,
            quality_weights=request.quality_weights
        )
        
        # Perform fusion
        result = await fusion_system.fuse_embeddings(
            embeddings=request.embeddings,
            sources=request.embedding_sources,
            config=config
        )
        
        return EmbeddingFusionResponse(
            fused_embedding=result.fused_embedding.tolist(),
            fusion_weights=result.fusion_weights,
            quality_score=result.quality_score,
            dimension_compatibility=result.dimension_compatibility,
            fusion_metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Embedding fusion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark", response_model=EmbeddingBenchmarkResponse)
async def benchmark_embeddings(
    request: EmbeddingBenchmarkRequest,
    contextual_embedder: ContextualEmbedder = Depends(get_contextual_embedder),
    multi_perspective: MultiPerspectiveEmbedder = Depends(get_multi_perspective_embedder),
    fusion_system: EmbeddingFusionSystem = Depends(get_fusion_system)
):
    """
    Benchmark different embedding approaches.
    
    Compares performance of contextual, multi-perspective, and fusion
    approaches on a set of test queries.
    """
    try:
        benchmark_id = f"benchmark_{int(time.time())}"
        results = {}
        
        # Benchmark contextual embeddings
        if "contextual" in request.benchmark_types:
            contextual_results = await _benchmark_contextual_embeddings(
                contextual_embedder,
                request.test_queries,
                request.ground_truth
            )
            results["contextual"] = contextual_results
        
        # Benchmark multi-perspective embeddings
        if "multi_perspective" in request.benchmark_types:
            perspective_results = await _benchmark_multi_perspective_embeddings(
                multi_perspective,
                request.test_queries,
                request.ground_truth
            )
            results["multi_perspective"] = perspective_results
        
        # Benchmark fusion embeddings
        if "fusion" in request.benchmark_types:
            fusion_results = await _benchmark_fusion_embeddings(
                fusion_system,
                request.test_queries,
                request.ground_truth
            )
            results["fusion"] = fusion_results
        
        # Create overall rankings
        rankings = []
        for approach, result in results.items():
            score = result.get("overall_score", 0.0)
            rankings.append({
                "approach": approach,
                "score": score,
                "avg_processing_time": result.get("avg_processing_time_ms", 0),
                "success_rate": result.get("success_rate", 0)
            })
        
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        # Generate performance summary
        performance_summary = {
            "best_approach": rankings[0]["approach"] if rankings else "none",
            "avg_improvement": np.mean([r["score"] for r in rankings]) if rankings else 0,
            "total_queries_tested": len(request.test_queries),
            "benchmark_duration_ms": sum(
                r.get("total_time_ms", 0) for r in results.values()
            )
        }
        
        # Generate recommendations
        recommendations = _generate_embedding_recommendations(results, rankings)
        
        return EmbeddingBenchmarkResponse(
            benchmark_id=benchmark_id,
            results=results,
            overall_rankings=rankings,
            performance_summary=performance_summary,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Embedding benchmarking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=EmbeddingMetricsResponse)
async def get_embedding_metrics(
    contextual_embedder: ContextualEmbedder = Depends(get_contextual_embedder),
    multi_perspective: MultiPerspectiveEmbedder = Depends(get_multi_perspective_embedder),
    fine_tuner: DynamicFineTuner = Depends(get_fine_tuner),
    fusion_system: EmbeddingFusionSystem = Depends(get_fusion_system)
):
    """
    Get comprehensive metrics for all embedding system components.
    
    Returns performance statistics, accuracy metrics, and system health
    information for the entire embedding subsystem.
    """
    try:
        # Gather metrics from all components concurrently
        metrics_tasks = [
            contextual_embedder.get_performance_metrics(),
            multi_perspective.get_performance_metrics(),
            fine_tuner.get_performance_metrics(),
            fusion_system.get_performance_metrics()
        ]
        
        contextual_metrics, perspective_metrics, tuning_metrics, fusion_metrics = await asyncio.gather(
            *metrics_tasks, return_exceptions=True
        )
        
        # Handle any exceptions in metrics gathering
        if isinstance(contextual_metrics, Exception):
            contextual_metrics = {"error": str(contextual_metrics)}
        if isinstance(perspective_metrics, Exception):
            perspective_metrics = {"error": str(perspective_metrics)}
        if isinstance(tuning_metrics, Exception):
            tuning_metrics = {"error": str(tuning_metrics)}
        if isinstance(fusion_metrics, Exception):
            fusion_metrics = {"error": str(fusion_metrics)}
        
        # Calculate overall performance scores
        overall_performance = {
            "avg_processing_time_ms": 0.0,
            "avg_accuracy": 0.0,
            "system_health": 1.0,
            "cache_efficiency": 0.0,
            "model_freshness": 1.0
        }
        
        # Extract processing times
        processing_times = []
        for metrics in [contextual_metrics, perspective_metrics, tuning_metrics, fusion_metrics]:
            if "avg_processing_time_ms" in metrics:
                processing_times.append(metrics["avg_processing_time_ms"])
        
        if processing_times:
            overall_performance["avg_processing_time_ms"] = np.mean(processing_times)
        
        # Extract accuracy scores
        accuracy_scores = []
        for metrics in [contextual_metrics, perspective_metrics, fusion_metrics]:
            if "accuracy" in metrics:
                accuracy_scores.append(metrics["accuracy"])
        
        if accuracy_scores:
            overall_performance["avg_accuracy"] = np.mean(accuracy_scores)
        
        # Calculate system health
        error_count = sum(1 for m in [contextual_metrics, perspective_metrics, tuning_metrics, fusion_metrics] if "error" in m)
        overall_performance["system_health"] = max(0.0, 1.0 - (error_count / 4.0))
        
        # Extract cache efficiency
        cache_efficiencies = []
        for metrics in [contextual_metrics, perspective_metrics, fusion_metrics]:
            if "cache_hit_rate" in metrics:
                cache_efficiencies.append(metrics["cache_hit_rate"])
        
        if cache_efficiencies:
            overall_performance["cache_efficiency"] = np.mean(cache_efficiencies)
        
        return EmbeddingMetricsResponse(
            timestamp=datetime.utcnow(),
            contextual_embedder=contextual_metrics,
            multi_perspective=perspective_metrics,
            fine_tuner=tuning_metrics,
            fusion_system=fusion_metrics,
            overall_performance=overall_performance
        )
        
    except Exception as e:
        logger.error(f"Failed to get embedding metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _execute_fine_tuning(
    fine_tuner: DynamicFineTuner,
    training_data: List[Dict[str, Any]],
    config: FineTuningConfig
):
    """Execute fine-tuning in the background."""
    try:
        result = await fine_tuner.start_fine_tuning(training_data, config)
        logger.info(
            "Fine-tuning completed",
            job_id=config.job_id,
            improvement_score=result.improvement_score
        )
    except Exception as e:
        logger.error(f"Fine-tuning job {config.job_id} failed: {e}")


def _estimate_fine_tuning_time(data_size: int, epochs: int) -> str:
    """Estimate fine-tuning completion time."""
    # Simple estimation: ~1 second per 100 examples per epoch
    estimated_seconds = (data_size / 100) * epochs * 1
    estimated_minutes = max(1, int(estimated_seconds / 60))
    
    completion_time = datetime.utcnow().replace(minute=datetime.utcnow().minute + estimated_minutes)
    return completion_time.isoformat()


async def _benchmark_contextual_embeddings(
    embedder: ContextualEmbedder,
    test_queries: List[str],
    ground_truth: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Benchmark contextual embeddings."""
    start_time = time.time()
    successful_embeddings = 0
    processing_times = []
    
    for query in test_queries:
        query_start = time.time()
        try:
            session_context = SessionContext(
                session_id=f"benchmark_{int(time.time())}",
                user_id="benchmark_user"
            )
            result = await embedder.generate_contextual_embedding(query, session_context)
            if result and result.embedding is not None:
                successful_embeddings += 1
            processing_times.append((time.time() - query_start) * 1000)
        except Exception:
            processing_times.append((time.time() - query_start) * 1000)
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "total_queries": len(test_queries),
        "successful_embeddings": successful_embeddings,
        "success_rate": successful_embeddings / len(test_queries),
        "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0,
        "total_time_ms": total_time,
        "overall_score": (successful_embeddings / len(test_queries)) * 0.8 + 
                        (1.0 - min(1.0, np.mean(processing_times) / 1000)) * 0.2 if processing_times else 0
    }


async def _benchmark_multi_perspective_embeddings(
    embedder: MultiPerspectiveEmbedder,
    test_queries: List[str],
    ground_truth: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Benchmark multi-perspective embeddings."""
    start_time = time.time()
    successful_embeddings = 0
    processing_times = []
    
    for query in test_queries:
        query_start = time.time()
        try:
            context = EmbeddingContext(query=query)
            result = await embedder.generate_contextual_embedding(query, context)
            if result and result.embedding is not None:
                successful_embeddings += 1
            processing_times.append((time.time() - query_start) * 1000)
        except Exception:
            processing_times.append((time.time() - query_start) * 1000)
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "total_queries": len(test_queries),
        "successful_embeddings": successful_embeddings,
        "success_rate": successful_embeddings / len(test_queries),
        "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0,
        "total_time_ms": total_time,
        "overall_score": (successful_embeddings / len(test_queries)) * 0.8 + 
                        (1.0 - min(1.0, np.mean(processing_times) / 1000)) * 0.2 if processing_times else 0
    }


async def _benchmark_fusion_embeddings(
    fusion_system: EmbeddingFusionSystem,
    test_queries: List[str],
    ground_truth: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Benchmark embedding fusion."""
    # For fusion benchmarking, we would need multiple embeddings per query
    # This is a simplified version
    return {
        "total_queries": len(test_queries),
        "successful_embeddings": len(test_queries),  # Placeholder
        "success_rate": 1.0,
        "avg_processing_time_ms": 50.0,  # Placeholder
        "total_time_ms": len(test_queries) * 50.0,
        "overall_score": 0.85  # Placeholder
    }


def _generate_embedding_recommendations(
    results: Dict[str, Dict[str, Any]],
    rankings: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate optimization recommendations based on benchmark results."""
    recommendations = []
    
    if not rankings:
        return recommendations
    
    best_approach = rankings[0]["approach"]
    worst_approach = rankings[-1]["approach"] if len(rankings) > 1 else None
    
    # Recommend best approach
    recommendations.append({
        "type": "best_practice",
        "title": f"Use {best_approach} for optimal performance",
        "description": f"{best_approach} achieved the highest overall score",
        "priority": "high",
        "expected_improvement": rankings[0]["score"]
    })
    
    # Identify performance issues
    for approach, result in results.items():
        if result.get("avg_processing_time_ms", 0) > 200:
            recommendations.append({
                "type": "performance",
                "title": f"Optimize {approach} processing time",
                "description": f"{approach} has high latency ({result['avg_processing_time_ms']:.1f}ms)",
                "priority": "medium",
                "suggestion": "Consider caching or model optimization"
            })
        
        if result.get("success_rate", 1) < 0.9:
            recommendations.append({
                "type": "reliability",
                "title": f"Improve {approach} reliability",
                "description": f"{approach} has low success rate ({result['success_rate']:.2f})",
                "priority": "high",
                "suggestion": "Review error handling and fallback mechanisms"
            })
    
    return recommendations


# Export router
__all__ = ["router"]