"""
Predictive Memory Management API endpoints.

Provides advanced prediction capabilities including usage pattern analysis,
auto-archiving, predictive preloading, and memory lifecycle optimization.
All processing is performed locally with zero external API calls.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ...core.prediction.usage_analyzer import UsageAnalyzer, AnalysisConfig, UsagePattern
from ...core.prediction.auto_archiver import AutoArchiver, ArchivingPolicy, ArchivingResult
from ...core.prediction.preloader import PredictivePreloader, PreloadingConfig, PreloadingResult
from ...core.prediction.lifecycle import LifecycleOptimizer, MemoryStage, LifecycleAnalysisResult
from ...core.memory.manager import MemoryManager
from ...core.cache.redis_cache import RedisCache
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class UsageAnalysisRequest(BaseModel):
    """Request for usage pattern analysis."""
    
    user_id: str = Field(..., description="User identifier")
    analysis_period_days: int = Field(30, ge=1, le=365, description="Analysis period in days")
    include_patterns: bool = Field(True, description="Include detailed pattern analysis")
    pattern_types: Optional[List[str]] = Field(None, description="Specific pattern types to analyze")


class UsageAnalysisResponse(BaseModel):
    """Response for usage pattern analysis."""
    
    user_id: str = Field(..., description="User identifier")
    analysis_period_days: int = Field(..., description="Analysis period used")
    patterns: List[Dict[str, Any]] = Field(..., description="Identified usage patterns")
    recommendations: List[Dict[str, Any]] = Field(..., description="Optimization recommendations")
    metrics: Dict[str, Any] = Field(..., description="Analysis metrics and statistics")
    timestamp: datetime = Field(..., description="Analysis timestamp")


class ArchivingRequest(BaseModel):
    """Request for auto-archiving operation."""
    
    user_id: Optional[str] = Field(None, description="User identifier (optional for global)")
    policy_name: str = Field("default", description="Archiving policy to use")
    dry_run: bool = Field(True, description="Whether to perform a dry run")
    max_memories: int = Field(1000, ge=1, le=10000, description="Maximum memories to process")
    force_archive: bool = Field(False, description="Force archive even if uncertain")


class ArchivingResponse(BaseModel):
    """Response for auto-archiving operation."""
    
    operation_id: str = Field(..., description="Operation identifier")
    dry_run: bool = Field(..., description="Whether this was a dry run")
    memories_processed: int = Field(..., description="Number of memories processed")
    memories_archived: int = Field(..., description="Number of memories archived")
    memories_restored: int = Field(..., description="Number of memories restored")
    space_saved_mb: float = Field(..., description="Estimated space saved in MB")
    recommendations: List[Dict[str, Any]] = Field(..., description="Additional recommendations")
    duration_ms: float = Field(..., description="Operation duration in milliseconds")


class PreloadingRequest(BaseModel):
    """Request for predictive preloading."""
    
    user_id: str = Field(..., description="User identifier")
    preload_strategy: str = Field("adaptive", description="Preloading strategy")
    max_preload_count: int = Field(50, ge=1, le=500, description="Maximum items to preload")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence for preloading")
    include_context: bool = Field(True, description="Include contextual information")


class PreloadingResponse(BaseModel):
    """Response for predictive preloading."""
    
    user_id: str = Field(..., description="User identifier")
    preloaded_memories: List[Dict[str, Any]] = Field(..., description="Preloaded memory information")
    preload_score: float = Field(..., description="Overall preload quality score")
    cache_hit_prediction: float = Field(..., description="Predicted cache hit rate improvement")
    strategy_used: str = Field(..., description="Preloading strategy used")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")


class LifecycleAnalysisRequest(BaseModel):
    """Request for memory lifecycle analysis."""
    
    user_id: Optional[str] = Field(None, description="User identifier (optional for global)")
    memory_ids: Optional[List[str]] = Field(None, description="Specific memory IDs to analyze")
    include_predictions: bool = Field(True, description="Include transition predictions")
    report_format: str = Field("json", description="Report format (json, summary, csv)")
    analysis_depth: str = Field("standard", description="Analysis depth (quick, standard, comprehensive)")


class LifecycleAnalysisResponse(BaseModel):
    """Response for memory lifecycle analysis."""
    
    analysis_id: str = Field(..., description="Analysis identifier")
    user_id: Optional[str] = Field(..., description="User identifier")
    memories_analyzed: int = Field(..., description="Number of memories analyzed")
    stage_distribution: Dict[str, int] = Field(..., description="Distribution across lifecycle stages")
    transition_matrix: Dict[str, Dict[str, float]] = Field(..., description="Stage transition probabilities")
    recommendations: List[Dict[str, Any]] = Field(..., description="Lifecycle optimization recommendations")
    predictions: Optional[Dict[str, Any]] = Field(None, description="Transition predictions")
    report_url: Optional[str] = Field(None, description="URL for detailed report download")


class PredictionMetricsResponse(BaseModel):
    """Response for prediction system metrics."""
    
    timestamp: datetime = Field(..., description="Metrics timestamp")
    usage_analyzer: Dict[str, Any] = Field(..., description="Usage analyzer metrics")
    auto_archiver: Dict[str, Any] = Field(..., description="Auto-archiver metrics")
    preloader: Dict[str, Any] = Field(..., description="Preloader metrics")
    lifecycle_optimizer: Dict[str, Any] = Field(..., description="Lifecycle optimizer metrics")
    overall_performance: Dict[str, float] = Field(..., description="Overall system performance")


# Dependencies
async def get_usage_analyzer() -> UsageAnalyzer:
    """Get usage analyzer instance."""
    try:
        memory_manager = get_provider(ProviderType.MEMORY_MANAGER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return UsageAnalyzer(memory_manager=memory_manager, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get usage analyzer: {e}")
        raise HTTPException(status_code=500, detail="Usage analyzer unavailable")


async def get_auto_archiver() -> AutoArchiver:
    """Get auto-archiver instance."""
    try:
        memory_manager = get_provider(ProviderType.MEMORY_MANAGER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return AutoArchiver(memory_manager=memory_manager, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get auto-archiver: {e}")
        raise HTTPException(status_code=500, detail="Auto-archiver unavailable")


async def get_preloader() -> PredictivePreloader:
    """Get predictive preloader instance."""
    try:
        memory_manager = get_provider(ProviderType.MEMORY_MANAGER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return PredictivePreloader(memory_manager=memory_manager, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get preloader: {e}")
        raise HTTPException(status_code=500, detail="Preloader unavailable")


async def get_lifecycle_optimizer() -> LifecycleOptimizer:
    """Get lifecycle optimizer instance."""
    try:
        memory_manager = get_provider(ProviderType.MEMORY_MANAGER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return LifecycleOptimizer(memory_manager=memory_manager, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get lifecycle optimizer: {e}")
        raise HTTPException(status_code=500, detail="Lifecycle optimizer unavailable")


# API Endpoints

@router.post("/analyze-usage", response_model=UsageAnalysisResponse)
async def analyze_usage_patterns(
    request: UsageAnalysisRequest,
    analyzer: UsageAnalyzer = Depends(get_usage_analyzer)
):
    """
    Analyze user memory usage patterns and provide optimization recommendations.
    
    Performs comprehensive analysis of access patterns, temporal trends,
    and user behavior to identify optimization opportunities.
    """
    try:
        start_time = datetime.utcnow()
        
        # Configure analysis
        config = AnalysisConfig(
            analysis_period_days=request.analysis_period_days,
            pattern_types=request.pattern_types or ["temporal", "access", "search", "session"],
            include_recommendations=True
        )
        
        # Perform analysis
        analysis_result = await analyzer.analyze_user_patterns(
            user_id=request.user_id,
            config=config
        )
        
        # Extract patterns and recommendations
        patterns = []
        for pattern in analysis_result.patterns:
            patterns.append({
                "type": pattern.pattern_type.value,
                "description": pattern.description,
                "confidence": pattern.confidence,
                "impact": pattern.impact_score,
                "metadata": pattern.metadata
            })
        
        recommendations = []
        for rec in analysis_result.recommendations:
            recommendations.append({
                "type": rec.recommendation_type.value,
                "title": rec.title,
                "description": rec.description,
                "priority": rec.priority.value,
                "expected_impact": rec.expected_impact,
                "implementation": rec.implementation_steps
            })
        
        # Compile metrics
        metrics = {
            "analysis_duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            "patterns_found": len(patterns),
            "recommendations_generated": len(recommendations),
            "confidence_scores": [p["confidence"] for p in patterns],
            "coverage_percentage": analysis_result.coverage_percentage,
            "data_quality_score": analysis_result.data_quality_score
        }
        
        return UsageAnalysisResponse(
            user_id=request.user_id,
            analysis_period_days=request.analysis_period_days,
            patterns=patterns,
            recommendations=recommendations,
            metrics=metrics,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Usage pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-archive", response_model=ArchivingResponse)
async def perform_auto_archiving(
    request: ArchivingRequest,
    background_tasks: BackgroundTasks,
    archiver: AutoArchiver = Depends(get_auto_archiver)
):
    """
    Perform intelligent auto-archiving of memories based on usage patterns.
    
    Uses ML-driven analysis to identify memories for archiving while
    ensuring important memories remain accessible.
    """
    try:
        start_time = datetime.utcnow()
        operation_id = f"archive_{int(start_time.timestamp())}"
        
        # Get or create archiving policy
        policy = await archiver.get_policy(request.policy_name)
        if not policy:
            # Create default policy if not found
            policy = ArchivingPolicy(
                name=request.policy_name,
                min_age_days=30,
                max_access_frequency=0.1,
                importance_threshold=0.3,
                consider_relationships=True,
                auto_restore_threshold=0.8
            )
        
        # Perform archiving operation
        archiving_result = await archiver.archive_memories(
            user_id=request.user_id,
            policy=policy,
            max_memories=request.max_memories,
            dry_run=request.dry_run,
            force=request.force_archive
        )
        
        # Schedule background optimization if not dry run
        if not request.dry_run:
            background_tasks.add_task(
                archiver.optimize_storage,
                user_id=request.user_id
            )
        
        # Calculate metrics
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ArchivingResponse(
            operation_id=operation_id,
            dry_run=request.dry_run,
            memories_processed=archiving_result.memories_processed,
            memories_archived=archiving_result.memories_archived,
            memories_restored=archiving_result.memories_restored,
            space_saved_mb=archiving_result.space_saved_bytes / (1024 * 1024),
            recommendations=archiving_result.recommendations,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Auto-archiving failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preload", response_model=PreloadingResponse)
async def perform_predictive_preloading(
    request: PreloadingRequest,
    preloader: PredictivePreloader = Depends(get_preloader)
):
    """
    Perform predictive preloading of memories based on usage patterns.
    
    Uses ML models to predict which memories are likely to be accessed
    and preloads them to improve query performance.
    """
    try:
        start_time = datetime.utcnow()
        
        # Configure preloading
        config = PreloadingConfig(
            strategy=request.preload_strategy,
            max_preload_count=request.max_preload_count,
            confidence_threshold=request.confidence_threshold,
            include_context=request.include_context
        )
        
        # Perform preloading
        preloading_result = await preloader.preload_for_user(
            user_id=request.user_id,
            config=config
        )
        
        # Format preloaded memories
        preloaded_memories = []
        for memory_info in preloading_result.preloaded_memories:
            preloaded_memories.append({
                "memory_id": memory_info["memory_id"],
                "prediction_confidence": memory_info["confidence"],
                "predicted_access_time": memory_info.get("predicted_access_time"),
                "preload_reason": memory_info.get("reason", "pattern_based"),
                "context": memory_info.get("context", {}) if request.include_context else {}
            })
        
        # Calculate execution time
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return PreloadingResponse(
            user_id=request.user_id,
            preloaded_memories=preloaded_memories,
            preload_score=preloading_result.quality_score,
            cache_hit_prediction=preloading_result.predicted_cache_improvement,
            strategy_used=preloading_result.strategy_used,
            execution_time_ms=execution_time_ms
        )
        
    except Exception as e:
        logger.error(f"Predictive preloading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lifecycle-analysis", response_model=LifecycleAnalysisResponse)
async def perform_lifecycle_analysis(
    request: LifecycleAnalysisRequest,
    background_tasks: BackgroundTasks,
    optimizer: LifecycleOptimizer = Depends(get_lifecycle_optimizer)
):
    """
    Perform comprehensive memory lifecycle analysis.
    
    Analyzes memory transitions through lifecycle stages and provides
    optimization recommendations and transition predictions.
    """
    try:
        start_time = datetime.utcnow()
        analysis_id = f"lifecycle_{int(start_time.timestamp())}"
        
        # Determine scope of analysis
        if request.memory_ids:
            # Analyze specific memories
            if request.analysis_depth == "comprehensive":
                results = await optimizer.batch_lifecycle_analysis(
                    memory_ids=request.memory_ids,
                    batch_size=50
                )
            else:
                # Quick analysis for specific memories
                results = {}
                for memory_id in request.memory_ids[:100]:  # Limit for quick analysis
                    try:
                        result = await optimizer.analyze_memory_lifecycle(memory_id)
                        results[memory_id] = result
                    except Exception as e:
                        logger.warning(f"Failed to analyze memory {memory_id}: {e}")
                        continue
        
        elif request.user_id:
            # Analyze all user memories
            if request.report_format in ["summary", "csv"]:
                # Generate comprehensive report
                report = await optimizer.export_lifecycle_report(
                    user_id=request.user_id,
                    format=request.report_format,
                    include_predictions=request.include_predictions
                )
                
                # Store report for download (in background)
                if request.report_format != "summary":
                    background_tasks.add_task(
                        _store_lifecycle_report,
                        analysis_id,
                        report
                    )
                
                return LifecycleAnalysisResponse(
                    analysis_id=analysis_id,
                    user_id=request.user_id,
                    memories_analyzed=report.get("summary", {}).get("analyzed_memories", 0),
                    stage_distribution=report.get("summary", {}).get("stage_distribution", {}),
                    transition_matrix=report.get("stage_analytics", {}).get("transition_matrix", {}),
                    recommendations=report.get("recommendations", []),
                    predictions=report.get("predictions") if request.include_predictions else None,
                    report_url=f"/api/v1/prediction/reports/{analysis_id}" if request.report_format != "summary" else None
                )
            else:
                # Standard analysis for user
                user_memories = await optimizer._get_user_memories(request.user_id)
                memory_ids = [m["id"] for m in user_memories[:500]]  # Limit for performance
                results = await optimizer.batch_lifecycle_analysis(memory_ids)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either user_id or memory_ids must be provided"
            )
        
        # Process results for standard analysis
        if 'results' in locals():
            # Calculate stage distribution
            stage_distribution = {}
            for result in results.values():
                stage = result.current_stage.value
                stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
            
            # Build transition matrix
            transition_matrix = optimizer._build_transition_matrix(results.values())
            
            # Generate recommendations
            recommendations = []
            for memory_id, result in list(results.items())[:10]:  # Top 10 for recommendations
                if result.optimization_recommendations:
                    for rec in result.optimization_recommendations[:2]:  # Top 2 per memory
                        recommendations.append({
                            "memory_id": memory_id,
                            "type": rec.recommendation_type.value,
                            "description": rec.description,
                            "priority": rec.priority.value,
                            "expected_impact": rec.expected_impact
                        })
            
            # Generate predictions if requested
            predictions = None
            if request.include_predictions:
                predictions = {}
                for memory_id in list(results.keys())[:20]:  # Limit predictions for performance
                    try:
                        prediction = await optimizer.predict_next_transition(memory_id)
                        predictions[memory_id] = {
                            "next_stage": prediction.predicted_stage.value,
                            "confidence": prediction.confidence,
                            "days_to_transition": prediction.time_to_transition_days
                        }
                    except Exception as e:
                        logger.warning(f"Prediction failed for {memory_id}: {e}")
                        continue
            
            return LifecycleAnalysisResponse(
                analysis_id=analysis_id,
                user_id=request.user_id,
                memories_analyzed=len(results),
                stage_distribution=stage_distribution,
                transition_matrix=transition_matrix,
                recommendations=recommendations,
                predictions=predictions
            )
        
    except Exception as e:
        logger.error(f"Lifecycle analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=PredictionMetricsResponse)
async def get_prediction_metrics(
    analyzer: UsageAnalyzer = Depends(get_usage_analyzer),
    archiver: AutoArchiver = Depends(get_auto_archiver),
    preloader: PredictivePreloader = Depends(get_preloader),
    optimizer: LifecycleOptimizer = Depends(get_lifecycle_optimizer)
):
    """
    Get comprehensive metrics for all prediction system components.
    
    Returns performance statistics, accuracy metrics, and system health
    information for the entire prediction subsystem.
    """
    try:
        # Gather metrics from all components concurrently
        metrics_tasks = [
            analyzer.get_performance_metrics(),
            archiver.get_performance_metrics(),
            preloader.get_performance_metrics(),
            optimizer.get_performance_metrics()
        ]
        
        usage_metrics, archiver_metrics, preloader_metrics, lifecycle_metrics = await asyncio.gather(
            *metrics_tasks, return_exceptions=True
        )
        
        # Handle any exceptions in metrics gathering
        if isinstance(usage_metrics, Exception):
            usage_metrics = {"error": str(usage_metrics)}
        if isinstance(archiver_metrics, Exception):
            archiver_metrics = {"error": str(archiver_metrics)}
        if isinstance(preloader_metrics, Exception):
            preloader_metrics = {"error": str(preloader_metrics)}
        if isinstance(lifecycle_metrics, Exception):
            lifecycle_metrics = {"error": str(lifecycle_metrics)}
        
        # Calculate overall performance scores
        overall_performance = {
            "accuracy_score": 0.0,
            "latency_score": 0.0,
            "cache_efficiency": 0.0,
            "prediction_confidence": 0.0,
            "system_health": 1.0
        }
        
        # Extract accuracy scores
        accuracy_scores = []
        if "accuracy" in usage_metrics:
            accuracy_scores.append(usage_metrics["accuracy"])
        if "prediction_accuracy" in preloader_metrics:
            accuracy_scores.append(preloader_metrics["prediction_accuracy"])
        
        if accuracy_scores:
            overall_performance["accuracy_score"] = sum(accuracy_scores) / len(accuracy_scores)
        
        # Extract latency information
        latency_scores = []
        for metrics in [usage_metrics, archiver_metrics, preloader_metrics, lifecycle_metrics]:
            if "avg_processing_time_ms" in metrics:
                # Convert to score (lower latency = higher score)
                latency_ms = metrics["avg_processing_time_ms"]
                latency_score = max(0.0, 1.0 - (latency_ms / 1000.0))  # Normalize to 0-1
                latency_scores.append(latency_score)
        
        if latency_scores:
            overall_performance["latency_score"] = sum(latency_scores) / len(latency_scores)
        
        # Extract cache efficiency
        if "cache_hit_rate" in preloader_metrics:
            overall_performance["cache_efficiency"] = preloader_metrics["cache_hit_rate"]
        
        # Calculate system health
        error_count = sum(1 for m in [usage_metrics, archiver_metrics, preloader_metrics, lifecycle_metrics] if "error" in m)
        overall_performance["system_health"] = max(0.0, 1.0 - (error_count / 4.0))
        
        return PredictionMetricsResponse(
            timestamp=datetime.utcnow(),
            usage_analyzer=usage_metrics,
            auto_archiver=archiver_metrics,
            preloader=preloader_metrics,
            lifecycle_optimizer=lifecycle_metrics,
            overall_performance=overall_performance
        )
        
    except Exception as e:
        logger.error(f"Failed to get prediction metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{analysis_id}")
async def download_lifecycle_report(analysis_id: str):
    """
    Download a previously generated lifecycle analysis report.
    
    Returns the full report in the requested format (JSON, CSV, etc.)
    """
    try:
        # This would retrieve the stored report
        # For now, return a placeholder
        return {
            "analysis_id": analysis_id,
            "status": "Report download not yet implemented",
            "note": "Reports are generated but download functionality is pending"
        }
        
    except Exception as e:
        logger.error(f"Failed to download report {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _store_lifecycle_report(analysis_id: str, report: Dict[str, Any]):
    """Store lifecycle report for later download."""
    try:
        # This would store the report in a file system or database
        # For now, just log the storage attempt
        logger.info(
            "Lifecycle report stored",
            analysis_id=analysis_id,
            report_size=len(str(report))
        )
    except Exception as e:
        logger.error(f"Failed to store lifecycle report {analysis_id}: {e}")


# Export router
__all__ = ["router"]