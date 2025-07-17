"""
Auto-Scaling & Performance API endpoints.

Provides advanced performance optimization and auto-scaling capabilities including
query optimization, auto-scaling controls, and performance monitoring.
All processing is performed locally with zero external API calls.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ...infrastructure.scaling.auto_scaler import AutoScalingEngine, ScalingStrategy, MetricType
from ...core.optimization.query_optimizer import QueryOptimizer, QueryOptimization, IndexRecommendation
from ...core.cache.redis_cache import RedisCache
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class QueryAnalysisRequest(BaseModel):
    """Request for query analysis."""
    
    query: str = Field(..., description="SQL query to analyze")
    include_optimization: bool = Field(True, description="Include optimization suggestions")
    optimization_level: str = Field("standard", description="Optimization level: basic, standard, aggressive")


class QueryAnalysisResponse(BaseModel):
    """Response for query analysis."""
    
    query: str = Field(..., description="Original query")
    query_type: str = Field(..., description="Type of SQL query")
    complexity_score: float = Field(..., description="Query complexity score")
    estimated_cost: float = Field(..., description="Estimated execution cost")
    tables_involved: List[str] = Field(..., description="Tables involved in query")
    optimization_suggestions: Optional[Dict[str, Any]] = Field(None, description="Optimization suggestions")
    index_recommendations: List[Dict[str, Any]] = Field(..., description="Index recommendations")
    analysis_time_ms: float = Field(..., description="Analysis time in milliseconds")


class QueryOptimizationRequest(BaseModel):
    """Request for query optimization."""
    
    query: str = Field(..., description="SQL query to optimize")
    optimization_level: str = Field("standard", description="Optimization level: basic, standard, aggressive")
    apply_recommendations: bool = Field(True, description="Apply optimization recommendations")


class QueryOptimizationResponse(BaseModel):
    """Response for query optimization."""
    
    original_query: str = Field(..., description="Original query")
    optimized_query: str = Field(..., description="Optimized query")
    optimization_strategies: List[str] = Field(..., description="Applied optimization strategies")
    estimated_improvement: float = Field(..., description="Estimated performance improvement")
    index_recommendations: List[Dict[str, Any]] = Field(..., description="Index recommendations")
    warnings: List[str] = Field(..., description="Optimization warnings")
    optimization_time_ms: float = Field(..., description="Optimization time in milliseconds")


class AutoScalingConfigRequest(BaseModel):
    """Request for auto-scaling configuration."""
    
    strategy: str = Field("hybrid", description="Scaling strategy: reactive, predictive, hybrid, conservative, aggressive")
    min_instances: int = Field(1, ge=1, le=100, description="Minimum number of instances")
    max_instances: int = Field(10, ge=1, le=100, description="Maximum number of instances")
    target_cpu_utilization: float = Field(70.0, ge=10.0, le=95.0, description="Target CPU utilization percentage")
    target_memory_utilization: float = Field(80.0, ge=10.0, le=95.0, description="Target memory utilization percentage")
    scale_up_cooldown: int = Field(300, ge=60, le=3600, description="Scale up cooldown in seconds")
    scale_down_cooldown: int = Field(600, ge=60, le=3600, description="Scale down cooldown in seconds")


class AutoScalingConfigResponse(BaseModel):
    """Response for auto-scaling configuration."""
    
    strategy: str = Field(..., description="Active scaling strategy")
    min_instances: int = Field(..., description="Minimum instances")
    max_instances: int = Field(..., description="Maximum instances")
    current_instances: int = Field(..., description="Current number of instances")
    target_instances: int = Field(..., description="Target number of instances")
    scaling_enabled: bool = Field(..., description="Whether auto-scaling is enabled")
    last_scaling_action: Optional[str] = Field(None, description="Last scaling action taken")


class PerformanceMetricsResponse(BaseModel):
    """Response for performance system metrics."""
    
    timestamp: datetime = Field(..., description="Metrics timestamp")
    auto_scaler: Dict[str, Any] = Field(..., description="Auto-scaler metrics")
    query_optimizer: Dict[str, Any] = Field(..., description="Query optimizer metrics")
    system_resources: Dict[str, Any] = Field(..., description="System resource metrics")
    performance_insights: Dict[str, Any] = Field(..., description="Performance insights")


class PerformanceInsightsRequest(BaseModel):
    """Request for performance insights."""
    
    time_window_hours: int = Field(24, ge=1, le=168, description="Time window for analysis in hours")
    include_recommendations: bool = Field(True, description="Include optimization recommendations")
    metric_types: Optional[List[str]] = Field(None, description="Specific metrics to analyze")


class PerformanceInsightsResponse(BaseModel):
    """Response for performance insights."""
    
    time_window_hours: int = Field(..., description="Analysis time window")
    summary: Dict[str, Any] = Field(..., description="Performance summary")
    slow_queries: List[Dict[str, Any]] = Field(..., description="Slow query analysis")
    resource_usage: Dict[str, Any] = Field(..., description="Resource usage patterns")
    optimization_opportunities: List[Dict[str, Any]] = Field(..., description="Optimization opportunities")
    system_recommendations: List[Dict[str, Any]] = Field(..., description="System-wide recommendations")
    trends: Dict[str, Any] = Field(..., description="Performance trends")


# Dependencies
async def get_auto_scaler() -> AutoScalingEngine:
    """Get auto-scaling engine instance."""
    try:
        cache = get_provider(ProviderType.CACHE, "default")
        return AutoScalingEngine(cache=cache)
    except Exception as e:
        logger.error(f"Failed to get auto-scaler: {e}")
        raise HTTPException(status_code=500, detail="Auto-scaler unavailable")


async def get_query_optimizer() -> QueryOptimizer:
    """Get query optimizer instance."""
    try:
        cache = get_provider(ProviderType.CACHE, "default")
        return QueryOptimizer(cache=cache)
    except Exception as e:
        logger.error(f"Failed to get query optimizer: {e}")
        raise HTTPException(status_code=500, detail="Query optimizer unavailable")


# API Endpoints

@router.post("/query/analyze", response_model=QueryAnalysisResponse)
async def analyze_query(
    request: QueryAnalysisRequest,
    optimizer: QueryOptimizer = Depends(get_query_optimizer)
):
    """
    Analyze SQL query structure and performance characteristics.
    
    Provides detailed analysis of query complexity, cost estimation,
    and optimization opportunities using local SQL parsing.
    """
    try:
        start_time = time.time()
        
        # Analyze the query
        analysis = await optimizer.analyze_query(request.query)
        
        # Get optimization suggestions if requested
        optimization_suggestions = None
        if request.include_optimization:
            optimization_result = await optimizer.optimize_query(
                request.query,
                request.optimization_level
            )
            optimization_suggestions = {
                "optimized_query": optimization_result.optimized_query,
                "strategies": [s.value for s in optimization_result.optimization_strategies],
                "estimated_improvement": optimization_result.estimated_improvement,
                "warnings": optimization_result.warnings
            }
        
        # Format index recommendations
        index_recommendations = []
        if "index_recommendations" in analysis:
            for rec in analysis["index_recommendations"]:
                index_recommendations.append({
                    "table_name": rec.table_name,
                    "columns": rec.columns,
                    "index_type": rec.index_type,
                    "estimated_benefit": rec.estimated_benefit,
                    "reasoning": rec.reasoning,
                    "confidence": rec.confidence
                })
        
        analysis_time = (time.time() - start_time) * 1000
        
        return QueryAnalysisResponse(
            query=request.query,
            query_type=analysis.get("query_type", "unknown"),
            complexity_score=analysis.get("complexity_score", 0.0),
            estimated_cost=analysis.get("estimated_cost", 0.0),
            tables_involved=analysis.get("tables", []),
            optimization_suggestions=optimization_suggestions,
            index_recommendations=index_recommendations,
            analysis_time_ms=analysis_time
        )
        
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/optimize", response_model=QueryOptimizationResponse)
async def optimize_query(
    request: QueryOptimizationRequest,
    optimizer: QueryOptimizer = Depends(get_query_optimizer)
):
    """
    Optimize SQL query for better performance.
    
    Applies various optimization strategies including query rewriting,
    index recommendations, and structural improvements.
    """
    try:
        start_time = time.time()
        
        # Optimize the query
        optimization_result = await optimizer.optimize_query(
            request.query,
            request.optimization_level
        )
        
        # Format index recommendations
        index_recommendations = []
        for rec in optimization_result.index_recommendations:
            index_recommendations.append({
                "table_name": rec.table_name,
                "columns": rec.columns,
                "index_type": rec.index_type,
                "estimated_benefit": rec.estimated_benefit,
                "creation_cost": rec.creation_cost,
                "maintenance_cost": rec.maintenance_cost,
                "reasoning": rec.reasoning,
                "confidence": rec.confidence
            })
        
        optimization_time = (time.time() - start_time) * 1000
        
        return QueryOptimizationResponse(
            original_query=optimization_result.original_query,
            optimized_query=optimization_result.optimized_query,
            optimization_strategies=[s.value for s in optimization_result.optimization_strategies],
            estimated_improvement=optimization_result.estimated_improvement,
            index_recommendations=index_recommendations,
            warnings=optimization_result.warnings,
            optimization_time_ms=optimization_time
        )
        
    except Exception as e:
        logger.error(f"Query optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scaling/configure", response_model=AutoScalingConfigResponse)
async def configure_auto_scaling(
    request: AutoScalingConfigRequest,
    background_tasks: BackgroundTasks,
    scaler: AutoScalingEngine = Depends(get_auto_scaler)
):
    """
    Configure auto-scaling parameters and strategy.
    
    Updates auto-scaling settings including strategy, thresholds,
    and instance limits for optimal resource utilization.
    """
    try:
        # Validate strategy
        try:
            strategy = ScalingStrategy(request.strategy)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid scaling strategy: {request.strategy}")
        
        # Update configuration
        scaler.strategy = strategy
        scaler.safety_controller.min_instances = request.min_instances
        scaler.safety_controller.max_instances = request.max_instances
        scaler.safety_controller.scale_up_cooldown = request.scale_up_cooldown
        scaler.safety_controller.scale_down_cooldown = request.scale_down_cooldown
        
        # Update scaling rules for CPU and memory
        cpu_rule = next((rule for rule in scaler.scaling_rules if rule.metric_type == MetricType.CPU_USAGE), None)
        if cpu_rule:
            cpu_rule.scale_up_threshold = request.target_cpu_utilization
            cpu_rule.scale_down_threshold = request.target_cpu_utilization * 0.7
        
        memory_rule = next((rule for rule in scaler.scaling_rules if rule.metric_type == MetricType.MEMORY_USAGE), None)
        if memory_rule:
            memory_rule.scale_up_threshold = request.target_memory_utilization
            memory_rule.scale_down_threshold = request.target_memory_utilization * 0.7
        
        # Start auto-scaling if not already running
        if not scaler.running:
            background_tasks.add_task(scaler.start)
        
        status = scaler.get_current_status()
        
        return AutoScalingConfigResponse(
            strategy=strategy.value,
            min_instances=request.min_instances,
            max_instances=request.max_instances,
            current_instances=status["current_instances"],
            target_instances=status["target_instances"],
            scaling_enabled=status["running"],
            last_scaling_action=status["statistics"].get("last_action")
        )
        
    except Exception as e:
        logger.error(f"Auto-scaling configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scaling/status", response_model=AutoScalingConfigResponse)
async def get_auto_scaling_status(
    scaler: AutoScalingEngine = Depends(get_auto_scaler)
):
    """
    Get current auto-scaling status and configuration.
    
    Returns current scaling parameters, instance counts, and
    recent scaling activity.
    """
    try:
        status = scaler.get_current_status()
        
        return AutoScalingConfigResponse(
            strategy=status["strategy"],
            min_instances=scaler.safety_controller.min_instances,
            max_instances=scaler.safety_controller.max_instances,
            current_instances=status["current_instances"],
            target_instances=status["target_instances"],
            scaling_enabled=status["running"],
            last_scaling_action=status["statistics"].get("last_action")
        )
        
    except Exception as e:
        logger.error(f"Failed to get auto-scaling status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scaling/manual")
async def manual_scaling_action(
    action: str = Query(..., description="Scaling action: scale_up, scale_down, set_instances"),
    target_instances: Optional[int] = Query(None, ge=1, le=100, description="Target instance count for set_instances"),
    scaler: AutoScalingEngine = Depends(get_auto_scaler)
):
    """
    Perform manual scaling actions.
    
    Allows manual override of auto-scaling for specific scaling needs
    while respecting safety constraints.
    """
    try:
        if action == "scale_up":
            await scaler.scale_up()
            return {"action": "scale_up", "message": "Scaling up by 1 instance"}
        
        elif action == "scale_down":
            await scaler.scale_down()
            return {"action": "scale_down", "message": "Scaling down by 1 instance"}
        
        elif action == "set_instances":
            if target_instances is None:
                raise HTTPException(status_code=400, detail="target_instances required for set_instances action")
            
            # Respect safety constraints
            min_instances = scaler.safety_controller.min_instances
            max_instances = scaler.safety_controller.max_instances
            
            if target_instances < min_instances:
                raise HTTPException(status_code=400, detail=f"Target instances below minimum ({min_instances})")
            if target_instances > max_instances:
                raise HTTPException(status_code=400, detail=f"Target instances above maximum ({max_instances})")
            
            scaler.current_instances = target_instances
            scaler.target_instances = target_instances
            
            return {
                "action": "set_instances",
                "target_instances": target_instances,
                "message": f"Set instance count to {target_instances}"
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")
        
    except Exception as e:
        logger.error(f"Manual scaling action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insights", response_model=PerformanceInsightsResponse)
async def get_performance_insights(
    request: PerformanceInsightsRequest,
    optimizer: QueryOptimizer = Depends(get_query_optimizer),
    scaler: AutoScalingEngine = Depends(get_auto_scaler)
):
    """
    Get comprehensive performance insights and recommendations.
    
    Analyzes query performance, resource usage patterns, and provides
    actionable recommendations for system optimization.
    """
    try:
        # Get query performance insights
        query_insights = await optimizer.get_performance_insights(request.time_window_hours)
        
        # Get system recommendations
        system_recommendations = await optimizer.generate_system_recommendations()
        
        # Get auto-scaling statistics
        scaling_stats = await scaler.get_scaling_stats()
        
        # Calculate performance trends
        trends = {
            "query_performance": "stable",
            "resource_utilization": "stable",
            "scaling_frequency": "normal"
        }
        
        # Determine query performance trend
        if query_insights.get("summary", {}).get("queries_with_degrading_performance", 0) > 0:
            trends["query_performance"] = "degrading"
        elif query_insights.get("summary", {}).get("queries_with_improving_performance", 0) > 0:
            trends["query_performance"] = "improving"
        
        # Determine resource utilization trend
        resource_usage = query_insights.get("resource_usage", {})
        if resource_usage.get("avg_cpu_usage", 0) > 80 or resource_usage.get("avg_memory_usage", 0) > 80:
            trends["resource_utilization"] = "high"
        elif resource_usage.get("avg_cpu_usage", 0) < 20 and resource_usage.get("avg_memory_usage", 0) < 20:
            trends["resource_utilization"] = "low"
        
        # Combine recommendations
        all_recommendations = system_recommendations + [
            {
                "type": "auto_scaling",
                "priority": "medium",
                "title": "Review Auto-Scaling Configuration",
                "description": "Current auto-scaling strategy may need adjustment",
                "action": f"Consider tuning {scaler.strategy.value} strategy parameters",
                "estimated_impact": "medium"
            }
        ]
        
        return PerformanceInsightsResponse(
            time_window_hours=request.time_window_hours,
            summary=query_insights.get("summary", {}),
            slow_queries=query_insights.get("slow_queries", []),
            resource_usage=query_insights.get("resource_usage", {}),
            optimization_opportunities=query_insights.get("optimization_opportunities", []),
            system_recommendations=all_recommendations,
            trends=trends
        )
        
    except Exception as e:
        logger.error(f"Performance insights generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    optimizer: QueryOptimizer = Depends(get_query_optimizer),
    scaler: AutoScalingEngine = Depends(get_auto_scaler)
):
    """
    Get comprehensive performance system metrics.
    
    Returns detailed metrics from all performance subsystems including
    query optimization, auto-scaling, and resource utilization.
    """
    try:
        # Gather metrics from all components concurrently
        metrics_tasks = [
            optimizer.get_performance_metrics(),
            scaler.get_scaling_stats()
        ]
        
        optimizer_metrics, scaler_metrics = await asyncio.gather(
            *metrics_tasks, return_exceptions=True
        )
        
        # Handle any exceptions in metrics gathering
        if isinstance(optimizer_metrics, Exception):
            optimizer_metrics = {"error": str(optimizer_metrics)}
        if isinstance(scaler_metrics, Exception):
            scaler_metrics = {"error": str(scaler_metrics)}
        
        # Get current system resources
        import psutil
        system_resources = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
            "network_io": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {}
        }
        
        # Generate performance insights summary
        performance_insights = {
            "overall_health": "good",
            "critical_issues": 0,
            "optimization_opportunities": len(optimizer_metrics.get("performance_tracking", {}).get("optimization_opportunities", 0)),
            "system_efficiency": "high"
        }
        
        # Assess overall health
        if system_resources["cpu_percent"] > 90 or system_resources["memory_percent"] > 90:
            performance_insights["overall_health"] = "critical"
            performance_insights["critical_issues"] += 1
        elif system_resources["cpu_percent"] > 75 or system_resources["memory_percent"] > 75:
            performance_insights["overall_health"] = "warning"
        
        # Assess system efficiency
        if (optimizer_metrics.get("performance_tracking", {}).get("avg_query_execution_time", 0) > 1.0 or
            optimizer_metrics.get("performance_tracking", {}).get("slow_query_count", 0) > 10):
            performance_insights["system_efficiency"] = "medium"
        
        return PerformanceMetricsResponse(
            timestamp=datetime.utcnow(),
            auto_scaler=scaler_metrics,
            query_optimizer=optimizer_metrics,
            system_resources=system_resources,
            performance_insights=performance_insights
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/track-performance")
async def track_query_performance(
    query: str = Query(..., description="SQL query that was executed"),
    execution_time: float = Query(..., description="Query execution time in seconds"),
    cpu_usage: Optional[float] = Query(None, description="CPU usage percentage during execution"),
    memory_usage: Optional[float] = Query(None, description="Memory usage percentage during execution"),
    optimizer: QueryOptimizer = Depends(get_query_optimizer)
):
    """
    Track performance metrics for a specific query execution.
    
    Records execution metrics for performance analysis and optimization
    opportunity identification.
    """
    try:
        await optimizer.track_query_performance(
            query=query,
            execution_time=execution_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage
        )
        
        return {
            "message": "Query performance tracked successfully",
            "query_hash": hash(query) % (10**8),  # Simple hash for reference
            "execution_time": execution_time,
            "tracked_metrics": {
                "execution_time": True,
                "cpu_usage": cpu_usage is not None,
                "memory_usage": memory_usage is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Query performance tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/scaling/stop")
async def stop_auto_scaling(
    scaler: AutoScalingEngine = Depends(get_auto_scaler)
):
    """
    Stop auto-scaling operations.
    
    Disables automatic scaling while maintaining current instance count.
    Manual scaling operations will still be available.
    """
    try:
        if scaler.running:
            await scaler.stop()
            return {"message": "Auto-scaling stopped successfully", "scaling_enabled": False}
        else:
            return {"message": "Auto-scaling was not running", "scaling_enabled": False}
        
    except Exception as e:
        logger.error(f"Failed to stop auto-scaling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scaling/start")
async def start_auto_scaling(
    background_tasks: BackgroundTasks,
    scaler: AutoScalingEngine = Depends(get_auto_scaler)
):
    """
    Start auto-scaling operations.
    
    Enables automatic scaling based on configured strategy and thresholds.
    """
    try:
        if not scaler.running:
            background_tasks.add_task(scaler.start)
            return {"message": "Auto-scaling started successfully", "scaling_enabled": True}
        else:
            return {"message": "Auto-scaling is already running", "scaling_enabled": True}
        
    except Exception as e:
        logger.error(f"Failed to start auto-scaling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ["router"]