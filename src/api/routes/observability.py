"""
Enhanced Observability API endpoints.

Provides comprehensive system monitoring, performance analytics, anomaly detection,
and optimization recommendations. All processing is performed locally with zero external API calls.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ...core.observability.dashboard import MemoryQualityDashboard, QualityMetric, AlertConfig
from ...core.observability.telemetry import TelemetryCollector, TelemetryData
from ...core.observability.metrics import MetricsCollector, SystemMetrics
from ...core.observability.tracing import DistributedTracer, TraceContext
from ...core.cache.redis_cache import RedisCache
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Request/Response Models
class MetricsQueryRequest(BaseModel):
    """Request for metrics query."""
    
    metric_types: Optional[List[str]] = Field(None, description="Specific metric types to query")
    time_range_minutes: int = Field(60, ge=1, le=10080, description="Time range in minutes")
    aggregation: str = Field("avg", description="Aggregation method: avg, min, max, sum, count")
    granularity: str = Field("1m", description="Data granularity: 1m, 5m, 15m, 1h, 6h, 24h")


class MetricsQueryResponse(BaseModel):
    """Response for metrics query."""
    
    metrics: Dict[str, List[Dict[str, Any]]] = Field(..., description="Time series metrics data")
    time_range: Dict[str, str] = Field(..., description="Query time range")
    aggregation: str = Field(..., description="Applied aggregation method")
    total_data_points: int = Field(..., description="Total number of data points")


class QualityAnalysisRequest(BaseModel):
    """Request for memory quality analysis."""
    
    user_id: Optional[str] = Field(None, description="User ID to analyze (optional for global)")
    analysis_depth: str = Field("standard", description="Analysis depth: quick, standard, comprehensive")
    include_recommendations: bool = Field(True, description="Include optimization recommendations")
    time_window_hours: int = Field(24, ge=1, le=168, description="Analysis time window")


class QualityAnalysisResponse(BaseModel):
    """Response for memory quality analysis."""
    
    overall_quality_score: float = Field(..., description="Overall quality score (0-100)")
    quality_metrics: Dict[str, Any] = Field(..., description="Detailed quality metrics")
    trend_analysis: Dict[str, Any] = Field(..., description="Quality trend analysis")
    anomalies_detected: List[Dict[str, Any]] = Field(..., description="Detected anomalies")
    recommendations: List[Dict[str, Any]] = Field(..., description="Optimization recommendations")
    analysis_metadata: Dict[str, Any] = Field(..., description="Analysis metadata")


class AlertConfigRequest(BaseModel):
    """Request to configure monitoring alerts."""
    
    alert_name: str = Field(..., description="Alert configuration name")
    metric_type: str = Field(..., description="Metric type to monitor")
    threshold_value: float = Field(..., description="Alert threshold value")
    comparison_operator: str = Field("greater_than", description="Comparison: greater_than, less_than, equals")
    time_window_minutes: int = Field(5, ge=1, le=60, description="Time window for evaluation")
    severity: str = Field("medium", description="Alert severity: low, medium, high, critical")
    enabled: bool = Field(True, description="Whether alert is enabled")


class AlertConfigResponse(BaseModel):
    """Response for alert configuration."""
    
    alert_id: str = Field(..., description="Alert configuration ID")
    alert_name: str = Field(..., description="Alert name")
    configuration: Dict[str, Any] = Field(..., description="Alert configuration details")
    status: str = Field(..., description="Alert status")
    created_at: datetime = Field(..., description="Creation timestamp")


class TraceQueryRequest(BaseModel):
    """Request for distributed trace query."""
    
    trace_id: Optional[str] = Field(None, description="Specific trace ID to query")
    operation_name: Optional[str] = Field(None, description="Operation name filter")
    service_name: Optional[str] = Field(None, description="Service name filter")
    min_duration_ms: Optional[float] = Field(None, description="Minimum trace duration in ms")
    time_range_hours: int = Field(1, ge=1, le=24, description="Time range in hours")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of traces")


class TraceQueryResponse(BaseModel):
    """Response for distributed trace query."""
    
    traces: List[Dict[str, Any]] = Field(..., description="Trace data")
    total_traces: int = Field(..., description="Total number of traces found")
    query_performance: Dict[str, Any] = Field(..., description="Query performance metrics")
    trace_statistics: Dict[str, Any] = Field(..., description="Trace statistics summary")


class PerformanceAnalysisRequest(BaseModel):
    """Request for performance analysis."""
    
    analysis_type: str = Field("comprehensive", description="Analysis type: quick, standard, comprehensive")
    time_range_hours: int = Field(6, ge=1, le=168, description="Analysis time range")
    include_predictions: bool = Field(True, description="Include performance predictions")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")


class PerformanceAnalysisResponse(BaseModel):
    """Response for performance analysis."""
    
    performance_score: float = Field(..., description="Overall performance score (0-100)")
    bottlenecks: List[Dict[str, Any]] = Field(..., description="Identified performance bottlenecks")
    resource_utilization: Dict[str, Any] = Field(..., description="Resource utilization analysis")
    performance_trends: Dict[str, Any] = Field(..., description="Performance trend analysis")
    predictions: Optional[Dict[str, Any]] = Field(None, description="Performance predictions")
    optimization_opportunities: List[Dict[str, Any]] = Field(..., description="Optimization opportunities")


class SystemHealthResponse(BaseModel):
    """Response for system health status."""
    
    overall_health: str = Field(..., description="Overall system health: excellent, good, fair, poor, critical")
    health_score: float = Field(..., description="Numeric health score (0-100)")
    component_health: Dict[str, Any] = Field(..., description="Individual component health status")
    active_alerts: List[Dict[str, Any]] = Field(..., description="Currently active alerts")
    recent_incidents: List[Dict[str, Any]] = Field(..., description="Recent incidents and resolutions")
    system_uptime: Dict[str, Any] = Field(..., description="System uptime information")


# Dependencies
async def get_dashboard() -> MemoryQualityDashboard:
    """Get memory quality dashboard instance."""
    try:
        memory_manager = get_provider(ProviderType.MEMORY_MANAGER, "default")
        cache = get_provider(ProviderType.CACHE, "default")
        return MemoryQualityDashboard(memory_manager=memory_manager, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get dashboard: {e}")
        raise HTTPException(status_code=500, detail="Dashboard unavailable")


async def get_telemetry_collector() -> TelemetryCollector:
    """Get telemetry collector instance."""
    try:
        return TelemetryCollector()
    except Exception as e:
        logger.error(f"Failed to get telemetry collector: {e}")
        raise HTTPException(status_code=500, detail="Telemetry system unavailable")


async def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance."""
    try:
        cache = get_provider(ProviderType.CACHE, "default")
        return MetricsCollector(cache=cache)
    except Exception as e:
        logger.error(f"Failed to get metrics collector: {e}")
        raise HTTPException(status_code=500, detail="Metrics system unavailable")


async def get_tracer() -> DistributedTracer:
    """Get distributed tracer instance."""
    try:
        return DistributedTracer()
    except Exception as e:
        logger.error(f"Failed to get tracer: {e}")
        raise HTTPException(status_code=500, detail="Tracing system unavailable")


# API Endpoints

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    dashboard: MemoryQualityDashboard = Depends(get_dashboard),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Get comprehensive system health status.
    
    Returns overall system health, component status, active alerts,
    and recent incidents for complete system visibility.
    """
    try:
        # Get system health from dashboard
        health_data = await dashboard.get_system_health()
        
        # Get metrics for additional health indicators
        system_metrics = await metrics.get_current_metrics()
        
        # Calculate overall health score
        health_score = health_data.get("overall_score", 0.0)
        
        # Determine health status
        if health_score >= 95:
            overall_health = "excellent"
        elif health_score >= 85:
            overall_health = "good"
        elif health_score >= 70:
            overall_health = "fair"
        elif health_score >= 50:
            overall_health = "poor"
        else:
            overall_health = "critical"
        
        # Get component health
        component_health = {
            "memory_system": health_data.get("memory_system_health", "unknown"),
            "database": health_data.get("database_health", "unknown"),
            "cache": health_data.get("cache_health", "unknown"),
            "api": health_data.get("api_health", "unknown"),
            "background_tasks": health_data.get("background_tasks_health", "unknown")
        }
        
        # Get active alerts
        active_alerts = health_data.get("active_alerts", [])
        
        # Get recent incidents
        recent_incidents = health_data.get("recent_incidents", [])
        
        # System uptime
        system_uptime = {
            "current_uptime_seconds": system_metrics.get("uptime_seconds", 0),
            "last_restart": system_metrics.get("last_restart", "unknown"),
            "restart_count_24h": system_metrics.get("restart_count_24h", 0)
        }
        
        return SystemHealthResponse(
            overall_health=overall_health,
            health_score=health_score,
            component_health=component_health,
            active_alerts=active_alerts,
            recent_incidents=recent_incidents,
            system_uptime=system_uptime
        )
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/query", response_model=MetricsQueryResponse)
async def query_metrics(
    request: MetricsQueryRequest,
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Query system metrics with flexible filtering and aggregation.
    
    Retrieves time series metrics data with configurable aggregation
    and granularity for detailed performance analysis.
    """
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=request.time_range_minutes)
        
        # Query metrics
        metrics_data = await metrics.query_metrics(
            metric_types=request.metric_types,
            start_time=start_time,
            end_time=end_time,
            aggregation=request.aggregation,
            granularity=request.granularity
        )
        
        # Count total data points
        total_data_points = sum(len(data) for data in metrics_data.values())
        
        return MetricsQueryResponse(
            metrics=metrics_data,
            time_range={
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            aggregation=request.aggregation,
            total_data_points=total_data_points
        )
        
    except Exception as e:
        logger.error(f"Metrics query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality/analyze", response_model=QualityAnalysisResponse)
async def analyze_memory_quality(
    request: QualityAnalysisRequest,
    dashboard: MemoryQualityDashboard = Depends(get_dashboard)
):
    """
    Perform comprehensive memory quality analysis.
    
    Analyzes memory quality metrics, detects anomalies, identifies trends,
    and provides actionable optimization recommendations.
    """
    try:
        # Perform quality analysis
        analysis_result = await dashboard.analyze_memory_quality(
            user_id=request.user_id,
            analysis_depth=request.analysis_depth,
            time_window_hours=request.time_window_hours
        )
        
        # Get quality metrics
        quality_metrics = analysis_result.get("quality_metrics", {})
        overall_score = quality_metrics.get("overall_quality_score", 0.0)
        
        # Get trend analysis
        trend_analysis = analysis_result.get("trend_analysis", {})
        
        # Get anomalies
        anomalies = analysis_result.get("anomalies_detected", [])
        
        # Get recommendations if requested
        recommendations = []
        if request.include_recommendations:
            recommendations = analysis_result.get("recommendations", [])
        
        # Analysis metadata
        analysis_metadata = {
            "analysis_depth": request.analysis_depth,
            "time_window_hours": request.time_window_hours,
            "user_scope": "single_user" if request.user_id else "global",
            "analysis_duration_ms": analysis_result.get("analysis_duration_ms", 0),
            "data_points_analyzed": analysis_result.get("data_points_analyzed", 0)
        }
        
        return QualityAnalysisResponse(
            overall_quality_score=overall_score,
            quality_metrics=quality_metrics,
            trend_analysis=trend_analysis,
            anomalies_detected=anomalies,
            recommendations=recommendations,
            analysis_metadata=analysis_metadata
        )
        
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/configure", response_model=AlertConfigResponse)
async def configure_alert(
    request: AlertConfigRequest,
    dashboard: MemoryQualityDashboard = Depends(get_dashboard)
):
    """
    Configure monitoring alerts for system metrics.
    
    Sets up automated alerts based on metric thresholds with
    configurable severity levels and notification settings.
    """
    try:
        # Create alert configuration
        alert_config = AlertConfig(
            name=request.alert_name,
            metric_type=request.metric_type,
            threshold_value=request.threshold_value,
            comparison_operator=request.comparison_operator,
            time_window_minutes=request.time_window_minutes,
            severity=request.severity,
            enabled=request.enabled
        )
        
        # Register alert with dashboard
        alert_id = await dashboard.configure_alert(alert_config)
        
        return AlertConfigResponse(
            alert_id=alert_id,
            alert_name=request.alert_name,
            configuration={
                "metric_type": request.metric_type,
                "threshold_value": request.threshold_value,
                "comparison_operator": request.comparison_operator,
                "time_window_minutes": request.time_window_minutes,
                "severity": request.severity,
                "enabled": request.enabled
            },
            status="active" if request.enabled else "inactive",
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Alert configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def list_alerts(
    status: Optional[str] = Query(None, description="Filter by status: active, inactive, triggered"),
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
    dashboard: MemoryQualityDashboard = Depends(get_dashboard)
):
    """
    List configured monitoring alerts.
    
    Returns all configured alerts with optional filtering by status and severity.
    """
    try:
        alerts = await dashboard.list_alerts(status=status, severity=severity)
        
        return {
            "alerts": [
                {
                    "alert_id": alert["alert_id"],
                    "name": alert["name"],
                    "metric_type": alert["metric_type"],
                    "threshold_value": alert["threshold_value"],
                    "comparison_operator": alert["comparison_operator"],
                    "severity": alert["severity"],
                    "status": alert["status"],
                    "last_triggered": alert.get("last_triggered"),
                    "trigger_count_24h": alert.get("trigger_count_24h", 0),
                    "created_at": alert["created_at"]
                }
                for alert in alerts
            ],
            "total_alerts": len(alerts),
            "filters_applied": {
                "status": status,
                "severity": severity
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/traces/query", response_model=TraceQueryResponse)
async def query_traces(
    request: TraceQueryRequest,
    tracer: DistributedTracer = Depends(get_tracer)
):
    """
    Query distributed traces for performance analysis.
    
    Searches trace data with flexible filtering for detailed
    performance investigation and bottleneck identification.
    """
    try:
        start_time = time.time()
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time_query = end_time - timedelta(hours=request.time_range_hours)
        
        # Query traces
        traces = await tracer.query_traces(
            trace_id=request.trace_id,
            operation_name=request.operation_name,
            service_name=request.service_name,
            min_duration_ms=request.min_duration_ms,
            start_time=start_time_query,
            end_time=end_time,
            limit=request.limit
        )
        
        # Calculate trace statistics
        if traces:
            durations = [trace.get("duration_ms", 0) for trace in traces]
            trace_statistics = {
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
                "error_rate": len([t for t in traces if t.get("error", False)]) / len(traces),
                "unique_operations": len(set(t.get("operation_name", "") for t in traces)),
                "unique_services": len(set(t.get("service_name", "") for t in traces))
            }
        else:
            trace_statistics = {
                "avg_duration_ms": 0,
                "min_duration_ms": 0,
                "max_duration_ms": 0,
                "p95_duration_ms": 0,
                "error_rate": 0,
                "unique_operations": 0,
                "unique_services": 0
            }
        
        query_performance = {
            "query_duration_ms": (time.time() - start_time) * 1000,
            "traces_scanned": len(traces),
            "time_range_hours": request.time_range_hours
        }
        
        return TraceQueryResponse(
            traces=traces,
            total_traces=len(traces),
            query_performance=query_performance,
            trace_statistics=trace_statistics
        )
        
    except Exception as e:
        logger.error(f"Trace query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/analyze", response_model=PerformanceAnalysisResponse)
async def analyze_performance(
    request: PerformanceAnalysisRequest,
    dashboard: MemoryQualityDashboard = Depends(get_dashboard),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Perform comprehensive performance analysis.
    
    Analyzes system performance across multiple dimensions, identifies
    bottlenecks, and provides optimization recommendations.
    """
    try:
        # Perform performance analysis
        analysis_tasks = [
            dashboard.analyze_performance(
                analysis_type=request.analysis_type,
                time_range_hours=request.time_range_hours,
                focus_areas=request.focus_areas
            ),
            metrics.analyze_performance_trends(
                time_range_hours=request.time_range_hours
            )
        ]
        
        dashboard_analysis, metrics_analysis = await asyncio.gather(*analysis_tasks)
        
        # Combine analysis results
        performance_score = (
            dashboard_analysis.get("performance_score", 0) +
            metrics_analysis.get("performance_score", 0)
        ) / 2
        
        # Identify bottlenecks
        bottlenecks = []
        bottlenecks.extend(dashboard_analysis.get("bottlenecks", []))
        bottlenecks.extend(metrics_analysis.get("bottlenecks", []))
        
        # Resource utilization analysis
        resource_utilization = {
            "cpu": metrics_analysis.get("cpu_utilization", {}),
            "memory": metrics_analysis.get("memory_utilization", {}),
            "disk": metrics_analysis.get("disk_utilization", {}),
            "network": metrics_analysis.get("network_utilization", {}),
            "database": dashboard_analysis.get("database_utilization", {})
        }
        
        # Performance trends
        performance_trends = {
            "response_time_trend": dashboard_analysis.get("response_time_trend", "stable"),
            "throughput_trend": dashboard_analysis.get("throughput_trend", "stable"),
            "error_rate_trend": dashboard_analysis.get("error_rate_trend", "stable"),
            "resource_usage_trend": metrics_analysis.get("resource_usage_trend", "stable")
        }
        
        # Predictions if requested
        predictions = None
        if request.include_predictions:
            predictions = {
                "performance_forecast": dashboard_analysis.get("performance_forecast", {}),
                "resource_forecasts": metrics_analysis.get("resource_forecasts", {}),
                "capacity_planning": dashboard_analysis.get("capacity_planning", {})
            }
        
        # Optimization opportunities
        optimization_opportunities = []
        optimization_opportunities.extend(dashboard_analysis.get("optimization_opportunities", []))
        optimization_opportunities.extend(metrics_analysis.get("optimization_opportunities", []))
        
        return PerformanceAnalysisResponse(
            performance_score=performance_score,
            bottlenecks=bottlenecks,
            resource_utilization=resource_utilization,
            performance_trends=performance_trends,
            predictions=predictions,
            optimization_opportunities=optimization_opportunities
        )
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry/export")
async def export_telemetry_data(
    format: str = Query("json", description="Export format: json, csv, prometheus"),
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range in hours"),
    metric_types: Optional[List[str]] = Query(None, description="Specific metric types to export"),
    telemetry: TelemetryCollector = Depends(get_telemetry_collector)
):
    """
    Export telemetry data in various formats.
    
    Exports collected telemetry data for external analysis,
    monitoring systems, or long-term storage.
    """
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        # Export telemetry data
        exported_data = await telemetry.export_data(
            start_time=start_time,
            end_time=end_time,
            format=format,
            metric_types=metric_types
        )
        
        return {
            "export_format": format,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_hours": time_range_hours
            },
            "data": exported_data,
            "export_metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "data_points": len(exported_data) if isinstance(exported_data, list) else 0,
                "metric_types_included": metric_types or "all"
            }
        }
        
    except Exception as e:
        logger.error(f"Telemetry export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/widgets")
async def create_dashboard_widget(
    widget_type: str = Query(..., description="Widget type: gauge, chart, table, heatmap"),
    metric_type: str = Query(..., description="Metric type to display"),
    configuration: Dict[str, Any] = {},
    dashboard: MemoryQualityDashboard = Depends(get_dashboard)
):
    """
    Create custom dashboard widget.
    
    Creates interactive dashboard widgets for real-time monitoring
    with configurable visualizations and metrics.
    """
    try:
        # Create widget
        widget = await dashboard.create_widget(
            widget_type=widget_type,
            metric_type=metric_type,
            configuration=configuration
        )
        
        return {
            "widget_id": widget["widget_id"],
            "widget_type": widget_type,
            "metric_type": metric_type,
            "configuration": configuration,
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Widget creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/widgets")
async def list_dashboard_widgets(
    dashboard: MemoryQualityDashboard = Depends(get_dashboard)
):
    """
    List all dashboard widgets.
    
    Returns all configured dashboard widgets with their current
    status and configuration details.
    """
    try:
        widgets = await dashboard.list_widgets()
        
        return {
            "widgets": widgets,
            "total_widgets": len(widgets),
            "widget_types": list(set(w.get("widget_type") for w in widgets)),
            "metrics_displayed": list(set(w.get("metric_type") for w in widgets))
        }
        
    except Exception as e:
        logger.error(f"Failed to list widgets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/{alert_id}")
async def delete_alert(
    alert_id: str,
    dashboard: MemoryQualityDashboard = Depends(get_dashboard)
):
    """
    Delete monitoring alert configuration.
    
    Removes alert configuration and stops monitoring for the specified alert.
    """
    try:
        success = await dashboard.delete_alert(alert_id)
        
        if success:
            return {"message": "Alert deleted successfully", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ["router"]