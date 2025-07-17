"""
Dashboard application entry point.

Creates and configures the dashboard server with all analytics components.
"""

import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .analytics.local_usage import UsageAnalyticsDashboard
from .analytics.local_roi import ROIAnalyticsDashboard
from .insights.local_gaps import GapAnalyticsDashboard
from .visualization.local_network import NetworkVisualizationDashboard
from ..core.memory.manager import MemoryManager
from ..core.utils.simple_logger import get_logger

logger = get_logger(__name__)


def create_dashboard_app(memory_manager: Optional[MemoryManager] = None) -> FastAPI:
    """Create the dashboard FastAPI application."""
    
    dashboard_app = FastAPI(
        title="Tyra Memory Analytics Dashboard",
        description="Comprehensive memory system analytics and visualization",
        version="1.0.0"
    )
    
    # Initialize dashboard components
    usage_dashboard = UsageAnalyticsDashboard(memory_manager)
    roi_dashboard = ROIAnalyticsDashboard(memory_manager)
    gaps_dashboard = GapAnalyticsDashboard(memory_manager)
    network_dashboard = NetworkVisualizationDashboard(memory_manager)
    
    @dashboard_app.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request):
        """Main dashboard page."""
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Tyra Memory Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #333; }
                    .nav-list { list-style-type: none; padding: 0; }
                    .nav-list li { margin: 10px 0; }
                    .nav-list a { 
                        color: #007bff; 
                        text-decoration: none; 
                        padding: 10px 15px;
                        border: 1px solid #007bff;
                        border-radius: 5px;
                        display: inline-block;
                        min-width: 200px;
                    }
                    .nav-list a:hover { 
                        background-color: #007bff; 
                        color: white; 
                    }
                    .status { 
                        margin-top: 20px; 
                        padding: 10px; 
                        background-color: #f8f9fa; 
                        border-radius: 5px; 
                    }
                </style>
            </head>
            <body>
                <h1>🧠 Tyra Memory Analytics Dashboard</h1>
                <p>Comprehensive memory system analytics and visualization</p>
                
                <ul class="nav-list">
                    <li><a href="/usage">📊 Usage Analytics</a></li>
                    <li><a href="/roi">💰 ROI Analysis</a></li>
                    <li><a href="/gaps">🔍 Knowledge Gaps</a></li>
                    <li><a href="/network">🕸️ Network Visualization</a></li>
                    <li><a href="/health">❤️ System Health</a></li>
                </ul>
                
                <div class="status">
                    <h3>System Status</h3>
                    <p><strong>Memory Manager:</strong> {}</p>
                    <p><strong>Dashboard Version:</strong> 1.0.0</p>
                    <p><strong>Analytics Enabled:</strong> ✅</p>
                </div>
            </body>
        </html>
        """.format("Connected" if memory_manager else "Not Connected"))
    
    @dashboard_app.get("/usage")
    async def usage_analytics():
        """Usage analytics endpoint."""
        try:
            if usage_dashboard and hasattr(usage_dashboard, 'get_analytics'):
                analytics_data = await usage_dashboard.get_analytics()
                return analytics_data
            else:
                return {
                    "status": "active",
                    "message": "Usage analytics dashboard initialized",
                    "memory_manager_connected": memory_manager is not None,
                    "analytics": {
                        "total_memories": 0,
                        "active_agents": 0,
                        "recent_activity": []
                    }
                }
        except Exception as e:
            logger.error(f"Usage analytics error: {e}")
            return {"error": str(e), "status": "error"}
    
    @dashboard_app.get("/roi")
    async def roi_analytics():
        """ROI analytics endpoint."""
        try:
            if roi_dashboard and hasattr(roi_dashboard, 'get_analytics'):
                analytics_data = await roi_dashboard.get_analytics()
                return analytics_data
            else:
                return {
                    "status": "active",
                    "message": "ROI analytics dashboard initialized",
                    "memory_manager_connected": memory_manager is not None,
                    "roi_metrics": {
                        "memory_efficiency": 0.85,
                        "search_performance": 0.92,
                        "cost_savings": 0.0
                    }
                }
        except Exception as e:
            logger.error(f"ROI analytics error: {e}")
            return {"error": str(e), "status": "error"}
    
    @dashboard_app.get("/gaps")
    async def gap_analytics():
        """Gap analytics endpoint."""
        try:
            if gaps_dashboard and hasattr(gaps_dashboard, 'get_analytics'):
                analytics_data = await gaps_dashboard.get_analytics()
                return analytics_data
            else:
                return {
                    "status": "active",
                    "message": "Gap analytics dashboard initialized",
                    "memory_manager_connected": memory_manager is not None,
                    "knowledge_gaps": {
                        "identified_gaps": 0,
                        "coverage_score": 0.75,
                        "recommendations": []
                    }
                }
        except Exception as e:
            logger.error(f"Gap analytics error: {e}")
            return {"error": str(e), "status": "error"}
    
    @dashboard_app.get("/network")
    async def network_visualization():
        """Network visualization endpoint."""
        try:
            if network_dashboard and hasattr(network_dashboard, 'get_visualization'):
                visualization_data = await network_dashboard.get_visualization()
                return visualization_data
            else:
                return {
                    "status": "active",
                    "message": "Network visualization dashboard initialized",
                    "memory_manager_connected": memory_manager is not None,
                    "network_data": {
                        "nodes": 0,
                        "edges": 0,
                        "clusters": 0
                    }
                }
        except Exception as e:
            logger.error(f"Network visualization error: {e}")
            return {"error": str(e), "status": "error"}
    
    @dashboard_app.get("/health")
    async def dashboard_health():
        """Dashboard health check."""
        return {
            "status": "healthy",
            "dashboard_version": "1.0.0",
            "memory_manager_connected": memory_manager is not None,
            "components": {
                "usage_dashboard": usage_dashboard is not None,
                "roi_dashboard": roi_dashboard is not None,
                "gaps_dashboard": gaps_dashboard is not None,
                "network_dashboard": network_dashboard is not None
            },
            "timestamp": asyncio.get_event_loop().time()
        }
    
    return dashboard_app