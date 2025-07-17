"""
Local Interactive Network Visualization System.

This module provides comprehensive 3D memory network visualization using
plotly for interactive exploration, dash for web interface, WebSocket
for real-time updates, and local caching for performance optimization.
"""

import asyncio
import json
import math
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ...core.memory.manager import MemoryManager
from ...core.graph.neo4j_client import Neo4jClient
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class VisualizationMode(str, Enum):
    """Visualization display modes."""
    NETWORK_3D = "network_3d"           # 3D network graph
    NETWORK_2D = "network_2d"           # 2D network graph  
    HIERARCHICAL = "hierarchical"        # Hierarchical tree layout
    CIRCULAR = "circular"               # Circular layout
    TIMELINE = "timeline"               # Temporal timeline view
    CLUSTER = "cluster"                 # Clustered communities
    FORCE_DIRECTED = "force_directed"   # Force-directed layout
    MATRIX = "matrix"                   # Adjacency matrix view


class NodeSizeMode(str, Enum):
    """Node sizing strategies."""
    UNIFORM = "uniform"                 # All nodes same size
    DEGREE = "degree"                   # Size by connection count
    IMPORTANCE = "importance"           # Size by graph centrality
    FREQUENCY = "frequency"             # Size by access frequency
    RECENCY = "recency"                 # Size by recent activity
    CONTENT_LENGTH = "content_length"   # Size by content amount


class EdgeColorMode(str, Enum):
    """Edge coloring strategies."""
    UNIFORM = "uniform"                 # All edges same color
    STRENGTH = "strength"               # Color by connection strength
    TYPE = "type"                       # Color by relationship type
    RECENCY = "recency"                 # Color by creation time
    ACTIVITY = "activity"               # Color by usage frequency


class LayoutAlgorithm(str, Enum):
    """Graph layout algorithms."""
    SPRING = "spring"                   # Spring-force layout
    KAMADA_KAWAI = "kamada_kawai"      # Kamada-Kawai algorithm
    CIRCULAR = "circular"               # Circular positioning
    SHELL = "shell"                     # Shell layout
    SPECTRAL = "spectral"              # Spectral layout
    RANDOM = "random"                   # Random positioning
    FRUCHTERMAN_REINGOLD = "fruchterman_reingold"  # FR algorithm


@dataclass
class VisualizationConfig:
    """Configuration for network visualization."""
    mode: VisualizationMode = VisualizationMode.NETWORK_3D
    layout_algorithm: LayoutAlgorithm = LayoutAlgorithm.SPRING
    node_size_mode: NodeSizeMode = NodeSizeMode.IMPORTANCE
    edge_color_mode: EdgeColorMode = EdgeColorMode.STRENGTH
    max_nodes: int = 500
    max_edges: int = 1000
    show_labels: bool = True
    show_edge_labels: bool = False
    enable_clustering: bool = True
    enable_filtering: bool = True
    enable_search: bool = True
    animation_enabled: bool = True
    real_time_updates: bool = True
    cache_duration_minutes: int = 5


class NetworkNode(BaseModel):
    """Network visualization node."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    node_id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Display label")
    content: str = Field(..., description="Node content/description")
    node_type: str = Field(default="memory", description="Type of node")
    
    # Position coordinates
    x: float = Field(default=0.0, description="X coordinate")
    y: float = Field(default=0.0, description="Y coordinate") 
    z: float = Field(default=0.0, description="Z coordinate")
    
    # Visual properties
    size: float = Field(default=10.0, ge=1.0, le=100.0, description="Node size")
    color: str = Field(default="#1f77b4", description="Node color")
    opacity: float = Field(default=0.8, ge=0.0, le=1.0, description="Node opacity")
    
    # Metadata
    degree: int = Field(default=0, description="Number of connections")
    centrality: float = Field(default=0.0, description="Centrality score")
    cluster_id: Optional[str] = Field(default=None, description="Cluster membership")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    last_accessed: Optional[datetime] = Field(default=None, description="Last access time")
    access_count: int = Field(default=0, description="Access frequency")


class NetworkEdge(BaseModel):
    """Network visualization edge."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    edge_id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    label: str = Field(default="", description="Edge label")
    edge_type: str = Field(default="similarity", description="Type of relationship")
    
    # Edge properties
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Connection strength")
    weight: float = Field(default=1.0, ge=0.0, description="Edge weight")
    color: str = Field(default="#888888", description="Edge color")
    width: float = Field(default=1.0, ge=0.1, le=10.0, description="Edge width")
    opacity: float = Field(default=0.6, ge=0.0, le=1.0, description="Edge opacity")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    validation_status: str = Field(default="pending", description="Validation status")


class NetworkGraph(BaseModel):
    """Complete network graph for visualization."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    graph_id: str = Field(..., description="Unique graph identifier")
    nodes: List[NetworkNode] = Field(..., description="Graph nodes")
    edges: List[NetworkEdge] = Field(..., description="Graph edges")
    
    # Graph metadata
    node_count: int = Field(..., description="Total number of nodes")
    edge_count: int = Field(..., description="Total number of edges")
    density: float = Field(..., description="Graph density")
    avg_clustering: float = Field(..., description="Average clustering coefficient")
    diameter: Optional[int] = Field(default=None, description="Graph diameter")
    
    # Visualization settings
    config: VisualizationConfig = Field(default_factory=VisualizationConfig)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation time")


class LocalNetworkVisualizer:
    """Local Interactive Network Visualization System."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        graph_client: Neo4jClient,
        cache: Optional[RedisCache] = None
    ):
        self.memory_manager = memory_manager
        self.graph_client = graph_client
        self.cache = cache
        
        # Graph analysis
        self.nx_graph = nx.Graph()
        self.node_positions = {}
        self.cluster_assignments = {}
        
        # Dash app
        self.app = None
        self.is_running = False
        
        # Caching
        self.layout_cache = {}
        self.graph_cache = {}
        self.last_update = datetime.utcnow()
        
        logger.info("LocalNetworkVisualizer initialized")
    
    async def load_memory_network(
        self,
        user_id: str,
        config: Optional[VisualizationConfig] = None
    ) -> NetworkGraph:
        """Load memory network for visualization."""
        try:
            if config is None:
                config = VisualizationConfig()
            
            # Check cache first
            cache_key = f"network_vis:{user_id}:{hash(str(config))}"
            if self.cache:
                cached = await self.cache.get(cache_key)
                if cached:
                    logger.info("Loaded network from cache", user_id=user_id)
                    return NetworkGraph.model_validate_json(cached)
            
            # Load memories and relationships
            memories = await self.memory_manager.get_all_memories(user_id)
            relationships = await self.graph_client.get_all_relationships(user_id)
            
            # Build NetworkX graph for analysis
            await self._build_networkx_graph(memories, relationships)
            
            # Create visualization nodes
            nodes = await self._create_visualization_nodes(memories, config)
            
            # Create visualization edges  
            edges = await self._create_visualization_edges(relationships, config)
            
            # Apply layout algorithm
            await self._apply_layout_algorithm(nodes, config.layout_algorithm)
            
            # Calculate graph metrics
            metrics = await self._calculate_graph_metrics()
            
            # Create network graph
            network_graph = NetworkGraph(
                graph_id=f"network_{user_id}_{datetime.utcnow().isoformat()}",
                nodes=nodes[:config.max_nodes],
                edges=edges[:config.max_edges],
                node_count=len(nodes),
                edge_count=len(edges),
                config=config,
                **metrics
            )
            
            # Cache result
            if self.cache:
                await self.cache.set(
                    cache_key, 
                    network_graph.model_dump_json(),
                    expire_minutes=config.cache_duration_minutes
                )
            
            logger.info(
                "Network loaded for visualization",
                user_id=user_id,
                nodes=len(nodes),
                edges=len(edges)
            )
            
            return network_graph
            
        except Exception as e:
            logger.error("Failed to load memory network", error=str(e), user_id=user_id)
            raise
    
    async def _build_networkx_graph(self, memories: List[Any], relationships: List[Any]) -> None:
        """Build NetworkX graph for analysis."""
        self.nx_graph.clear()
        
        # Add nodes (memories)
        for memory in memories:
            self.nx_graph.add_node(
                memory.id,
                content=memory.content,
                created_at=memory.created_at,
                tags=getattr(memory, 'tags', []),
                access_count=getattr(memory, 'access_count', 0)
            )
        
        # Add edges (relationships)
        for rel in relationships:
            if rel.source_id in self.nx_graph and rel.target_id in self.nx_graph:
                self.nx_graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    weight=rel.strength,
                    relationship_type=rel.relationship_type,
                    confidence=getattr(rel, 'confidence', 1.0)
                )
    
    async def _create_visualization_nodes(
        self,
        memories: List[Any],
        config: VisualizationConfig
    ) -> List[NetworkNode]:
        """Create visualization nodes from memories."""
        nodes = []
        
        # Calculate centrality measures
        centrality = nx.degree_centrality(self.nx_graph)
        betweenness = nx.betweenness_centrality(self.nx_graph)
        
        # Detect communities if clustering enabled
        if config.enable_clustering:
            self.cluster_assignments = await self._detect_communities()
        
        for memory in memories:
            if memory.id not in self.nx_graph:
                continue
                
            # Calculate node size
            size = await self._calculate_node_size(memory, config.node_size_mode)
            
            # Calculate node color
            color = await self._calculate_node_color(memory, config)
            
            # Get cluster assignment
            cluster_id = self.cluster_assignments.get(memory.id)
            
            node = NetworkNode(
                node_id=memory.id,
                label=memory.content[:50] + "..." if len(memory.content) > 50 else memory.content,
                content=memory.content,
                size=size,
                color=color,
                degree=self.nx_graph.degree(memory.id),
                centrality=centrality.get(memory.id, 0.0),
                cluster_id=cluster_id,
                tags=getattr(memory, 'tags', []),
                created_at=memory.created_at,
                last_accessed=getattr(memory, 'last_accessed', None),
                access_count=getattr(memory, 'access_count', 0)
            )
            
            nodes.append(node)
        
        return nodes
    
    async def _create_visualization_edges(
        self,
        relationships: List[Any], 
        config: VisualizationConfig
    ) -> List[NetworkEdge]:
        """Create visualization edges from relationships."""
        edges = []
        
        for rel in relationships:
            if rel.source_id not in self.nx_graph or rel.target_id not in self.nx_graph:
                continue
            
            # Calculate edge color
            color = await self._calculate_edge_color(rel, config.edge_color_mode)
            
            # Calculate edge width based on strength
            width = max(0.5, min(5.0, rel.strength * 5))
            
            edge = NetworkEdge(
                edge_id=f"{rel.source_id}_{rel.target_id}",
                source_id=rel.source_id,
                target_id=rel.target_id,
                label=getattr(rel, 'relationship_type', ''),
                edge_type=getattr(rel, 'relationship_type', 'similarity'),
                strength=rel.strength,
                weight=rel.strength,
                color=color,
                width=width,
                confidence=getattr(rel, 'confidence', 1.0),
                validation_status=getattr(rel, 'validation_status', 'validated')
            )
            
            edges.append(edge)
        
        return edges
    
    async def _apply_layout_algorithm(
        self,
        nodes: List[NetworkNode],
        algorithm: LayoutAlgorithm
    ) -> None:
        """Apply layout algorithm to position nodes."""
        # Check cache for this layout
        cache_key = f"layout_{algorithm}_{len(nodes)}_{hash(str([n.node_id for n in nodes]))}"
        if cache_key in self.layout_cache:
            positions = self.layout_cache[cache_key]
        else:
            # Calculate new layout
            if algorithm == LayoutAlgorithm.SPRING:
                positions = nx.spring_layout(self.nx_graph, dim=3, k=1, iterations=50)
            elif algorithm == LayoutAlgorithm.KAMADA_KAWAI:
                positions = nx.kamada_kawai_layout(self.nx_graph, dim=3)
            elif algorithm == LayoutAlgorithm.CIRCULAR:
                positions = nx.circular_layout(self.nx_graph)
                # Add z-coordinate for 3D
                for node_id in positions:
                    positions[node_id] = (*positions[node_id], 0)
            elif algorithm == LayoutAlgorithm.SPECTRAL:
                positions = nx.spectral_layout(self.nx_graph)
                # Add z-coordinate for 3D
                for node_id in positions:
                    positions[node_id] = (*positions[node_id], 0)
            else:
                # Default to spring layout
                positions = nx.spring_layout(self.nx_graph, dim=3, k=1, iterations=50)
            
            # Cache layout
            self.layout_cache[cache_key] = positions
        
        # Apply positions to nodes
        for node in nodes:
            if node.node_id in positions:
                pos = positions[node.node_id]
                node.x = float(pos[0])
                node.y = float(pos[1])
                node.z = float(pos[2]) if len(pos) > 2 else 0.0
    
    async def _calculate_node_size(self, memory: Any, size_mode: NodeSizeMode) -> float:
        """Calculate node size based on sizing mode."""
        base_size = 10.0
        
        if size_mode == NodeSizeMode.UNIFORM:
            return base_size
        elif size_mode == NodeSizeMode.DEGREE:
            degree = self.nx_graph.degree(memory.id)
            return base_size + (degree * 2)
        elif size_mode == NodeSizeMode.IMPORTANCE:
            centrality = nx.degree_centrality(self.nx_graph).get(memory.id, 0)
            return base_size + (centrality * 30)
        elif size_mode == NodeSizeMode.FREQUENCY:
            access_count = getattr(memory, 'access_count', 0)
            return base_size + min(30, access_count * 2)
        elif size_mode == NodeSizeMode.RECENCY:
            if hasattr(memory, 'last_accessed') and memory.last_accessed:
                days_ago = (datetime.utcnow() - memory.last_accessed).days
                recency_score = max(0, 30 - days_ago) / 30
                return base_size + (recency_score * 20)
            return base_size
        elif size_mode == NodeSizeMode.CONTENT_LENGTH:
            content_length = len(memory.content)
            return base_size + min(30, content_length / 50)
        
        return base_size
    
    async def _calculate_node_color(self, memory: Any, config: VisualizationConfig) -> str:
        """Calculate node color based on properties."""
        # Color by cluster if clustering enabled
        if config.enable_clustering and memory.id in self.cluster_assignments:
            cluster_id = self.cluster_assignments[memory.id]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            return colors[hash(str(cluster_id)) % len(colors)]
        
        # Color by tags
        if hasattr(memory, 'tags') and memory.tags:
            primary_tag = memory.tags[0]
            return f"#{hash(primary_tag) % 0xFFFFFF:06x}"
        
        # Default color
        return "#1f77b4"
    
    async def _calculate_edge_color(self, relationship: Any, color_mode: EdgeColorMode) -> str:
        """Calculate edge color based on coloring mode."""
        if color_mode == EdgeColorMode.UNIFORM:
            return "#888888"
        elif color_mode == EdgeColorMode.STRENGTH:
            # Color from red (weak) to green (strong)
            strength = relationship.strength
            r = int(255 * (1 - strength))
            g = int(255 * strength)
            return f"rgb({r},{g},0)"
        elif color_mode == EdgeColorMode.TYPE:
            rel_type = getattr(relationship, 'relationship_type', 'similarity')
            type_colors = {
                'similarity': '#1f77b4',
                'causal': '#ff7f0e', 
                'temporal': '#2ca02c',
                'entity': '#d62728',
                'topic': '#9467bd'
            }
            return type_colors.get(rel_type, '#888888')
        elif color_mode == EdgeColorMode.RECENCY:
            if hasattr(relationship, 'created_at'):
                days_ago = (datetime.utcnow() - relationship.created_at).days
                intensity = max(0, min(255, 255 - days_ago * 5))
                return f"rgb({intensity},{intensity},{intensity})"
            return "#888888"
        
        return "#888888"
    
    async def _detect_communities(self) -> Dict[str, str]:
        """Detect communities in the graph."""
        try:
            if len(self.nx_graph) < 3:
                return {}
            
            # Use Louvain community detection
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(self.nx_graph)
            
            assignments = {}
            for i, community in enumerate(communities):
                for node_id in community:
                    assignments[node_id] = f"cluster_{i}"
            
            return assignments
            
        except Exception as e:
            logger.warning("Community detection failed", error=str(e))
            return {}
    
    async def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """Calculate graph-level metrics."""
        if len(self.nx_graph) == 0:
            return {
                "density": 0.0,
                "avg_clustering": 0.0,
                "diameter": None
            }
        
        density = nx.density(self.nx_graph)
        
        try:
            avg_clustering = nx.average_clustering(self.nx_graph)
        except:
            avg_clustering = 0.0
        
        try:
            if nx.is_connected(self.nx_graph):
                diameter = nx.diameter(self.nx_graph)
            else:
                diameter = None
        except:
            diameter = None
        
        return {
            "density": density,
            "avg_clustering": avg_clustering, 
            "diameter": diameter
        }
    
    def create_dash_app(self) -> dash.Dash:
        """Create Dash application for interactive visualization."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("Memory Network Visualization", className="text-center mb-4"),
                        
                        # Controls
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Visualization Mode:"),
                                        dcc.Dropdown(
                                            id="viz-mode-dropdown",
                                            options=[
                                                {"label": "3D Network", "value": "network_3d"},
                                                {"label": "2D Network", "value": "network_2d"},
                                                {"label": "Hierarchical", "value": "hierarchical"},
                                                {"label": "Timeline", "value": "timeline"}
                                            ],
                                            value="network_3d"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Layout Algorithm:"),
                                        dcc.Dropdown(
                                            id="layout-dropdown",
                                            options=[
                                                {"label": "Spring", "value": "spring"},
                                                {"label": "Kamada-Kawai", "value": "kamada_kawai"},
                                                {"label": "Circular", "value": "circular"},
                                                {"label": "Spectral", "value": "spectral"}
                                            ],
                                            value="spring"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("Node Size Mode:"),
                                        dcc.Dropdown(
                                            id="node-size-dropdown",
                                            options=[
                                                {"label": "Importance", "value": "importance"},
                                                {"label": "Degree", "value": "degree"},
                                                {"label": "Frequency", "value": "frequency"},
                                                {"label": "Uniform", "value": "uniform"}
                                            ],
                                            value="importance"
                                        )
                                    ], width=3),
                                    
                                    dbc.Col([
                                        html.Label("User ID:"),
                                        dcc.Input(
                                            id="user-id-input",
                                            type="text",
                                            placeholder="Enter user ID",
                                            value="default_user"
                                        )
                                    ], width=3)
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Update Visualization",
                                            id="update-btn",
                                            color="primary",
                                            className="mt-3"
                                        )
                                    ], width=2),
                                    
                                    dbc.Col([
                                        dbc.Switch(
                                            id="real-time-switch",
                                            label="Real-time Updates",
                                            value=True
                                        )
                                    ], width=2),
                                    
                                    dbc.Col([
                                        dbc.Switch(
                                            id="clustering-switch", 
                                            label="Show Clusters",
                                            value=True
                                        )
                                    ], width=2)
                                ], className="mt-2")
                            ])
                        ], className="mb-4"),
                        
                        # Main visualization
                        dcc.Graph(
                            id="network-graph",
                            style={"height": "700px"},
                            config={"displayModeBar": True}
                        ),
                        
                        # Statistics
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("Graph Statistics", className="card-title"),
                                        html.Div(id="graph-stats")
                                    ])
                                ])
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("Selected Node Info", className="card-title"),
                                        html.Div(id="node-info")
                                    ])
                                ])
                            ], width=6)
                        ], className="mt-4")
                        
                    ], width=12)
                ])
            ], fluid=True),
            
            # Store for graph data
            dcc.Store(id="graph-data-store"),
            
            # Interval for real-time updates
            dcc.Interval(
                id="update-interval",
                interval=30000,  # 30 seconds
                n_intervals=0,
                disabled=False
            )
        ])
        
        # Callbacks
        self._setup_dash_callbacks(app)
        
        return app
    
    def _setup_dash_callbacks(self, app: dash.Dash) -> None:
        """Setup Dash application callbacks."""
        
        @app.callback(
            [Output("network-graph", "figure"),
             Output("graph-stats", "children"),
             Output("graph-data-store", "data")],
            [Input("update-btn", "n_clicks"),
             Input("update-interval", "n_intervals")],
            [State("user-id-input", "value"),
             State("viz-mode-dropdown", "value"),
             State("layout-dropdown", "value"),
             State("node-size-dropdown", "value"),
             State("clustering-switch", "value"),
             State("real-time-switch", "value")]
        )
        def update_visualization(n_clicks, n_intervals, user_id, viz_mode, layout, node_size, clustering, real_time):
            if not user_id:
                raise PreventUpdate
            
            # Create configuration
            config = VisualizationConfig(
                mode=VisualizationMode(viz_mode),
                layout_algorithm=LayoutAlgorithm(layout),
                node_size_mode=NodeSizeMode(node_size),
                enable_clustering=clustering,
                real_time_updates=real_time
            )
            
            # Load network data (this would be async in real implementation)
            # For demo purposes, we'll create a simple example
            try:
                # This would call: network_graph = await self.load_memory_network(user_id, config)
                network_graph = self._create_example_network(config)
                
                # Create plotly figure
                fig = self._create_plotly_figure(network_graph)
                
                # Create statistics
                stats = self._create_graph_statistics(network_graph)
                
                return fig, stats, network_graph.model_dump()
                
            except Exception as e:
                logger.error("Failed to update visualization", error=str(e))
                return {}, html.Div("Error loading visualization"), {}
        
        @app.callback(
            Output("node-info", "children"),
            [Input("network-graph", "clickData")],
            [State("graph-data-store", "data")]
        )
        def display_node_info(click_data, graph_data):
            if not click_data or not graph_data:
                return "Click on a node to see details"
            
            try:
                point = click_data["points"][0]
                node_id = point.get("customdata", "")
                
                # Find node in graph data
                for node_data in graph_data.get("nodes", []):
                    if node_data["node_id"] == node_id:
                        return html.Div([
                            html.P(f"ID: {node_data['node_id']}"),
                            html.P(f"Content: {node_data['content'][:100]}..."),
                            html.P(f"Connections: {node_data['degree']}"),
                            html.P(f"Centrality: {node_data['centrality']:.3f}"),
                            html.P(f"Cluster: {node_data.get('cluster_id', 'None')}")
                        ])
                
                return "Node information not found"
                
            except Exception as e:
                return f"Error displaying node info: {str(e)}"
    
    def _create_example_network(self, config: VisualizationConfig) -> NetworkGraph:
        """Create example network for demonstration."""
        # This is a placeholder - in real implementation this would load from database
        nodes = [
            NetworkNode(
                node_id=f"node_{i}",
                label=f"Memory {i}",
                content=f"This is memory content {i}",
                x=np.random.uniform(-1, 1),
                y=np.random.uniform(-1, 1),
                z=np.random.uniform(-1, 1),
                size=np.random.uniform(10, 30),
                degree=np.random.randint(1, 5),
                centrality=np.random.uniform(0, 1)
            )
            for i in range(20)
        ]
        
        edges = []
        for i in range(len(nodes)):
            for j in range(i+1, min(i+3, len(nodes))):
                edges.append(NetworkEdge(
                    edge_id=f"edge_{i}_{j}",
                    source_id=nodes[i].node_id,
                    target_id=nodes[j].node_id,
                    strength=np.random.uniform(0.3, 1.0),
                    width=np.random.uniform(1, 3)
                ))
        
        return NetworkGraph(
            graph_id="example_graph",
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
            density=0.3,
            avg_clustering=0.4,
            config=config
        )
    
    def _create_plotly_figure(self, network_graph: NetworkGraph) -> go.Figure:
        """Create plotly figure from network graph."""
        nodes = network_graph.nodes
        edges = network_graph.edges
        
        if network_graph.config.mode == VisualizationMode.NETWORK_3D:
            return self._create_3d_network_plot(nodes, edges)
        elif network_graph.config.mode == VisualizationMode.NETWORK_2D:
            return self._create_2d_network_plot(nodes, edges)
        else:
            return self._create_3d_network_plot(nodes, edges)  # Default to 3D
    
    def _create_3d_network_plot(self, nodes: List[NetworkNode], edges: List[NetworkEdge]) -> go.Figure:
        """Create 3D network visualization."""
        # Create edge traces
        edge_traces = []
        for edge in edges:
            source_node = next((n for n in nodes if n.node_id == edge.source_id), None)
            target_node = next((n for n in nodes if n.node_id == edge.target_id), None)
            
            if source_node and target_node:
                edge_trace = go.Scatter3d(
                    x=[source_node.x, target_node.x, None],
                    y=[source_node.y, target_node.y, None],
                    z=[source_node.z, target_node.z, None],
                    mode='lines',
                    line=dict(color=edge.color, width=edge.width),
                    opacity=edge.opacity,
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter3d(
            x=[n.x for n in nodes],
            y=[n.y for n in nodes], 
            z=[n.z for n in nodes],
            mode='markers+text' if len(nodes) < 50 else 'markers',
            marker=dict(
                size=[n.size for n in nodes],
                color=[n.color for n in nodes],
                opacity=[n.opacity for n in nodes],
                line=dict(width=2, color='white')
            ),
            text=[n.label for n in nodes] if len(nodes) < 50 else None,
            textposition='middle center',
            customdata=[n.node_id for n in nodes],
            hovertemplate='<b>%{text}</b><br>Connections: %{marker.size}<extra></extra>',
            name='Memories'
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title="3D Memory Network",
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='black'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def _create_2d_network_plot(self, nodes: List[NetworkNode], edges: List[NetworkEdge]) -> go.Figure:
        """Create 2D network visualization."""
        # Create edge traces
        edge_traces = []
        for edge in edges:
            source_node = next((n for n in nodes if n.node_id == edge.source_id), None)
            target_node = next((n for n in nodes if n.node_id == edge.target_id), None)
            
            if source_node and target_node:
                edge_trace = go.Scatter(
                    x=[source_node.x, target_node.x, None],
                    y=[source_node.y, target_node.y, None],
                    mode='lines',
                    line=dict(color=edge.color, width=edge.width),
                    opacity=edge.opacity,
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[n.x for n in nodes],
            y=[n.y for n in nodes],
            mode='markers+text' if len(nodes) < 50 else 'markers',
            marker=dict(
                size=[n.size for n in nodes],
                color=[n.color for n in nodes],
                opacity=[n.opacity for n in nodes],
                line=dict(width=2, color='white')
            ),
            text=[n.label for n in nodes] if len(nodes) < 50 else None,
            textposition='middle center',
            customdata=[n.node_id for n in nodes],
            hovertemplate='<b>%{text}</b><br>Connections: %{marker.size}<extra></extra>',
            name='Memories'
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title="2D Memory Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='black',
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def _create_graph_statistics(self, network_graph: NetworkGraph) -> html.Div:
        """Create graph statistics display."""
        return html.Div([
            html.P(f"Nodes: {network_graph.node_count}"),
            html.P(f"Edges: {network_graph.edge_count}"),
            html.P(f"Density: {network_graph.density:.3f}"),
            html.P(f"Avg Clustering: {network_graph.avg_clustering:.3f}"),
            html.P(f"Diameter: {network_graph.diameter or 'N/A'}")
        ])
    
    async def start_server(self, host: str = "127.0.0.1", port: int = 8050) -> None:
        """Start the Dash visualization server."""
        if self.is_running:
            logger.warning("Visualization server already running")
            return
        
        try:
            self.app = self.create_dash_app()
            self.is_running = True
            
            logger.info(
                "Starting visualization server",
                host=host,
                port=port
            )
            
            # Run in separate thread to avoid blocking
            import threading
            server_thread = threading.Thread(
                target=lambda: self.app.run_server(
                    host=host,
                    port=port,
                    debug=False,
                    use_reloader=False
                )
            )
            server_thread.daemon = True
            server_thread.start()
            
        except Exception as e:
            logger.error("Failed to start visualization server", error=str(e))
            self.is_running = False
            raise
    
    async def stop_server(self) -> None:
        """Stop the visualization server."""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info("Visualization server stopped")
    
    async def update_real_time(self, user_id: str, event_data: Dict[str, Any]) -> None:
        """Handle real-time updates to visualization."""
        if not self.is_running:
            return
        
        try:
            # Invalidate cache for this user
            cache_pattern = f"network_vis:{user_id}:*"
            if self.cache:
                await self.cache.delete_pattern(cache_pattern)
            
            # Clear layout cache
            self.layout_cache.clear()
            
            logger.info("Real-time visualization update triggered", user_id=user_id)
            
        except Exception as e:
            logger.error("Failed to handle real-time update", error=str(e))


# Module exports
__all__ = [
    "LocalNetworkVisualizer",
    "NetworkGraph", 
    "NetworkNode",
    "NetworkEdge",
    "VisualizationConfig",
    "VisualizationMode",
    "NodeSizeMode",
    "EdgeColorMode",
    "LayoutAlgorithm"
]