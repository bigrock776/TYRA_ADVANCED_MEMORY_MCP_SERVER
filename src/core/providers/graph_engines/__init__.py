"""
Graph engine providers module for Tyra MCP Memory Server.

This module provides various graph database implementations with a pluggable
architecture for easy swapping of graph storage backends.
"""

from .neo4j import Neo4jEngine
from .registry import (
    GraphEngineProviderRegistry,
    create_graph_engine,
    get_available_providers,
)

# Legacy dictionary for backwards compatibility
GRAPH_ENGINE_PROVIDERS = {
    "neo4j": Neo4jEngine,
}

__all__ = [
    "Neo4jEngine",
    "GraphEngineProviderRegistry",
    "get_available_providers",
    "create_graph_engine",
    "GRAPH_ENGINE_PROVIDERS",
]
