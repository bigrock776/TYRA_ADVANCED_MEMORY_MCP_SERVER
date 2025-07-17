"""
Knowledge Graph API endpoints.

Provides access to the temporal knowledge graph stored in Neo4j,
including entity extraction, relationship mapping, and graph queries.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.graph.graphiti_integration import GraphitiManager
from ...core.graph.neo4j_client import Neo4jClient
from ...core.graph.causal_inference import CausalInferenceEngine, CausalRelationType, CausalInferenceMethod
from ...core.utils.logger import get_logger
from ...core.utils.registry import ProviderType, get_provider

logger = get_logger(__name__)

router = APIRouter()


# Enums
class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    TOOL = "tool"
    DOCUMENT = "document"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""

    RELATED_TO = "related_to"
    MENTIONS = "mentions"
    CREATED_BY = "created_by"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    HAPPENED_AT = "happened_at"
    CAUSED_BY = "caused_by"
    SIMILAR_TO = "similar_to"


class GraphQueryType(str, Enum):
    """Types of graph queries."""

    NEIGHBORS = "neighbors"
    PATH = "path"
    SUBGRAPH = "subgraph"
    TEMPORAL = "temporal"
    PATTERN = "pattern"


# Request/Response Models
class Entity(BaseModel):
    """Knowledge graph entity."""

    id: str = Field(..., description="Entity ID")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    properties: Dict[str, Any] = Field(default={}, description="Entity properties")
    confidence: float = Field(..., description="Extraction confidence")
    first_seen: datetime = Field(..., description="First occurrence timestamp")
    last_seen: datetime = Field(..., description="Most recent occurrence timestamp")
    occurrence_count: int = Field(..., description="Number of occurrences")


class Relationship(BaseModel):
    """Relationship between entities."""

    id: str = Field(..., description="Relationship ID")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: RelationshipType = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(
        default={}, description="Relationship properties"
    )
    confidence: float = Field(..., description="Relationship confidence")
    created_at: datetime = Field(..., description="Creation timestamp")


class GraphNode(BaseModel):
    """Node in graph visualization."""

    id: str
    label: str
    type: str
    properties: Dict[str, Any] = {}
    x: Optional[float] = None
    y: Optional[float] = None


class GraphEdge(BaseModel):
    """Edge in graph visualization."""

    id: str
    source: str
    target: str
    label: str
    properties: Dict[str, Any] = {}


class GraphData(BaseModel):
    """Graph data for visualization."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = {}


class EntityExtractionRequest(BaseModel):
    """Request to extract entities from text."""

    text: str = Field(..., description="Text to extract entities from")
    types: Optional[List[EntityType]] = Field(
        None, description="Entity types to extract"
    )
    min_confidence: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )


class GraphQueryRequest(BaseModel):
    """Request for graph queries."""

    query_type: GraphQueryType = Field(..., description="Type of graph query")
    entity_ids: List[str] = Field(..., description="Entity IDs to query")
    depth: int = Field(2, ge=1, le=5, description="Query depth")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")


class TemporalQueryRequest(BaseModel):
    """Request for temporal graph queries."""

    start_time: datetime = Field(..., description="Start of time range")
    end_time: datetime = Field(..., description="End of time range")
    entity_types: Optional[List[EntityType]] = Field(
        None, description="Filter by entity types"
    )
    relationship_types: Optional[List[RelationshipType]] = Field(
        None, description="Filter by relationship types"
    )
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")


class CausalAnalysisRequest(BaseModel):
    """Request for causal inference analysis."""
    
    entity_ids: List[str] = Field(..., description="Entity IDs to analyze for causal relationships")
    method: CausalInferenceMethod = Field(CausalInferenceMethod.GRANGER_CAUSALITY, description="Causal inference method")
    time_window_days: int = Field(30, ge=1, le=365, description="Time window for analysis")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")


class CausalModelExportRequest(BaseModel):
    """Request for causal model export."""
    
    format: str = Field("json", description="Export format: json, graphml, dot, cytoscape")
    include_metadata: bool = Field(True, description="Include metadata in export")
    entity_filter: Optional[List[str]] = Field(None, description="Filter by specific entities")


class CausalModelImportRequest(BaseModel):
    """Request for causal model import."""
    
    model_data: Dict[str, Any] = Field(..., description="Causal model data to import")
    format: str = Field("json", description="Import format: json, graphml, dot, cytoscape")
    merge_strategy: str = Field("merge", description="Strategy: merge, replace, append")


class CausalValidationRequest(BaseModel):
    """Request for causal relationship validation."""
    
    relationships: List[Dict[str, Any]] = Field(..., description="Causal relationships to validate")
    validation_data: Optional[List[Dict[str, Any]]] = Field(None, description="Additional validation data")


# Dependencies
async def get_neo4j_client() -> Neo4jClient:
    """Get Neo4j client instance."""
    try:
        return get_provider(ProviderType.GRAPH_CLIENT, "neo4j")
    except Exception as e:
        logger.error(f"Failed to get Neo4j client: {e}")
        raise HTTPException(status_code=500, detail="Graph database unavailable")


async def get_graphiti_manager() -> GraphitiManager:
    """Get Graphiti manager instance."""
    try:
        return get_provider(ProviderType.GRAPH_MANAGER, "graphiti")
    except Exception as e:
        logger.error(f"Failed to get Graphiti manager: {e}")
        raise HTTPException(status_code=500, detail="Graph manager unavailable")


async def get_causal_inference_engine() -> CausalInferenceEngine:
    """Get causal inference engine instance."""
    try:
        neo4j_client = get_provider(ProviderType.GRAPH_CLIENT, "neo4j")
        cache = get_provider(ProviderType.CACHE, "default")
        return CausalInferenceEngine(neo4j_client=neo4j_client, cache=cache)
    except Exception as e:
        logger.error(f"Failed to get causal inference engine: {e}")
        raise HTTPException(status_code=500, detail="Causal inference engine unavailable")


@router.post("/entities/extract", response_model=List[Entity])
async def extract_entities(
    request: EntityExtractionRequest,
    graphiti: GraphitiManager = Depends(get_graphiti_manager),
):
    """
    Extract entities from text.

    Uses NLP to identify and extract entities with their types and properties.
    """
    try:
        # Extract entities
        extracted = await graphiti.extract_entities(
            text=request.text,
            entity_types=request.types,
            min_confidence=request.min_confidence,
        )

        # Convert to response format
        entities = []
        for entity_data in extracted:
            entities.append(
                Entity(
                    id=entity_data["id"],
                    name=entity_data["name"],
                    type=EntityType(entity_data["type"]),
                    properties=entity_data.get("properties", {}),
                    confidence=entity_data["confidence"],
                    first_seen=entity_data["first_seen"],
                    last_seen=entity_data["last_seen"],
                    occurrence_count=entity_data["occurrence_count"],
                )
            )

        return entities

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}", response_model=Entity)
async def get_entity(
    entity_id: str, neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Get details of a specific entity.

    Returns full entity information including properties and metadata.
    """
    try:
        entity_data = await neo4j.get_entity(entity_id)

        if not entity_data:
            raise HTTPException(status_code=404, detail="Entity not found")

        return Entity(
            id=entity_data["id"],
            name=entity_data["name"],
            type=EntityType(entity_data["type"]),
            properties=entity_data.get("properties", {}),
            confidence=entity_data["confidence"],
            first_seen=entity_data["first_seen"],
            last_seen=entity_data["last_seen"],
            occurrence_count=entity_data["occurrence_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities", response_model=List[Entity])
async def list_entities(
    entity_type: Optional[EntityType] = Query(
        None, description="Filter by entity type"
    ),
    search: Optional[str] = Query(None, description="Search in entity names"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    List entities in the knowledge graph.

    Supports filtering by type and searching by name.
    """
    try:
        # Query entities
        entity_list = await neo4j.list_entities(
            entity_type=entity_type, search=search, limit=limit, offset=offset
        )

        # Convert to response format
        entities = []
        for entity_data in entity_list:
            entities.append(
                Entity(
                    id=entity_data["id"],
                    name=entity_data["name"],
                    type=EntityType(entity_data["type"]),
                    properties=entity_data.get("properties", {}),
                    confidence=entity_data["confidence"],
                    first_seen=entity_data["first_seen"],
                    last_seen=entity_data["last_seen"],
                    occurrence_count=entity_data["occurrence_count"],
                )
            )

        return entities

    except Exception as e:
        logger.error(f"Failed to list entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships/{entity_id}", response_model=List[Relationship])
async def get_entity_relationships(
    entity_id: str,
    relationship_type: Optional[RelationshipType] = Query(
        None, description="Filter by relationship type"
    ),
    direction: str = Query(
        "both", description="Relationship direction: in, out, or both"
    ),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Get relationships for an entity.

    Returns all relationships where the entity is either source or target.
    """
    try:
        # Validate direction
        if direction not in ["in", "out", "both"]:
            raise HTTPException(status_code=400, detail="Invalid direction")

        # Query relationships
        relationships_data = await neo4j.get_relationships(
            entity_id=entity_id,
            relationship_type=relationship_type,
            direction=direction,
        )

        # Convert to response format
        relationships = []
        for rel_data in relationships_data:
            relationships.append(
                Relationship(
                    id=rel_data["id"],
                    source_id=rel_data["source_id"],
                    target_id=rel_data["target_id"],
                    type=RelationshipType(rel_data["type"]),
                    properties=rel_data.get("properties", {}),
                    confidence=rel_data["confidence"],
                    created_at=rel_data["created_at"],
                )
            )

        return relationships

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get relationships for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=GraphData)
async def query_graph(
    request: GraphQueryRequest, neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """
    Execute complex graph queries.

    Supports various query types including neighbor search, path finding,
    and subgraph extraction.
    """
    try:
        result = None

        if request.query_type == GraphQueryType.NEIGHBORS:
            # Get neighbors up to specified depth
            result = await neo4j.get_neighbors(
                entity_ids=request.entity_ids,
                depth=request.depth,
                filters=request.filters,
            )

        elif request.query_type == GraphQueryType.PATH:
            # Find paths between entities
            if len(request.entity_ids) < 2:
                raise HTTPException(
                    status_code=400, detail="Path query requires at least 2 entities"
                )

            result = await neo4j.find_paths(
                start_id=request.entity_ids[0],
                end_id=request.entity_ids[1],
                max_depth=request.depth,
            )

        elif request.query_type == GraphQueryType.SUBGRAPH:
            # Extract subgraph around entities
            result = await neo4j.get_subgraph(
                entity_ids=request.entity_ids,
                depth=request.depth,
                filters=request.filters,
            )

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported query type: {request.query_type}"
            )

        # Convert to graph visualization format
        return _convert_to_graph_data(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/temporal/query", response_model=GraphData)
async def temporal_graph_query(
    request: TemporalQueryRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Query the temporal knowledge graph.

    Retrieves entities and relationships within a specific time range.
    """
    try:
        # Execute temporal query
        result = await neo4j.temporal_query(
            start_time=request.start_time,
            end_time=request.end_time,
            entity_types=request.entity_types,
            relationship_types=request.relationship_types,
            limit=request.limit,
        )

        # Convert to graph visualization format
        return _convert_to_graph_data(result)

    except Exception as e:
        logger.error(f"Temporal query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memories/{memory_id}/graph")
async def add_memory_to_graph(
    memory_id: str, graphiti: GraphitiManager = Depends(get_graphiti_manager)
):
    """
    Add a memory to the knowledge graph.

    Extracts entities and relationships from the memory and adds them to the graph.
    """
    try:
        # Process memory and add to graph
        result = await graphiti.add_memory_to_graph(memory_id)

        return {
            "memory_id": memory_id,
            "entities_added": result["entities_added"],
            "relationships_added": result["relationships_added"],
            "message": "Memory successfully added to knowledge graph",
        }

    except Exception as e:
        logger.error(f"Failed to add memory {memory_id} to graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_graph_statistics(neo4j: Neo4jClient = Depends(get_neo4j_client)):
    """
    Get knowledge graph statistics.

    Returns information about the size and composition of the graph.
    """
    try:
        stats = await neo4j.get_statistics()

        return {
            "total_entities": stats["total_entities"],
            "total_relationships": stats["total_relationships"],
            "entity_types": stats["entity_types"],
            "relationship_types": stats["relationship_types"],
            "avg_relationships_per_entity": stats["avg_relationships_per_entity"],
            "most_connected_entities": stats["most_connected_entities"],
            "temporal_range": stats["temporal_range"],
        }

    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualize")
async def visualize_graph(
    entity_ids: List[str] = Query(..., description="Entity IDs to visualize"),
    depth: int = Query(2, ge=1, le=3, description="Visualization depth"),
    layout: str = Query(
        "force", description="Layout algorithm: force, hierarchical, circular"
    ),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Generate graph visualization data.

    Returns graph data with layout coordinates for visualization.
    """
    try:
        # Get subgraph
        subgraph = await neo4j.get_subgraph(entity_ids=entity_ids, depth=depth)

        # Convert to visualization format
        graph_data = _convert_to_graph_data(subgraph)

        # Apply layout algorithm
        if layout == "force":
            graph_data = _apply_force_layout(graph_data)
        elif layout == "hierarchical":
            graph_data = _apply_hierarchical_layout(graph_data)
        elif layout == "circular":
            graph_data = _apply_circular_layout(graph_data)

        return graph_data

    except Exception as e:
        logger.error(f"Graph visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal/analyze")
async def analyze_causal_relationships(
    request: CausalAnalysisRequest,
    causal_engine: CausalInferenceEngine = Depends(get_causal_inference_engine)
):
    """
    Analyze causal relationships between entities.
    
    Uses advanced causal inference methods to identify and validate
    causal relationships in the knowledge graph.
    """
    try:
        # Perform causal analysis
        analysis_result = await causal_engine.infer_causal_relationships(
            entity_pairs=[(request.entity_ids[i], request.entity_ids[i+1]) 
                         for i in range(len(request.entity_ids)-1)],
            method=request.method,
            time_window_days=request.time_window_days,
            confidence_threshold=request.confidence_threshold
        )
        
        # Format results
        causal_relationships = []
        for claim in analysis_result.causal_claims:
            causal_relationships.append({
                "cause_entity": claim.cause_entity,
                "effect_entity": claim.effect_entity,
                "relationship_type": claim.relationship_type.value,
                "confidence": claim.confidence,
                "confidence_level": claim.confidence_level.value,
                "evidence": [
                    {
                        "type": evidence.evidence_type,
                        "strength": evidence.strength,
                        "p_value": evidence.p_value,
                        "method": evidence.method_used
                    }
                    for evidence in claim.evidence
                ],
                "temporal_order": claim.temporal_precedence,
                "effect_size": claim.effect_size
            })
        
        return {
            "analysis_id": analysis_result.analysis_id,
            "method_used": request.method.value,
            "causal_relationships": causal_relationships,
            "summary": {
                "total_relationships": len(causal_relationships),
                "high_confidence_count": len([r for r in causal_relationships if r["confidence"] >= 0.8]),
                "average_confidence": sum(r["confidence"] for r in causal_relationships) / len(causal_relationships) if causal_relationships else 0
            },
            "metadata": analysis_result.metadata
        }
        
    except Exception as e:
        logger.error(f"Causal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal/export")
async def export_causal_model(
    request: CausalModelExportRequest,
    causal_engine: CausalInferenceEngine = Depends(get_causal_inference_engine)
):
    """
    Export causal model in various formats.
    
    Exports the current causal model with relationships and metadata
    in the specified format for external analysis or visualization.
    """
    try:
        # Export causal model
        exported_model = await causal_engine.export_causal_model(
            format=request.format,
            include_metadata=request.include_metadata
        )
        
        # Filter by entities if specified
        if request.entity_filter:
            # Filter the exported model to include only specified entities
            filtered_model = _filter_causal_model(exported_model, request.entity_filter)
            exported_model = filtered_model
        
        return {
            "format": request.format,
            "model_data": exported_model,
            "export_timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "total_entities": len(exported_model.get("entities", [])),
                "total_relationships": len(exported_model.get("relationships", [])),
                "includes_metadata": request.include_metadata
            }
        }
        
    except Exception as e:
        logger.error(f"Causal model export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal/import")
async def import_causal_model(
    request: CausalModelImportRequest,
    causal_engine: CausalInferenceEngine = Depends(get_causal_inference_engine)
):
    """
    Import causal model from external data.
    
    Imports causal relationships and entities from external sources
    using various merge strategies to integrate with existing model.
    """
    try:
        # Import causal model
        import_result = await causal_engine.import_causal_model(
            model_data=request.model_data,
            format=request.format,
            merge_strategy=request.merge_strategy
        )
        
        return {
            "import_status": "success",
            "merge_strategy": request.merge_strategy,
            "imported_counts": {
                "entities_added": import_result.get("entities_added", 0),
                "entities_updated": import_result.get("entities_updated", 0),
                "relationships_added": import_result.get("relationships_added", 0),
                "relationships_updated": import_result.get("relationships_updated", 0)
            },
            "conflicts_resolved": import_result.get("conflicts_resolved", []),
            "import_timestamp": datetime.utcnow().isoformat(),
            "warnings": import_result.get("warnings", [])
        }
        
    except Exception as e:
        logger.error(f"Causal model import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal/validate")
async def validate_causal_relationships(
    request: CausalValidationRequest,
    causal_engine: CausalInferenceEngine = Depends(get_causal_inference_engine)
):
    """
    Validate causal relationships using statistical tests.
    
    Performs comprehensive validation of proposed causal relationships
    using multiple statistical and causal inference methods.
    """
    try:
        # Validate causal relationships
        validation_result = await causal_engine.validate_causal_relationships(
            relationships=request.relationships,
            validation_data=request.validation_data
        )
        
        # Format validation results
        validation_summary = []
        for relationship, result in validation_result["relationship_results"].items():
            validation_summary.append({
                "relationship": relationship,
                "is_valid": result["is_valid"],
                "confidence_score": result["confidence_score"],
                "validation_tests": result["tests_performed"],
                "evidence_strength": result["evidence_strength"],
                "potential_confounders": result.get("potential_confounders", []),
                "recommendations": result.get("recommendations", [])
            })
        
        return {
            "validation_id": validation_result["validation_id"],
            "overall_validity_score": validation_result["overall_score"],
            "relationships_validated": len(validation_summary),
            "valid_relationships": len([r for r in validation_summary if r["is_valid"]]),
            "validation_summary": validation_summary,
            "methodology": validation_result["methodology"],
            "limitations": validation_result.get("limitations", []),
            "recommendations": validation_result.get("overall_recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Causal relationship validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal/insights")
async def generate_causal_insights(
    entity_ids: List[str] = Query(..., description="Entity IDs to analyze"),
    time_window_days: int = Query(30, ge=1, le=365, description="Analysis time window"),
    include_recommendations: bool = Query(True, description="Include actionable recommendations"),
    causal_engine: CausalInferenceEngine = Depends(get_causal_inference_engine)
):
    """
    Generate actionable causal insights for entities.
    
    Provides high-level insights about causal relationships and their
    implications, including actionable recommendations.
    """
    try:
        # Generate causal insights
        insights = await causal_engine.generate_causal_insights(
            entity_ids=entity_ids,
            time_window_days=time_window_days,
            include_recommendations=include_recommendations
        )
        
        return {
            "insights_id": insights["insights_id"],
            "entities_analyzed": entity_ids,
            "key_insights": insights["key_insights"],
            "causal_patterns": insights["causal_patterns"],
            "intervention_opportunities": insights.get("intervention_opportunities", []),
            "risk_factors": insights.get("risk_factors", []),
            "actionable_recommendations": insights.get("recommendations", []) if include_recommendations else [],
            "confidence_assessment": insights["confidence_assessment"],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Causal insights generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _filter_causal_model(model_data: Dict[str, Any], entity_filter: List[str]) -> Dict[str, Any]:
    """Filter causal model to include only specified entities."""
    filtered_model = {
        "format": model_data.get("format", "json"),
        "metadata": model_data.get("metadata", {}),
        "entities": [],
        "relationships": []
    }
    
    # Filter entities
    if "entities" in model_data:
        filtered_model["entities"] = [
            entity for entity in model_data["entities"]
            if entity.get("id") in entity_filter or entity.get("name") in entity_filter
        ]
    
    # Filter relationships to only include those between filtered entities
    if "relationships" in model_data:
        entity_ids = {e.get("id") for e in filtered_model["entities"]}
        entity_names = {e.get("name") for e in filtered_model["entities"]}
        
        filtered_model["relationships"] = [
            rel for rel in model_data["relationships"]
            if (rel.get("source") in entity_ids or rel.get("source") in entity_names) and
               (rel.get("target") in entity_ids or rel.get("target") in entity_names)
        ]
    
    return filtered_model


def _convert_to_graph_data(raw_data: Dict[str, Any]) -> GraphData:
    """Convert raw graph data to visualization format."""
    nodes = []
    edges = []

    # Process nodes
    for node_data in raw_data.get("nodes", []):
        nodes.append(
            GraphNode(
                id=node_data["id"],
                label=node_data.get("name", node_data["id"]),
                type=node_data.get("type", "unknown"),
                properties=node_data.get("properties", {}),
            )
        )

    # Process edges
    for edge_data in raw_data.get("edges", []):
        edges.append(
            GraphEdge(
                id=edge_data["id"],
                source=edge_data["source"],
                target=edge_data["target"],
                label=edge_data.get("type", "related"),
                properties=edge_data.get("properties", {}),
            )
        )

    return GraphData(
        nodes=nodes,
        edges=edges,
        metadata={"node_count": len(nodes), "edge_count": len(edges)},
    )


def _apply_force_layout(graph_data: GraphData) -> GraphData:
    """Apply force-directed layout to graph using spring simulation."""
    import math
    import random
    
    if not graph_data.nodes:
        return graph_data
    
    # Initialize positions randomly if not set
    for node in graph_data.nodes:
        if not hasattr(node, 'x') or node.x is None:
            node.x = random.uniform(-50, 50)
        if not hasattr(node, 'y') or node.y is None:
            node.y = random.uniform(-50, 50)
    
    # Create adjacency information for forces
    adjacency = {}
    for edge in graph_data.edges:
        if edge.source not in adjacency:
            adjacency[edge.source] = []
        if edge.target not in adjacency:
            adjacency[edge.target] = []
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)
    
    # Simulation parameters
    iterations = 50
    dt = 0.1
    spring_strength = 0.5
    repulsion_strength = 100
    damping = 0.9
    
    # Create node index mapping
    node_index = {node.id: i for i, node in enumerate(graph_data.nodes)}
    
    for iteration in range(iterations):
        forces_x = [0.0] * len(graph_data.nodes)
        forces_y = [0.0] * len(graph_data.nodes)
        
        # Calculate repulsive forces between all nodes
        for i, node1 in enumerate(graph_data.nodes):
            for j, node2 in enumerate(graph_data.nodes):
                if i != j:
                    dx = node1.x - node2.x
                    dy = node1.y - node2.y
                    distance = math.sqrt(dx*dx + dy*dy) + 0.01  # Avoid division by zero
                    
                    # Repulsive force (inversely proportional to distance)
                    force = repulsion_strength / (distance * distance)
                    forces_x[i] += force * dx / distance
                    forces_y[i] += force * dy / distance
        
        # Calculate attractive forces for connected nodes
        for edge in graph_data.edges:
            source_idx = node_index.get(edge.source)
            target_idx = node_index.get(edge.target)
            
            if source_idx is not None and target_idx is not None:
                source_node = graph_data.nodes[source_idx]
                target_node = graph_data.nodes[target_idx]
                
                dx = target_node.x - source_node.x
                dy = target_node.y - source_node.y
                distance = math.sqrt(dx*dx + dy*dy) + 0.01
                
                # Spring force (proportional to distance)
                force = spring_strength * distance
                fx = force * dx / distance
                fy = force * dy / distance
                
                forces_x[source_idx] += fx
                forces_y[source_idx] += fy
                forces_x[target_idx] -= fx
                forces_y[target_idx] -= fy
        
        # Apply forces and update positions
        for i, node in enumerate(graph_data.nodes):
            # Apply damping
            forces_x[i] *= damping
            forces_y[i] *= damping
            
            # Update positions
            node.x += forces_x[i] * dt
            node.y += forces_y[i] * dt
            
            # Keep nodes within reasonable bounds
            node.x = max(-200, min(200, node.x))
            node.y = max(-200, min(200, node.y))
    
    return graph_data


def _apply_hierarchical_layout(graph_data: GraphData) -> GraphData:
    """Apply hierarchical layout to graph with proper layering."""
    if not graph_data.nodes:
        return graph_data
    
    # Build adjacency list and find root nodes (nodes with no incoming edges)
    incoming_edges = {node.id: 0 for node in graph_data.nodes}
    outgoing = {node.id: [] for node in graph_data.nodes}
    
    for edge in graph_data.edges:
        incoming_edges[edge.target] = incoming_edges.get(edge.target, 0) + 1
        outgoing[edge.source] = outgoing.get(edge.source, [])
        outgoing[edge.source].append(edge.target)
    
    # Find root nodes (no incoming edges)
    root_nodes = [node_id for node_id, count in incoming_edges.items() if count == 0]
    
    # If no clear roots, pick nodes with highest degree
    if not root_nodes:
        node_degrees = {}
        for edge in graph_data.edges:
            node_degrees[edge.source] = node_degrees.get(edge.source, 0) + 1
            node_degrees[edge.target] = node_degrees.get(edge.target, 0) + 1
        
        if node_degrees:
            max_degree = max(node_degrees.values())
            root_nodes = [node_id for node_id, degree in node_degrees.items() if degree == max_degree][:3]
        else:
            root_nodes = [graph_data.nodes[0].id] if graph_data.nodes else []
    
    # Assign layers using BFS
    layers = {}
    visited = set()
    queue = [(node_id, 0) for node_id in root_nodes]
    
    while queue:
        node_id, layer = queue.pop(0)
        if node_id in visited:
            continue
            
        visited.add(node_id)
        layers[node_id] = layer
        
        # Add children to next layer
        for child_id in outgoing.get(node_id, []):
            if child_id not in visited:
                queue.append((child_id, layer + 1))
    
    # Handle unvisited nodes (disconnected components)
    max_layer = max(layers.values()) if layers else 0
    for node in graph_data.nodes:
        if node.id not in layers:
            max_layer += 1
            layers[node.id] = max_layer
    
    # Organize nodes by layer
    layer_nodes = {}
    for node_id, layer in layers.items():
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node_id)
    
    # Position nodes
    layer_height = 80
    node_spacing = 60
    
    node_positions = {}
    for layer, node_ids in layer_nodes.items():
        y = layer * layer_height
        width = len(node_ids) * node_spacing
        start_x = -width / 2
        
        for i, node_id in enumerate(node_ids):
            x = start_x + (i * node_spacing)
            node_positions[node_id] = (x, y)
    
    # Apply positions to nodes
    for node in graph_data.nodes:
        if node.id in node_positions:
            node.x, node.y = node_positions[node.id]
        else:
            # Fallback position
            node.x = 0
            node.y = 0
    
    return graph_data


def _apply_circular_layout(graph_data: GraphData) -> GraphData:
    """Apply circular layout to graph with smart node grouping."""
    import math
    
    if not graph_data.nodes:
        return graph_data
    
    # Group connected components for better visualization
    components = _find_connected_components(graph_data)
    
    if len(components) == 1:
        # Single component - use one circle
        _position_nodes_in_circle(graph_data.nodes, radius=120, center_x=0, center_y=0)
    else:
        # Multiple components - arrange in concentric circles or separate circles
        if len(components) <= 3:
            # Few components - use separate circles
            component_radius = 80
            circle_spacing = 200
            
            for i, component_nodes in enumerate(components):
                # Calculate position for this component's circle
                angle = (2 * math.pi * i) / len(components)
                center_x = circle_spacing * math.cos(angle)
                center_y = circle_spacing * math.sin(angle)
                
                # Position nodes in this component's circle
                _position_nodes_in_circle(component_nodes, component_radius, center_x, center_y)
        else:
            # Many components - use concentric circles
            base_radius = 60
            radius_increment = 50
            
            for i, component_nodes in enumerate(components):
                radius = base_radius + (i * radius_increment)
                _position_nodes_in_circle(component_nodes, radius, 0, 0)
    
    return graph_data


def _find_connected_components(graph_data: GraphData) -> List[List]:
    """Find connected components in the graph."""
    # Build adjacency list
    adjacency = {}
    for node in graph_data.nodes:
        adjacency[node.id] = []
    
    for edge in graph_data.edges:
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)
    
    visited = set()
    components = []
    
    for node in graph_data.nodes:
        if node.id not in visited:
            # Start new component
            component = []
            queue = [node.id]
            
            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                # Find the actual node object
                current_node = next((n for n in graph_data.nodes if n.id == current_id), None)
                if current_node:
                    component.append(current_node)
                
                # Add neighbors to queue
                for neighbor_id in adjacency.get(current_id, []):
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
            
            if component:
                components.append(component)
    
    return components


def _position_nodes_in_circle(nodes: List, radius: float, center_x: float, center_y: float):
    """Position a list of nodes in a circle."""
    import math
    
    n = len(nodes)
    if n == 0:
        return
    
    if n == 1:
        # Single node at center
        nodes[0].x = center_x
        nodes[0].y = center_y
        return
    
    # Calculate optimal radius based on number of nodes
    # Ensure nodes don't overlap (minimum distance between nodes)
    min_node_distance = 30
    circumference_needed = n * min_node_distance
    min_radius = circumference_needed / (2 * math.pi)
    actual_radius = max(radius, min_radius)
    
    # Position nodes around circle
    for i, node in enumerate(nodes):
        angle = (2 * math.pi * i) / n
        node.x = center_x + actual_radius * math.cos(angle)
        node.y = center_y + actual_radius * math.sin(angle)
