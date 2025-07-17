"""
Core graph module for temporal knowledge graph operations.

This module provides graph engine interfaces and implementations for
managing temporal knowledge graphs with advanced entity relationship
tracking and semantic search capabilities.
"""

from .graphiti_integration import GraphitiManager
from .neo4j_client import Neo4jClient
from .reasoning_engine import (
    MultiHopReasoningEngine,
    ReasoningType,
    ReasoningStrategy,
    ConfidenceLevel,
    ReasoningNode,
    ReasoningEdge,
    ReasoningPath,
    ReasoningQuery,
    ReasoningResult,
    PathScorer,
    ExplanationGenerator
)
from .causal_inference import (
    CausalInferenceEngine,
    CausalRelationType,
    CausalInferenceMethod,
    CausalEvidence,
    CausalClaim,
    CausalGraph,
    GrangerCausalityAnalyzer,
    PearlCausalAnalyzer
)
from .temporal_evolution import (
    TemporalKnowledgeEvolutionEngine,
    EvolutionType,
    ChangeDetectionMethod,
    TemporalGranularity,
    TemporalSnapshot,
    EvolutionEvent,
    TemporalAlignment,
    ConceptDriftDetector,
    TemporalEmbeddingAligner,
    EvolutionPatternDetector
)
from .recommender import (
    GraphBasedRecommendationEngine,
    RecommendationType,
    RecommendationStrategy,
    RecommendationItem,
    RecommendationSet,
    UserProfile,
    ContentBasedRecommender,
    GraphEmbeddingRecommender,
    DiversityOptimizer
)

__all__ = [
    # Core graph components
    "GraphitiManager", 
    "Neo4jClient",
    
    # Multi-hop reasoning
    "MultiHopReasoningEngine",
    "ReasoningType",
    "ReasoningStrategy", 
    "ConfidenceLevel",
    "ReasoningNode",
    "ReasoningEdge",
    "ReasoningPath",
    "ReasoningQuery",
    "ReasoningResult",
    "PathScorer",
    "ExplanationGenerator",
    
    # Causal inference
    "CausalInferenceEngine",
    "CausalRelationType",
    "CausalInferenceMethod",
    "CausalEvidence",
    "CausalClaim",
    "CausalGraph",
    "GrangerCausalityAnalyzer",
    "PearlCausalAnalyzer",
    
    # Temporal evolution
    "TemporalKnowledgeEvolutionEngine",
    "EvolutionType",
    "ChangeDetectionMethod",
    "TemporalGranularity",
    "TemporalSnapshot",
    "EvolutionEvent",
    "TemporalAlignment",
    "ConceptDriftDetector",
    "TemporalEmbeddingAligner",
    "EvolutionPatternDetector",
    
    # Recommendation engine
    "GraphBasedRecommendationEngine",
    "RecommendationType",
    "RecommendationStrategy",
    "RecommendationItem",
    "RecommendationSet",
    "UserProfile",
    "ContentBasedRecommender",
    "GraphEmbeddingRecommender",
    "DiversityOptimizer"
]