"""
Core embeddings module for centralized embedding operations.

This module provides a unified interface for all embedding operations,
coordinating between different providers and implementing shared functionality,
including context-aware embeddings, multi-perspective embeddings, dynamic fine-tuning,
and advanced embedding fusion.
"""

from .manager import EmbeddingManager
from .models import EmbeddingRequest, EmbeddingResponse

# Context-aware embeddings
from .contextual import (
    SessionAwareEmbedder,
    ContextualEmbedding,
    SessionContext,
    ContextType,
    FusionStrategy as ContextFusionStrategy,
    ContextDecayModel,
    ContextVectorFusion
)

# Multi-perspective embeddings
from .multi_perspective import (
    MultiPerspectiveEmbedder,
    MultiPerspectiveResult,
    PerspectiveEmbedding,
    PerspectiveModel,
    PerspectiveType,
    RoutingStrategy,
    ConfidenceMethod,
    PerspectiveRouter,
    ConfidenceEstimator
)

# Dynamic fine-tuning
from .fine_tuning import (
    DynamicFineTuner,
    FineTuningConfig,
    FineTuningStrategy,
    LearningObjective,
    TrainingExample,
    ModelVersion,
    ModelVersionManager,
    OnlineLearningOptimizer
)

# Embedding fusion
from .fusion import (
    EmbeddingFusionEngine,
    FusionResult,
    EmbeddingSource,
    FusionConfig,
    FusionStrategy,
    QualityMetric,
    QualityAssessor,
    AdaptiveFusionOptimizer,
    AttentionFusionNetwork
)

__all__ = [
    # Core components
    "EmbeddingManager",
    "EmbeddingRequest", 
    "EmbeddingResponse",
    
    # Context-aware embeddings
    "SessionAwareEmbedder",
    "ContextualEmbedding",
    "SessionContext",
    "ContextType",
    "ContextFusionStrategy",
    "ContextDecayModel",
    "ContextVectorFusion",
    
    # Multi-perspective embeddings
    "MultiPerspectiveEmbedder",
    "MultiPerspectiveResult",
    "PerspectiveEmbedding",
    "PerspectiveModel",
    "PerspectiveType",
    "RoutingStrategy",
    "ConfidenceMethod", 
    "PerspectiveRouter",
    "ConfidenceEstimator",
    
    # Dynamic fine-tuning
    "DynamicFineTuner",
    "FineTuningConfig",
    "FineTuningStrategy",
    "LearningObjective",
    "TrainingExample",
    "ModelVersion",
    "ModelVersionManager",
    "OnlineLearningOptimizer",
    
    # Embedding fusion
    "EmbeddingFusionEngine",
    "FusionResult",
    "EmbeddingSource",
    "FusionConfig",
    "FusionStrategy",
    "QualityMetric",
    "QualityAssessor",
    "AdaptiveFusionOptimizer",
    "AttentionFusionNetwork"
]