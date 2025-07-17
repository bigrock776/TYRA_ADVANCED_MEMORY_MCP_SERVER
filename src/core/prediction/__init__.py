"""
Predictive Memory Management Module.

This module provides comprehensive predictive analytics for memory management including:
- Usage pattern analysis with local ML clustering
- Smart auto-archiving with local scoring algorithms
- Predictive preloading using Markov chains and n-gram models
- Memory lifecycle optimization with local analytics

All processing is performed locally with zero external API calls.
"""

from .usage_analyzer import (
    UsageAnalyzer,
    UsagePattern,
    AccessPattern,
    UserBehaviorModel,
    PopularityScore,
    PatternType,
    AccessFrequency
)

from .auto_archiver import (
    AutoArchiver,
    ArchivingPolicy,
    ImportanceScore,
    ArchivingRule,
    ArchivingAction,
    PolicyEngine,
    RestoreTrigger
)

from .preloader import (
    PredictivePreloader,
    CachePrediction,
    QueryAnticipator,
    MarkovChainPredictor,
    PreloadingStrategy,
    PriorityQueue,
    SuccessTracker
)

from .lifecycle import (
    MemoryLifecycleOptimizer,
    LifecycleStage,
    TransitionPrediction,
    OptimizationRecommendation,
    PerformanceMetrics,
    AgingAlgorithm
)

__all__ = [
    # Usage analysis
    "UsageAnalyzer",
    "UsagePattern",
    "AccessPattern",
    "UserBehaviorModel",
    "PopularityScore",
    "PatternType",
    "AccessFrequency",
    
    # Auto-archiving
    "AutoArchiver",
    "ArchivingPolicy",
    "ImportanceScore",
    "ArchivingRule",
    "ArchivingAction",
    "PolicyEngine",
    "RestoreTrigger",
    
    # Predictive preloading
    "PredictivePreloader",
    "CachePrediction",
    "QueryAnticipator",
    "MarkovChainPredictor",
    "PreloadingStrategy",
    "PriorityQueue",
    "SuccessTracker",
    
    # Lifecycle optimization
    "MemoryLifecycleOptimizer",
    "LifecycleStage",
    "TransitionPrediction",
    "OptimizationRecommendation",
    "PerformanceMetrics",
    "AgingAlgorithm"
]