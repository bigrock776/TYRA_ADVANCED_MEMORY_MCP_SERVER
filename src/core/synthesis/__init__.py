"""
Memory Synthesis Module.

This module provides advanced memory synthesis capabilities including:
- Deduplication with semantic similarity
- AI-powered summarization with anti-hallucination
- Cross-memory pattern detection
- Temporal memory evolution analysis

All operations are performed locally with zero external API calls.
"""

from .deduplication import (
    DeduplicationEngine,
    DuplicateGroup,
    DuplicateType,
    MergeStrategy,
    DuplicationMetrics
)

from .summarization import (
    SummarizationEngine,
    MemorySummary,
    SummarizationType,
    SummaryQuality,
    QualityMetrics
)

from .pattern_detector import (
    PatternDetector,
    PatternCluster,
    PatternType,
    ClusteringMethod,
    PatternInsight,
    KnowledgeGap,
    PatternAnalysisResult
)

from .temporal_analysis import (
    TemporalAnalyzer,
    ConceptEvolution,
    EvolutionType,
    TrendDirection,
    TemporalInsight,
    LearningProgression,
    TemporalAnalysisResult
)

__all__ = [
    # Deduplication
    "DeduplicationEngine",
    "DuplicateGroup", 
    "DuplicateType",
    "MergeStrategy",
    "DuplicationMetrics",
    # Summarization
    "SummarizationEngine",
    "MemorySummary",
    "SummarizationType",
    "SummaryQuality",
    "QualityMetrics",
    # Pattern Detection
    "PatternDetector",
    "PatternCluster",
    "PatternType",
    "ClusteringMethod",
    "PatternInsight",
    "KnowledgeGap",
    "PatternAnalysisResult",
    # Temporal Analysis
    "TemporalAnalyzer",
    "ConceptEvolution",
    "EvolutionType",
    "TrendDirection",
    "TemporalInsight",
    "LearningProgression",
    "TemporalAnalysisResult"
]