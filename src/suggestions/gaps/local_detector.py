"""
Local Missing Knowledge Detection System.

This module provides intelligent detection of knowledge gaps using local
topic modeling, prioritization algorithms, filling suggestions, resolution
tracking, and impact assessment for comprehensive knowledge management.
"""

import asyncio
import math
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict, Counter
import spacy
import re

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ...core.embeddings.embedder import Embedder
from ...core.memory.manager import MemoryManager
from ...core.graph.neo4j_client import Neo4jClient
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class GapType(str, Enum):
    """Types of knowledge gaps that can be detected."""
    CONCEPTUAL_GAP = "conceptual_gap"          # Missing core concepts
    PROCEDURAL_GAP = "procedural_gap"          # Missing how-to knowledge
    FACTUAL_GAP = "factual_gap"                # Missing factual information
    CONTEXTUAL_GAP = "contextual_gap"          # Missing context or background
    RELATIONSHIP_GAP = "relationship_gap"      # Missing connections between concepts
    TEMPORAL_GAP = "temporal_gap"              # Missing time-based progression
    DOMAIN_GAP = "domain_gap"                  # Missing domain knowledge
    PREREQUISITE_GAP = "prerequisite_gap"      # Missing foundational knowledge
    BRIDGING_GAP = "bridging_gap"              # Missing connecting knowledge
    UPDATE_GAP = "update_gap"                  # Outdated knowledge needs refresh


class Priority(str, Enum):
    """Priority levels for addressing knowledge gaps."""
    CRITICAL = "critical"      # Blocks understanding/progress
    HIGH = "high"             # Significantly impacts effectiveness
    MEDIUM = "medium"         # Moderately useful to fill
    LOW = "low"               # Nice to have
    OPTIONAL = "optional"     # Minimal impact


class ImpactLevel(str, Enum):
    """Expected impact of filling the knowledge gap."""
    TRANSFORMATIVE = "transformative"  # Major improvement in understanding
    SIGNIFICANT = "significant"        # Notable improvement
    MODERATE = "moderate"              # Some improvement
    MINIMAL = "minimal"                # Small improvement
    UNCERTAIN = "uncertain"            # Impact unclear


class FillingSuggestion(BaseModel):
    """Suggestion for how to fill a knowledge gap."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    suggestion_type: str = Field(..., description="Type of filling suggestion")
    description: str = Field(..., min_length=10, description="Detailed suggestion description")
    resources: List[str] = Field(..., description="Suggested resources or methods")
    effort_estimate: str = Field(..., description="Estimated effort to implement")
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of success")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites for implementation")


@dataclass
class GapDetectionContext:
    """Context for knowledge gap detection."""
    user_id: str
    analysis_scope: str = "all"  # "all", "domain", "recent", "frequent"
    domain_focus: Optional[str] = None
    time_window_days: int = 90
    min_gap_significance: float = 0.3
    max_gaps: int = 20
    include_minor_gaps: bool = False
    prioritize_actionable: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)


class KnowledgeGap(BaseModel):
    """Detected knowledge gap with comprehensive analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    gap_id: str = Field(..., description="Unique gap identifier")
    gap_type: GapType = Field(..., description="Type of knowledge gap")
    title: str = Field(..., min_length=5, description="Clear gap title")
    description: str = Field(..., min_length=20, description="Detailed gap description")
    
    # Priority and impact
    priority: Priority = Field(..., description="Gap priority level")
    impact_level: ImpactLevel = Field(..., description="Expected impact of filling gap")
    significance_score: float = Field(..., ge=0.0, le=1.0, description="Gap significance score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in gap detection")
    
    # Gap details
    missing_concepts: List[str] = Field(..., description="Specific missing concepts")
    related_memories: List[str] = Field(..., description="Memory IDs related to this gap")
    knowledge_cluster: Optional[str] = Field(None, description="Knowledge cluster this gap belongs to")
    domain_area: str = Field(..., description="Domain or area of knowledge")
    
    # Context and evidence
    gap_indicators: List[str] = Field(..., description="Evidence indicating this gap")
    surrounding_knowledge: Dict[str, List[str]] = Field(..., description="Related knowledge that exists")
    knowledge_prerequisites: List[str] = Field(..., description="Prerequisites for understanding")
    
    # Filling suggestions
    filling_suggestions: List[FillingSuggestion] = Field(..., description="Suggestions for filling gap")
    learning_path: List[str] = Field(..., description="Suggested learning sequence")
    estimated_effort: str = Field(..., description="Estimated effort to fill gap")
    
    # Tracking and metrics
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
    last_analyzed: datetime = Field(default_factory=datetime.utcnow, description="Last analysis timestamp")
    fill_attempts: List[Dict[str, Any]] = Field(default_factory=list, description="Attempts to fill gap")
    resolution_status: str = Field("open", description="Gap resolution status")
    
    # Metadata
    tags: Set[str] = Field(default_factory=set, description="Gap tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GapDetectionResult(BaseModel):
    """Complete gap detection analysis result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    gaps: List[KnowledgeGap] = Field(..., description="Detected knowledge gaps")
    total_memories_analyzed: int = Field(..., ge=0, description="Total memories analyzed")
    processing_time_ms: float = Field(..., ge=0.0, description="Analysis processing time")
    detection_strategies: List[str] = Field(..., description="Detection strategies used")
    
    # Analysis metrics
    knowledge_coverage: Dict[str, float] = Field(..., description="Knowledge coverage by domain")
    gap_distribution: Dict[str, int] = Field(..., description="Distribution of gap types")
    priority_breakdown: Dict[str, int] = Field(..., description="Priority level breakdown")
    domain_analysis: Dict[str, Any] = Field(..., description="Domain-specific analysis")
    
    # Actionable insights
    immediate_actions: List[str] = Field(..., description="Immediate actions to take")
    learning_recommendations: List[str] = Field(..., description="Learning recommendations")
    knowledge_goals: List[str] = Field(..., description="Suggested knowledge goals")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")


class LocalKnowledgeGapDetector:
    """
    Local Missing Knowledge Detection System.
    
    Intelligently detects knowledge gaps using topic modeling, clustering,
    relationship analysis, and domain knowledge assessment with local algorithms.
    """
    
    def __init__(
        self,
        embedder: Embedder,
        memory_manager: MemoryManager,
        graph_client: Optional[Neo4jClient] = None
    ):
        """Initialize local knowledge gap detector."""
        self.embedder = embedder
        self.memory_manager = memory_manager
        self.graph_client = graph_client
        
        # Analysis components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=15,
            random_state=42,
            max_iter=20
        )
        self.nmf_model = NMF(
            n_components=15,
            random_state=42,
            max_iter=200
        )
        self.scaler = StandardScaler()
        
        # Knowledge domain definitions
        self.knowledge_domains = {
            "technology": ["programming", "software", "hardware", "algorithm", "database", "api", "framework"],
            "business": ["strategy", "marketing", "finance", "management", "operations", "sales", "customer"],
            "science": ["research", "experiment", "theory", "analysis", "methodology", "hypothesis", "data"],
            "education": ["learning", "teaching", "course", "curriculum", "skill", "training", "knowledge"],
            "health": ["medicine", "treatment", "diagnosis", "therapy", "wellness", "prevention", "health"],
            "creative": ["design", "art", "writing", "creative", "innovation", "brainstorming", "inspiration"],
            "personal": ["goal", "habit", "productivity", "organization", "planning", "reflection", "growth"]
        }
        
        # Gap detection patterns
        self.gap_indicators = {
            "incomplete_sentences": [r"need to learn", r"don't understand", r"unclear about", r"missing information"],
            "question_patterns": [r"\?$", r"^how", r"^what", r"^why", r"^when", r"^where", r"^who"],
            "uncertainty_markers": ["maybe", "perhaps", "not sure", "unclear", "confused", "don't know"],
            "follow_up_needed": ["research more", "look into", "investigate", "find out", "learn about"],
            "prerequisite_mentions": ["need to understand", "first need", "before", "prerequisite", "foundation"]
        }
        
        # NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic NLP")
            self.nlp = None
        
        # Caching
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.topic_cache: Dict[str, Dict[str, Any]] = {}
        self.domain_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'gaps_detected': 0,
            'gaps_filled': 0,
            'average_processing_time': 0.0,
            'detection_by_type': {gtype.value: 0 for gtype in GapType},
            'priority_distribution': {priority.value: 0 for priority in Priority},
            'accuracy_feedback': []
        }
        
        logger.info("Initialized LocalKnowledgeGapDetector")
    
    async def detect_knowledge_gaps(
        self,
        context: GapDetectionContext
    ) -> GapDetectionResult:
        """
        Detect knowledge gaps based on context and analysis.
        
        Args:
            context: Gap detection context and parameters
            
        Returns:
            Comprehensive gap detection results with actionable insights
        """
        start_time = datetime.utcnow()
        self.detection_stats['total_detections'] += 1
        
        try:
            # Get memories for analysis
            memories = await self._get_memories_for_analysis(context)
            
            if len(memories) < 5:
                return GapDetectionResult(
                    gaps=[],
                    total_memories_analyzed=len(memories),
                    processing_time_ms=0.0,
                    detection_strategies=[],
                    knowledge_coverage={},
                    gap_distribution={},
                    priority_breakdown={},
                    domain_analysis={"status": "insufficient_memories"},
                    immediate_actions=["Add more memories to enable gap detection"],
                    learning_recommendations=[],
                    knowledge_goals=[]
                )
            
            # Perform gap detection using multiple strategies
            gaps = []
            strategies_used = []
            
            # Strategy 1: Topic-based gap detection
            topic_gaps = await self._detect_topic_gaps(memories, context)
            gaps.extend(topic_gaps)
            if topic_gaps:
                strategies_used.append("topic_analysis")
            
            # Strategy 2: Conceptual gap detection
            conceptual_gaps = await self._detect_conceptual_gaps(memories, context)
            gaps.extend(conceptual_gaps)
            if conceptual_gaps:
                strategies_used.append("conceptual_analysis")
            
            # Strategy 3: Relationship gap detection
            relationship_gaps = await self._detect_relationship_gaps(memories, context)
            gaps.extend(relationship_gaps)
            if relationship_gaps:
                strategies_used.append("relationship_analysis")
            
            # Strategy 4: Domain knowledge gap detection
            domain_gaps = await self._detect_domain_gaps(memories, context)
            gaps.extend(domain_gaps)
            if domain_gaps:
                strategies_used.append("domain_analysis")
            
            # Strategy 5: Procedural gap detection
            procedural_gaps = await self._detect_procedural_gaps(memories, context)
            gaps.extend(procedural_gaps)
            if procedural_gaps:
                strategies_used.append("procedural_analysis")
            
            # Strategy 6: Temporal/sequence gap detection
            temporal_gaps = await self._detect_temporal_gaps(memories, context)
            gaps.extend(temporal_gaps)
            if temporal_gaps:
                strategies_used.append("temporal_analysis")
            
            # Strategy 7: Prerequisite gap detection
            prerequisite_gaps = await self._detect_prerequisite_gaps(memories, context)
            gaps.extend(prerequisite_gaps)
            if prerequisite_gaps:
                strategies_used.append("prerequisite_analysis")
            
            # Strategy 8: Update/currency gap detection
            update_gaps = await self._detect_update_gaps(memories, context)
            gaps.extend(update_gaps)
            if update_gaps:
                strategies_used.append("currency_analysis")
            
            # Remove duplicates and filter by significance
            unique_gaps = self._deduplicate_gaps(gaps)
            significant_gaps = [
                gap for gap in unique_gaps 
                if gap.significance_score >= context.min_gap_significance
            ]
            
            # Prioritize and rank gaps
            prioritized_gaps = self._prioritize_gaps(significant_gaps, context)
            
            # Enhance gaps with filling suggestions
            enhanced_gaps = await self._enhance_gaps_with_suggestions(
                prioritized_gaps, memories, context
            )
            
            # Limit to max gaps
            final_gaps = enhanced_gaps[:context.max_gaps]
            
            # Calculate comprehensive metrics
            knowledge_coverage = await self._calculate_knowledge_coverage(memories, final_gaps)
            gap_distribution = Counter(gap.gap_type for gap in final_gaps)
            priority_breakdown = Counter(gap.priority for gap in final_gaps)
            domain_analysis = await self._analyze_domain_coverage(memories, final_gaps)
            
            # Generate actionable insights
            immediate_actions = self._generate_immediate_actions(final_gaps)
            learning_recommendations = self._generate_learning_recommendations(final_gaps, memories)
            knowledge_goals = self._generate_knowledge_goals(final_gaps, domain_analysis)
            
            # Update statistics
            for gap in final_gaps:
                self.detection_stats['detection_by_type'][gap.gap_type.value] += 1
                self.detection_stats['priority_distribution'][gap.priority.value] += 1
            
            self.detection_stats['gaps_detected'] += len(final_gaps)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.detection_stats['average_processing_time'] = (
                (self.detection_stats['average_processing_time'] * 
                 (self.detection_stats['total_detections'] - 1) + processing_time) /
                self.detection_stats['total_detections']
            )
            
            logger.info(
                "Detected knowledge gaps",
                user_id=context.user_id,
                gaps_count=len(final_gaps),
                memories_analyzed=len(memories),
                processing_time_ms=processing_time,
                strategies=strategies_used
            )
            
            return GapDetectionResult(
                gaps=final_gaps,
                total_memories_analyzed=len(memories),
                processing_time_ms=processing_time,
                detection_strategies=strategies_used,
                knowledge_coverage=knowledge_coverage,
                gap_distribution=dict(gap_distribution),
                priority_breakdown=dict(priority_breakdown),
                domain_analysis=domain_analysis,
                immediate_actions=immediate_actions,
                learning_recommendations=learning_recommendations,
                knowledge_goals=knowledge_goals
            )
            
        except Exception as e:
            logger.error("Error detecting knowledge gaps", error=str(e))
            return GapDetectionResult(
                gaps=[],
                total_memories_analyzed=0,
                processing_time_ms=0.0,
                detection_strategies=[],
                knowledge_coverage={},
                gap_distribution={},
                priority_breakdown={},
                domain_analysis={"error": str(e)},
                immediate_actions=["System error - please try again"],
                learning_recommendations=[],
                knowledge_goals=[]
            )
    
    async def _get_memories_for_analysis(self, context: GapDetectionContext) -> List[Dict[str, Any]]:
        """Get memories for gap detection analysis."""
        try:
            if context.analysis_scope == "recent":
                cutoff_date = datetime.utcnow() - timedelta(days=context.time_window_days)
                memories = await self.memory_manager.get_memories_for_user(
                    context.user_id,
                    created_after=cutoff_date,
                    limit=300
                )
            elif context.analysis_scope == "domain" and context.domain_focus:
                # Filter by domain - simplified implementation
                memories = await self.memory_manager.get_memories_for_user(
                    context.user_id,
                    limit=300
                )
                # Filter by domain keywords
                domain_keywords = self.knowledge_domains.get(context.domain_focus, [])
                filtered_memories = []
                for memory in memories:
                    content_lower = memory.content.lower()
                    if any(keyword in content_lower for keyword in domain_keywords):
                        filtered_memories.append(memory)
                memories = filtered_memories
            else:
                memories = await self.memory_manager.get_memories_for_user(
                    context.user_id,
                    limit=300
                )
            
            # Convert to analysis format
            memory_list = []
            for memory in memories:
                memory_dict = {
                    "id": memory.id,
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "tags": getattr(memory, 'tags', []),
                    "entities": getattr(memory, 'entities', [])
                }
                memory_list.append(memory_dict)
            
            logger.debug(f"Retrieved {len(memory_list)} memories for gap detection")
            return memory_list
            
        except Exception as e:
            logger.error("Error retrieving memories for gap analysis", error=str(e))
            return []
    
    async def _detect_topic_gaps(
        self,
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Detect gaps in topic coverage using topic modeling."""
        gaps = []
        
        try:
            if len(memories) < 10:
                return gaps
            
            # Extract text content
            texts = [memory["content"] for memory in memories]
            
            # Perform topic modeling
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Use both LDA and NMF for topic discovery
            lda_topics = self.lda_model.fit_transform(tfidf_matrix)
            nmf_topics = self.nmf_model.fit_transform(tfidf_matrix)
            
            # Analyze topic coverage and identify gaps
            topic_analysis = self._analyze_topic_coverage(
                lda_topics, nmf_topics, texts, memories
            )
            
            # Identify potential topic gaps
            if topic_analysis["coverage_gaps"]:
                for gap_info in topic_analysis["coverage_gaps"]:
                    gap = self._create_topic_gap(gap_info, memories, context)
                    gaps.append(gap)
            
        except Exception as e:
            logger.error("Error in topic gap detection", error=str(e))
        
        return gaps
    
    def _analyze_topic_coverage(
        self,
        lda_topics: np.ndarray,
        nmf_topics: np.ndarray,
        texts: List[str],
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze topic coverage to identify gaps."""
        # Get feature names (topics/keywords)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Analyze LDA topics
        lda_topic_words = []
        for topic_idx in range(self.lda_model.n_components):
            topic_words = [
                feature_names[i] for i in 
                self.lda_model.components_[topic_idx].argsort()[:-10-1:-1]
            ]
            lda_topic_words.append(topic_words)
        
        # Analyze NMF topics
        nmf_topic_words = []
        for topic_idx in range(self.nmf_model.n_components):
            topic_words = [
                feature_names[i] for i in 
                self.nmf_model.components_[topic_idx].argsort()[:-10-1:-1]
            ]
            nmf_topic_words.append(topic_words)
        
        # Identify topic coverage patterns
        topic_strengths = np.mean(lda_topics, axis=0)
        weak_topics = [i for i, strength in enumerate(topic_strengths) if strength < 0.05]
        
        # Identify potential coverage gaps
        coverage_gaps = []
        for topic_idx in weak_topics:
            if topic_idx < len(lda_topic_words):
                topic_words = lda_topic_words[topic_idx]
                # Check if this represents a genuine knowledge gap
                if self._is_significant_topic_gap(topic_words, texts):
                    coverage_gaps.append({
                        "topic_index": topic_idx,
                        "topic_words": topic_words,
                        "strength": topic_strengths[topic_idx],
                        "gap_type": "topic_coverage"
                    })
        
        # Look for cross-topic connections that might be missing
        cross_topic_gaps = self._identify_cross_topic_gaps(lda_topic_words, texts)
        coverage_gaps.extend(cross_topic_gaps)
        
        return {
            "lda_topics": lda_topic_words,
            "nmf_topics": nmf_topic_words,
            "topic_strengths": topic_strengths.tolist(),
            "weak_topics": weak_topics,
            "coverage_gaps": coverage_gaps,
            "topic_diversity": len([s for s in topic_strengths if s > 0.02]) / len(topic_strengths)
        }
    
    def _is_significant_topic_gap(self, topic_words: List[str], texts: List[str]) -> bool:
        """Determine if a weak topic represents a significant knowledge gap."""
        # Check if topic words appear in questions or uncertainty contexts
        topic_mentions = 0
        gap_indicators = 0
        
        for text in texts:
            text_lower = text.lower()
            
            # Count topic word mentions
            for word in topic_words:
                if word in text_lower:
                    topic_mentions += 1
            
            # Look for gap indicators around topic words
            for word in topic_words:
                if word in text_lower:
                    # Check for uncertainty markers near this word
                    word_index = text_lower.find(word)
                    context = text_lower[max(0, word_index-50):word_index+50]
                    
                    for pattern_list in self.gap_indicators.values():
                        for pattern in pattern_list:
                            if re.search(pattern, context):
                                gap_indicators += 1
                                break
        
        # Significant gap if topic is mentioned but with uncertainty
        return gap_indicators > 0 and (gap_indicators / max(topic_mentions, 1)) > 0.3
    
    def _identify_cross_topic_gaps(
        self,
        topic_words_list: List[List[str]],
        texts: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify gaps in cross-topic connections."""
        cross_gaps = []
        
        # Look for topics that should be connected but aren't
        for i, topic1_words in enumerate(topic_words_list):
            for j, topic2_words in enumerate(topic_words_list[i+1:], i+1):
                # Check if these topics have natural connections
                connection_strength = self._calculate_topic_connection_strength(
                    topic1_words, topic2_words, texts
                )
                
                if connection_strength < 0.2:  # Weak connection
                    # Check if this represents a knowledge gap
                    if self._should_be_connected(topic1_words, topic2_words):
                        cross_gaps.append({
                            "topic_indices": [i, j],
                            "topic1_words": topic1_words,
                            "topic2_words": topic2_words,
                            "connection_strength": connection_strength,
                            "gap_type": "cross_topic_connection"
                        })
        
        return cross_gaps
    
    def _calculate_topic_connection_strength(
        self,
        topic1_words: List[str],
        topic2_words: List[str],
        texts: List[str]
    ) -> float:
        """Calculate connection strength between two topics."""
        co_occurrences = 0
        total_mentions = 0
        
        for text in texts:
            text_lower = text.lower()
            
            topic1_mentioned = any(word in text_lower for word in topic1_words)
            topic2_mentioned = any(word in text_lower for word in topic2_words)
            
            if topic1_mentioned or topic2_mentioned:
                total_mentions += 1
                
                if topic1_mentioned and topic2_mentioned:
                    co_occurrences += 1
        
        return co_occurrences / max(total_mentions, 1)
    
    def _should_be_connected(self, topic1_words: List[str], topic2_words: List[str]) -> bool:
        """Determine if two topics should naturally be connected."""
        # Simple heuristic: check for semantic similarity or domain relationships
        
        # Check for domain overlap
        topic1_domains = set()
        topic2_domains = set()
        
        for domain, keywords in self.knowledge_domains.items():
            if any(word in keywords for word in topic1_words):
                topic1_domains.add(domain)
            if any(word in keywords for word in topic2_words):
                topic2_domains.add(domain)
        
        # Should be connected if in same domain
        return bool(topic1_domains.intersection(topic2_domains))
    
    def _create_topic_gap(
        self,
        gap_info: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> KnowledgeGap:
        """Create a topic-based knowledge gap."""
        gap_id = hashlib.md5(f"topic_{gap_info['topic_index']}_{context.user_id}".encode()).hexdigest()[:16]
        
        topic_words = gap_info["topic_words"]
        strength = gap_info["strength"]
        
        if gap_info["gap_type"] == "topic_coverage":
            title = f"Missing Knowledge in {topic_words[0].title()} Area"
            description = f"Topic analysis reveals insufficient coverage of {', '.join(topic_words[:3])} concepts."
            gap_type = GapType.CONCEPTUAL_GAP
        else:
            title = f"Missing Connections Between {topic_words[0].title()} Topics"
            description = f"Weak connections detected between {', '.join(topic_words[:2])} concepts."
            gap_type = GapType.RELATIONSHIP_GAP
        
        # Determine priority based on strength and domain
        if strength < 0.02:
            priority = Priority.HIGH
            significance = 0.8
        elif strength < 0.04:
            priority = Priority.MEDIUM
            significance = 0.6
        else:
            priority = Priority.LOW
            significance = 0.4
        
        # Find related memories
        related_memories = []
        for memory in memories:
            content_lower = memory["content"].lower()
            if any(word in content_lower for word in topic_words):
                related_memories.append(memory["id"])
        
        # Determine domain
        domain = self._identify_domain(topic_words)
        
        # Generate filling suggestions
        filling_suggestions = self._generate_topic_filling_suggestions(topic_words, gap_type)
        
        # Generate learning path
        learning_path = [
            f"Research foundational concepts in {topic_words[0]}",
            f"Study relationships between {topic_words[0]} and {topic_words[1] if len(topic_words) > 1 else 'related concepts'}",
            f"Practice applying {topic_words[0]} concepts",
            f"Connect {topic_words[0]} to existing knowledge"
        ]
        
        return KnowledgeGap(
            gap_id=gap_id,
            gap_type=gap_type,
            title=title,
            description=description,
            priority=priority,
            impact_level=ImpactLevel.MODERATE,
            significance_score=significance,
            confidence=0.7,
            missing_concepts=topic_words[:5],
            related_memories=related_memories[:10],
            domain_area=domain,
            gap_indicators=[f"Weak topic coverage (strength: {strength:.3f})"],
            surrounding_knowledge={"related_topics": [w for w in topic_words if w not in topic_words[:5]]},
            knowledge_prerequisites=[f"Basic understanding of {topic_words[0]}"],
            filling_suggestions=filling_suggestions,
            learning_path=learning_path,
            estimated_effort="2-4 weeks",
            tags={"topic", "conceptual", domain}
        )
    
    def _identify_domain(self, words: List[str]) -> str:
        """Identify the domain for a set of words."""
        domain_scores = {}
        
        for domain, keywords in self.knowledge_domains.items():
            score = sum(1 for word in words if word in keywords)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"
    
    def _generate_topic_filling_suggestions(
        self,
        topic_words: List[str],
        gap_type: GapType
    ) -> List[FillingSuggestion]:
        """Generate suggestions for filling topic gaps."""
        suggestions = []
        
        primary_concept = topic_words[0]
        
        # Research suggestion
        suggestions.append(FillingSuggestion(
            suggestion_type="research",
            description=f"Conduct comprehensive research on {primary_concept} and related concepts",
            resources=[
                f"Search for authoritative sources on {primary_concept}",
                f"Find academic papers or books about {primary_concept}",
                f"Look for expert explanations and tutorials"
            ],
            effort_estimate="1-2 weeks",
            success_probability=0.8,
            prerequisites=["Access to research resources", "Basic reading comprehension"]
        ))
        
        # Practical application suggestion
        suggestions.append(FillingSuggestion(
            suggestion_type="practical_application",
            description=f"Find practical applications or examples of {primary_concept}",
            resources=[
                f"Look for case studies involving {primary_concept}",
                f"Find hands-on exercises or projects",
                f"Connect with practitioners in the field"
            ],
            effort_estimate="2-3 weeks",
            success_probability=0.7,
            prerequisites=[f"Basic understanding of {primary_concept}"]
        ))
        
        # Expert consultation suggestion
        if gap_type in [GapType.CONCEPTUAL_GAP, GapType.DOMAIN_GAP]:
            suggestions.append(FillingSuggestion(
                suggestion_type="expert_consultation",
                description=f"Consult with experts or take structured learning about {primary_concept}",
                resources=[
                    f"Find online courses about {primary_concept}",
                    f"Join communities or forums focused on {primary_concept}",
                    f"Seek mentorship or professional guidance"
                ],
                effort_estimate="4-8 weeks",
                success_probability=0.9,
                prerequisites=["Time commitment", "Learning motivation"]
            ))
        
        return suggestions
    
    async def _detect_conceptual_gaps(
        self,
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Detect conceptual knowledge gaps."""
        gaps = []
        
        try:
            # Look for explicit gap indicators in content
            for memory in memories:
                content = memory["content"]
                
                # Check for uncertainty markers
                uncertainty_count = 0
                for pattern_list in self.gap_indicators.values():
                    for pattern in pattern_list:
                        matches = len(re.findall(pattern, content.lower()))
                        uncertainty_count += matches
                
                # If memory has many uncertainty markers, analyze for gaps
                if uncertainty_count >= 3:
                    conceptual_gaps = await self._analyze_memory_for_conceptual_gaps(
                        memory, memories, context
                    )
                    gaps.extend(conceptual_gaps)
                    
        except Exception as e:
            logger.error("Error in conceptual gap detection", error=str(e))
        
        return gaps
    
    async def _analyze_memory_for_conceptual_gaps(
        self,
        memory: Dict[str, Any],
        all_memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Analyze a specific memory for conceptual gaps."""
        gaps = []
        
        try:
            content = memory["content"]
            
            # Extract concepts mentioned with uncertainty
            uncertain_concepts = self._extract_uncertain_concepts(content)
            
            for concept in uncertain_concepts:
                # Check if this concept is adequately covered elsewhere
                coverage = self._assess_concept_coverage(concept, all_memories)
                
                if coverage < 0.3:  # Insufficient coverage
                    gap = self._create_conceptual_gap(
                        concept, memory, coverage, context
                    )
                    gaps.append(gap)
                    
        except Exception as e:
            logger.warning(f"Error analyzing memory for conceptual gaps: {e}")
        
        return gaps
    
    def _extract_uncertain_concepts(self, content: str) -> List[str]:
        """Extract concepts mentioned with uncertainty markers."""
        uncertain_concepts = []
        content_lower = content.lower()
        
        # Look for patterns like "not sure about X", "unclear on Y", etc.
        uncertainty_patterns = [
            r"not sure about ([a-zA-Z\s]+)",
            r"unclear on ([a-zA-Z\s]+)",
            r"don't understand ([a-zA-Z\s]+)",
            r"confused about ([a-zA-Z\s]+)",
            r"need to learn ([a-zA-Z\s]+)",
            r"what is ([a-zA-Z\s]+)\?",
            r"how does ([a-zA-Z\s]+) work"
        ]
        
        for pattern in uncertainty_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                concept = match.strip()
                if len(concept) > 2 and len(concept.split()) <= 4:  # Reasonable concept length
                    uncertain_concepts.append(concept)
        
        return uncertain_concepts
    
    def _assess_concept_coverage(self, concept: str, memories: List[Dict[str, Any]]) -> float:
        """Assess how well a concept is covered in the knowledge base."""
        concept_mentions = 0
        detailed_coverage = 0
        total_memories = len(memories)
        
        for memory in memories:
            content_lower = memory["content"].lower()
            
            if concept in content_lower:
                concept_mentions += 1
                
                # Check for detailed coverage (longer explanations)
                concept_context = self._extract_concept_context(concept, content_lower)
                if len(concept_context) > 100:  # Substantial content
                    detailed_coverage += 1
        
        # Coverage score based on mentions and detail
        mention_score = min(concept_mentions / total_memories * 10, 1.0)
        detail_score = min(detailed_coverage / max(concept_mentions, 1), 1.0)
        
        return (mention_score + detail_score) / 2
    
    def _extract_concept_context(self, concept: str, content: str) -> str:
        """Extract context around a concept mention."""
        concept_index = content.find(concept)
        if concept_index == -1:
            return ""
        
        start = max(0, concept_index - 200)
        end = min(len(content), concept_index + len(concept) + 200)
        
        return content[start:end]
    
    def _create_conceptual_gap(
        self,
        concept: str,
        source_memory: Dict[str, Any],
        coverage: float,
        context: GapDetectionContext
    ) -> KnowledgeGap:
        """Create a conceptual knowledge gap."""
        gap_id = hashlib.md5(f"conceptual_{concept}_{context.user_id}".encode()).hexdigest()[:16]
        
        title = f"Insufficient Understanding of {concept.title()}"
        description = f"Analysis indicates uncertainty about '{concept}' with limited coverage ({coverage:.1%}) in your knowledge base."
        
        # Priority based on coverage
        if coverage < 0.1:
            priority = Priority.HIGH
            significance = 0.9
            impact = ImpactLevel.SIGNIFICANT
        elif coverage < 0.2:
            priority = Priority.MEDIUM
            significance = 0.7
            impact = ImpactLevel.MODERATE
        else:
            priority = Priority.LOW
            significance = 0.5
            impact = ImpactLevel.MINIMAL
        
        domain = self._identify_domain([concept])
        
        filling_suggestions = [
            FillingSuggestion(
                suggestion_type="focused_research",
                description=f"Research {concept} thoroughly to build solid understanding",
                resources=[
                    f"Find authoritative definitions of {concept}",
                    f"Look for examples and applications of {concept}",
                    f"Study related concepts and context"
                ],
                effort_estimate="1-2 weeks",
                success_probability=0.8
            ),
            FillingSuggestion(
                suggestion_type="practical_exploration",
                description=f"Find practical applications or examples of {concept}",
                resources=[
                    f"Look for case studies involving {concept}",
                    f"Find tutorials or guides about {concept}",
                    f"Practice using or applying {concept}"
                ],
                effort_estimate="2-3 weeks",
                success_probability=0.7
            )
        ]
        
        learning_path = [
            f"Define and understand {concept}",
            f"Learn the context and background of {concept}",
            f"Study examples and applications of {concept}",
            f"Practice using {concept} in relevant contexts"
        ]
        
        return KnowledgeGap(
            gap_id=gap_id,
            gap_type=GapType.CONCEPTUAL_GAP,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            significance_score=significance,
            confidence=0.8,
            missing_concepts=[concept],
            related_memories=[source_memory["id"]],
            domain_area=domain,
            gap_indicators=[f"Uncertainty markers around '{concept}'", f"Low coverage: {coverage:.1%}"],
            surrounding_knowledge={"source_context": source_memory["content"][:200]},
            knowledge_prerequisites=[f"Basic familiarity with {domain} domain"],
            filling_suggestions=filling_suggestions,
            learning_path=learning_path,
            estimated_effort="2-4 weeks",
            tags={"conceptual", "understanding", domain}
        )
    
    async def _detect_relationship_gaps(
        self,
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Detect gaps in relationships between concepts."""
        gaps = []
        
        try:
            # Build concept network from memories
            concept_network = await self._build_concept_network(memories)
            
            # Identify weak or missing connections
            weak_connections = self._identify_weak_connections(concept_network)
            
            for connection_info in weak_connections:
                gap = self._create_relationship_gap(connection_info, memories, context)
                gaps.append(gap)
                
        except Exception as e:
            logger.error("Error in relationship gap detection", error=str(e))
        
        return gaps
    
    async def _build_concept_network(self, memories: List[Dict[str, Any]]) -> nx.Graph:
        """Build a network of concepts and their relationships."""
        G = nx.Graph()
        
        # Extract concepts from each memory
        memory_concepts = {}
        for memory in memories:
            concepts = self._extract_concepts_from_memory(memory)
            memory_concepts[memory["id"]] = concepts
            
            # Add concepts as nodes
            for concept in concepts:
                if not G.has_node(concept):
                    G.add_node(concept, mentions=0)
                G.nodes[concept]["mentions"] += 1
        
        # Add edges based on co-occurrence
        for memory_id, concepts in memory_concepts.items():
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    if G.has_edge(concept1, concept2):
                        G.edges[concept1, concept2]["weight"] += 1
                    else:
                        G.add_edge(concept1, concept2, weight=1)
        
        return G
    
    def _extract_concepts_from_memory(self, memory: Dict[str, Any]) -> List[str]:
        """Extract key concepts from a memory."""
        concepts = []
        content = memory["content"]
        
        # Use NLP if available
        if self.nlp:
            try:
                doc = self.nlp(content)
                # Extract nouns and noun phrases
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                        concepts.append(token.lemma_.lower())
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:  # Limit phrase length
                        concepts.append(chunk.text.lower())
                        
            except Exception as e:
                logger.warning(f"NLP processing failed: {e}")
        
        # Fallback: extract capitalized words and domain keywords
        words = content.split()
        for word in words:
            clean_word = word.strip('.,!?;:"()[]').lower()
            if len(clean_word) > 3:
                # Check if it's a domain keyword
                for domain_keywords in self.knowledge_domains.values():
                    if clean_word in domain_keywords:
                        concepts.append(clean_word)
                        break
        
        # Also include tags and entities
        concepts.extend(memory.get("tags", []))
        concepts.extend(memory.get("entities", []))
        
        return list(set(concepts))  # Remove duplicates
    
    def _identify_weak_connections(self, concept_network: nx.Graph) -> List[Dict[str, Any]]:
        """Identify weak or missing connections in the concept network."""
        weak_connections = []
        
        # Find concepts that should be connected but aren't
        nodes = list(concept_network.nodes())
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Check if nodes should be connected based on domain similarity
                if self._should_be_connected_concepts(node1, node2):
                    if not concept_network.has_edge(node1, node2):
                        # Missing connection
                        weak_connections.append({
                            "concept1": node1,
                            "concept2": node2,
                            "connection_strength": 0.0,
                            "gap_type": "missing_connection"
                        })
                    else:
                        # Check if existing connection is weak
                        weight = concept_network.edges[node1, node2]["weight"]
                        expected_weight = self._calculate_expected_connection_strength(
                            node1, node2, concept_network
                        )
                        
                        if weight < expected_weight * 0.5:  # Much weaker than expected
                            weak_connections.append({
                                "concept1": node1,
                                "concept2": node2,
                                "connection_strength": weight / expected_weight,
                                "gap_type": "weak_connection"
                            })
        
        return weak_connections[:10]  # Limit to most significant gaps
    
    def _should_be_connected_concepts(self, concept1: str, concept2: str) -> bool:
        """Determine if two concepts should naturally be connected."""
        # Check for domain relationships
        concept1_domains = set()
        concept2_domains = set()
        
        for domain, keywords in self.knowledge_domains.items():
            if concept1 in keywords:
                concept1_domains.add(domain)
            if concept2 in keywords:
                concept2_domains.add(domain)
        
        # Should be connected if in same domain
        if concept1_domains.intersection(concept2_domains):
            return True
        
        # Check for semantic similarity (simplified)
        if self._are_semantically_similar(concept1, concept2):
            return True
        
        return False
    
    def _are_semantically_similar(self, concept1: str, concept2: str) -> bool:
        """Check if concepts are semantically similar (simplified)."""
        # Simple heuristic: share word roots or are related terms
        concept1_words = set(concept1.split())
        concept2_words = set(concept2.split())
        
        # Check for word overlap
        if concept1_words.intersection(concept2_words):
            return True
        
        # Check for common prefixes/suffixes
        if (concept1[:4] == concept2[:4] and len(concept1) > 4 and len(concept2) > 4):
            return True
        
        return False
    
    def _calculate_expected_connection_strength(
        self,
        concept1: str,
        concept2: str,
        network: nx.Graph
    ) -> float:
        """Calculate expected connection strength between concepts."""
        # Simple heuristic based on node degrees
        degree1 = network.degree(concept1)
        degree2 = network.degree(concept2)
        
        # Concepts with higher degrees should have stronger connections
        return math.sqrt(degree1 * degree2) / 10
    
    def _create_relationship_gap(
        self,
        connection_info: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> KnowledgeGap:
        """Create a relationship knowledge gap."""
        concept1 = connection_info["concept1"]
        concept2 = connection_info["concept2"]
        gap_id = hashlib.md5(f"relationship_{concept1}_{concept2}_{context.user_id}".encode()).hexdigest()[:16]
        
        if connection_info["gap_type"] == "missing_connection":
            title = f"Missing Connection Between {concept1.title()} and {concept2.title()}"
            description = f"These concepts appear related but lack explicit connections in your knowledge base."
            significance = 0.7
        else:
            title = f"Weak Connection Between {concept1.title()} and {concept2.title()}"
            description = f"Relationship between these concepts is underdeveloped (strength: {connection_info['connection_strength']:.2f})."
            significance = 0.5
        
        # Find related memories
        related_memories = []
        for memory in memories:
            content_lower = memory["content"].lower()
            if concept1 in content_lower or concept2 in content_lower:
                related_memories.append(memory["id"])
        
        domain = self._identify_domain([concept1, concept2])
        
        filling_suggestions = [
            FillingSuggestion(
                suggestion_type="relationship_research",
                description=f"Research the relationship between {concept1} and {concept2}",
                resources=[
                    f"Study how {concept1} relates to {concept2}",
                    f"Find examples where both concepts interact",
                    f"Look for comparative analyses"
                ],
                effort_estimate="1-2 weeks",
                success_probability=0.8
            )
        ]
        
        learning_path = [
            f"Understand {concept1} individually",
            f"Understand {concept2} individually", 
            f"Study the relationship between {concept1} and {concept2}",
            f"Find practical examples of their interaction"
        ]
        
        return KnowledgeGap(
            gap_id=gap_id,
            gap_type=GapType.RELATIONSHIP_GAP,
            title=title,
            description=description,
            priority=Priority.MEDIUM,
            impact_level=ImpactLevel.MODERATE,
            significance_score=significance,
            confidence=0.6,
            missing_concepts=[f"{concept1}-{concept2} relationship"],
            related_memories=related_memories[:5],
            domain_area=domain,
            gap_indicators=[f"Weak/missing connection in concept network"],
            surrounding_knowledge={"concept1": concept1, "concept2": concept2},
            knowledge_prerequisites=[f"Understanding of {concept1}", f"Understanding of {concept2}"],
            filling_suggestions=filling_suggestions,
            learning_path=learning_path,
            estimated_effort="2-3 weeks",
            tags={"relationship", "connection", domain}
        )
    
    async def _detect_domain_gaps(
        self,
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Detect gaps in domain knowledge coverage."""
        gaps = []
        
        try:
            # Analyze domain coverage
            domain_coverage = self._analyze_domain_coverage_detailed(memories)
            
            # Identify underrepresented domains
            for domain, coverage_info in domain_coverage.items():
                if coverage_info["coverage_score"] < 0.3 and coverage_info["importance"] > 0.5:
                    gap = self._create_domain_gap(domain, coverage_info, memories, context)
                    gaps.append(gap)
                    
        except Exception as e:
            logger.error("Error in domain gap detection", error=str(e))
        
        return gaps
    
    def _analyze_domain_coverage_detailed(self, memories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze coverage of different knowledge domains."""
        domain_coverage = {}
        
        for domain, keywords in self.knowledge_domains.items():
            coverage_info = {
                "keyword_mentions": 0,
                "memory_count": 0,
                "total_content_length": 0,
                "coverage_score": 0.0,
                "importance": 0.5,  # Default importance
                "related_memories": []
            }
            
            for memory in memories:
                content_lower = memory["content"].lower()
                domain_mentions = sum(1 for keyword in keywords if keyword in content_lower)
                
                if domain_mentions > 0:
                    coverage_info["keyword_mentions"] += domain_mentions
                    coverage_info["memory_count"] += 1
                    coverage_info["total_content_length"] += len(memory["content"])
                    coverage_info["related_memories"].append(memory["id"])
            
            # Calculate coverage score
            if len(memories) > 0:
                memory_ratio = coverage_info["memory_count"] / len(memories)
                keyword_density = coverage_info["keyword_mentions"] / max(len(keywords), 1)
                avg_content_length = coverage_info["total_content_length"] / max(coverage_info["memory_count"], 1)
                
                coverage_info["coverage_score"] = (
                    memory_ratio * 0.4 +
                    min(keyword_density, 1.0) * 0.4 +
                    min(avg_content_length / 500, 1.0) * 0.2  # Normalize by expected length
                )
            
            # Estimate importance based on keyword frequency across all memories
            total_mentions = sum(
                sum(1 for keyword in keywords 
                    for memory in memories 
                    if keyword in memory["content"].lower())
                for keyword in keywords
            )
            coverage_info["importance"] = min(total_mentions / len(memories) / len(keywords), 1.0)
            
            domain_coverage[domain] = coverage_info
        
        return domain_coverage
    
    def _create_domain_gap(
        self,
        domain: str,
        coverage_info: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> KnowledgeGap:
        """Create a domain knowledge gap."""
        gap_id = hashlib.md5(f"domain_{domain}_{context.user_id}".encode()).hexdigest()[:16]
        
        coverage_score = coverage_info["coverage_score"]
        importance = coverage_info["importance"]
        
        title = f"Limited Knowledge in {domain.title()} Domain"
        description = f"Coverage analysis shows minimal {domain} knowledge (score: {coverage_score:.2f}) despite apparent relevance."
        
        # Priority based on importance and coverage gap
        gap_size = 1 - coverage_score
        if importance > 0.7 and gap_size > 0.7:
            priority = Priority.HIGH
            significance = 0.8
            impact = ImpactLevel.SIGNIFICANT
        elif importance > 0.5 and gap_size > 0.5:
            priority = Priority.MEDIUM
            significance = 0.6
            impact = ImpactLevel.MODERATE
        else:
            priority = Priority.LOW
            significance = 0.4
            impact = ImpactLevel.MINIMAL
        
        domain_keywords = self.knowledge_domains[domain]
        
        filling_suggestions = [
            FillingSuggestion(
                suggestion_type="domain_exploration",
                description=f"Systematically explore {domain} domain knowledge",
                resources=[
                    f"Study foundational concepts in {domain}",
                    f"Find introductory resources about {domain}",
                    f"Connect with {domain} experts or communities"
                ],
                effort_estimate="4-8 weeks",
                success_probability=0.8,
                prerequisites=["Time commitment", "Learning motivation"]
            ),
            FillingSuggestion(
                suggestion_type="targeted_learning",
                description=f"Focus on specific {domain} areas of immediate relevance",
                resources=[
                    f"Identify most relevant {domain} topics",
                    f"Find practical applications in {domain}",
                    f"Start with basics and build progressively"
                ],
                effort_estimate="2-4 weeks",
                success_probability=0.7
            )
        ]
        
        learning_path = [
            f"Assess current {domain} knowledge level",
            f"Identify priority {domain} topics",
            f"Study foundational {domain} concepts",
            f"Explore advanced {domain} applications",
            f"Practice applying {domain} knowledge"
        ]
        
        return KnowledgeGap(
            gap_id=gap_id,
            gap_type=GapType.DOMAIN_GAP,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            significance_score=significance,
            confidence=0.7,
            missing_concepts=domain_keywords[:5],
            related_memories=coverage_info["related_memories"][:5],
            domain_area=domain,
            gap_indicators=[f"Low domain coverage: {coverage_score:.2f}", f"Domain importance: {importance:.2f}"],
            surrounding_knowledge={"domain_keywords": domain_keywords},
            knowledge_prerequisites=[f"Basic familiarity with {domain} terminology"],
            filling_suggestions=filling_suggestions,
            learning_path=learning_path,
            estimated_effort="4-12 weeks",
            tags={"domain", "comprehensive", domain}
        )
    
    async def _detect_procedural_gaps(
        self,
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Detect gaps in procedural (how-to) knowledge."""
        gaps = []
        
        try:
            # Look for procedural knowledge patterns
            procedural_patterns = [
                r"how to ([a-zA-Z\s]+)",
                r"steps to ([a-zA-Z\s]+)",
                r"process of ([a-zA-Z\s]+)",
                r"method for ([a-zA-Z\s]+)",
                r"way to ([a-zA-Z\s]+)"
            ]
            
            mentioned_procedures = set()
            incomplete_procedures = []
            
            for memory in memories:
                content = memory["content"]
                content_lower = content.lower()
                
                # Find mentioned procedures
                for pattern in procedural_patterns:
                    matches = re.findall(pattern, content_lower)
                    mentioned_procedures.update(matches)
                
                # Look for incomplete procedure descriptions
                if any(marker in content_lower for marker in ["step", "process", "method", "procedure"]):
                    # Check if it's incomplete
                    if self._is_incomplete_procedure(content):
                        incomplete_procedures.append({
                            "memory": memory,
                            "incomplete_aspects": self._identify_incomplete_aspects(content)
                        })
            
            # Create gaps for incomplete procedures
            for proc_info in incomplete_procedures:
                gap = self._create_procedural_gap(proc_info, memories, context)
                gaps.append(gap)
                
        except Exception as e:
            logger.error("Error in procedural gap detection", error=str(e))
        
        return gaps
    
    def _is_incomplete_procedure(self, content: str) -> bool:
        """Check if a procedure description is incomplete."""
        content_lower = content.lower()
        
        # Look for incompleteness indicators
        incomplete_indicators = [
            "need to figure out",
            "not sure how",
            "unclear process",
            "missing steps",
            "incomplete",
            "todo",
            "tbd",
            "to be determined"
        ]
        
        return any(indicator in content_lower for indicator in incomplete_indicators)
    
    def _identify_incomplete_aspects(self, content: str) -> List[str]:
        """Identify what aspects of a procedure are incomplete."""
        aspects = []
        content_lower = content.lower()
        
        if "step" in content_lower and ("missing" in content_lower or "unclear" in content_lower):
            aspects.append("missing_steps")
        
        if "detail" in content_lower and ("need" in content_lower or "lack" in content_lower):
            aspects.append("insufficient_detail")
        
        if "example" in content_lower and ("need" in content_lower or "want" in content_lower):
            aspects.append("missing_examples")
        
        if not aspects:
            aspects.append("general_incompleteness")
        
        return aspects
    
    def _create_procedural_gap(
        self,
        proc_info: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> KnowledgeGap:
        """Create a procedural knowledge gap."""
        memory = proc_info["memory"]
        aspects = proc_info["incomplete_aspects"]
        
        gap_id = hashlib.md5(f"procedural_{memory['id']}_{context.user_id}".encode()).hexdigest()[:16]
        
        title = "Incomplete Procedural Knowledge"
        description = f"Procedural knowledge gaps identified: {', '.join(aspects)}"
        
        priority = Priority.MEDIUM
        significance = 0.6
        impact = ImpactLevel.MODERATE
        
        # Extract the procedure topic
        content = memory["content"]
        procedure_topic = self._extract_procedure_topic(content)
        domain = self._identify_domain([procedure_topic]) if procedure_topic else "general"
        
        filling_suggestions = [
            FillingSuggestion(
                suggestion_type="step_completion",
                description="Complete the missing procedural steps",
                resources=[
                    "Research detailed step-by-step guides",
                    "Find expert tutorials or documentation",
                    "Practice the procedure to identify gaps"
                ],
                effort_estimate="1-2 weeks",
                success_probability=0.8
            ),
            FillingSuggestion(
                suggestion_type="practical_validation",
                description="Validate the procedure through practice",
                resources=[
                    "Test the procedure in practice",
                    "Get feedback from others",
                    "Refine based on experience"
                ],
                effort_estimate="2-3 weeks",
                success_probability=0.7
            )
        ]
        
        learning_path = [
            "Identify specific procedural gaps",
            "Research complete procedure documentation",
            "Practice the procedure step-by-step",
            "Document lessons learned and refinements"
        ]
        
        return KnowledgeGap(
            gap_id=gap_id,
            gap_type=GapType.PROCEDURAL_GAP,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            significance_score=significance,
            confidence=0.7,
            missing_concepts=[f"{procedure_topic} procedure" if procedure_topic else "procedure steps"],
            related_memories=[memory["id"]],
            domain_area=domain,
            gap_indicators=[f"Incomplete aspects: {', '.join(aspects)}"],
            surrounding_knowledge={"partial_procedure": content[:200]},
            knowledge_prerequisites=["Basic understanding of the domain"],
            filling_suggestions=filling_suggestions,
            learning_path=learning_path,
            estimated_effort="2-4 weeks",
            tags={"procedural", "how-to", domain}
        )
    
    def _extract_procedure_topic(self, content: str) -> Optional[str]:
        """Extract the main topic of a procedure."""
        # Simple extraction - look for topics after procedural keywords
        procedural_keywords = ["how to", "steps to", "process of", "method for"]
        
        for keyword in procedural_keywords:
            if keyword in content.lower():
                start_index = content.lower().find(keyword) + len(keyword)
                remaining = content[start_index:start_index+50].strip()
                
                # Extract the topic (first few words)
                words = remaining.split()[:3]
                if words:
                    return " ".join(words).strip('.,!?;:"()[]')
        
        return None
    
    async def _detect_temporal_gaps(
        self,
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Detect gaps in temporal sequences or progressions."""
        gaps = []
        
        try:
            # Group memories by topic and analyze temporal patterns
            topic_timelines = self._build_topic_timelines(memories)
            
            for topic, timeline in topic_timelines.items():
                if len(timeline) >= 3:  # Need sufficient data points
                    temporal_gaps = self._identify_temporal_gaps_in_timeline(timeline, topic)
                    for gap_info in temporal_gaps:
                        gap = self._create_temporal_gap(gap_info, memories, context)
                        gaps.append(gap)
                        
        except Exception as e:
            logger.error("Error in temporal gap detection", error=str(e))
        
        return gaps
    
    def _build_topic_timelines(self, memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Build timelines for different topics."""
        # Group memories by primary topic/tag
        topic_groups = defaultdict(list)
        
        for memory in memories:
            # Use first tag as primary topic, or extract from content
            primary_topic = None
            
            if memory.get("tags"):
                primary_topic = memory["tags"][0]
            else:
                # Extract topic from content (simplified)
                words = memory["content"].split()[:10]  # First 10 words
                for word in words:
                    if len(word) > 4 and word.isalpha():
                        primary_topic = word.lower()
                        break
            
            if primary_topic:
                topic_groups[primary_topic].append({
                    "memory": memory,
                    "timestamp": memory["created_at"],
                    "content": memory["content"]
                })
        
        # Sort each timeline by timestamp
        for topic in topic_groups:
            topic_groups[topic].sort(key=lambda x: x["timestamp"])
        
        return dict(topic_groups)
    
    def _identify_temporal_gaps_in_timeline(
        self,
        timeline: List[Dict[str, Any]],
        topic: str
    ) -> List[Dict[str, Any]]:
        """Identify gaps in a temporal sequence."""
        gaps = []
        
        # Analyze time intervals between memories
        intervals = []
        for i in range(1, len(timeline)):
            interval = (timeline[i]["timestamp"] - timeline[i-1]["timestamp"]).days
            intervals.append(interval)
        
        if not intervals:
            return gaps
        
        # Find unusually large gaps
        avg_interval = sum(intervals) / len(intervals)
        std_interval = np.std(intervals) if len(intervals) > 1 else 0
        
        for i, interval in enumerate(intervals):
            if interval > avg_interval + 2 * std_interval and interval > 30:  # Significant gap
                gaps.append({
                    "topic": topic,
                    "gap_start": timeline[i]["timestamp"],
                    "gap_end": timeline[i+1]["timestamp"],
                    "gap_duration": interval,
                    "before_memory": timeline[i]["memory"],
                    "after_memory": timeline[i+1]["memory"]
                })
        
        return gaps
    
    def _create_temporal_gap(
        self,
        gap_info: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> KnowledgeGap:
        """Create a temporal knowledge gap."""
        topic = gap_info["topic"]
        duration = gap_info["gap_duration"]
        
        gap_id = hashlib.md5(f"temporal_{topic}_{duration}_{context.user_id}".encode()).hexdigest()[:16]
        
        title = f"Knowledge Gap in {topic.title()} Timeline"
        description = f"Detected {duration}-day gap in {topic} knowledge progression, suggesting missing updates or developments."
        
        # Priority based on gap size
        if duration > 180:  # 6 months
            priority = Priority.HIGH
            significance = 0.8
            impact = ImpactLevel.SIGNIFICANT
        elif duration > 90:  # 3 months
            priority = Priority.MEDIUM
            significance = 0.6
            impact = ImpactLevel.MODERATE
        else:
            priority = Priority.LOW
            significance = 0.4
            impact = ImpactLevel.MINIMAL
        
        domain = self._identify_domain([topic])
        
        filling_suggestions = [
            FillingSuggestion(
                suggestion_type="timeline_update",
                description=f"Update knowledge about {topic} developments during the gap period",
                resources=[
                    f"Research what happened in {topic} between {gap_info['gap_start'].strftime('%Y-%m')} and {gap_info['gap_end'].strftime('%Y-%m')}",
                    f"Find recent updates and changes in {topic}",
                    f"Connect with current {topic} developments"
                ],
                effort_estimate="2-4 weeks",
                success_probability=0.7
            )
        ]
        
        learning_path = [
            f"Review last known state of {topic}",
            f"Research developments during gap period",
            f"Update understanding of current {topic} state",
            f"Fill in missing progression steps"
        ]
        
        return KnowledgeGap(
            gap_id=gap_id,
            gap_type=GapType.TEMPORAL_GAP,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            significance_score=significance,
            confidence=0.6,
            missing_concepts=[f"{topic} developments"],
            related_memories=[gap_info["before_memory"]["id"], gap_info["after_memory"]["id"]],
            domain_area=domain,
            gap_indicators=[f"{duration}-day gap in timeline"],
            surrounding_knowledge={
                "before_state": gap_info["before_memory"]["content"][:100],
                "after_state": gap_info["after_memory"]["content"][:100]
            },
            knowledge_prerequisites=[f"Basic understanding of {topic}"],
            filling_suggestions=filling_suggestions,
            learning_path=learning_path,
            estimated_effort="2-6 weeks",
            tags={"temporal", "timeline", domain, topic}
        )
    
    async def _detect_prerequisite_gaps(
        self,
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Detect gaps in prerequisite knowledge."""
        gaps = []
        
        try:
            # Look for mentions of advanced concepts without foundational knowledge
            for memory in memories:
                content = memory["content"]
                
                # Identify advanced concepts
                advanced_concepts = self._identify_advanced_concepts(content)
                
                for concept in advanced_concepts:
                    # Check if prerequisites are covered
                    prerequisites = self._identify_prerequisites(concept)
                    missing_prerequisites = []
                    
                    for prereq in prerequisites:
                        if not self._is_prerequisite_covered(prereq, memories):
                            missing_prerequisites.append(prereq)
                    
                    if missing_prerequisites:
                        gap = self._create_prerequisite_gap(
                            concept, missing_prerequisites, memory, memories, context
                        )
                        gaps.append(gap)
                        
        except Exception as e:
            logger.error("Error in prerequisite gap detection", error=str(e))
        
        return gaps
    
    def _identify_advanced_concepts(self, content: str) -> List[str]:
        """Identify advanced concepts that likely have prerequisites."""
        # This is a simplified heuristic - look for technical terms
        advanced_indicators = [
            "algorithm", "implementation", "optimization", "architecture",
            "framework", "methodology", "paradigm", "strategy", "analysis"
        ]
        
        advanced_concepts = []
        content_lower = content.lower()
        
        for indicator in advanced_indicators:
            if indicator in content_lower:
                # Extract the concept around the indicator
                words = content.split()
                for i, word in enumerate(words):
                    if indicator in word.lower():
                        # Get surrounding context
                        start = max(0, i-2)
                        end = min(len(words), i+3)
                        concept = " ".join(words[start:end])
                        advanced_concepts.append(concept)
                        break
        
        return advanced_concepts
    
    def _identify_prerequisites(self, concept: str) -> List[str]:
        """Identify likely prerequisites for a concept."""
        # Simplified prerequisite mapping
        concept_lower = concept.lower()
        prerequisites = []
        
        if "algorithm" in concept_lower:
            prerequisites.extend(["basic programming", "data structures", "complexity analysis"])
        elif "framework" in concept_lower:
            prerequisites.extend(["programming fundamentals", "software architecture"])
        elif "optimization" in concept_lower:
            prerequisites.extend(["basic mathematics", "problem analysis"])
        elif "analysis" in concept_lower:
            prerequisites.extend(["data collection", "statistical basics"])
        
        # General prerequisites for technical concepts
        if any(tech in concept_lower for tech in ["implementation", "development", "system"]):
            prerequisites.append("technical fundamentals")
        
        return prerequisites
    
    def _is_prerequisite_covered(self, prerequisite: str, memories: List[Dict[str, Any]]) -> bool:
        """Check if a prerequisite is adequately covered."""
        coverage_count = 0
        
        for memory in memories:
            content_lower = memory["content"].lower()
            if prerequisite.lower() in content_lower:
                coverage_count += 1
        
        # Consider covered if mentioned in at least 2 memories
        return coverage_count >= 2
    
    def _create_prerequisite_gap(
        self,
        concept: str,
        missing_prerequisites: List[str],
        source_memory: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> KnowledgeGap:
        """Create a prerequisite knowledge gap."""
        gap_id = hashlib.md5(f"prerequisite_{concept}_{context.user_id}".encode()).hexdigest()[:16]
        
        title = f"Missing Prerequisites for {concept}"
        description = f"Advanced concept '{concept}' mentioned without adequate prerequisite knowledge: {', '.join(missing_prerequisites)}"
        
        priority = Priority.HIGH
        significance = 0.8
        impact = ImpactLevel.SIGNIFICANT
        
        domain = self._identify_domain([concept])
        
        filling_suggestions = [
            FillingSuggestion(
                suggestion_type="prerequisite_study",
                description="Study prerequisite concepts before advancing",
                resources=[
                    f"Learn about {missing_prerequisites[0]} fundamentals",
                    f"Find introductory resources for {', '.join(missing_prerequisites[:2])}",
                    "Build foundational understanding progressively"
                ],
                effort_estimate="4-8 weeks",
                success_probability=0.9,
                prerequisites=["Basic learning commitment"]
            )
        ]
        
        learning_path = [
            f"Identify specific prerequisite gaps",
            f"Study {missing_prerequisites[0]} basics",
            f"Learn {', '.join(missing_prerequisites[1:3])}",
            f"Return to {concept} with solid foundation"
        ]
        
        return KnowledgeGap(
            gap_id=gap_id,
            gap_type=GapType.PREREQUISITE_GAP,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            significance_score=significance,
            confidence=0.8,
            missing_concepts=missing_prerequisites,
            related_memories=[source_memory["id"]],
            domain_area=domain,
            gap_indicators=[f"Advanced concept without prerequisites"],
            surrounding_knowledge={"advanced_concept": concept},
            knowledge_prerequisites=missing_prerequisites,
            filling_suggestions=filling_suggestions,
            learning_path=learning_path,
            estimated_effort="4-12 weeks",
            tags={"prerequisite", "foundational", domain}
        )
    
    async def _detect_update_gaps(
        self,
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Detect knowledge that may be outdated and need updates."""
        gaps = []
        
        try:
            # Find potentially outdated memories
            cutoff_date = datetime.utcnow() - timedelta(days=365)  # 1 year old
            
            for memory in memories:
                if memory["created_at"] < cutoff_date:
                    # Check if this is likely to become outdated
                    if self._is_likely_outdated(memory):
                        gap = self._create_update_gap(memory, memories, context)
                        gaps.append(gap)
                        
        except Exception as e:
            logger.error("Error in update gap detection", error=str(e))
        
        return gaps
    
    def _is_likely_outdated(self, memory: Dict[str, Any]) -> bool:
        """Check if memory content is likely to become outdated."""
        content_lower = memory["content"].lower()
        
        # Look for time-sensitive content
        outdated_indicators = [
            "current", "latest", "new", "recent", "today", "now",
            "version", "update", "release", "trending", "modern"
        ]
        
        # Look for rapidly changing domains
        fast_changing_domains = [
            "technology", "software", "ai", "machine learning",
            "web", "mobile", "crypto", "startup", "market"
        ]
        
        has_time_sensitive = any(indicator in content_lower for indicator in outdated_indicators)
        has_fast_changing = any(domain in content_lower for domain in fast_changing_domains)
        
        return has_time_sensitive or has_fast_changing
    
    def _create_update_gap(
        self,
        memory: Dict[str, Any],
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> KnowledgeGap:
        """Create an update knowledge gap."""
        gap_id = hashlib.md5(f"update_{memory['id']}_{context.user_id}".encode()).hexdigest()[:16]
        
        age_days = (datetime.utcnow() - memory["created_at"]).days
        
        title = f"Potentially Outdated Knowledge"
        description = f"Memory from {age_days} days ago may contain outdated information requiring updates."
        
        priority = Priority.MEDIUM
        significance = 0.5
        impact = ImpactLevel.MODERATE
        
        # Extract main topic
        content_preview = memory["content"][:100]
        main_topic = self._extract_main_topic(memory["content"])
        domain = self._identify_domain([main_topic]) if main_topic else "general"
        
        filling_suggestions = [
            FillingSuggestion(
                suggestion_type="currency_check",
                description="Check if information is still current and accurate",
                resources=[
                    f"Research current state of {main_topic}",
                    "Find recent updates and changes",
                    "Verify accuracy of key facts"
                ],
                effort_estimate="1-2 weeks",
                success_probability=0.8
            )
        ]
        
        learning_path = [
            "Review existing knowledge",
            "Research current developments",
            "Update outdated information", 
            "Document changes and updates"
        ]
        
        return KnowledgeGap(
            gap_id=gap_id,
            gap_type=GapType.UPDATE_GAP,
            title=title,
            description=description,
            priority=priority,
            impact_level=impact,
            significance_score=significance,
            confidence=0.6,
            missing_concepts=[f"updated {main_topic}"],
            related_memories=[memory["id"]],
            domain_area=domain,
            gap_indicators=[f"Memory age: {age_days} days"],
            surrounding_knowledge={"old_content": content_preview},
            knowledge_prerequisites=["Access to current information"],
            filling_suggestions=filling_suggestions,
            learning_path=learning_path,
            estimated_effort="1-3 weeks",
            tags={"update", "currency", domain}
        )
    
    def _extract_main_topic(self, content: str) -> Optional[str]:
        """Extract the main topic from content."""
        # Simple extraction - get most frequent meaningful words
        words = content.lower().split()
        
        # Filter stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        meaningful_words = [
            word.strip('.,!?;:"()[]') for word in words 
            if len(word) > 3 and word not in stop_words
        ]
        
        if meaningful_words:
            # Return most frequent word as main topic
            word_counts = Counter(meaningful_words)
            return word_counts.most_common(1)[0][0]
        
        return None
    
    # Continue with enhancement and utility methods...
    
    async def _enhance_gaps_with_suggestions(
        self,
        gaps: List[KnowledgeGap],
        memories: List[Dict[str, Any]],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Enhance gaps with additional filling suggestions and context."""
        enhanced_gaps = []
        
        for gap in gaps:
            # Add contextual suggestions based on existing knowledge
            additional_suggestions = self._generate_contextual_suggestions(gap, memories)
            gap.filling_suggestions.extend(additional_suggestions)
            
            # Enhance learning path with specific steps
            enhanced_path = self._enhance_learning_path(gap, memories)
            gap.learning_path = enhanced_path
            
            # Add dependencies between gaps
            dependencies = self._identify_gap_dependencies(gap, gaps)
            gap.metadata["dependencies"] = dependencies
            
            enhanced_gaps.append(gap)
        
        return enhanced_gaps
    
    def _generate_contextual_suggestions(
        self,
        gap: KnowledgeGap,
        memories: List[Dict[str, Any]]
    ) -> List[FillingSuggestion]:
        """Generate additional contextual suggestions."""
        suggestions = []
        
        # Look for related memories that might provide context
        related_content = []
        for memory_id in gap.related_memories:
            memory = next((m for m in memories if m["id"] == memory_id), None)
            if memory:
                related_content.append(memory["content"])
        
        if related_content:
            suggestions.append(FillingSuggestion(
                suggestion_type="context_building",
                description="Build on existing related knowledge",
                resources=[
                    "Review related memories for connections",
                    "Identify patterns in existing knowledge",
                    "Build bridges between known and unknown"
                ],
                effort_estimate="1-2 weeks",
                success_probability=0.7
            ))
        
        return suggestions
    
    def _enhance_learning_path(
        self,
        gap: KnowledgeGap,
        memories: List[Dict[str, Any]]
    ) -> List[str]:
        """Enhance learning path with specific, actionable steps."""
        enhanced_path = []
        
        # Start with assessment
        enhanced_path.append(f"Assess current understanding of {gap.domain_area}")
        
        # Add domain-specific steps
        if gap.gap_type == GapType.CONCEPTUAL_GAP:
            enhanced_path.extend([
                f"Define key concepts: {', '.join(gap.missing_concepts[:3])}",
                f"Study examples and applications",
                f"Practice explaining concepts to others"
            ])
        elif gap.gap_type == GapType.PROCEDURAL_GAP:
            enhanced_path.extend([
                f"Find step-by-step guides",
                f"Practice procedure in safe environment",
                f"Document lessons learned and variations"
            ])
        elif gap.gap_type == GapType.RELATIONSHIP_GAP:
            enhanced_path.extend([
                f"Study each concept individually",
                f"Research how concepts interact",
                f"Find real-world examples of relationships"
            ])
        
        # Add verification step
        enhanced_path.append("Test and validate new knowledge")
        
        return enhanced_path
    
    def _identify_gap_dependencies(
        self,
        gap: KnowledgeGap,
        all_gaps: List[KnowledgeGap]
    ) -> List[str]:
        """Identify dependencies between gaps."""
        dependencies = []
        
        for other_gap in all_gaps:
            if other_gap.gap_id != gap.gap_id:
                # Check if other gap is a prerequisite
                if self._is_prerequisite_gap(other_gap, gap):
                    dependencies.append(other_gap.gap_id)
        
        return dependencies
    
    def _is_prerequisite_gap(self, potential_prereq: KnowledgeGap, current_gap: KnowledgeGap) -> bool:
        """Check if one gap is a prerequisite for another."""
        # Check domain overlap
        if potential_prereq.domain_area == current_gap.domain_area:
            # Check if prerequisite gap has foundational concepts
            if potential_prereq.gap_type == GapType.PREREQUISITE_GAP:
                return True
            
            # Check if concepts overlap
            prereq_concepts = set(potential_prereq.missing_concepts)
            current_prereqs = set(current_gap.knowledge_prerequisites)
            
            if prereq_concepts.intersection(current_prereqs):
                return True
        
        return False
    
    def _deduplicate_gaps(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """Remove duplicate gaps."""
        seen_combinations = set()
        unique_gaps = []
        
        for gap in gaps:
            # Create key based on type, domain, and main concepts
            key = (
                gap.gap_type,
                gap.domain_area,
                tuple(sorted(gap.missing_concepts[:3]))
            )
            
            if key not in seen_combinations:
                seen_combinations.add(key)
                unique_gaps.append(gap)
        
        return unique_gaps
    
    def _prioritize_gaps(
        self,
        gaps: List[KnowledgeGap],
        context: GapDetectionContext
    ) -> List[KnowledgeGap]:
        """Prioritize gaps based on significance and context."""
        # Calculate composite priority scores
        for gap in gaps:
            priority_weights = {
                Priority.CRITICAL: 5,
                Priority.HIGH: 4,
                Priority.MEDIUM: 3,
                Priority.LOW: 2,
                Priority.OPTIONAL: 1
            }
            
            impact_weights = {
                ImpactLevel.TRANSFORMATIVE: 5,
                ImpactLevel.SIGNIFICANT: 4,
                ImpactLevel.MODERATE: 3,
                ImpactLevel.MINIMAL: 2,
                ImpactLevel.UNCERTAIN: 1
            }
            
            priority_score = priority_weights.get(gap.priority, 1)
            impact_score = impact_weights.get(gap.impact_level, 1)
            
            # Composite score
            composite_score = (
                priority_score * 0.4 +
                impact_score * 0.3 +
                gap.significance_score * 5 * 0.2 +
                gap.confidence * 5 * 0.1
            )
            
            gap.metadata["composite_score"] = composite_score
        
        # Sort by composite score
        gaps.sort(key=lambda x: x.metadata.get("composite_score", 0), reverse=True)
        
        # Filter out low priority if not requested
        if not context.include_minor_gaps:
            gaps = [gap for gap in gaps if gap.priority != Priority.OPTIONAL]
        
        return gaps
    
    async def _calculate_knowledge_coverage(
        self,
        memories: List[Dict[str, Any]],
        gaps: List[KnowledgeGap]
    ) -> Dict[str, float]:
        """Calculate knowledge coverage metrics."""
        domain_coverage = {}
        
        # Calculate coverage for each domain
        for domain in self.knowledge_domains.keys():
            domain_memories = []
            domain_gaps = []
            
            for memory in memories:
                content_lower = memory["content"].lower()
                domain_keywords = self.knowledge_domains[domain]
                if any(keyword in content_lower for keyword in domain_keywords):
                    domain_memories.append(memory)
            
            for gap in gaps:
                if gap.domain_area == domain:
                    domain_gaps.append(gap)
            
            # Calculate coverage score
            if len(domain_memories) > 0:
                # Factor in both quantity and gap density
                memory_score = min(len(domain_memories) / 10, 1.0)  # Normalize to 10 memories
                gap_penalty = len(domain_gaps) * 0.1  # Each gap reduces coverage
                coverage = max(0, memory_score - gap_penalty)
            else:
                coverage = 0.0
            
            domain_coverage[domain] = coverage
        
        return domain_coverage
    
    async def _analyze_domain_coverage(
        self,
        memories: List[Dict[str, Any]],
        gaps: List[KnowledgeGap]
    ) -> Dict[str, Any]:
        """Analyze domain coverage in detail."""
        analysis = {
            "total_domains": len(self.knowledge_domains),
            "covered_domains": 0,
            "gap_density_by_domain": {},
            "strength_by_domain": {},
            "recommendations": []
        }
        
        for domain, keywords in self.knowledge_domains.items():
            # Count domain-related memories
            domain_memory_count = 0
            total_domain_content = 0
            
            for memory in memories:
                content_lower = memory["content"].lower()
                if any(keyword in content_lower for keyword in keywords):
                    domain_memory_count += 1
                    total_domain_content += len(memory["content"])
            
            # Count domain gaps
            domain_gap_count = len([gap for gap in gaps if gap.domain_area == domain])
            
            # Calculate metrics
            if domain_memory_count > 0:
                analysis["covered_domains"] += 1
                analysis["strength_by_domain"][domain] = {
                    "memory_count": domain_memory_count,
                    "avg_content_length": total_domain_content / domain_memory_count,
                    "coverage_strength": min(domain_memory_count / 5, 1.0)
                }
            else:
                analysis["strength_by_domain"][domain] = {
                    "memory_count": 0,
                    "avg_content_length": 0,
                    "coverage_strength": 0.0
                }
            
            analysis["gap_density_by_domain"][domain] = domain_gap_count
        
        # Generate domain recommendations
        weak_domains = [
            domain for domain, strength in analysis["strength_by_domain"].items()
            if strength["coverage_strength"] < 0.3
        ]
        
        if weak_domains:
            analysis["recommendations"].append(
                f"Focus on strengthening knowledge in: {', '.join(weak_domains[:3])}"
            )
        
        return analysis
    
    def _generate_immediate_actions(self, gaps: List[KnowledgeGap]) -> List[str]:
        """Generate immediate actionable items."""
        actions = []
        
        # Prioritize critical and high priority gaps
        urgent_gaps = [gap for gap in gaps if gap.priority in [Priority.CRITICAL, Priority.HIGH]]
        
        for gap in urgent_gaps[:3]:  # Top 3 urgent gaps
            if gap.gap_type == GapType.PREREQUISITE_GAP:
                actions.append(f"Immediately study prerequisites for {gap.missing_concepts[0]}")
            elif gap.gap_type == GapType.CONCEPTUAL_GAP:
                actions.append(f"Research and define {gap.missing_concepts[0]}")
            elif gap.gap_type == GapType.PROCEDURAL_GAP:
                actions.append(f"Find step-by-step guide for incomplete procedure")
            else:
                actions.append(f"Address {gap.title.lower()}")
        
        # Add general actions
        if len(gaps) > 5:
            actions.append("Create a systematic learning plan to address knowledge gaps")
        
        return actions
    
    def _generate_learning_recommendations(
        self,
        gaps: List[KnowledgeGap],
        memories: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate learning recommendations."""
        recommendations = []
        
        # Analyze gap patterns
        gap_types = Counter(gap.gap_type for gap in gaps)
        domains = Counter(gap.domain_area for gap in gaps)
        
        # Type-based recommendations
        if gap_types[GapType.CONCEPTUAL_GAP] > 2:
            recommendations.append("Focus on building stronger conceptual foundations")
        
        if gap_types[GapType.PROCEDURAL_GAP] > 1:
            recommendations.append("Seek hands-on practice and step-by-step tutorials")
        
        if gap_types[GapType.RELATIONSHIP_GAP] > 1:
            recommendations.append("Study how different concepts connect and interact")
        
        # Domain-based recommendations
        top_gap_domain = domains.most_common(1)[0][0] if domains else None
        if top_gap_domain and domains[top_gap_domain] > 1:
            recommendations.append(f"Consider structured learning in {top_gap_domain} domain")
        
        # General recommendations
        if len(gaps) > 3:
            recommendations.append("Establish regular learning schedule to systematically address gaps")
        
        return recommendations
    
    def _generate_knowledge_goals(
        self,
        gaps: List[KnowledgeGap],
        domain_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate strategic knowledge goals."""
        goals = []
        
        # Domain-based goals
        weak_domains = [
            domain for domain, strength in domain_analysis.get("strength_by_domain", {}).items()
            if strength["coverage_strength"] < 0.5
        ]
        
        for domain in weak_domains[:2]:
            goals.append(f"Achieve proficient understanding in {domain} domain")
        
        # Gap-type based goals
        gap_types = Counter(gap.gap_type for gap in gaps)
        
        if gap_types[GapType.PREREQUISITE_GAP] > 0:
            goals.append("Build solid foundational knowledge before advancing")
        
        if gap_types[GapType.PROCEDURAL_GAP] > 1:
            goals.append("Master key procedures through hands-on practice")
        
        # Impact-based goals
        high_impact_gaps = [gap for gap in gaps if gap.impact_level in [ImpactLevel.TRANSFORMATIVE, ImpactLevel.SIGNIFICANT]]
        if high_impact_gaps:
            goals.append(f"Focus on high-impact knowledge areas for maximum benefit")
        
        return goals
    
    def get_detection_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about gap detection."""
        return {
            "stats": self.detection_stats.copy(),
            "cache_status": {
                "embedding_cache_size": len(self.embedding_cache),
                "topic_cache_size": len(self.topic_cache),
                "domain_cache_size": len(self.domain_cache)
            },
            "performance": {
                "average_processing_time": self.detection_stats['average_processing_time'],
                "total_gaps_detected": self.detection_stats['gaps_detected'],
                "gaps_filled": self.detection_stats['gaps_filled']
            },
            "detection_patterns": {
                "by_type": self.detection_stats['detection_by_type'],
                "by_priority": self.detection_stats['priority_distribution']
            }
        }
    
    async def record_gap_resolution(
        self,
        gap_id: str,
        resolution_type: str,
        success: bool,
        notes: Optional[str] = None
    ) -> None:
        """Record resolution attempt for a knowledge gap."""
        resolution_record = {
            "gap_id": gap_id,
            "resolution_type": resolution_type,
            "success": success,
            "notes": notes,
            "timestamp": datetime.utcnow()
        }
        
        # This would be stored in the gap tracking system
        logger.info(
            "Recorded gap resolution",
            gap_id=gap_id,
            success=success,
            resolution_type=resolution_type
        )
        
        if success:
            self.detection_stats['gaps_filled'] += 1


# Example usage
async def example_usage():
    """Example of using LocalKnowledgeGapDetector."""
    from ...core.embeddings.embedder import Embedder
    from ...core.memory.manager import MemoryManager
    
    # Initialize components
    embedder = Embedder()
    memory_manager = MemoryManager()
    
    # Create gap detector
    detector = LocalKnowledgeGapDetector(embedder, memory_manager)
    
    # Create detection context
    context = GapDetectionContext(
        user_id="user123",
        analysis_scope="all",
        max_gaps=10,
        min_gap_significance=0.3,
        include_minor_gaps=False
    )
    
    # Detect knowledge gaps
    result = await detector.detect_knowledge_gaps(context)
    
    print(f"Detected {len(result.gaps)} knowledge gaps")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Detection strategies: {result.detection_strategies}")
    
    for i, gap in enumerate(result.gaps[:5], 1):
        print(f"\n{i}. {gap.title}")
        print(f"   Type: {gap.gap_type} | Priority: {gap.priority}")
        print(f"   Significance: {gap.significance_score:.2f} | Confidence: {gap.confidence:.2f}")
        print(f"   Description: {gap.description}")
        print(f"   Missing concepts: {', '.join(gap.missing_concepts[:3])}")
        print(f"   Filling suggestions: {len(gap.filling_suggestions)}")
    
    print(f"\nImmediate actions: {result.immediate_actions}")
    print(f"Learning recommendations: {result.learning_recommendations}")
    print(f"Knowledge goals: {result.knowledge_goals}")
    
    # Record resolution attempt
    if result.gaps:
        await detector.record_gap_resolution(
            gap_id=result.gaps[0].gap_id,
            resolution_type="research",
            success=True,
            notes="Completed initial research phase"
        )
    
    # Get analytics
    analytics = detector.get_detection_analytics()
    print(f"\nAnalytics: {analytics}")


if __name__ == "__main__":
    asyncio.run(example_usage())