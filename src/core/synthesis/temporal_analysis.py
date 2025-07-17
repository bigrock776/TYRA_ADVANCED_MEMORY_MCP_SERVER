"""
Temporal Memory Evolution Analysis with Time Series and Graph Analytics.

This module analyzes how memories and knowledge evolve over time, tracking
concept changes, learning progressions, and temporal relationships.
All processing uses local algorithms with networkx and time series analysis.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import spacy
from scipy import stats
from scipy.signal import find_peaks
import structlog
from pydantic import BaseModel, Field, ConfigDict
import json
import hashlib

from ...models.memory import Memory
from ..embeddings.embedder import Embedder
from ..cache.redis_cache import RedisCache
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class EvolutionType(str, Enum):
    """Types of temporal evolution patterns."""
    CONCEPT_DRIFT = "concept_drift"  # Gradual change in understanding
    SUDDEN_SHIFT = "sudden_shift"  # Abrupt change in perspective
    KNOWLEDGE_GROWTH = "knowledge_growth"  # Accumulating information
    REFINEMENT = "refinement"  # Improving accuracy/detail
    CONTRADICTION = "contradiction"  # Conflicting information
    SYNTHESIS = "synthesis"  # Combining multiple sources


class TrendDirection(str, Enum):
    """Direction of temporal trends."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    CYCLICAL = "cyclical"
    VOLATILE = "volatile"


@dataclass
class TemporalCluster:
    """Represents memories clustered by time period."""
    time_period: Tuple[datetime, datetime]
    memories: List[Memory]
    dominant_topics: List[str]
    average_embedding: np.ndarray
    concept_drift_score: float = 0.0
    novelty_score: float = 0.0


@dataclass
class ConceptEvolution:
    """Tracks how a concept evolves over time."""
    concept: str
    timeline: List[Tuple[datetime, Memory]]
    evolution_type: EvolutionType
    confidence: float
    key_changes: List[str]
    stability_score: float
    trend_direction: TrendDirection


class TemporalInsight(BaseModel):
    """Insights from temporal analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    insight_type: str = Field(description="Type of temporal insight")
    description: str = Field(description="Human-readable description")
    time_span: Tuple[datetime, datetime] = Field(description="Time period covered")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in insight")
    supporting_memories: List[str] = Field(description="Memory IDs supporting insight")
    trend_data: Dict[str, Any] = Field(default_factory=dict, description="Trend statistics")


class LearningProgression(BaseModel):
    """Represents learning progression over time."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    topic: str = Field(description="Topic or domain")
    start_date: datetime = Field(description="Start of learning period")
    end_date: datetime = Field(description="End of learning period")
    progression_score: float = Field(ge=0.0, le=1.0, description="Learning progression score")
    complexity_trend: TrendDirection = Field(description="Complexity change over time")
    milestones: List[Dict[str, Any]] = Field(description="Key learning milestones")
    knowledge_gaps: List[str] = Field(description="Identified gaps in progression")


class TemporalAnalysisResult(BaseModel):
    """Complete temporal analysis results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    concept_evolutions: List[ConceptEvolution] = Field(description="Concept evolution patterns")
    learning_progressions: List[LearningProgression] = Field(description="Learning trends")
    temporal_insights: List[TemporalInsight] = Field(description="Time-based insights")
    activity_patterns: Dict[str, Any] = Field(description="Activity pattern analysis")
    knowledge_velocity: Dict[str, float] = Field(description="Rate of knowledge acquisition")
    memory_lifecycle: Dict[str, Any] = Field(description="Memory lifecycle analytics")


class TemporalAnalyzer:
    """
    Advanced temporal memory evolution analyzer.
    
    Features:
    - Concept drift detection using embedding space analysis
    - Learning progression tracking with complexity scoring
    - Temporal graph construction for relationship analysis
    - Time series analysis for trend detection
    - Memory lifecycle analytics
    - Knowledge velocity measurement
    """
    
    def __init__(
        self,
        embedder: Embedder,
        cache: Optional[RedisCache] = None,
        time_window_days: int = 30,
        min_memories_per_window: int = 5,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize the temporal analyzer.
        
        Args:
            embedder: Embedder for generating vector representations
            cache: Redis cache for performance optimization
            time_window_days: Size of temporal windows for analysis
            min_memories_per_window: Minimum memories required per window
            spacy_model: SpaCy model for NLP tasks
        """
        self.embedder = embedder
        self.cache = cache
        self.time_window_days = time_window_days
        self.min_memories_per_window = min_memories_per_window
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load(spacy_model)
        except:
            logger.warning(f"SpaCy model {spacy_model} not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        logger.info(
            "Initialized temporal analyzer",
            time_window_days=time_window_days,
            min_memories_per_window=min_memories_per_window
        )
    
    async def analyze_temporal_evolution(
        self,
        memories: List[Memory],
        concepts_to_track: Optional[List[str]] = None
    ) -> TemporalAnalysisResult:
        """
        Perform comprehensive temporal analysis of memory evolution.
        
        Args:
            memories: List of memories to analyze
            concepts_to_track: Specific concepts to track (auto-detected if None)
            
        Returns:
            Complete temporal analysis results
        """
        if not memories:
            return TemporalAnalysisResult(
                concept_evolutions=[],
                learning_progressions=[],
                temporal_insights=[],
                activity_patterns={},
                knowledge_velocity={},
                memory_lifecycle={}
            )
        
        start_time = datetime.utcnow()
        
        # Sort memories by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        # Create temporal clusters
        temporal_clusters = await self._create_temporal_clusters(sorted_memories)
        
        # Detect concept evolutions
        concept_evolutions = await self._detect_concept_evolutions(
            sorted_memories, 
            concepts_to_track
        )
        
        # Analyze learning progressions
        learning_progressions = await self._analyze_learning_progressions(
            sorted_memories,
            temporal_clusters
        )
        
        # Generate temporal insights
        temporal_insights = await self._generate_temporal_insights(
            sorted_memories,
            concept_evolutions,
            temporal_clusters
        )
        
        # Analyze activity patterns
        activity_patterns = await self._analyze_activity_patterns(sorted_memories)
        
        # Calculate knowledge velocity
        knowledge_velocity = await self._calculate_knowledge_velocity(
            sorted_memories,
            temporal_clusters
        )
        
        # Analyze memory lifecycle
        memory_lifecycle = await self._analyze_memory_lifecycle(sorted_memories)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            "Temporal analysis complete",
            memories_count=len(memories),
            concept_evolutions=len(concept_evolutions),
            learning_progressions=len(learning_progressions),
            temporal_insights=len(temporal_insights),
            processing_time_s=processing_time
        )
        
        return TemporalAnalysisResult(
            concept_evolutions=concept_evolutions,
            learning_progressions=learning_progressions,
            temporal_insights=temporal_insights,
            activity_patterns=activity_patterns,
            knowledge_velocity=knowledge_velocity,
            memory_lifecycle=memory_lifecycle
        )
    
    async def _create_temporal_clusters(
        self, 
        memories: List[Memory]
    ) -> List[TemporalCluster]:
        """Create clusters of memories based on time windows."""
        if not memories:
            return []
        
        clusters = []
        start_date = memories[0].created_at
        end_date = memories[-1].created_at
        
        # Create time windows
        current_date = start_date
        window_delta = timedelta(days=self.time_window_days)
        
        while current_date < end_date:
            window_end = current_date + window_delta
            
            # Get memories in this time window
            window_memories = [
                m for m in memories 
                if current_date <= m.created_at < window_end
            ]
            
            if len(window_memories) >= self.min_memories_per_window:
                # Extract dominant topics
                texts = [m.content for m in window_memories]
                dominant_topics = await self._extract_dominant_topics(texts)
                
                # Calculate average embedding
                embeddings = await self.embedder.embed_batch(texts)
                average_embedding = np.mean(embeddings, axis=0)
                
                # Calculate novelty score (compared to previous clusters)
                novelty_score = 0.0
                if clusters:
                    prev_embedding = clusters[-1].average_embedding
                    similarity = cosine_similarity(
                        [average_embedding], 
                        [prev_embedding]
                    )[0][0]
                    novelty_score = 1.0 - similarity
                
                cluster = TemporalCluster(
                    time_period=(current_date, window_end),
                    memories=window_memories,
                    dominant_topics=dominant_topics,
                    average_embedding=average_embedding,
                    novelty_score=novelty_score
                )
                
                clusters.append(cluster)
            
            current_date = window_end
        
        # Calculate concept drift scores
        for i in range(1, len(clusters)):
            prev_embedding = clusters[i-1].average_embedding
            curr_embedding = clusters[i].average_embedding
            
            similarity = cosine_similarity([prev_embedding], [curr_embedding])[0][0]
            drift_score = 1.0 - similarity
            clusters[i].concept_drift_score = drift_score
        
        return clusters
    
    async def _extract_dominant_topics(self, texts: List[str]) -> List[str]:
        """Extract dominant topics from a collection of texts."""
        if not texts:
            return []
        
        # Combine all texts
        combined_text = ' '.join(texts)
        doc = self.nlp(combined_text)
        
        # Extract noun phrases and entities
        topics = []
        
        # Get noun phrases
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4:  # Multi-word phrases
                topics.append(chunk.text.lower())
        
        # Get named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                topics.append(ent.text.lower())
        
        # Count frequency and return top topics
        topic_counts = Counter(topics)
        return [topic for topic, _ in topic_counts.most_common(5)]
    
    async def _detect_concept_evolutions(
        self,
        memories: List[Memory],
        concepts_to_track: Optional[List[str]] = None
    ) -> List[ConceptEvolution]:
        """Detect how concepts evolve over time."""
        if not concepts_to_track:
            # Auto-detect concepts from memory content
            concepts_to_track = await self._auto_detect_concepts(memories)
        
        evolutions = []
        
        for concept in concepts_to_track:
            # Find memories mentioning this concept
            concept_memories = []
            for memory in memories:
                if concept.lower() in memory.content.lower():
                    concept_memories.append(memory)
            
            if len(concept_memories) < 3:  # Need minimum memories to track evolution
                continue
            
            # Sort by timestamp
            concept_memories.sort(key=lambda m: m.created_at)
            
            # Analyze evolution pattern
            evolution = await self._analyze_concept_evolution(concept, concept_memories)
            if evolution:
                evolutions.append(evolution)
        
        return evolutions
    
    async def _auto_detect_concepts(self, memories: List[Memory]) -> List[str]:
        """Automatically detect important concepts to track."""
        concept_frequency = defaultdict(int)
        
        for memory in memories:
            doc = self.nlp(memory.content)
            
            # Extract important noun phrases
            for chunk in doc.noun_chunks:
                if 2 <= len(chunk.text.split()) <= 3:
                    concept_frequency[chunk.text.lower()] += 1
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                    concept_frequency[ent.text.lower()] += 1
        
        # Return concepts mentioned in multiple memories
        frequent_concepts = [
            concept for concept, freq in concept_frequency.items() 
            if freq >= 3  # Minimum frequency threshold
        ]
        
        return frequent_concepts[:20]  # Limit to top 20 concepts
    
    async def _analyze_concept_evolution(
        self,
        concept: str,
        concept_memories: List[Memory]
    ) -> Optional[ConceptEvolution]:
        """Analyze how a specific concept evolves over time."""
        if len(concept_memories) < 3:
            return None
        
        # Create timeline
        timeline = [(m.created_at, m) for m in concept_memories]
        
        # Get embeddings for concept contexts
        concept_texts = []
        for memory in concept_memories:
            # Extract sentences containing the concept
            doc = self.nlp(memory.content)
            concept_sentences = []
            for sent in doc.sents:
                if concept.lower() in sent.text.lower():
                    concept_sentences.append(sent.text)
            
            if concept_sentences:
                concept_texts.append(' '.join(concept_sentences))
            else:
                concept_texts.append(memory.content)
        
        # Analyze embedding trajectory
        embeddings = await self.embedder.embed_batch(concept_texts)
        
        # Detect evolution type
        evolution_type = await self._classify_evolution_type(embeddings, timeline)
        
        # Calculate stability score
        stability_score = await self._calculate_stability_score(embeddings)
        
        # Determine trend direction
        trend_direction = await self._determine_trend_direction(embeddings)
        
        # Extract key changes
        key_changes = await self._extract_key_changes(concept_memories, concept)
        
        # Calculate confidence based on data quality
        confidence = min(1.0, len(concept_memories) / 10)  # More memories = higher confidence
        confidence *= stability_score  # Adjust by stability
        
        return ConceptEvolution(
            concept=concept,
            timeline=timeline,
            evolution_type=evolution_type,
            confidence=confidence,
            key_changes=key_changes,
            stability_score=stability_score,
            trend_direction=trend_direction
        )
    
    async def _classify_evolution_type(
        self,
        embeddings: np.ndarray,
        timeline: List[Tuple[datetime, Memory]]
    ) -> EvolutionType:
        """Classify the type of evolution based on embedding trajectory."""
        if len(embeddings) < 3:
            return EvolutionType.KNOWLEDGE_GROWTH
        
        # Calculate consecutive similarities
        similarities = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        similarity_variance = np.var(similarities)
        
        # Check for sudden changes
        sudden_changes = [s for s in similarities if s < 0.7]  # Low similarity threshold
        
        # Classify evolution type
        if len(sudden_changes) > len(similarities) * 0.3:  # >30% sudden changes
            return EvolutionType.SUDDEN_SHIFT
        elif similarity_variance > 0.1:  # High variance
            return EvolutionType.CONTRADICTION
        elif avg_similarity > 0.9:  # Very stable
            return EvolutionType.REFINEMENT
        elif avg_similarity > 0.8:  # Moderate stability
            return EvolutionType.KNOWLEDGE_GROWTH
        elif avg_similarity > 0.6:  # Lower stability
            return EvolutionType.CONCEPT_DRIFT
        else:
            return EvolutionType.SYNTHESIS
    
    async def _calculate_stability_score(self, embeddings: np.ndarray) -> float:
        """Calculate how stable the concept evolution is."""
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        # Stability is average similarity
        return float(np.mean(similarities))
    
    async def _determine_trend_direction(self, embeddings: np.ndarray) -> TrendDirection:
        """Determine the overall trend direction of concept evolution."""
        if len(embeddings) < 3:
            return TrendDirection.STABLE
        
        # Project to 1D using PCA
        pca = PCA(n_components=1)
        trend_values = pca.fit_transform(embeddings).flatten()
        
        # Calculate trend
        x = np.arange(len(trend_values))
        slope, _, r_value, _, _ = stats.linregress(x, trend_values)
        
        # Classify based on slope and correlation
        if abs(r_value) < 0.3:  # Weak correlation
            # Check for cyclical patterns
            if len(trend_values) > 6:
                peaks, _ = find_peaks(trend_values)
                valleys, _ = find_peaks(-trend_values)
                if len(peaks) > 1 and len(valleys) > 1:
                    return TrendDirection.CYCLICAL
            
            return TrendDirection.VOLATILE if np.std(trend_values) > 0.5 else TrendDirection.STABLE
        
        elif slope > 0.1:
            return TrendDirection.INCREASING
        elif slope < -0.1:
            return TrendDirection.DECREASING
        else:
            return TrendDirection.STABLE
    
    async def _extract_key_changes(
        self,
        memories: List[Memory],
        concept: str
    ) -> List[str]:
        """Extract key changes in how a concept is discussed."""
        changes = []
        
        # Simple approach: look for sentiment or descriptive changes
        prev_descriptors = set()
        
        for i, memory in enumerate(memories):
            doc = self.nlp(memory.content)
            
            # Find sentences with the concept
            concept_sentences = [
                sent for sent in doc.sents 
                if concept.lower() in sent.text.lower()
            ]
            
            if not concept_sentences:
                continue
            
            # Extract descriptive words around the concept
            current_descriptors = set()
            for sent in concept_sentences:
                sent_doc = self.nlp(sent.text)
                for token in sent_doc:
                    if (token.pos_ in ['ADJ', 'VERB'] and 
                        not token.is_stop and 
                        len(token.text) > 3):
                        current_descriptors.add(token.lemma_.lower())
            
            # Compare with previous descriptors
            if i > 0:
                new_descriptors = current_descriptors - prev_descriptors
                removed_descriptors = prev_descriptors - current_descriptors
                
                if new_descriptors:
                    changes.append(f"New perspectives: {', '.join(list(new_descriptors)[:3])}")
                
                if removed_descriptors and len(removed_descriptors) > 2:
                    changes.append(f"Shifted away from: {', '.join(list(removed_descriptors)[:3])}")
            
            prev_descriptors = current_descriptors
        
        return changes[:5]  # Limit to 5 key changes
    
    async def _analyze_learning_progressions(
        self,
        memories: List[Memory],
        temporal_clusters: List[TemporalCluster]
    ) -> List[LearningProgression]:
        """Analyze learning progressions over time."""
        progressions = []
        
        # Group memories by topic
        topic_memories = defaultdict(list)
        
        for cluster in temporal_clusters:
            for topic in cluster.dominant_topics:
                topic_memories[topic].extend(cluster.memories)
        
        # Analyze progression for each topic
        for topic, topic_mems in topic_memories.items():
            if len(topic_mems) < 5:  # Need minimum memories
                continue
            
            # Sort by time
            topic_mems.sort(key=lambda m: m.created_at)
            
            # Analyze complexity progression
            complexity_scores = await self._calculate_complexity_scores(topic_mems)
            
            # Determine trend
            if len(complexity_scores) > 2:
                x = np.arange(len(complexity_scores))
                slope, _, r_value, _, _ = stats.linregress(x, complexity_scores)
                
                if slope > 0.1 and r_value > 0.3:
                    complexity_trend = TrendDirection.INCREASING
                elif slope < -0.1 and r_value > 0.3:
                    complexity_trend = TrendDirection.DECREASING
                else:
                    complexity_trend = TrendDirection.STABLE
            else:
                complexity_trend = TrendDirection.STABLE
            
            # Calculate progression score
            progression_score = await self._calculate_progression_score(
                topic_mems, 
                complexity_scores
            )
            
            # Identify milestones
            milestones = await self._identify_learning_milestones(
                topic_mems, 
                complexity_scores
            )
            
            # Identify knowledge gaps
            knowledge_gaps = await self._identify_topic_knowledge_gaps(topic_mems, topic)
            
            progression = LearningProgression(
                topic=topic,
                start_date=topic_mems[0].created_at,
                end_date=topic_mems[-1].created_at,
                progression_score=progression_score,
                complexity_trend=complexity_trend,
                milestones=milestones,
                knowledge_gaps=knowledge_gaps
            )
            
            progressions.append(progression)
        
        return progressions
    
    async def _calculate_complexity_scores(self, memories: List[Memory]) -> List[float]:
        """Calculate complexity scores for memories."""
        scores = []
        
        for memory in memories:
            doc = self.nlp(memory.content)
            
            # Complexity indicators
            sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
            avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
            
            # Vocabulary complexity
            complex_words = sum(1 for token in doc if len(token.text) > 6)
            total_words = len([token for token in doc if token.is_alpha])
            vocab_complexity = complex_words / max(total_words, 1)
            
            # Technical terms (entities, technical vocabulary)
            technical_score = len(doc.ents) / max(total_words, 1)
            
            # Combine scores
            complexity = (
                min(avg_sentence_length / 20, 1.0) * 0.4 +  # Sentence complexity
                vocab_complexity * 0.4 +  # Vocabulary complexity
                technical_score * 0.2  # Technical content
            )
            
            scores.append(complexity)
        
        return scores
    
    async def _calculate_progression_score(
        self,
        memories: List[Memory],
        complexity_scores: List[float]
    ) -> float:
        """Calculate overall learning progression score."""
        if len(memories) < 2:
            return 0.5
        
        # Factor 1: Complexity progression
        complexity_trend = 0.0
        if len(complexity_scores) > 1:
            x = np.arange(len(complexity_scores))
            slope, _, r_value, _, _ = stats.linregress(x, complexity_scores)
            complexity_trend = max(0, slope) * abs(r_value)
        
        # Factor 2: Temporal consistency (regular learning)
        time_gaps = []
        for i in range(1, len(memories)):
            gap = (memories[i].created_at - memories[i-1].created_at).days
            time_gaps.append(gap)
        
        consistency_score = 1.0 / (1.0 + np.std(time_gaps)) if time_gaps else 0.5
        
        # Factor 3: Volume growth
        time_span = (memories[-1].created_at - memories[0].created_at).days
        volume_score = min(1.0, len(memories) / max(time_span / 30, 1))  # Memories per month
        
        # Combine factors
        progression_score = (
            complexity_trend * 0.4 +
            consistency_score * 0.3 +
            volume_score * 0.3
        )
        
        return min(1.0, progression_score)
    
    async def _identify_learning_milestones(
        self,
        memories: List[Memory],
        complexity_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Identify key learning milestones."""
        milestones = []
        
        if len(memories) < 3:
            return milestones
        
        # Find complexity peaks (learning breakthroughs)
        peaks, _ = find_peaks(complexity_scores, height=np.mean(complexity_scores))
        
        for peak_idx in peaks:
            if peak_idx < len(memories):
                milestone = {
                    "date": memories[peak_idx].created_at,
                    "type": "complexity_peak",
                    "description": f"Learning breakthrough in topic complexity",
                    "complexity_score": complexity_scores[peak_idx],
                    "memory_id": memories[peak_idx].id
                }
                milestones.append(milestone)
        
        # Find periods of high activity (learning spurts)
        if len(memories) > 5:
            time_windows = []
            window_size = min(7, len(memories) // 3)  # Days or memories/3
            
            for i in range(len(memories) - window_size + 1):
                window_memories = memories[i:i + window_size]
                time_span = (window_memories[-1].created_at - window_memories[0].created_at).days
                if time_span > 0:
                    activity_rate = len(window_memories) / time_span
                    time_windows.append((i, activity_rate))
            
            # Find high activity periods
            if time_windows:
                activity_rates = [rate for _, rate in time_windows]
                threshold = np.mean(activity_rates) + np.std(activity_rates)
                
                for i, rate in time_windows:
                    if rate > threshold:
                        milestone = {
                            "date": memories[i].created_at,
                            "type": "learning_spurt",
                            "description": f"High learning activity period",
                            "activity_rate": rate,
                            "memory_count": window_size
                        }
                        milestones.append(milestone)
        
        return milestones[:5]  # Limit to top 5 milestones
    
    async def _identify_topic_knowledge_gaps(
        self,
        memories: List[Memory],
        topic: str
    ) -> List[str]:
        """Identify knowledge gaps for a specific topic."""
        gaps = []
        
        # Extract all subtopics/aspects mentioned
        subtopics = set()
        for memory in memories:
            doc = self.nlp(memory.content)
            
            # Extract noun phrases related to the topic
            for chunk in doc.noun_chunks:
                if topic.lower() in chunk.text.lower() and len(chunk.text.split()) > 1:
                    subtopics.add(chunk.text.lower())
        
        # Common knowledge areas that might be missing
        # This is a simplified approach - in practice, you'd use domain knowledge
        potential_aspects = [
            "definition", "history", "applications", "examples", 
            "benefits", "drawbacks", "comparison", "future", "process"
        ]
        
        # Check which aspects are under-represented
        aspect_coverage = {}
        for aspect in potential_aspects:
            coverage = sum(1 for memory in memories if aspect in memory.content.lower())
            aspect_coverage[aspect] = coverage
        
        # Identify gaps (aspects with low coverage)
        avg_coverage = np.mean(list(aspect_coverage.values()))
        for aspect, coverage in aspect_coverage.items():
            if coverage < avg_coverage * 0.5:  # Less than half average coverage
                gaps.append(f"Limited coverage of {topic} {aspect}")
        
        return gaps[:3]  # Top 3 gaps
    
    async def _generate_temporal_insights(
        self,
        memories: List[Memory],
        concept_evolutions: List[ConceptEvolution],
        temporal_clusters: List[TemporalCluster]
    ) -> List[TemporalInsight]:
        """Generate insights from temporal analysis."""
        insights = []
        
        if not memories:
            return insights
        
        # Insight: Most active learning periods
        if len(temporal_clusters) > 1:
            cluster_sizes = [(c.time_period, len(c.memories)) for c in temporal_clusters]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            
            if cluster_sizes:
                most_active_period, memory_count = cluster_sizes[0]
                insight = TemporalInsight(
                    insight_type="peak_learning_period",
                    description=f"Most active learning period with {memory_count} memories",
                    time_span=most_active_period,
                    confidence=0.9,
                    supporting_memories=[],  # Would need to track specific memories
                    trend_data={"memory_count": memory_count}
                )
                insights.append(insight)
        
        # Insight: Concept stability analysis
        stable_concepts = [c for c in concept_evolutions if c.stability_score > 0.8]
        if stable_concepts:
            concept_names = [c.concept for c in stable_concepts[:3]]
            insight = TemporalInsight(
                insight_type="stable_concepts",
                description=f"Stable understanding of: {', '.join(concept_names)}",
                time_span=(memories[0].created_at, memories[-1].created_at),
                confidence=np.mean([c.stability_score for c in stable_concepts]),
                supporting_memories=[],
                trend_data={"stable_concepts": concept_names}
            )
            insights.append(insight)
        
        # Insight: Rapid concept evolution
        evolving_concepts = [c for c in concept_evolutions if c.stability_score < 0.5]
        if evolving_concepts:
            concept_names = [c.concept for c in evolving_concepts[:3]]
            insight = TemporalInsight(
                insight_type="evolving_concepts",
                description=f"Rapidly evolving understanding of: {', '.join(concept_names)}",
                time_span=(memories[0].created_at, memories[-1].created_at),
                confidence=0.8,
                supporting_memories=[],
                trend_data={"evolving_concepts": concept_names}
            )
            insights.append(insight)
        
        # Insight: Learning consistency
        time_gaps = []
        for i in range(1, len(memories)):
            gap = (memories[i].created_at - memories[i-1].created_at).days
            time_gaps.append(gap)
        
        if time_gaps:
            avg_gap = np.mean(time_gaps)
            consistency = 1.0 / (1.0 + np.std(time_gaps))
            
            if consistency > 0.7:
                pattern = "consistent"
            elif consistency > 0.4:
                pattern = "moderately consistent"
            else:
                pattern = "irregular"
            
            insight = TemporalInsight(
                insight_type="learning_consistency",
                description=f"Learning pattern is {pattern} with average {avg_gap:.1f} day gaps",
                time_span=(memories[0].created_at, memories[-1].created_at),
                confidence=consistency,
                supporting_memories=[],
                trend_data={
                    "average_gap_days": avg_gap,
                    "consistency_score": consistency,
                    "pattern": pattern
                }
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_activity_patterns(self, memories: List[Memory]) -> Dict[str, Any]:
        """Analyze activity patterns over time."""
        if not memories:
            return {}
        
        timestamps = [m.created_at for m in memories]
        
        # Create time series
        df = pd.DataFrame({
            'timestamp': timestamps,
            'memory_count': [1] * len(timestamps)
        })
        df.set_index('timestamp', inplace=True)
        
        # Resample by day
        daily_counts = df.resample('D').sum()
        
        patterns = {
            'total_days_active': len(daily_counts[daily_counts['memory_count'] > 0]),
            'max_daily_memories': int(daily_counts['memory_count'].max()),
            'avg_daily_memories': float(daily_counts['memory_count'].mean()),
            'most_active_day': daily_counts['memory_count'].idxmax().strftime('%Y-%m-%d'),
            'activity_distribution': {
                'weekday': self._analyze_weekday_pattern(timestamps),
                'hourly': self._analyze_hourly_pattern(timestamps)
            }
        }
        
        return patterns
    
    def _analyze_weekday_pattern(self, timestamps: List[datetime]) -> Dict[str, int]:
        """Analyze memory creation by day of week."""
        weekday_counts = defaultdict(int)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for ts in timestamps:
            weekday_counts[day_names[ts.weekday()]] += 1
        
        return dict(weekday_counts)
    
    def _analyze_hourly_pattern(self, timestamps: List[datetime]) -> Dict[int, int]:
        """Analyze memory creation by hour of day."""
        hourly_counts = defaultdict(int)
        
        for ts in timestamps:
            hourly_counts[ts.hour] += 1
        
        return dict(hourly_counts)
    
    async def _calculate_knowledge_velocity(
        self,
        memories: List[Memory],
        temporal_clusters: List[TemporalCluster]
    ) -> Dict[str, float]:
        """Calculate rate of knowledge acquisition over time."""
        if not temporal_clusters:
            return {}
        
        velocity_metrics = {}
        
        # Overall memory velocity (memories per day)
        total_days = (memories[-1].created_at - memories[0].created_at).days
        if total_days > 0:
            velocity_metrics['memories_per_day'] = len(memories) / total_days
        
        # Topic velocity (new topics per time period)
        all_topics = set()
        topic_timeline = []
        
        for cluster in temporal_clusters:
            new_topics = set(cluster.dominant_topics) - all_topics
            all_topics.update(cluster.dominant_topics)
            
            period_days = (cluster.time_period[1] - cluster.time_period[0]).days
            if period_days > 0:
                topic_timeline.append({
                    'period': cluster.time_period,
                    'new_topics': len(new_topics),
                    'topics_per_day': len(new_topics) / period_days
                })
        
        if topic_timeline:
            velocity_metrics['avg_new_topics_per_day'] = np.mean([
                t['topics_per_day'] for t in topic_timeline
            ])
        
        # Novelty velocity (how quickly new concepts are introduced)
        novelty_scores = [c.novelty_score for c in temporal_clusters if c.novelty_score > 0]
        if novelty_scores:
            velocity_metrics['avg_novelty_score'] = np.mean(novelty_scores)
        
        return velocity_metrics
    
    async def _analyze_memory_lifecycle(self, memories: List[Memory]) -> Dict[str, Any]:
        """Analyze memory lifecycle patterns."""
        if not memories:
            return {}
        
        # Calculate memory age distribution
        now = datetime.utcnow()
        memory_ages = [(now - m.created_at).days for m in memories]
        
        lifecycle_data = {
            'age_distribution': {
                'min_age_days': min(memory_ages),
                'max_age_days': max(memory_ages),
                'avg_age_days': np.mean(memory_ages),
                'median_age_days': np.median(memory_ages)
            },
            'retention_patterns': {
                'memories_last_7_days': sum(1 for age in memory_ages if age <= 7),
                'memories_last_30_days': sum(1 for age in memory_ages if age <= 30),
                'memories_last_90_days': sum(1 for age in memory_ages if age <= 90)
            }
        }
        
        # Analyze memory content evolution (length, complexity)
        memory_lengths = [len(m.content) for m in memories]
        if memory_lengths:
            lifecycle_data['content_evolution'] = {
                'avg_length': np.mean(memory_lengths),
                'length_trend': self._calculate_length_trend(memories)
            }
        
        return lifecycle_data
    
    def _calculate_length_trend(self, memories: List[Memory]) -> str:
        """Calculate trend in memory content length over time."""
        if len(memories) < 3:
            return "insufficient_data"
        
        lengths = [len(m.content) for m in memories]
        x = np.arange(len(lengths))
        slope, _, r_value, _, _ = stats.linregress(x, lengths)
        
        if abs(r_value) < 0.3:
            return "stable"
        elif slope > 10:  # More than 10 characters increase per memory
            return "increasing"
        elif slope < -10:
            return "decreasing"
        else:
            return "stable"