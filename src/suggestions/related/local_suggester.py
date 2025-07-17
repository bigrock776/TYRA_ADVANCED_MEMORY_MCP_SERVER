"""
Local Memory Suggester for Related Content.

This module provides intelligent suggestions for related memories using local
similarity algorithms, contextual relevance analysis, ML-based ranking,
and feedback learning for continuous improvement.
"""

import asyncio
import math
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import spacy
from collections import defaultdict, Counter
import json
import hashlib

import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator

from ...core.embeddings.embedder import Embedder
from ...core.memory.manager import MemoryManager
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class SuggestionType(str, Enum):
    """Types of memory suggestions."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    ENTITY_OVERLAP = "entity_overlap"
    TOPIC_SIMILARITY = "topic_similarity"
    USER_BEHAVIOR = "user_behavior"
    COLLABORATIVE = "collaborative"


class RelevanceScore(str, Enum):
    """Relevance score categories."""
    VERY_HIGH = "very_high"    # 0.9-1.0
    HIGH = "high"              # 0.7-0.89
    MEDIUM = "medium"          # 0.5-0.69
    LOW = "low"                # 0.3-0.49
    VERY_LOW = "very_low"      # 0.0-0.29


@dataclass
class SuggestionContext:
    """Context for generating suggestions."""
    user_id: str
    current_memory_id: Optional[str] = None
    query_text: Optional[str] = None
    recent_memories: List[str] = field(default_factory=list)
    user_topics: Set[str] = field(default_factory=set)
    interaction_history: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MemorySuggestion(BaseModel):
    """Structured memory suggestion with metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    memory_id: str = Field(..., description="ID of suggested memory")
    suggestion_type: SuggestionType = Field(..., description="Type of suggestion")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    relevance_category: RelevanceScore = Field(..., description="Relevance category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in suggestion")
    explanation: str = Field(..., min_length=10, description="Why this memory is suggested")
    similarity_factors: Dict[str, float] = Field(..., description="Breakdown of similarity factors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Suggestion timestamp")
    
    @field_validator('relevance_score')
    @classmethod
    def set_relevance_category(cls, v):
        """Automatically set relevance category based on score."""
        return v
    
    def __post_init__(self):
        """Set relevance category after initialization."""
        if self.relevance_score >= 0.9:
            self.relevance_category = RelevanceScore.VERY_HIGH
        elif self.relevance_score >= 0.7:
            self.relevance_category = RelevanceScore.HIGH
        elif self.relevance_score >= 0.5:
            self.relevance_category = RelevanceScore.MEDIUM
        elif self.relevance_score >= 0.3:
            self.relevance_category = RelevanceScore.LOW
        else:
            self.relevance_category = RelevanceScore.VERY_LOW


class SuggestionResult(BaseModel):
    """Complete suggestion result with analytics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    suggestions: List[MemorySuggestion] = Field(..., description="List of memory suggestions")
    total_candidates: int = Field(..., ge=0, description="Total memories considered")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    suggestion_strategies: List[str] = Field(..., description="Strategies used for suggestions")
    context_analysis: Dict[str, Any] = Field(..., description="Analysis of suggestion context")
    feedback_opportunities: List[str] = Field(default_factory=list, description="Ways to improve suggestions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Result timestamp")


class LocalMemorySuggester:
    """
    Local Memory Suggester for Related Content.
    
    Provides intelligent memory suggestions using multiple local algorithms:
    - Semantic similarity via embeddings
    - Temporal proximity analysis
    - Entity overlap detection
    - Topic modeling with LDA
    - User behavior patterns
    - Collaborative filtering
    """
    
    def __init__(
        self,
        embedder: Embedder,
        memory_manager: MemoryManager,
        cache_size: int = 1000
    ):
        """Initialize local memory suggester."""
        self.embedder = embedder
        self.memory_manager = memory_manager
        self.cache_size = cache_size
        
        # Similarity computation
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=20,
            random_state=42
        )
        
        # Machine learning models
        self.relevance_model: Optional[RandomForestRegressor] = None
        self.user_preference_models: Dict[str, Any] = {}
        
        # Caching
        self.similarity_cache: Dict[str, Dict[str, float]] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.topic_cache: Dict[str, List[float]] = {}
        
        # Analytics and feedback
        self.suggestion_history: List[Dict[str, Any]] = []
        self.feedback_data: List[Dict[str, Any]] = []
        self.user_interactions: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic NLP")
            self.nlp = None
        
        # Performance tracking
        self.suggestion_stats = {
            'total_suggestions': 0,
            'successful_suggestions': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'feedback_received': 0,
            'model_improvements': 0
        }
        
        logger.info("Initialized LocalMemorySuggester")
    
    async def suggest_related_memories(
        self,
        context: SuggestionContext,
        max_suggestions: int = 10,
        min_relevance: float = 0.3,
        diversify: bool = True
    ) -> SuggestionResult:
        """
        Generate related memory suggestions based on context.
        
        Args:
            context: Suggestion context with user and memory information
            max_suggestions: Maximum number of suggestions to return
            min_relevance: Minimum relevance score for suggestions
            diversify: Whether to diversify suggestions across types
            
        Returns:
            Complete suggestion result with ranked memories
        """
        start_time = datetime.utcnow()
        self.suggestion_stats['total_suggestions'] += 1
        
        try:
            # Get candidate memories
            candidates = await self._get_candidate_memories(context)
            
            if not candidates:
                return SuggestionResult(
                    suggestions=[],
                    total_candidates=0,
                    processing_time_ms=0.0,
                    suggestion_strategies=[],
                    context_analysis={"status": "no_candidates"},
                    feedback_opportunities=["Add more memories to improve suggestions"]
                )
            
            # Generate suggestions using multiple strategies
            suggestions = []
            strategies_used = []
            
            # Strategy 1: Semantic similarity
            semantic_suggestions = await self._semantic_similarity_suggestions(
                context, candidates, max_suggestions // 2
            )
            suggestions.extend(semantic_suggestions)
            if semantic_suggestions:
                strategies_used.append("semantic_similarity")
            
            # Strategy 2: Temporal proximity
            temporal_suggestions = await self._temporal_proximity_suggestions(
                context, candidates, max_suggestions // 4
            )
            suggestions.extend(temporal_suggestions)
            if temporal_suggestions:
                strategies_used.append("temporal_proximity")
            
            # Strategy 3: Entity overlap
            entity_suggestions = await self._entity_overlap_suggestions(
                context, candidates, max_suggestions // 4
            )
            suggestions.extend(entity_suggestions)
            if entity_suggestions:
                strategies_used.append("entity_overlap")
            
            # Strategy 4: Topic similarity
            topic_suggestions = await self._topic_similarity_suggestions(
                context, candidates, max_suggestions // 4
            )
            suggestions.extend(topic_suggestions)
            if topic_suggestions:
                strategies_used.append("topic_similarity")
            
            # Strategy 5: User behavior patterns
            behavior_suggestions = await self._user_behavior_suggestions(
                context, candidates, max_suggestions // 4
            )
            suggestions.extend(behavior_suggestions)
            if behavior_suggestions:
                strategies_used.append("user_behavior")
            
            # Remove duplicates and filter by relevance
            unique_suggestions = self._deduplicate_suggestions(suggestions)
            filtered_suggestions = [
                s for s in unique_suggestions 
                if s.relevance_score >= min_relevance
            ]
            
            # Apply ML-based reranking if model available
            if self.relevance_model and len(filtered_suggestions) > 1:
                filtered_suggestions = await self._ml_rerank_suggestions(
                    context, filtered_suggestions
                )
                strategies_used.append("ml_reranking")
            
            # Diversify suggestions if requested
            if diversify and len(filtered_suggestions) > max_suggestions:
                filtered_suggestions = self._diversify_suggestions(
                    filtered_suggestions, max_suggestions
                )
                strategies_used.append("diversification")
            
            # Limit to max suggestions
            final_suggestions = filtered_suggestions[:max_suggestions]
            
            # Update relevance categories
            for suggestion in final_suggestions:
                suggestion.__post_init__()
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.suggestion_stats['average_processing_time'] = (
                (self.suggestion_stats['average_processing_time'] * 
                 (self.suggestion_stats['total_suggestions'] - 1) + processing_time) /
                self.suggestion_stats['total_suggestions']
            )
            
            # Context analysis
            context_analysis = {
                "user_id": context.user_id,
                "has_current_memory": context.current_memory_id is not None,
                "has_query": context.query_text is not None,
                "recent_memories_count": len(context.recent_memories),
                "user_topics_count": len(context.user_topics),
                "candidates_processed": len(candidates),
                "suggestions_generated": len(final_suggestions)
            }
            
            # Feedback opportunities
            feedback_opportunities = []
            if len(final_suggestions) < max_suggestions // 2:
                feedback_opportunities.append("Consider adding more diverse memories")
            if not any(s.relevance_score > 0.8 for s in final_suggestions):
                feedback_opportunities.append("High-quality matches not found - refine search")
            if len(strategies_used) < 3:
                feedback_opportunities.append("Limited strategy coverage - expand memory base")
            
            # Store suggestion for feedback learning
            suggestion_record = {
                "context": context,
                "suggestions": final_suggestions,
                "strategies": strategies_used,
                "timestamp": datetime.utcnow()
            }
            self.suggestion_history.append(suggestion_record)
            
            # Limit history size
            if len(self.suggestion_history) > 1000:
                self.suggestion_history = self.suggestion_history[-500:]
            
            if final_suggestions:
                self.suggestion_stats['successful_suggestions'] += 1
            
            logger.info(
                "Generated memory suggestions",
                user_id=context.user_id,
                suggestions_count=len(final_suggestions),
                processing_time_ms=processing_time,
                strategies=strategies_used
            )
            
            return SuggestionResult(
                suggestions=final_suggestions,
                total_candidates=len(candidates),
                processing_time_ms=processing_time,
                suggestion_strategies=strategies_used,
                context_analysis=context_analysis,
                feedback_opportunities=feedback_opportunities
            )
            
        except Exception as e:
            logger.error("Error generating memory suggestions", error=str(e))
            return SuggestionResult(
                suggestions=[],
                total_candidates=0,
                processing_time_ms=0.0,
                suggestion_strategies=[],
                context_analysis={"error": str(e)},
                feedback_opportunities=["System error - please try again"]
            )
    
    async def _get_candidate_memories(self, context: SuggestionContext) -> List[Dict[str, Any]]:
        """Get candidate memories for suggestion generation."""
        try:
            # Get recent memories for the user
            memories = await self.memory_manager.get_memories_for_user(
                context.user_id,
                limit=500  # Get substantial candidate set
            )
            
            # Filter out current memory if specified
            if context.current_memory_id:
                memories = [m for m in memories if m.id != context.current_memory_id]
            
            # Convert to dictionaries with necessary fields
            candidates = []
            for memory in memories:
                candidate = {
                    "id": memory.id,
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "tags": getattr(memory, 'tags', []),
                    "entities": getattr(memory, 'entities', [])
                }
                candidates.append(candidate)
            
            logger.debug(f"Retrieved {len(candidates)} candidate memories")
            return candidates
            
        except Exception as e:
            logger.error("Error retrieving candidate memories", error=str(e))
            return []
    
    async def _semantic_similarity_suggestions(
        self,
        context: SuggestionContext,
        candidates: List[Dict[str, Any]],
        max_suggestions: int
    ) -> List[MemorySuggestion]:
        """Generate suggestions based on semantic similarity."""
        if not context.query_text and not context.current_memory_id:
            return []
        
        try:
            # Get reference text for comparison
            reference_text = context.query_text
            if not reference_text and context.current_memory_id:
                # Get current memory content
                current_memory = await self.memory_manager.get_memory(context.current_memory_id)
                if current_memory:
                    reference_text = current_memory.content
            
            if not reference_text:
                return []
            
            # Get embedding for reference text
            reference_embedding = await self._get_or_compute_embedding(reference_text)
            if reference_embedding is None:
                return []
            
            # Compute similarities
            suggestions = []
            for candidate in candidates:
                try:
                    candidate_embedding = await self._get_or_compute_embedding(candidate["content"])
                    if candidate_embedding is None:
                        continue
                    
                    # Compute cosine similarity
                    similarity = float(cosine_similarity(
                        reference_embedding.reshape(1, -1),
                        candidate_embedding.reshape(1, -1)
                    )[0][0])
                    
                    if similarity > 0.3:  # Threshold for consideration
                        explanation = f"Semantically similar (similarity: {similarity:.2f})"
                        
                        suggestion = MemorySuggestion(
                            memory_id=candidate["id"],
                            suggestion_type=SuggestionType.SEMANTIC_SIMILARITY,
                            relevance_score=similarity,
                            relevance_category=RelevanceScore.MEDIUM,  # Will be updated
                            confidence=min(similarity + 0.1, 1.0),
                            explanation=explanation,
                            similarity_factors={"semantic_similarity": similarity},
                            metadata={
                                "reference_length": len(reference_text),
                                "candidate_length": len(candidate["content"]),
                                "computation_method": "cosine_similarity"
                            }
                        )
                        suggestions.append(suggestion)
                        
                except Exception as e:
                    logger.warning(f"Error computing similarity for candidate {candidate['id']}: {e}")
                    continue
            
            # Sort by relevance and return top suggestions
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error("Error in semantic similarity suggestions", error=str(e))
            return []
    
    async def _temporal_proximity_suggestions(
        self,
        context: SuggestionContext,
        candidates: List[Dict[str, Any]],
        max_suggestions: int
    ) -> List[MemorySuggestion]:
        """Generate suggestions based on temporal proximity."""
        if not context.current_memory_id:
            return []
        
        try:
            # Get current memory timestamp
            current_memory = await self.memory_manager.get_memory(context.current_memory_id)
            if not current_memory:
                return []
            
            reference_time = current_memory.created_at
            
            suggestions = []
            for candidate in candidates:
                try:
                    candidate_time = candidate["created_at"]
                    
                    # Calculate time difference
                    time_diff = abs((reference_time - candidate_time).total_seconds())
                    
                    # Convert to relevance score (closer in time = higher score)
                    # Use exponential decay: score = exp(-time_diff / time_scale)
                    time_scale = 7 * 24 * 3600  # 1 week in seconds
                    temporal_score = math.exp(-time_diff / time_scale)
                    
                    if temporal_score > 0.1:  # Threshold for consideration
                        time_description = self._humanize_time_diff(time_diff)
                        explanation = f"Created {time_description} (temporal proximity: {temporal_score:.2f})"
                        
                        suggestion = MemorySuggestion(
                            memory_id=candidate["id"],
                            suggestion_type=SuggestionType.TEMPORAL_PROXIMITY,
                            relevance_score=temporal_score,
                            relevance_category=RelevanceScore.MEDIUM,
                            confidence=temporal_score,
                            explanation=explanation,
                            similarity_factors={"temporal_proximity": temporal_score},
                            metadata={
                                "time_difference_seconds": time_diff,
                                "reference_time": reference_time.isoformat(),
                                "candidate_time": candidate_time.isoformat()
                            }
                        )
                        suggestions.append(suggestion)
                        
                except Exception as e:
                    logger.warning(f"Error computing temporal proximity for candidate {candidate['id']}: {e}")
                    continue
            
            # Sort by relevance and return top suggestions
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error("Error in temporal proximity suggestions", error=str(e))
            return []
    
    async def _entity_overlap_suggestions(
        self,
        context: SuggestionContext,
        candidates: List[Dict[str, Any]],
        max_suggestions: int
    ) -> List[MemorySuggestion]:
        """Generate suggestions based on entity overlap."""
        try:
            # Get reference entities
            reference_entities = set()
            
            if context.query_text:
                reference_entities.update(self._extract_entities(context.query_text))
            
            if context.current_memory_id:
                current_memory = await self.memory_manager.get_memory(context.current_memory_id)
                if current_memory:
                    reference_entities.update(self._extract_entities(current_memory.content))
                    # Also use any stored entities
                    if hasattr(current_memory, 'entities'):
                        reference_entities.update(current_memory.entities)
            
            if not reference_entities:
                return []
            
            suggestions = []
            for candidate in candidates:
                try:
                    # Extract entities from candidate
                    candidate_entities = set(self._extract_entities(candidate["content"]))
                    if "entities" in candidate:
                        candidate_entities.update(candidate["entities"])
                    
                    if candidate_entities:
                        # Calculate entity overlap
                        overlap = len(reference_entities.intersection(candidate_entities))
                        union = len(reference_entities.union(candidate_entities))
                        
                        if overlap > 0 and union > 0:
                            # Jaccard similarity
                            entity_score = overlap / union
                            
                            # Boost score based on number of overlapping entities
                            entity_score *= (1 + math.log(overlap + 1) / 10)
                            entity_score = min(entity_score, 1.0)
                            
                            if entity_score > 0.1:
                                overlapping_entities = reference_entities.intersection(candidate_entities)
                                explanation = f"Shares entities: {', '.join(list(overlapping_entities)[:3])} (overlap: {entity_score:.2f})"
                                
                                suggestion = MemorySuggestion(
                                    memory_id=candidate["id"],
                                    suggestion_type=SuggestionType.ENTITY_OVERLAP,
                                    relevance_score=entity_score,
                                    relevance_category=RelevanceScore.MEDIUM,
                                    confidence=entity_score,
                                    explanation=explanation,
                                    similarity_factors={"entity_overlap": entity_score},
                                    metadata={
                                        "overlapping_entities": list(overlapping_entities),
                                        "reference_entity_count": len(reference_entities),
                                        "candidate_entity_count": len(candidate_entities),
                                        "overlap_count": overlap
                                    }
                                )
                                suggestions.append(suggestion)
                        
                except Exception as e:
                    logger.warning(f"Error computing entity overlap for candidate {candidate['id']}: {e}")
                    continue
            
            # Sort by relevance and return top suggestions
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error("Error in entity overlap suggestions", error=str(e))
            return []
    
    async def _topic_similarity_suggestions(
        self,
        context: SuggestionContext,
        candidates: List[Dict[str, Any]],
        max_suggestions: int
    ) -> List[MemorySuggestion]:
        """Generate suggestions based on topic similarity using LDA."""
        try:
            # Prepare text corpus
            texts = []
            candidate_indices = {}
            
            # Add reference text
            reference_text = context.query_text or ""
            if context.current_memory_id:
                current_memory = await self.memory_manager.get_memory(context.current_memory_id)
                if current_memory:
                    reference_text = current_memory.content
            
            if not reference_text:
                return []
            
            texts.append(reference_text)
            
            # Add candidate texts
            for i, candidate in enumerate(candidates):
                texts.append(candidate["content"])
                candidate_indices[i + 1] = candidate["id"]  # +1 because reference is at index 0
            
            if len(texts) < 2:
                return []
            
            # Compute TF-IDF
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            except ValueError as e:
                logger.warning(f"TF-IDF computation failed: {e}")
                return []
            
            # Compute LDA topics
            try:
                lda_topics = self.lda_model.fit_transform(tfidf_matrix)
            except Exception as e:
                logger.warning(f"LDA computation failed: {e}")
                return []
            
            # Get reference topic distribution
            reference_topics = lda_topics[0]
            
            suggestions = []
            for candidate_idx, memory_id in candidate_indices.items():
                try:
                    candidate_topics = lda_topics[candidate_idx]
                    
                    # Compute topic similarity (cosine similarity of topic distributions)
                    topic_similarity = float(cosine_similarity(
                        reference_topics.reshape(1, -1),
                        candidate_topics.reshape(1, -1)
                    )[0][0])
                    
                    if topic_similarity > 0.2:
                        # Find dominant topics
                        ref_dominant = np.argmax(reference_topics)
                        cand_dominant = np.argmax(candidate_topics)
                        
                        explanation = f"Similar topics (similarity: {topic_similarity:.2f})"
                        if ref_dominant == cand_dominant:
                            explanation += f" - both focus on topic {ref_dominant}"
                        
                        suggestion = MemorySuggestion(
                            memory_id=memory_id,
                            suggestion_type=SuggestionType.TOPIC_SIMILARITY,
                            relevance_score=topic_similarity,
                            relevance_category=RelevanceScore.MEDIUM,
                            confidence=topic_similarity,
                            explanation=explanation,
                            similarity_factors={"topic_similarity": topic_similarity},
                            metadata={
                                "reference_dominant_topic": int(ref_dominant),
                                "candidate_dominant_topic": int(cand_dominant),
                                "topic_distribution": candidate_topics.tolist()
                            }
                        )
                        suggestions.append(suggestion)
                        
                except Exception as e:
                    logger.warning(f"Error computing topic similarity for candidate {memory_id}: {e}")
                    continue
            
            # Sort by relevance and return top suggestions
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error("Error in topic similarity suggestions", error=str(e))
            return []
    
    async def _user_behavior_suggestions(
        self,
        context: SuggestionContext,
        candidates: List[Dict[str, Any]],
        max_suggestions: int
    ) -> List[MemorySuggestion]:
        """Generate suggestions based on user behavior patterns."""
        try:
            user_id = context.user_id
            user_data = self.user_interactions.get(user_id, {})
            
            if not user_data:
                return []
            
            # Get user's memory access patterns
            accessed_memories = user_data.get('accessed_memories', {})
            memory_ratings = user_data.get('memory_ratings', {})
            topic_preferences = user_data.get('topic_preferences', {})
            
            suggestions = []
            for candidate in candidates:
                try:
                    memory_id = candidate["id"]
                    behavior_score = 0.0
                    factors = {}
                    
                    # Factor 1: Previous access frequency
                    access_count = accessed_memories.get(memory_id, 0)
                    if access_count > 0:
                        access_score = min(access_count / 10.0, 0.3)  # Max 0.3 from access
                        behavior_score += access_score
                        factors["access_frequency"] = access_score
                    
                    # Factor 2: Previous ratings
                    if memory_id in memory_ratings:
                        rating_score = memory_ratings[memory_id] / 5.0  # Normalize to 0-1
                        behavior_score += rating_score * 0.4  # Max 0.4 from rating
                        factors["user_rating"] = rating_score
                    
                    # Factor 3: Topic preferences
                    candidate_topics = self._extract_topics(candidate["content"])
                    topic_score = 0.0
                    for topic in candidate_topics:
                        if topic in topic_preferences:
                            topic_score += topic_preferences[topic]
                    
                    if candidate_topics:
                        topic_score = min(topic_score / len(candidate_topics), 0.3)  # Max 0.3 from topics
                        behavior_score += topic_score
                        factors["topic_preference"] = topic_score
                    
                    # Factor 4: Recency bias (recently accessed = higher score)
                    last_access = user_data.get('last_access', {}).get(memory_id)
                    if last_access:
                        days_since = (datetime.utcnow() - last_access).days
                        recency_score = max(0, 0.2 * math.exp(-days_since / 30))  # Decay over 30 days
                        behavior_score += recency_score
                        factors["recency"] = recency_score
                    
                    if behavior_score > 0.1:
                        explanation = f"Matches your behavior patterns (score: {behavior_score:.2f})"
                        if "user_rating" in factors:
                            explanation += f" - you rated this {memory_ratings[memory_id]}/5"
                        
                        suggestion = MemorySuggestion(
                            memory_id=memory_id,
                            suggestion_type=SuggestionType.USER_BEHAVIOR,
                            relevance_score=behavior_score,
                            relevance_category=RelevanceScore.MEDIUM,
                            confidence=behavior_score,
                            explanation=explanation,
                            similarity_factors=factors,
                            metadata={
                                "access_count": access_count,
                                "user_rating": memory_ratings.get(memory_id),
                                "dominant_topics": candidate_topics[:3]
                            }
                        )
                        suggestions.append(suggestion)
                        
                except Exception as e:
                    logger.warning(f"Error computing user behavior score for candidate {candidate['id']}: {e}")
                    continue
            
            # Sort by relevance and return top suggestions
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error("Error in user behavior suggestions", error=str(e))
            return []
    
    async def _ml_rerank_suggestions(
        self,
        context: SuggestionContext,
        suggestions: List[MemorySuggestion]
    ) -> List[MemorySuggestion]:
        """Rerank suggestions using trained ML model."""
        try:
            if not self.relevance_model or len(suggestions) < 2:
                return suggestions
            
            # Extract features for each suggestion
            features = []
            for suggestion in suggestions:
                feature_vector = self._extract_suggestion_features(suggestion, context)
                features.append(feature_vector)
            
            # Predict relevance scores
            feature_matrix = np.array(features)
            predicted_scores = self.relevance_model.predict(feature_matrix)
            
            # Update suggestion scores with weighted combination
            for i, suggestion in enumerate(suggestions):
                original_score = suggestion.relevance_score
                predicted_score = max(0, min(1, predicted_scores[i]))  # Clamp to [0,1]
                
                # Weighted combination: 70% original, 30% ML prediction
                combined_score = 0.7 * original_score + 0.3 * predicted_score
                suggestion.relevance_score = combined_score
                suggestion.similarity_factors["ml_prediction"] = predicted_score
            
            # Re-sort by updated scores
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.debug("Reranked suggestions using ML model")
            return suggestions
            
        except Exception as e:
            logger.error("Error in ML reranking", error=str(e))
            return suggestions
    
    def _extract_suggestion_features(
        self,
        suggestion: MemorySuggestion,
        context: SuggestionContext
    ) -> List[float]:
        """Extract feature vector for ML model."""
        features = []
        
        # Basic scores
        features.append(suggestion.relevance_score)
        features.append(suggestion.confidence)
        
        # Similarity factors
        factors = suggestion.similarity_factors
        features.append(factors.get("semantic_similarity", 0.0))
        features.append(factors.get("temporal_proximity", 0.0))
        features.append(factors.get("entity_overlap", 0.0))
        features.append(factors.get("topic_similarity", 0.0))
        
        # Context features
        features.append(1.0 if context.current_memory_id else 0.0)
        features.append(1.0 if context.query_text else 0.0)
        features.append(len(context.recent_memories) / 10.0)  # Normalize
        features.append(len(context.user_topics) / 20.0)  # Normalize
        
        # Suggestion type (one-hot encoding)
        suggestion_types = list(SuggestionType)
        for stype in suggestion_types:
            features.append(1.0 if suggestion.suggestion_type == stype else 0.0)
        
        return features
    
    def _diversify_suggestions(
        self,
        suggestions: List[MemorySuggestion],
        max_suggestions: int
    ) -> List[MemorySuggestion]:
        """Diversify suggestions across different types and characteristics."""
        if len(suggestions) <= max_suggestions:
            return suggestions
        
        # Group by suggestion type
        type_groups = defaultdict(list)
        for suggestion in suggestions:
            type_groups[suggestion.suggestion_type].append(suggestion)
        
        # Select diverse suggestions
        diverse_suggestions = []
        remaining_slots = max_suggestions
        
        # First, take top suggestion from each type
        for stype, group in type_groups.items():
            if remaining_slots > 0 and group:
                diverse_suggestions.append(group[0])
                remaining_slots -= 1
        
        # Fill remaining slots with highest scoring suggestions
        all_remaining = [
            s for s in suggestions 
            if s not in diverse_suggestions
        ]
        all_remaining.sort(key=lambda x: x.relevance_score, reverse=True)
        
        diverse_suggestions.extend(all_remaining[:remaining_slots])
        
        # Sort final list by relevance
        diverse_suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return diverse_suggestions[:max_suggestions]
    
    def _deduplicate_suggestions(self, suggestions: List[MemorySuggestion]) -> List[MemorySuggestion]:
        """Remove duplicate suggestions, keeping the highest scored version."""
        seen_memories = {}
        unique_suggestions = []
        
        for suggestion in suggestions:
            memory_id = suggestion.memory_id
            if memory_id not in seen_memories:
                seen_memories[memory_id] = suggestion
                unique_suggestions.append(suggestion)
            else:
                # Keep the higher scored suggestion
                existing = seen_memories[memory_id]
                if suggestion.relevance_score > existing.relevance_score:
                    unique_suggestions.remove(existing)
                    unique_suggestions.append(suggestion)
                    seen_memories[memory_id] = suggestion
        
        return unique_suggestions
    
    async def _get_or_compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache or compute it."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            self.suggestion_stats['cache_hits'] += 1
            return self.embedding_cache[text_hash]
        
        try:
            embedding = await self.embedder.embed_text(text)
            
            # Cache the embedding
            self.embedding_cache[text_hash] = embedding
            self.suggestion_stats['cache_misses'] += 1
            
            # Limit cache size
            if len(self.embedding_cache) > self.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self.embedding_cache.keys())[:100]
                for key in oldest_keys:
                    del self.embedding_cache[key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        if not self.nlp:
            # Fallback: simple capitalized word extraction
            words = text.split()
            entities = [word.strip('.,!?;:"()[]') for word in words if word[0].isupper() and len(word) > 2]
            return list(set(entities))
        
        try:
            doc = self.nlp(text)
            entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]]
            return list(set(entities))
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics/keywords from text."""
        # Simple topic extraction using TF-IDF
        try:
            # Create a small corpus with just this text
            tfidf = TfidfVectorizer(max_features=10, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform([text])
            
            # Get feature names (topics/keywords)
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top topics
            topic_scores = list(zip(feature_names, scores))
            topic_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [topic for topic, score in topic_scores[:5] if score > 0]
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return []
    
    def _humanize_time_diff(self, seconds: float) -> str:
        """Convert time difference to human-readable format."""
        if seconds < 3600:
            return f"{int(seconds // 60)} minutes ago"
        elif seconds < 86400:
            return f"{int(seconds // 3600)} hours ago"
        elif seconds < 604800:
            return f"{int(seconds // 86400)} days ago"
        elif seconds < 2592000:
            return f"{int(seconds // 604800)} weeks ago"
        else:
            return f"{int(seconds // 2592000)} months ago"
    
    async def record_feedback(
        self,
        suggestion_id: str,
        feedback_type: str,
        feedback_value: Union[float, bool, str],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record user feedback for suggestion improvement."""
        feedback_record = {
            "suggestion_id": suggestion_id,
            "feedback_type": feedback_type,
            "feedback_value": feedback_value,
            "context": context or {},
            "timestamp": datetime.utcnow()
        }
        
        self.feedback_data.append(feedback_record)
        self.suggestion_stats['feedback_received'] += 1
        
        # Trigger model retraining if enough feedback collected
        if len(self.feedback_data) >= 100 and len(self.feedback_data) % 50 == 0:
            await self._retrain_relevance_model()
        
        logger.info(
            "Recorded suggestion feedback",
            suggestion_id=suggestion_id,
            feedback_type=feedback_type,
            total_feedback=len(self.feedback_data)
        )
    
    async def _retrain_relevance_model(self) -> None:
        """Retrain the relevance model using collected feedback."""
        try:
            if len(self.feedback_data) < 50:
                return
            
            # Prepare training data
            X = []
            y = []
            
            for feedback in self.feedback_data:
                if feedback["feedback_type"] == "relevance_rating":
                    # Find corresponding suggestion from history
                    suggestion_found = False
                    for record in self.suggestion_history:
                        for suggestion in record["suggestions"]:
                            if suggestion.memory_id == feedback["suggestion_id"]:
                                feature_vector = self._extract_suggestion_features(
                                    suggestion, record["context"]
                                )
                                X.append(feature_vector)
                                y.append(float(feedback["feedback_value"]))
                                suggestion_found = True
                                break
                        if suggestion_found:
                            break
            
            if len(X) < 20:  # Need minimum training data
                return
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.relevance_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            self.relevance_model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.relevance_model.score(X_train, y_train)
            test_score = self.relevance_model.score(X_test, y_test)
            
            self.suggestion_stats['model_improvements'] += 1
            
            logger.info(
                "Retrained relevance model",
                training_samples=len(X_train),
                train_score=train_score,
                test_score=test_score
            )
            
        except Exception as e:
            logger.error("Error retraining relevance model", error=str(e))
    
    def save_model(self, model_path: str) -> None:
        """Save the trained model to disk."""
        try:
            model_data = {
                "relevance_model": self.relevance_model,
                "tfidf_vectorizer": self.tfidf_vectorizer,
                "lda_model": self.lda_model,
                "user_preference_models": self.user_preference_models,
                "suggestion_stats": self.suggestion_stats
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Saved suggestion models to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from disk."""
        try:
            model_data = joblib.load(model_path)
            
            self.relevance_model = model_data.get("relevance_model")
            self.tfidf_vectorizer = model_data.get("tfidf_vectorizer", self.tfidf_vectorizer)
            self.lda_model = model_data.get("lda_model", self.lda_model)
            self.user_preference_models = model_data.get("user_preference_models", {})
            self.suggestion_stats.update(model_data.get("suggestion_stats", {}))
            
            logger.info(f"Loaded suggestion models from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_suggestion_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about suggestion performance."""
        return {
            "stats": self.suggestion_stats.copy(),
            "cache_efficiency": {
                "cache_hits": self.suggestion_stats['cache_hits'],
                "cache_misses": self.suggestion_stats['cache_misses'],
                "hit_rate": (
                    self.suggestion_stats['cache_hits'] / 
                    max(self.suggestion_stats['cache_hits'] + self.suggestion_stats['cache_misses'], 1)
                )
            },
            "model_status": {
                "relevance_model_trained": self.relevance_model is not None,
                "feedback_data_size": len(self.feedback_data),
                "suggestion_history_size": len(self.suggestion_history)
            },
            "user_interaction_stats": {
                "total_users": len(self.user_interactions),
                "users_with_preferences": len([
                    u for u in self.user_interactions.values() 
                    if u.get('topic_preferences')
                ])
            }
        }


# Example usage
async def example_usage():
    """Example of using LocalMemorySuggester."""
    from ...core.embeddings.embedder import Embedder
    from ...core.memory.manager import MemoryManager
    
    # Initialize components (mocked for example)
    embedder = Embedder()  # Initialize with actual embedder
    memory_manager = MemoryManager()  # Initialize with actual memory manager
    
    # Create suggester
    suggester = LocalMemorySuggester(embedder, memory_manager)
    
    # Create suggestion context
    context = SuggestionContext(
        user_id="user123",
        query_text="machine learning algorithms",
        recent_memories=["mem1", "mem2", "mem3"],
        user_topics={"ai", "programming", "data science"}
    )
    
    # Generate suggestions
    result = await suggester.suggest_related_memories(
        context=context,
        max_suggestions=5,
        min_relevance=0.3,
        diversify=True
    )
    
    print(f"Generated {len(result.suggestions)} suggestions")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Strategies used: {result.suggestion_strategies}")
    
    for i, suggestion in enumerate(result.suggestions, 1):
        print(f"\n{i}. Memory: {suggestion.memory_id}")
        print(f"   Type: {suggestion.suggestion_type}")
        print(f"   Relevance: {suggestion.relevance_score:.3f} ({suggestion.relevance_category})")
        print(f"   Explanation: {suggestion.explanation}")
    
    # Record feedback
    if result.suggestions:
        await suggester.record_feedback(
            suggestion_id=result.suggestions[0].memory_id,
            feedback_type="relevance_rating",
            feedback_value=4.5  # Rating out of 5
        )
    
    # Get analytics
    analytics = suggester.get_suggestion_analytics()
    print(f"\nAnalytics: {analytics}")


if __name__ == "__main__":
    asyncio.run(example_usage())