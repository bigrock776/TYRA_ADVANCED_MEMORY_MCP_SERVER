"""
Enhanced Dynamic Reranking with Query Intent Analysis.

This module provides advanced reranking capabilities that adapt to query intent,
context, and user preferences. All processing is performed locally with zero
external API calls.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import json
import heapq
import math

# ML and NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Transformer imports for local reranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

# spaCy for linguistic analysis
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import structlog
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent

from ...models.memory import Memory
from ..embeddings.embedder import Embedder
from ..cache.redis_cache import RedisCache
from ..utils.config import settings

logger = structlog.get_logger(__name__)


class QueryIntent(str, Enum):
    """Types of query intents for adaptive reranking."""
    FACTUAL = "factual"              # Seeking specific facts
    PROCEDURAL = "procedural"        # How-to questions
    CONCEPTUAL = "conceptual"        # Understanding concepts
    COMPARATIVE = "comparative"      # Comparing items
    ANALYTICAL = "analytical"        # Deep analysis
    CREATIVE = "creative"           # Creative/brainstorming
    TEMPORAL = "temporal"           # Time-related queries
    SPATIAL = "spatial"             # Location-related queries
    PERSONAL = "personal"           # Personal experiences
    EXPLORATORY = "exploratory"     # Open-ended exploration


class RerankingStrategy(str, Enum):
    """Different reranking strategies."""
    SEMANTIC = "semantic"           # Pure semantic similarity
    RELEVANCE = "relevance"         # Relevance-focused
    RECENCY = "recency"            # Time-based ranking
    POPULARITY = "popularity"       # Usage-based ranking
    DIVERSITY = "diversity"         # Maximize diversity
    HYBRID = "hybrid"              # Combination approach
    INTENT_ADAPTIVE = "intent_adaptive"  # Intent-based adaptation


class ContextType(str, Enum):
    """Types of context for reranking."""
    CONVERSATION = "conversation"   # Chat context
    DOCUMENT = "document"          # Document context
    TASK = "task"                  # Task-specific context
    USER_PROFILE = "user_profile"  # User preferences
    TEMPORAL = "temporal"          # Time context
    DOMAIN = "domain"              # Domain-specific context


@dataclass
class QueryContext:
    """Context information for query processing."""
    intent: QueryIntent
    conversation_history: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[str] = None
    temporal_context: Optional[datetime] = None
    task_context: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class RankingFeatures:
    """Features extracted for ranking calculation."""
    semantic_similarity: float
    keyword_overlap: float
    recency_score: float
    popularity_score: float
    intent_alignment: float
    context_relevance: float
    diversity_penalty: float
    quality_score: float
    confidence: float


class RerankingResult(BaseModel):
    """Result from reranking operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    memory_id: str = Field(description="Memory ID")
    original_rank: int = Field(description="Original ranking position")
    new_rank: int = Field(description="New ranking position")
    score: float = Field(description="Final ranking score")
    features: Dict[str, float] = Field(description="Individual feature scores")
    explanation: str = Field(description="Ranking explanation")


class DynamicReranker:
    """
    Advanced dynamic reranking system with query intent analysis.
    
    Features:
    - Query intent detection using local NLP models
    - Multi-factor ranking with adaptive weights
    - Context-aware reranking strategies
    - Learning from user interactions
    - Cross-encoder based semantic reranking
    - Diversity-aware result optimization
    - Temporal and popularity-based adjustments
    """
    
    def __init__(
        self,
        embedder: Embedder,
        cache: Optional[RedisCache] = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        spacy_model: str = "en_core_web_sm",
        diversity_threshold: float = 0.85,
        learning_rate: float = 0.01
    ):
        """
        Initialize the dynamic reranker.
        
        Args:
            embedder: Text embedder for similarity calculation
            cache: Optional Redis cache for performance
            cross_encoder_model: Cross-encoder model for reranking
            spacy_model: spaCy model for linguistic analysis
            diversity_threshold: Threshold for diversity enforcement
            learning_rate: Learning rate for adaptive weights
        """
        self.embedder = embedder
        self.cache = cache
        self.diversity_threshold = diversity_threshold
        self.learning_rate = learning_rate
        
        # Initialize cross-encoder for semantic reranking
        self.cross_encoder = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info(f"Loaded cross-encoder: {cross_encoder_model}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
        
        # Initialize spaCy for linguistic analysis
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(f"SpaCy model {spacy_model} not found")
        
        # Initialize intent detection patterns
        self._init_intent_patterns()
        
        # Initialize adaptive weights
        self.adaptive_weights = {
            'semantic_similarity': 0.3,
            'keyword_overlap': 0.2,
            'recency_score': 0.1,
            'popularity_score': 0.1,
            'intent_alignment': 0.2,
            'context_relevance': 0.1
        }
        
        # Feature extractors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Learning components
        self.ranking_model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Usage statistics for popularity scoring
        self.usage_stats = defaultdict(int)
        self.interaction_history = []
        
        logger.info(
            "Initialized dynamic reranker",
            cross_encoder_available=CROSS_ENCODER_AVAILABLE,
            spacy_available=SPACY_AVAILABLE,
            diversity_threshold=diversity_threshold
        )
    
    def _init_intent_patterns(self):
        """Initialize patterns for query intent detection."""
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(?:what is|what are|who is|who are|where is|where are|when is|when are)\b',
                r'\b(?:define|definition|meaning|explain|describe)\b',
                r'\b(?:facts about|information about|details about)\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\b(?:how to|how do|how can|steps to|guide to|tutorial)\b',
                r'\b(?:instruction|procedure|process|method|way to)\b',
                r'\b(?:create|make|build|setup|configure|install)\b'
            ],
            QueryIntent.COMPARATIVE: [
                r'\b(?:compare|comparison|difference|versus|vs|better than)\b',
                r'\b(?:pros and cons|advantages|disadvantages|similarities)\b',
                r'\b(?:which is better|best option|choose between)\b'
            ],
            QueryIntent.ANALYTICAL: [
                r'\b(?:analyze|analysis|examine|evaluate|assess|study)\b',
                r'\b(?:why does|why is|reason|cause|effect|impact)\b',
                r'\b(?:implications|consequences|significance|importance)\b'
            ],
            QueryIntent.TEMPORAL: [
                r'\b(?:when|timeline|schedule|date|time|duration|period)\b',
                r'\b(?:history|historical|past|future|recent|latest)\b',
                r'\b(?:before|after|during|since|until)\b'
            ],
            QueryIntent.CREATIVE: [
                r'\b(?:ideas|brainstorm|creative|innovative|suggestions)\b',
                r'\b(?:imagine|invent|design|create|generate)\b',
                r'\b(?:alternatives|options|possibilities|scenarios)\b'
            ]
        }
    
    async def rerank_memories(
        self,
        query: str,
        memories: List[Memory],
        context: Optional[QueryContext] = None,
        strategy: RerankingStrategy = RerankingStrategy.INTENT_ADAPTIVE,
        top_k: Optional[int] = None
    ) -> List[RerankingResult]:
        """
        Rerank memories based on query, context, and intent.
        
        Args:
            query: Search query
            memories: List of memories to rerank
            context: Optional query context
            strategy: Reranking strategy to use
            top_k: Number of top results to return
            
        Returns:
            List of reranking results with scores and explanations
        """
        if not memories:
            return []
        
        # Detect query intent if not provided in context
        if context is None:
            intent = await self._detect_query_intent(query)
            context = QueryContext(intent=intent)
        
        # Extract features for all memories
        features_list = await self._extract_ranking_features(query, memories, context)
        
        # Calculate ranking scores based on strategy
        scores = await self._calculate_ranking_scores(
            query, memories, features_list, context, strategy
        )
        
        # Apply diversity filtering if needed
        if strategy in [RerankingStrategy.DIVERSITY, RerankingStrategy.HYBRID]:
            scores = await self._apply_diversity_filtering(memories, scores, features_list)
        
        # Create ranking results
        results = []
        for i, (memory, features, score) in enumerate(zip(memories, features_list, scores)):
            explanation = self._generate_ranking_explanation(features, context, strategy)
            
            result = RerankingResult(
                memory_id=memory.id,
                original_rank=i,
                new_rank=0,  # Will be set after sorting
                score=score,
                features=features.__dict__,
                explanation=explanation
            )
            results.append(result)
        
        # Sort by score and assign new ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i
        
        # Return top-k if specified
        if top_k:
            results = results[:top_k]
        
        logger.info(
            "Completed memory reranking",
            query_intent=context.intent.value,
            strategy=strategy.value,
            total_memories=len(memories),
            top_k=top_k or len(results)
        )
        
        return results
    
    async def _detect_query_intent(self, query: str) -> QueryIntent:
        """Detect the intent of a query using pattern matching and NLP."""
        query_lower = query.lower()
        
        # Score each intent based on pattern matches
        intent_scores = defaultdict(float)
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                import re
                matches = len(re.findall(pattern, query_lower))
                intent_scores[intent] += matches
        
        # Additional linguistic analysis with spaCy
        if self.nlp:
            doc = self.nlp(query)
            
            # Question word analysis
            question_words = {'what', 'who', 'where', 'when', 'why', 'how'}
            for token in doc:
                if token.lemma_.lower() in question_words:
                    if token.lemma_.lower() == 'how':
                        intent_scores[QueryIntent.PROCEDURAL] += 2
                    elif token.lemma_.lower() in ['what', 'who', 'where', 'when']:
                        intent_scores[QueryIntent.FACTUAL] += 2
                    elif token.lemma_.lower() == 'why':
                        intent_scores[QueryIntent.ANALYTICAL] += 2
            
            # POS tag analysis
            verb_count = len([token for token in doc if token.pos_ == 'VERB'])
            noun_count = len([token for token in doc if token.pos_ == 'NOUN'])
            
            if verb_count > noun_count:
                intent_scores[QueryIntent.PROCEDURAL] += 1
        
        # Return intent with highest score, default to exploratory
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            return best_intent
        else:
            return QueryIntent.EXPLORATORY
    
    async def _extract_ranking_features(
        self,
        query: str,
        memories: List[Memory],
        context: QueryContext
    ) -> List[RankingFeatures]:
        """Extract ranking features for all memories."""
        features_list = []
        
        # Generate query embedding
        query_embedding = await self.embedder.embed(query)
        
        # Extract features for each memory
        for memory in memories:
            features = await self._extract_single_memory_features(
                query, query_embedding, memory, context
            )
            features_list.append(features)
        
        return features_list
    
    async def _extract_single_memory_features(
        self,
        query: str,
        query_embedding: np.ndarray,
        memory: Memory,
        context: QueryContext
    ) -> RankingFeatures:
        """Extract ranking features for a single memory."""
        
        # Semantic similarity
        if hasattr(memory, 'embedding') and memory.embedding is not None:
            semantic_similarity = float(cosine_similarity([query_embedding], [memory.embedding])[0][0])
        else:
            # Fallback: generate embedding and calculate similarity
            memory_embedding = await self.embedder.embed(memory.content)
            semantic_similarity = float(cosine_similarity([query_embedding], [memory_embedding])[0][0])
        
        # Cross-encoder similarity (if available)
        if self.cross_encoder:
            try:
                cross_encoder_score = self.cross_encoder.predict([(query, memory.content)])[0]
                # Normalize to 0-1 range
                semantic_similarity = max(semantic_similarity, (cross_encoder_score + 1) / 2)
            except Exception as e:
                logger.warning(f"Cross-encoder prediction failed: {e}")
        
        # Keyword overlap
        keyword_overlap = self._calculate_keyword_overlap(query, memory.content)
        
        # Recency score
        recency_score = self._calculate_recency_score(memory)
        
        # Popularity score
        popularity_score = self._calculate_popularity_score(memory)
        
        # Intent alignment
        intent_alignment = await self._calculate_intent_alignment(memory, context.intent)
        
        # Context relevance
        context_relevance = await self._calculate_context_relevance(memory, context)
        
        # Quality score
        quality_score = self._calculate_quality_score(memory)
        
        # Confidence score
        confidence = self._calculate_confidence_score(
            semantic_similarity, keyword_overlap, intent_alignment
        )
        
        return RankingFeatures(
            semantic_similarity=semantic_similarity,
            keyword_overlap=keyword_overlap,
            recency_score=recency_score,
            popularity_score=popularity_score,
            intent_alignment=intent_alignment,
            context_relevance=context_relevance,
            diversity_penalty=0.0,  # Will be calculated later
            quality_score=quality_score,
            confidence=confidence
        )
    
    def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        """Calculate keyword overlap between query and content."""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            vectors = vectorizer.fit_transform([query, content])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except:
            # Fallback: simple word overlap
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            if not query_words:
                return 0.0
            overlap = len(query_words.intersection(content_words))
            return overlap / len(query_words)
    
    def _calculate_recency_score(self, memory: Memory) -> float:
        """Calculate recency score based on memory timestamp."""
        if not hasattr(memory, 'created_at') or memory.created_at is None:
            return 0.5  # Neutral score for unknown timestamps
        
        # Calculate days since creation
        now = datetime.utcnow()
        days_old = (now - memory.created_at).days
        
        # Exponential decay: score = e^(-days/30)
        recency_score = math.exp(-days_old / 30.0)
        return min(recency_score, 1.0)
    
    def _calculate_popularity_score(self, memory: Memory) -> float:
        """Calculate popularity score based on usage statistics."""
        usage_count = self.usage_stats.get(memory.id, 0)
        
        if not self.usage_stats:
            return 0.5  # Neutral score if no usage data
        
        max_usage = max(self.usage_stats.values())
        if max_usage == 0:
            return 0.5
        
        return usage_count / max_usage
    
    async def _calculate_intent_alignment(self, memory: Memory, intent: QueryIntent) -> float:
        """Calculate how well memory content aligns with query intent."""
        content = memory.content.lower()
        
        # Intent-specific scoring
        if intent == QueryIntent.FACTUAL:
            # Look for definitions, facts, specific information
            factual_indicators = ['definition', 'fact', 'information', 'data', 'statistic']
            score = sum(1 for indicator in factual_indicators if indicator in content)
            return min(score / 3.0, 1.0)
        
        elif intent == QueryIntent.PROCEDURAL:
            # Look for steps, instructions, procedures
            procedural_indicators = ['step', 'instruction', 'procedure', 'method', 'process']
            score = sum(1 for indicator in procedural_indicators if indicator in content)
            return min(score / 3.0, 1.0)
        
        elif intent == QueryIntent.ANALYTICAL:
            # Look for analysis, explanations, reasoning
            analytical_indicators = ['analysis', 'reason', 'because', 'therefore', 'explanation']
            score = sum(1 for indicator in analytical_indicators if indicator in content)
            return min(score / 3.0, 1.0)
        
        elif intent == QueryIntent.COMPARATIVE:
            # Look for comparisons, differences, similarities
            comparative_indicators = ['compare', 'versus', 'difference', 'similar', 'unlike']
            score = sum(1 for indicator in comparative_indicators if indicator in content)
            return min(score / 3.0, 1.0)
        
        elif intent == QueryIntent.TEMPORAL:
            # Look for time-related content
            temporal_indicators = ['time', 'date', 'when', 'before', 'after', 'during']
            score = sum(1 for indicator in temporal_indicators if indicator in content)
            return min(score / 3.0, 1.0)
        
        else:
            # Default neutral score for other intents
            return 0.5
    
    async def _calculate_context_relevance(self, memory: Memory, context: QueryContext) -> float:
        """Calculate relevance based on contextual factors."""
        relevance = 0.0
        
        # Conversation history relevance
        if context.conversation_history:
            for msg in context.conversation_history[-3:]:  # Last 3 messages
                similarity = self._calculate_keyword_overlap(msg, memory.content)
                relevance += similarity * 0.2  # Weight by recency
        
        # Domain relevance
        if context.domain and hasattr(memory, 'metadata'):
            memory_domain = memory.metadata.get('domain', '')
            if memory_domain == context.domain:
                relevance += 0.3
        
        # Task context relevance
        if context.task_context:
            task_similarity = self._calculate_keyword_overlap(context.task_context, memory.content)
            relevance += task_similarity * 0.3
        
        return min(relevance, 1.0)
    
    def _calculate_quality_score(self, memory: Memory) -> float:
        """Calculate content quality score."""
        content = memory.content
        
        # Basic quality indicators
        quality = 0.0
        
        # Length penalty for very short or very long content
        length = len(content.split())
        if 10 <= length <= 500:
            quality += 0.3
        elif length > 500:
            quality += 0.1
        
        # Sentence structure
        sentences = content.split('.')
        if len(sentences) > 1:
            quality += 0.2
        
        # Presence of structured content
        if any(indicator in content.lower() for indicator in ['1.', '2.', '-', '*']):
            quality += 0.2
        
        # Grammar and readability (basic check)
        if content.count('?') + content.count('!') < len(sentences) * 0.5:
            quality += 0.2
        
        # Metadata quality
        if hasattr(memory, 'metadata') and memory.metadata:
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _calculate_confidence_score(
        self, 
        semantic_sim: float, 
        keyword_overlap: float, 
        intent_alignment: float
    ) -> float:
        """Calculate overall confidence in the ranking."""
        # Weighted average of key similarity metrics
        confidence = (
            semantic_sim * 0.5 +
            keyword_overlap * 0.3 +
            intent_alignment * 0.2
        )
        return min(confidence, 1.0)
    
    async def _calculate_ranking_scores(
        self,
        query: str,
        memories: List[Memory],
        features_list: List[RankingFeatures],
        context: QueryContext,
        strategy: RerankingStrategy
    ) -> List[float]:
        """Calculate final ranking scores based on strategy."""
        
        if strategy == RerankingStrategy.SEMANTIC:
            return [features.semantic_similarity for features in features_list]
        
        elif strategy == RerankingStrategy.RECENCY:
            return [features.recency_score for features in features_list]
        
        elif strategy == RerankingStrategy.POPULARITY:
            return [features.popularity_score for features in features_list]
        
        elif strategy == RerankingStrategy.INTENT_ADAPTIVE:
            return await self._calculate_intent_adaptive_scores(features_list, context)
        
        elif strategy == RerankingStrategy.HYBRID:
            return await self._calculate_hybrid_scores(features_list, context)
        
        else:  # RELEVANCE or default
            return await self._calculate_relevance_scores(features_list)
    
    async def _calculate_intent_adaptive_scores(
        self,
        features_list: List[RankingFeatures],
        context: QueryContext
    ) -> List[float]:
        """Calculate scores with intent-adaptive weighting."""
        
        # Adjust weights based on query intent
        weights = self.adaptive_weights.copy()
        
        if context.intent == QueryIntent.FACTUAL:
            weights['semantic_similarity'] = 0.4
            weights['intent_alignment'] = 0.3
            weights['keyword_overlap'] = 0.2
            weights['recency_score'] = 0.05
            weights['popularity_score'] = 0.05
        
        elif context.intent == QueryIntent.PROCEDURAL:
            weights['intent_alignment'] = 0.4
            weights['semantic_similarity'] = 0.3
            weights['context_relevance'] = 0.2
            weights['keyword_overlap'] = 0.1
        
        elif context.intent == QueryIntent.TEMPORAL:
            weights['recency_score'] = 0.4
            weights['semantic_similarity'] = 0.3
            weights['intent_alignment'] = 0.2
            weights['keyword_overlap'] = 0.1
        
        elif context.intent == QueryIntent.ANALYTICAL:
            weights['semantic_similarity'] = 0.35
            weights['intent_alignment'] = 0.35
            weights['context_relevance'] = 0.2
            weights['keyword_overlap'] = 0.1
        
        # Calculate weighted scores
        scores = []
        for features in features_list:
            score = (
                features.semantic_similarity * weights['semantic_similarity'] +
                features.keyword_overlap * weights['keyword_overlap'] +
                features.recency_score * weights['recency_score'] +
                features.popularity_score * weights['popularity_score'] +
                features.intent_alignment * weights['intent_alignment'] +
                features.context_relevance * weights['context_relevance']
            )
            scores.append(score)
        
        return scores
    
    async def _calculate_hybrid_scores(
        self,
        features_list: List[RankingFeatures],
        context: QueryContext
    ) -> List[float]:
        """Calculate hybrid scores combining multiple strategies."""
        
        # Get intent-adaptive scores
        intent_scores = await self._calculate_intent_adaptive_scores(features_list, context)
        
        # Calculate additional components
        scores = []
        for i, features in enumerate(features_list):
            # Base score from intent-adaptive
            base_score = intent_scores[i]
            
            # Quality boost
            quality_boost = features.quality_score * 0.1
            
            # Confidence adjustment
            confidence_adjustment = features.confidence * 0.05
            
            # Final hybrid score
            hybrid_score = base_score + quality_boost + confidence_adjustment
            scores.append(min(hybrid_score, 1.0))
        
        return scores
    
    async def _calculate_relevance_scores(
        self,
        features_list: List[RankingFeatures]
    ) -> List[float]:
        """Calculate relevance-focused scores."""
        scores = []
        for features in features_list:
            # Simple relevance = semantic + keyword + quality
            relevance = (
                features.semantic_similarity * 0.5 +
                features.keyword_overlap * 0.3 +
                features.quality_score * 0.2
            )
            scores.append(relevance)
        
        return scores
    
    async def _apply_diversity_filtering(
        self,
        memories: List[Memory],
        scores: List[float],
        features_list: List[RankingFeatures]
    ) -> List[float]:
        """Apply diversity filtering to avoid redundant results."""
        
        # Create list of (index, score) pairs
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select diverse results
        selected_indices = []
        selected_embeddings = []
        
        for idx, score in indexed_scores:
            memory = memories[idx]
            
            # Check similarity with already selected memories
            is_diverse = True
            if hasattr(memory, 'embedding') and memory.embedding is not None:
                current_embedding = memory.embedding
                
                for selected_embedding in selected_embeddings:
                    similarity = cosine_similarity([current_embedding], [selected_embedding])[0][0]
                    if similarity > self.diversity_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    selected_indices.append(idx)
                    selected_embeddings.append(current_embedding)
            else:
                # If no embedding, include by default
                selected_indices.append(idx)
        
        # Apply diversity penalty to non-selected items
        adjusted_scores = scores.copy()
        for i, score in enumerate(adjusted_scores):
            if i not in selected_indices:
                adjusted_scores[i] = score * 0.7  # Diversity penalty
        
        return adjusted_scores
    
    def _generate_ranking_explanation(
        self,
        features: RankingFeatures,
        context: QueryContext,
        strategy: RerankingStrategy
    ) -> str:
        """Generate human-readable explanation for ranking decision."""
        
        explanations = []
        
        # Semantic similarity
        if features.semantic_similarity > 0.8:
            explanations.append("High semantic similarity to query")
        elif features.semantic_similarity > 0.6:
            explanations.append("Moderate semantic similarity to query")
        
        # Intent alignment
        if features.intent_alignment > 0.7:
            explanations.append(f"Strong alignment with {context.intent.value} intent")
        
        # Keyword overlap
        if features.keyword_overlap > 0.5:
            explanations.append("Good keyword overlap with query")
        
        # Recency
        if features.recency_score > 0.8:
            explanations.append("Recent content")
        
        # Popularity
        if features.popularity_score > 0.7:
            explanations.append("Frequently accessed content")
        
        # Context relevance
        if features.context_relevance > 0.6:
            explanations.append("Relevant to conversation context")
        
        # Quality
        if features.quality_score > 0.8:
            explanations.append("High-quality content")
        
        # Strategy-specific explanations
        if strategy == RerankingStrategy.INTENT_ADAPTIVE:
            explanations.append(f"Adapted for {context.intent.value} queries")
        elif strategy == RerankingStrategy.DIVERSITY:
            explanations.append("Selected for result diversity")
        
        if not explanations:
            explanations = ["Basic relevance match"]
        
        return "; ".join(explanations)
    
    async def learn_from_interaction(
        self,
        query: str,
        selected_memory_ids: List[str],
        feedback_scores: Optional[List[float]] = None
    ):
        """Learn from user interactions to improve ranking."""
        
        # Update usage statistics
        for memory_id in selected_memory_ids:
            self.usage_stats[memory_id] += 1
        
        # Store interaction for future learning
        interaction = {
            'query': query,
            'selected_ids': selected_memory_ids,
            'feedback_scores': feedback_scores,
            'timestamp': datetime.utcnow()
        }
        self.interaction_history.append(interaction)
        
        # Keep only recent interactions (last 1000)
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
        
        # Adaptive weight adjustment based on feedback
        if feedback_scores and len(feedback_scores) == len(selected_memory_ids):
            await self._adjust_adaptive_weights(feedback_scores)
    
    async def _adjust_adaptive_weights(self, feedback_scores: List[float]):
        """Adjust adaptive weights based on user feedback."""
        
        # Simple gradient-based adjustment
        avg_feedback = np.mean(feedback_scores)
        
        if avg_feedback > 0.7:
            # Positive feedback - strengthen current weights
            for key in self.adaptive_weights:
                self.adaptive_weights[key] *= (1 + self.learning_rate)
        elif avg_feedback < 0.3:
            # Negative feedback - adjust weights
            for key in self.adaptive_weights:
                self.adaptive_weights[key] *= (1 - self.learning_rate)
        
        # Normalize weights
        total_weight = sum(self.adaptive_weights.values())
        for key in self.adaptive_weights:
            self.adaptive_weights[key] /= total_weight
    
    async def get_ranking_analytics(self) -> Dict[str, Any]:
        """Get analytics about ranking performance."""
        
        analytics = {
            'total_interactions': len(self.interaction_history),
            'unique_memories_accessed': len(self.usage_stats),
            'top_memories': sorted(
                self.usage_stats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'current_weights': self.adaptive_weights.copy(),
            'is_trained': self.is_trained
        }
        
        if self.interaction_history:
            recent_queries = [
                interaction['query'] 
                for interaction in self.interaction_history[-10:]
            ]
            analytics['recent_queries'] = recent_queries
        
        return analytics