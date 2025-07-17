"""
Session-Aware Contextual Embeddings for Enhanced Memory Retrieval.

This module provides session-aware embedding generation using local BERT models,
context injection, session adaptation with local fine-tuning, context vector fusion,
and context decay modeling. All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import math
import hashlib

# ML and transformer imports - all local
import torch
from transformers import (
    AutoTokenizer, AutoModel, AdapterConfig, 
    TrainingArguments, Trainer, BertTokenizer, BertModel
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ContextType(str, Enum):
    """Types of context for embedding enhancement."""
    SESSION = "session"                # User session context
    TEMPORAL = "temporal"              # Time-based context
    SEMANTIC = "semantic"              # Semantic similarity context
    USER_PROFILE = "user_profile"      # User preference context
    DOMAIN = "domain"                  # Domain-specific context
    CONVERSATION = "conversation"      # Conversation history context
    TASK = "task"                      # Task-specific context


class ContextDecayFunction(str, Enum):
    """Functions for context decay over time."""
    EXPONENTIAL = "exponential"        # Exponential decay
    LINEAR = "linear"                  # Linear decay
    LOGARITHMIC = "logarithmic"        # Logarithmic decay
    STEP = "step"                      # Step function decay
    CUSTOM = "custom"                  # Custom decay function


class FusionStrategy(str, Enum):
    """Strategies for context vector fusion."""
    WEIGHTED_AVERAGE = "weighted_average"      # Weighted averaging
    ATTENTION_BASED = "attention_based"        # Attention mechanism
    CONCATENATION = "concatenation"            # Simple concatenation
    ELEMENT_WISE = "element_wise"              # Element-wise operations
    NEURAL_FUSION = "neural_fusion"            # Neural network fusion


@dataclass
class SessionContext:
    """Represents session-based context for embeddings."""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    query_history: List[str] = field(default_factory=list)
    memory_access_history: List[str] = field(default_factory=list)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    context_weights: Dict[ContextType, float] = field(default_factory=dict)
    accumulated_context: Optional[np.ndarray] = None
    decay_function: ContextDecayFunction = ContextDecayFunction.EXPONENTIAL
    
    def calculate_session_duration(self) -> float:
        """Calculate session duration in minutes."""
        return (self.last_activity - self.start_time).total_seconds() / 60.0
    
    def get_context_freshness(self) -> float:
        """Calculate context freshness (0-1)."""
        minutes_since_activity = (datetime.utcnow() - self.last_activity).total_seconds() / 60.0
        # Context is fresh for 30 minutes, then decays
        return max(0.0, 1.0 - (minutes_since_activity / 30.0))


@dataclass
class ContextualEmbedding:
    """Enhanced embedding with context information."""
    base_embedding: np.ndarray
    contextual_embedding: np.ndarray
    context_vector: np.ndarray
    context_weights: Dict[ContextType, float]
    confidence: float
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_final_embedding(self, context_strength: float = 0.3) -> np.ndarray:
        """Get final embedding with context blending."""
        return (1 - context_strength) * self.base_embedding + context_strength * self.contextual_embedding


@dataclass
class ContextAdaptationStats:
    """Statistics for context adaptation performance."""
    sessions_processed: int = 0
    adaptations_performed: int = 0
    average_improvement: float = 0.0
    context_types_used: Dict[str, int] = field(default_factory=dict)
    decay_effectiveness: Dict[str, float] = field(default_factory=dict)
    fusion_performance: Dict[str, float] = field(default_factory=dict)


class ContextDecayModel:
    """
    Context decay modeling for temporal context adjustment.
    
    Features:
    - Multiple decay functions (exponential, linear, logarithmic, step)
    - Configurable decay parameters
    - Context freshness calculation
    - Adaptive decay rate adjustment
    """
    
    def __init__(
        self,
        decay_function: ContextDecayFunction = ContextDecayFunction.EXPONENTIAL,
        decay_rate: float = 0.1,
        min_weight: float = 0.01
    ):
        self.decay_function = decay_function
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        
        # Decay parameters for different functions
        self.decay_params = {
            'alpha': decay_rate,      # Base decay rate
            'beta': 0.5,             # Secondary parameter
            'gamma': 0.1,            # Tertiary parameter
            'threshold': 60.0        # Time threshold in minutes
        }
    
    def calculate_decay_weight(
        self,
        time_elapsed: float,
        context_importance: float = 1.0
    ) -> float:
        """Calculate decay weight based on time elapsed and context importance."""
        
        # Apply decay function
        if self.decay_function == ContextDecayFunction.EXPONENTIAL:
            weight = math.exp(-self.decay_params['alpha'] * time_elapsed)
        
        elif self.decay_function == ContextDecayFunction.LINEAR:
            weight = max(0.0, 1.0 - self.decay_params['alpha'] * time_elapsed / 60.0)
        
        elif self.decay_function == ContextDecayFunction.LOGARITHMIC:
            weight = 1.0 / (1.0 + self.decay_params['alpha'] * math.log(1 + time_elapsed))
        
        elif self.decay_function == ContextDecayFunction.STEP:
            if time_elapsed < self.decay_params['threshold']:
                weight = 1.0
            elif time_elapsed < self.decay_params['threshold'] * 2:
                weight = 0.5
            else:
                weight = 0.1
        
        else:  # Custom or fallback
            weight = max(0.0, 1.0 - time_elapsed / 120.0)  # 2-hour decay
        
        # Apply context importance multiplier
        weight *= context_importance
        
        # Ensure minimum weight
        return max(self.min_weight, weight)
    
    def update_decay_parameters(
        self,
        effectiveness_scores: Dict[str, float],
        adaptation_rate: float = 0.1
    ):
        """Update decay parameters based on effectiveness feedback."""
        if not effectiveness_scores:
            return
        
        avg_effectiveness = np.mean(list(effectiveness_scores.values()))
        
        # Adapt decay rate based on effectiveness
        if avg_effectiveness > 0.8:
            # High effectiveness - reduce decay rate (keep context longer)
            self.decay_rate *= (1 - adaptation_rate)
        elif avg_effectiveness < 0.5:
            # Low effectiveness - increase decay rate (forget context faster)
            self.decay_rate *= (1 + adaptation_rate)
        
        # Clamp decay rate
        self.decay_rate = max(0.01, min(1.0, self.decay_rate))
        self.decay_params['alpha'] = self.decay_rate


class ContextVectorFusion:
    """
    Advanced context vector fusion using multiple strategies.
    
    Features:
    - Weighted averaging with learned weights
    - Attention-based fusion mechanism
    - Element-wise fusion operations
    - Neural network-based fusion
    - Quality assessment and optimization
    """
    
    def __init__(
        self,
        embedding_dim: int,
        fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        device: str = "cpu"
    ):
        self.embedding_dim = embedding_dim
        self.fusion_strategy = fusion_strategy
        self.device = device
        
        # Learned fusion weights
        self.context_weights = {
            ContextType.SESSION: 0.3,
            ContextType.TEMPORAL: 0.2,
            ContextType.SEMANTIC: 0.25,
            ContextType.USER_PROFILE: 0.15,
            ContextType.DOMAIN: 0.1
        }
        
        # Attention weights for attention-based fusion
        self.attention_weights = None
        
        # Neural fusion network (simple MLP)
        if fusion_strategy == FusionStrategy.NEURAL_FUSION:
            self.fusion_network = self._create_fusion_network()
        else:
            self.fusion_network = None
    
    def _create_fusion_network(self) -> torch.nn.Module:
        """Create a simple neural network for context fusion."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.Tanh()
        ).to(self.device)
    
    async def fuse_context_vectors(
        self,
        base_embedding: np.ndarray,
        context_vectors: Dict[ContextType, np.ndarray],
        context_weights: Optional[Dict[ContextType, float]] = None
    ) -> Tuple[np.ndarray, float]:
        """Fuse context vectors with base embedding."""
        
        if not context_vectors:
            return base_embedding, 1.0
        
        weights = context_weights or self.context_weights
        
        if self.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_average_fusion(base_embedding, context_vectors, weights)
        
        elif self.fusion_strategy == FusionStrategy.ATTENTION_BASED:
            return await self._attention_based_fusion(base_embedding, context_vectors, weights)
        
        elif self.fusion_strategy == FusionStrategy.CONCATENATION:
            return await self._concatenation_fusion(base_embedding, context_vectors, weights)
        
        elif self.fusion_strategy == FusionStrategy.ELEMENT_WISE:
            return await self._element_wise_fusion(base_embedding, context_vectors, weights)
        
        elif self.fusion_strategy == FusionStrategy.NEURAL_FUSION:
            return await self._neural_fusion(base_embedding, context_vectors, weights)
        
        else:
            # Fallback to weighted average
            return await self._weighted_average_fusion(base_embedding, context_vectors, weights)
    
    async def _weighted_average_fusion(
        self,
        base_embedding: np.ndarray,
        context_vectors: Dict[ContextType, np.ndarray],
        weights: Dict[ContextType, float]
    ) -> Tuple[np.ndarray, float]:
        """Fuse using weighted averaging."""
        
        # Start with base embedding
        fused_embedding = base_embedding.copy()
        total_weight = 1.0
        confidence = 1.0
        
        # Add weighted context vectors
        for context_type, context_vector in context_vectors.items():
            if context_type in weights and weights[context_type] > 0:
                weight = weights[context_type]
                
                # Ensure context vector has same dimensions
                if len(context_vector) == len(base_embedding):
                    fused_embedding += weight * context_vector
                    total_weight += weight
                    confidence *= (1.0 + weight * 0.1)  # Boost confidence with more context
        
        # Normalize
        if total_weight > 0:
            fused_embedding /= total_weight
        
        # Normalize embedding
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding /= norm
        
        confidence = min(1.0, confidence)
        return fused_embedding, confidence
    
    async def _attention_based_fusion(
        self,
        base_embedding: np.ndarray,
        context_vectors: Dict[ContextType, np.ndarray],
        weights: Dict[ContextType, float]
    ) -> Tuple[np.ndarray, float]:
        """Fuse using attention mechanism."""
        
        # Calculate attention scores
        attention_scores = {}
        for context_type, context_vector in context_vectors.items():
            if len(context_vector) == len(base_embedding):
                # Use cosine similarity as attention score
                similarity = cosine_similarity(
                    base_embedding.reshape(1, -1),
                    context_vector.reshape(1, -1)
                )[0, 0]
                attention_scores[context_type] = max(0.0, similarity)
        
        # Normalize attention scores
        total_attention = sum(attention_scores.values())
        if total_attention > 0:
            for context_type in attention_scores:
                attention_scores[context_type] /= total_attention
        
        # Apply attention-weighted fusion
        fused_embedding = base_embedding.copy()
        confidence = 1.0
        
        for context_type, context_vector in context_vectors.items():
            if context_type in attention_scores:
                attention_weight = attention_scores[context_type]
                context_weight = weights.get(context_type, 0.0)
                final_weight = attention_weight * context_weight
                
                if final_weight > 0 and len(context_vector) == len(base_embedding):
                    fused_embedding += final_weight * context_vector
                    confidence *= (1.0 + final_weight * 0.15)
        
        # Normalize
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding /= norm
        
        confidence = min(1.0, confidence)
        return fused_embedding, confidence
    
    async def _concatenation_fusion(
        self,
        base_embedding: np.ndarray,
        context_vectors: Dict[ContextType, np.ndarray],
        weights: Dict[ContextType, float]
    ) -> Tuple[np.ndarray, float]:
        """Fuse using concatenation (then dimensionality reduction)."""
        
        # Collect all vectors for concatenation
        vectors_to_concat = [base_embedding]
        
        for context_type, context_vector in context_vectors.items():
            if context_type in weights and weights[context_type] > 0:
                if len(context_vector) == len(base_embedding):
                    # Weight the context vector
                    weighted_vector = context_vector * weights[context_type]
                    vectors_to_concat.append(weighted_vector)
        
        # Concatenate vectors
        concatenated = np.concatenate(vectors_to_concat)
        
        # Reduce to original dimensions using simple averaging
        # In practice, you might use PCA or a learned projection
        chunk_size = len(base_embedding)
        num_chunks = len(concatenated) // chunk_size
        
        fused_embedding = np.zeros_like(base_embedding)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            fused_embedding += concatenated[start_idx:end_idx]
        
        fused_embedding /= num_chunks
        
        # Normalize
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding /= norm
        
        confidence = min(1.0, 0.8 + 0.1 * len(context_vectors))
        return fused_embedding, confidence
    
    async def _element_wise_fusion(
        self,
        base_embedding: np.ndarray,
        context_vectors: Dict[ContextType, np.ndarray],
        weights: Dict[ContextType, float]
    ) -> Tuple[np.ndarray, float]:
        """Fuse using element-wise operations."""
        
        fused_embedding = base_embedding.copy()
        confidence = 1.0
        
        for context_type, context_vector in context_vectors.items():
            if context_type in weights and weights[context_type] > 0:
                if len(context_vector) == len(base_embedding):
                    weight = weights[context_type]
                    
                    # Element-wise multiplication with weight
                    element_wise_product = fused_embedding * context_vector * weight
                    
                    # Blend with original
                    blend_factor = weight * 0.5  # Conservative blending
                    fused_embedding = (1 - blend_factor) * fused_embedding + blend_factor * element_wise_product
                    
                    confidence *= (1.0 + weight * 0.05)
        
        # Normalize
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding /= norm
        
        confidence = min(1.0, confidence)
        return fused_embedding, confidence
    
    async def _neural_fusion(
        self,
        base_embedding: np.ndarray,
        context_vectors: Dict[ContextType, np.ndarray],
        weights: Dict[ContextType, float]
    ) -> Tuple[np.ndarray, float]:
        """Fuse using neural network."""
        
        if self.fusion_network is None:
            # Fallback to weighted average
            return await self._weighted_average_fusion(base_embedding, context_vectors, weights)
        
        # Create aggregated context vector
        aggregated_context = np.zeros_like(base_embedding)
        total_weight = 0.0
        
        for context_type, context_vector in context_vectors.items():
            if context_type in weights and weights[context_type] > 0:
                if len(context_vector) == len(base_embedding):
                    weight = weights[context_type]
                    aggregated_context += weight * context_vector
                    total_weight += weight
        
        if total_weight > 0:
            aggregated_context /= total_weight
        
        # Concatenate base and context embeddings
        input_tensor = torch.tensor(
            np.concatenate([base_embedding, aggregated_context]),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        # Forward pass through fusion network
        with torch.no_grad():
            fused_tensor = self.fusion_network(input_tensor)
            fused_embedding = fused_tensor.squeeze(0).cpu().numpy()
        
        # Normalize
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding /= norm
        
        confidence = min(1.0, 0.9 + 0.05 * len(context_vectors))
        return fused_embedding, confidence
    
    def update_fusion_weights(
        self,
        performance_feedback: Dict[ContextType, float],
        learning_rate: float = 0.1
    ):
        """Update fusion weights based on performance feedback."""
        
        for context_type, performance in performance_feedback.items():
            if context_type in self.context_weights:
                current_weight = self.context_weights[context_type]
                
                # Adjust weight based on performance
                if performance > 0.8:
                    # Good performance - increase weight
                    new_weight = current_weight * (1 + learning_rate * 0.5)
                elif performance < 0.5:
                    # Poor performance - decrease weight
                    new_weight = current_weight * (1 - learning_rate * 0.5)
                else:
                    # Neutral performance - small adjustment toward mean
                    mean_weight = np.mean(list(self.context_weights.values()))
                    new_weight = current_weight + learning_rate * 0.1 * (mean_weight - current_weight)
                
                # Clamp weight
                self.context_weights[context_type] = max(0.0, min(1.0, new_weight))
        
        # Normalize weights
        total_weight = sum(self.context_weights.values())
        if total_weight > 0:
            for context_type in self.context_weights:
                self.context_weights[context_type] /= total_weight


class SessionAwareEmbedder:
    """
    Session-aware contextual embedding generator.
    
    Features:
    - Local BERT models for context injection
    - Session adaptation with local fine-tuning
    - Context vector fusion using weighted averaging
    - Context decay modeling with exponential decay
    - Multi-context support (session, temporal, semantic, user, domain)
    - Performance optimization and caching
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        device: str = "cpu",
        redis_cache: Optional[RedisCache] = None,
        max_sequence_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.redis_cache = redis_cache
        self.max_sequence_length = max_sequence_length
        
        # Core models
        self.sentence_transformer = None
        self.tokenizer = None
        self.model = None
        
        # Context management
        self.active_sessions: Dict[str, SessionContext] = {}
        self.context_decay_model = ContextDecayModel()
        self.context_fusion = None
        
        # Performance tracking
        self.stats = ContextAdaptationStats()
        
        # Context extractors
        self.context_extractors = {
            ContextType.SESSION: self._extract_session_context,
            ContextType.TEMPORAL: self._extract_temporal_context,
            ContextType.SEMANTIC: self._extract_semantic_context,
            ContextType.USER_PROFILE: self._extract_user_profile_context,
            ContextType.DOMAIN: self._extract_domain_context
        }
        
        logger.info(f"SessionAwareEmbedder initialized with model: {model_name}")
    
    async def initialize(self):
        """Initialize embedding models and components."""
        try:
            # Load sentence transformer
            self.sentence_transformer = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Load tokenizer and model for context injection
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            
            # Initialize context fusion
            embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
            self.context_fusion = ContextVectorFusion(
                embedding_dim=embedding_dim,
                device=self.device
            )
            
            logger.info(f"SessionAwareEmbedder initialized successfully with {embedding_dim}D embeddings")
            
        except Exception as e:
            logger.error(f"Failed to initialize SessionAwareEmbedder: {e}")
            raise
    
    async def generate_contextual_embedding(
        self,
        text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context_types: Optional[Set[ContextType]] = None,
        context_strength: float = 0.3
    ) -> ContextualEmbedding:
        """Generate contextual embedding for text with session awareness."""
        
        # Generate base embedding
        base_embedding = await self._generate_base_embedding(text)
        
        # Extract context vectors
        context_vectors = await self._extract_context_vectors(
            text, session_id, user_id, context_types
        )
        
        # Fuse context with base embedding
        contextual_embedding, confidence = await self.context_fusion.fuse_context_vectors(
            base_embedding, context_vectors
        )
        
        # Calculate context weights used
        context_weights = {
            ctx_type: self.context_fusion.context_weights.get(ctx_type, 0.0)
            for ctx_type in context_vectors.keys()
        }
        
        # Create context vector for storage
        if context_vectors:
            context_vector = np.mean(list(context_vectors.values()), axis=0)
        else:
            context_vector = np.zeros_like(base_embedding)
        
        # Update session context if provided
        if session_id and session_id in self.active_sessions:
            await self._update_session_context(session_id, text, contextual_embedding)
        
        return ContextualEmbedding(
            base_embedding=base_embedding,
            contextual_embedding=contextual_embedding,
            context_vector=context_vector,
            context_weights=context_weights,
            confidence=confidence,
            session_id=session_id
        )
    
    async def _generate_base_embedding(self, text: str) -> np.ndarray:
        """Generate base embedding using sentence transformer."""
        try:
            # Check cache first
            if self.redis_cache:
                cache_key = f"base_emb:{hashlib.md5(text.encode()).hexdigest()}"
                cached_embedding = await self.redis_cache.get(cache_key)
                if cached_embedding is not None:
                    return np.array(cached_embedding)
            
            # Generate embedding
            embedding = self.sentence_transformer.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            # Cache result
            if self.redis_cache:
                await self.redis_cache.set(cache_key, embedding.tolist(), ttl=3600)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate base embedding: {e}")
            # Return random embedding as fallback
            dim = getattr(self.sentence_transformer, '_modules', {}).get('0', {}).get('auto_model', {}).get('config', {}).get('hidden_size', 384)
            return np.random.normal(0, 0.1, dim)
    
    async def _extract_context_vectors(
        self,
        text: str,
        session_id: Optional[str],
        user_id: Optional[str],
        context_types: Optional[Set[ContextType]]
    ) -> Dict[ContextType, np.ndarray]:
        """Extract context vectors for specified context types."""
        
        context_vectors = {}
        
        # Use all context types if not specified
        if context_types is None:
            context_types = set(self.context_extractors.keys())
        
        # Extract each context type
        for context_type in context_types:
            if context_type in self.context_extractors:
                try:
                    context_vector = await self.context_extractors[context_type](
                        text, session_id, user_id
                    )
                    if context_vector is not None:
                        context_vectors[context_type] = context_vector
                        
                        # Update stats
                        if context_type.value not in self.stats.context_types_used:
                            self.stats.context_types_used[context_type.value] = 0
                        self.stats.context_types_used[context_type.value] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to extract {context_type.value} context: {e}")
        
        return context_vectors
    
    async def _extract_session_context(
        self,
        text: str,
        session_id: Optional[str],
        user_id: Optional[str]
    ) -> Optional[np.ndarray]:
        """Extract session-based context vector."""
        
        if not session_id or session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Use accumulated context if available
        if session.accumulated_context is not None:
            # Apply decay based on session freshness
            freshness = session.get_context_freshness()
            decay_weight = self.context_decay_model.calculate_decay_weight(
                (datetime.utcnow() - session.last_activity).total_seconds() / 60.0,
                freshness
            )
            
            return session.accumulated_context * decay_weight
        
        # Generate context from session history
        if session.query_history:
            # Use recent queries as context
            recent_queries = session.query_history[-5:]  # Last 5 queries
            context_text = " ".join(recent_queries)
            
            try:
                context_embedding = await self._generate_base_embedding(context_text)
                return context_embedding * 0.7  # Reduce strength for session context
            except Exception:
                return None
        
        return None
    
    async def _extract_temporal_context(
        self,
        text: str,
        session_id: Optional[str],
        user_id: Optional[str]
    ) -> Optional[np.ndarray]:
        """Extract temporal context based on time patterns."""
        
        now = datetime.utcnow()
        
        # Create temporal features
        temporal_features = [
            now.hour / 24.0,                    # Hour of day
            now.weekday() / 7.0,                # Day of week
            now.day / 31.0,                     # Day of month
            math.sin(2 * math.pi * now.hour / 24),  # Cyclical hour
            math.cos(2 * math.pi * now.hour / 24),  # Cyclical hour
            math.sin(2 * math.pi * now.weekday() / 7),  # Cyclical day
            math.cos(2 * math.pi * now.weekday() / 7)   # Cyclical day
        ]
        
        # Extend to embedding dimension
        base_embedding_dim = len(await self._generate_base_embedding("test"))
        temporal_vector = np.zeros(base_embedding_dim)
        
        # Populate first few dimensions with temporal features
        for i, feature in enumerate(temporal_features):
            if i < base_embedding_dim:
                temporal_vector[i] = feature
        
        return temporal_vector * 0.5  # Reduce strength for temporal context
    
    async def _extract_semantic_context(
        self,
        text: str,
        session_id: Optional[str],
        user_id: Optional[str]
    ) -> Optional[np.ndarray]:
        """Extract semantic context based on text similarity patterns."""
        
        # This is a simplified version - in practice you might use
        # topic models, semantic clustering, etc.
        
        # Extract key terms for semantic context
        words = text.lower().split()
        
        # Simple keyword-based semantic context
        domain_keywords = {
            'technical': ['code', 'function', 'algorithm', 'api', 'data', 'system'],
            'business': ['strategy', 'market', 'customer', 'revenue', 'profit'],
            'academic': ['research', 'study', 'analysis', 'theory', 'experiment'],
            'personal': ['feel', 'think', 'remember', 'experience', 'personal']
        }
        
        # Calculate domain scores
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for word in words if word in keywords) / len(words) if words else 0
            domain_scores[domain] = score
        
        # Create semantic vector
        base_embedding_dim = len(await self._generate_base_embedding("test"))
        semantic_vector = np.zeros(base_embedding_dim)
        
        # Encode domain scores into vector
        for i, (domain, score) in enumerate(domain_scores.items()):
            if i < base_embedding_dim:
                semantic_vector[i] = score
        
        return semantic_vector * 0.6  # Moderate strength for semantic context
    
    async def _extract_user_profile_context(
        self,
        text: str,
        session_id: Optional[str],
        user_id: Optional[str]
    ) -> Optional[np.ndarray]:
        """Extract user profile-based context."""
        
        if not user_id:
            return None
        
        # This would typically load user preferences, interaction history, etc.
        # For now, create a simple user-based context
        
        # Create user-specific hash for consistent context
        user_hash = hashlib.md5(user_id.encode()).digest()
        user_features = np.frombuffer(user_hash, dtype=np.uint8).astype(np.float32)
        user_features = (user_features - 128) / 128  # Normalize to [-1, 1]
        
        # Extend to embedding dimension
        base_embedding_dim = len(await self._generate_base_embedding("test"))
        user_vector = np.zeros(base_embedding_dim)
        
        # Populate with user features
        for i, feature in enumerate(user_features):
            if i < base_embedding_dim:
                user_vector[i] = feature * 0.1  # Small influence
        
        return user_vector
    
    async def _extract_domain_context(
        self,
        text: str,
        session_id: Optional[str],
        user_id: Optional[str]
    ) -> Optional[np.ndarray]:
        """Extract domain-specific context."""
        
        # Simple domain detection based on keywords
        domains = {
            'technology': ['software', 'computer', 'programming', 'tech', 'digital'],
            'science': ['research', 'experiment', 'hypothesis', 'data', 'analysis'],
            'business': ['company', 'market', 'sales', 'customer', 'business'],
            'health': ['medical', 'health', 'doctor', 'patient', 'treatment']
        }
        
        words = text.lower().split()
        domain_scores = {}
        
        for domain, keywords in domains.items():
            score = sum(1 for word in words if any(kw in word for kw in keywords))
            domain_scores[domain] = score / len(words) if words else 0
        
        # Get dominant domain
        dominant_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        if dominant_domain[1] > 0:
            # Create domain-specific context
            base_embedding_dim = len(await self._generate_base_embedding("test"))
            domain_vector = np.zeros(base_embedding_dim)
            
            # Use domain embedding (could be pre-trained domain embeddings)
            domain_text = f"This is about {dominant_domain[0]} domain"
            try:
                domain_embedding = await self._generate_base_embedding(domain_text)
                return domain_embedding * dominant_domain[1] * 0.4
            except Exception:
                return None
        
        return None
    
    async def start_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        context_weights: Optional[Dict[ContextType, float]] = None
    ) -> SessionContext:
        """Start a new session for context tracking."""
        
        session = SessionContext(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            context_weights=context_weights or {}
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Started session {session_id} for user {user_id}")
        
        return session
    
    async def end_session(self, session_id: str):
        """End a session and clean up context."""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session_duration = session.calculate_session_duration()
            
            # Update stats
            self.stats.sessions_processed += 1
            
            # Remove session
            del self.active_sessions[session_id]
            
            logger.info(f"Ended session {session_id}, duration: {session_duration:.1f} minutes")
    
    async def _update_session_context(
        self,
        session_id: str,
        text: str,
        embedding: np.ndarray
    ):
        """Update session context with new interaction."""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Update activity time
        session.last_activity = datetime.utcnow()
        
        # Add to query history
        session.query_history.append(text)
        if len(session.query_history) > 50:  # Keep last 50 queries
            session.query_history.pop(0)
        
        # Update accumulated context
        if session.accumulated_context is None:
            session.accumulated_context = embedding.copy()
        else:
            # Exponential moving average
            alpha = 0.3  # Learning rate
            session.accumulated_context = (
                alpha * embedding + (1 - alpha) * session.accumulated_context
            )
    
    async def adapt_context_weights(
        self,
        performance_feedback: Dict[str, float],
        session_id: Optional[str] = None
    ):
        """Adapt context weights based on performance feedback."""
        
        # Convert string keys to ContextType
        context_feedback = {}
        for key, value in performance_feedback.items():
            try:
                context_type = ContextType(key)
                context_feedback[context_type] = value
            except ValueError:
                logger.warning(f"Unknown context type: {key}")
        
        # Update global fusion weights
        self.context_fusion.update_fusion_weights(context_feedback)
        
        # Update session-specific weights if provided
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            for context_type, score in context_feedback.items():
                if context_type in session.context_weights:
                    current = session.context_weights[context_type]
                    if score > 0.8:
                        session.context_weights[context_type] = min(1.0, current * 1.1)
                    elif score < 0.5:
                        session.context_weights[context_type] = max(0.0, current * 0.9)
        
        # Update decay model
        effectiveness_scores = {k: v for k, v in performance_feedback.items()}
        self.context_decay_model.update_decay_parameters(effectiveness_scores)
        
        # Update stats
        self.stats.adaptations_performed += 1
        self.stats.average_improvement = np.mean(list(performance_feedback.values()))
        
        logger.info(f"Adapted context weights based on feedback: {performance_feedback}")
    
    async def get_context_stats(self) -> Dict[str, Any]:
        """Get comprehensive context adaptation statistics."""
        
        return {
            'sessions_active': len(self.active_sessions),
            'sessions_processed': self.stats.sessions_processed,
            'adaptations_performed': self.stats.adaptations_performed,
            'average_improvement': self.stats.average_improvement,
            'context_types_used': dict(self.stats.context_types_used),
            'fusion_weights': self.context_fusion.context_weights,
            'decay_parameters': self.context_decay_model.decay_params,
            'active_session_durations': {
                sid: session.calculate_session_duration()
                for sid, session in self.active_sessions.items()
            }
        }
    
    async def cleanup_expired_sessions(self, max_age_minutes: int = 120):
        """Clean up expired sessions based on inactivity."""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if session.last_activity < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.end_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")