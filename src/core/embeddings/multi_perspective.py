"""
Multi-Perspective Embeddings for Enhanced Context Understanding.

This module provides multi-perspective embedding generation using multiple local embedding models,
perspective switching with local routing, confidence scoring using ensemble methods,
and perspective recommendation using similarity metrics. All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import math
import hashlib

# ML and transformer imports - all local
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class PerspectiveType(str, Enum):
    """Types of embedding perspectives."""
    GENERAL = "general"                    # General-purpose embeddings
    DOMAIN_SPECIFIC = "domain_specific"    # Domain-specialized embeddings
    SEMANTIC = "semantic"                  # Semantic understanding focus
    SYNTACTIC = "syntactic"                # Syntactic structure focus
    CONTEXTUAL = "contextual"              # Context-aware embeddings
    MULTILINGUAL = "multilingual"          # Cross-language embeddings
    CODE = "code"                          # Code-specific embeddings
    SCIENTIFIC = "scientific"              # Scientific domain embeddings


class RoutingStrategy(str, Enum):
    """Strategies for routing text to appropriate perspectives."""
    CONTENT_BASED = "content_based"        # Route based on content analysis
    SIMILARITY_BASED = "similarity_based"  # Route based on similarity patterns
    ENSEMBLE = "ensemble"                  # Use ensemble of all perspectives
    LEARNED = "learned"                    # ML-learned routing
    HYBRID = "hybrid"                      # Combination of strategies
    USER_PREFERENCE = "user_preference"    # Based on user preferences


class ConfidenceMethod(str, Enum):
    """Methods for calculating perspective confidence."""
    INTERNAL_CONSISTENCY = "internal_consistency"  # Self-consistency metrics
    CROSS_VALIDATION = "cross_validation"          # Cross-perspective validation
    ENSEMBLE_AGREEMENT = "ensemble_agreement"      # Agreement across perspectives
    LEARNED_CONFIDENCE = "learned_confidence"      # ML-learned confidence
    HYBRID_SCORING = "hybrid_scoring"              # Multiple methods combined


@dataclass
class PerspectiveModel:
    """Configuration for a single embedding perspective."""
    perspective_type: PerspectiveType
    model_name: str
    model_path: Optional[str] = None
    device: str = "cpu"
    max_sequence_length: int = 512
    embedding_dimension: Optional[int] = None
    specialization_keywords: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    weight: float = 1.0
    
    # Model instance (loaded at runtime)
    model_instance: Optional[SentenceTransformer] = None
    is_loaded: bool = False


@dataclass
class PerspectiveEmbedding:
    """Embedding result from a single perspective."""
    perspective_type: PerspectiveType
    embedding: np.ndarray
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    model_name: str = ""
    
    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.confidence >= 0.9:
            return "very_high"
        elif self.confidence >= 0.75:
            return "high"
        elif self.confidence >= 0.5:
            return "medium"
        else:
            return "low"


@dataclass
class MultiPerspectiveResult:
    """Result containing embeddings from multiple perspectives."""
    text: str
    perspectives: Dict[PerspectiveType, PerspectiveEmbedding]
    recommended_perspective: PerspectiveType
    ensemble_embedding: np.ndarray
    confidence_scores: Dict[PerspectiveType, float]
    routing_strategy: RoutingStrategy
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_best_perspective(self) -> PerspectiveEmbedding:
        """Get the embedding from the best-performing perspective."""
        return self.perspectives[self.recommended_perspective]
    
    def get_weighted_average_embedding(self) -> np.ndarray:
        """Get weighted average of all perspective embeddings."""
        if not self.perspectives:
            return self.ensemble_embedding
        
        weighted_embeddings = []
        total_weight = 0.0
        
        for perspective_type, perspective_embedding in self.perspectives.items():
            weight = self.confidence_scores.get(perspective_type, 0.0)
            if weight > 0:
                weighted_embeddings.append(weight * perspective_embedding.embedding)
                total_weight += weight
        
        if total_weight > 0:
            result = np.sum(weighted_embeddings, axis=0) / total_weight
            # Normalize
            norm = np.linalg.norm(result)
            if norm > 0:
                result /= norm
            return result
        
        return self.ensemble_embedding


class PerspectiveRouter:
    """
    Smart routing system for selecting appropriate embedding perspectives.
    
    Features:
    - Content-based routing using keyword analysis
    - Similarity-based routing using embedding comparisons
    - ML-learned routing with Random Forest classifier
    - Ensemble routing combining multiple perspectives
    - Performance-based adaptive routing
    """
    
    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID,
        confidence_threshold: float = 0.6
    ):
        self.routing_strategy = routing_strategy
        self.confidence_threshold = confidence_threshold
        
        # Content-based routing rules
        self.content_rules = {
            PerspectiveType.CODE: [
                'function', 'class', 'method', 'algorithm', 'code', 'programming',
                'variable', 'loop', 'if', 'else', 'import', 'def', 'return'
            ],
            PerspectiveType.SCIENTIFIC: [
                'research', 'study', 'hypothesis', 'experiment', 'analysis',
                'data', 'statistical', 'method', 'result', 'conclusion'
            ],
            PerspectiveType.DOMAIN_SPECIFIC: [
                'technical', 'specification', 'requirement', 'implementation',
                'architecture', 'design', 'protocol', 'standard'
            ],
            PerspectiveType.CONTEXTUAL: [
                'context', 'situation', 'environment', 'background',
                'circumstance', 'setting', 'condition'
            ]
        }
        
        # ML components for learned routing
        self.routing_classifier: Optional[RandomForestClassifier] = None
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        # Performance tracking
        self.routing_history: List[Dict[str, Any]] = []
        self.perspective_performance: Dict[PerspectiveType, List[float]] = defaultdict(list)
    
    async def route_to_perspectives(
        self,
        text: str,
        available_perspectives: Set[PerspectiveType],
        user_preferences: Optional[Dict[PerspectiveType, float]] = None
    ) -> Dict[PerspectiveType, float]:
        """Route text to appropriate perspectives with confidence weights."""
        
        if self.routing_strategy == RoutingStrategy.CONTENT_BASED:
            return await self._content_based_routing(text, available_perspectives)
        
        elif self.routing_strategy == RoutingStrategy.SIMILARITY_BASED:
            return await self._similarity_based_routing(text, available_perspectives)
        
        elif self.routing_strategy == RoutingStrategy.ENSEMBLE:
            return await self._ensemble_routing(text, available_perspectives)
        
        elif self.routing_strategy == RoutingStrategy.LEARNED:
            return await self._learned_routing(text, available_perspectives)
        
        elif self.routing_strategy == RoutingStrategy.USER_PREFERENCE:
            return await self._user_preference_routing(text, available_perspectives, user_preferences)
        
        elif self.routing_strategy == RoutingStrategy.HYBRID:
            return await self._hybrid_routing(text, available_perspectives, user_preferences)
        
        else:
            # Default: equal weights for all perspectives
            return {perspective: 1.0 for perspective in available_perspectives}
    
    async def _content_based_routing(
        self,
        text: str,
        available_perspectives: Set[PerspectiveType]
    ) -> Dict[PerspectiveType, float]:
        """Route based on content analysis and keyword matching."""
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        routing_weights = {}
        
        for perspective in available_perspectives:
            if perspective in self.content_rules:
                keywords = self.content_rules[perspective]
                # Calculate keyword overlap
                overlap = len(words.intersection(set(keywords)))
                relevance_score = overlap / len(keywords) if keywords else 0.0
                
                # Boost score if multiple keywords match
                if overlap > 1:
                    relevance_score *= (1.0 + 0.1 * overlap)
                
                routing_weights[perspective] = min(1.0, relevance_score)
            else:
                # Default weight for perspectives without specific rules
                routing_weights[perspective] = 0.5
        
        # Ensure at least one perspective has decent weight
        if all(weight < 0.3 for weight in routing_weights.values()):
            # Fall back to general perspective or equal weights
            if PerspectiveType.GENERAL in routing_weights:
                routing_weights[PerspectiveType.GENERAL] = 0.8
            else:
                # Equal weights for all
                for perspective in routing_weights:
                    routing_weights[perspective] = 0.6
        
        return routing_weights
    
    async def _similarity_based_routing(
        self,
        text: str,
        available_perspectives: Set[PerspectiveType]
    ) -> Dict[PerspectiveType, float]:
        """Route based on similarity to perspective prototypes."""
        
        # This would typically use pre-computed prototype embeddings
        # For now, use a simplified approach based on text characteristics
        
        routing_weights = {}
        text_features = await self._extract_text_features(text)
        
        # Define prototype features for each perspective
        perspective_prototypes = {
            PerspectiveType.CODE: {
                'avg_word_length': 8.0,
                'special_char_ratio': 0.3,
                'camelcase_ratio': 0.4,
                'numeric_ratio': 0.2
            },
            PerspectiveType.SCIENTIFIC: {
                'avg_word_length': 7.5,
                'formal_word_ratio': 0.6,
                'passive_voice_ratio': 0.3,
                'technical_term_ratio': 0.4
            },
            PerspectiveType.GENERAL: {
                'avg_word_length': 5.5,
                'common_word_ratio': 0.8,
                'sentence_length': 15.0,
                'readability_score': 0.7
            }
        }
        
        # Calculate similarity to prototypes
        for perspective in available_perspectives:
            if perspective in perspective_prototypes:
                prototype = perspective_prototypes[perspective]
                similarity = await self._calculate_feature_similarity(text_features, prototype)
                routing_weights[perspective] = similarity
            else:
                routing_weights[perspective] = 0.5  # Default weight
        
        return routing_weights
    
    async def _ensemble_routing(
        self,
        text: str,
        available_perspectives: Set[PerspectiveType]
    ) -> Dict[PerspectiveType, float]:
        """Use ensemble approach - all perspectives with equal weight."""
        
        # Equal weights but adjusted by historical performance
        routing_weights = {}
        
        for perspective in available_perspectives:
            base_weight = 1.0
            
            # Adjust based on historical performance
            if perspective in self.perspective_performance:
                performance_scores = self.perspective_performance[perspective]
                if performance_scores:
                    avg_performance = np.mean(performance_scores[-10:])  # Last 10 scores
                    # Scale performance to weight adjustment
                    performance_factor = 0.5 + avg_performance * 0.5
                    base_weight *= performance_factor
            
            routing_weights[perspective] = base_weight
        
        return routing_weights
    
    async def _learned_routing(
        self,
        text: str,
        available_perspectives: Set[PerspectiveType]
    ) -> Dict[PerspectiveType, float]:
        """Use ML model for learned routing decisions."""
        
        if not self.is_trained or self.routing_classifier is None:
            # Fall back to content-based routing
            return await self._content_based_routing(text, available_perspectives)
        
        try:
            # Extract features for ML model
            features = await self._extract_ml_features(text)
            features_scaled = self.feature_scaler.transform([features])
            
            # Get predictions
            probabilities = self.routing_classifier.predict_proba(features_scaled)[0]
            
            # Map probabilities to available perspectives
            routing_weights = {}
            perspective_list = list(available_perspectives)
            
            for i, perspective in enumerate(perspective_list):
                if i < len(probabilities):
                    routing_weights[perspective] = float(probabilities[i])
                else:
                    routing_weights[perspective] = 0.5
            
            return routing_weights
            
        except Exception as e:
            logger.warning(f"Learned routing failed: {e}, falling back to content-based")
            return await self._content_based_routing(text, available_perspectives)
    
    async def _user_preference_routing(
        self,
        text: str,
        available_perspectives: Set[PerspectiveType],
        user_preferences: Optional[Dict[PerspectiveType, float]]
    ) -> Dict[PerspectiveType, float]:
        """Route based on user preferences."""
        
        if not user_preferences:
            return await self._content_based_routing(text, available_perspectives)
        
        routing_weights = {}
        
        for perspective in available_perspectives:
            # Use user preference or default weight
            preference_weight = user_preferences.get(perspective, 0.5)
            routing_weights[perspective] = preference_weight
        
        # Normalize weights
        total_weight = sum(routing_weights.values())
        if total_weight > 0:
            for perspective in routing_weights:
                routing_weights[perspective] /= total_weight
        
        return routing_weights
    
    async def _hybrid_routing(
        self,
        text: str,
        available_perspectives: Set[PerspectiveType],
        user_preferences: Optional[Dict[PerspectiveType, float]]
    ) -> Dict[PerspectiveType, float]:
        """Combine multiple routing strategies."""
        
        # Get weights from different strategies
        content_weights = await self._content_based_routing(text, available_perspectives)
        ensemble_weights = await self._ensemble_routing(text, available_perspectives)
        
        # Combine weights
        routing_weights = {}
        for perspective in available_perspectives:
            combined_weight = (
                0.5 * content_weights.get(perspective, 0.0) +
                0.3 * ensemble_weights.get(perspective, 0.0)
            )
            
            # Add user preferences if available
            if user_preferences and perspective in user_preferences:
                combined_weight += 0.2 * user_preferences[perspective]
            else:
                combined_weight += 0.2 * 0.5  # Default preference
            
            routing_weights[perspective] = combined_weight
        
        return routing_weights
    
    async def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract statistical features from text."""
        
        words = text.split()
        chars = list(text)
        
        features = {
            'length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'special_char_ratio': sum(1 for c in chars if not c.isalnum() and not c.isspace()) / len(chars) if chars else 0,
            'digit_ratio': sum(1 for c in chars if c.isdigit()) / len(chars) if chars else 0,
            'uppercase_ratio': sum(1 for c in chars if c.isupper()) / len(chars) if chars else 0,
            'camelcase_count': sum(1 for word in words if any(c.isupper() for c in word[1:])),
            'underscore_count': text.count('_'),
            'parentheses_count': text.count('(') + text.count(')'),
            'bracket_count': text.count('[') + text.count(']'),
            'brace_count': text.count('{') + text.count('}')
        }
        
        # Normalize counts by text length
        if features['length'] > 0:
            for key in ['camelcase_count', 'underscore_count', 'parentheses_count', 'bracket_count', 'brace_count']:
                features[f'{key}_ratio'] = features[key] / features['length']
        
        return features
    
    async def _calculate_feature_similarity(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """Calculate similarity between feature dictionaries."""
        
        common_keys = set(features1.keys()).intersection(set(features2.keys()))
        if not common_keys:
            return 0.0
        
        # Calculate cosine similarity
        vec1 = np.array([features1[key] for key in common_keys])
        vec2 = np.array([features2[key] for key in common_keys])
        
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return max(0.0, similarity)
    
    async def _extract_ml_features(self, text: str) -> List[float]:
        """Extract features for ML routing model."""
        
        text_features = await self._extract_text_features(text)
        
        # Convert to feature vector
        feature_names = [
            'length', 'word_count', 'avg_word_length', 'sentence_count',
            'special_char_ratio', 'digit_ratio', 'uppercase_ratio',
            'camelcase_count_ratio', 'underscore_count_ratio',
            'parentheses_count_ratio', 'bracket_count_ratio', 'brace_count_ratio'
        ]
        
        features = []
        for name in feature_names:
            features.append(text_features.get(name, 0.0))
        
        return features
    
    async def train_routing_model(self, training_data: List[Dict[str, Any]]):
        """Train ML model for learned routing."""
        
        if len(training_data) < 50:
            logger.warning("Insufficient data for routing model training")
            return
        
        try:
            # Extract features and labels
            X = []
            y = []
            
            for data in training_data:
                text = data.get('text', '')
                best_perspective = data.get('best_perspective', 'general')
                
                features = await self._extract_ml_features(text)
                X.append(features)
                y.append(best_perspective)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train classifier
            self.routing_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            self.routing_classifier.fit(X_scaled, y)
            
            self.is_trained = True
            
            logger.info(f"Routing model trained on {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train routing model: {e}")
    
    def record_routing_feedback(
        self,
        text: str,
        chosen_perspective: PerspectiveType,
        performance_score: float
    ):
        """Record feedback for routing performance."""
        
        self.routing_history.append({
            'text': text,
            'perspective': chosen_perspective,
            'performance': performance_score,
            'timestamp': datetime.utcnow()
        })
        
        # Update perspective performance
        self.perspective_performance[chosen_perspective].append(performance_score)
        
        # Keep only recent history
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-800:]  # Keep last 800
        
        for perspective in self.perspective_performance:
            if len(self.perspective_performance[perspective]) > 100:
                self.perspective_performance[perspective] = self.perspective_performance[perspective][-80:]


class ConfidenceEstimator:
    """
    Confidence estimation for multi-perspective embeddings.
    
    Features:
    - Internal consistency checking
    - Cross-perspective validation
    - Ensemble agreement scoring
    - Learned confidence models
    - Hybrid confidence scoring
    """
    
    def __init__(
        self,
        confidence_method: ConfidenceMethod = ConfidenceMethod.HYBRID_SCORING,
        min_confidence: float = 0.1
    ):
        self.confidence_method = confidence_method
        self.min_confidence = min_confidence
        
        # For learned confidence
        self.confidence_model: Optional[RandomForestClassifier] = None
        self.confidence_scaler = StandardScaler()
        self.confidence_trained = False
    
    async def estimate_confidence(
        self,
        text: str,
        perspectives: Dict[PerspectiveType, PerspectiveEmbedding],
        routing_weights: Dict[PerspectiveType, float]
    ) -> Dict[PerspectiveType, float]:
        """Estimate confidence for each perspective embedding."""
        
        if self.confidence_method == ConfidenceMethod.INTERNAL_CONSISTENCY:
            return await self._internal_consistency_confidence(perspectives)
        
        elif self.confidence_method == ConfidenceMethod.CROSS_VALIDATION:
            return await self._cross_validation_confidence(perspectives)
        
        elif self.confidence_method == ConfidenceMethod.ENSEMBLE_AGREEMENT:
            return await self._ensemble_agreement_confidence(perspectives)
        
        elif self.confidence_method == ConfidenceMethod.LEARNED_CONFIDENCE:
            return await self._learned_confidence(text, perspectives, routing_weights)
        
        elif self.confidence_method == ConfidenceMethod.HYBRID_SCORING:
            return await self._hybrid_confidence_scoring(text, perspectives, routing_weights)
        
        else:
            # Default: use routing weights as confidence
            return {perspective: weight for perspective, weight in routing_weights.items()}
    
    async def _internal_consistency_confidence(
        self,
        perspectives: Dict[PerspectiveType, PerspectiveEmbedding]
    ) -> Dict[PerspectiveType, float]:
        """Estimate confidence based on internal consistency metrics."""
        
        confidence_scores = {}
        
        for perspective_type, embedding_result in perspectives.items():
            # Use embedding magnitude and distribution as consistency metrics
            embedding = embedding_result.embedding
            
            # Calculate various consistency metrics
            magnitude = np.linalg.norm(embedding)
            entropy = -np.sum(np.abs(embedding) * np.log(np.abs(embedding) + 1e-10))
            uniformity = 1.0 - np.std(embedding)
            
            # Combine metrics into confidence score
            confidence = (
                0.4 * min(1.0, magnitude) +
                0.3 * min(1.0, entropy / 10.0) +
                0.3 * max(0.0, uniformity)
            )
            
            confidence_scores[perspective_type] = max(self.min_confidence, confidence)
        
        return confidence_scores
    
    async def _cross_validation_confidence(
        self,
        perspectives: Dict[PerspectiveType, PerspectiveEmbedding]
    ) -> Dict[PerspectiveType, float]:
        """Estimate confidence based on cross-perspective validation."""
        
        confidence_scores = {}
        
        if len(perspectives) < 2:
            # Can't do cross-validation with only one perspective
            return {p: 0.8 for p in perspectives}
        
        # Calculate pairwise similarities
        perspective_list = list(perspectives.keys())
        embeddings = [perspectives[p].embedding for p in perspective_list]
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        for i, perspective_type in enumerate(perspective_list):
            # Average similarity with other perspectives
            other_similarities = [
                similarity_matrix[i][j] for j in range(len(perspective_list)) if i != j
            ]
            
            if other_similarities:
                avg_similarity = np.mean(other_similarities)
                # Higher similarity with others suggests higher confidence
                confidence = max(self.min_confidence, avg_similarity)
            else:
                confidence = 0.8  # Default for single perspective
            
            confidence_scores[perspective_type] = confidence
        
        return confidence_scores
    
    async def _ensemble_agreement_confidence(
        self,
        perspectives: Dict[PerspectiveType, PerspectiveEmbedding]
    ) -> Dict[PerspectiveType, float]:
        """Estimate confidence based on ensemble agreement."""
        
        if len(perspectives) < 2:
            return {p: 0.8 for p in perspectives}
        
        # Calculate centroid of all embeddings
        embeddings = list(perspectives.values())
        centroid = np.mean([emb.embedding for emb in embeddings], axis=0)
        
        confidence_scores = {}
        
        for perspective_type, embedding_result in perspectives.items():
            # Similarity to ensemble centroid
            similarity = cosine_similarity(
                embedding_result.embedding.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0, 0]
            
            # Higher similarity to consensus suggests higher confidence
            confidence = max(self.min_confidence, similarity)
            confidence_scores[perspective_type] = confidence
        
        return confidence_scores
    
    async def _learned_confidence(
        self,
        text: str,
        perspectives: Dict[PerspectiveType, PerspectiveEmbedding],
        routing_weights: Dict[PerspectiveType, float]
    ) -> Dict[PerspectiveType, float]:
        """Use learned model for confidence estimation."""
        
        if not self.confidence_trained or self.confidence_model is None:
            # Fall back to ensemble agreement
            return await self._ensemble_agreement_confidence(perspectives)
        
        try:
            confidence_scores = {}
            
            for perspective_type, embedding_result in perspectives.items():
                # Extract features for confidence prediction
                features = await self._extract_confidence_features(
                    text, embedding_result, routing_weights.get(perspective_type, 0.0)
                )
                
                features_scaled = self.confidence_scaler.transform([features])
                confidence = self.confidence_model.predict(features_scaled)[0]
                
                confidence_scores[perspective_type] = max(self.min_confidence, confidence)
            
            return confidence_scores
            
        except Exception as e:
            logger.warning(f"Learned confidence estimation failed: {e}")
            return await self._ensemble_agreement_confidence(perspectives)
    
    async def _hybrid_confidence_scoring(
        self,
        text: str,
        perspectives: Dict[PerspectiveType, PerspectiveEmbedding],
        routing_weights: Dict[PerspectiveType, float]
    ) -> Dict[PerspectiveType, float]:
        """Combine multiple confidence estimation methods."""
        
        # Get confidence from different methods
        internal_scores = await self._internal_consistency_confidence(perspectives)
        agreement_scores = await self._ensemble_agreement_confidence(perspectives)
        
        # Combine scores
        confidence_scores = {}
        
        for perspective_type in perspectives:
            internal_conf = internal_scores.get(perspective_type, 0.5)
            agreement_conf = agreement_scores.get(perspective_type, 0.5)
            routing_conf = routing_weights.get(perspective_type, 0.5)
            
            # Weighted combination
            combined_confidence = (
                0.4 * internal_conf +
                0.4 * agreement_conf +
                0.2 * routing_conf
            )
            
            confidence_scores[perspective_type] = max(self.min_confidence, combined_confidence)
        
        return confidence_scores
    
    async def _extract_confidence_features(
        self,
        text: str,
        embedding_result: PerspectiveEmbedding,
        routing_weight: float
    ) -> List[float]:
        """Extract features for learned confidence estimation."""
        
        embedding = embedding_result.embedding
        
        features = [
            len(text),                                    # Text length
            len(text.split()),                           # Word count
            np.linalg.norm(embedding),                   # Embedding magnitude
            np.std(embedding),                           # Embedding std
            np.mean(np.abs(embedding)),                  # Mean absolute value
            routing_weight,                              # Routing confidence
            embedding_result.processing_time,            # Processing time
            len([x for x in embedding if abs(x) > 0.1]) # Number of significant dimensions
        ]
        
        return features
    
    async def train_confidence_model(self, training_data: List[Dict[str, Any]]):
        """Train model for learned confidence estimation."""
        
        if len(training_data) < 100:
            logger.warning("Insufficient data for confidence model training")
            return
        
        try:
            X = []
            y = []
            
            for data in training_data:
                text = data.get('text', '')
                embedding = np.array(data.get('embedding', []))
                routing_weight = data.get('routing_weight', 0.5)
                actual_confidence = data.get('actual_confidence', 0.5)
                
                if len(embedding) > 0:
                    # Create fake embedding result for feature extraction
                    fake_result = PerspectiveEmbedding(
                        perspective_type=PerspectiveType.GENERAL,
                        embedding=embedding,
                        confidence=0.0,
                        processing_time=data.get('processing_time', 0.1)
                    )
                    
                    features = await self._extract_confidence_features(
                        text, fake_result, routing_weight
                    )
                    
                    X.append(features)
                    y.append(actual_confidence)
            
            if len(X) < 50:
                logger.warning("Insufficient valid training samples")
                return
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.confidence_scaler.fit_transform(X)
            
            # Train model
            self.confidence_model = RandomForestClassifier(
                n_estimators=50,
                random_state=42
            )
            self.confidence_model.fit(X_scaled, y)
            
            self.confidence_trained = True
            
            logger.info(f"Confidence model trained on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train confidence model: {e}")


class MultiPerspectiveEmbedder:
    """
    Multi-perspective embedding generator using multiple local embedding models.
    
    Features:
    - Multiple local embedding models for different perspectives
    - Smart perspective routing with local routing algorithms
    - Confidence scoring using ensemble methods
    - Perspective recommendation using similarity metrics
    - Performance optimization and adaptive learning
    """
    
    def __init__(
        self,
        perspective_models: Dict[PerspectiveType, PerspectiveModel],
        redis_cache: Optional[RedisCache] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID,
        confidence_method: ConfidenceMethod = ConfidenceMethod.HYBRID_SCORING
    ):
        self.perspective_models = perspective_models
        self.redis_cache = redis_cache
        
        # Core components
        self.router = PerspectiveRouter(routing_strategy)
        self.confidence_estimator = ConfidenceEstimator(confidence_method)
        
        # Performance tracking
        self.performance_stats = {
            'requests_processed': 0,
            'perspective_usage': defaultdict(int),
            'average_confidence': defaultdict(list),
            'routing_accuracy': [],
            'processing_times': defaultdict(list)
        }
        
        logger.info(f"MultiPerspectiveEmbedder initialized with {len(perspective_models)} perspectives")
    
    async def initialize(self):
        """Initialize all perspective models."""
        
        for perspective_type, model_config in self.perspective_models.items():
            try:
                # Load model
                if model_config.model_path:
                    model_instance = SentenceTransformer(
                        model_config.model_path,
                        device=model_config.device
                    )
                else:
                    model_instance = SentenceTransformer(
                        model_config.model_name,
                        device=model_config.device
                    )
                
                model_config.model_instance = model_instance
                model_config.is_loaded = True
                
                # Get embedding dimension
                if model_config.embedding_dimension is None:
                    model_config.embedding_dimension = model_instance.get_sentence_embedding_dimension()
                
                logger.info(f"Loaded {perspective_type.value} perspective model: {model_config.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {perspective_type.value} model: {e}")
                model_config.is_loaded = False
        
        # Check if any models loaded successfully
        loaded_models = [p for p in self.perspective_models.values() if p.is_loaded]
        if not loaded_models:
            raise RuntimeError("No perspective models loaded successfully")
        
        logger.info(f"MultiPerspectiveEmbedder initialized with {len(loaded_models)} active perspectives")
    
    async def generate_multi_perspective_embedding(
        self,
        text: str,
        user_preferences: Optional[Dict[PerspectiveType, float]] = None,
        required_perspectives: Optional[Set[PerspectiveType]] = None
    ) -> MultiPerspectiveResult:
        """Generate embeddings from multiple perspectives."""
        
        start_time = datetime.utcnow()
        
        # Determine available perspectives
        available_perspectives = {
            p_type for p_type, model in self.perspective_models.items()
            if model.is_loaded
        }
        
        if required_perspectives:
            available_perspectives = available_perspectives.intersection(required_perspectives)
        
        if not available_perspectives:
            raise ValueError("No available perspectives for embedding generation")
        
        # Route to perspectives
        routing_weights = await self.router.route_to_perspectives(
            text, available_perspectives, user_preferences
        )
        
        # Generate embeddings from selected perspectives
        perspective_embeddings = {}
        
        for perspective_type in available_perspectives:
            if routing_weights.get(perspective_type, 0.0) > 0.1:  # Threshold for inclusion
                try:
                    embedding = await self._generate_perspective_embedding(
                        text, perspective_type
                    )
                    perspective_embeddings[perspective_type] = embedding
                    
                    # Update usage stats
                    self.performance_stats['perspective_usage'][perspective_type] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to generate {perspective_type.value} embedding: {e}")
        
        if not perspective_embeddings:
            raise RuntimeError("Failed to generate any perspective embeddings")
        
        # Estimate confidence for each perspective
        confidence_scores = await self.confidence_estimator.estimate_confidence(
            text, perspective_embeddings, routing_weights
        )
        
        # Determine recommended perspective
        recommended_perspective = max(
            confidence_scores.items(),
            key=lambda x: x[1] * routing_weights.get(x[0], 0.0)
        )[0]
        
        # Generate ensemble embedding
        ensemble_embedding = await self._generate_ensemble_embedding(
            perspective_embeddings, confidence_scores
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update performance stats
        self.performance_stats['requests_processed'] += 1
        for perspective_type, confidence in confidence_scores.items():
            self.performance_stats['average_confidence'][perspective_type].append(confidence)
        
        result = MultiPerspectiveResult(
            text=text,
            perspectives=perspective_embeddings,
            recommended_perspective=recommended_perspective,
            ensemble_embedding=ensemble_embedding,
            confidence_scores=confidence_scores,
            routing_strategy=self.router.routing_strategy,
            processing_time=processing_time,
            metadata={
                'routing_weights': routing_weights,
                'available_perspectives': list(available_perspectives),
                'user_preferences': user_preferences
            }
        )
        
        return result
    
    async def _generate_perspective_embedding(
        self,
        text: str,
        perspective_type: PerspectiveType
    ) -> PerspectiveEmbedding:
        """Generate embedding from a specific perspective."""
        
        model_config = self.perspective_models[perspective_type]
        
        if not model_config.is_loaded or model_config.model_instance is None:
            raise ValueError(f"Perspective {perspective_type.value} not loaded")
        
        start_time = datetime.utcnow()
        
        # Check cache first
        cache_key = f"perspective_emb:{perspective_type.value}:{hashlib.md5(text.encode()).hexdigest()}"
        if self.redis_cache:
            cached_result = await self.redis_cache.get(cache_key)
            if cached_result is not None:
                return PerspectiveEmbedding(
                    perspective_type=perspective_type,
                    embedding=np.array(cached_result['embedding']),
                    confidence=cached_result['confidence'],
                    metadata={'cached': True},
                    processing_time=cached_result['processing_time'],
                    model_name=model_config.model_name
                )
        
        try:
            # Generate embedding
            embedding = model_config.model_instance.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Calculate initial confidence (can be improved with more sophisticated methods)
            confidence = await self._calculate_initial_confidence(
                text, embedding, model_config
            )
            
            result = PerspectiveEmbedding(
                perspective_type=perspective_type,
                embedding=embedding,
                confidence=confidence,
                metadata={'cached': False},
                processing_time=processing_time,
                model_name=model_config.model_name
            )
            
            # Cache result
            if self.redis_cache:
                cache_data = {
                    'embedding': embedding.tolist(),
                    'confidence': confidence,
                    'processing_time': processing_time
                }
                await self.redis_cache.set(cache_key, cache_data, ttl=1800)  # 30 minutes
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate {perspective_type.value} embedding: {e}")
            raise
    
    async def _calculate_initial_confidence(
        self,
        text: str,
        embedding: np.ndarray,
        model_config: PerspectiveModel
    ) -> float:
        """Calculate initial confidence for perspective embedding."""
        
        # Simple confidence based on embedding properties and text-model fit
        base_confidence = 0.7
        
        # Adjust based on text length
        text_length = len(text)
        if text_length < 10:
            base_confidence *= 0.8  # Lower confidence for very short text
        elif text_length > 1000:
            base_confidence *= 0.9  # Slightly lower for very long text
        
        # Adjust based on specialization keywords
        if model_config.specialization_keywords:
            text_lower = text.lower()
            keyword_matches = sum(
                1 for keyword in model_config.specialization_keywords
                if keyword in text_lower
            )
            keyword_boost = min(0.2, keyword_matches * 0.05)
            base_confidence += keyword_boost
        
        # Adjust based on embedding magnitude (normalized embeddings should be ~1)
        magnitude = np.linalg.norm(embedding)
        if 0.9 <= magnitude <= 1.1:
            base_confidence += 0.05  # Boost for well-normalized embeddings
        
        return min(1.0, max(0.1, base_confidence))
    
    async def _generate_ensemble_embedding(
        self,
        perspective_embeddings: Dict[PerspectiveType, PerspectiveEmbedding],
        confidence_scores: Dict[PerspectiveType, float]
    ) -> np.ndarray:
        """Generate ensemble embedding from multiple perspectives."""
        
        if not perspective_embeddings:
            raise ValueError("No perspective embeddings provided")
        
        # Weighted average based on confidence scores
        weighted_embeddings = []
        total_weight = 0.0
        
        for perspective_type, embedding_result in perspective_embeddings.items():
            confidence = confidence_scores.get(perspective_type, 0.0)
            if confidence > 0:
                weighted_embeddings.append(confidence * embedding_result.embedding)
                total_weight += confidence
        
        if total_weight > 0:
            ensemble_embedding = np.sum(weighted_embeddings, axis=0) / total_weight
        else:
            # Fallback to simple average
            embeddings = [emb.embedding for emb in perspective_embeddings.values()]
            ensemble_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(ensemble_embedding)
        if norm > 0:
            ensemble_embedding /= norm
        
        return ensemble_embedding
    
    async def recommend_perspective(
        self,
        text: str,
        user_feedback: Optional[Dict[PerspectiveType, float]] = None
    ) -> Tuple[PerspectiveType, float]:
        """Recommend the best perspective for given text."""
        
        available_perspectives = {
            p_type for p_type, model in self.perspective_models.items()
            if model.is_loaded
        }
        
        # Get routing weights
        routing_weights = await self.router.route_to_perspectives(
            text, available_perspectives
        )
        
        # Adjust with user feedback if provided
        if user_feedback:
            for perspective_type, feedback_score in user_feedback.items():
                if perspective_type in routing_weights:
                    routing_weights[perspective_type] *= (0.5 + 0.5 * feedback_score)
        
        # Find best perspective
        best_perspective = max(routing_weights.items(), key=lambda x: x[1])
        
        return best_perspective[0], best_perspective[1]
    
    async def update_performance_feedback(
        self,
        text: str,
        chosen_perspective: PerspectiveType,
        performance_score: float
    ):
        """Update performance feedback for adaptive learning."""
        
        # Update router feedback
        self.router.record_routing_feedback(text, chosen_perspective, performance_score)
        
        # Update performance stats
        self.performance_stats['routing_accuracy'].append(performance_score)
        if len(self.performance_stats['routing_accuracy']) > 1000:
            self.performance_stats['routing_accuracy'] = self.performance_stats['routing_accuracy'][-800:]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        # Calculate average confidence by perspective
        avg_confidence = {}
        for perspective_type, confidences in self.performance_stats['average_confidence'].items():
            if confidences:
                avg_confidence[perspective_type.value] = np.mean(confidences[-100:])  # Last 100
        
        return {
            'requests_processed': self.performance_stats['requests_processed'],
            'perspective_usage': dict(self.performance_stats['perspective_usage']),
            'average_confidence_by_perspective': avg_confidence,
            'overall_routing_accuracy': np.mean(self.performance_stats['routing_accuracy'][-100:]) if self.performance_stats['routing_accuracy'] else 0.0,
            'loaded_perspectives': [
                p_type.value for p_type, model in self.perspective_models.items()
                if model.is_loaded
            ],
            'router_strategy': self.router.routing_strategy.value,
            'confidence_method': self.confidence_estimator.confidence_method.value
        }
    
    async def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached embeddings."""
        
        if not self.redis_cache:
            return
        
        try:
            # This would implement cache cleanup logic
            # For now, just log the action
            logger.info(f"Cache cleanup requested for entries older than {max_age_hours} hours")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    async def analyze_perspective_effectiveness(
        self,
        user_id: Optional[str] = None,
        analysis_window_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze the effectiveness of different perspectives for retrieval."""
        try:
            # Collect usage statistics for each perspective
            perspective_stats = {}
            
            for perspective_type in PerspectiveType:
                stats = {
                    "total_requests": 0,
                    "avg_confidence": 0.0,
                    "avg_processing_time_ms": 0.0,
                    "success_rate": 1.0,
                    "user_satisfaction": 0.0,
                    "retrieval_accuracy": 0.0,
                    "cache_hit_rate": 0.0
                }
                
                # Get perspective-specific metrics
                if perspective_type in self.perspective_metrics:
                    metrics = self.perspective_metrics[perspective_type]
                    stats.update({
                        "total_requests": metrics.total_requests,
                        "avg_confidence": metrics.avg_confidence,
                        "avg_processing_time_ms": metrics.avg_processing_time_ms,
                        "success_rate": metrics.success_rate,
                        "cache_hit_rate": getattr(metrics, 'cache_hit_rate', 0.0)
                    })
                
                perspective_stats[perspective_type.value] = stats
            
            # Calculate overall effectiveness scores
            effectiveness_analysis = {
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_window_days": analysis_window_days,
                "user_id": user_id,
                "perspective_statistics": perspective_stats,
                "recommendations": [],
                "optimization_opportunities": []
            }
            
            # Identify best performing perspectives
            best_perspectives = []
            for p_type, stats in perspective_stats.items():
                if stats["success_rate"] > 0.8 and stats["avg_confidence"] > 0.7:
                    best_perspectives.append({
                        "perspective": p_type,
                        "score": (stats["success_rate"] + stats["avg_confidence"]) / 2,
                        "reason": "High success rate and confidence"
                    })
            
            # Sort by effectiveness score
            best_perspectives.sort(key=lambda x: x["score"], reverse=True)
            effectiveness_analysis["best_perspectives"] = best_perspectives[:3]
            
            # Generate recommendations
            recommendations = []
            
            # Check for underperforming perspectives
            for p_type, stats in perspective_stats.items():
                if stats["success_rate"] < 0.6:
                    recommendations.append({
                        "type": "improvement",
                        "perspective": p_type,
                        "issue": "Low success rate",
                        "recommendation": f"Consider retraining or replacing {p_type} perspective model",
                        "priority": "high"
                    })
                
                if stats["avg_processing_time_ms"] > 500:
                    recommendations.append({
                        "type": "performance",
                        "perspective": p_type,
                        "issue": "High processing time",
                        "recommendation": f"Optimize {p_type} perspective for faster processing",
                        "priority": "medium"
                    })
            
            # Check for perspective imbalance
            request_counts = [stats["total_requests"] for stats in perspective_stats.values()]
            if request_counts and max(request_counts) > 10 * min([r for r in request_counts if r > 0]):
                recommendations.append({
                    "type": "balance",
                    "issue": "Perspective usage imbalance",
                    "recommendation": "Consider promoting underused perspectives or demoting overused ones",
                    "priority": "low"
                })
            
            effectiveness_analysis["recommendations"] = recommendations
            
            # Identify optimization opportunities
            optimizations = []
            
            # Cache optimization
            low_cache_perspectives = [
                p for p, stats in perspective_stats.items() 
                if stats["cache_hit_rate"] < 0.3 and stats["total_requests"] > 10
            ]
            if low_cache_perspectives:
                optimizations.append({
                    "type": "caching",
                    "description": f"Improve caching for {', '.join(low_cache_perspectives)}",
                    "expected_benefit": "Reduced latency and resource usage"
                })
            
            # Model combination optimization
            high_performing = [p for p, stats in perspective_stats.items() if stats["avg_confidence"] > 0.8]
            if len(high_performing) > 1:
                optimizations.append({
                    "type": "ensemble",
                    "description": f"Create ensemble of top perspectives: {', '.join(high_performing)}",
                    "expected_benefit": "Improved accuracy through model combination"
                })
            
            effectiveness_analysis["optimization_opportunities"] = optimizations
            
            return effectiveness_analysis
            
        except Exception as e:
            logger.error(f"Perspective effectiveness analysis failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def auto_optimize_perspectives(
        self,
        effectiveness_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Automatically optimize perspective selection and configuration."""
        try:
            if not effectiveness_analysis:
                effectiveness_analysis = await self.analyze_perspective_effectiveness()
            
            if "error" in effectiveness_analysis:
                return {"error": "Cannot optimize without valid effectiveness analysis"}
            
            optimization_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "optimizations_applied": [],
                "performance_improvements": {},
                "recommendations_implemented": []
            }
            
            recommendations = effectiveness_analysis.get("recommendations", [])
            
            # Implement high-priority recommendations automatically
            for rec in recommendations:
                if rec.get("priority") == "high" and rec.get("type") == "improvement":
                    perspective = rec.get("perspective")
                    if perspective and perspective in self.perspective_weights:
                        # Reduce weight of underperforming perspective
                        old_weight = self.perspective_weights[perspective]
                        new_weight = max(0.1, old_weight * 0.7)  # Reduce by 30%, minimum 0.1
                        self.perspective_weights[perspective] = new_weight
                        
                        optimization_results["optimizations_applied"].append({
                            "type": "weight_reduction",
                            "perspective": perspective,
                            "old_weight": old_weight,
                            "new_weight": new_weight,
                            "reason": rec.get("issue", "Performance issue")
                        })
            
            # Implement caching optimizations
            optimizations = effectiveness_analysis.get("optimization_opportunities", [])
            for opt in optimizations:
                if opt.get("type") == "caching":
                    # Increase cache TTL for low-hit perspectives
                    if hasattr(self, 'cache_ttl'):
                        old_ttl = self.cache_ttl
                        self.cache_ttl = min(3600, old_ttl * 1.5)  # Increase TTL by 50%, max 1 hour
                        
                        optimization_results["optimizations_applied"].append({
                            "type": "cache_ttl_increase",
                            "old_ttl_seconds": old_ttl,
                            "new_ttl_seconds": self.cache_ttl,
                            "reason": "Improve cache efficiency"
                        })
            
            # Normalize perspective weights
            total_weight = sum(self.perspective_weights.values())
            if total_weight > 0:
                for perspective in self.perspective_weights:
                    self.perspective_weights[perspective] /= total_weight
            
            # Calculate expected performance improvements
            stats = effectiveness_analysis.get("perspective_statistics", {})
            before_avg_confidence = np.mean([
                s.get("avg_confidence", 0) for s in stats.values()
            ]) if stats else 0
            
            # Estimate improvement (simplified calculation)
            estimated_improvement = len(optimization_results["optimizations_applied"]) * 0.05
            optimization_results["performance_improvements"] = {
                "estimated_confidence_improvement": estimated_improvement,
                "estimated_latency_reduction_ms": len(optimization_results["optimizations_applied"]) * 10,
                "estimated_cache_hit_improvement": 0.1 if any(
                    opt["type"] == "cache_ttl_increase" 
                    for opt in optimization_results["optimizations_applied"]
                ) else 0
            }
            
            logger.info(
                "Auto-optimization completed",
                optimizations_count=len(optimization_results["optimizations_applied"]),
                estimated_improvement=estimated_improvement
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Auto-optimization failed: {e}")
            return {"error": str(e)}
    
    async def benchmark_perspectives(
        self,
        test_queries: List[str],
        ground_truth: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Benchmark different perspectives against test queries."""
        try:
            if not test_queries:
                return {"error": "No test queries provided"}
            
            benchmark_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "test_queries_count": len(test_queries),
                "perspective_results": {},
                "overall_rankings": [],
                "performance_matrix": {}
            }
            
            # Test each perspective individually
            for perspective_type in PerspectiveType:
                perspective_results = {
                    "total_queries": len(test_queries),
                    "successful_embeddings": 0,
                    "avg_processing_time_ms": 0.0,
                    "avg_confidence": 0.0,
                    "errors": [],
                    "query_results": []
                }
                
                processing_times = []
                confidences = []
                
                for i, query in enumerate(test_queries):
                    start_time = time.time()
                    
                    try:
                        # Generate embedding with specific perspective
                        result = await self.generate_contextual_embedding(
                            text=query,
                            context=EmbeddingContext(
                                query=query,
                                perspective_type=perspective_type,
                                include_metadata=True
                            )
                        )
                        
                        processing_time = (time.time() - start_time) * 1000
                        processing_times.append(processing_time)
                        
                        if result and result.embedding is not None:
                            perspective_results["successful_embeddings"] += 1
                            confidences.append(result.confidence)
                            
                            query_result = {
                                "query_index": i,
                                "processing_time_ms": processing_time,
                                "confidence": result.confidence,
                                "embedding_dimension": len(result.embedding),
                                "metadata": result.metadata
                            }
                            
                            # Compare with ground truth if available
                            if ground_truth and i < len(ground_truth):
                                truth = ground_truth[i]
                                if "expected_confidence" in truth:
                                    query_result["confidence_diff"] = abs(
                                        result.confidence - truth["expected_confidence"]
                                    )
                            
                            perspective_results["query_results"].append(query_result)
                        
                    except Exception as e:
                        perspective_results["errors"].append({
                            "query_index": i,
                            "error": str(e)
                        })
                
                # Calculate averages
                if processing_times:
                    perspective_results["avg_processing_time_ms"] = np.mean(processing_times)
                if confidences:
                    perspective_results["avg_confidence"] = np.mean(confidences)
                
                # Calculate success rate
                perspective_results["success_rate"] = (
                    perspective_results["successful_embeddings"] / len(test_queries)
                )
                
                benchmark_results["perspective_results"][perspective_type.value] = perspective_results
            
            # Create overall rankings
            rankings = []
            for perspective, results in benchmark_results["perspective_results"].items():
                score = (
                    results["success_rate"] * 0.4 +
                    results["avg_confidence"] * 0.3 +
                    (1.0 - min(1.0, results["avg_processing_time_ms"] / 1000.0)) * 0.3
                )
                rankings.append({
                    "perspective": perspective,
                    "score": score,
                    "success_rate": results["success_rate"],
                    "avg_confidence": results["avg_confidence"],
                    "avg_processing_time_ms": results["avg_processing_time_ms"]
                })
            
            rankings.sort(key=lambda x: x["score"], reverse=True)
            benchmark_results["overall_rankings"] = rankings
            
            # Create performance matrix
            metrics = ["success_rate", "avg_confidence", "avg_processing_time_ms"]
            performance_matrix = {}
            
            for metric in metrics:
                performance_matrix[metric] = {}
                for perspective, results in benchmark_results["perspective_results"].items():
                    performance_matrix[metric][perspective] = results[metric]
            
            benchmark_results["performance_matrix"] = performance_matrix
            
            logger.info(
                "Perspective benchmarking completed",
                test_queries_count=len(test_queries),
                perspectives_tested=len(benchmark_results["perspective_results"]),
                top_performer=rankings[0]["perspective"] if rankings else "none"
            )
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Perspective benchmarking failed: {e}")
            return {"error": str(e)}
    
    async def export_perspective_configuration(self) -> Dict[str, Any]:
        """Export current perspective configuration for backup or transfer."""
        try:
            config = {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0",
                "perspective_weights": dict(self.perspective_weights),
                "perspective_models": {},
                "perspective_metrics": {},
                "router_config": {
                    "selection_strategy": getattr(self, 'selection_strategy', 'confidence_weighted'),
                    "min_confidence_threshold": getattr(self, 'min_confidence_threshold', 0.5),
                    "enable_fallback": getattr(self, 'enable_fallback', True)
                },
                "fusion_config": {
                    "fusion_method": getattr(self, 'fusion_method', 'weighted_average'),
                    "normalization": getattr(self, 'normalization', True),
                    "dimension_alignment": getattr(self, 'dimension_alignment', True)
                }
            }
            
            # Export perspective model configurations
            for perspective_type in PerspectiveType:
                if perspective_type in self.perspective_models:
                    model_info = self.perspective_models[perspective_type]
                    config["perspective_models"][perspective_type.value] = {
                        "model_name": getattr(model_info, 'model_name', 'unknown'),
                        "model_path": getattr(model_info, 'model_path', 'unknown'),
                        "embedding_dimension": getattr(model_info, 'embedding_dimension', 'unknown'),
                        "max_sequence_length": getattr(model_info, 'max_sequence_length', 'unknown'),
                        "last_updated": getattr(model_info, 'last_updated', datetime.utcnow()).isoformat()
                    }
            
            # Export perspective metrics
            for perspective_type in PerspectiveType:
                if perspective_type in self.perspective_metrics:
                    metrics = self.perspective_metrics[perspective_type]
                    config["perspective_metrics"][perspective_type.value] = {
                        "total_requests": metrics.total_requests,
                        "avg_confidence": metrics.avg_confidence,
                        "avg_processing_time_ms": metrics.avg_processing_time_ms,
                        "success_rate": metrics.success_rate,
                        "last_updated": metrics.last_updated.isoformat()
                    }
            
            return config
            
        except Exception as e:
            logger.error(f"Configuration export failed: {e}")
            return {"error": str(e)}
    
    async def import_perspective_configuration(self, config: Dict[str, Any]) -> bool:
        """Import perspective configuration from backup."""
        try:
            if "version" not in config:
                raise ValueError("Invalid configuration format - missing version")
            
            # Import perspective weights
            if "perspective_weights" in config:
                for perspective_name, weight in config["perspective_weights"].items():
                    try:
                        perspective_type = PerspectiveType(perspective_name)
                        self.perspective_weights[perspective_type] = float(weight)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid perspective weight for {perspective_name}: {e}")
            
            # Import router configuration
            if "router_config" in config:
                router_config = config["router_config"]
                self.selection_strategy = router_config.get("selection_strategy", "confidence_weighted")
                self.min_confidence_threshold = router_config.get("min_confidence_threshold", 0.5)
                self.enable_fallback = router_config.get("enable_fallback", True)
            
            # Import fusion configuration
            if "fusion_config" in config:
                fusion_config = config["fusion_config"]
                self.fusion_method = fusion_config.get("fusion_method", "weighted_average")
                self.normalization = fusion_config.get("normalization", True)
                self.dimension_alignment = fusion_config.get("dimension_alignment", True)
            
            logger.info(
                "Perspective configuration imported successfully",
                perspectives_imported=len(config.get("perspective_weights", {})),
                config_version=config.get("version", "unknown")
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration import failed: {e}")
            return False


# Export main components
__all__ = [
    "MultiPerspectiveEmbedder",
    "PerspectiveType",
    "PerspectiveModel", 
    "EmbeddingContext",
    "PerspectiveResult",
    "PerspectiveMetrics",
    "PerspectiveRouter"
]