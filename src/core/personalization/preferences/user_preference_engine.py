"""
User Preference Learning Engine for Memory Personalization.

This module provides comprehensive user preference learning capabilities with collaborative filtering,
preference extraction using local algorithms, profile management with local storage,
and evolution tracking. All processing is performed locally with zero external dependencies.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import pickle
import hashlib
import threading
from pathlib import Path

# ML and recommendation imports - all local
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
import implicit
from surprise import Dataset, Reader, SVD, SVDpp, NMF as SurpriseNMF
from surprise import accuracy
from surprise.model_selection import train_test_split
import lightfm
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ....core.cache.redis_cache import RedisCache
from ....core.utils.config import settings

logger = structlog.get_logger(__name__)


class PreferenceType(str, Enum):
    """Types of user preferences."""
    CONTENT_TOPIC = "content_topic"         # Topic-based preferences
    MEMORY_TYPE = "memory_type"             # Type of memory content
    INTERACTION_STYLE = "interaction_style" # How user interacts
    TEMPORAL_PATTERN = "temporal_pattern"   # When user accesses
    COMPLEXITY_LEVEL = "complexity_level"   # Content complexity preference
    SOURCE_PREFERENCE = "source_preference" # Preferred information sources
    FORMAT_PREFERENCE = "format_preference" # Text, structured, etc.
    CONTEXT_PREFERENCE = "context_preference" # Work, personal, etc.


class InteractionType(str, Enum):
    """Types of user interactions."""
    VIEW = "view"                           # Memory accessed/viewed
    SAVE = "save"                          # Memory saved/bookmarked
    EDIT = "edit"                          # Memory modified
    DELETE = "delete"                      # Memory deleted
    SEARCH = "search"                      # Search performed
    SHARE = "share"                        # Memory shared
    LIKE = "like"                          # Positive feedback
    DISLIKE = "dislike"                    # Negative feedback
    COMMENT = "comment"                    # Comment added
    TAG = "tag"                            # Tag applied


class LearningStrategy(str, Enum):
    """Strategies for preference learning."""
    COLLABORATIVE = "collaborative"         # Collaborative filtering
    CONTENT_BASED = "content_based"        # Content-based filtering
    HYBRID = "hybrid"                      # Hybrid approach
    MATRIX_FACTORIZATION = "matrix_factorization" # Matrix factorization
    DEEP_LEARNING = "deep_learning"        # Deep learning (local)
    CLUSTERING = "clustering"              # Clustering-based


@dataclass
class UserInteraction:
    """Represents a user interaction with memory."""
    user_id: str
    memory_id: str
    interaction_type: InteractionType
    timestamp: datetime
    duration: Optional[float] = None  # seconds
    rating: Optional[float] = None    # explicit rating 1-5
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "memory_id": self.memory_id,
            "interaction_type": self.interaction_type.value,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "rating": self.rating,
            "context": self.context,
            "metadata": self.metadata
        }


@dataclass
class UserPreference:
    """Represents a learned user preference."""
    user_id: str
    preference_type: PreferenceType
    preference_value: str
    strength: float  # 0-1 confidence score
    timestamp: datetime
    source: str  # how this preference was learned
    evidence_count: int = 1
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "preference_type": self.preference_type.value,
            "preference_value": self.preference_value,
            "strength": self.strength,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "metadata": self.metadata
        }


@dataclass
class UserProfile:
    """Complete user preference profile."""
    user_id: str
    preferences: Dict[PreferenceType, List[UserPreference]]
    interaction_count: int
    profile_strength: float  # overall confidence in profile
    created_at: datetime
    last_updated: datetime
    feature_vector: Optional[np.ndarray] = None
    similarity_cache: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_preferences(self, preference_type: PreferenceType, limit: int = 5) -> List[UserPreference]:
        """Get top preferences of a specific type."""
        prefs = self.preferences.get(preference_type, [])
        return sorted(prefs, key=lambda x: x.strength, reverse=True)[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "preferences": {
                ptype.value: [pref.to_dict() for pref in prefs]
                for ptype, prefs in self.preferences.items()
            },
            "interaction_count": self.interaction_count,
            "profile_strength": self.profile_strength,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "feature_vector": self.feature_vector.tolist() if self.feature_vector is not None else None,
            "metadata": self.metadata
        }


class CollaborativeFilteringEngine:
    """Implements collaborative filtering for preference learning."""
    
    def __init__(self, min_interactions: int = 5, similarity_threshold: float = 0.1):
        self.min_interactions = min_interactions
        self.similarity_threshold = similarity_threshold
        self.user_item_matrix: Optional[csr_matrix] = None
        self.user_index: Dict[str, int] = {}
        self.item_index: Dict[str, int] = {}
        self.reverse_user_index: Dict[int, str] = {}
        self.reverse_item_index: Dict[int, str] = {}
        self.model = None
        self.user_similarities: Dict[str, Dict[str, float]] = {}
    
    def build_user_item_matrix(self, interactions: List[UserInteraction]) -> csr_matrix:
        """Build user-item interaction matrix."""
        # Create user and item mappings
        users = set(interaction.user_id for interaction in interactions)
        items = set(interaction.memory_id for interaction in interactions)
        
        self.user_index = {user: idx for idx, user in enumerate(sorted(users))}
        self.item_index = {item: idx for idx, item in enumerate(sorted(items))}
        self.reverse_user_index = {idx: user for user, idx in self.user_index.items()}
        self.reverse_item_index = {idx: item for item, idx in self.item_index.items()}
        
        # Build interaction matrix
        data = []
        rows = []
        cols = []
        
        for interaction in interactions:
            user_idx = self.user_index[interaction.user_id]
            item_idx = self.item_index[interaction.memory_id]
            
            # Calculate implicit rating based on interaction type
            rating = self._calculate_implicit_rating(interaction)
            
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(rating)
        
        matrix = csr_matrix((data, (rows, cols)), 
                           shape=(len(self.user_index), len(self.item_index)))
        
        self.user_item_matrix = matrix
        return matrix
    
    def _calculate_implicit_rating(self, interaction: UserInteraction) -> float:
        """Calculate implicit rating from interaction."""
        base_ratings = {
            InteractionType.VIEW: 1.0,
            InteractionType.SAVE: 3.0,
            InteractionType.EDIT: 4.0,
            InteractionType.DELETE: -2.0,
            InteractionType.SEARCH: 0.5,
            InteractionType.SHARE: 4.5,
            InteractionType.LIKE: 5.0,
            InteractionType.DISLIKE: -3.0,
            InteractionType.COMMENT: 3.5,
            InteractionType.TAG: 3.0
        }
        
        rating = base_ratings.get(interaction.interaction_type, 1.0)
        
        # Adjust based on duration if available
        if interaction.duration is not None:
            if interaction.duration > 300:  # 5 minutes
                rating *= 1.5
            elif interaction.duration < 10:  # 10 seconds
                rating *= 0.5
        
        # Use explicit rating if available
        if interaction.rating is not None:
            rating = interaction.rating
        
        return max(-5.0, min(5.0, rating))  # Clamp to [-5, 5]
    
    def train_collaborative_model(self) -> bool:
        """Train collaborative filtering model."""
        if self.user_item_matrix is None:
            return False
        
        try:
            # Use implicit library for matrix factorization
            self.model = implicit.als.AlternatingLeastSquares(
                factors=50,
                regularization=0.01,
                iterations=20,
                random_state=42
            )
            
            # Convert to format expected by implicit (item-user matrix)
            item_user_matrix = self.user_item_matrix.T.tocsr()
            
            # Train model
            self.model.fit(item_user_matrix)
            
            logger.info("Collaborative filtering model trained successfully",
                       users=len(self.user_index), items=len(self.item_index))
            return True
            
        except Exception as e:
            logger.error("Failed to train collaborative filtering model", error=str(e))
            return False
    
    def calculate_user_similarities(self, user_id: str, top_k: int = 50) -> Dict[str, float]:
        """Calculate similarities between users."""
        if user_id not in self.user_index or self.user_item_matrix is None:
            return {}
        
        user_idx = self.user_index[user_id]
        user_vector = self.user_item_matrix[user_idx].toarray().flatten()
        
        similarities = {}
        
        for other_user_id, other_idx in self.user_index.items():
            if other_user_id == user_id:
                continue
            
            other_vector = self.user_item_matrix[other_idx].toarray().flatten()
            
            # Calculate cosine similarity
            if np.linalg.norm(user_vector) > 0 and np.linalg.norm(other_vector) > 0:
                similarity = cosine_similarity([user_vector], [other_vector])[0][0]
                
                if similarity > self.similarity_threshold:
                    similarities[other_user_id] = similarity
        
        # Cache and return top similarities
        top_similarities = dict(sorted(similarities.items(), 
                                     key=lambda x: x[1], reverse=True)[:top_k])
        self.user_similarities[user_id] = top_similarities
        
        return top_similarities
    
    def get_user_recommendations(self, user_id: str, exclude_seen: bool = True, 
                               top_k: int = 20) -> List[Tuple[str, float]]:
        """Get item recommendations for a user."""
        if (user_id not in self.user_index or self.model is None or 
            self.user_item_matrix is None):
            return []
        
        try:
            user_idx = self.user_index[user_id]
            
            # Get recommendations from model
            item_user_matrix = self.user_item_matrix.T.tocsr()
            recommendations = self.model.recommend(
                user_idx, 
                item_user_matrix, 
                N=top_k,
                filter_already_liked_items=exclude_seen
            )
            
            # Convert back to memory IDs with scores
            results = []
            for item_idx, score in recommendations:
                memory_id = self.reverse_item_index[item_idx]
                results.append((memory_id, float(score)))
            
            return results
            
        except Exception as e:
            logger.error("Failed to get user recommendations", user_id=user_id, error=str(e))
            return []


class ContentBasedEngine:
    """Implements content-based filtering for preference learning."""
    
    def __init__(self, feature_dim: int = 100):
        self.feature_dim = feature_dim
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.memory_features: Dict[str, np.ndarray] = {}
        self.memory_metadata: Dict[str, Dict[str, Any]] = {}
        self.user_content_profiles: Dict[str, np.ndarray] = {}
        self.scaler = StandardScaler()
    
    def extract_content_features(self, memory_data: List[Dict[str, Any]]) -> bool:
        """Extract features from memory content."""
        try:
            # Prepare text content
            texts = []
            memory_ids = []
            
            for memory in memory_data:
                memory_id = memory.get('id', '')
                content = memory.get('content', '')
                
                # Combine title, content, and tags
                full_text = ' '.join([
                    memory.get('title', ''),
                    content,
                    ' '.join(memory.get('tags', []))
                ])
                
                texts.append(full_text)
                memory_ids.append(memory_id)
                self.memory_metadata[memory_id] = memory
            
            if not texts:
                return False
            
            # Extract TF-IDF features
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
            
            # Apply dimensionality reduction if needed
            if tfidf_features.shape[1] > self.feature_dim:
                svd = TruncatedSVD(n_components=self.feature_dim, random_state=42)
                reduced_features = svd.fit_transform(tfidf_features.toarray())
            else:
                reduced_features = tfidf_features.toarray()
            
            # Normalize features
            reduced_features = self.scaler.fit_transform(reduced_features)
            
            # Store features
            for memory_id, features in zip(memory_ids, reduced_features):
                self.memory_features[memory_id] = features
            
            logger.info("Content features extracted", 
                       memories=len(memory_ids), feature_dim=reduced_features.shape[1])
            return True
            
        except Exception as e:
            logger.error("Failed to extract content features", error=str(e))
            return False
    
    def build_user_content_profile(self, user_id: str, 
                                 interactions: List[UserInteraction]) -> np.ndarray:
        """Build content-based profile for a user."""
        if not self.memory_features:
            return np.zeros(self.feature_dim)
        
        # Filter interactions for this user
        user_interactions = [i for i in interactions if i.user_id == user_id]
        
        if not user_interactions:
            return np.zeros(self.feature_dim)
        
        # Weighted average of content features
        weighted_features = []
        weights = []
        
        for interaction in user_interactions:
            if interaction.memory_id in self.memory_features:
                features = self.memory_features[interaction.memory_id]
                weight = self._calculate_implicit_rating(interaction)
                
                weighted_features.append(features * weight)
                weights.append(weight)
        
        if not weighted_features:
            return np.zeros(self.feature_dim)
        
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            profile = np.sum(weighted_features, axis=0) / total_weight
        else:
            profile = np.mean(weighted_features, axis=0)
        
        # Normalize profile
        if np.linalg.norm(profile) > 0:
            profile = profile / np.linalg.norm(profile)
        
        self.user_content_profiles[user_id] = profile
        return profile
    
    def _calculate_implicit_rating(self, interaction: UserInteraction) -> float:
        """Calculate implicit rating from interaction (same as collaborative)."""
        base_ratings = {
            InteractionType.VIEW: 1.0,
            InteractionType.SAVE: 3.0,
            InteractionType.EDIT: 4.0,
            InteractionType.DELETE: -2.0,
            InteractionType.SEARCH: 0.5,
            InteractionType.SHARE: 4.5,
            InteractionType.LIKE: 5.0,
            InteractionType.DISLIKE: -3.0,
            InteractionType.COMMENT: 3.5,
            InteractionType.TAG: 3.0
        }
        
        rating = base_ratings.get(interaction.interaction_type, 1.0)
        
        if interaction.duration is not None:
            if interaction.duration > 300:
                rating *= 1.5
            elif interaction.duration < 10:
                rating *= 0.5
        
        if interaction.rating is not None:
            rating = interaction.rating
        
        return max(-5.0, min(5.0, rating))
    
    def get_content_recommendations(self, user_id: str, exclude_seen: Set[str] = None,
                                  top_k: int = 20) -> List[Tuple[str, float]]:
        """Get content-based recommendations for a user."""
        if user_id not in self.user_content_profiles:
            return []
        
        exclude_seen = exclude_seen or set()
        user_profile = self.user_content_profiles[user_id]
        
        recommendations = []
        
        for memory_id, memory_features in self.memory_features.items():
            if memory_id in exclude_seen:
                continue
            
            # Calculate similarity
            similarity = cosine_similarity([user_profile], [memory_features])[0][0]
            recommendations.append((memory_id, similarity))
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]


class PreferenceEvolutionTracker:
    """Tracks how user preferences evolve over time."""
    
    def __init__(self, window_size: int = 30):  # days
        self.window_size = window_size
        self.preference_history: Dict[str, List[Tuple[datetime, Dict[str, float]]]] = {}
        self.trend_analysis: Dict[str, Dict[str, Any]] = {}
    
    def record_preference_snapshot(self, user_id: str, preferences: Dict[str, float]):
        """Record a snapshot of user preferences."""
        timestamp = datetime.utcnow()
        
        if user_id not in self.preference_history:
            self.preference_history[user_id] = []
        
        self.preference_history[user_id].append((timestamp, preferences.copy()))
        
        # Keep only recent history
        cutoff_date = timestamp - timedelta(days=self.window_size)
        self.preference_history[user_id] = [
            (ts, prefs) for ts, prefs in self.preference_history[user_id]
            if ts >= cutoff_date
        ]
    
    def analyze_preference_trends(self, user_id: str) -> Dict[str, Any]:
        """Analyze trends in user preferences."""
        if user_id not in self.preference_history:
            return {}
        
        history = self.preference_history[user_id]
        if len(history) < 2:
            return {}
        
        # Get all preference keys
        all_keys = set()
        for _, prefs in history:
            all_keys.update(prefs.keys())
        
        trends = {}
        
        for key in all_keys:
            values = []
            timestamps = []
            
            for ts, prefs in history:
                if key in prefs:
                    values.append(prefs[key])
                    timestamps.append(ts)
            
            if len(values) >= 2:
                # Calculate trend
                time_diffs = [(ts - timestamps[0]).total_seconds() / 86400 for ts in timestamps]
                correlation, p_value = pearsonr(time_diffs, values)
                
                trends[key] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'trend_direction': 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable',
                    'current_value': values[-1],
                    'initial_value': values[0],
                    'change_rate': (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
                }
        
        self.trend_analysis[user_id] = trends
        return trends
    
    def predict_future_preferences(self, user_id: str, days_ahead: int = 7) -> Dict[str, float]:
        """Predict future preferences based on trends."""
        trends = self.trend_analysis.get(user_id, {})
        
        predictions = {}
        
        for key, trend_data in trends.items():
            current_value = trend_data['current_value']
            change_rate = trend_data['change_rate']
            
            # Simple linear prediction
            predicted_value = current_value + (change_rate * days_ahead)
            predictions[key] = max(0.0, min(1.0, predicted_value))  # Clamp to [0, 1]
        
        return predictions


class UserPreferenceEngine:
    """
    Complete User Preference Learning Engine.
    
    Provides comprehensive preference learning with collaborative filtering,
    content-based recommendations, and preference evolution tracking.
    """
    
    def __init__(self, cache: Optional[RedisCache] = None, storage_path: str = "./user_preferences"):
        self.cache = cache
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core engines
        self.collaborative_engine = CollaborativeFilteringEngine()
        self.content_engine = ContentBasedEngine()
        self.evolution_tracker = PreferenceEvolutionTracker()
        
        # Data storage
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interactions: List[UserInteraction] = []
        self.memory_data: List[Dict[str, Any]] = []
        
        # Configuration
        self.learning_strategy = LearningStrategy.HYBRID
        self.update_threshold = 10  # minimum interactions before update
        self.profile_update_interval = timedelta(hours=6)
        
        # Threading for async operations
        self.update_lock = threading.Lock()
        
        logger.info("User preference engine initialized")
    
    async def record_interaction(self, interaction: UserInteraction) -> bool:
        """Record a user interaction."""
        try:
            self.interactions.append(interaction)
            
            # Cache interaction if available
            if self.cache:
                await self.cache.lpush(
                    f"user_interactions:{interaction.user_id}",
                    json.dumps(interaction.to_dict()),
                    max_length=1000
                )
            
            # Trigger profile update if enough new interactions
            await self._maybe_update_user_profile(interaction.user_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to record interaction", error=str(e))
            return False
    
    async def update_memory_data(self, memory_data: List[Dict[str, Any]]) -> bool:
        """Update memory content data for feature extraction."""
        try:
            self.memory_data = memory_data
            
            # Extract content features
            success = self.content_engine.extract_content_features(memory_data)
            
            if success:
                logger.info("Memory data updated", count=len(memory_data))
            
            return success
            
        except Exception as e:
            logger.error("Failed to update memory data", error=str(e))
            return False
    
    async def _maybe_update_user_profile(self, user_id: str):
        """Update user profile if conditions are met."""
        with self.update_lock:
            # Check if user has enough interactions
            user_interactions = [i for i in self.interactions if i.user_id == user_id]
            
            if len(user_interactions) < self.update_threshold:
                return
            
            # Check if enough time has passed
            if user_id in self.user_profiles:
                last_update = self.user_profiles[user_id].last_updated
                if datetime.utcnow() - last_update < self.profile_update_interval:
                    return
            
            # Update profile
            await self._update_user_profile(user_id)
    
    async def _update_user_profile(self, user_id: str) -> UserProfile:
        """Update user profile with latest preferences."""
        try:
            # Get user interactions
            user_interactions = [i for i in self.interactions if i.user_id == user_id]
            
            if not user_interactions:
                # Create empty profile
                profile = UserProfile(
                    user_id=user_id,
                    preferences={},
                    interaction_count=0,
                    profile_strength=0.0,
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
                self.user_profiles[user_id] = profile
                return profile
            
            # Learn preferences using different strategies
            learned_preferences = await self._learn_user_preferences(user_id, user_interactions)
            
            # Calculate profile strength
            profile_strength = min(1.0, len(user_interactions) / 100.0)  # Asymptotic to 1.0
            
            # Create or update profile
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                profile.preferences = learned_preferences
                profile.interaction_count = len(user_interactions)
                profile.profile_strength = profile_strength
                profile.last_updated = datetime.utcnow()
            else:
                profile = UserProfile(
                    user_id=user_id,
                    preferences=learned_preferences,
                    interaction_count=len(user_interactions),
                    profile_strength=profile_strength,
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
            
            # Build content-based profile
            if self.memory_data:
                content_profile = self.content_engine.build_user_content_profile(
                    user_id, user_interactions
                )
                profile.feature_vector = content_profile
            
            self.user_profiles[user_id] = profile
            
            # Record preference snapshot for evolution tracking
            preference_snapshot = self._extract_preference_snapshot(profile)
            self.evolution_tracker.record_preference_snapshot(user_id, preference_snapshot)
            
            # Save profile
            await self._save_user_profile(profile)
            
            logger.info("User profile updated", user_id=user_id, 
                       interactions=len(user_interactions), strength=profile_strength)
            
            return profile
            
        except Exception as e:
            logger.error("Failed to update user profile", user_id=user_id, error=str(e))
            return self.user_profiles.get(user_id, UserProfile(
                user_id=user_id, preferences={}, interaction_count=0,
                profile_strength=0.0, created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            ))
    
    async def _learn_user_preferences(self, user_id: str, 
                                    interactions: List[UserInteraction]) -> Dict[PreferenceType, List[UserPreference]]:
        """Learn user preferences from interactions."""
        preferences = defaultdict(list)
        
        try:
            # Content topic preferences
            topic_prefs = await self._learn_topic_preferences(user_id, interactions)
            preferences[PreferenceType.CONTENT_TOPIC].extend(topic_prefs)
            
            # Memory type preferences
            type_prefs = await self._learn_memory_type_preferences(user_id, interactions)
            preferences[PreferenceType.MEMORY_TYPE].extend(type_prefs)
            
            # Interaction style preferences
            style_prefs = await self._learn_interaction_style_preferences(user_id, interactions)
            preferences[PreferenceType.INTERACTION_STYLE].extend(style_prefs)
            
            # Temporal pattern preferences
            temporal_prefs = await self._learn_temporal_preferences(user_id, interactions)
            preferences[PreferenceType.TEMPORAL_PATTERN].extend(temporal_prefs)
            
            # Context preferences
            context_prefs = await self._learn_context_preferences(user_id, interactions)
            preferences[PreferenceType.CONTEXT_PREFERENCE].extend(context_prefs)
            
        except Exception as e:
            logger.error("Failed to learn user preferences", user_id=user_id, error=str(e))
        
        return dict(preferences)
    
    async def _learn_topic_preferences(self, user_id: str, 
                                     interactions: List[UserInteraction]) -> List[UserPreference]:
        """Learn topic preferences from user interactions."""
        preferences = []
        
        try:
            # Extract topics from memory metadata
            topic_scores = defaultdict(float)
            topic_counts = defaultdict(int)
            
            for interaction in interactions:
                memory_id = interaction.memory_id
                rating = self.content_engine._calculate_implicit_rating(interaction)
                
                # Get memory metadata
                memory_meta = self.content_engine.memory_metadata.get(memory_id, {})
                tags = memory_meta.get('tags', [])
                
                for tag in tags:
                    topic_scores[tag] += rating
                    topic_counts[tag] += 1
            
            # Calculate average scores and create preferences
            for topic, total_score in topic_scores.items():
                count = topic_counts[topic]
                avg_score = total_score / count
                strength = min(1.0, count / 10.0)  # More interactions = higher confidence
                
                if avg_score > 0 and strength > 0.1:
                    pref = UserPreference(
                        user_id=user_id,
                        preference_type=PreferenceType.CONTENT_TOPIC,
                        preference_value=topic,
                        strength=strength,
                        timestamp=datetime.utcnow(),
                        source="interaction_analysis",
                        evidence_count=count
                    )
                    preferences.append(pref)
            
        except Exception as e:
            logger.error("Failed to learn topic preferences", user_id=user_id, error=str(e))
        
        return preferences
    
    async def _learn_memory_type_preferences(self, user_id: str,
                                           interactions: List[UserInteraction]) -> List[UserPreference]:
        """Learn memory type preferences."""
        preferences = []
        
        try:
            type_scores = defaultdict(float)
            type_counts = defaultdict(int)
            
            for interaction in interactions:
                memory_id = interaction.memory_id
                rating = self.content_engine._calculate_implicit_rating(interaction)
                
                memory_meta = self.content_engine.memory_metadata.get(memory_id, {})
                memory_type = memory_meta.get('type', 'general')
                
                type_scores[memory_type] += rating
                type_counts[memory_type] += 1
            
            for mem_type, total_score in type_scores.items():
                count = type_counts[mem_type]
                avg_score = total_score / count
                strength = min(1.0, count / 5.0)
                
                if avg_score > 0 and strength > 0.1:
                    pref = UserPreference(
                        user_id=user_id,
                        preference_type=PreferenceType.MEMORY_TYPE,
                        preference_value=mem_type,
                        strength=strength,
                        timestamp=datetime.utcnow(),
                        source="type_analysis",
                        evidence_count=count
                    )
                    preferences.append(pref)
                    
        except Exception as e:
            logger.error("Failed to learn memory type preferences", user_id=user_id, error=str(e))
        
        return preferences
    
    async def _learn_interaction_style_preferences(self, user_id: str,
                                                 interactions: List[UserInteraction]) -> List[UserPreference]:
        """Learn interaction style preferences."""
        preferences = []
        
        try:
            style_scores = defaultdict(float)
            style_counts = defaultdict(int)
            
            for interaction in interactions:
                interaction_type = interaction.interaction_type.value
                rating = 1.0  # All interactions show some preference
                
                # Weight different interaction types
                if interaction.interaction_type in [InteractionType.SAVE, InteractionType.SHARE]:
                    rating = 2.0
                elif interaction.interaction_type in [InteractionType.LIKE, InteractionType.EDIT]:
                    rating = 1.5
                
                style_scores[interaction_type] += rating
                style_counts[interaction_type] += 1
            
            total_interactions = len(interactions)
            
            for style, total_score in style_scores.items():
                count = style_counts[style]
                proportion = count / total_interactions
                strength = min(1.0, proportion * 2)  # Higher proportion = higher strength
                
                if strength > 0.1:
                    pref = UserPreference(
                        user_id=user_id,
                        preference_type=PreferenceType.INTERACTION_STYLE,
                        preference_value=style,
                        strength=strength,
                        timestamp=datetime.utcnow(),
                        source="interaction_pattern_analysis",
                        evidence_count=count
                    )
                    preferences.append(pref)
                    
        except Exception as e:
            logger.error("Failed to learn interaction style preferences", user_id=user_id, error=str(e))
        
        return preferences
    
    async def _learn_temporal_preferences(self, user_id: str,
                                        interactions: List[UserInteraction]) -> List[UserPreference]:
        """Learn temporal pattern preferences."""
        preferences = []
        
        try:
            hour_scores = defaultdict(float)
            hour_counts = defaultdict(int)
            
            for interaction in interactions:
                hour = interaction.timestamp.hour
                rating = self.content_engine._calculate_implicit_rating(interaction)
                
                hour_scores[hour] += rating
                hour_counts[hour] += 1
            
            # Find peak hours
            total_interactions = len(interactions)
            
            for hour, total_score in hour_scores.items():
                count = hour_counts[hour]
                avg_score = total_score / count
                proportion = count / total_interactions
                
                # High activity periods are preferences
                if proportion > 0.1:  # More than 10% of activity
                    strength = min(1.0, proportion * 3)
                    
                    time_period = self._get_time_period(hour)
                    
                    pref = UserPreference(
                        user_id=user_id,
                        preference_type=PreferenceType.TEMPORAL_PATTERN,
                        preference_value=time_period,
                        strength=strength,
                        timestamp=datetime.utcnow(),
                        source="temporal_analysis",
                        evidence_count=count,
                        metadata={"hour": hour, "avg_score": avg_score}
                    )
                    preferences.append(pref)
                    
        except Exception as e:
            logger.error("Failed to learn temporal preferences", user_id=user_id, error=str(e))
        
        return preferences
    
    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    async def _learn_context_preferences(self, user_id: str,
                                       interactions: List[UserInteraction]) -> List[UserPreference]:
        """Learn context preferences from interaction metadata."""
        preferences = []
        
        try:
            context_scores = defaultdict(float)
            context_counts = defaultdict(int)
            
            for interaction in interactions:
                context = interaction.context
                rating = self.content_engine._calculate_implicit_rating(interaction)
                
                for key, value in context.items():
                    context_key = f"{key}:{value}"
                    context_scores[context_key] += rating
                    context_counts[context_key] += 1
            
            for context_key, total_score in context_scores.items():
                count = context_counts[context_key]
                avg_score = total_score / count
                strength = min(1.0, count / 5.0)
                
                if avg_score > 0 and strength > 0.2:
                    pref = UserPreference(
                        user_id=user_id,
                        preference_type=PreferenceType.CONTEXT_PREFERENCE,
                        preference_value=context_key,
                        strength=strength,
                        timestamp=datetime.utcnow(),
                        source="context_analysis",
                        evidence_count=count
                    )
                    preferences.append(pref)
                    
        except Exception as e:
            logger.error("Failed to learn context preferences", user_id=user_id, error=str(e))
        
        return preferences
    
    def _extract_preference_snapshot(self, profile: UserProfile) -> Dict[str, float]:
        """Extract a snapshot of preferences for trend tracking."""
        snapshot = {}
        
        for pref_type, prefs in profile.preferences.items():
            for pref in prefs:
                key = f"{pref_type.value}:{pref.preference_value}"
                snapshot[key] = pref.strength
        
        return snapshot
    
    async def get_user_recommendations(self, user_id: str, exclude_seen: Set[str] = None,
                                     top_k: int = 20) -> List[Tuple[str, float]]:
        """Get personalized recommendations for a user."""
        exclude_seen = exclude_seen or set()
        
        if self.learning_strategy == LearningStrategy.COLLABORATIVE:
            return self.collaborative_engine.get_user_recommendations(user_id, True, top_k)
        
        elif self.learning_strategy == LearningStrategy.CONTENT_BASED:
            return self.content_engine.get_content_recommendations(user_id, exclude_seen, top_k)
        
        elif self.learning_strategy == LearningStrategy.HYBRID:
            # Combine collaborative and content-based
            collab_recs = self.collaborative_engine.get_user_recommendations(user_id, True, top_k)
            content_recs = self.content_engine.get_content_recommendations(user_id, exclude_seen, top_k)
            
            # Merge and weight recommendations
            combined_scores = defaultdict(float)
            
            for memory_id, score in collab_recs:
                combined_scores[memory_id] += score * 0.6  # 60% collaborative
            
            for memory_id, score in content_recs:
                combined_scores[memory_id] += score * 0.4  # 40% content-based
            
            # Sort and return top recommendations
            recommendations = [(mem_id, score) for mem_id, score in combined_scores.items()]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:top_k]
        
        return []
    
    async def retrain_models(self) -> bool:
        """Retrain recommendation models with latest data."""
        try:
            # Build matrices and train collaborative filtering
            if self.interactions:
                self.collaborative_engine.build_user_item_matrix(self.interactions)
                self.collaborative_engine.train_collaborative_model()
            
            # Update content features
            if self.memory_data:
                self.content_engine.extract_content_features(self.memory_data)
            
            # Update all user profiles
            user_ids = set(interaction.user_id for interaction in self.interactions)
            for user_id in user_ids:
                await self._update_user_profile(user_id)
            
            logger.info("Models retrained successfully", users=len(user_ids), 
                       interactions=len(self.interactions))
            return True
            
        except Exception as e:
            logger.error("Failed to retrain models", error=str(e))
            return False
    
    async def _save_user_profile(self, profile: UserProfile) -> bool:
        """Save user profile to disk."""
        try:
            profile_file = self.storage_path / f"profile_{profile.user_id}.json"
            
            with open(profile_file, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error("Failed to save user profile", user_id=profile.user_id, error=str(e))
            return False
    
    async def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from disk."""
        try:
            profile_file = self.storage_path / f"profile_{user_id}.json"
            
            if not profile_file.exists():
                return None
            
            with open(profile_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct profile
            preferences = {}
            for ptype_str, prefs_data in data.get('preferences', {}).items():
                ptype = PreferenceType(ptype_str)
                prefs = []
                
                for pref_data in prefs_data:
                    pref = UserPreference(
                        user_id=pref_data['user_id'],
                        preference_type=PreferenceType(pref_data['preference_type']),
                        preference_value=pref_data['preference_value'],
                        strength=pref_data['strength'],
                        timestamp=datetime.fromisoformat(pref_data['timestamp']),
                        source=pref_data['source'],
                        evidence_count=pref_data.get('evidence_count', 1),
                        last_updated=datetime.fromisoformat(pref_data['last_updated']) if pref_data.get('last_updated') else None,
                        metadata=pref_data.get('metadata', {})
                    )
                    prefs.append(pref)
                
                preferences[ptype] = prefs
            
            profile = UserProfile(
                user_id=data['user_id'],
                preferences=preferences,
                interaction_count=data['interaction_count'],
                profile_strength=data['profile_strength'],
                created_at=datetime.fromisoformat(data['created_at']),
                last_updated=datetime.fromisoformat(data['last_updated']),
                feature_vector=np.array(data['feature_vector']) if data.get('feature_vector') else None,
                metadata=data.get('metadata', {})
            )
            
            self.user_profiles[user_id] = profile
            return profile
            
        except Exception as e:
            logger.error("Failed to load user profile", user_id=user_id, error=str(e))
            return None
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from memory."""
        return self.user_profiles.get(user_id)
    
    def get_preference_evolution(self, user_id: str) -> Dict[str, Any]:
        """Get preference evolution analysis for a user."""
        return self.evolution_tracker.analyze_preference_trends(user_id)
    
    def predict_user_preferences(self, user_id: str, days_ahead: int = 7) -> Dict[str, float]:
        """Predict future user preferences."""
        return self.evolution_tracker.predict_future_preferences(user_id, days_ahead)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide preference learning metrics."""
        total_users = len(self.user_profiles)
        total_interactions = len(self.interactions)
        
        # Calculate average profile strength
        avg_strength = 0.0
        if self.user_profiles:
            avg_strength = np.mean([p.profile_strength for p in self.user_profiles.values()])
        
        # Interaction type distribution
        interaction_dist = defaultdict(int)
        for interaction in self.interactions:
            interaction_dist[interaction.interaction_type.value] += 1
        
        return {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "average_profile_strength": avg_strength,
            "interaction_distribution": dict(interaction_dist),
            "learning_strategy": self.learning_strategy.value,
            "collaborative_users": len(self.collaborative_engine.user_index),
            "content_memories": len(self.content_engine.memory_features)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform preference engine health check."""
        try:
            # Test preference learning with dummy data
            test_interaction = UserInteraction(
                user_id="test_user",
                memory_id="test_memory",
                interaction_type=InteractionType.VIEW,
                timestamp=datetime.utcnow()
            )
            
            success = await self.record_interaction(test_interaction)
            
            return {
                "status": "healthy" if success else "unhealthy",
                "test_interaction_recorded": success,
                "system_metrics": self.get_system_metrics(),
                "engines_available": {
                    "collaborative": self.collaborative_engine is not None,
                    "content_based": self.content_engine is not None,
                    "evolution_tracker": self.evolution_tracker is not None
                }
            }
            
        except Exception as e:
            logger.error("Preference engine health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "system_metrics": self.get_system_metrics()
            }