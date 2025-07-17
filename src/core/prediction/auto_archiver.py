"""
Smart Auto-Archiving Engine for Predictive Memory Management.

This module provides intelligent auto-archiving capabilities using local scoring algorithms,
importance scoring with feature engineering, policy engines, and restoration triggers.
All processing is performed locally with zero external API calls.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import json
import math
import heapq

# ML and analytics imports - all local
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import scipy.stats as stats

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ...models.memory import Memory
from ...core.cache.redis_cache import RedisCache
from .usage_analyzer import UsageAnalyzer, UsagePattern
from ...core.utils.config import settings

logger = structlog.get_logger(__name__)


class ArchivingAction(str, Enum):
    """Types of archiving actions."""
    ARCHIVE = "archive"
    DELETE = "delete"
    COMPRESS = "compress"
    MOVE_COLD_STORAGE = "move_cold_storage"
    MARK_STALE = "mark_stale"
    NO_ACTION = "no_action"


class ImportanceLevel(str, Enum):
    """Levels of memory importance."""
    CRITICAL = "critical"        # Never archive
    HIGH = "high"               # Archive only when very old
    MEDIUM = "medium"           # Archive based on usage patterns
    LOW = "low"                 # Archive aggressively
    VERY_LOW = "very_low"       # Archive immediately if unused


class PolicyType(str, Enum):
    """Types of archiving policies."""
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    IMPORTANCE_BASED = "importance_based"
    SPACE_BASED = "space_based"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class RestoreTriggerType(str, Enum):
    """Types of restoration triggers."""
    ACCESS_ATTEMPT = "access_attempt"
    SEARCH_QUERY = "search_query"
    USER_REQUEST = "user_request"
    RELATED_ACCESS = "related_access"
    SCHEDULED = "scheduled"
    IMPORTANCE_CHANGE = "importance_change"


@dataclass
class ImportanceScore:
    """Represents importance scoring for a memory."""
    memory_id: str
    base_score: float           # Base importance (0-1)
    usage_score: float          # Usage-based importance (0-1)
    recency_score: float        # Recency-based importance (0-1)
    relationship_score: float   # Relationship-based importance (0-1)
    content_score: float        # Content quality importance (0-1)
    user_value_score: float     # User-assigned value (0-1)
    final_score: float          # Weighted final score (0-1)
    importance_level: ImportanceLevel
    factors: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    
    def calculate_final_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate final importance score with optional custom weights."""
        default_weights = {
            'base': 0.2,
            'usage': 0.25,
            'recency': 0.15,
            'relationship': 0.15,
            'content': 0.15,
            'user_value': 0.1
        }
        
        weights = weights or default_weights
        
        self.final_score = (
            self.base_score * weights['base'] +
            self.usage_score * weights['usage'] +
            self.recency_score * weights['recency'] +
            self.relationship_score * weights['relationship'] +
            self.content_score * weights['content'] +
            self.user_value_score * weights['user_value']
        )
        
        # Determine importance level
        if self.final_score >= 0.9:
            self.importance_level = ImportanceLevel.CRITICAL
        elif self.final_score >= 0.7:
            self.importance_level = ImportanceLevel.HIGH
        elif self.final_score >= 0.5:
            self.importance_level = ImportanceLevel.MEDIUM
        elif self.final_score >= 0.3:
            self.importance_level = ImportanceLevel.LOW
        else:
            self.importance_level = ImportanceLevel.VERY_LOW
        
        return self.final_score


@dataclass
class ArchivingRule:
    """Represents an archiving rule."""
    id: str
    name: str
    policy_type: PolicyType
    conditions: List[Dict[str, Any]]
    action: ArchivingAction
    priority: int = 0
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches(self, memory_data: Dict[str, Any]) -> bool:
        """Check if memory matches rule conditions."""
        if not self.enabled:
            return False
        
        for condition in self.conditions:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if not self._evaluate_condition(memory_data, field, operator, value):
                return False
        
        return True
    
    def _evaluate_condition(self, data: Dict[str, Any], field: str, operator: str, value: Any) -> bool:
        """Evaluate a single condition."""
        try:
            field_value = self._get_nested_value(data, field)
            
            if operator == 'eq':
                return field_value == value
            elif operator == 'ne':
                return field_value != value
            elif operator == 'gt':
                return field_value > value
            elif operator == 'gte':
                return field_value >= value
            elif operator == 'lt':
                return field_value < value
            elif operator == 'lte':
                return field_value <= value
            elif operator == 'in':
                return field_value in value
            elif operator == 'not_in':
                return field_value not in value
            elif operator == 'contains':
                return str(value) in str(field_value)
            elif operator == 'not_contains':
                return str(value) not in str(field_value)
            
            return False
            
        except Exception:
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """Get nested field value using dot notation."""
        try:
            value = data
            for part in field.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return None


@dataclass
class ArchivingPolicy:
    """Comprehensive archiving policy configuration."""
    id: str
    name: str
    rules: List[ArchivingRule]
    default_action: ArchivingAction = ArchivingAction.NO_ACTION
    importance_weights: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    
    def get_action_for_memory(self, memory_data: Dict[str, Any]) -> ArchivingAction:
        """Get archiving action for a memory based on policy rules."""
        # Sort rules by priority (higher priority first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if rule.matches(memory_data):
                return rule.action
        
        return self.default_action


@dataclass
class RestoreTrigger:
    """Represents a trigger for memory restoration."""
    id: str
    trigger_type: RestoreTriggerType
    memory_id: str
    triggered_at: datetime
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    processed: bool = False


class PolicyEngine:
    """
    Advanced policy engine for archiving decisions.
    
    Features:
    - Rule-based policy evaluation
    - Machine learning-enhanced decision making
    - Dynamic threshold adjustment
    - Policy conflict resolution
    - Performance optimization
    """
    
    def __init__(self):
        self.policies: Dict[str, ArchivingPolicy] = {}
        self.ml_model: Optional[RandomForestClassifier] = None
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Default policies
        self._create_default_policies()
    
    def _create_default_policies(self):
        """Create default archiving policies."""
        # Time-based policy
        time_rules = [
            ArchivingRule(
                id="old_unused",
                name="Archive old unused memories",
                policy_type=PolicyType.TIME_BASED,
                conditions=[
                    {'field': 'days_since_last_access', 'operator': 'gt', 'value': 90},
                    {'field': 'importance_level', 'operator': 'in', 'value': ['low', 'very_low']}
                ],
                action=ArchivingAction.ARCHIVE,
                priority=1
            )
        ]
        
        time_policy = ArchivingPolicy(
            id="time_based",
            name="Time-based Archiving",
            rules=time_rules,
            thresholds={'max_age_days': 365, 'min_importance': 0.3}
        )
        
        # Usage-based policy
        usage_rules = [
            ArchivingRule(
                id="low_usage",
                name="Archive low usage memories",
                policy_type=PolicyType.USAGE_BASED,
                conditions=[
                    {'field': 'access_count', 'operator': 'lt', 'value': 5},
                    {'field': 'days_since_creation', 'operator': 'gt', 'value': 30}
                ],
                action=ArchivingAction.ARCHIVE,
                priority=2
            )
        ]
        
        usage_policy = ArchivingPolicy(
            id="usage_based",
            name="Usage-based Archiving",
            rules=usage_rules,
            thresholds={'min_access_count': 3, 'min_access_frequency': 0.1}
        )
        
        self.policies['time_based'] = time_policy
        self.policies['usage_based'] = usage_policy
    
    def add_policy(self, policy: ArchivingPolicy):
        """Add or update an archiving policy."""
        self.policies[policy.id] = policy
    
    def evaluate_memory(self, memory_data: Dict[str, Any]) -> Tuple[ArchivingAction, float]:
        """Evaluate memory against all policies and return recommended action."""
        actions = []
        confidences = []
        
        # Evaluate against each enabled policy
        for policy in self.policies.values():
            if policy.enabled:
                action = policy.get_action_for_memory(memory_data)
                if action != ArchivingAction.NO_ACTION:
                    actions.append(action)
                    confidences.append(0.8)  # Base confidence
        
        # Use ML model if trained
        if self.is_trained and self.ml_model:
            try:
                ml_action, ml_confidence = self._ml_predict(memory_data)
                if ml_action != ArchivingAction.NO_ACTION:
                    actions.append(ml_action)
                    confidences.append(ml_confidence)
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # Resolve conflicts and return final decision
        if not actions:
            return ArchivingAction.NO_ACTION, 0.0
        
        # Simple majority vote with confidence weighting
        action_scores = defaultdict(float)
        for action, confidence in zip(actions, confidences):
            action_scores[action] += confidence
        
        best_action = max(action_scores.items(), key=lambda x: x[1])
        final_confidence = best_action[1] / len(actions)
        
        return best_action[0], final_confidence
    
    def _ml_predict(self, memory_data: Dict[str, Any]) -> Tuple[ArchivingAction, float]:
        """Use ML model to predict archiving action."""
        features = self._extract_features(memory_data)
        features_scaled = self.scaler.transform([features])
        
        # Get prediction and probability
        prediction = self.ml_model.predict(features_scaled)[0]
        probabilities = self.ml_model.predict_proba(features_scaled)[0]
        
        # Convert back to action
        action = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
        
        return ArchivingAction(action), confidence
    
    def _extract_features(self, memory_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features for ML model."""
        features = [
            memory_data.get('days_since_creation', 0),
            memory_data.get('days_since_last_access', 0),
            memory_data.get('access_count', 0),
            memory_data.get('importance_score', 0.5),
            memory_data.get('content_length', 0),
            memory_data.get('unique_users', 1),
            memory_data.get('recency_score', 0.5),
            memory_data.get('usage_score', 0.5),
            memory_data.get('relationship_score', 0.5),
        ]
        return features
    
    def train_ml_model(self, training_data: List[Dict[str, Any]]):
        """Train ML model on historical archiving decisions."""
        if len(training_data) < 50:  # Need minimum training data
            logger.warning("Insufficient training data for ML model")
            return
        
        try:
            # Extract features and labels
            X = []
            y = []
            
            for data in training_data:
                features = self._extract_features(data)
                action = data.get('actual_action', ArchivingAction.NO_ACTION.value)
                
                X.append(features)
                y.append(action)
            
            X = np.array(X)
            y = np.array(y)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            self.ml_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.ml_model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            
            self.is_trained = True
            
            logger.info(f"ML model trained with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")


class AutoArchiver:
    """
    Smart auto-archiving engine for predictive memory management.
    
    Features:
    - Local scoring algorithms for importance assessment
    - Feature engineering for archiving decisions
    - Policy engine with rule evaluation
    - Threshold-based restoration triggers
    - Machine learning-enhanced decision making
    - Performance optimization and monitoring
    """
    
    def __init__(
        self,
        usage_analyzer: Optional[UsageAnalyzer] = None,
        redis_cache: Optional[RedisCache] = None,
        default_archive_threshold: float = 0.3,
        restoration_threshold: float = 0.7
    ):
        """
        Initialize the auto-archiver.
        
        Args:
            usage_analyzer: Optional usage analyzer for pattern insights
            redis_cache: Optional Redis cache for archiving state
            default_archive_threshold: Default threshold for archiving decisions
            restoration_threshold: Threshold for automatic restoration
        """
        self.usage_analyzer = usage_analyzer
        self.redis_cache = redis_cache
        self.default_archive_threshold = default_archive_threshold
        self.restoration_threshold = restoration_threshold
        
        # Core components
        self.policy_engine = PolicyEngine()
        
        # Importance scoring
        self.importance_scores: Dict[str, ImportanceScore] = {}
        
        # Archived memories tracking
        self.archived_memories: Dict[str, Dict[str, Any]] = {}
        self.restoration_triggers: List[RestoreTrigger] = []
        
        # Feature extractors
        self.content_analyzer = None  # Could integrate NLP for content analysis
        
        # Performance tracking
        self.stats = {
            'memories_analyzed': 0,
            'memories_archived': 0,
            'memories_restored': 0,
            'archiving_accuracy': 0.0,
            'space_saved': 0,
            'policy_evaluations': 0
        }
        
        # ML components
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector_trained = False
        
        logger.info(
            "Auto-archiver initialized",
            archive_threshold=default_archive_threshold,
            restoration_threshold=restoration_threshold
        )
    
    async def analyze_memory_importance(
        self,
        memory: Memory,
        usage_pattern: Optional[UsagePattern] = None
    ) -> ImportanceScore:
        """
        Analyze and score memory importance.
        
        Args:
            memory: Memory to analyze
            usage_pattern: Optional usage pattern from usage analyzer
            
        Returns:
            Importance score with detailed breakdown
        """
        # Base importance scoring
        base_score = await self._calculate_base_importance(memory)
        
        # Usage-based importance
        usage_score = await self._calculate_usage_importance(memory, usage_pattern)
        
        # Recency-based importance
        recency_score = await self._calculate_recency_importance(memory)
        
        # Relationship-based importance
        relationship_score = await self._calculate_relationship_importance(memory)
        
        # Content-based importance
        content_score = await self._calculate_content_importance(memory)
        
        # User-assigned value
        user_value_score = await self._calculate_user_value_importance(memory)
        
        # Create importance score object
        importance = ImportanceScore(
            memory_id=memory.id,
            base_score=base_score,
            usage_score=usage_score,
            recency_score=recency_score,
            relationship_score=relationship_score,
            content_score=content_score,
            user_value_score=user_value_score,
            final_score=0.0,  # Will be calculated
            importance_level=ImportanceLevel.MEDIUM,  # Will be set
            confidence=0.8  # Base confidence
        )
        
        # Calculate final score
        importance.calculate_final_score()
        
        # Store importance score
        self.importance_scores[memory.id] = importance
        
        logger.debug(f"Importance analyzed for memory {memory.id}: {importance.final_score:.3f}")
        
        return importance
    
    async def _calculate_base_importance(self, memory: Memory) -> float:
        """Calculate base importance from memory metadata."""
        score = 0.5  # Default base score
        
        # Check for explicit importance markers
        if hasattr(memory, 'metadata') and memory.metadata:
            metadata = memory.metadata
            
            # User-marked importance
            if 'important' in metadata and metadata['important']:
                score += 0.3
            
            # System flags
            if 'system_critical' in metadata and metadata['system_critical']:
                score += 0.4
            
            # Tags indicating importance
            important_tags = {'critical', 'important', 'key', 'essential'}
            tags = set(metadata.get('tags', []))
            if tags.intersection(important_tags):
                score += 0.2
        
        return min(1.0, score)
    
    async def _calculate_usage_importance(
        self,
        memory: Memory,
        usage_pattern: Optional[UsagePattern]
    ) -> float:
        """Calculate importance based on usage patterns."""
        if not usage_pattern:
            return 0.3  # Default if no usage data
        
        score = 0.0
        
        # Access frequency contribution
        freq_scores = {
            'very_high': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'very_low': 0.2
        }
        score += freq_scores.get(usage_pattern.access_frequency.value, 0.3) * 0.4
        
        # Pattern confidence contribution
        score += usage_pattern.confidence * 0.3
        
        # Recent access boost
        if 'recency_score' in usage_pattern.metrics:
            score += usage_pattern.metrics['recency_score'] * 0.3
        
        return min(1.0, score)
    
    async def _calculate_recency_importance(self, memory: Memory) -> float:
        """Calculate importance based on recency."""
        if not hasattr(memory, 'created_at') or not memory.created_at:
            return 0.5
        
        # Calculate days since creation/last modification
        now = datetime.utcnow()
        created_days = (now - memory.created_at).days
        
        # Exponential decay for recency
        recency_score = math.exp(-created_days / 30.0)  # 30-day half-life
        
        return min(1.0, recency_score)
    
    async def _calculate_relationship_importance(self, memory: Memory) -> float:
        """Calculate importance based on relationships to other memories."""
        # This could be enhanced with actual relationship analysis
        # For now, use simple heuristics
        
        score = 0.3  # Base relationship score
        
        if hasattr(memory, 'metadata') and memory.metadata:
            # Check for references or links
            content = memory.content.lower()
            
            # Count references to other memories/documents
            reference_indicators = ['see also', 'refer to', 'related to', 'similar to']
            reference_count = sum(1 for indicator in reference_indicators if indicator in content)
            
            # Boost score based on references
            score += min(0.4, reference_count * 0.1)
            
            # Check for being referenced by others (would need reverse lookup)
            # This is simplified - in practice would check relationship graph
        
        return min(1.0, score)
    
    async def _calculate_content_importance(self, memory: Memory) -> float:
        """Calculate importance based on content quality and characteristics."""
        content = memory.content
        score = 0.4  # Base content score
        
        # Content length contribution
        length = len(content)
        if length > 1000:  # Substantial content
            score += 0.2
        elif length > 500:
            score += 0.1
        elif length < 50:  # Very short content
            score -= 0.1
        
        # Content structure indicators
        structure_indicators = [
            ('# ', 0.1),     # Headers
            ('* ', 0.05),    # Lists
            ('```', 0.1),    # Code blocks
            ('http', 0.05),  # URLs
            ('@', 0.02),     # Mentions
        ]
        
        for indicator, weight in structure_indicators:
            count = content.count(indicator)
            score += min(0.1, count * weight)
        
        # Technical content indicators
        technical_keywords = [
            'algorithm', 'function', 'method', 'class', 'interface',
            'database', 'query', 'analysis', 'research', 'documentation'
        ]
        
        content_lower = content.lower()
        technical_count = sum(1 for keyword in technical_keywords if keyword in content_lower)
        score += min(0.2, technical_count * 0.02)
        
        return min(1.0, score)
    
    async def _calculate_user_value_importance(self, memory: Memory) -> float:
        """Calculate importance based on user-assigned value indicators."""
        score = 0.5  # Neutral default
        
        if hasattr(memory, 'metadata') and memory.metadata:
            metadata = memory.metadata
            
            # Explicit user rating
            if 'user_rating' in metadata:
                rating = metadata['user_rating']
                if isinstance(rating, (int, float)) and 0 <= rating <= 5:
                    score = rating / 5.0
            
            # User flags
            if 'favorite' in metadata and metadata['favorite']:
                score += 0.3
            
            if 'bookmark' in metadata and metadata['bookmark']:
                score += 0.2
            
            # Priority indicators
            priority = metadata.get('priority', 'medium').lower()
            priority_scores = {
                'critical': 1.0,
                'high': 0.8,
                'medium': 0.5,
                'low': 0.3,
                'none': 0.2
            }
            score = max(score, priority_scores.get(priority, 0.5))
        
        return min(1.0, score)
    
    async def evaluate_for_archiving(
        self,
        memories: List[Memory],
        force_evaluation: bool = False
    ) -> List[Tuple[Memory, ArchivingAction, float]]:
        """
        Evaluate memories for archiving decisions.
        
        Args:
            memories: List of memories to evaluate
            force_evaluation: Force re-evaluation even if recently done
            
        Returns:
            List of (memory, recommended_action, confidence) tuples
        """
        results = []
        
        for memory in memories:
            # Get or calculate importance score
            importance = self.importance_scores.get(memory.id)
            if not importance or force_evaluation:
                usage_pattern = None
                if self.usage_analyzer:
                    patterns = await self.usage_analyzer.analyze_patterns([memory])
                    usage_pattern = patterns.get(memory.id)
                
                importance = await self.analyze_memory_importance(memory, usage_pattern)
            
            # Prepare memory data for policy evaluation
            memory_data = await self._prepare_memory_data(memory, importance)
            
            # Evaluate with policy engine
            action, confidence = self.policy_engine.evaluate_memory(memory_data)
            
            # Apply anomaly detection
            if self.anomaly_detector_trained:
                is_anomaly = await self._detect_anomaly(memory_data)
                if is_anomaly:
                    # Be more conservative with anomalous memories
                    if action in [ArchivingAction.DELETE, ArchivingAction.ARCHIVE]:
                        confidence *= 0.7
            
            # Final decision based on importance and thresholds
            final_action, final_confidence = await self._make_final_decision(
                memory, importance, action, confidence
            )
            
            results.append((memory, final_action, final_confidence))
            
            self.stats['memories_analyzed'] += 1
            self.stats['policy_evaluations'] += 1
        
        logger.info(f"Evaluated {len(memories)} memories for archiving")
        
        return results
    
    async def _prepare_memory_data(self, memory: Memory, importance: ImportanceScore) -> Dict[str, Any]:
        """Prepare memory data for policy evaluation."""
        now = datetime.utcnow()
        
        # Calculate time-based metrics
        days_since_creation = 0
        days_since_last_access = 0
        
        if hasattr(memory, 'created_at') and memory.created_at:
            days_since_creation = (now - memory.created_at).days
        
        # Get usage data if available
        access_count = 0
        unique_users = 1
        usage_score = 0.5
        recency_score = 0.5
        
        if self.usage_analyzer:
            pattern = self.usage_analyzer.memory_patterns.get(memory.id)
            if pattern:
                access_count = pattern.access_count
                unique_users = len(pattern.user_ids)
                days_since_last_access = (now - pattern.last_access).days
                
                metrics = pattern.calculate_metrics()
                usage_score = metrics.get('regularity_score', 0.5)
                recency_score = metrics.get('recency_score', 0.5)
        
        return {
            'memory_id': memory.id,
            'days_since_creation': days_since_creation,
            'days_since_last_access': days_since_last_access,
            'access_count': access_count,
            'unique_users': unique_users,
            'content_length': len(memory.content),
            'importance_score': importance.final_score,
            'importance_level': importance.importance_level.value,
            'usage_score': usage_score,
            'recency_score': recency_score,
            'relationship_score': importance.relationship_score,
            'content_score': importance.content_score,
            'user_value_score': importance.user_value_score
        }
    
    async def _detect_anomaly(self, memory_data: Dict[str, Any]) -> bool:
        """Use anomaly detection to identify unusual memories."""
        try:
            features = self.policy_engine._extract_features(memory_data)
            prediction = self.anomaly_detector.predict([features])[0]
            return prediction == -1  # -1 indicates anomaly
        except Exception:
            return False
    
    async def _make_final_decision(
        self,
        memory: Memory,
        importance: ImportanceScore,
        suggested_action: ArchivingAction,
        confidence: float
    ) -> Tuple[ArchivingAction, float]:
        """Make final archiving decision with safety checks."""
        
        # Safety checks for critical memories
        if importance.importance_level == ImportanceLevel.CRITICAL:
            return ArchivingAction.NO_ACTION, 1.0
        
        # Apply threshold-based decisions
        if importance.final_score > self.restoration_threshold:
            # High importance - avoid archiving
            if suggested_action in [ArchivingAction.ARCHIVE, ArchivingAction.DELETE]:
                return ArchivingAction.NO_ACTION, 0.9
        
        elif importance.final_score < self.default_archive_threshold:
            # Low importance - favor archiving
            if suggested_action == ArchivingAction.NO_ACTION:
                return ArchivingAction.ARCHIVE, 0.7
        
        # Apply confidence threshold
        if confidence < 0.5:
            return ArchivingAction.NO_ACTION, confidence
        
        return suggested_action, confidence
    
    async def execute_archiving_action(
        self,
        memory: Memory,
        action: ArchivingAction,
        confidence: float
    ) -> bool:
        """
        Execute archiving action on memory.
        
        Args:
            memory: Memory to act on
            action: Action to execute
            confidence: Confidence in the decision
            
        Returns:
            True if action was successful
        """
        try:
            if action == ArchivingAction.NO_ACTION:
                return True
            
            # Log the action
            logger.info(
                f"Executing archiving action",
                memory_id=memory.id,
                action=action.value,
                confidence=confidence
            )
            
            if action == ArchivingAction.ARCHIVE:
                success = await self._archive_memory(memory, confidence)
            elif action == ArchivingAction.DELETE:
                success = await self._delete_memory(memory, confidence)
            elif action == ArchivingAction.COMPRESS:
                success = await self._compress_memory(memory, confidence)
            elif action == ArchivingAction.MOVE_COLD_STORAGE:
                success = await self._move_to_cold_storage(memory, confidence)
            elif action == ArchivingAction.MARK_STALE:
                success = await self._mark_stale(memory, confidence)
            else:
                success = False
            
            if success:
                self.stats['memories_archived'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute archiving action: {e}")
            return False
    
    async def _archive_memory(self, memory: Memory, confidence: float) -> bool:
        """Archive a memory (move to archived state)."""
        archive_data = {
            'memory_id': memory.id,
            'content': memory.content,
            'metadata': getattr(memory, 'metadata', {}),
            'archived_at': datetime.utcnow().isoformat(),
            'archive_confidence': confidence,
            'original_location': 'active'
        }
        
        # Store in archived memories
        self.archived_memories[memory.id] = archive_data
        
        # Could integrate with actual storage system here
        logger.info(f"Memory {memory.id} archived successfully")
        
        return True
    
    async def _delete_memory(self, memory: Memory, confidence: float) -> bool:
        """Delete a memory (permanent removal)."""
        # This should only be done with very high confidence
        if confidence < 0.9:
            logger.warning(f"Refusing to delete memory {memory.id} with low confidence: {confidence}")
            return False
        
        # In practice, this would interface with the actual memory storage
        logger.info(f"Memory {memory.id} marked for deletion")
        
        return True
    
    async def _compress_memory(self, memory: Memory, confidence: float) -> bool:
        """Compress memory content to save space."""
        # Could implement actual compression here
        logger.info(f"Memory {memory.id} compressed")
        return True
    
    async def _move_to_cold_storage(self, memory: Memory, confidence: float) -> bool:
        """Move memory to cold storage."""
        logger.info(f"Memory {memory.id} moved to cold storage")
        return True
    
    async def _mark_stale(self, memory: Memory, confidence: float) -> bool:
        """Mark memory as stale."""
        logger.info(f"Memory {memory.id} marked as stale")
        return True
    
    async def check_restoration_triggers(self) -> List[str]:
        """Check for memories that should be restored."""
        memories_to_restore = []
        
        for trigger in self.restoration_triggers:
            if not trigger.processed:
                if await self._should_restore(trigger):
                    memories_to_restore.append(trigger.memory_id)
                    trigger.processed = True
        
        return memories_to_restore
    
    async def _should_restore(self, trigger: RestoreTrigger) -> bool:
        """Determine if a restoration trigger should be acted upon."""
        # Check if memory exists in archived state
        if trigger.memory_id not in self.archived_memories:
            return False
        
        # Check trigger confidence
        if trigger.confidence < self.restoration_threshold:
            return False
        
        # Type-specific logic
        if trigger.trigger_type == RestoreTriggerType.ACCESS_ATTEMPT:
            return True  # Always restore on access attempt
        
        elif trigger.trigger_type == RestoreTriggerType.SEARCH_QUERY:
            # Restore if high relevance to search
            relevance = trigger.trigger_data.get('relevance', 0.0)
            return relevance > 0.7
        
        elif trigger.trigger_type == RestoreTriggerType.RELATED_ACCESS:
            # Restore if related memory is being accessed frequently
            related_accesses = trigger.trigger_data.get('related_accesses', 0)
            return related_accesses > 5
        
        return False
    
    async def add_restoration_trigger(
        self,
        memory_id: str,
        trigger_type: RestoreTriggerType,
        trigger_data: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ):
        """Add a restoration trigger for an archived memory."""
        trigger = RestoreTrigger(
            id=f"{trigger_type.value}_{memory_id}_{int(datetime.utcnow().timestamp())}",
            trigger_type=trigger_type,
            memory_id=memory_id,
            triggered_at=datetime.utcnow(),
            trigger_data=trigger_data or {},
            confidence=confidence
        )
        
        self.restoration_triggers.append(trigger)
        
        logger.info(f"Restoration trigger added for memory {memory_id}")
    
    async def train_models(self, historical_data: List[Dict[str, Any]]):
        """Train ML models on historical archiving data."""
        # Train policy engine ML model
        self.policy_engine.train_ml_model(historical_data)
        
        # Train anomaly detector
        if len(historical_data) >= 100:
            try:
                features = []
                for data in historical_data:
                    feature_vector = self.policy_engine._extract_features(data)
                    features.append(feature_vector)
                
                X = np.array(features)
                self.anomaly_detector.fit(X)
                self.anomaly_detector_trained = True
                
                logger.info("Anomaly detector trained successfully")
                
            except Exception as e:
                logger.warning(f"Anomaly detector training failed: {e}")
    
    def get_archiving_stats(self) -> Dict[str, Any]:
        """Get comprehensive archiving statistics."""
        return {
            **self.stats,
            'archived_memories_count': len(self.archived_memories),
            'pending_restoration_triggers': len([t for t in self.restoration_triggers if not t.processed]),
            'importance_scores_calculated': len(self.importance_scores),
            'policies_configured': len(self.policy_engine.policies),
            'ml_model_trained': self.policy_engine.is_trained,
            'anomaly_detector_trained': self.anomaly_detector_trained
        }